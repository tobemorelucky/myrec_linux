"""Experimental single-step relation-disagreement attention on frozen ASPCF design.

Only interest attention changes. Complement is NOT a pure collaborative view.
No claim that relation disagreement is an error or that this is established novelty.
"""
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF
from models.sequential.llmmi_utils import check_nan_inf


def relation_difference(semantic, complement, valid):
    """Raw within-view cosine difference, masked on BOTH axes, no self edges.

    Normalize only valid vectors; finite padding content cannot influence a relation.
    eps=1e-8 gives zero cosine for zero vectors, without inventing similarities.
    """
    s = F.normalize(semantic.masked_fill(~valid[..., None], 0), dim=-1, eps=1e-8)
    c = F.normalize(complement.masked_fill(~valid[..., None], 0), dim=-1, eps=1e-8)
    length = valid.shape[1]
    edges = valid[:, :, None] & valid[:, None, :]
    edges = edges & ~torch.eye(length, dtype=torch.bool, device=valid.device)[None]
    delta = (torch.bmm(s, s.transpose(1, 2)) -
             torch.bmm(c, c.transpose(1, 2))).masked_fill(~edges, 0)
    return delta, edges


def masked_attention(logits, valid):
    """Matches original softmax for nonempty histories; empty rows safely zero."""
    masked = logits.masked_fill(~valid[:, None, :], float("-inf"))
    masked = torch.where(valid.any(-1)[:, None, None], masked, torch.zeros_like(masked))
    return F.softmax(masked, dim=-1).masked_fill(~valid[:, None, :], 0)


class RelationDisagreementAttention(nn.Module):
    def __init__(self, original, eta_init=0.0, eta_max=0.1):
        super().__init__()
        if not (math.isfinite(eta_max) and eta_max > 0 and
                math.isfinite(eta_init) and abs(eta_init) < eta_max):
            raise ValueError("Require finite eta_max>0 and abs(eta_init)<eta_max")
        # Reuse actual initialized parameters: no new Q/K/V, no RNG consumption.
        self.K, self.attn_size = original.K, original.attn_size
        self.query = original.query
        self.Wq, self.Wk, self.Wv = original.Wq, original.Wk, original.Wv
        self.eta_max = float(eta_max)
        self.eta_raw = nn.Parameter(torch.tensor(math.atanh(eta_init / eta_max)))

    @property
    def eta(self):
        return self.eta_max * self.eta_raw.tanh()

    def forward(self, history_emb, lengths, semantic, complement, return_details=False):
        batch, length, _ = history_emb.shape
        valid = torch.arange(length, device=lengths.device)[None] < lengths[:, None]
        query = self.Wq(self.query).unsqueeze(0).expand(batch, -1, -1)
        keys = self.Wk(history_emb)
        values = self.Wv(history_emb)
        logits = torch.bmm(query, keys.transpose(1, 2)) / math.sqrt(self.attn_size)
        a0 = masked_attention(logits, valid)
        delta, edges = relation_difference(semantic, complement, valid)
        # E = A0 @ D; normalize by the actual remaining attention mass per
        # destination, rather than dividing by padded L or diluting with self edges.
        mass = torch.bmm(a0, edges.to(a0.dtype))
        evidence = torch.bmm(a0, delta) / mass.clamp_min(1e-8)
        evidence = evidence.masked_fill(~valid[:, None, :], 0)
        correction = self.eta * evidence
        attention = masked_attention(logits + correction, valid)
        interests = torch.bmm(attention, values)
        if return_details:
            return interests, attention, dict(
                baseline_attention=a0, relation_difference=delta,
                relation_evidence=evidence, relation_mass=mass,
                attention_correction=correction, eta=self.eta)
        return interests, attention


class LLMMIRecASPCFRelationAttention(LLMMIRecASPCF):
    extra_log_args = LLMMIRecASPCF.extra_log_args + ["relation_eta_init", "relation_eta_max"]

    @staticmethod
    def parse_model_args(parser):
        parser = LLMMIRecASPCF.parse_model_args(parser)
        parser.add_argument("--relation_eta_init", type=float, default=0.0)
        parser.add_argument("--relation_eta_max", type=float, default=0.1)
        return parser

    def __init__(self, args, corpus):
        if args.item_encoder != "aspcf" or args.aspcf_gate_mode != "basic":
            raise ValueError("This candidate requires the unchanged basic ASPCF encoder")
        super().__init__(args, corpus)
        self.extractor = RelationDisagreementAttention(
            self.extractor, args.relation_eta_init, args.relation_eta_max)
        logging.info("[RelationAttention] signed eta=%s*tanh(theta), init=%s; #params: %s",
                     args.relation_eta_max, args.relation_eta_init, self.count_variables())

    def forward(self, feed_dict, return_intermediate=False):
        history, lengths = feed_dict["history_items"], feed_dict["lengths"]
        candidates = feed_dict["item_id"]
        _, length = history.shape
        device = history.device

        # Identical ItemEncoder instance and initialization for history/candidates.
        hist_out = self.item_encoder(history, return_components=True)
        raw = hist_out["emb"]
        cand_out = (self.item_encoder(candidates, return_components=True)
                    if return_intermediate else None)
        cand = cand_out["emb"] if cand_out is not None else self.item_encoder(candidates)

        # Original reverse-position and dropout path, unchanged.
        valid_his = (history > 0).long()
        len_range = torch.arange(self.max_his, device=device)
        position = (lengths[:, None] - len_range[None, :length]) * valid_his
        positioned = self.dropout(raw + self.position_emb(position))
        result = self.extractor(positioned, lengths, hist_out["semantic"],
                                hist_out["complement"], return_intermediate)
        vectors, attention = result[:2]
        vectors = self.dropout(vectors)
        weights = self.aggregator(raw, lengths)
        user = (vectors * weights[:, :, None]).sum(dim=1)
        prediction = (user[:, None, :] * cand).sum(dim=-1)

        # Preserve original relation-ID collection, sampling and parent loss.
        out = {"prediction": prediction}
        if self.training and self.lambda_relation > 0:
            ids = torch.unique(torch.cat([history.reshape(-1), candidates.reshape(-1)]))
            ids = ids[ids != 0]
            if ids.numel() > self.relation_sample_size:
                ids = ids[torch.randperm(ids.numel(), device=device)[:self.relation_sample_size]]
            out["_relation_ids"] = ids

        if not self._first_batch_checked:
            self._first_batch_checked = True
            for name, tensor in [("history_vectors", raw), ("interest_vectors", vectors),
                                 ("interest_weights", weights), ("candidate_vectors", cand),
                                 ("prediction", prediction)]:
                check_nan_inf(tensor, name)
            logging.info("[RelationAttention] First-batch finite check passed.")

        if return_intermediate:
            out.update(interest_vectors=vectors, attention_maps=attention,
                       interest_weights=weights, user_vector=user,
                       history_vectors=raw, candidate_vectors=cand,
                       history_semantic=hist_out["semantic"],
                       history_complement=hist_out["complement"],
                       history_alpha_sem=hist_out["alpha_sem"],
                       history_alpha_comp=hist_out["alpha_comp"],
                       candidate_semantic=cand_out["semantic"],
                       candidate_complement=cand_out["complement"],
                       candidate_alpha_sem=cand_out["alpha_sem"],
                       candidate_alpha_comp=cand_out["alpha_comp"])
            out.update(result[2])
        return out
