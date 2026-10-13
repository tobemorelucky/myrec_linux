"""Research prototype: shared dual-view nonlinear refinement of ASPCF interests."""
import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.BaseModel import GeneralModel
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF
from models.sequential.LLMMIRecASPCFSampledSoftmax import (
    LLMMIRecASPCFSampledSoftmax, candidate_mask)


class DualViewInterestRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(192, 128)
        self.output = nn.Linear(64, 64)
        nn.init.normal_(self.input.weight, std=.01)
        nn.init.normal_(self.input.bias, std=.01)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features):
        return self.output(F.glu(self.input(features), dim=-1))


class LLMMIRecASPCFDualViewRefine(LLMMIRecASPCFSampledSoftmax):
    def __init__(self, args, corpus):
        if (int(args.history_max) != 20 or args.item_encoder != "aspcf"
                or int(args.emb_size) != 64 or int(args.semantic_dim) != 32
                or int(args.complement_dim) != 32):
            raise ValueError("Requires H20 / original ASPCF / D64 / semantic32 / complement32")
        super().__init__(args, corpus)
        # Preserve all original initialization and subsequent training RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261012)
            self.refiner = DualViewInterestRefiner()
        self._refine_grad_stats = None
        logging.info("[DualViewRefine] shared 192->128 GLU64->64, zero output; added=%d total=%d",
                     sum(p.numel() for p in self.refiner.parameters()), self.count_variables())

    def _refine(self, out, history, lengths):
        raw = out["history_vectors"]
        B, L, _ = raw.shape
        valid = ((torch.arange(L, device=raw.device)[None, :] < lengths[:, None])
                 & (history != 0))
        attention = out["attention_maps"].masked_fill(~valid[:, None, :], 0)
        semantic = out["history_semantic"].masked_fill(~valid[:, :, None], 0)
        complement = out["history_complement"].masked_fill(~valid[:, :, None], 0)
        S = torch.bmm(attention, semantic)
        C = torch.bmm(attention, complement)
        masked_raw = raw.masked_fill(~valid[:, :, None], 0)
        mean = masked_raw.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        last = masked_raw[torch.arange(B, device=raw.device), (lengths-1).clamp_min(0)]
        # Reuse the original aggregator's LayerNorm parameters and exact context definition.
        context = self.aggregator.ln(mean + last)
        features = torch.cat([out["interest_vectors"], S, C,
                              context[:, None, :].expand(-1, self.K, -1)], dim=-1)
        residual = self.refiner(features).masked_fill((lengths == 0)[:, None, None], 0)
        refined = out["interest_vectors"] + residual
        user = (refined * out["interest_weights"][:, :, None]).sum(1)
        return user, refined, residual, S, C, context, features

    def forward(self, feed_dict, return_intermediate=False):
        # Exactly one original history encoding and one candidate encoding.
        # Bypass SS.forward to avoid constructing an unused baseline BxB matrix.
        out = LLMMIRecASPCF.forward(self, feed_dict, return_intermediate=True)
        user, refined, residual, S, C, context, features = self._refine(
            out, feed_dict["history_items"], feed_dict["lengths"])
        out["user_vector"] = user
        out["prediction"] = (user[:, None, :] * out["candidate_vectors"]).sum(-1)
        if self.training:
            positives = feed_dict["item_id"][:, 0]
            allowed, duplicate, clicked = candidate_mask(
                self.train_membership, feed_dict["user_id"], positives)
            out["_ss_scores"] = user @ out["candidate_vectors"][:, 0, :].T
            out["_ss_allowed"] = allowed
            out["_ss_duplicate"] = duplicate.sum()
            out["_ss_clicked"] = clicked.sum()
        if return_intermediate:
            out.update(refined_interest_vectors=refined, interest_residual=residual,
                       interest_semantic=S, interest_complement=C,
                       refinement_context=context, refinement_features=features)
        else:
            for key in list(out):
                if key not in ("prediction", "_relation_ids") and not key.startswith("_ss_"):
                    del out[key]
        return out

    def loss(self, out):
        loss = super().loss(out)
        if self.training and "_ss_scores" in out and self._refine_grad_stats is None:
            self._refine_grad_stats = {}
            for name, parameter in [("output", self.refiner.output.weight),
                                     ("input", self.refiner.input.weight)]:
                def capture(grad, key=name):
                    if key not in self._refine_grad_stats:
                        self._refine_grad_stats[key] = grad.detach().norm()
                    return grad
                self._refine_grad_stats[name+"_handle"] = parameter.register_hook(capture)
        return loss

    def _flush_stats(self):
        super()._flush_stats()
        if self._refine_grad_stats is not None:
            values = []
            for key in ("output", "input"):
                self._refine_grad_stats[key+"_handle"].remove()
                values.append(float(self._refine_grad_stats[key].cpu())
                              if key in self._refine_grad_stats else float("nan"))
            logging.info("[DualViewRefineStats] epoch=%s first_batch_output_grad=%.9g "
                         "first_batch_input_grad=%.9g output_weight_norm=%.9g",
                         self._ss_epoch, values[0], values[1],
                         float(self.refiner.output.weight.detach().norm().cpu()))
            self._refine_grad_stats = None

    class Dataset(LLMMIRecASPCFSampledSoftmax.Dataset):
        def _get_feed_dict(self, index):
            feed = GeneralModel.Dataset._get_feed_dict(self, index)
            pos = int(self.data["position"][index])
            seq = self.corpus.user_his[feed["user_id"]][max(0, pos-20):pos]
            feed["history_items"] = np.array([x[0] for x in seq])
            feed["history_times"] = np.array([x[1] for x in seq])
            feed["lengths"] = len(seq)
            return feed
