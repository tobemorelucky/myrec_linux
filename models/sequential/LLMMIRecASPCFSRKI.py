"""SRKI research prototype. Pivoted Cholesky is classical kernel approximation.
Positive-support pooling does NOT guarantee semantic clustering or full history coverage.
"""
import json
import math
from pathlib import Path
import torch
from torch import nn
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF
from models.sequential.llmmi_components import ItemEncoder
from models.sequential.llmmi_utils import check_nan_inf

NORM_EPS = 1e-8
RANK_TOL = 1e-6  # Numerical rank tolerance on unit-diagonal Gram matrices, not searched.


def masked_softmax(logits, mask):
    masked = logits.masked_fill(~mask, -torch.inf)
    safe = torch.where(mask.any(-1, keepdim=True), masked, torch.zeros_like(masked))
    return safe.softmax(-1) * mask.to(logits.dtype)


@torch.no_grad()
def residual_support(features, valid, K, diagnostics=False):
    """Kernel detached for every source. No user loop or host synchronization.
    Signed Cholesky factors drive residual updates; only pooling rectifies their signs.
    The first pivot is the latest nonzero semantic row. Ties later use first index.
    """
    if K < 1:
        raise ValueError("K must be >=1")
    B, L, _ = features.shape
    dtype = torch.float64 if features.dtype == torch.float64 else torch.float32
    x = torch.where(valid[..., None], features.to(dtype), 0.0)
    norms = x.norm(dim=-1)
    semantic_valid = valid & (norms > NORM_EPS)
    z = x / norms.clamp_min(NORM_EPS)[..., None] * semantic_valid[..., None]
    G = torch.bmm(z, z.transpose(1, 2))
    R = G.clone()
    selected = torch.zeros_like(valid)
    factors, active_slots, indices = [], [], []
    rs = [R.clone()] if diagnostics else []
    energies = [R.diagonal(dim1=1, dim2=2).sum(-1)] if diagnostics else []
    rows = torch.arange(B, device=x.device)
    if L:
        last = torch.arange(L, device=x.device)[None].expand(B, -1).masked_fill(
            ~semantic_valid, -1).max(-1).values.clamp_min(0)
    for k in range(min(K, L)):
        diagonal = R.diagonal(dim1=1, dim2=2)
        available = semantic_valid & ~selected
        scores = diagonal.clamp_min(0).masked_fill(~available, -torch.inf)
        pivot = last if k == 0 else scores.argmax(-1)
        pivot_value = diagonal[rows, pivot]
        active = available[rows, pivot] & (pivot_value > RANK_TOL)
        divisor = torch.where(active, pivot_value, torch.ones_like(pivot_value)).sqrt()
        q = R.gather(2, pivot[:, None, None].expand(-1, L, 1)).squeeze(-1)
        q = q / divisor[:, None] * active[:, None] * semantic_valid
        factors.append(q)
        active_slots.append(active)
        indices.append(torch.where(active, pivot, torch.full_like(pivot, -1)))
        selected.scatter_(1, pivot[:, None], selected.gather(1, pivot[:, None]) | active[:, None])
        R = R - q[:, :, None] * q[:, None, :]
        R = (R + R.transpose(1, 2)) * 0.5
        if diagnostics:
            rs.append(R.clone())
            energies.append(R.diagonal(dim1=1, dim2=2).sum(-1))
    for _ in range(K - len(factors)):
        factors.append(x.new_zeros(B, L))
        active_slots.append(torch.zeros(B, dtype=torch.bool, device=x.device))
        indices.append(torch.full((B,), -1, device=x.device, dtype=torch.long))
        if diagnostics:
            rs.append(R.clone())
            energies.append(R.diagonal(dim1=1, dim2=2).sum(-1))
    q = torch.stack(factors, 1)
    semantic_slots = torch.stack(active_slots, 1)
    mass = q.clamp_min(0).square() * valid[:, None]
    base = mass / mass.sum(-1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)
    # No semantic evidence: one explicitly nonsemantic fallback; never fabricate K slots.
    fallback = ~semantic_valid.any(-1) & valid.any(-1)
    if L:
        uniform = valid.to(dtype) / valid.sum(-1, keepdim=True).clamp_min(1)
        base[:, 0] = torch.where(fallback[:, None], uniform, base[:, 0])
    mask = semantic_slots.clone()
    mask[:, 0] |= fallback
    info = dict(base_attention=base, interest_mask=mask,
                semantic_slot_mask=semantic_slots, fallback_used=fallback)
    if diagnostics:
        info.update(kernel=G, factors=q, pivots=torch.stack(indices, 1),
                    residuals=torch.stack(rs, 1), residual_energy=torch.stack(energies, 1),
                    semantic_valid=semantic_valid)
    return info


class SRKIExtractor(nn.Module):
    def __init__(self, K, dim=64):
        super().__init__()
        self.K = int(K)
        if self.K < 1:
            raise ValueError("K must be >=1")
        self.Wv = nn.Linear(dim, dim)
        # A bias would cancel in normalization and be unused.
        self.importance = nn.Linear(dim, 1, bias=False)

    def forward(self, history, features, valid, diagnostics=False):
        info = residual_support(features, valid, self.K, diagnostics)
        base = info["base_attention"].to(history.dtype)
        if history.shape[1] == 0:
            info.update(interest_vectors=history.new_zeros(history.shape[0], self.K, history.shape[-1]),
                        attention_maps=base)
            return info
        supported = base > 0
        logbase = torch.where(supported, base, torch.ones_like(base)).log()
        logits = self.importance(history).squeeze(-1)
        attention = masked_softmax(logbase + logits[:, None, :], supported)
        values = self.Wv(history) * valid[..., None]
        info.update(interest_vectors=torch.bmm(attention, values), attention_maps=attention)
        if diagnostics:
            info.update(importance_logits=logits, values=values)
        return info


class ContentAggregator(nn.Module):
    def __init__(self, dim=64, K=None):
        super().__init__()
        # Omit the provably inactive scoring projection for K1.
        self.Wq = nn.Linear(dim, dim) if K != 1 else None
        self.scale = math.sqrt(dim)

    def forward(self, interests, last, mask):
        if self.Wq is None:
            weights = mask.to(interests.dtype)
        else:
            query = self.Wq(last)
            weights = masked_softmax((interests * query[:, None]).sum(-1) / self.scale, mask)
        return (interests * weights[..., None]).sum(1), weights


class LLMMIRecASPCFSRKI(LLMMIRecASPCF):
    VERSION = "srki-prototype-v1"
    extra_log_args = LLMMIRecASPCF.extra_log_args + ["srki_kernel_source", "srki_permutation_seed"]

    @staticmethod
    def parse_model_args(parser):
        parser = LLMMIRecASPCF.parse_model_args(parser)
        parser.add_argument("--srki_kernel_source", choices=["llm", "aspcf", "permuted_llm"], default="llm")
        parser.add_argument("--srki_permutation_seed", type=int, default=20261016)
        parser.set_defaults(K=2)
        return parser

    def __init__(self, args, corpus):
        if args.K < 1 or args.history_max != 20 or args.emb_size != 64:
            raise ValueError("SRKI requires positive K, H20, D64")
        if args.item_encoder != "aspcf" or args.semantic_rank != 512:
            raise ValueError("SRKI uses unchanged ASPCF and existing first512 semantic source")
        self.srki_kernel_source = args.srki_kernel_source
        self.srki_permutation_seed = int(args.srki_permutation_seed)
        super().__init__(args, corpus)
        # One independent global item-row permutation; original ItemEncoder table stays intact.
        rng = torch.Generator(device="cpu").manual_seed(self.srki_permutation_seed)
        mapping = torch.cat([torch.zeros(1, dtype=torch.long),
                             torch.randperm(self.item_num - 1, generator=rng) + 1])
        self.register_buffer("srki_semantic_permutation", mapping, persistent=False)

    def _define_params(self, llm_table):
        self.item_encoder = ItemEncoder(
            item_num=self.item_num, emb_size=self.emb_size, mode="aspcf", llm_table=llm_table,
            semantic_rank=self.semantic_rank, semantic_dim=self.semantic_dim,
            semantic_hidden=self.semantic_hidden, complement_dim=self.complement_dim,
            tail_hidden=self.tail_hidden, complement_hidden=self.complement_hidden,
            gate_hidden=self.gate_hidden, aspcf_gate_mode=self.aspcf_gate_mode)
        self.extractor = SRKIExtractor(self.K, self.emb_size)
        self.aggregator = ContentAggregator(self.emb_size, self.K)
        self.dropout = nn.Dropout(self.dropout_p)
        # No unused old query, position table, or history-only MLP aggregator.

    def encode_history(self, history, lengths):
        valid = (torch.arange(history.shape[1], device=history.device)[None] < lengths[:, None]) & (history > 0)
        raw = self.item_encoder(history)
        if self.srki_kernel_source == "aspcf":
            features = raw.detach()  # Mixed representation, not pure ID or fixed semantic truth.
        else:
            ids = self.srki_semantic_permutation[history] if self.srki_kernel_source == "permuted_llm" else history
            features = self.item_encoder.llm_table[ids][..., :self.semantic_rank]
        return raw, features.detach(), valid

    def forward(self, feed_dict, return_intermediate=False):
        history, lengths = feed_dict["history_items"], feed_dict["lengths"]
        raw, features, valid = self.encode_history(history, lengths)
        state = self.extractor(self.dropout(raw), features, valid, return_intermediate)
        interests = self.dropout(state["interest_vectors"])
        if history.shape[1]:
            last = raw[torch.arange(raw.shape[0], device=raw.device), (lengths - 1).clamp_min(0)]
            last = last * (lengths > 0)[:, None]
        else:
            last = raw.new_zeros(raw.shape[0], self.emb_size)
        user, weights = self.aggregator(interests, last, state["interest_mask"])
        ids = feed_dict["item_id"]
        candidates = self.item_encoder(ids)
        prediction = (user[:, None] * candidates).sum(-1)
        out = dict(prediction=prediction, interest_vectors=interests, interest_mask=state["interest_mask"])
        if self.training and self.lambda_relation > 0:
            unique = torch.unique(torch.cat([history.reshape(-1), ids.reshape(-1)]))
            unique = unique[unique != 0]
            if unique.numel() > self.relation_sample_size:
                idx = torch.randperm(unique.numel(), device=history.device)[:self.relation_sample_size]
                unique = unique[idx]
            out["_relation_ids"] = unique
        if return_intermediate:
            out.update(state)
            out.update(interest_vectors=interests, interest_weights=weights, user_vector=user,
                       history_vectors=raw, candidate_vectors=candidates)
        if not self._first_batch_checked:
            self._first_batch_checked = True
            for name, tensor in [("srki_history", raw), ("srki_interests", interests),
                                 ("srki_candidates", candidates), ("srki_prediction", prediction)]:
                check_nan_inf(tensor, name)
        return out

    def checkpoint_metadata(self):
        return dict(version=self.VERSION, K=self.K, dim=self.emb_size, H=self.max_his,
                    kernel_source=self.srki_kernel_source, permutation_seed=self.srki_permutation_seed,
                    llm_emb_path=self.llm_emb_path, semantic_rank=self.semantic_rank,
                    norm_eps=NORM_EPS, rank_tol=RANK_TOL)

    def save_model(self, model_path=None):
        path = model_path or self.model_path
        super().save_model(path)
        Path(str(path) + ".srki.json").write_text(json.dumps(self.checkpoint_metadata(), indent=2))

    def load_model(self, model_path=None):
        path = model_path or self.model_path
        metadata = json.loads(Path(str(path) + ".srki.json").read_text())
        if metadata != self.checkpoint_metadata():
            raise ValueError("SRKI checkpoint configuration mismatch, including K/kernel/asset path")
        super().load_model(path)
