"""ELECTOR-SIL-inspired adaptive episodic memory for ASPCF.

The item encoder, learned K queries, history aggregator, filtered batch Softmax
and relation objective are inherited. This module adds no auxiliary objective.
Only the exported candidate scores receive the already locked frequency prior;
the ranking loss remains exactly the original uncorrected batch Softmax.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from models.sequential.LLMMIRecASPCFSampledSoftmax import LLMMIRecASPCFSampledSoftmax as SS


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_softmax(logits, allowed):
    """Zero output and finite derivatives for fully masked rows."""
    weights = F.softmax(logits.masked_fill(~allowed, torch.finfo(logits.dtype).min), -1)
    weights = weights * allowed.to(weights.dtype)
    return weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)


def segment_ids(gaps, lengths, mode, threshold, min_len=2, max_len=6, fixed_len=4):
    """Online greedy boundaries; only gaps up to the current position are used.

    The unfinished last segment can be a singleton. It is never merged using
    knowledge of the suffix, so extending a prefix cannot revise old boundaries.
    The max_len guard is a structural cap, not another fitted threshold.
    """
    if mode not in ("fixed", "semantic"):
        raise ValueError(mode)
    if not 1 <= min_len <= max_len or fixed_len < 1:
        raise ValueError("Invalid segment lengths")
    B, L = gaps.shape
    valid = torch.arange(L, device=gaps.device)[None] < lengths[:, None]
    running = torch.zeros_like(lengths)
    current = torch.zeros_like(lengths)
    output = []
    for t in range(L):  # <=H iterations; vectorized over the batch.
        if mode == "fixed":
            cut = running >= fixed_len
        else:
            cut = (running >= max_len) | ((running >= min_len) & (gaps[:, t] > threshold))
        cut = cut & valid[:, t] & (t > 0)
        current = current + cut.long()
        running = torch.where(cut, 0, running) + valid[:, t].long()
        output.append(current)
    ids = torch.stack(output, 1) if L else gaps.new_zeros((B, 0), dtype=torch.long)
    return ids, valid


class SILRecomposer(nn.Module):
    """Shared interest-conditioned read of segment and recent-behavior memories."""
    def __init__(self, dim=64, heads=2, recent=3, use_recent=True, use_local=True):
        super().__init__()
        if dim % heads or heads < 1 or recent < 1:
            raise ValueError("Invalid SIL dimensions")
        self.dim, self.heads, self.recent = dim, heads, recent
        self.use_recent, self.use_local = use_recent, use_local
        self.local_input_norm = nn.LayerNorm(dim)
        self.local_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.local_out = nn.Linear(dim, dim, bias=False)
        self.local_output_norm = nn.LayerNorm(dim)
        self.interest_norm = nn.LayerNorm(dim)
        self.recent_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.read_query = nn.Linear(dim, dim, bias=False)
        self.recent_query = nn.Linear(dim, dim, bias=False)
        self.read_key = nn.Linear(dim, dim, bias=False)
        self.read_value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.output.weight)

    def local_bank(self, x, ids, valid):
        B, L, D = x.shape
        x = x.masked_fill(~valid[:, :, None], 0)
        if L == 0 or not self.use_local:
            return x
        qkv = self.local_qkv(self.local_input_norm(x))
        q, k, value = [a.reshape(B, L, self.heads, D // self.heads).transpose(1, 2)
                       for a in qkv.chunk(3, -1)]
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(D // self.heads)
        order = torch.arange(L, device=x.device)
        allowed = ((ids[:, :, None] == ids[:, None, :])
                   & (order[:, None] >= order[None, :])
                   & valid[:, :, None] & valid[:, None, :])
        weights = safe_softmax(logits, allowed[:, None])
        local = (weights @ value).transpose(1, 2).reshape(B, L, D)
        local = self.local_output_norm(x + self.local_out(local))
        return local.masked_fill(~valid[:, :, None], 0)

    def forward(self, x, interests, ids, valid, lengths):
        B, L, D = x.shape
        if interests.ndim != 3 or interests.shape[0] != B or interests.shape[-1] != D:
            raise ValueError("Expected history [B,L,D], interests [B,K,D]")
        if L == 0:
            return torch.zeros_like(interests), {}
        local = self.local_bank(x, ids, valid)
        membership = F.one_hot(ids, num_classes=L).to(x.dtype) * valid[:, :, None]
        sizes = membership.sum(1)
        memory = torch.bmm(membership.transpose(1, 2), local) / sizes.clamp_min(1)[:, :, None]
        memory_valid = sizes > 0
        order = torch.arange(L, device=x.device)[None]
        recent_valid = valid & (order >= (lengths - self.recent).clamp_min(0)[:, None])
        masked = x.masked_fill(~recent_valid[:, :, None], 0)
        recent = masked.sum(1) / recent_valid.sum(1).clamp_min(1)[:, None]
        last = x[torch.arange(B, device=x.device), (lengths - 1).clamp_min(0)]
        recent = (recent + last).masked_fill((lengths == 0)[:, None], 0)
        query = self.read_query(self.interest_norm(interests))
        if self.use_recent:
            query = query + self.recent_query(self.recent_norm(recent))[:, None]
        # Fine-grained memory: one slot for each of the latest real positions.
        # Repeated item IDs at different positions are distinct behavior events.
        fine_positions = lengths[:, None] - self.recent + torch.arange(self.recent, device=x.device)[None]
        fine_valid = (fine_positions >= 0) & (fine_positions < lengths[:, None])
        gather = fine_positions.clamp(0, L - 1)
        fine_memory = x.gather(1, gather[:, :, None].expand(-1, -1, D))
        fine_memory = fine_memory.masked_fill(~fine_valid[:, :, None], 0)
        # Multi-item segment summaries intentionally coexist with their details.
        # A singleton segment whose only event is already in the recent branch
        # is masked out; it must not receive a second copy of attention mass.
        recent_segment_membership = membership * recent_valid[:, :, None]
        singleton_recent = (sizes == 1) & (recent_segment_membership.sum(1) > 0)
        segment_valid = memory_valid & ~singleton_recent
        joint_memory = torch.cat([memory, fine_memory], dim=1)
        joint_valid = torch.cat([segment_valid, fine_valid], dim=1)
        normalized_memory = self.memory_norm(joint_memory)
        logits = torch.bmm(query, self.read_key(normalized_memory).transpose(1, 2)) / math.sqrt(D)
        # One shared softmax, not the sum of two independently normalized reads.
        attention = safe_softmax(logits, joint_valid[:, None])
        read = torch.bmm(attention, self.read_value(joint_memory))
        delta = self.output(read).masked_fill((lengths < 2)[:, None, None], 0)
        return delta, dict(segment_sizes=sizes, memory_attention=attention,
                           segment_memories=memory, recent_context=recent,
                           segment_memory_valid=segment_valid, joint_memory_valid=joint_valid,
                           behavior_memories=fine_memory, behavior_memory_valid=fine_valid,
                           behavior_positions=fine_positions.masked_fill(~fine_valid, -1))


class LLMMIRecASPCFAdaptiveSIL(SS):
    extra_log_args = SS.extra_log_args + [
        "sil_mode", "sil_asset_path", "sil_heads", "sil_recent", "sil_use_recent", "sil_use_local"]

    @staticmethod
    def parse_model_args(parser):
        parser = SS.parse_model_args(parser)
        parser.add_argument("--sil_mode", choices=["baseline", "fixed", "semantic"], default="semantic")
        parser.add_argument("--sil_asset_path", type=str, default="")
        parser.add_argument("--sil_heads", type=int, default=2)
        parser.add_argument("--sil_recent", type=int, default=3)
        parser.add_argument("--sil_use_recent", type=int, choices=[0, 1], default=1)
        parser.add_argument("--sil_use_local", type=int, choices=[0, 1], default=1)
        return parser

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.sil_mode = getattr(args, "sil_mode", "semantic")
        asset_path = Path(getattr(args, "sil_asset_path", "")).expanduser()
        if not asset_path.is_file():
            raise ValueError("An explicitly prepared Train-only --sil_asset_path is required")
        self.sil_asset_sha256 = file_sha(asset_path)
        asset = json.loads(asset_path.read_text())
        if (asset["schema"] != "adaptive_sil_train_v1" or asset["n_items"] != self.item_num
                or asset["dataset"] != getattr(args, "dataset", asset["dataset"])
                or asset["history_max"] != self.max_his
                or asset["semantic_dimension"] != self.item_encoder.llm_table.shape[1]):
            raise ValueError("SIL asset/config mismatch")
        actual = corpus.data_df["train"][["user_id", "item_id", "position"]].to_numpy(np.int64)
        if array_sha(actual) != asset["train_frame_sha256"]:
            raise ValueError("SIL asset is not from this Train frame")
        if file_sha(args.llm_emb_path) != asset["embedding_sha256"]:
            raise ValueError("SIL asset belongs to a different LLM table")
        counts = np.bincount(actual[:, 1], minlength=self.item_num)
        if not np.array_equal(counts, np.asarray(asset["train_item_counts"])):
            raise ValueError("SIL prior is not the raw Train item frequency")
        if asset["lambda_pop"] != .30 or asset["raw_pop_lambda"] != .30 * asset["locked_scale"]:
            raise ValueError("The previously locked frequency prior must be unchanged")
        self.sil_threshold = float(asset["threshold"])
        if not math.isfinite(self.sil_threshold) or not 0 <= self.sil_threshold <= 2:
            raise ValueError("Invalid semantic threshold")
        self.sil_min_len = int(asset["min_segment_length"])
        self.sil_max_len = int(asset["max_segment_length"])
        self.sil_fixed_len = int(asset["fixed_segment_length"])
        self.sil_raw_pop_lambda = float(asset["raw_pop_lambda"])
        self.register_buffer("sil_log_frequency", torch.as_tensor(np.log1p(counts), dtype=torch.float32))
        if self.sil_mode == "baseline":
            self.sil_block = None
        else:
            # Isolate new parameter initialization from ASPCF RNG/state.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20261022)
                self.sil_block = SILRecomposer(
                    self.emb_size, getattr(args, "sil_heads", 2), getattr(args, "sil_recent", 3),
                    bool(getattr(args, "sil_use_recent", 1)), bool(getattr(args, "sil_use_local", 1)))
        if self.sil_mode == "semantic":
            self.register_buffer("sil_semantic_unit",
                                 F.normalize(self.item_encoder.llm_table.detach(), dim=-1), persistent=False)

    def semantic_gaps(self, history):
        z = self.sil_semantic_unit[history]
        first = z.new_zeros(history.shape[0], 1)
        return torch.cat([first, (1 - (z[:, 1:] * z[:, :-1]).sum(-1)).clamp(0, 2)], 1)

    def recompose_interests(self, history, raw_history, lengths, interests):
        if self.sil_block is None:
            return interests, {}
        gaps = self.semantic_gaps(history) if self.sil_mode == "semantic" else raw_history.new_zeros(history.shape)
        ids, valid = segment_ids(gaps, lengths, self.sil_mode, self.sil_threshold,
                                 self.sil_min_len, self.sil_max_len, self.sil_fixed_len)
        delta, info = self.sil_block(raw_history, interests, ids, valid, lengths)
        return interests + delta, dict(sil_delta=delta, sil_segment_ids=ids, **info)

    def corrected_scores(self, raw_scores, candidates):
        # Match the established float64 post-processing and duplicate slot handling.
        frequency = self.sil_log_frequency[candidates].double()
        return raw_scores.double() + self.sil_raw_pop_lambda * (frequency - frequency.mean(-1, keepdim=True))

    def forward(self, feed_dict, return_intermediate=False):
        lengths = feed_dict["lengths"]
        safe_feed = dict(feed_dict)
        if feed_dict["history_items"].shape[1] == 0:
            safe_feed["history_items"] = feed_dict["history_items"].new_zeros((len(lengths), 1))
        # Avoid the inherited all-masked softmax on empty rows, then zero their V.
        safe_feed["lengths"] = lengths.clamp_min(1)
        out = super().forward(safe_feed, True)
        original = out["interest_vectors"].masked_fill((lengths == 0)[:, None, None], 0)
        out["attention_maps"] = out["attention_maps"].masked_fill((lengths == 0)[:, None, None], 0)
        updated, info = self.recompose_interests(
            safe_feed["history_items"], out["history_vectors"], lengths, original)
        user = (updated * out["interest_weights"][:, :, None]).sum(1)
        raw_scores = (user[:, None] * out["candidate_vectors"]).sum(-1)
        out.update(interest_vectors=updated, user_vector=user,
                   prediction=self.corrected_scores(raw_scores, feed_dict["item_id"]))
        if self.training:
            out["_ss_scores"] = user @ out["candidate_vectors"][:, 0].T
        if return_intermediate:
            out.update(base_interest_vectors=original, raw_prediction=raw_scores, **info)
        else:
            out = {k: v for k, v in out.items()
                   if k in ("prediction", "_relation_ids") or k.startswith("_ss_")}
        return out
