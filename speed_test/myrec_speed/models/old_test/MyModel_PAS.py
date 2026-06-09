# -*- coding: UTF-8 -*-
"""
MyModel_PAS.py

Replaceable implementation for the paper/code line:
  PoMRec backbone + LLM semantic alignment/injection + PAS history-only sampler + TIC/IPD.

PAS = Period-aware Semantic-Collaborative Adaptive Sampler.
Important properties:
  1) History-only second module: PAS never uses target positive/negative items.
  2) Task-decoupled evidence selection: separate masks for extractor and aggregator.
  3) Period-aware quota: long/mid/recent segments are protected to reduce over-denoising.
  4) Semantic-collaborative signals: sampler uses aligned semantic and CF item views.
  5) Training uses straight-through period top-k masks by default; warmup makes it stable.
"""

import logging
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import SequentialModel


# =========================
# Utils: load embedding tables
# =========================
def _ensure_2d_np(x):
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"embedding must be 2D array, got shape {x.shape}")
    return x


def _load_llm_table_pkl(path: str, expected_num_items_plus1: int) -> torch.Tensor:
    arr = pickle.load(open(path, "rb"))
    arr = _ensure_2d_np(arr)

    N1 = expected_num_items_plus1
    if arr.shape[0] == N1:
        table = arr
    elif arr.shape[0] == N1 - 1:
        table = np.vstack([np.zeros((1, arr.shape[1]), dtype=arr.dtype), arr])
    else:
        d = arr.shape[1]
        table = np.zeros((N1, d), dtype=arr.dtype)
        take = min(arr.shape[0], N1)
        table[:take] = arr[:take]
    return torch.tensor(table, dtype=torch.float32)


def _load_srs_emb_pkl(path: str, expected_num_items_plus1: int) -> torch.Tensor:
    arr = pickle.load(open(path, "rb"))
    arr = _ensure_2d_np(arr)

    N1 = expected_num_items_plus1
    if arr.shape[0] == N1:
        table = arr
    elif arr.shape[0] == N1 - 1:
        table = np.vstack([np.zeros((1, arr.shape[1]), dtype=arr.dtype), arr])
    else:
        d = arr.shape[1]
        table = np.zeros((N1, d), dtype=arr.dtype)
        take = min(arr.shape[0], N1)
        table[:take] = arr[:take]
    return torch.tensor(table, dtype=torch.float32)


# =========================
# Alignment loss
# =========================
class InfoNCEAlign(nn.Module):
    def __init__(self, tau: float = 0.2):
        super().__init__()
        self.tau = float(tau)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        X = F.normalize(X, dim=-1)
        Y = F.normalize(Y, dim=-1)
        logits = (X @ Y.t()) / self.tau
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


# =========================
# PoMRec-style multi-interest extractor
# =========================
class MultiInterestExtractor(nn.Module):
    def __init__(
        self,
        k: int,
        item_num: int,
        emb_size: int,
        attn_size: int,
        max_his: int,
        prompt_num: int,
        lamb: float,
        use_llmemb: int = 0,
        llm_emb_path: str = "",
        srs_emb_path: str = "",
        llm_fuse: int = 1,
        gamma_init: float = 0.05,
        gamma_trainable: int = 1,
    ):
        super().__init__()
        self.K = int(k)
        self.max_his = int(max_his)
        self.prompt_num = int(prompt_num)
        self.lamb = float(lamb)
        self.emb_size = int(emb_size)

        self.use_llmemb = int(use_llmemb)
        self.llm_fuse = int(llm_fuse)
        self.gamma_trainable = int(gamma_trainable)

        self.i_embeddings = nn.Embedding(item_num, emb_size)
        self.p_embeddings = nn.Embedding(max_his + 1, emb_size)

        self.srs_emb = None
        if self.use_llmemb:
            if not llm_emb_path:
                raise ValueError("use_llmemb=1 but llm_emb_path is empty")
            llm_table = _load_llm_table_pkl(llm_emb_path, expected_num_items_plus1=item_num)
            self.register_buffer("llm_table", llm_table, persistent=False)
            d_llm = llm_table.size(1)
            hid = max(emb_size, d_llm // 2)
            self.adapter = nn.Sequential(
                nn.Linear(d_llm, hid),
                nn.GELU(),
                nn.Linear(hid, emb_size),
                nn.LayerNorm(emb_size),
            )
            if self.gamma_trainable:
                self.log_gamma = nn.Parameter(torch.log(torch.exp(torch.tensor(gamma_init)) - 1.0))
            else:
                self.register_buffer("gamma", torch.tensor(float(gamma_init)))
            if srs_emb_path:
                srs_table = _load_srs_emb_pkl(srs_emb_path, expected_num_items_plus1=item_num)
                self.srs_emb = nn.Embedding.from_pretrained(srs_table, freeze=True)

        self.max_prompt = 5
        pad_len = max(0, self.max_prompt - self.prompt_num)
        self.register_buffer("prompt_pad", torch.ones(pad_len, emb_size), persistent=False)
        self.prompt1 = nn.Embedding(self.prompt_num, emb_size)
        self.prompt2 = nn.Embedding(self.prompt_num, emb_size)

        self.W1 = nn.Linear(emb_size, attn_size)
        self.W2 = nn.Linear(attn_size, self.K)
        self.W3 = nn.Linear(emb_size, attn_size)
        self.W4 = nn.Linear(attn_size, 1)

        # legacy LGD runtime flags for fallback
        self.logic_denoise_b = 0.0
        self.use_logic_denoise = 0
        self.logic_denoise_alpha = 1.0
        self.logic_denoise_topk = 0
        self.logic_denoise_r = 0.15

    # ----- getters -----
    def get_cf_emb(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.i_embeddings(item_ids)

    def get_llm_emb(self, item_ids: torch.Tensor) -> torch.Tensor:
        if not self.use_llmemb:
            raise RuntimeError("get_llm_emb called but use_llmemb=0")
        z = self.llm_table[item_ids]
        return self.adapter(z)

    def get_anchor_emb(self, item_ids: torch.Tensor) -> torch.Tensor:
        if self.use_llmemb and (self.srs_emb is not None):
            return self.srs_emb(item_ids)
        return self.get_cf_emb(item_ids)

    def get_gamma(self) -> torch.Tensor:
        if hasattr(self, "log_gamma"):
            return F.softplus(self.log_gamma)
        return self.gamma

    def get_item_emb(self, item_ids: torch.Tensor) -> torch.Tensor:
        e_cf = self.get_cf_emb(item_ids)
        if (not self.use_llmemb) or (not self.llm_fuse):
            return e_cf
        e_llm = self.get_llm_emb(item_ids)
        g = self.get_gamma()
        return e_cf + g * e_llm

    # ----- reusable helpers -----
    @staticmethod
    def value2attn(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = values.masked_fill(mask.unsqueeze(-1) == 0, -np.inf)
        values = values.transpose(-1, -2)
        attn = (values - values.max()).softmax(dim=-1)
        return attn.masked_fill(torch.isnan(attn), 0)

    def _position_index(self, history: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, seq_len = history.shape
        device = history.device
        valid_his = (history > 0).long()
        len_range = torch.arange(self.max_his, device=device)
        position = (lengths[:, None] - len_range[None, :seq_len]) * valid_his
        return position

    def embed_history_fused(self, history: torch.Tensor, lengths: torch.Tensor):
        valid_his = (history > 0).long()
        pos = self._position_index(history, lengths)
        his_vectors = self.get_item_emb(history) + self.p_embeddings(pos)
        return his_vectors, valid_his

    def embed_history_cf(self, history: torch.Tensor, lengths: torch.Tensor, add_pos: bool = True):
        valid_his = (history > 0).long()
        his_vectors = self.get_cf_emb(history)
        if add_pos:
            pos = self._position_index(history, lengths)
            his_vectors = his_vectors + self.p_embeddings(pos)
        return his_vectors, valid_his

    def embed_history_sem(self, history: torch.Tensor, lengths: torch.Tensor, add_pos: bool = True):
        if not self.use_llmemb:
            raise RuntimeError("embed_history_sem called but use_llmemb=0")
        valid_his = (history > 0).long()
        his_vectors = self.get_llm_emb(history)
        if add_pos:
            pos = self._position_index(history, lengths)
            his_vectors = his_vectors + self.p_embeddings(pos)
        return his_vectors, valid_his

    def _extend_mask(self, valid_his: torch.Tensor) -> torch.Tensor:
        B = valid_his.size(0)
        device = valid_his.device
        return torch.cat([valid_his, torch.ones([B, self.max_prompt], device=device, dtype=valid_his.dtype)], dim=1)

    def _append_prompt_bias(self, bias: torch.Tensor, B: int, out_dim: int, device, dtype):
        if bias is None:
            return None
        prompt_bias = torch.zeros(B, self.max_prompt, out_dim, device=device, dtype=dtype)
        return torch.cat([bias, prompt_bias], dim=1)

    def extract_interests_from_vectors(
        self,
        his_vectors: torch.Tensor,
        valid_his: torch.Tensor,
        attn_bias: torch.Tensor = None,
        sample_bias: torch.Tensor = None,
    ):
        B = his_vectors.size(0)
        device = his_vectors.device
        valid_his_ext = self._extend_mask(valid_his)

        prompt1 = torch.cat([self.prompt_pad.to(device), self.prompt1.weight], dim=0)
        prompt1 = prompt1.unsqueeze(0).expand(B, -1, -1)
        his_vectors_prompt1 = torch.cat([his_vectors, prompt1], dim=1)

        attn_score = self.W2(self.W1(his_vectors_prompt1).tanh())  # (B,L+P,K)
        if attn_bias is not None:
            if attn_bias.dim() != 3 or attn_bias.size(1) != his_vectors.size(1):
                raise ValueError(f"attn_bias length mismatch: {attn_bias.size()} vs history len {his_vectors.size()}")
            if attn_bias.size(-1) == 1:
                attn_bias = attn_bias.expand(-1, -1, self.K)
            attn_score = attn_score + self._append_prompt_bias(attn_bias, B, self.K, device, attn_bias.dtype)
        if sample_bias is not None:
            if sample_bias.dim() == 2:
                sample_bias = sample_bias[:, :, None]
            if sample_bias.size(-1) == 1:
                sample_bias = sample_bias.expand(-1, -1, self.K)
            attn_score = attn_score + self._append_prompt_bias(sample_bias, B, self.K, device, sample_bias.dtype)

        attn = self.value2attn(attn_score, valid_his_ext)
        interest_vectors = (his_vectors_prompt1[:, None, :, :] * attn[:, :, :, None]).sum(-2)

        var = []
        for kk in range(self.K):
            x_mean_2 = (his_vectors_prompt1 - interest_vectors[:, kk:kk + 1, :]) ** 2
            var_k = torch.matmul(attn[:, kk:kk + 1, :], x_mean_2)
            var.append(torch.sqrt(var_k + 1e-12))
        variance = torch.cat(var, dim=1)
        interest_vectors = interest_vectors + self.lamb * variance
        return interest_vectors, attn

    def predict_distribution_from_vectors(
        self,
        his_vectors: torch.Tensor,
        valid_his: torch.Tensor,
        sample_bias: torch.Tensor = None,
    ):
        B = his_vectors.size(0)
        device = his_vectors.device
        valid_his_ext = self._extend_mask(valid_his)

        prompt2 = torch.cat([self.prompt_pad.to(device), self.prompt2.weight], dim=0)
        prompt2 = prompt2.unsqueeze(0).expand(B, -1, -1)
        his_vectors_prompt2 = torch.cat([his_vectors, prompt2], dim=1)

        distri_pred = self.W4(self.W3(his_vectors_prompt2).tanh())  # (B,L+P,1)
        if sample_bias is not None:
            if sample_bias.dim() == 2:
                sample_bias = sample_bias[:, :, None]
            if sample_bias.size(-1) != 1:
                sample_bias = sample_bias.mean(dim=-1, keepdim=True)
            distri_pred = distri_pred + self._append_prompt_bias(sample_bias, B, 1, device, sample_bias.dtype)
        distri_pred = self.value2attn(distri_pred, valid_his_ext)
        distri_vectors = torch.matmul(distri_pred, his_vectors_prompt2).squeeze(1)
        return distri_vectors, distri_pred

    def forward(self, history: torch.Tensor, lengths: torch.Tensor, q_vec=None):
        his_vectors, valid_his = self.embed_history_fused(history, lengths)

        # legacy fallback LGD
        if getattr(self, "use_logic_denoise", 0) and (q_vec is not None):
            his_for_gate = self.get_cf_emb(history)
            q = F.normalize(q_vec, dim=-1)
            hv = F.normalize(his_for_gate, dim=-1)
            sim = (hv * q[:, None, :]).sum(dim=-1)
            b = float(getattr(self, "logic_denoise_b", 0.0))
            gate = torch.sigmoid(self.logic_denoise_alpha * (sim - b))
            gate = gate * (history > 0).float()
            topk = int(getattr(self, "logic_denoise_topk", 0))
            if topk > 0 and topk < sim.size(1):
                kth = torch.topk(gate, k=topk, dim=1).values[:, -1]
                gate = gate * 0.9 + 0.1 * (gate >= kth[:, None]).float()
            r = float(getattr(self, "logic_denoise_r", 0.15))
            gate = gate[:, :, None]
            his_vectors = his_vectors * (1 - r) + his_vectors * gate * r

        interest_vectors, _ = self.extract_interests_from_vectors(his_vectors, valid_his)
        distri_vectors, _ = self.predict_distribution_from_vectors(his_vectors, valid_his)
        return interest_vectors, distri_vectors


# =========================
# PAS: Period-aware semantic-collaborative adaptive sampler
# =========================
class PeriodAwareSampler(nn.Module):
    """
    History-only sampler.
    It learns two task-specific evidence masks:
      - extractor mask: cleaner content evidence for multi-interest extraction
      - aggregator mask: broader transition evidence for interest distribution prediction

    No target item, no positive/negative labels are used in this module.
    """
    def __init__(
        self,
        emb_size: int,
        hidden_size: int = 128,
        temp: float = 0.5,
        hard: int = 1,
        use_gumbel: int = 0,
        e_quota_long: int = 2,
        e_quota_mid: int = 3,
        e_quota_recent: int = 5,
        a_quota_long: int = 3,
        a_quota_mid: int = 4,
        a_quota_recent: int = 6,
        rate_e: float = 0.60,
        rate_a: float = 0.75,
    ):
        super().__init__()
        self.temp = float(temp)
        self.hard = int(hard)
        self.use_gumbel = int(use_gumbel)
        self.e_quotas = (int(e_quota_long), int(e_quota_mid), int(e_quota_recent))
        self.a_quotas = (int(a_quota_long), int(a_quota_mid), int(a_quota_recent))
        self.rate_e = float(rate_e)
        self.rate_a = float(rate_a)

        in_dim = emb_size * 3 + 5  # cf, sem, cf*sem, cosine, rel_pos, period one-hot(3)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.head_e = nn.Linear(hidden_size, 1)
        self.head_a = nn.Linear(hidden_size, 1)

    @staticmethod
    def _safe_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(dim=-1)

    @staticmethod
    def _period_ids(L: int, device):
        # 0: long, 1: mid, 2: recent. Index order follows the sequence order.
        idx = torch.arange(L, device=device)
        b1 = L // 3
        b2 = (2 * L) // 3
        pid = torch.zeros(L, device=device, dtype=torch.long)
        pid[idx >= b1] = 1
        pid[idx >= b2] = 2
        return pid

    @staticmethod
    def _sample_gumbel(shape, device, eps: float = 1e-10):
        u = torch.rand(shape, device=device).clamp_(eps, 1.0 - eps)
        return -torch.log(-torch.log(u))

    def _period_topk_mask(self, score: torch.Tensor, valid: torch.Tensor, quotas) -> torch.Tensor:
        # score: (B,L), valid: (B,L) float/bool
        B, L = score.shape
        device = score.device
        valid_bool = valid.bool()
        period_ids = self._period_ids(L, device)
        out = torch.zeros_like(score)

        for p, quota in enumerate(quotas):
            if quota <= 0:
                continue
            seg = (period_ids == p)[None, :].expand(B, -1)
            allowed = valid_bool & seg
            k = min(int(quota), L)
            masked_score = score.masked_fill(~allowed, -1e9)
            top_idx = torch.topk(masked_score, k=k, dim=1).indices
            m = torch.zeros_like(score)
            m.scatter_(1, top_idx, 1.0)
            out = torch.maximum(out, m * allowed.float())
        return out * valid.float()

    def _make_mask(self, logits: torch.Tensor, valid: torch.Tensor, quotas, module_coef: float):
        # prob is differentiable. hard mask is ST-estimated during training.
        score = logits
        if self.training and self.use_gumbel:
            score = score + self._sample_gumbel(score.shape, score.device)
        prob = torch.sigmoid(score / max(self.temp, 1e-6)) * valid.float()

        if self.hard:
            hard = self._period_topk_mask(score, valid, quotas)
            # straight-through mask: hard in forward, prob in backward
            st = hard.detach() - prob.detach() + prob
        else:
            st = prob

        # warmup: at coefficient 0, all valid items are fully available;
        # at coefficient 1, PAS selection is fully applied.
        coef = float(module_coef)
        gate_eff = ((1.0 - coef) + coef * st).clamp(0.0, 1.0) * valid.float()
        return gate_eff, st, prob

    def _period_coverage_loss(self, mask: torch.Tensor, valid: torch.Tensor, quotas) -> torch.Tensor:
        B, L = mask.shape
        device = mask.device
        period_ids = self._period_ids(L, device)
        losses = []
        for p, quota in enumerate(quotas):
            if quota <= 0:
                continue
            seg = (period_ids == p).float()[None, :]
            valid_count = (valid.float() * seg).sum(dim=1)
            required = torch.minimum(torch.full_like(valid_count, float(quota)), valid_count)
            selected = (mask * valid.float() * seg).sum(dim=1)
            denom = required.clamp_min(1.0)
            losses.append((F.relu(required - selected) / denom).mean())
        if not losses:
            return torch.zeros([], device=device)
        return torch.stack(losses).mean()

    def forward(
        self,
        his_cf_raw: torch.Tensor,
        his_sem_raw: torch.Tensor,
        valid_his: torch.Tensor,
        module_coef: float = 1.0,
    ):
        B, L, D = his_cf_raw.shape
        device = his_cf_raw.device
        valid = valid_his.float()

        cos_sc = self._safe_cos(his_cf_raw, his_sem_raw) * valid  # (B,L)
        rel_pos = torch.linspace(0.0, 1.0, steps=L, device=device)[None, :, None].expand(B, -1, -1)
        period_ids = self._period_ids(L, device)
        period_onehot = F.one_hot(period_ids, num_classes=3).float()[None, :, :].expand(B, -1, -1)

        feat = torch.cat(
            [his_cf_raw, his_sem_raw, his_cf_raw * his_sem_raw, cos_sc[:, :, None], rel_pos, period_onehot],
            dim=-1,
        )
        hidden = self.mlp(feat)
        logits_e = self.head_e(hidden).squeeze(-1).masked_fill(valid == 0, -1e9)
        logits_a = self.head_a(hidden).squeeze(-1).masked_fill(valid == 0, -1e9)

        gate_e, mask_e_st, prob_e = self._make_mask(logits_e, valid, self.e_quotas, module_coef)
        gate_a, mask_a_st, prob_a = self._make_mask(logits_a, valid, self.a_quotas, module_coef)

        # auxiliary losses, all history-only
        eps = 1e-8
        rate_e = (mask_e_st * valid).sum(dim=1) / (valid.sum(dim=1) + eps)
        rate_a = (mask_a_st * valid).sum(dim=1) / (valid.sum(dim=1) + eps)
        L_rate = ((rate_e - self.rate_e) ** 2 + (rate_a - self.rate_a) ** 2).mean()

        L_period = 0.5 * (
            self._period_coverage_loss(mask_e_st, valid, self.e_quotas) +
            self._period_coverage_loss(mask_a_st, valid, self.a_quotas)
        )

        # Select high semantic-CF consistency. Detach cos so this regularizes sampler only.
        cos_detach = cos_sc.detach()
        selected = (mask_e_st + mask_a_st).clamp(0.0, 1.0) * valid
        L_sc = -((selected * cos_detach).sum() / (selected.sum() + eps))

        aux = {
            "gate_e": gate_e,
            "gate_a": gate_a,
            "mask_e": mask_e_st,
            "mask_a": mask_a_st,
            "prob_e": prob_e,
            "prob_a": prob_a,
            "cos_sc": cos_sc.detach(),
            "loss_rate": L_rate,
            "loss_period": L_period,
            "loss_sc": L_sc,
        }
        return gate_e, gate_a, aux


# =========================
# MyModel + PAS
# =========================
class MyModel(SequentialModel):
    reader = "SeqReader"
    runner = "BaseRunner"

    extra_log_args = [
        "use_emile", "lambda_ipd",
        "use_pas", "lambda_sampler_rate", "lambda_sampler_period", "lambda_sampler_sc",
        "pas_rate_e", "pas_rate_a",
        # legacy fields
        "use_logic_denoise", "logic_denoise_topk", "logic_denoise_r",
        "use_logic_aggr", "lambda_logic_aggr", "logic_denoise_b",
    ]

    @staticmethod
    def parse_model_args(parser):
        # PoMRec backbone
        parser.add_argument("--emb_size", type=int, default=64)
        parser.add_argument("--attn_size", type=int, default=8)
        parser.add_argument("--K", type=int, default=3)
        parser.add_argument("--prompt_num", type=int, default=4)
        parser.add_argument("--n_layers", type=int, default=1)
        parser.add_argument("--lamb", type=float, default=3.0)

        # LLM alignment / fusion
        parser.add_argument("--use_llmemb", type=int, default=0)
        parser.add_argument("--llm_emb_path", type=str, default="")
        parser.add_argument("--srs_emb_path", type=str, default="")
        parser.add_argument("--alpha", type=float, default=0.001)
        parser.add_argument("--tau", type=float, default=0.2)
        parser.add_argument("--rat_alpha_warmup_steps", type=int, default=5000)
        parser.add_argument("--llm_fuse", type=int, default=1)
        parser.add_argument("--gamma_init", type=float, default=0.1)
        parser.add_argument("--gamma_trainable", type=int, default=0)

        # warm start
        parser.add_argument("--init_ckpt", type=str, default="")
        parser.add_argument("--init_strict", type=int, default=0)

        # TIC / IPD (legacy naming preserved)
        parser.add_argument("--use_emile", type=int, default=0)
        parser.add_argument("--lambda_ipd", type=float, default=0.05)
        parser.add_argument("--ipd_margin", type=float, default=0.2)
        parser.add_argument("--ilr_neg_weight", type=float, default=1.0)
        parser.add_argument("--emile_use_fused_itememb", type=int, default=0)
        parser.add_argument("--emile_warmup_steps", type=int, default=5000)

        # legacy LGD fallback; PAS normally uses only warmup_steps from this group
        parser.add_argument("--use_logic_denoise", type=int, default=0)
        parser.add_argument("--logic_denoise_alpha", type=float, default=1.0)
        parser.add_argument("--logic_denoise_warmup_steps", type=int, default=5000)
        parser.add_argument("--logic_denoise_topk", type=int, default=0)
        parser.add_argument("--logic_denoise_r", type=float, default=0.15)
        parser.add_argument("--logic_denoise_b", type=float, default=0.0)

        # PAS second module
        parser.add_argument("--use_pas", type=int, default=1)
        parser.add_argument("--pas_hidden", type=int, default=128)
        parser.add_argument("--pas_temp", type=float, default=0.5)
        parser.add_argument("--pas_hard", type=int, default=1)
        parser.add_argument("--pas_use_gumbel", type=int, default=0)
        parser.add_argument("--pas_e_quota_long", type=int, default=2)
        parser.add_argument("--pas_e_quota_mid", type=int, default=3)
        parser.add_argument("--pas_e_quota_recent", type=int, default=5)
        parser.add_argument("--pas_a_quota_long", type=int, default=3)
        parser.add_argument("--pas_a_quota_mid", type=int, default=4)
        parser.add_argument("--pas_a_quota_recent", type=int, default=6)
        parser.add_argument("--pas_rate_e", type=float, default=0.60)
        parser.add_argument("--pas_rate_a", type=float, default=0.75)
        parser.add_argument("--lambda_sampler_rate", type=float, default=0.01)
        parser.add_argument("--lambda_sampler_period", type=float, default=0.01)
        parser.add_argument("--lambda_sampler_sc", type=float, default=0.001)

        # legacy no-op args for compatibility
        parser.add_argument("--use_logic_aggr", type=int, default=0)
        parser.add_argument("--lambda_logic_aggr", type=float, default=0.0)
        parser.add_argument("--logic_lambda_max", type=float, default=0.10)
        parser.add_argument("--logic_support_temp", type=float, default=2.0)
        parser.add_argument("--logic_gate_a", type=float, default=8.0)
        parser.add_argument("--logic_gate_b", type=float, default=0.8)

        # old SADIR args accepted but ignored, so previous scripts fail less often
        parser.add_argument("--use_sadir", type=int, default=0)
        parser.add_argument("--lambda_gate_cons", type=float, default=0.0)

        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)

        self.emb_size = args.emb_size
        self.attn_size = args.attn_size
        self.K = args.K
        self.prompt_num = args.prompt_num
        self.n_layers = args.n_layers
        self.lamb = args.lamb
        self.max_his = args.history_max

        self.use_llmemb = int(getattr(args, "use_llmemb", 0))
        self.llm_emb_path = getattr(args, "llm_emb_path", "")
        self.srs_emb_path = getattr(args, "srs_emb_path", "")
        self.alpha = float(getattr(args, "alpha", 0.001))
        self.tau = float(getattr(args, "tau", 0.2))
        self.rat_alpha_warmup_steps = int(getattr(args, "rat_alpha_warmup_steps", 0))

        self.init_ckpt = getattr(args, "init_ckpt", "")
        self.init_strict = int(getattr(args, "init_strict", 0))

        self.llm_fuse = int(getattr(args, "llm_fuse", 1))
        self.gamma_init = float(getattr(args, "gamma_init", 0.05))
        self.gamma_trainable = int(getattr(args, "gamma_trainable", 1))

        self.use_emile = int(getattr(args, "use_emile", 0))
        self.lambda_ipd = float(getattr(args, "lambda_ipd", 0.1))
        self.ipd_margin = float(getattr(args, "ipd_margin", 0.2))
        self.emile_use_fused_itememb = int(getattr(args, "emile_use_fused_itememb", 0))
        self.emile_warmup_steps = int(getattr(args, "emile_warmup_steps", 5000))

        self.use_logic_denoise = int(getattr(args, "use_logic_denoise", 0))
        self.logic_denoise_alpha = float(getattr(args, "logic_denoise_alpha", 1.0))
        self.logic_denoise_warmup_steps = int(getattr(args, "logic_denoise_warmup_steps", 5000))
        self.logic_denoise_topk = int(getattr(args, "logic_denoise_topk", 0))
        self.logic_denoise_r = float(getattr(args, "logic_denoise_r", 0.15))
        self.logic_denoise_b = float(getattr(args, "logic_denoise_b", 0.0))

        self.use_pas = int(getattr(args, "use_pas", 1))
        self.pas_hidden = int(getattr(args, "pas_hidden", 128))
        self.pas_temp = float(getattr(args, "pas_temp", 0.5))
        self.pas_hard = int(getattr(args, "pas_hard", 1))
        self.pas_use_gumbel = int(getattr(args, "pas_use_gumbel", 0))
        self.pas_e_quotas = (
            int(getattr(args, "pas_e_quota_long", 2)),
            int(getattr(args, "pas_e_quota_mid", 3)),
            int(getattr(args, "pas_e_quota_recent", 5)),
        )
        self.pas_a_quotas = (
            int(getattr(args, "pas_a_quota_long", 3)),
            int(getattr(args, "pas_a_quota_mid", 4)),
            int(getattr(args, "pas_a_quota_recent", 6)),
        )
        self.pas_rate_e = float(getattr(args, "pas_rate_e", 0.60))
        self.pas_rate_a = float(getattr(args, "pas_rate_a", 0.75))
        self.lambda_sampler_rate = float(getattr(args, "lambda_sampler_rate", 0.01))
        self.lambda_sampler_period = float(getattr(args, "lambda_sampler_period", 0.01))
        self.lambda_sampler_sc = float(getattr(args, "lambda_sampler_sc", 0.001))

        self.use_logic_aggr = int(getattr(args, "use_logic_aggr", 0))
        self.lambda_logic_aggr = float(getattr(args, "lambda_logic_aggr", 0.0))

        self._define_params()
        self.apply(self.init_weights)

        if self.use_llmemb:
            self.align_loss_func = InfoNCEAlign(tau=self.tau)

        if self.use_llmemb and self.init_ckpt:
            self.load_model(self.init_ckpt, strict=bool(self.init_strict))
            logging.info(f"[MyModel-PAS] Warm-start from {self.init_ckpt} (strict={bool(self.init_strict)})")
        else:
            logging.info("[MyModel-PAS] Train from scratch")

        self.global_step = 0

    def _define_params(self):
        self.interest_extractor = MultiInterestExtractor(
            k=self.K,
            item_num=self.item_num,
            emb_size=self.emb_size,
            attn_size=self.attn_size,
            max_his=self.max_his,
            prompt_num=self.prompt_num,
            lamb=self.lamb,
            use_llmemb=self.use_llmemb,
            llm_emb_path=self.llm_emb_path,
            srs_emb_path=self.srs_emb_path,
            llm_fuse=self.llm_fuse,
            gamma_init=self.gamma_init,
            gamma_trainable=self.gamma_trainable,
        )

        self.proj = nn.Sequential()
        for i in range(max(0, self.n_layers - 1)):
            self.proj.add_module(f"proj_{i}", nn.Linear(self.emb_size, self.emb_size))
            self.proj.add_module(f"dropout_{i}", nn.Dropout(p=0.5))
            self.proj.add_module(f"relu_{i}", nn.ReLU(inplace=True))
        self.proj.add_module("proj_final", nn.Linear(self.emb_size, self.K))

        self.pas_sampler = PeriodAwareSampler(
            emb_size=self.emb_size,
            hidden_size=self.pas_hidden,
            temp=self.pas_temp,
            hard=self.pas_hard,
            use_gumbel=self.pas_use_gumbel,
            e_quota_long=self.pas_e_quotas[0],
            e_quota_mid=self.pas_e_quotas[1],
            e_quota_recent=self.pas_e_quotas[2],
            a_quota_long=self.pas_a_quotas[0],
            a_quota_mid=self.pas_a_quotas[1],
            a_quota_recent=self.pas_a_quotas[2],
            rate_e=self.pas_rate_e,
            rate_a=self.pas_rate_a,
        )

    def load_model(self, model_path=None, strict: bool = False):
        if model_path is None:
            model_path = self.model_path
        model_dict = self.state_dict()
        state_dict = torch.load(model_path, map_location="cpu")
        if strict:
            self.load_state_dict(state_dict, strict=True)
        else:
            exist = {}
            for k, v in state_dict.items():
                if k in model_dict and hasattr(v, "shape") and hasattr(model_dict[k], "shape"):
                    if v.shape == model_dict[k].shape:
                        exist[k] = v
            model_dict.update(exist)
            self.load_state_dict(model_dict, strict=False)
        logging.info("Load model from " + model_path)

    # -------------------------
    # helpers
    # -------------------------
    def _alpha_t(self) -> float:
        if self.rat_alpha_warmup_steps <= 0:
            return self.alpha
        t = min(self.global_step, self.rat_alpha_warmup_steps)
        return self.alpha * (t / float(self.rat_alpha_warmup_steps))

    def _emile_w(self) -> float:
        if self.emile_warmup_steps <= 0:
            return 1.0
        t = min(self.global_step, self.emile_warmup_steps)
        return t / float(self.emile_warmup_steps)

    def _pas_w(self) -> float:
        if self.logic_denoise_warmup_steps <= 0:
            return 1.0
        t = min(self.global_step, self.logic_denoise_warmup_steps)
        return t / float(self.logic_denoise_warmup_steps)

    @staticmethod
    def _cos_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        a = F.normalize(a, dim=-1, eps=eps)
        b = F.normalize(b, dim=-1, eps=eps)
        return (a * b).sum(dim=-1)

    @staticmethod
    def _cos_dist(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return 1.0 - MyModel._cos_sim(a, b, eps=eps)

    @staticmethod
    def _bpr_dist(pos_dist: torch.Tensor, neg_dist: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
        return F.softplus((pos_dist - neg_dist) + margin).mean()

    # =========================
    # forward
    # =========================
    def forward(self, feed_dict):
        self.global_step += 1

        i_ids = feed_dict["item_id"]          # (B, 1+neg)
        history = feed_dict["history_items"]  # (B, H)
        lengths = feed_dict["lengths"]        # (B,)

        # item candidate embeddings
        i_vectors = self.interest_extractor.get_item_emb(i_ids)  # (B,C,D)

        # Base fused historical representation
        his_fused, valid_his = self.interest_extractor.embed_history_fused(history, lengths)

        if self.use_pas:
            if not self.use_llmemb:
                raise RuntimeError("PAS requires use_llmemb=1 because semantic-CF sampler features are used.")
            # Raw item views without positional embedding are used for sampler evidence.
            his_cf_raw, _ = self.interest_extractor.embed_history_cf(history, lengths, add_pos=False)
            his_sem_raw, _ = self.interest_extractor.embed_history_sem(history, lengths, add_pos=False)
            module_coef = self._pas_w() if self.training else 1.0
            gate_e, gate_a, pas_aux = self.pas_sampler(
                his_cf_raw=his_cf_raw,
                his_sem_raw=his_sem_raw,
                valid_his=valid_his,
                module_coef=module_coef,
            )
            eps = 1e-8
            sample_bias_e = torch.log(gate_e + eps)
            sample_bias_a = torch.log(gate_a + eps)
            interest_vectors, _ = self.interest_extractor.extract_interests_from_vectors(
                his_fused, valid_his, sample_bias=sample_bias_e
            )
            distri_vectors, _ = self.interest_extractor.predict_distribution_from_vectors(
                his_fused, valid_his, sample_bias=sample_bias_a
            )
        elif self.use_logic_denoise and self.training:
            # Legacy fallback path: self-conditioned LGD from original model.
            iv0, dv0 = self.interest_extractor(history, lengths, q_vec=None)
            with torch.no_grad():
                base_logits0 = self.proj(dv0)
                w0 = torch.softmax(base_logits0, dim=-1)
                q_vec = (iv0 * w0[:, :, None]).sum(dim=1)
            self.interest_extractor.use_logic_denoise = 1
            self.interest_extractor.logic_denoise_alpha = self.logic_denoise_alpha * self._pas_w()
            self.interest_extractor.logic_denoise_topk = self.logic_denoise_topk
            self.interest_extractor.logic_denoise_r = self.logic_denoise_r * self._pas_w()
            self.interest_extractor.logic_denoise_b = self.logic_denoise_b
            interest_vectors, distri_vectors = self.interest_extractor(history, lengths, q_vec=q_vec)
            pas_aux = None
        else:
            interest_vectors, _ = self.interest_extractor.extract_interests_from_vectors(his_fused, valid_his)
            distri_vectors, _ = self.interest_extractor.predict_distribution_from_vectors(his_fused, valid_his)
            pas_aux = None

        # final aggregator prediction
        base_logits = self.proj(distri_vectors)
        w = torch.softmax(base_logits, dim=-1)
        user_vector = (interest_vectors * w[:, :, None]).sum(dim=1)
        prediction = (user_vector[:, None, :] * i_vectors).sum(dim=-1)

        out_dict = {"prediction": prediction}

        # stash interests for TIC/IPD
        if self.use_emile:
            out_dict["emile_interest_vectors"] = interest_vectors
            out_dict["emile_user_vector"] = user_vector
            out_dict["emile_w"] = w
            out_dict["emile_pos_ids"] = i_ids[:, 0]
            out_dict["emile_neg_ids"] = i_ids[:, 1] if i_ids.size(1) > 1 else None

        # alignment (positive items only; same as original model)
        if self.use_llmemb:
            pos_ids = i_ids[:, 0]
            mask = (pos_ids != 0)
            if mask.any():
                ids = pos_ids[mask]
                srs = self.interest_extractor.get_anchor_emb(ids)
                llm = self.interest_extractor.get_llm_emb(ids)
                out_dict["align_loss"] = self.align_loss_func(srs, llm)
            else:
                out_dict["align_loss"] = torch.zeros([], device=prediction.device)

        # PAS auxiliary losses (history-only; no target-conditioned supervision)
        if self.use_pas and pas_aux is not None:
            out_dict["pas_loss_rate"] = pas_aux["loss_rate"]
            out_dict["pas_loss_period"] = pas_aux["loss_period"]
            out_dict["pas_loss_sc"] = pas_aux["loss_sc"]
            out_dict["pas_gate_e_mean"] = pas_aux["gate_e"].detach().mean()
            out_dict["pas_gate_a_mean"] = pas_aux["gate_a"].detach().mean()

        # gamma debug
        if self.use_llmemb and self.llm_fuse and (self.global_step % 200 == 0):
            ie = self.interest_extractor
            with torch.no_grad():
                g = float(ie.get_gamma().detach().item())
                pos_dbg = i_ids[:, 0]
                pos_dbg = pos_dbg[pos_dbg != 0][:128]
                if pos_dbg.numel() > 0:
                    e_cf = ie.get_cf_emb(pos_dbg)
                    e_llm = ie.get_llm_emb(pos_dbg)
                    ratio = (g * e_llm).norm(dim=-1).mean() / (e_cf.norm(dim=-1).mean() + 1e-12)
                    print(f"[step {self.global_step}] gamma={g:.6f}  llm/cf_norm_ratio={float(ratio):.4f}")

        return out_dict

    # =========================
    # loss
    # =========================
    def loss(self, out_dict: dict):
        loss = super().loss(out_dict)

        # alignment
        if self.use_llmemb and ("align_loss" in out_dict):
            loss = loss + self._alpha_t() * out_dict["align_loss"]

        # TIC / IPD
        if self.use_emile:
            neg_ids = out_dict.get("emile_neg_ids", None)
            if neg_ids is not None:
                iv = out_dict["emile_interest_vectors"]  # (B,K,D)
                w = out_dict["emile_w"]
                pos_ids = out_dict["emile_pos_ids"]
                ie = self.interest_extractor

                if self.emile_use_fused_itememb:
                    pos_v = ie.get_item_emb(pos_ids)
                    neg_v = ie.get_item_emb(neg_ids)
                else:
                    pos_v = ie.get_cf_emb(pos_ids)
                    neg_v = ie.get_cf_emb(neg_ids)

                H_vec = (iv * w[:, :, None]).sum(dim=1)
                d_pos_H = self._cos_dist(pos_v, H_vec)
                d_neg_H = self._cos_dist(neg_v, H_vec)
                d_pos_h = self._cos_dist(pos_v[:, None, :], iv)
                d_pos_h_best = d_pos_h.min(dim=1).values
                m = self.ipd_margin
                L_ipd = (
                    self._bpr_dist(d_pos_H, d_pos_h_best, margin=m) +
                    self._bpr_dist(d_pos_H, d_neg_H, margin=m) +
                    self._bpr_dist(d_pos_h_best, d_neg_H, margin=m)
                )
                loss = loss + self._emile_w() * (self.lambda_ipd * L_ipd)
                out_dict["loss_ipd"] = L_ipd.detach()

        # PAS sampler auxiliary losses, warm-started.
        if self.use_pas and ("pas_loss_rate" in out_dict):
            w_pas = self._pas_w()
            L_rate = out_dict["pas_loss_rate"]
            L_period = out_dict["pas_loss_period"]
            L_sc = out_dict["pas_loss_sc"]
            loss = loss + w_pas * (
                self.lambda_sampler_rate * L_rate +
                self.lambda_sampler_period * L_period +
                self.lambda_sampler_sc * L_sc
            )
            out_dict["loss_pas_rate"] = L_rate.detach()
            out_dict["loss_pas_period"] = L_period.detach()
            out_dict["loss_pas_sc"] = L_sc.detach()

        return loss
