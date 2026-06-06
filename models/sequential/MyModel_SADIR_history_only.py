# -*- coding: UTF-8 -*-
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
# PoMRec-style extractor with reusable subroutines
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

    def embed_history_cf(self, history: torch.Tensor, lengths: torch.Tensor):
        valid_his = (history > 0).long()
        pos = self._position_index(history, lengths)
        his_vectors = self.get_cf_emb(history) + self.p_embeddings(pos)
        return his_vectors, valid_his

    def embed_history_sem(self, history: torch.Tensor, lengths: torch.Tensor):
        if not self.use_llmemb:
            raise RuntimeError("embed_history_sem called but use_llmemb=0")
        valid_his = (history > 0).long()
        pos = self._position_index(history, lengths)
        his_vectors = self.get_llm_emb(history) + self.p_embeddings(pos)
        return his_vectors, valid_his

    def _extend_mask(self, valid_his: torch.Tensor) -> torch.Tensor:
        B = valid_his.size(0)
        device = valid_his.device
        return torch.cat([valid_his, torch.ones([B, self.max_prompt], device=device, dtype=valid_his.dtype)], dim=1)

    def extract_interests_from_vectors(self, his_vectors: torch.Tensor, valid_his: torch.Tensor, attn_bias: torch.Tensor = None):
        B = his_vectors.size(0)
        device = his_vectors.device
        valid_his_ext = self._extend_mask(valid_his)

        prompt1 = torch.cat([self.prompt_pad.to(device), self.prompt1.weight], dim=0)
        prompt1 = prompt1.unsqueeze(0).expand(B, -1, -1)
        his_vectors_prompt1 = torch.cat([his_vectors, prompt1], dim=1)

        attn_score = self.W2(self.W1(his_vectors_prompt1).tanh())  # (B,L+P,K)
        if attn_bias is not None:
            if attn_bias.size(1) != his_vectors.size(1):
                raise ValueError(f"attn_bias length mismatch: {attn_bias.size()} vs history len {his_vectors.size()}")
            prompt_bias = torch.zeros(B, self.max_prompt, self.K, device=device, dtype=attn_bias.dtype)
            attn_score = attn_score + torch.cat([attn_bias, prompt_bias], dim=1)

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

    def predict_distribution_from_vectors(self, his_vectors: torch.Tensor, valid_his: torch.Tensor):
        B = his_vectors.size(0)
        device = his_vectors.device
        valid_his_ext = self._extend_mask(valid_his)

        prompt2 = torch.cat([self.prompt_pad.to(device), self.prompt2.weight], dim=0)
        prompt2 = prompt2.unsqueeze(0).expand(B, -1, -1)
        his_vectors_prompt2 = torch.cat([his_vectors, prompt2], dim=1)

        distri_pred = self.W4(self.W3(his_vectors_prompt2).tanh())
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
# SADIR modules
# =========================
class SemanticDriftRouter(nn.Module):
    """
    Semantic-aligned drift-aware task-decoupled routing.
    History-only: it uses historical CF/semantic views, coarse interests, and coarse weights,
    but does NOT consume target positive/negative items.
    It outputs two routing gates for extractor / aggregator and an interest-wise prior.
    """
    def __init__(
        self,
        emb_size: int,
        K: int,
        hidden_size: int = 128,
        assign_tau: float = 0.5,
        transition_lambda_e: float = 0.35,
        transition_lambda_a: float = 0.85,
        residual_e: float = 0.25,
        residual_a: float = 0.20,
        prior_lambda: float = 0.5,
    ):
        super().__init__()
        self.K = int(K)
        self.assign_tau = float(assign_tau)
        self.transition_lambda_e = float(transition_lambda_e)
        self.transition_lambda_a = float(transition_lambda_a)
        self.residual_e = float(residual_e)
        self.residual_a = float(residual_a)
        self.prior_lambda = float(prior_lambda)

        self.state_mlp = nn.Sequential(
            nn.Linear(emb_size * 3 + 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 3),
        )

    @staticmethod
    def _cos_batch_vs_interest(his: torch.Tensor, interests: torch.Tensor) -> torch.Tensor:
        # his: (B,L,D), interests: (B,K,D) -> (B,L,K)
        his = F.normalize(his, dim=-1)
        interests = F.normalize(interests, dim=-1)
        return torch.einsum("bld,bkd->blk", his, interests)

    @staticmethod
    def _js_div_from_probs(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # p,q: (...,K)
        m = 0.5 * (p + q)
        kl_pm = (p * ((p + eps).log() - (m + eps).log())).sum(dim=-1)
        kl_qm = (q * ((q + eps).log() - (m + eps).log())).sum(dim=-1)
        return 0.5 * (kl_pm + kl_qm)

    def forward(
        self,
        his_fused: torch.Tensor,
        his_cf: torch.Tensor,
        his_sem: torch.Tensor,
        valid_his: torch.Tensor,
        coarse_interests: torch.Tensor,
        coarse_weights: torch.Tensor,
        module_coef: float = 1.0,
    ):
        B, L, D = his_fused.shape
        device = his_fused.device
        mask = valid_his.float()

        # interest assignment under CF / semantic views
        a_cf = self._cos_batch_vs_interest(his_cf, coarse_interests)   # (B,L,K)
        a_sem = self._cos_batch_vs_interest(his_sem, coarse_interests) # (B,L,K)
        p_cf = torch.softmax(a_cf / self.assign_tau, dim=-1)
        p_sem = torch.softmax(a_sem / self.assign_tau, dim=-1)

        js = self._js_div_from_probs(p_cf, p_sem)
        consistency = torch.exp(-js).clamp(0.0, 1.0) * mask  # (B,L)

        q0 = (coarse_interests * coarse_weights[:, :, None]).sum(dim=1)  # (B,D)
        q0_expand = q0[:, None, :].expand(-1, L, -1)

        rel_pos = torch.linspace(0.0, 1.0, steps=L, device=device)[None, :, None].expand(B, -1, -1)
        features = torch.cat(
            [his_cf, his_sem, q0_expand, consistency[:, :, None], rel_pos], dim=-1
        )
        state_logits = self.state_mlp(features)
        state_probs = torch.softmax(state_logits, dim=-1) * mask[:, :, None]  # stable / transition / drift-noise

        p_stable = state_probs[..., 0]
        p_trans = state_probs[..., 1]
        # p_noise = state_probs[..., 2]

        conf_e = 0.5 + 0.5 * consistency
        conf_a = 0.7 + 0.3 * consistency
        gate_e = ((p_stable + self.transition_lambda_e * p_trans) * conf_e).clamp(0.0, 1.0) * mask
        gate_a = ((p_stable + self.transition_lambda_a * p_trans) * conf_a).clamp(0.0, 1.0) * mask

        re = self.residual_e * module_coef
        ra = self.residual_a * module_coef
        H_E = his_fused * ((1.0 - re) + re * gate_e[:, :, None])
        H_A = his_fused * ((1.0 - ra) + ra * gate_a[:, :, None])

        # interest-wise extractor prior (semantic+cf agreement weighted by extractor gate)
        prior = 0.5 * (p_cf + p_sem)
        prior = prior * gate_e[:, :, None]
        prior = prior / (prior.sum(dim=-1, keepdim=True) + 1e-8)
        attn_bias = self.prior_lambda * torch.log(prior + 1e-8)

        aux = {
            "state_probs": state_probs,
            "consistency": consistency,
            "gate_e": gate_e,
            "gate_a": gate_a,
            "attn_bias": attn_bias,
        }
        return H_E, H_A, aux


class InterestRefiner(nn.Module):
    def __init__(self, emb_size: int, num_heads: int = 4, ff_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        if emb_size % num_heads != 0:
            num_heads = 1
        self.mha = nn.MultiheadAttention(
            embed_dim=emb_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)
        hidden = max(emb_size, emb_size * ff_mult)
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_size),
        )

    def forward(self, coarse_interests: torch.Tensor, routed_history: torch.Tensor, valid_his: torch.Tensor):
        key_padding_mask = (valid_his == 0)
        attn_out, _ = self.mha(
            query=coarse_interests,
            key=routed_history,
            value=routed_history,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.ln1(coarse_interests + self.drop1(attn_out))
        x = self.ln2(x + self.drop2(self.ffn(x)))
        return x


# =========================
# MyModel + SADIR (history-only second module)
# =========================
class MyModel(SequentialModel):
    reader = "SeqReader"
    runner = "BaseRunner"

    extra_log_args = [
        "use_emile", "lambda_ipd",
        "use_logic_denoise", "logic_denoise_topk", "logic_denoise_r",
        "use_sadir", "lambda_gate_cons",
        # legacy fields
        "use_logic_aggr", "lambda_logic_aggr", "logic_lambda_max", "logic_support_temp",
        "logic_gate_a", "logic_gate_b", "logic_denoise_b"
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

        # legacy LGD fallback
        parser.add_argument("--use_logic_denoise", type=int, default=0)
        parser.add_argument("--logic_denoise_alpha", type=float, default=1.0)
        parser.add_argument("--logic_denoise_warmup_steps", type=int, default=5000)
        parser.add_argument("--logic_denoise_topk", type=int, default=0)
        parser.add_argument("--logic_denoise_r", type=float, default=0.15)
        parser.add_argument("--logic_denoise_b", type=float, default=0.0)

        # new SADIR module
        parser.add_argument("--use_sadir", type=int, default=1)
        parser.add_argument("--sadir_hidden", type=int, default=128)
        parser.add_argument("--sadir_assign_tau", type=float, default=0.5)
        parser.add_argument("--sadir_transition_lambda_e", type=float, default=0.35)
        parser.add_argument("--sadir_transition_lambda_a", type=float, default=0.85)
        parser.add_argument("--sadir_residual_e", type=float, default=0.25)
        parser.add_argument("--sadir_residual_a", type=float, default=0.20)
        parser.add_argument("--sadir_prior_lambda", type=float, default=0.50)
        parser.add_argument("--sadir_refiner_heads", type=int, default=4)
        parser.add_argument("--sadir_refiner_dropout", type=float, default=0.10)
        parser.add_argument("--lambda_gate_cons", type=float, default=0.001)

        # legacy no-op args for compatibility
        parser.add_argument("--use_logic_aggr", type=int, default=0)
        parser.add_argument("--lambda_logic_aggr", type=float, default=0.0)
        parser.add_argument("--logic_lambda_max", type=float, default=0.10)
        parser.add_argument("--logic_support_temp", type=float, default=2.0)
        parser.add_argument("--logic_gate_a", type=float, default=8.0)
        parser.add_argument("--logic_gate_b", type=float, default=0.8)

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

        self.use_sadir = int(getattr(args, "use_sadir", 1))
        self.sadir_hidden = int(getattr(args, "sadir_hidden", 128))
        self.sadir_assign_tau = float(getattr(args, "sadir_assign_tau", 0.5))
        self.sadir_transition_lambda_e = float(getattr(args, "sadir_transition_lambda_e", 0.35))
        self.sadir_transition_lambda_a = float(getattr(args, "sadir_transition_lambda_a", 0.85))
        self.sadir_residual_e = float(getattr(args, "sadir_residual_e", 0.25))
        self.sadir_residual_a = float(getattr(args, "sadir_residual_a", 0.20))
        self.sadir_prior_lambda = float(getattr(args, "sadir_prior_lambda", 0.50))
        self.sadir_refiner_heads = int(getattr(args, "sadir_refiner_heads", 4))
        self.sadir_refiner_dropout = float(getattr(args, "sadir_refiner_dropout", 0.10))
        self.lambda_gate_cons = float(getattr(args, "lambda_gate_cons", 0.001))

        self.use_logic_aggr = int(getattr(args, "use_logic_aggr", 0))
        self.lambda_logic_aggr = float(getattr(args, "lambda_logic_aggr", 0.0))

        self._define_params()
        self.apply(self.init_weights)

        if self.use_llmemb:
            self.align_loss_func = InfoNCEAlign(tau=self.tau)

        if self.use_llmemb and self.init_ckpt:
            self.load_model(self.init_ckpt, strict=bool(self.init_strict))
            logging.info(f"[MyModel-SADIR-HO] Warm-start from {self.init_ckpt} (strict={bool(self.init_strict)})")
        else:
            logging.info("[MyModel-SADIR-HO] Train from scratch")

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

        self.sadir_router = SemanticDriftRouter(
            emb_size=self.emb_size,
            K=self.K,
            hidden_size=self.sadir_hidden,
            assign_tau=self.sadir_assign_tau,
            transition_lambda_e=self.sadir_transition_lambda_e,
            transition_lambda_a=self.sadir_transition_lambda_a,
            residual_e=self.sadir_residual_e,
            residual_a=self.sadir_residual_a,
            prior_lambda=self.sadir_prior_lambda,
        )
        self.interest_refiner = InterestRefiner(
            emb_size=self.emb_size,
            num_heads=self.sadir_refiner_heads,
            dropout=self.sadir_refiner_dropout,
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

    def _sadir_w(self) -> float:
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

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
        return (x * mask).sum() / (mask.sum() + eps)

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

        # =========================
        # Coarse pass
        # =========================
        his_fused, valid_his = self.interest_extractor.embed_history_fused(history, lengths)
        coarse_interests, _ = self.interest_extractor.extract_interests_from_vectors(his_fused, valid_his)
        coarse_distri, _ = self.interest_extractor.predict_distribution_from_vectors(his_fused, valid_his)
        coarse_logits = self.proj(coarse_distri)
        coarse_w = torch.softmax(coarse_logits, dim=-1)

        if self.use_sadir:
            if not self.use_llmemb:
                raise RuntimeError("SADIR requires use_llmemb=1 because semantic-aligned drift signals are used.")

            his_cf, _ = self.interest_extractor.embed_history_cf(history, lengths)
            his_sem, _ = self.interest_extractor.embed_history_sem(history, lengths)
            module_coef = self._sadir_w() if self.training else 1.0

            H_E, H_A, aux = self.sadir_router(
                his_fused=his_fused,
                his_cf=his_cf,
                his_sem=his_sem,
                valid_his=valid_his,
                coarse_interests=coarse_interests.detach(),
                coarse_weights=coarse_w.detach(),
                module_coef=module_coef,
            )

            seed_interests, _ = self.interest_extractor.extract_interests_from_vectors(
                H_E, valid_his, attn_bias=aux["attn_bias"]
            )
            interest_vectors = self.interest_refiner(seed_interests, H_E, valid_his)
            distri_vectors, _ = self.interest_extractor.predict_distribution_from_vectors(H_A, valid_his)
        elif self.use_logic_denoise and self.training:
            # Legacy fallback path
            iv0, dv0 = self.interest_extractor(history, lengths, q_vec=None)
            with torch.no_grad():
                base_logits0 = self.proj(dv0)
                w0 = torch.softmax(base_logits0, dim=-1)
                q_vec = (iv0 * w0[:, :, None]).sum(dim=1)
            self.interest_extractor.use_logic_denoise = 1
            self.interest_extractor.logic_denoise_alpha = self.logic_denoise_alpha * self._sadir_w()
            self.interest_extractor.logic_denoise_topk = self.logic_denoise_topk
            self.interest_extractor.logic_denoise_r = self.logic_denoise_r * self._sadir_w()
            self.interest_extractor.logic_denoise_b = self.logic_denoise_b
            interest_vectors, distri_vectors = self.interest_extractor(history, lengths, q_vec=q_vec)
        else:
            interest_vectors = coarse_interests
            distri_vectors = coarse_distri

        # final aggregator prediction
        base_logits = self.proj(distri_vectors)
        w = torch.softmax(base_logits, dim=-1)
        user_vector = (interest_vectors * w[:, :, None]).sum(dim=1)
        prediction = (user_vector[:, None, :] * i_vectors).sum(dim=-1)

        out_dict = {"prediction": prediction}

        # stash refined interests for TIC/IPD
        if self.use_emile:
            out_dict["emile_interest_vectors"] = interest_vectors
            out_dict["emile_user_vector"] = user_vector
            out_dict["emile_w"] = w
            out_dict["emile_pos_ids"] = i_ids[:, 0]
            out_dict["emile_neg_ids"] = i_ids[:, 1] if i_ids.size(1) > 1 else None

        # alignment (positive items only)
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

        # SADIR auxiliary outputs (history-only; no target-conditioned supervision here)
        if self.use_sadir:
            out_dict["interest_weight"] = w
            out_dict["gate_e"] = aux["gate_e"]
            out_dict["gate_a"] = aux["gate_a"]
            out_dict["gate_mask"] = valid_his.float()
            out_dict["state_probs"] = aux["state_probs"]
            out_dict["consistency"] = aux["consistency"]

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

        # SADIR extractor/aggregator gate consistency (weak coupling)
        if self.use_sadir and ("gate_e" in out_dict):
            gate_e = out_dict["gate_e"]
            gate_a = out_dict["gate_a"]
            mask = out_dict["gate_mask"]
            L_gate = self._masked_mean((gate_e - gate_a) ** 2, mask)
            loss = loss + self._sadir_w() * (self.lambda_gate_cons * L_gate)
            out_dict["loss_gate_cons"] = L_gate.detach()

        return loss
