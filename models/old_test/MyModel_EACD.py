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
    """
    Read LLM embedding pkl -> table (N1, d), row0 padding ensured.
    Supports:
      - (N1, d) already has row0 padding
      - (N1-1, d) no row0 -> prepend zeros
      - otherwise -> truncate/pad to N1 (safe)
    """
    arr = pickle.load(open(path, "rb"))
    arr = _ensure_2d_np(arr)

    n1 = expected_num_items_plus1
    if arr.shape[0] == n1:
        table = arr
    elif arr.shape[0] == n1 - 1:
        table = np.vstack([np.zeros((1, arr.shape[1]), dtype=arr.dtype), arr])
    else:
        d = arr.shape[1]
        table = np.zeros((n1, d), dtype=arr.dtype)
        take = min(arr.shape[0], n1)
        table[:take] = arr[:take]

    return torch.tensor(table, dtype=torch.float32)


def _load_srs_emb_pkl(path: str, expected_num_items_plus1: int) -> torch.Tensor:
    """
    Load stage0 collaborative item emb -> table (N1, emb), row0 padding ensured.
    Supports (N1, emb) / (N1-1, emb).
    """
    arr = pickle.load(open(path, "rb"))
    arr = _ensure_2d_np(arr)

    n1 = expected_num_items_plus1
    if arr.shape[0] == n1:
        table = arr
    elif arr.shape[0] == n1 - 1:
        table = np.vstack([np.zeros((1, arr.shape[1]), dtype=arr.dtype), arr])
    else:
        d = arr.shape[1]
        table = np.zeros((n1, d), dtype=arr.dtype)
        take = min(arr.shape[0], n1)
        table[:take] = arr[:take]

    return torch.tensor(table, dtype=torch.float32)


# =========================
# Alignment loss
# =========================
class InfoNCEAlign(nn.Module):
    """
    Symmetric InfoNCE alignment:
      L = 0.5*(CE(sim(X,Y)/tau, diag) + CE(sim(Y,X)/tau, diag))
    """

    def __init__(self, tau: float = 0.2):
        super().__init__()
        self.tau = float(tau)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        logits = (x @ y.t()) / self.tau
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


# =========================
# Posterior denoiser
# =========================
class PosteriorDenoiser(nn.Module):
    """
    Extractor-Aggregator Collaborative Denoising (EACD)

    For each history item h_t, estimate a posterior over K interests + 1 noise bucket:
        p(z_t = k | h_t, V^(0), w^(0)),  k in {1..K}
        p(z_t = noise | h_t, V^(0), w^(0))

    Outputs:
        r_interest : (B, L, K)  posterior responsibility for each interest
        r_noise    : (B, L)     posterior probability of noise bucket
        g_keep     : (B, L)     global keep rate = 1 - r_noise
        mass       : (B, K)     evidence distribution induced by denoised history
    """

    def __init__(
        self,
        emb_size: int,
        k: int,
        tau_d: float = 0.2,
        beta_prior: float = 1.0,
        noise_a: float = 8.0,
        noise_b: float = 0.2,
        noise_c: float = 0.0,
    ):
        super().__init__()
        self.emb_size = int(emb_size)
        self.k = int(k)
        self.tau_d = float(tau_d)
        self.beta_prior = float(beta_prior)

        # trainable noise bucket parameters
        self.noise_a = nn.Parameter(torch.tensor(float(noise_a), dtype=torch.float32))
        self.noise_b = nn.Parameter(torch.tensor(float(noise_b), dtype=torch.float32))
        self.noise_c = nn.Parameter(torch.tensor(float(noise_c), dtype=torch.float32))

    def forward(
        self,
        his_cf: torch.Tensor,
        interest_proto: torch.Tensor,
        w_prior: torch.Tensor,
        valid_mask: torch.Tensor,
    ):
        """
        Args:
            his_cf:         (B, L, D)
            interest_proto: (B, K, D)
            w_prior:        (B, K)
            valid_mask:     (B, L), 1 for valid positions
        """
        h = F.normalize(his_cf, dim=-1)
        v = F.normalize(interest_proto, dim=-1)

        # (B, L, K)
        sim = torch.einsum("bld,bkd->blk", h, v)
        interest_logits = sim / self.tau_d + self.beta_prior * torch.log(w_prior[:, None, :] + 1e-8)

        # noise bucket: if an item is far from all interests, it should be assigned to noise more likely
        max_sim = sim.max(dim=-1).values  # (B, L)
        noise_logits = self.noise_a * (self.noise_b - max_sim) + self.noise_c

        all_logits = torch.cat([interest_logits, noise_logits.unsqueeze(-1)], dim=-1)  # (B, L, K+1)

        mask = valid_mask.unsqueeze(-1).float()
        all_logits = all_logits.masked_fill(mask == 0, -1e9)
        posterior = torch.softmax(all_logits, dim=-1) * mask

        r_interest = posterior[..., :self.k]              # (B, L, K)
        r_noise = posterior[..., self.k]                  # (B, L)
        g_keep = 1.0 - r_noise                            # (B, L)

        mass = r_interest.sum(dim=1)                      # (B, K)
        mass = mass / (mass.sum(dim=-1, keepdim=True) + 1e-8)

        return r_interest, r_noise, g_keep, mass


# =========================
# Multi-interest extractor
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

            hidden_llm = max(1, d_llm // 2)
            self.adapter = nn.Sequential(
                nn.Linear(d_llm, hidden_llm),
                nn.GELU(),
                nn.Linear(hidden_llm, emb_size),
                nn.LayerNorm(emb_size),
            )

            if self.gamma_trainable:
                gamma_tensor = torch.tensor(float(gamma_init), dtype=torch.float32)
                self.log_gamma = nn.Parameter(torch.log(torch.exp(gamma_tensor) - 1.0))
            else:
                self.register_buffer("gamma", torch.tensor(float(gamma_init), dtype=torch.float32))

            if srs_emb_path:
                srs_table = _load_srs_emb_pkl(srs_emb_path, expected_num_items_plus1=item_num)
                self.srs_emb = nn.Embedding.from_pretrained(srs_table, freeze=True)

        self.max_prompt = 5
        if self.prompt_num > self.max_prompt:
            raise ValueError(f"prompt_num={self.prompt_num} exceeds max_prompt={self.max_prompt}")
        pad_len = max(0, self.max_prompt - self.prompt_num)
        self.register_buffer("prompt_pad", torch.ones(pad_len, emb_size), persistent=False)
        self.prompt1 = nn.Embedding(self.prompt_num, emb_size)
        self.prompt2 = nn.Embedding(self.prompt_num, emb_size)

        self.W1 = nn.Linear(emb_size, attn_size)
        self.W2 = nn.Linear(attn_size, self.K)
        self.W3 = nn.Linear(emb_size, attn_size)
        self.W4 = nn.Linear(attn_size, 1)

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

    @staticmethod
    def value2attn(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = values.masked_fill(mask.unsqueeze(-1) == 0, -np.inf)
        values = values.transpose(-1, -2)
        attn = (values - values.max()).softmax(dim=-1)
        return attn.masked_fill(torch.isnan(attn), 0)

    def forward(
        self,
        history: torch.Tensor,
        lengths: torch.Tensor,
        interest_prior: torch.Tensor = None,
        global_gate: torch.Tensor = None,
        residual_keep_ratio: float = 0.15,
        prior_strength: float = 1.0,
    ):
        """
        Args:
            history:             (B, L)
            lengths:             (B,)
            interest_prior:      (B, L, K) or None
            global_gate:         (B, L) or None
            residual_keep_ratio: scalar in [0,1]
            prior_strength:      scale for interest-wise posterior prior
        """
        bsz, seq_len = history.shape
        device = history.device

        valid_his = (history > 0).long()
        his_vectors = self.get_item_emb(history)

        len_range = torch.arange(self.max_his, device=device)
        position = (lengths[:, None] - len_range[None, :seq_len]) * valid_his
        his_vectors = his_vectors + self.p_embeddings(position)

        if global_gate is not None:
            g = global_gate * valid_his.float()
            his_vectors = his_vectors * (1.0 - residual_keep_ratio) + his_vectors * g.unsqueeze(-1) * residual_keep_ratio

        valid_his_ext = torch.cat([valid_his, torch.ones([bsz, self.max_prompt], device=device, dtype=valid_his.dtype)], dim=1)

        # ---- Multi-interest extraction ----
        prompt1 = torch.cat([self.prompt_pad.to(device), self.prompt1.weight], dim=0)
        prompt1 = prompt1.unsqueeze(0).expand(bsz, -1, -1)
        his_vectors_prompt1 = torch.cat([his_vectors, prompt1], dim=1)

        attn_score = self.W2(self.W1(his_vectors_prompt1).tanh())  # (B, L+P, K)

        if interest_prior is not None:
            pad_prior = torch.ones(bsz, self.max_prompt, self.K, device=device, dtype=his_vectors.dtype) / float(self.K)
            prior_full = torch.cat([interest_prior, pad_prior], dim=1)
            attn_score = attn_score + prior_strength * torch.log(prior_full + 1e-8)

        attn_score = self.value2attn(attn_score, valid_his_ext)
        interest_vectors = (his_vectors_prompt1[:, None, :, :] * attn_score[:, :, :, None]).sum(-2)

        var = []
        for kk in range(self.K):
            x_mean_2 = (his_vectors_prompt1 - interest_vectors[:, kk:kk + 1, :]) ** 2
            var_k = torch.matmul(attn_score[:, kk:kk + 1, :], x_mean_2)
            var.append(torch.sqrt(var_k + 1e-12))
        variance = torch.cat(var, dim=1)
        interest_vectors = interest_vectors + self.lamb * variance

        # ---- Interest distribution prediction ----
        prompt2 = torch.cat([self.prompt_pad.to(device), self.prompt2.weight], dim=0)
        prompt2 = prompt2.unsqueeze(0).expand(bsz, -1, -1)
        his_vectors_prompt2 = torch.cat([his_vectors, prompt2], dim=1)

        distri_pred = self.W4(self.W3(his_vectors_prompt2).tanh())
        distri_pred = self.value2attn(distri_pred, valid_his_ext)
        distri_vectors = torch.matmul(distri_pred, his_vectors_prompt2).squeeze(1)

        return interest_vectors, distri_vectors


# =========================
# MyModel: PoMRec + LLMAlign + EACD + TIC(IPD-style)
# =========================
class MyModel(SequentialModel):
    reader = "SeqReader"
    runner = "BaseRunner"

    extra_log_args = [
        "use_emile", "lambda_ipd",
        "use_logic_denoise", "logic_denoise_topk", "logic_denoise_r",
        "lambda_ea", "denoise_tau", "denoise_prior_beta", "denoise_extractor_lambda",
        "use_logic_aggr", "lambda_logic_aggr", "logic_lambda_max", "logic_support_temp",
        "logic_gate_a", "logic_gate_b", "logic_denoise_b"
    ]

    @staticmethod
    def parse_model_args(parser):
        # ---- original PoMRec args ----
        parser.add_argument("--emb_size", type=int, default=64)
        parser.add_argument("--attn_size", type=int, default=8)
        parser.add_argument("--K", type=int, default=3)
        parser.add_argument("--prompt_num", type=int, default=4)
        parser.add_argument("--n_layers", type=int, default=1)
        parser.add_argument("--lamb", type=float, default=3.0)

        # ---- LLM alignment ----
        parser.add_argument("--use_llmemb", type=int, default=0)
        parser.add_argument("--llm_emb_path", type=str, default="")
        parser.add_argument("--srs_emb_path", type=str, default="")
        parser.add_argument("--alpha", type=float, default=0.001)
        parser.add_argument("--tau", type=float, default=0.2)
        parser.add_argument("--rat_alpha_warmup_steps", type=int, default=5000)

        # ---- fusion (cf + gamma * llm) ----
        parser.add_argument("--llm_fuse", type=int, default=1)
        parser.add_argument("--gamma_init", type=float, default=0.1)
        parser.add_argument("--gamma_trainable", type=int, default=0)

        # ---- warm start ----
        parser.add_argument("--init_ckpt", type=str, default="")
        parser.add_argument("--init_strict", type=int, default=0)

        # ---- target-interest consistency (kept under original names for compatibility) ----
        parser.add_argument("--use_emile", type=int, default=0)
        parser.add_argument("--lambda_ipd", type=float, default=0.05)
        parser.add_argument("--ipd_margin", type=float, default=0.2)
        parser.add_argument("--ilr_neg_weight", type=float, default=1.0)
        parser.add_argument("--emile_use_fused_itememb", type=int, default=0)
        parser.add_argument("--emile_warmup_steps", type=int, default=5000)

        # ---- EACD (reuses the old logic_denoise switch for minimal script change) ----
        parser.add_argument("--use_logic_denoise", type=int, default=0)
        parser.add_argument("--logic_denoise_alpha", type=float, default=8.0,
                            help="Noise bucket slope a_n in EACD.")
        parser.add_argument("--logic_denoise_warmup_steps", type=int, default=5000)
        parser.add_argument("--logic_denoise_topk", type=int, default=0,
                            help="Reserved for backward compatibility; not used in EACD.")
        parser.add_argument("--logic_denoise_r", type=float, default=0.15,
                            help="Residual keep ratio for global keep gate in EACD.")
        parser.add_argument("--logic_denoise_b", type=float, default=0.2,
                            help="Noise threshold b_n in EACD.")

        # ---- new denoiser-specific controls ----
        parser.add_argument("--denoise_tau", type=float, default=0.2)
        parser.add_argument("--denoise_prior_beta", type=float, default=1.0)
        parser.add_argument("--denoise_noise_c", type=float, default=0.0)
        parser.add_argument("--denoise_extractor_lambda", type=float, default=1.0,
                            help="Strength of posterior interest prior injected into extractor attention.")
        parser.add_argument("--lambda_ea", type=float, default=0.02,
                            help="KL consistency weight between denoised evidence mass and aggregator interest distribution.")

        # ---- legacy logic_aggr args kept only for script compatibility ----
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

        # llm
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

        # target-interest consistency
        self.use_emile = int(getattr(args, "use_emile", 0))
        self.lambda_ipd = float(getattr(args, "lambda_ipd", 0.1))
        self.ipd_margin = float(getattr(args, "ipd_margin", 0.2))
        self.emile_use_fused_itememb = int(getattr(args, "emile_use_fused_itememb", 0))
        self.emile_warmup_steps = int(getattr(args, "emile_warmup_steps", 5000))

        if self.use_emile:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(getattr(args, "random_seed", 1)) + 2027)
            self.register_buffer("emile_T", torch.randn(self.emb_size, generator=g))
        else:
            self.register_buffer("emile_T", torch.zeros(self.emb_size))

        # EACD
        self.use_logic_denoise = int(getattr(args, "use_logic_denoise", 0))
        self.logic_denoise_alpha = float(getattr(args, "logic_denoise_alpha", 8.0))
        self.logic_denoise_warmup_steps = int(getattr(args, "logic_denoise_warmup_steps", 5000))
        self.logic_denoise_topk = int(getattr(args, "logic_denoise_topk", 0))
        self.logic_denoise_r = float(getattr(args, "logic_denoise_r", 0.15))
        self.logic_denoise_b = float(getattr(args, "logic_denoise_b", 0.2))
        self.denoise_tau = float(getattr(args, "denoise_tau", 0.2))
        self.denoise_prior_beta = float(getattr(args, "denoise_prior_beta", 1.0))
        self.denoise_noise_c = float(getattr(args, "denoise_noise_c", 0.0))
        self.denoise_extractor_lambda = float(getattr(args, "denoise_extractor_lambda", 1.0))
        self.lambda_ea = float(getattr(args, "lambda_ea", 0.02))

        # legacy compatibility fields
        self.use_logic_aggr = int(getattr(args, "use_logic_aggr", 0))
        self.lambda_logic_aggr = float(getattr(args, "lambda_logic_aggr", 0.0))

        self._define_params()
        self.apply(self.init_weights)

        if self.use_llmemb:
            self.align_loss_func = InfoNCEAlign(tau=self.tau)

        if self.use_llmemb and self.init_ckpt:
            self.load_model(self.init_ckpt, strict=bool(self.init_strict))
            logging.info(f"[MyModel] Warm-start from {self.init_ckpt} (strict={bool(self.init_strict)})")
        else:
            logging.info("[MyModel] Train from scratch")

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

        self.posterior_denoiser = PosteriorDenoiser(
            emb_size=self.emb_size,
            k=self.K,
            tau_d=self.denoise_tau,
            beta_prior=self.denoise_prior_beta,
            noise_a=self.logic_denoise_alpha,
            noise_b=self.logic_denoise_b,
            noise_c=self.denoise_noise_c,
        )

        self.proj = nn.Sequential()
        for i in range(max(0, self.n_layers - 1)):
            self.proj.add_module(f"proj_{i}", nn.Linear(self.emb_size, self.emb_size))
            self.proj.add_module(f"dropout_{i}", nn.Dropout(p=0.5))
            self.proj.add_module(f"relu_{i}", nn.ReLU(inplace=True))
        self.proj.add_module("proj_final", nn.Linear(self.emb_size, self.K))

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
    # warmup helpers
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

    def _logic_denoise_w(self) -> float:
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
        valid_mask = (history > 0).float()     # (B, H)

        # ---------------------------------------------------------
        # Pass-1: no denoise, get interest prototypes and interest prior
        # Pass-2: EACD estimates posterior responsibilities over interests + noise bucket
        #         and feeds them back into extractor + aggregator consistency
        # ---------------------------------------------------------
        if self.use_logic_denoise and self.training:
            iv0, dv0 = self.interest_extractor(history, lengths)

            with torch.no_grad():
                logits0 = self.proj(dv0)              # (B, K)
                w0 = torch.softmax(logits0, dim=-1)   # (B, K)

            his_cf = self.interest_extractor.get_cf_emb(history)
            r_interest, r_noise, g_keep, mass = self.posterior_denoiser(
                his_cf=his_cf,
                interest_proto=iv0.detach(),
                w_prior=w0.detach(),
                valid_mask=valid_mask,
            )

            keep_ratio = self.logic_denoise_r * self._logic_denoise_w()
            prior_strength = self.denoise_extractor_lambda * self._logic_denoise_w()
            interest_vectors, distri_vectors = self.interest_extractor(
                history,
                lengths,
                interest_prior=r_interest,
                global_gate=g_keep,
                residual_keep_ratio=keep_ratio,
                prior_strength=prior_strength,
            )
        else:
            mass = None
            r_noise = None
            interest_vectors, distri_vectors = self.interest_extractor(history, lengths)

        i_vectors = self.interest_extractor.get_item_emb(i_ids)  # (B, C, D)
        base_logits = self.proj(distri_vectors)                  # (B, K)
        w = torch.softmax(base_logits, dim=-1)                   # (B, K)

        u_base = (interest_vectors * w[:, :, None]).sum(dim=1)   # (B, D)
        pred_base = (u_base[:, None, :] * i_vectors).sum(dim=-1) # (B, C)
        prediction = pred_base

        out_dict = {
            "prediction": prediction,
            "interest_weight": w,
        }

        if mass is not None:
            out_dict["denoise_mass"] = mass
            out_dict["r_noise"] = r_noise

        # target-interest consistency stash
        if self.use_emile:
            out_dict["emile_interest_vectors"] = interest_vectors
            out_dict["emile_user_vector"] = u_base
            out_dict["emile_w"] = w
            out_dict["emile_pos_ids"] = i_ids[:, 0]
            out_dict["emile_neg_ids"] = i_ids[:, 1] if i_ids.size(1) > 1 else None

        # alignment (pos only)
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

        # extractor-aggregator consistency
        if ("denoise_mass" in out_dict) and ("interest_weight" in out_dict) and (self.lambda_ea > 0):
            mass = out_dict["denoise_mass"].detach()
            w = out_dict["interest_weight"]
            ea_loss = F.kl_div(torch.log(w + 1e-8), mass, reduction="batchmean")
            loss = loss + self._logic_denoise_w() * (self.lambda_ea * ea_loss)
            out_dict["loss_ea"] = ea_loss.detach()

        # target-interest consistency (IPD-style)
        if self.use_emile:
            neg_ids = out_dict.get("emile_neg_ids", None)
            if neg_ids is not None:
                iv = out_dict["emile_interest_vectors"]
                w = out_dict["emile_w"]
                pos_ids = out_dict["emile_pos_ids"]

                ie = self.interest_extractor
                if self.emile_use_fused_itememb:
                    pos_v = ie.get_item_emb(pos_ids)
                    neg_v = ie.get_item_emb(neg_ids)
                else:
                    pos_v = ie.get_cf_emb(pos_ids)
                    neg_v = ie.get_cf_emb(neg_ids)

                h_vec = (iv * w[:, :, None]).sum(dim=1)
                d_pos_h = self._cos_dist(pos_v, h_vec)
                d_neg_h = self._cos_dist(neg_v, h_vec)
                d_pos_interest = self._cos_dist(pos_v[:, None, :], iv)
                d_pos_interest_best = d_pos_interest.min(dim=1).values

                m = self.ipd_margin
                l_ipd = (
                    self._bpr_dist(d_pos_h, d_pos_interest_best, margin=m) +
                    self._bpr_dist(d_pos_h, d_neg_h, margin=m) +
                    self._bpr_dist(d_pos_interest_best, d_neg_h, margin=m)
                )

                loss = loss + self._emile_w() * (self.lambda_ipd * l_ipd)
                out_dict["loss_ipd"] = l_ipd.detach()

        return loss
