# -*- coding: UTF-8 -*-
"""
MyModelV2: Re-submission main model.

PoMRec backbone + 3 modules:
  3.1 LLM Semantic Alignment & Controllable Injection
  3.2 SICR: Semantic-Intent Consistent Reallocation (NEW, replaces old LGD)
  3.3 IPD: Target-Interest Consistency (unchanged from MyModel)
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
    """Read LLM embedding pkl -> table (N1, d), row0 padding ensured."""
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
    """Load pretrained collaborative item emb -> table (N1, emb), row0 padding ensured."""
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
    """Symmetric InfoNCE: L = 0.5*(CE(sim(X,Y)/tau, diag) + CE(sim(Y,X)/tau, diag))"""

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
# MultiInterestExtractor (PoMRec backbone + LLM branch, clean — no LGD)
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

        # --- CF embedding (trainable) ---
        self.i_embeddings = nn.Embedding(item_num, emb_size)

        # --- Position embedding ---
        self.p_embeddings = nn.Embedding(max_his + 1, emb_size)

        # --- LLM table + adapter ---
        self.srs_emb = None
        if self.use_llmemb:
            if not llm_emb_path:
                raise ValueError("use_llmemb=1 but llm_emb_path is empty")

            llm_table = _load_llm_table_pkl(llm_emb_path, expected_num_items_plus1=item_num)
            self.register_buffer("llm_table", llm_table, persistent=False)
            d_llm = llm_table.size(1)

            self.adapter = nn.Sequential(
                nn.Linear(d_llm, d_llm // 2),
                nn.GELU(),
                nn.Linear(d_llm // 2, emb_size),
                nn.LayerNorm(emb_size),
            )

            if self.gamma_trainable:
                self.log_gamma = nn.Parameter(torch.log(torch.exp(torch.tensor(gamma_init)) - 1.0))
            else:
                self.register_buffer("gamma", torch.tensor(float(gamma_init)))

            if srs_emb_path:
                srs_table = _load_srs_emb_pkl(srs_emb_path, expected_num_items_plus1=item_num)
                self.srs_emb = nn.Embedding.from_pretrained(srs_table, freeze=True)

        # --- Prompts ---
        self.max_prompt = 5
        pad_len = max(0, self.max_prompt - self.prompt_num)
        self.register_buffer("prompt_pad", torch.ones(pad_len, emb_size), persistent=False)
        self.prompt1 = nn.Embedding(self.prompt_num, emb_size)
        self.prompt2 = nn.Embedding(self.prompt_num, emb_size)

        # --- Attention blocks ---
        self.W1 = nn.Linear(emb_size, attn_size)
        self.W2 = nn.Linear(attn_size, self.K)
        self.W3 = nn.Linear(emb_size, attn_size)
        self.W4 = nn.Linear(attn_size, 1)

    # ----- embedding getters -----
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

    # ----- attention helper -----
    @staticmethod
    def value2attn(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = values.masked_fill(mask.unsqueeze(-1) == 0, -np.inf)
        values = values.transpose(-1, -2)
        attn = (values - values.max()).softmax(dim=-1)
        return attn.masked_fill(torch.isnan(attn), 0)

    def forward(self, history, lengths, interest_bias=None, return_aux=False):
        """
        Args:
            history: (B, L)
            lengths: (B,)
            interest_bias: (B, K, L) optional SICR attention bias on history positions
            return_aux: if True, also return attention maps and valid mask
        Returns:
            interest_vectors: (B, K, D)
            distri_vectors: (B, D)
            aux (optional): {"attn_hist": (B,K,L), "valid_his": (B,L)}
        """
        B, seq_len = history.shape
        device = history.device

        valid_his = (history > 0).long()

        his_vectors = self.get_item_emb(history)

        len_range = torch.arange(self.max_his, device=device)
        position = (lengths[:, None] - len_range[None, :seq_len]) * valid_his
        his_vectors = his_vectors + self.p_embeddings(position)

        valid_his_ext = torch.cat([valid_his, torch.ones([B, self.max_prompt], device=device)], dim=1)

        # ---- Multi-Interest Extraction ----
        prompt1 = torch.cat([self.prompt_pad.to(device), self.prompt1.weight], dim=0)
        prompt1 = prompt1.unsqueeze(0).expand(B, -1, -1)
        his_vectors_prompt1 = torch.cat([his_vectors, prompt1], dim=1)

        attn_score = self.W2(self.W1(his_vectors_prompt1).tanh())  # (B, L+P, K)

        # SICR bias injection: only on real history positions, not prompts
        if interest_bias is not None:
            bias = interest_bias.transpose(1, 2)  # (B, K, L) -> (B, L, K)
            attn_score[:, :seq_len, :] = attn_score[:, :seq_len, :] + bias

        # Normalize to get attention maps: (B, K, L+P)
        attn_maps = self.value2attn(attn_score, valid_his_ext)

        interest_vectors = (his_vectors_prompt1[:, None, :, :] * attn_maps[:, :, :, None]).sum(-2)

        var = []
        for kk in range(self.K):
            x_mean_2 = (his_vectors_prompt1 - interest_vectors[:, kk:kk + 1, :]) ** 2
            var_k = torch.matmul(attn_maps[:, kk:kk + 1, :], x_mean_2)
            var.append(torch.sqrt(var_k + 1e-12))
        variance = torch.cat(var, 1)
        interest_vectors = interest_vectors + self.lamb * variance

        # ---- Interest Distribution Predict ----
        prompt2 = torch.cat([self.prompt_pad.to(device), self.prompt2.weight], dim=0)
        prompt2 = prompt2.unsqueeze(0).expand(B, -1, -1)
        his_vectors_prompt2 = torch.cat([his_vectors, prompt2], dim=1)

        distri_pred = self.W4(self.W3(his_vectors_prompt2).tanh())
        distri_maps = self.value2attn(distri_pred, valid_his_ext)
        distri_vectors = torch.matmul(distri_maps, his_vectors_prompt2).squeeze(1)

        if return_aux:
            # Extract history-only attention maps, renormalize over history
            attn_hist = attn_maps[:, :, :seq_len]  # (B, K, L)
            attn_hist = attn_hist / (attn_hist.sum(dim=-1, keepdim=True) + 1e-8)
            aux = {
                "attn_hist": attn_hist,
                "valid_his": (history > 0).float(),
            }
            return interest_vectors, distri_vectors, aux

        return interest_vectors, distri_vectors


# =========================
# MyModelV2: PoMRec + LLMAlign + SICR + IPD
# =========================
class MyModelV2(SequentialModel):
    reader = "SeqReader"
    runner = "BaseRunner"

    extra_log_args = [
        "emb_size", "lr",
        "use_emile", "lambda_ipd",
        "use_sicr",
        "sicr_beta",
        "sicr_sem_weight",
        "sicr_intent_weight",
        "sicr_warmup_steps",
        "sicr_detach",
    ]

    @staticmethod
    def parse_model_args(parser):
        # ---- PoMRec backbone ----
        parser.add_argument("--emb_size", type=int, default=64)
        parser.add_argument("--attn_size", type=int, default=8)
        parser.add_argument("--K", type=int, default=3)
        parser.add_argument("--prompt_num", type=int, default=4)
        parser.add_argument("--n_layers", type=int, default=1)
        parser.add_argument("--lamb", type=float, default=3.0)

        # ---- LLM embedding alignment / fusion ----
        parser.add_argument("--use_llmemb", type=int, default=0)
        parser.add_argument("--llm_emb_path", type=str, default="")
        parser.add_argument("--srs_emb_path", type=str, default="")
        parser.add_argument("--alpha", type=float, default=0.001)
        parser.add_argument("--tau", type=float, default=0.2)
        parser.add_argument("--rat_alpha_warmup_steps", type=int, default=5000)
        parser.add_argument("--llm_fuse", type=int, default=1)
        parser.add_argument("--gamma_init", type=float, default=0.1)
        parser.add_argument("--gamma_trainable", type=int, default=0)

        # ---- warm start ----
        parser.add_argument("--init_ckpt", type=str, default="")
        parser.add_argument("--init_strict", type=int, default=0)

        # ---- EMILE (IPD) ----
        parser.add_argument("--use_emile", type=int, default=0)
        parser.add_argument("--lambda_ipd", type=float, default=0.05)
        parser.add_argument("--ipd_margin", type=float, default=0.2)
        parser.add_argument("--ilr_neg_weight", type=float, default=1.0)
        parser.add_argument("--emile_use_fused_itememb", type=int, default=0)
        parser.add_argument("--emile_warmup_steps", type=int, default=5000)

        # ---- SICR (replaces LGD) ----
        parser.add_argument("--use_sicr", type=int, default=1,
                            help="1: enable semantic-intent consistent behavior reallocation")
        parser.add_argument("--sicr_beta", type=float, default=0.05,
                            help="strength of SICR attention bias")
        parser.add_argument("--sicr_sem_weight", type=float, default=0.2,
                            help="weight of semantic interest consistency in SICR")
        parser.add_argument("--sicr_intent_weight", type=float, default=0.3,
                            help="weight of intent consistency in SICR")
        parser.add_argument("--sicr_warmup_steps", type=int, default=5000,
                            help="warmup steps for SICR bias")
        parser.add_argument("--sicr_detach", type=int, default=1,
                            help="1: detach SICR bias for stable training")

        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)

        # base
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

        # emile
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

        # sicr
        self.use_sicr = int(getattr(args, "use_sicr", 1))
        self.sicr_beta = float(getattr(args, "sicr_beta", 0.05))
        self.sicr_sem_weight = float(getattr(args, "sicr_sem_weight", 0.2))
        self.sicr_intent_weight = float(getattr(args, "sicr_intent_weight", 0.3))
        self.sicr_warmup_steps = int(getattr(args, "sicr_warmup_steps", 5000))
        self.sicr_detach = int(getattr(args, "sicr_detach", 1))

        # build modules
        self._define_params()
        self.apply(self.init_weights)

        if self.use_llmemb:
            self.align_loss_func = InfoNCEAlign(tau=self.tau)

        # warm-start
        if self.use_llmemb and self.init_ckpt:
            self.load_model(self.init_ckpt, strict=bool(self.init_strict))
            logging.info(f"[MyModelV2] Warm-start from {self.init_ckpt} (strict={bool(self.init_strict)})")
        else:
            logging.info("[MyModelV2] Train from scratch")

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

    def _sicr_w(self) -> float:
        if self.sicr_warmup_steps <= 0:
            return 1.0
        t = min(self.global_step, self.sicr_warmup_steps)
        return t / float(self.sicr_warmup_steps)

    @staticmethod
    def _cos_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        a = F.normalize(a, dim=-1, eps=eps)
        b = F.normalize(b, dim=-1, eps=eps)
        return (a * b).sum(dim=-1)

    @staticmethod
    def _cos_dist(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return 1.0 - MyModelV2._cos_sim(a, b, eps=eps)

    @staticmethod
    def _bpr_dist(pos_dist: torch.Tensor, neg_dist: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
        return F.softplus((pos_dist - neg_dist) + margin).mean()

    # -------------------------
    # SICR: compute attention bias
    # -------------------------
    def compute_sicr_bias(self, history, iv0, w0, aux0):
        """
        Compute SICR attention bias from three consistency signals.

        Args:
            history: (B, L)
            iv0: (B, K, D) first-pass interest vectors
            w0: (B, K) first-pass interest distribution weights
            aux0: dict with "attn_hist" (B,K,L), "valid_his" (B,L)

        Returns:
            interest_bias: (B, K, L)
        """
        B, L = history.shape
        K = iv0.size(1)

        # 1. CF interest consistency: cos(cf_emb_history, interest_vectors)
        hist_cf = self.interest_extractor.get_cf_emb(history)  # (B, L, D)
        cf_score = self._cos_sim(
            hist_cf[:, None, :, :],   # (B, 1, L, D)
            iv0[:, :, None, :],        # (B, K, 1, D)
        )  # (B, K, L)

        # 2. Intent consistency: cos(cf_emb_history, aggregated query q0)
        q0 = (iv0 * w0[:, :, None]).sum(dim=1)  # (B, D)

        intent_score = self._cos_sim(
            hist_cf,                 # (B, L, D)
            q0[:, None, :],          # (B, 1, D)
        )  # (B, L)

        intent_score = intent_score[:, None, :].expand(-1, K, -1)  # (B, K, L)

        # 3. Semantic interest consistency
        if self.use_llmemb:
            hist_sem = self.interest_extractor.get_llm_emb(history)  # (B, L, D)
            attn_hist = aux0["attn_hist"]                            # (B, K, L)

            sem_centers = torch.matmul(attn_hist, hist_sem)          # (B, K, D)

            sem_score = self._cos_sim(
                hist_sem[:, None, :, :],    # (B, 1, L, D)
                sem_centers[:, :, None, :],  # (B, K, 1, D)
            )  # (B, K, L)
        else:
            sem_score = torch.zeros_like(cf_score)

        # 4. Compose bias
        bias = cf_score \
             + self.sicr_sem_weight * sem_score \
             + self.sicr_intent_weight * intent_score

        # Mask padding positions
        valid = (history > 0).float()
        bias = bias * valid[:, None, :]

        # Stabilize and scale
        bias = torch.clamp(bias, min=-2.0, max=2.0)
        bias = self.sicr_beta * self._sicr_w() * bias

        if self.sicr_detach:
            bias = bias.detach()

        return bias

    # =========================
    # forward
    # =========================
    def forward(self, feed_dict):
        self.global_step += 1

        i_ids = feed_dict["item_id"]          # (B, 1+neg)
        history = feed_dict["history_items"]  # (B, H)
        lengths = feed_dict["lengths"]        # (B,)

        # =========================================================
        # SICR: two-pass interest extraction with consistency bias
        # =========================================================
        if self.use_sicr:
            # Pass 1: extract without bias, get attention maps
            iv0, dv0, aux0 = self.interest_extractor(
                history, lengths,
                interest_bias=None,
                return_aux=True,
            )

            logits0 = self.proj(dv0)              # (B, K)
            w0 = torch.softmax(logits0, dim=-1)   # (B, K)

            # Compute SICR bias from first-pass results
            interest_bias = self.compute_sicr_bias(history, iv0, w0, aux0)

            # Pass 2: extract with SICR bias
            interest_vectors, distri_vectors = self.interest_extractor(
                history, lengths,
                interest_bias=interest_bias,
                return_aux=False,
            )

            # Debug
            if self.training and self.global_step % 1000 == 0:
                logging.info(
                    f"[SICR] step={self.global_step} "
                    f"w={self._sicr_w():.4f} "
                    f"beta={self.sicr_beta:.4f} "
                    f"bias_mean={interest_bias.mean().item():.6f} "
                    f"bias_abs={interest_bias.abs().mean().item():.6f}"
                )
        else:
            interest_vectors, distri_vectors = self.interest_extractor(
                history, lengths,
                interest_bias=None,
                return_aux=False,
            )

        # item embeddings for candidates
        i_vectors = self.interest_extractor.get_item_emb(i_ids)  # (B, C, D)

        # base intent logits (user-only)
        base_logits = self.proj(distri_vectors)  # (B, K)
        w = torch.softmax(base_logits, dim=-1)   # (B, K)

        # Baseline prediction
        u_base = (interest_vectors * w[:, :, None]).sum(dim=1)   # (B, D)
        prediction = (u_base[:, None, :] * i_vectors).sum(dim=-1)  # (B, C)

        out_dict = {"prediction": prediction}

        # EMILE stash
        if self.use_emile:
            out_dict["emile_interest_vectors"] = interest_vectors
            out_dict["emile_user_vector"] = u_base
            out_dict["emile_w"] = w
            out_dict["emile_pos_ids"] = i_ids[:, 0]
            out_dict["emile_neg_ids"] = i_ids[:, 1] if i_ids.size(1) > 1 else None

        # Alignment (pos only)
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

        # EMILE (IPD)
        if self.use_emile:
            neg_ids = out_dict.get("emile_neg_ids", None)
            if neg_ids is not None:
                iv = out_dict["emile_interest_vectors"]  # (B, K, D)
                w = out_dict["emile_w"]                  # (B, K)
                pos_ids = out_dict["emile_pos_ids"]      # (B,)
                neg_ids = neg_ids                        # (B,)

                ie = self.interest_extractor

                if self.emile_use_fused_itememb:
                    pos_v = ie.get_item_emb(pos_ids)
                    neg_v = ie.get_item_emb(neg_ids)
                else:
                    pos_v = ie.get_cf_emb(pos_ids)
                    neg_v = ie.get_cf_emb(neg_ids)

                # IPD
                H_vec = (iv * w[:, :, None]).sum(dim=1)
                d_pos_H = self._cos_dist(pos_v, H_vec)
                d_neg_H = self._cos_dist(neg_v, H_vec)
                d_pos_h = self._cos_dist(pos_v[:, None, :], iv)      # (B, K)
                d_pos_h_best = d_pos_h.min(dim=1).values             # (B,)

                m = self.ipd_margin
                L_ipd = (
                    self._bpr_dist(d_pos_H, d_pos_h_best, margin=m) +
                    self._bpr_dist(d_pos_H, d_neg_H, margin=m) +
                    self._bpr_dist(d_pos_h_best, d_neg_H, margin=m)
                )

                w_em = self._emile_w()
                loss = loss + w_em * (self.lambda_ipd * L_ipd)
                out_dict["loss_ipd"] = L_ipd.detach()

        return loss
