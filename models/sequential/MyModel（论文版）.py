# -*- coding: UTF-8 -*-
import logging
import pickle
import math
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
    """
    Load stage0 collaborative item emb -> table (N1, emb), row0 padding ensured.
    Supports (N1, emb) / (N1-1, emb).
    """
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
# Alignment loss (stable)
# =========================
class InfoNCEAlign(nn.Module):
    """
    Symmetric InfoNCE alignment:
      L = 0.5*(CE(sim(X,Y)/tau, diag) + CE(sim(Y,X)/tau, diag))
    """
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
# MultiInterestExtractor (PoMRec backbone) + LLM branch + LGD
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
                # softplus^{-1}(gamma_init)
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

        # LGD runtime flags (set by outer model)
        self.logic_denoise_b = 0.0
        self.use_logic_denoise = 0
        self.logic_denoise_alpha = 1.0
        self.logic_denoise_topk = 0
        self.logic_denoise_r = 0.15

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

    def forward(self, history: torch.Tensor, lengths: torch.Tensor, q_vec=None):
        B, seq_len = history.shape
        device = history.device

        valid_his = (history > 0).long()

        his_vectors = self.get_item_emb(history)

        len_range = torch.arange(self.max_his, device=device)
        position = (lengths[:, None] - len_range[None, :seq_len]) * valid_his
        his_vectors = his_vectors + self.p_embeddings(position)

        # -----------------------------
        # Logic-guided denoise (TRAIN only; outer model controls q_vec + alpha/r)
        # -----------------------------
        if getattr(self, "use_logic_denoise", 0) and (q_vec is not None):
            his_for_gate = self.get_cf_emb(history)  # more stable
            q = F.normalize(q_vec, dim=-1)
            hv = F.normalize(his_for_gate, dim=-1)
            sim = (hv * q[:, None, :]).sum(dim=-1)

            b = float(getattr(self, "logic_denoise_b", 0.0))
            gate = torch.sigmoid(self.logic_denoise_alpha * (sim - b))
            gate = gate * (history > 0).float()

            if self.training:
                if not hasattr(self, "debug_step"):
                    self.debug_step = 0
                self.debug_step += 1
                if self.debug_step % 1000 == 0:
                    with torch.no_grad():
                        m_gate = float(gate.mean().detach().cpu())
                        p50 = float((gate > 0.5).float().mean().detach().cpu())
                        p80 = float((gate > 0.8).float().mean().detach().cpu())
                        m_sim = float(sim.mean().detach().cpu())
                        print(
                            f"[LGD] step={self.debug_step} alpha={float(self.logic_denoise_alpha):.3f} b={b:.3f} "
                            f"r={float(getattr(self, 'logic_denoise_r', 0.0)):.3f} topk={int(getattr(self, 'logic_denoise_topk', 0))} "
                            f"sim_mean={m_sim:.4f} gate_mean={m_gate:.4f} gate>0.5={p50:.4f} gate>0.8={p80:.4f}"
                        )

            topk = int(getattr(self, "logic_denoise_topk", 0))
            if topk > 0 and topk < sim.size(1):
                kth = torch.topk(gate, k=topk, dim=1).values[:, -1]
                gate = gate * 0.9 + 0.1 * (gate >= kth[:, None]).float()

            r = float(getattr(self, "logic_denoise_r", 0.15))
            gate = gate[:, :, None]
            his_vectors = his_vectors * (1 - r) + his_vectors * gate * r

        valid_his_ext = torch.cat([valid_his, torch.ones([B, self.max_prompt], device=device)], dim=1)

        # ---- Multi-Interest Extraction ----
        prompt1 = torch.cat([self.prompt_pad.to(device), self.prompt1.weight], dim=0)
        prompt1 = prompt1.unsqueeze(0).expand(B, -1, -1)
        his_vectors_prompt1 = torch.cat([his_vectors, prompt1], dim=1)

        attn_score = self.W2(self.W1(his_vectors_prompt1).tanh())
        attn_score = self.value2attn(attn_score, valid_his_ext)

        interest_vectors = (his_vectors_prompt1[:, None, :, :] * attn_score[:, :, :, None]).sum(-2)

        var = []
        for kk in range(self.K):
            x_mean_2 = (his_vectors_prompt1 - interest_vectors[:, kk:kk + 1, :]) ** 2
            var_k = torch.matmul(attn_score[:, kk:kk + 1, :], x_mean_2)
            var.append(torch.sqrt(var_k + 1e-12))
        variance = torch.cat(var, 1)
        interest_vectors = interest_vectors + self.lamb * variance

        # ---- Interest Distribution Predict ----
        prompt2 = torch.cat([self.prompt_pad.to(device), self.prompt2.weight], dim=0)
        prompt2 = prompt2.unsqueeze(0).expand(B, -1, -1)
        his_vectors_prompt2 = torch.cat([his_vectors, prompt2], dim=1)

        distri_pred = self.W4(self.W3(his_vectors_prompt2).tanh())
        distri_pred = self.value2attn(distri_pred, valid_his_ext)
        distri_vectors = torch.matmul(distri_pred, his_vectors_prompt2).squeeze(1)

        return interest_vectors, distri_vectors


# =========================
# MyModel: PoMRec + LLMAlign + EMILE(IPD) + LGD
# (logic_aggr disabled/removed from forward path)
# =========================
class MyModel(SequentialModel):
    reader = "SeqReader"
    runner = "BaseRunner"

    extra_log_args = [
        "use_emile", "lambda_ipd",
        "use_logic_denoise", "logic_denoise_topk", "logic_denoise_r",
        # 仍保留这些字段，避免旧脚本/日志断裂（可删）
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

        # ---- LLMEmb alignment (regularization) ----
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

        # ---- EMILE(IPD) ----
        parser.add_argument("--use_emile", type=int, default=0)
        parser.add_argument("--lambda_ipd", type=float, default=0.05)
        parser.add_argument("--ipd_margin", type=float, default=0.2)
        parser.add_argument("--ilr_neg_weight", type=float, default=1.0)
        parser.add_argument("--emile_use_fused_itememb", type=int, default=0)
        parser.add_argument("--emile_warmup_steps", type=int, default=5000)

        # ---- LGD ----
        parser.add_argument("--use_logic_denoise", type=int, default=0)
        parser.add_argument("--logic_denoise_alpha", type=float, default=1.0)
        parser.add_argument("--logic_denoise_warmup_steps", type=int, default=5000)
        parser.add_argument("--logic_denoise_topk", type=int, default=0)
        parser.add_argument("--logic_denoise_r", type=float, default=0.15)
        parser.add_argument("--logic_denoise_b", type=float, default=0.0)

        # ---- logic_aggr 参数仍保留（兼容旧脚本），但 forward 中不再使用 ----
        parser.add_argument("--use_logic_aggr", type=int, default=0)
        parser.add_argument("--lambda_logic_aggr", type=float, default=0.0)
        parser.add_argument("--logic_lambda_max", type=float, default=0.10)
        parser.add_argument("--logic_support_temp", type=float, default=2.0)
        parser.add_argument("--logic_gate_a", type=float, default=8.0)
        parser.add_argument("--logic_gate_b", type=float, default=0.8)

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

        # denoise
        self.use_logic_denoise = int(getattr(args, "use_logic_denoise", 0))
        self.logic_denoise_alpha = float(getattr(args, "logic_denoise_alpha", 1.0))
        self.logic_denoise_warmup_steps = int(getattr(args, "logic_denoise_warmup_steps", 5000))
        self.logic_denoise_topk = int(getattr(args, "logic_denoise_topk", 0))
        self.logic_denoise_r = float(getattr(args, "logic_denoise_r", 0.15))
        self.logic_denoise_b = float(getattr(args, "logic_denoise_b", 0.0))

        # logic_aggr 仍读取（兼容），但 forward 不使用
        self.use_logic_aggr = int(getattr(args, "use_logic_aggr", 0))
        self.lambda_logic_aggr = float(getattr(args, "lambda_logic_aggr", 0.0))

        # build modules
        self._define_params()
        self.apply(self.init_weights)

        if self.use_llmemb:
            self.align_loss_func = InfoNCEAlign(tau=self.tau)

        # warm-start
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

        # =========================================================
        # Self-conditioned LGD (NO label leakage):
        #   Pass-1: run extractor without denoise -> get dv0 as query
        #   Pass-2: run extractor with denoise using q_vec
        # =========================================================
        if self.use_logic_denoise and self.training:
            self.interest_extractor.use_logic_denoise = 0
            iv0, dv0 = self.interest_extractor(history, lengths, q_vec=None)

            with torch.no_grad():
                base_logits0 = self.proj(dv0)              # (B,K)
                w0 = torch.softmax(base_logits0, dim=-1)   # (B,K)
                q_vec = (iv0 * w0[:, :, None]).sum(dim=1)  # (B,D)

            self.interest_extractor.use_logic_denoise = 1
            self.interest_extractor.logic_denoise_alpha = self.logic_denoise_alpha * self._logic_denoise_w()
            self.interest_extractor.logic_denoise_topk = self.logic_denoise_topk
            self.interest_extractor.logic_denoise_r = self.logic_denoise_r * self._logic_denoise_w()
            # 兼容：仍然把 b 挂到 extractor（如果你未来要用 b-sweep）
            self.interest_extractor.logic_denoise_b = self.logic_denoise_b

            interest_vectors, distri_vectors = self.interest_extractor(history, lengths, q_vec=q_vec)
        else:
            self.interest_extractor.use_logic_denoise = 0
            interest_vectors, distri_vectors = self.interest_extractor(history, lengths, q_vec=None)

        # item embeddings for candidates
        i_vectors = self.interest_extractor.get_item_emb(i_ids)  # (B,C,D)

        # base intent logits (user-only)
        base_logits = self.proj(distri_vectors)  # (B,K)
        w = torch.softmax(base_logits, dim=-1)   # (B,K)

        # Baseline prediction (always)
        u_base = (interest_vectors * w[:, :, None]).sum(dim=1)   # (B,D)
        pred_base = (u_base[:, None, :] * i_vectors).sum(dim=-1) # (B,C)

        # -------- logic_aggr 已移除：永远用 pred_base --------
        prediction = pred_base

        out_dict = {"prediction": prediction}

        # EMILE stash
        if self.use_emile:
            user_vector_for_emile = (interest_vectors * w[:, :, None]).sum(dim=1)
            out_dict["emile_interest_vectors"] = interest_vectors
            out_dict["emile_user_vector"] = user_vector_for_emile
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

        # EMILE(IPD)
        if self.use_emile:
            neg_ids = out_dict.get("emile_neg_ids", None)
            if neg_ids is not None:
                iv = out_dict["emile_interest_vectors"]  # (B,K,D)
                uv = out_dict["emile_user_vector"]       # (B,D)  # kept for future
                w = out_dict["emile_w"]                  # (B,K)
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
                d_pos_h = self._cos_dist(pos_v[:, None, :], iv)      # (B,K)
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