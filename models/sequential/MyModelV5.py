# -*- coding: UTF-8 -*-
"""
MyModelV5: Clean re-submission main model.

PoMRec backbone + 3 modules:
  3.1 LLM Semantic Alignment & Controllable Injection
  3.2 SSID: Semantic-Supervised Interest Disentanglement (training-only auxiliary loss)
  3.3 IPD: Target-Interest Consistency (unchanged from MyModel)
"""

import logging
import math
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

    def forward(self, history, lengths, return_aux=False):
        """
        Args:
            history: (B, L)
            lengths: (B,)
            return_aux: if True, return attn_hist and valid_his
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

        attn_score = self.W2(self.W1(his_vectors_prompt1).tanh())
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
        distri_pred = self.value2attn(distri_pred, valid_his_ext)
        distri_vectors = torch.matmul(distri_pred, his_vectors_prompt2).squeeze(1)

        if return_aux:
            attn_hist = attn_maps[:, :, :seq_len]  # (B, K, L) — history only, no prompts
            attn_hist = attn_hist / (attn_hist.sum(dim=-1, keepdim=True) + 1e-8)
            aux = {
                "attn_hist": attn_hist,
                "valid_his": (history > 0).float(),
            }
            return interest_vectors, distri_vectors, aux

        return interest_vectors, distri_vectors


# =========================
# MyModelV5: PoMRec + LLMAlign + DSPC + IPD
# =========================
class MyModelV5(SequentialModel):
    reader = "SeqReader"
    runner = "BaseRunner"

    extra_log_args = [
        "emb_size", "lr",
        "use_emile", "lambda_ipd",
        "use_dspc",
        "dspc_scale",
        "dspc_temp",
        "dspc_gate",
        "dspc_gate_alpha",
        "dspc_norm",
        "dspc_detach_sem",
        "dspc_dropout",
        "use_ssid",
        "lambda_ssid",
        "ssid_temp",
        "ssid_warmup_steps",
        "ssid_detach_sem",
        "ssid_detach_attn",
        "ssid_use_proto_norm",
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

        # ---- DSPC: Dynamic Semantic Prototype Calibration ----
        parser.add_argument("--use_dspc", type=int, default=0,
                            help="1: enable dynamic semantic prototype calibration")
        parser.add_argument("--dspc_scale", type=float, default=0.05,
                            help="residual scale for DSPC calibration")
        parser.add_argument("--dspc_temp", type=float, default=0.5,
                            help="temperature for semantic prototype attention")
        parser.add_argument("--dspc_gate", type=int, default=1,
                            help="1: gate DSPC calibration by interest-prototype similarity")
        parser.add_argument("--dspc_gate_alpha", type=float, default=2.0,
                            help="sharpness of DSPC gate sigmoid")
        parser.add_argument("--dspc_norm", type=int, default=1,
                            help="1: layernorm calibrated interest vectors")
        parser.add_argument("--dspc_detach_sem", type=int, default=1,
                            help="1: detach semantic embeddings in DSPC")
        parser.add_argument("--dspc_dropout", type=float, default=0.0,
                            help="dropout on DSPC delta")

        # ---- SSID: Semantic-Supervised Interest Disentanglement ----
        parser.add_argument("--use_ssid", type=int, default=0,
                            help="1: enable semantic-supervised interest disentanglement loss")
        parser.add_argument("--lambda_ssid", type=float, default=0.001,
                            help="weight for SSID contrastive loss")
        parser.add_argument("--ssid_temp", type=float, default=0.2,
                            help="temperature for SSID contrastive loss")
        parser.add_argument("--ssid_warmup_steps", type=int, default=5000,
                            help="warmup steps for SSID loss")
        parser.add_argument("--ssid_detach_sem", type=int, default=1,
                            help="1: detach semantic embeddings in SSID")
        parser.add_argument("--ssid_detach_attn", type=int, default=1,
                            help="1: detach attention maps in SSID")
        parser.add_argument("--ssid_use_proto_norm", type=int, default=1,
                            help="1: use normalized vectors in SSID contrastive loss")

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

        # dspc
        self.use_dspc = int(getattr(args, "use_dspc", 0))
        self.dspc_scale = float(getattr(args, "dspc_scale", 0.05))
        self.dspc_temp = float(getattr(args, "dspc_temp", 0.5))
        self.dspc_gate = int(getattr(args, "dspc_gate", 1))
        self.dspc_gate_alpha = float(getattr(args, "dspc_gate_alpha", 2.0))
        self.dspc_norm = int(getattr(args, "dspc_norm", 1))
        self.dspc_detach_sem = int(getattr(args, "dspc_detach_sem", 1))
        self.dspc_dropout_p = float(getattr(args, "dspc_dropout", 0.0))

        self.dspc_ln = nn.LayerNorm(self.emb_size)
        self.dspc_dropout = nn.Dropout(self.dspc_dropout_p)

        # ssid
        self.use_ssid = int(getattr(args, "use_ssid", 0))
        self.lambda_ssid = float(getattr(args, "lambda_ssid", 0.001))
        self.ssid_temp = float(getattr(args, "ssid_temp", 0.2))
        self.ssid_warmup_steps = int(getattr(args, "ssid_warmup_steps", 5000))
        self.ssid_detach_sem = int(getattr(args, "ssid_detach_sem", 1))
        self.ssid_detach_attn = int(getattr(args, "ssid_detach_attn", 1))
        self.ssid_use_proto_norm = int(getattr(args, "ssid_use_proto_norm", 1))

        # build modules
        self._define_params()
        self.apply(self.init_weights)

        if self.use_llmemb:
            self.align_loss_func = InfoNCEAlign(tau=self.tau)

        # warm-start
        if self.use_llmemb and self.init_ckpt:
            self.load_model(self.init_ckpt, strict=bool(self.init_strict))
            logging.info(f"[MyModelV5] Warm-start from {self.init_ckpt} (strict={bool(self.init_strict)})")
        else:
            logging.info("[MyModelV5] Train from scratch")

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

    def _ssid_w(self) -> float:
        if self.ssid_warmup_steps <= 0:
            return 1.0
        t = min(self.global_step, self.ssid_warmup_steps)
        return t / float(self.ssid_warmup_steps)

    @staticmethod
    def _cos_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        a = F.normalize(a, dim=-1, eps=eps)
        b = F.normalize(b, dim=-1, eps=eps)
        return (a * b).sum(dim=-1)

    @staticmethod
    def _cos_dist(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return 1.0 - MyModelV5._cos_sim(a, b, eps=eps)

    @staticmethod
    def _bpr_dist(pos_dist: torch.Tensor, neg_dist: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
        return F.softplus((pos_dist - neg_dist) + margin).mean()

    # -------------------------
    # DSPC: Dynamic Semantic Prototype Calibration
    # -------------------------
    def compute_dspc_interest(self, history, interest_vectors):
        """
        Calibrate interest vectors with semantic prototypes retrieved from
        LLM-adapted history item embeddings.

        Each interest vector serves as a query to attend over history
        semantic embeddings, producing an interest-specific semantic prototype.
        A gated residual then calibrates the original interest vector.

        No target item used — no label leakage.
        No new loss added.

        Args:
            history: (B, L)
            interest_vectors: (B, K, D)

        Returns:
            calibrated_interest_vectors: (B, K, D)
            aux: dict with dspc_attn, dspc_gate_mean, dspc_proto_norm
        """
        hist_sem = self.interest_extractor.get_llm_emb(history)  # (B, L, D)

        if self.dspc_detach_sem:
            hist_sem = hist_sem.detach()

        valid = (history > 0)  # (B, L)

        # Attention: interest_vectors as queries over history semantic embeddings
        q = F.normalize(interest_vectors, dim=-1)         # (B, K, D)
        h = F.normalize(hist_sem, dim=-1)                 # (B, L, D)

        score = torch.einsum("bkd,bld->bkl", q, h) / self.dspc_temp
        score = score.masked_fill(~valid[:, None, :], -1e9)

        attn = torch.softmax(score, dim=-1)               # (B, K, L)
        sem_proto = torch.matmul(attn, hist_sem)          # (B, K, D)

        # Gate: control calibration strength per interest
        if self.dspc_gate:
            sim = (F.normalize(interest_vectors, dim=-1)
                   * F.normalize(sem_proto, dim=-1)).sum(dim=-1)  # (B, K)
            gate = torch.sigmoid(self.dspc_gate_alpha * sim)      # (B, K)
        else:
            gate = torch.ones(interest_vectors.size()[:2],
                              device=interest_vectors.device)

        # Residual calibration
        delta = gate[:, :, None] * sem_proto
        delta = self.dspc_dropout(delta)

        calibrated = interest_vectors + self.dspc_scale * delta

        if self.dspc_norm:
            calibrated = self.dspc_ln(calibrated)

        aux = {
            "dspc_attn": attn.detach(),
            "dspc_gate_mean": gate.detach().mean(),
            "dspc_proto_norm": sem_proto.detach().norm(dim=-1).mean(),
        }

        return calibrated, aux

    # -------------------------
    # SSID: Semantic-Supervised Interest Disentanglement loss
    # -------------------------
    def compute_ssid_loss(self, out_dict):
        """
        Contrastive loss aligning each interest vector with its
        attention-weighted semantic prototype.

        Encourages each interest to attend to semantically consistent
        history items, without changing forward prediction.

        Args:
            out_dict: dict with "ssid_history", "ssid_interest_vectors",
                      "ssid_attn_hist"

        Returns:
            loss: scalar tensor
            aux: dict with debug info
        """
        history = out_dict["ssid_history"]                     # (B, L)
        interest_vectors = out_dict["ssid_interest_vectors"]   # (B, K, D)
        attn_hist = out_dict["ssid_attn_hist"]                 # (B, K, L)

        hist_sem = self.interest_extractor.get_llm_emb(history)  # (B, L, D)

        if self.ssid_detach_sem:
            hist_sem = hist_sem.detach()

        if self.ssid_detach_attn:
            attn_hist = attn_hist.detach()

        # Build semantic prototype per interest via attention pooling
        sem_proto = torch.matmul(attn_hist, hist_sem)  # (B, K, D)

        if self.ssid_use_proto_norm:
            q = F.normalize(interest_vectors, dim=-1)
            p = F.normalize(sem_proto, dim=-1)
        else:
            q = interest_vectors
            p = sem_proto

        # Interest-prototype matching: diagonal should be highest
        logits = torch.matmul(q, p.transpose(1, 2)) / self.ssid_temp  # (B, K, K)

        B = logits.size(0)
        K = logits.size(1)
        labels = torch.arange(K, device=logits.device)
        labels = labels.unsqueeze(0).expand(B, -1).reshape(-1)

        loss = F.cross_entropy(
            logits.reshape(-1, K),
            labels,
        )

        aux = {
            "ssid_logits_diag": logits.diagonal(dim1=1, dim2=2).detach().mean(),
            "ssid_proto_norm": sem_proto.detach().norm(dim=-1).mean(),
        }

        return loss, aux

    # =========================
    # forward
    # =========================
    def forward(self, feed_dict):
        self.global_step += 1

        i_ids = feed_dict["item_id"]          # (B, 1+neg)
        history = feed_dict["history_items"]  # (B, H)
        lengths = feed_dict["lengths"]        # (B,)

        # Standard single-pass extraction (no token injection)
        interest_vectors, distri_vectors, aux = self.interest_extractor(
            history, lengths,
            return_aux=True,
        )

        # DSPC: post-hoc semantic prototype calibration (default off)
        dspc_aux = None
        if self.use_dspc and self.use_llmemb:
            interest_vectors, dspc_aux = self.compute_dspc_interest(
                history, interest_vectors,
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

        # SSID stash: interest vectors and attention maps for auxiliary contrastive loss
        if self.use_ssid:
            out_dict["ssid_interest_vectors"] = interest_vectors
            out_dict["ssid_attn_hist"] = aux["attn_hist"]
            out_dict["ssid_history"] = history

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

        # DSPC debug log
        if dspc_aux is not None and self.training and self.global_step % 1000 == 0:
            attn = dspc_aux["dspc_attn"]  # (B, K, L)
            entropy = -(attn * torch.log(attn + 1e-8)).sum(dim=-1).mean()
            logging.info(
                f"[DSPC] step={self.global_step} "
                f"scale={self.dspc_scale:.4f} "
                f"temp={self.dspc_temp:.3f} "
                f"gate_mean={dspc_aux['dspc_gate_mean'].item():.4f} "
                f"proto_norm={dspc_aux['dspc_proto_norm'].item():.4f} "
                f"attn_entropy={entropy.item():.4f}"
            )

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

        # SSID: semantic-supervised interest disentanglement
        if self.use_ssid and ("ssid_interest_vectors" in out_dict):
            L_ssid, ssid_aux = self.compute_ssid_loss(out_dict)
            loss = loss + self._ssid_w() * self.lambda_ssid * L_ssid
            out_dict["loss_ssid"] = L_ssid.detach()

            if self.training and self.global_step % 1000 == 0:
                logging.info(
                    f"[SSID] step={self.global_step} "
                    f"w={self._ssid_w():.4f} "
                    f"lambda={self.lambda_ssid:.6f} "
                    f"loss={L_ssid.item():.6f} "
                    f"diag_mean={ssid_aux['ssid_logits_diag'].item():.4f} "
                    f"proto_norm={ssid_aux['ssid_proto_norm'].item():.4f} "
                    f"temp={self.ssid_temp:.3f}"
                )

        return loss
