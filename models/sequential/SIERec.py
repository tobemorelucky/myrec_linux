# -*- coding: UTF-8 -*-
"""
SIERec: PoMRec backbone + LLM semantic alignment/injection (clean skeleton).

Phase 1 — minimal working model:
  - PoMRec multi-interest extractor
  - LLM embedding adapter + residual fusion (cf + gamma * llm)
  - InfoNCE alignment loss
  - BPR main recommendation loss

Phase 2 — MVTC CF-only target interest distillation:
  - Uses training positive item to build soft teacher distribution over K interests
  - KL divergence supervises model's own interest distribution (base_logits)
  - No target leakage at inference (model only uses user history)

Future phases will add: SPIP, semantic teacher, SIDE.
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
# MultiInterestExtractor (PoMRec backbone + LLM branch, clean)
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

    def forward(self, history: torch.Tensor, lengths: torch.Tensor):
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
# SIERec: PoMRec + LLMAlign (Phase 1 clean skeleton)
# =========================
class SIERec(SequentialModel):
    reader = "SeqReader"
    runner = "BaseRunner"

    extra_log_args = ["emb_size", "lr",
                     "use_mvtc", "lambda_mvtc", "mvtc_temp", "mvtc_warmup_steps"]

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

        # ---- MVTC: CF-only target interest distillation ----
        parser.add_argument("--use_mvtc", type=int, default=0,
                            help="1: enable MVTC CF-only target interest distillation")
        parser.add_argument("--lambda_mvtc", type=float, default=0.03,
                            help="weight for MVTC calibration loss")
        parser.add_argument("--mvtc_temp", type=float, default=0.5,
                            help="temperature for target interest distillation")
        parser.add_argument("--mvtc_warmup_steps", type=int, default=0,
                            help="warmup steps for MVTC loss; 0 means no warmup")

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

        # mvtc
        self.use_mvtc = int(getattr(args, "use_mvtc", 0))
        self.lambda_mvtc = float(getattr(args, "lambda_mvtc", 0.03))
        self.mvtc_temp = float(getattr(args, "mvtc_temp", 0.5))
        self.mvtc_warmup_steps = int(getattr(args, "mvtc_warmup_steps", 0))

        # build modules
        self._define_params()
        self.apply(self.init_weights)

        if self.use_llmemb:
            self.align_loss_func = InfoNCEAlign(tau=self.tau)

        # warm-start
        if self.use_llmemb and self.init_ckpt:
            self.load_model(self.init_ckpt, strict=bool(self.init_strict))
            logging.info(f"[SIERec] Warm-start from {self.init_ckpt} (strict={bool(self.init_strict)})")
        else:
            logging.info("[SIERec] Train from scratch")

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

    @staticmethod
    def _cos_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        a = F.normalize(a, dim=-1, eps=eps)
        b = F.normalize(b, dim=-1, eps=eps)
        return (a * b).sum(dim=-1)

    def _mvtc_w(self) -> float:
        if self.mvtc_warmup_steps <= 0:
            return 1.0
        t = min(self.global_step, self.mvtc_warmup_steps)
        return t / float(self.mvtc_warmup_steps)

    # =========================
    # forward
    # =========================
    def forward(self, feed_dict):
        self.global_step += 1

        i_ids = feed_dict["item_id"]          # (B, 1+neg)
        history = feed_dict["history_items"]  # (B, H)
        lengths = feed_dict["lengths"]        # (B,)

        interest_vectors, distri_vectors = self.interest_extractor(history, lengths)

        i_vectors = self.interest_extractor.get_item_emb(i_ids)

        base_logits = self.proj(distri_vectors)  # (B, K)
        w = torch.softmax(base_logits, dim=-1)   # (B, K)

        user_vector = (interest_vectors * w[:, :, None]).sum(dim=1)   # (B, D)
        prediction = (user_vector[:, None, :] * i_vectors).sum(dim=-1)  # (B, C)

        out_dict = {"prediction": prediction}

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

        # MVTC uses the positive item only to construct an auxiliary teacher distribution
        # during training. During inference, the model only uses base_logits predicted from
        # user history. Therefore, this does not introduce target leakage at inference time.
        if self.use_mvtc:
            out_dict["intent_logits"] = base_logits          # (B, K)
            out_dict["interest_vectors"] = interest_vectors  # (B, K, D)
            out_dict["pos_ids"] = i_ids[:, 0]                # (B,)

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

        if self.use_llmemb and ("align_loss" in out_dict):
            loss = loss + self._alpha_t() * out_dict["align_loss"]

        # MVTC: CF-only target interest distillation
        if self.use_mvtc and ("intent_logits" in out_dict):
            logits = out_dict["intent_logits"]              # (B, K)
            iv = out_dict["interest_vectors"]               # (B, K, D)
            pos_ids = out_dict["pos_ids"]                   # (B,)

            # CF target item embedding (no LLM fusion)
            pos_cf = self.interest_extractor.get_cf_emb(pos_ids)  # (B, D)

            # CF-only target interest teacher distribution
            cf_logits = self._cos_sim(iv, pos_cf[:, None, :])     # (B, K)
            p_cf = torch.softmax(cf_logits / self.mvtc_temp, dim=-1)

            # model predicted interest distribution
            log_q = torch.log_softmax(logits / self.mvtc_temp, dim=-1)

            # KL teacher -> student (temperature-scaled)
            L_mvtc = F.kl_div(
                log_q,
                p_cf.detach(),
                reduction="batchmean"
            ) * (self.mvtc_temp ** 2)

            loss = loss + self._mvtc_w() * self.lambda_mvtc * L_mvtc
            out_dict["loss_mvtc"] = L_mvtc.detach()

        return loss
