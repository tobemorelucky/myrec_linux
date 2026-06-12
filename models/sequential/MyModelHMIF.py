# -*- coding: UTF-8 -*-
"""
MyModelHMIF: PoMRec backbone + LLM + IPD + HMIF.

HMIF: Hybrid Multi-Interest Fusion.
  - Lightweight scoring mechanism that combines aggregated user-vector
    prediction with interest-level candidate matching.
  - No auxiliary loss, no offline preprocessing, no codebook.
  - Prediction only; interest vectors and user representation unchanged.

Modules:
  3.1 LLM Semantic Alignment & Controllable Injection
  3.2 HMIF: Hybrid Multi-Interest Fusion scoring
  3.3 IPD: Target-Interest Consistency
"""

import logging
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import SequentialModel


# =========================
# Utils
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

    def forward(self, X, Y):
        X = F.normalize(X, dim=-1)
        Y = F.normalize(Y, dim=-1)
        logits = (X @ Y.t()) / self.tau
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


# =========================
# MultiInterestExtractor (clean PoMRec + LLM, no extras)
# =========================
class MultiInterestExtractor(nn.Module):
    def __init__(
        self, k, item_num, emb_size, attn_size, max_his, prompt_num, lamb,
        use_llmemb=0, llm_emb_path="", srs_emb_path="",
        llm_fuse=1, gamma_init=0.05, gamma_trainable=1,
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
            self.adapter = nn.Sequential(
                nn.Linear(d_llm, d_llm // 2), nn.GELU(),
                nn.Linear(d_llm // 2, emb_size), nn.LayerNorm(emb_size),
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

    def get_cf_emb(self, item_ids):
        return self.i_embeddings(item_ids)

    def get_llm_emb(self, item_ids):
        if not self.use_llmemb:
            raise RuntimeError("get_llm_emb called but use_llmemb=0")
        return self.adapter(self.llm_table[item_ids])

    def get_anchor_emb(self, item_ids):
        if self.use_llmemb and (self.srs_emb is not None):
            return self.srs_emb(item_ids)
        return self.get_cf_emb(item_ids)

    def get_gamma(self):
        if hasattr(self, "log_gamma"):
            return F.softplus(self.log_gamma)
        return self.gamma

    def get_item_emb(self, item_ids):
        e_cf = self.get_cf_emb(item_ids)
        if (not self.use_llmemb) or (not self.llm_fuse):
            return e_cf
        return e_cf + self.get_gamma() * self.get_llm_emb(item_ids)

    @staticmethod
    def value2attn(values, mask):
        values = values.masked_fill(mask.unsqueeze(-1) == 0, -np.inf)
        values = values.transpose(-1, -2)
        attn = (values - values.max()).softmax(dim=-1)
        return attn.masked_fill(torch.isnan(attn), 0)

    def forward(self, history, lengths):
        B, seq_len = history.shape
        device = history.device
        valid_his = (history > 0).long()
        his_vectors = self.get_item_emb(history)
        len_range = torch.arange(self.max_his, device=device)
        position = (lengths[:, None] - len_range[None, :seq_len]) * valid_his
        his_vectors = his_vectors + self.p_embeddings(position)
        valid_his_ext = torch.cat([valid_his, torch.ones([B, self.max_prompt], device=device)], dim=1)

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

        prompt2 = torch.cat([self.prompt_pad.to(device), self.prompt2.weight], dim=0)
        prompt2 = prompt2.unsqueeze(0).expand(B, -1, -1)
        his_vectors_prompt2 = torch.cat([his_vectors, prompt2], dim=1)
        distri_pred = self.W4(self.W3(his_vectors_prompt2).tanh())
        distri_maps = self.value2attn(distri_pred, valid_his_ext)
        distri_vectors = torch.matmul(distri_maps, his_vectors_prompt2).squeeze(1)

        return interest_vectors, distri_vectors


# =========================
# MyModelHMIF: PoMRec + LLM + HMIF + IPD
# =========================
class MyModelHMIF(SequentialModel):
    reader = "SeqReader"
    runner = "BaseRunner"
    extra_log_args = [
        "emb_size", "lr", "use_emile", "lambda_ipd",
        "use_hmif", "hmif_eta", "hmif_mode", "hmif_temp",
    ]

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--emb_size", type=int, default=64)
        parser.add_argument("--attn_size", type=int, default=8)
        parser.add_argument("--K", type=int, default=3)
        parser.add_argument("--prompt_num", type=int, default=4)
        parser.add_argument("--n_layers", type=int, default=1)
        parser.add_argument("--lamb", type=float, default=3.0)

        parser.add_argument("--use_llmemb", type=int, default=0)
        parser.add_argument("--llm_emb_path", type=str, default="")
        parser.add_argument("--srs_emb_path", type=str, default="")
        parser.add_argument("--alpha", type=float, default=0.001)
        parser.add_argument("--tau", type=float, default=0.2)
        parser.add_argument("--rat_alpha_warmup_steps", type=int, default=5000)
        parser.add_argument("--llm_fuse", type=int, default=1)
        parser.add_argument("--gamma_init", type=float, default=0.1)
        parser.add_argument("--gamma_trainable", type=int, default=0)

        parser.add_argument("--init_ckpt", type=str, default="")
        parser.add_argument("--init_strict", type=int, default=0)

        # ---- IPD / EMILE ----
        parser.add_argument("--use_emile", type=int, default=0)
        parser.add_argument("--lambda_ipd", type=float, default=0.05)
        parser.add_argument("--ipd_margin", type=float, default=0.2)
        parser.add_argument("--emile_use_fused_itememb", type=int, default=0)
        parser.add_argument("--emile_warmup_steps", type=int, default=5000)

        # ---- HMIF: Hybrid Multi-Interest Fusion ----
        parser.add_argument("--use_hmif", type=int, default=1,
                            help="1: enable hybrid multi-interest fusion scoring")
        parser.add_argument("--hmif_eta", type=float, default=0.1,
                            help="weight of interest-level matching in final prediction")
        parser.add_argument("--hmif_mode", type=str, default="logsumexp",
                            help="interest aggregation mode: max or logsumexp")
        parser.add_argument("--hmif_temp", type=float, default=0.2,
                            help="temperature for logsumexp mode")
        parser.add_argument("--hmif_detach_weight", type=int, default=0,
                            help="1: detach interest weights in HMIF")

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

        # IPD / EMILE
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

        # HMIF
        self.use_hmif = int(getattr(args, "use_hmif", 1))
        self.hmif_eta = float(getattr(args, "hmif_eta", 0.1))
        self.hmif_eta = max(0.0, min(1.0, self.hmif_eta))
        self.hmif_mode = str(getattr(args, "hmif_mode", "logsumexp"))
        self.hmif_temp = float(getattr(args, "hmif_temp", 0.2))
        self.hmif_detach_weight = int(getattr(args, "hmif_detach_weight", 0))

        self._define_params()
        self.apply(self.init_weights)
        if self.use_llmemb:
            self.align_loss_func = InfoNCEAlign(tau=self.tau)
        if self.use_llmemb and self.init_ckpt:
            self.load_model(self.init_ckpt, strict=bool(self.init_strict))
            logging.info(f"[MyModelHMIF] Warm-start from {self.init_ckpt}")
        else:
            logging.info("[MyModelHMIF] Train from scratch")
        self.global_step = 0

    def _define_params(self):
        self.interest_extractor = MultiInterestExtractor(
            k=self.K, item_num=self.item_num, emb_size=self.emb_size,
            attn_size=self.attn_size, max_his=self.max_his, prompt_num=self.prompt_num,
            lamb=self.lamb, use_llmemb=self.use_llmemb, llm_emb_path=self.llm_emb_path,
            srs_emb_path=self.srs_emb_path, llm_fuse=self.llm_fuse,
            gamma_init=self.gamma_init, gamma_trainable=self.gamma_trainable,
        )
        self.proj = nn.Sequential()
        for i in range(max(0, self.n_layers - 1)):
            self.proj.add_module(f"proj_{i}", nn.Linear(self.emb_size, self.emb_size))
            self.proj.add_module(f"dropout_{i}", nn.Dropout(p=0.5))
            self.proj.add_module(f"relu_{i}", nn.ReLU(inplace=True))
        self.proj.add_module("proj_final", nn.Linear(self.emb_size, self.K))

    def load_model(self, model_path=None, strict=False):
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
    def _alpha_t(self):
        if self.rat_alpha_warmup_steps <= 0:
            return self.alpha
        t = min(self.global_step, self.rat_alpha_warmup_steps)
        return self.alpha * (t / float(self.rat_alpha_warmup_steps))

    def _emile_w(self):
        if self.emile_warmup_steps <= 0:
            return 1.0
        t = min(self.global_step, self.emile_warmup_steps)
        return t / float(self.emile_warmup_steps)

    @staticmethod
    def _cos_sim(a, b, eps=1e-8):
        a = F.normalize(a, dim=-1, eps=eps)
        b = F.normalize(b, dim=-1, eps=eps)
        return (a * b).sum(dim=-1)

    @staticmethod
    def _cos_dist(a, b, eps=1e-8):
        return 1.0 - MyModelHMIF._cos_sim(a, b, eps=eps)

    @staticmethod
    def _bpr_dist(pos_dist, neg_dist, margin=0.0):
        return F.softplus((pos_dist - neg_dist) + margin).mean()

    # =========================
    # forward (single-pass, HMIF scoring)
    # =========================
    def forward(self, feed_dict):
        self.global_step += 1
        i_ids = feed_dict["item_id"]
        history = feed_dict["history_items"]
        lengths = feed_dict["lengths"]

        interest_vectors, distri_vectors = self.interest_extractor(history, lengths)

        i_vectors = self.interest_extractor.get_item_emb(i_ids)
        base_logits = self.proj(distri_vectors)
        w = torch.softmax(base_logits, dim=-1)

        batch_size = interest_vectors.size(0)

        # Aggregated prediction (standard)
        u_base = (interest_vectors * w[:, :, None]).sum(dim=1)
        prediction_agg = (u_base[:, None, :] * i_vectors).sum(dim=-1)

        if self.use_hmif:
            # Interest-level matching scores: (B, K, N)
            w_match = w.detach() if self.hmif_detach_weight else w
            interest_scores = torch.einsum("bkd,bnd->bkn", interest_vectors, i_vectors)

            if self.hmif_mode == "max":
                prediction_mi = interest_scores.max(dim=1).values
            else:  # logsumexp
                prediction_mi = torch.logsumexp(
                    interest_scores / self.hmif_temp, dim=1
                ) * self.hmif_temp

            prediction = (1 - self.hmif_eta) * prediction_agg + self.hmif_eta * prediction_mi
        else:
            prediction = prediction_agg

        out_dict = {"prediction": prediction.view(batch_size, -1)}

        # IPD stash
        if self.use_emile:
            out_dict["emile_interest_vectors"] = interest_vectors
            out_dict["emile_user_vector"] = u_base
            out_dict["emile_w"] = w
            out_dict["emile_pos_ids"] = i_ids[:, 0]
            out_dict["emile_neg_ids"] = i_ids[:, 1] if i_ids.size(1) > 1 else None

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

        # HMIF debug
        if self.use_hmif and self.training and self.global_step % 1000 == 0:
            logging.info(
                f"[HMIF] step={self.global_step} "
                f"eta={self.hmif_eta:.3f} "
                f"mode={self.hmif_mode} "
                f"temp={self.hmif_temp:.3f} "
                f"agg_score_mean={prediction_agg.mean().item():.4f} "
                f"mi_score_mean={prediction_mi.mean().item():.4f} "
                f"prediction_mean={prediction.mean().item():.4f}"
            )

        if self.use_llmemb and self.llm_fuse and (self.global_step % 200 == 0):
            ie = self.interest_extractor
            with torch.no_grad():
                g = float(ie.get_gamma().detach().item())
                pos_dbg = i_ids[:, 0]; pos_dbg = pos_dbg[pos_dbg != 0][:128]
                if pos_dbg.numel() > 0:
                    e_cf = ie.get_cf_emb(pos_dbg)
                    e_llm = ie.get_llm_emb(pos_dbg)
                    ratio = (g * e_llm).norm(dim=-1).mean() / (e_cf.norm(dim=-1).mean() + 1e-12)
                    print(f"[step {self.global_step}] gamma={g:.6f}  llm/cf_norm_ratio={float(ratio):.4f}")

        return out_dict

    # =========================
    # loss
    # =========================
    def loss(self, out_dict):
        loss = super().loss(out_dict)

        if self.use_llmemb and ("align_loss" in out_dict):
            loss = loss + self._alpha_t() * out_dict["align_loss"]

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
                w_em = self._emile_w()
                loss = loss + w_em * (self.lambda_ipd * L_ipd)
                out_dict["loss_ipd"] = L_ipd.detach()

        return loss
