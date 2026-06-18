# -*- coding: UTF-8 -*-
"""
MyModelITIRL: PoMRec backbone + LLM + IPD + ITIRL.

ITIRL: IPD-Guided Target-routed Interest-level Ranking.
  - Reuses IPD's target-interest distances as routing signal
    (instead of recomputing from interest-target similarity).
  - Confidence-gated BPR at the interest level.
  - Prediction unchanged; no inference cost increase.

Modules:
  3.1 LLM Semantic Alignment & Controllable Injection
  3.2 ITIRL: IPD-Guided Target-routed Interest-level Ranking (aux loss)
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
    if arr.shape[0] == N1: table = arr
    elif arr.shape[0] == N1 - 1:
        table = np.vstack([np.zeros((1, arr.shape[1]), dtype=arr.dtype), arr])
    else:
        d = arr.shape[1]; table = np.zeros((N1, d), dtype=arr.dtype)
        take = min(arr.shape[0], N1); table[:take] = arr[:take]
    return torch.tensor(table, dtype=torch.float32)


def _load_srs_emb_pkl(path: str, expected_num_items_plus1: int) -> torch.Tensor:
    arr = pickle.load(open(path, "rb")); arr = _ensure_2d_np(arr)
    N1 = expected_num_items_plus1
    if arr.shape[0] == N1: table = arr
    elif arr.shape[0] == N1 - 1:
        table = np.vstack([np.zeros((1, arr.shape[1]), dtype=arr.dtype), arr])
    else:
        d = arr.shape[1]; table = np.zeros((N1, d), dtype=arr.dtype)
        take = min(arr.shape[0], N1); table[:take] = arr[:take]
    return torch.tensor(table, dtype=torch.float32)


# =========================
# Alignment loss
# =========================
class InfoNCEAlign(nn.Module):
    def __init__(self, tau=0.2): super().__init__(); self.tau = float(tau)
    def forward(self, X, Y):
        X = F.normalize(X, dim=-1); Y = F.normalize(Y, dim=-1)
        logits = (X @ Y.t()) / self.tau
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


# =========================
# MultiInterestExtractor (clean PoMRec + LLM)
# =========================
class MultiInterestExtractor(nn.Module):
    def __init__(self, k, item_num, emb_size, attn_size, max_his, prompt_num, lamb,
                 use_llmemb=0, llm_emb_path="", srs_emb_path="",
                 llm_fuse=1, gamma_init=0.05, gamma_trainable=1):
        super().__init__()
        self.K = int(k); self.max_his = int(max_his); self.prompt_num = int(prompt_num)
        self.lamb = float(lamb); self.emb_size = int(emb_size)
        self.use_llmemb = int(use_llmemb); self.llm_fuse = int(llm_fuse)
        self.gamma_trainable = int(gamma_trainable)
        self.i_embeddings = nn.Embedding(item_num, emb_size)
        self.p_embeddings = nn.Embedding(max_his + 1, emb_size)
        self.srs_emb = None
        if self.use_llmemb:
            if not llm_emb_path: raise ValueError("use_llmemb=1 but llm_emb_path is empty")
            llm_table = _load_llm_table_pkl(llm_emb_path, expected_num_items_plus1=item_num)
            self.register_buffer("llm_table", llm_table, persistent=False)
            d_llm = llm_table.size(1)
            self.adapter = nn.Sequential(
                nn.Linear(d_llm, d_llm // 2), nn.GELU(),
                nn.Linear(d_llm // 2, emb_size), nn.LayerNorm(emb_size))
            if self.gamma_trainable:
                self.log_gamma = nn.Parameter(torch.log(torch.exp(torch.tensor(gamma_init)) - 1.0))
            else: self.register_buffer("gamma", torch.tensor(float(gamma_init)))
            if srs_emb_path:
                srs_table = _load_srs_emb_pkl(srs_emb_path, expected_num_items_plus1=item_num)
                self.srs_emb = nn.Embedding.from_pretrained(srs_table, freeze=True)
        self.max_prompt = 5
        pad_len = max(0, self.max_prompt - self.prompt_num)
        self.register_buffer("prompt_pad", torch.ones(pad_len, emb_size), persistent=False)
        self.prompt1 = nn.Embedding(self.prompt_num, emb_size)
        self.prompt2 = nn.Embedding(self.prompt_num, emb_size)
        self.W1 = nn.Linear(emb_size, attn_size); self.W2 = nn.Linear(attn_size, self.K)
        self.W3 = nn.Linear(emb_size, attn_size); self.W4 = nn.Linear(attn_size, 1)

    def get_cf_emb(self, item_ids): return self.i_embeddings(item_ids)
    def get_llm_emb(self, item_ids):
        if not self.use_llmemb: raise RuntimeError("get_llm_emb called but use_llmemb=0")
        return self.adapter(self.llm_table[item_ids])
    def get_anchor_emb(self, item_ids):
        if self.use_llmemb and (self.srs_emb is not None): return self.srs_emb(item_ids)
        return self.get_cf_emb(item_ids)
    def get_gamma(self):
        if hasattr(self, "log_gamma"): return F.softplus(self.log_gamma)
        return self.gamma
    def get_item_emb(self, item_ids):
        e_cf = self.get_cf_emb(item_ids)
        if (not self.use_llmemb) or (not self.llm_fuse): return e_cf
        return e_cf + self.get_gamma() * self.get_llm_emb(item_ids)

    @staticmethod
    def value2attn(values, mask):
        values = values.masked_fill(mask.unsqueeze(-1) == 0, -np.inf)
        values = values.transpose(-1, -2)
        attn = (values - values.max()).softmax(dim=-1)
        return attn.masked_fill(torch.isnan(attn), 0)

    def forward(self, history, lengths):
        B, seq_len = history.shape; device = history.device
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
# MyModelITIRL: PoMRec + LLM + ITIRL + IPD
# =========================
class MyModelITIRL(SequentialModel):
    reader = "SeqReader"; runner = "BaseRunner"
    extra_log_args = ["emb_size", "lr", "use_emile", "lambda_ipd",
                      "use_itirl", "lambda_itirl", "itirl_route_source", "itirl_gate_type"]

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
        # IPD
        parser.add_argument("--use_emile", type=int, default=0)
        parser.add_argument("--lambda_ipd", type=float, default=0.05)
        parser.add_argument("--ipd_margin", type=float, default=0.2)
        parser.add_argument("--emile_use_fused_itememb", type=int, default=0)
        parser.add_argument("--emile_warmup_steps", type=int, default=5000)
        # ITIRL
        parser.add_argument("--use_itirl", type=int, default=1)
        parser.add_argument("--lambda_itirl", type=float, default=0.01)
        parser.add_argument("--itirl_warmup_steps", type=int, default=5000)
        parser.add_argument("--itirl_route_source", type=str, default="ipd")
        parser.add_argument("--itirl_gate_type", type=str, default="margin")
        parser.add_argument("--itirl_conf_threshold", type=float, default=0.45)
        parser.add_argument("--itirl_margin_threshold", type=float, default=0.10)
        parser.add_argument("--itirl_gate_mode", type=str, default="linear")
        parser.add_argument("--itirl_gate_temp", type=float, default=0.05)
        parser.add_argument("--itirl_score_norm", type=int, default=0)
        parser.add_argument("--itirl_neg_reduce", type=str, default="mean")
        parser.add_argument("--itirl_loss_normalize", type=str, default="batch")
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size; self.attn_size = args.attn_size
        self.K = args.K; self.prompt_num = args.prompt_num
        self.n_layers = args.n_layers; self.lamb = args.lamb; self.max_his = args.history_max
        self.use_llmemb = int(getattr(args, "use_llmemb", 0))
        self.llm_emb_path = getattr(args, "llm_emb_path", "")
        self.srs_emb_path = getattr(args, "srs_emb_path", "")
        self.alpha = float(getattr(args, "alpha", 0.001))
        self.tau = float(getattr(args, "tau", 0.2))
        self.rat_alpha_warmup_steps = int(getattr(args, "rat_alpha_warmup_steps", 0))
        self.init_ckpt = getattr(args, "init_ckpt", ""); self.init_strict = int(getattr(args, "init_strict", 0))
        self.llm_fuse = int(getattr(args, "llm_fuse", 1))
        self.gamma_init = float(getattr(args, "gamma_init", 0.05))
        self.gamma_trainable = int(getattr(args, "gamma_trainable", 1))
        # IPD
        self.use_emile = int(getattr(args, "use_emile", 0))
        self.lambda_ipd = float(getattr(args, "lambda_ipd", 0.1))
        self.ipd_margin = float(getattr(args, "ipd_margin", 0.2))
        self.emile_use_fused_itememb = int(getattr(args, "emile_use_fused_itememb", 0))
        self.emile_warmup_steps = int(getattr(args, "emile_warmup_steps", 5000))
        if self.use_emile:
            g = torch.Generator(device="cpu"); g.manual_seed(int(getattr(args, "random_seed", 1)) + 2027)
            self.register_buffer("emile_T", torch.randn(self.emb_size, generator=g))
        else: self.register_buffer("emile_T", torch.zeros(self.emb_size))
        # ITIRL
        self.use_itirl = int(getattr(args, "use_itirl", 1))
        self.lambda_itirl = float(getattr(args, "lambda_itirl", 0.01))
        self.itirl_warmup_steps = int(getattr(args, "itirl_warmup_steps", 5000))
        self.itirl_route_source = str(getattr(args, "itirl_route_source", "ipd"))
        self.itirl_gate_type = str(getattr(args, "itirl_gate_type", "margin"))
        self.itirl_conf_threshold = float(getattr(args, "itirl_conf_threshold", 0.45))
        self.itirl_margin_threshold = float(getattr(args, "itirl_margin_threshold", 0.10))
        self.itirl_gate_mode = str(getattr(args, "itirl_gate_mode", "linear"))
        self.itirl_gate_temp = float(getattr(args, "itirl_gate_temp", 0.05))
        self.itirl_score_norm = int(getattr(args, "itirl_score_norm", 0))
        self.itirl_neg_reduce = str(getattr(args, "itirl_neg_reduce", "mean"))
        self.itirl_loss_normalize = str(getattr(args, "itirl_loss_normalize", "batch"))
        self._define_params(); self.apply(self.init_weights)
        if self.use_llmemb: self.align_loss_func = InfoNCEAlign(tau=self.tau)
        if self.use_llmemb and self.init_ckpt:
            self.load_model(self.init_ckpt, strict=bool(self.init_strict))
            logging.info(f"[MyModelITIRL] Warm-start from {self.init_ckpt}")
        else: logging.info("[MyModelITIRL] Train from scratch")
        self.global_step = 0

    def _define_params(self):
        self.interest_extractor = MultiInterestExtractor(
            k=self.K, item_num=self.item_num, emb_size=self.emb_size,
            attn_size=self.attn_size, max_his=self.max_his, prompt_num=self.prompt_num,
            lamb=self.lamb, use_llmemb=self.use_llmemb, llm_emb_path=self.llm_emb_path,
            srs_emb_path=self.srs_emb_path, llm_fuse=self.llm_fuse,
            gamma_init=self.gamma_init, gamma_trainable=self.gamma_trainable)
        self.proj = nn.Sequential()
        for i in range(max(0, self.n_layers - 1)):
            self.proj.add_module(f"proj_{i}", nn.Linear(self.emb_size, self.emb_size))
            self.proj.add_module(f"dropout_{i}", nn.Dropout(p=0.5))
            self.proj.add_module(f"relu_{i}", nn.ReLU(inplace=True))
        self.proj.add_module("proj_final", nn.Linear(self.emb_size, self.K))

    def load_model(self, model_path=None, strict=False):
        if model_path is None: model_path = self.model_path
        model_dict = self.state_dict(); state_dict = torch.load(model_path, map_location="cpu")
        if strict: self.load_state_dict(state_dict, strict=True)
        else:
            exist = {k: v for k, v in state_dict.items()
                     if k in model_dict and hasattr(v, "shape") and hasattr(model_dict[k], "shape")
                     and v.shape == model_dict[k].shape}
            model_dict.update(exist); self.load_state_dict(model_dict, strict=False)
        logging.info("Load model from " + model_path)

    def _alpha_t(self):
        if self.rat_alpha_warmup_steps <= 0: return self.alpha
        return self.alpha * min(self.global_step, self.rat_alpha_warmup_steps) / float(self.rat_alpha_warmup_steps)
    def _emile_w(self):
        if self.emile_warmup_steps <= 0: return 1.0
        return min(self.global_step, self.emile_warmup_steps) / float(self.emile_warmup_steps)
    def _itirl_w(self):
        if self.itirl_warmup_steps <= 0: return 1.0
        return min(self.global_step, self.itirl_warmup_steps) / float(self.itirl_warmup_steps)

    @staticmethod
    def _cos_sim(a, b, eps=1e-8):
        return (F.normalize(a, dim=-1, eps=eps) * F.normalize(b, dim=-1, eps=eps)).sum(dim=-1)
    @staticmethod
    def _cos_dist(a, b, eps=1e-8): return 1.0 - MyModelITIRL._cos_sim(a, b, eps=eps)
    @staticmethod
    def _bpr_dist(pos_dist, neg_dist, margin=0.0): return F.softplus((pos_dist - neg_dist) + margin).mean()

    # -------------------------
    # ITIRL loss
    # -------------------------
    def compute_itirl_loss(self, out_dict):
        interest_vectors = out_dict["itirl_interest_vectors"]  # (B, K, D)
        item_vectors = out_dict["itirl_item_vectors"]          # (B, N, D)
        if item_vectors.size(1) <= 1:
            return torch.zeros([], device=item_vectors.device), {}

        # Route from IPD or weight
        if self.itirl_route_source == "ipd" and "itirl_route_prob_ipd" in out_dict:
            route_prob = out_dict["itirl_route_prob_ipd"]
        else:
            route_prob = out_dict.get("itirl_route_prob_weight",
                        torch.ones(interest_vectors.size()[:2], device=interest_vectors.device) / self.K)
        route_prob = route_prob.detach()

        conf, route_idx = route_prob.max(dim=1)
        top2 = torch.topk(route_prob, k=min(2, route_prob.size(1)), dim=1).values
        margin = top2[:, 0] - top2[:, 1] if top2.size(1) >= 2 else conf

        q = F.normalize(interest_vectors, dim=-1) if self.itirl_score_norm else interest_vectors
        p = F.normalize(item_vectors, dim=-1) if self.itirl_score_norm else item_vectors
        scores = torch.einsum("bkd,bnd->bkn", q, p)
        pos_scores = scores[:, :, 0]; neg_scores = scores[:, :, 1:]

        selected_pos = pos_scores.gather(1, route_idx[:, None]).squeeze(1)
        idx_expanded = route_idx[:, None, None].expand(-1, 1, neg_scores.size(-1))
        selected_neg = neg_scores.gather(1, idx_expanded).squeeze(1)

        if self.itirl_neg_reduce == "max":
            loss_vec = -F.logsigmoid(selected_pos - selected_neg.max(dim=1).values)
        else:
            loss_vec = -F.logsigmoid(selected_pos[:, None] - selected_neg).mean(dim=1)

        # Gate
        if self.itirl_gate_type == "none":
            gate = torch.ones_like(loss_vec)
        else:
            base = conf if self.itirl_gate_type == "conf" else margin
            threshold = self.itirl_conf_threshold if self.itirl_gate_type == "conf" else self.itirl_margin_threshold
            if self.itirl_gate_mode == "sigmoid":
                gate = torch.sigmoid((base - threshold) / self.itirl_gate_temp)
            else:
                gate = ((base - threshold) / (1.0 - threshold)).clamp(min=0.0, max=1.0)
        gate = gate.detach()

        if self.itirl_loss_normalize == "active":
            loss_itirl = (gate * loss_vec).sum() / (gate.sum() + 1e-8)
        else:
            loss_itirl = (gate * loss_vec).mean()

        aux = {"conf_mean": conf.mean(), "margin_mean": margin.float().mean(),
               "gate_mean": gate.mean(), "active_rate": (gate > 0).float().mean(),
               "selected_pos_mean": selected_pos.mean(), "selected_neg_mean": selected_neg.mean()}
        return loss_itirl, aux

    # =========================
    # forward
    # =========================
    def forward(self, feed_dict):
        self.global_step += 1
        i_ids = feed_dict["item_id"]; history = feed_dict["history_items"]; lengths = feed_dict["lengths"]

        interest_vectors, distri_vectors = self.interest_extractor(history, lengths)
        i_vectors = self.interest_extractor.get_item_emb(i_ids)
        base_logits = self.proj(distri_vectors)
        w = torch.softmax(base_logits, dim=-1)
        u_base = (interest_vectors * w[:, :, None]).sum(dim=1)
        prediction = (u_base[:, None, :] * i_vectors).sum(dim=-1)
        out_dict = {"prediction": prediction}

        if self.use_emile:
            out_dict["emile_interest_vectors"] = interest_vectors
            out_dict["emile_user_vector"] = u_base; out_dict["emile_w"] = w
            out_dict["emile_pos_ids"] = i_ids[:, 0]
            out_dict["emile_neg_ids"] = i_ids[:, 1] if i_ids.size(1) > 1 else None

        # ITIRL stash
        if self.use_itirl:
            out_dict["itirl_interest_vectors"] = interest_vectors
            out_dict["itirl_item_vectors"] = i_vectors
            # IPD routing: use cos_dist from pos item to each interest
            pos_cf = self.interest_extractor.get_cf_emb(i_ids[:, 0])
            d_pos_h = self._cos_dist(pos_cf[:, None, :], interest_vectors)  # (B, K)
            out_dict["itirl_route_prob_ipd"] = F.softmax(-d_pos_h, dim=1).detach()
            out_dict["itirl_route_prob_weight"] = w.detach()

        if self.use_llmemb:
            pos_ids = i_ids[:, 0]; mask = (pos_ids != 0)
            if mask.any():
                ids = pos_ids[mask]
                out_dict["align_loss"] = self.align_loss_func(
                    self.interest_extractor.get_anchor_emb(ids),
                    self.interest_extractor.get_llm_emb(ids))
            else: out_dict["align_loss"] = torch.zeros([], device=prediction.device)

        if self.use_llmemb and self.llm_fuse and (self.global_step % 200 == 0):
            ie = self.interest_extractor
            with torch.no_grad():
                g = float(ie.get_gamma().detach().item())
                pos_dbg = i_ids[:, 0]; pos_dbg = pos_dbg[pos_dbg != 0][:128]
                if pos_dbg.numel() > 0:
                    ratio = (g * ie.get_llm_emb(pos_dbg)).norm(dim=-1).mean() / (ie.get_cf_emb(pos_dbg).norm(dim=-1).mean() + 1e-12)
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
                iv = out_dict["emile_interest_vectors"]; w = out_dict["emile_w"]
                pos_ids = out_dict["emile_pos_ids"]; ie = self.interest_extractor
                pos_v = ie.get_item_emb(pos_ids) if self.emile_use_fused_itememb else ie.get_cf_emb(pos_ids)
                neg_v = ie.get_item_emb(neg_ids) if self.emile_use_fused_itememb else ie.get_cf_emb(neg_ids)
                H_vec = (iv * w[:, :, None]).sum(dim=1)
                d_pos_h = self._cos_dist(pos_v[:, None, :], iv)
                L_ipd = (
                    self._bpr_dist(self._cos_dist(pos_v, H_vec), d_pos_h.min(dim=1).values, margin=self.ipd_margin) +
                    self._bpr_dist(self._cos_dist(pos_v, H_vec), self._cos_dist(neg_v, H_vec), margin=self.ipd_margin) +
                    self._bpr_dist(d_pos_h.min(dim=1).values, self._cos_dist(neg_v, H_vec), margin=self.ipd_margin))
                loss = loss + self._emile_w() * (self.lambda_ipd * L_ipd)
                out_dict["loss_ipd"] = L_ipd.detach()

        if self.use_itirl and ("itirl_interest_vectors" in out_dict):
            loss_itirl, aux = self.compute_itirl_loss(out_dict)
            loss = loss + self._itirl_w() * self.lambda_itirl * loss_itirl
            out_dict["loss_itirl"] = loss_itirl.detach()
            if self.training and self.global_step % 1000 == 0:
                logging.info(
                    f"[ITIRL] step={self.global_step} w={self._itirl_w():.4f} lambda={self.lambda_itirl:.4f} "
                    f"loss={loss_itirl.item():.6f} route_source={self.itirl_route_source} "
                    f"gate_type={self.itirl_gate_type} conf_mean={aux['conf_mean'].item():.4f} "
                    f"margin_mean={aux['margin_mean'].item():.4f} gate_mean={aux['gate_mean'].item():.4f} "
                    f"active_rate={aux['active_rate'].item():.4f} "
                    f"pos_mean={aux['selected_pos_mean'].item():.4f} neg_mean={aux['selected_neg_mean'].item():.4f}")
        return loss
