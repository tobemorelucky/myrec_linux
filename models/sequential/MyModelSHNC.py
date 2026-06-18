# -*- coding: UTF-8 -*-
"""
MyModelSHNC: PoMRec backbone + LLM + IPD + SHNC.

SHNC: Semantic Hard Negative Calibration.
  - Uses offline pre-computed semantic nearest-neighbor table.
  - During training, for each positive item, samples hard negatives
    from semantically similar items and applies BPR loss.
  - Prediction and inference completely unchanged.

Modules:
  3.1 LLM Semantic Alignment & Controllable Injection
  3.2 SHNC: Semantic Hard Negative Calibration (auxiliary loss)
  3.3 IPD: Target-Interest Consistency
"""

import logging, pickle, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from models.BaseModel import SequentialModel

# ========================= Utils =========================
def _ensure_2d_np(x):
    x = np.asarray(x)
    if x.ndim != 2: raise ValueError(f"embedding must be 2D, got {x.shape}")
    return x

def _load_llm_table_pkl(path, N1):
    arr = pickle.load(open(path, "rb")); arr = _ensure_2d_np(arr)
    if arr.shape[0] == N1: table = arr
    elif arr.shape[0] == N1 - 1: table = np.vstack([np.zeros((1, arr.shape[1]), dtype=arr.dtype), arr])
    else:
        d = arr.shape[1]; table = np.zeros((N1, d), dtype=arr.dtype)
        table[: min(arr.shape[0], N1)] = arr[: min(arr.shape[0], N1)]
    return torch.tensor(table, dtype=torch.float32)

def _load_srs_emb_pkl(path, N1):
    arr = pickle.load(open(path, "rb")); arr = _ensure_2d_np(arr)
    if arr.shape[0] == N1: table = arr
    elif arr.shape[0] == N1 - 1: table = np.vstack([np.zeros((1, arr.shape[1]), dtype=arr.dtype), arr])
    else:
        d = arr.shape[1]; table = np.zeros((N1, d), dtype=arr.dtype)
        table[: min(arr.shape[0], N1)] = arr[: min(arr.shape[0], N1)]
    return torch.tensor(table, dtype=torch.float32)

# ========================= Alignment =========================
class InfoNCEAlign(nn.Module):
    def __init__(self, tau=0.2): super().__init__(); self.tau = float(tau)
    def forward(self, X, Y):
        X, Y = F.normalize(X, dim=-1), F.normalize(Y, dim=-1)
        logits = (X @ Y.t()) / self.tau
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

# ========================= MultiInterestExtractor =========================
class MultiInterestExtractor(nn.Module):
    def __init__(self, k, item_num, emb_size, attn_size, max_his, prompt_num, lamb,
                 use_llmemb=0, llm_emb_path="", srs_emb_path="", llm_fuse=1, gamma_init=0.05, gamma_trainable=1):
        super().__init__()
        self.K = int(k); self.max_his = int(max_his); self.prompt_num = int(prompt_num)
        self.lamb = float(lamb); self.emb_size = int(emb_size)
        self.use_llmemb = int(use_llmemb); self.llm_fuse = int(llm_fuse); self.gamma_trainable = int(gamma_trainable)
        self.i_embeddings = nn.Embedding(item_num, emb_size)
        self.p_embeddings = nn.Embedding(max_his + 1, emb_size)
        self.srs_emb = None
        if self.use_llmemb:
            if not llm_emb_path: raise ValueError("use_llmemb=1 but llm_emb_path is empty")
            self.register_buffer("llm_table", _load_llm_table_pkl(llm_emb_path, item_num), persistent=False)
            d_llm = self.llm_table.size(1)
            self.adapter = nn.Sequential(nn.Linear(d_llm, d_llm // 2), nn.GELU(),
                                         nn.Linear(d_llm // 2, emb_size), nn.LayerNorm(emb_size))
            if self.gamma_trainable: self.log_gamma = nn.Parameter(torch.log(torch.exp(torch.tensor(gamma_init)) - 1.0))
            else: self.register_buffer("gamma", torch.tensor(float(gamma_init)))
            if srs_emb_path: self.srs_emb = nn.Embedding.from_pretrained(
                _load_srs_emb_pkl(srs_emb_path, item_num), freeze=True)
        self.max_prompt = 5
        self.register_buffer("prompt_pad", torch.ones(max(0, self.max_prompt - self.prompt_num), emb_size), persistent=False)
        self.prompt1 = nn.Embedding(self.prompt_num, emb_size); self.prompt2 = nn.Embedding(self.prompt_num, emb_size)
        self.W1 = nn.Linear(emb_size, attn_size); self.W2 = nn.Linear(attn_size, self.K)
        self.W3 = nn.Linear(emb_size, attn_size); self.W4 = nn.Linear(attn_size, 1)

    def get_cf_emb(self, ids): return self.i_embeddings(ids)
    def get_llm_emb(self, ids):
        if not self.use_llmemb: raise RuntimeError("get_llm_emb called but use_llmemb=0")
        return self.adapter(self.llm_table[ids])
    def get_anchor_emb(self, ids):
        if self.use_llmemb and self.srs_emb is not None: return self.srs_emb(ids)
        return self.get_cf_emb(ids)
    def get_gamma(self):
        if hasattr(self, "log_gamma"): return F.softplus(self.log_gamma)
        return self.gamma
    def get_item_emb(self, ids):
        e_cf = self.get_cf_emb(ids)
        if (not self.use_llmemb) or (not self.llm_fuse): return e_cf
        return e_cf + self.get_gamma() * self.get_llm_emb(ids)

    @staticmethod
    def value2attn(values, mask):
        values = values.masked_fill(mask.unsqueeze(-1) == 0, -np.inf)
        values = values.transpose(-1, -2); attn = (values - values.max()).softmax(dim=-1)
        return attn.masked_fill(torch.isnan(attn), 0)

    def forward(self, history, lengths):
        B, seq_len = history.shape; device = history.device
        valid_his = (history > 0).long()
        his_vectors = self.get_item_emb(history)
        len_range = torch.arange(self.max_his, device=device)
        position = (lengths[:, None] - len_range[None, :seq_len]) * valid_his
        his_vectors = his_vectors + self.p_embeddings(position)
        valid_his_ext = torch.cat([valid_his, torch.ones([B, self.max_prompt], device=device)], dim=1)
        prompt1 = torch.cat([self.prompt_pad.to(device), self.prompt1.weight], dim=0).unsqueeze(0).expand(B, -1, -1)
        his_vectors_prompt1 = torch.cat([his_vectors, prompt1], dim=1)
        attn_maps = self.value2attn(self.W2(self.W1(his_vectors_prompt1).tanh()), valid_his_ext)
        interest_vectors = (his_vectors_prompt1[:, None, :, :] * attn_maps[:, :, :, None]).sum(-2)
        var = []
        for kk in range(self.K):
            x_mean_2 = (his_vectors_prompt1 - interest_vectors[:, kk:kk + 1, :]) ** 2
            var.append(torch.sqrt(torch.matmul(attn_maps[:, kk:kk + 1, :], x_mean_2) + 1e-12))
        interest_vectors = interest_vectors + self.lamb * torch.cat(var, 1)
        prompt2 = torch.cat([self.prompt_pad.to(device), self.prompt2.weight], dim=0).unsqueeze(0).expand(B, -1, -1)
        his_vectors_prompt2 = torch.cat([his_vectors, prompt2], dim=1)
        distri_maps = self.value2attn(self.W4(self.W3(his_vectors_prompt2).tanh()), valid_his_ext)
        distri_vectors = torch.matmul(distri_maps, his_vectors_prompt2).squeeze(1)
        return interest_vectors, distri_vectors


# ========================= MyModelSHNC =========================
class MyModelSHNC(SequentialModel):
    reader = "SeqReader"; runner = "BaseRunner"
    extra_log_args = ["emb_size", "lr", "use_emile", "lambda_ipd",
                      "use_shnc", "lambda_shnc", "shnc_num", "shnc_sample_mode",
                      "shnc_filter_history", "shnc_rank_start", "shnc_rank_end"]

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
        parser.add_argument("--use_emile", type=int, default=0)
        parser.add_argument("--lambda_ipd", type=float, default=0.05)
        parser.add_argument("--ipd_margin", type=float, default=0.2)
        parser.add_argument("--emile_use_fused_itememb", type=int, default=0)
        parser.add_argument("--emile_warmup_steps", type=int, default=5000)
        # SHNC
        parser.add_argument("--use_shnc", type=int, default=1)
        parser.add_argument("--shnc_path", type=str, default="")
        parser.add_argument("--lambda_shnc", type=float, default=0.01)
        parser.add_argument("--shnc_warmup_steps", type=int, default=5000)
        parser.add_argument("--shnc_num", type=int, default=1)
        parser.add_argument("--shnc_neg_reduce", type=str, default="mean")
        parser.add_argument("--shnc_score_norm", type=int, default=0)
        parser.add_argument("--shnc_detach_user", type=int, default=0)
        parser.add_argument("--shnc_sample_mode", type=str, default="random")
        parser.add_argument("--shnc_filter_history", type=int, default=1)
        parser.add_argument("--shnc_rank_start", type=int, default=0)
        parser.add_argument("--shnc_rank_end", type=int, default=100)
        parser.add_argument("--shnc_resample_attempts", type=int, default=5)
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size; self.attn_size = args.attn_size
        self.K = args.K; self.prompt_num = args.prompt_num; self.n_layers = args.n_layers
        self.lamb = args.lamb; self.max_his = args.history_max
        self.use_llmemb = int(getattr(args, "use_llmemb", 0))
        self.llm_emb_path = getattr(args, "llm_emb_path", "")
        self.srs_emb_path = getattr(args, "srs_emb_path", "")
        self.alpha = float(getattr(args, "alpha", 0.001)); self.tau = float(getattr(args, "tau", 0.2))
        self.rat_alpha_warmup_steps = int(getattr(args, "rat_alpha_warmup_steps", 0))
        self.init_ckpt = getattr(args, "init_ckpt", ""); self.init_strict = int(getattr(args, "init_strict", 0))
        self.llm_fuse = int(getattr(args, "llm_fuse", 1))
        self.gamma_init = float(getattr(args, "gamma_init", 0.05))
        self.gamma_trainable = int(getattr(args, "gamma_trainable", 1))
        # IPD
        self.use_emile = int(getattr(args, "use_emile", 0))
        self.lambda_ipd = float(getattr(args, "lambda_ipd", 0.1)); self.ipd_margin = float(getattr(args, "ipd_margin", 0.2))
        self.emile_use_fused_itememb = int(getattr(args, "emile_use_fused_itememb", 0))
        self.emile_warmup_steps = int(getattr(args, "emile_warmup_steps", 5000))
        if self.use_emile:
            g = torch.Generator(device="cpu"); g.manual_seed(int(getattr(args, "random_seed", 1)) + 2027)
            self.register_buffer("emile_T", torch.randn(self.emb_size, generator=g))
        else: self.register_buffer("emile_T", torch.zeros(self.emb_size))
        # SHNC
        self.use_shnc = int(getattr(args, "use_shnc", 1))
        self.shnc_path = getattr(args, "shnc_path", "")
        self.lambda_shnc = float(getattr(args, "lambda_shnc", 0.01))
        self.shnc_warmup_steps = int(getattr(args, "shnc_warmup_steps", 5000))
        self.shnc_num = int(getattr(args, "shnc_num", 1))
        self.shnc_neg_reduce = str(getattr(args, "shnc_neg_reduce", "mean"))
        self.shnc_score_norm = int(getattr(args, "shnc_score_norm", 0))
        self.shnc_detach_user = int(getattr(args, "shnc_detach_user", 0))
        self.shnc_sample_mode = str(getattr(args, "shnc_sample_mode", "random"))
        self.shnc_filter_history = int(getattr(args, "shnc_filter_history", 1))
        self.shnc_rank_start = int(getattr(args, "shnc_rank_start", 0))
        self.shnc_rank_end = int(getattr(args, "shnc_rank_end", 100))
        self.shnc_resample_attempts = int(getattr(args, "shnc_resample_attempts", 5))
        if self.use_shnc:
            if not self.shnc_path: raise ValueError("use_shnc=1 requires --shnc_path")
            hn = torch.from_numpy(np.asarray(pickle.load(open(self.shnc_path, "rb")))).long()
            self.register_buffer("semantic_hardneg_table", hn, persistent=False)
            logging.info(f"[SHNC] Loaded hardneg table: {hn.shape}")
        self._define_params(); self.apply(self.init_weights)
        if self.use_llmemb: self.align_loss_func = InfoNCEAlign(tau=self.tau)
        if self.use_llmemb and self.init_ckpt:
            self.load_model(self.init_ckpt, strict=bool(self.init_strict))
            logging.info(f"[MyModelSHNC] Warm-start from {self.init_ckpt}")
        else: logging.info("[MyModelSHNC] Train from scratch")
        self.global_step = 0

    def _define_params(self):
        self.interest_extractor = MultiInterestExtractor(
            k=self.K, item_num=self.item_num, emb_size=self.emb_size, attn_size=self.attn_size,
            max_his=self.max_his, prompt_num=self.prompt_num, lamb=self.lamb,
            use_llmemb=self.use_llmemb, llm_emb_path=self.llm_emb_path, srs_emb_path=self.srs_emb_path,
            llm_fuse=self.llm_fuse, gamma_init=self.gamma_init, gamma_trainable=self.gamma_trainable)
        self.proj = nn.Sequential()
        for i in range(max(0, self.n_layers - 1)):
            self.proj.add_module(f"proj_{i}", nn.Linear(self.emb_size, self.emb_size))
            self.proj.add_module(f"dropout_{i}", nn.Dropout(p=0.5)); self.proj.add_module(f"relu_{i}", nn.ReLU(inplace=True))
        self.proj.add_module("proj_final", nn.Linear(self.emb_size, self.K))

    def load_model(self, model_path=None, strict=False):
        if model_path is None: model_path = self.model_path
        model_dict = self.state_dict(); state_dict = torch.load(model_path, map_location="cpu")
        if strict: self.load_state_dict(state_dict, strict=True)
        else:
            exist = {k: v for k, v in state_dict.items() if k in model_dict
                     and hasattr(v, "shape") and hasattr(model_dict[k], "shape")
                     and v.shape == model_dict[k].shape}
            model_dict.update(exist); self.load_state_dict(model_dict, strict=False)
        logging.info("Load model from " + model_path)

    def _alpha_t(self):
        if self.rat_alpha_warmup_steps <= 0: return self.alpha
        return self.alpha * min(self.global_step, self.rat_alpha_warmup_steps) / float(self.rat_alpha_warmup_steps)
    def _emile_w(self):
        if self.emile_warmup_steps <= 0: return 1.0
        return min(self.global_step, self.emile_warmup_steps) / float(self.emile_warmup_steps)
    def _shnc_w(self):
        if self.shnc_warmup_steps <= 0: return 1.0
        return min(self.global_step, self.shnc_warmup_steps) / float(self.shnc_warmup_steps)

    @staticmethod
    def _cos_sim(a, b, eps=1e-8): return (F.normalize(a, dim=-1, eps=eps) * F.normalize(b, dim=-1, eps=eps)).sum(dim=-1)
    @staticmethod
    def _cos_dist(a, b, eps=1e-8): return 1.0 - MyModelSHNC._cos_sim(a, b, eps=eps)
    @staticmethod
    def _bpr_dist(pos_dist, neg_dist, margin=0.0): return F.softplus((pos_dist - neg_dist) + margin).mean()

    # -------------------------
    # Item vector getter (consistent with forward recommendation)
    # -------------------------
    def get_item_vectors_for_shnc(self, item_ids):
        return self.interest_extractor.get_item_emb(item_ids)

    # -------------------------
    # SHNC loss
    # -------------------------
    def compute_shnc_loss(self, out_dict):
        pos_item_id = out_dict["shnc_pos_item_id"]           # (B,)
        user_vector = out_dict["shnc_user_vector"]            # (B, D)
        pos_score = out_dict["shnc_pos_score"]                # (B,)
        history_items = out_dict.get("shnc_history_items", None)  # (B, L)

        B = pos_item_id.size(0)
        full_topk = self.semantic_hardneg_table.size(1)

        # Rank window
        rank_start = max(0, min(self.shnc_rank_start, full_topk))
        rank_end = max(rank_start + 1, min(self.shnc_rank_end, full_topk))
        cand_full = self.semantic_hardneg_table[pos_item_id]  # (B, full_topk)
        cand = cand_full[:, rank_start:rank_end]              # (B, window)

        # History filtering
        total_valid = 0; total_filtered = 0
        hard_neg_ids_list = []
        for b in range(B):
            c = cand[b]  # (window,)
            # Build exclusion set
            excluded = {pos_item_id[b].item(), 0}
            if self.shnc_filter_history and history_items is not None:
                for h in history_items[b].tolist():
                    if h > 0:
                        excluded.add(h)
            # Filter
            valid_c = [v.item() for v in c if v.item() not in excluded]
            total_valid += len(valid_c)
            total_filtered += (c.size(0) - len(valid_c))

            K = self.shnc_num
            if len(valid_c) >= K:
                sel = np.random.choice(valid_c, size=K, replace=False)
            else:
                # Fallback: use unfiltered cand excluding 0 and pos
                fallback = [v.item() for v in cand_full[b] if v.item() not in {0, pos_item_id[b].item()}]
                if len(fallback) >= K:
                    sel = np.random.choice(fallback, size=K, replace=False)
                elif len(fallback) > 0:
                    sel = np.random.choice(fallback, size=K, replace=True)
                else:
                    sel = np.full(K, max(1, pos_item_id[b].item()), dtype=np.int64)
            hard_neg_ids_list.append(torch.tensor(sel, device=c.device, dtype=torch.long))

        hard_neg_ids = torch.stack(hard_neg_ids_list, dim=0)  # (B, shnc_num)
        valid_neg_rate = total_valid / max(1, B * self.shnc_num)
        filtered_rate = total_filtered / max(1, B * (rank_end - rank_start))

        hard_neg_emb = self.get_item_vectors_for_shnc(hard_neg_ids)  # (B, shnc_num, D)
        if self.shnc_score_norm:
            uv = F.normalize(user_vector, dim=-1)
            hne = F.normalize(hard_neg_emb, dim=-1)
        else:
            uv, hne = user_vector, hard_neg_emb
        hard_neg_scores = (uv[:, None, :] * hne).sum(dim=-1)  # (B, shnc_num)

        if self.shnc_neg_reduce == "max":
            hard_score = hard_neg_scores.max(dim=1).values
            loss_shnc = -F.logsigmoid(pos_score - hard_score).mean()
        else:
            loss_shnc = -F.logsigmoid(pos_score[:, None] - hard_neg_scores).mean()

        aux = {"pos_score_mean": pos_score.detach().mean(),
               "hard_neg_score_mean": hard_neg_scores.detach().mean(),
               "valid_neg_rate": valid_neg_rate,
               "filtered_rate": filtered_rate}
        return loss_shnc, aux

    # ========================= forward =========================
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
            out_dict["emile_interest_vectors"] = interest_vectors; out_dict["emile_user_vector"] = u_base
            out_dict["emile_w"] = w; out_dict["emile_pos_ids"] = i_ids[:, 0]
            out_dict["emile_neg_ids"] = i_ids[:, 1] if i_ids.size(1) > 1 else None

        if self.use_shnc:
            out_dict["shnc_user_vector"] = u_base
            out_dict["shnc_pos_item_id"] = i_ids[:, 0]
            out_dict["shnc_pos_score"] = prediction[:, 0]
            out_dict["shnc_history_items"] = history

        if self.use_llmemb:
            pos_ids = i_ids[:, 0]; mask = (pos_ids != 0)
            if mask.any():
                ids = pos_ids[mask]
                out_dict["align_loss"] = self.align_loss_func(
                    self.interest_extractor.get_anchor_emb(ids), self.interest_extractor.get_llm_emb(ids))
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

    # ========================= loss =========================
    def loss(self, out_dict):
        loss = super().loss(out_dict)
        if self.use_llmemb and ("align_loss" in out_dict): loss = loss + self._alpha_t() * out_dict["align_loss"]
        if self.use_emile:
            neg_ids = out_dict.get("emile_neg_ids", None)
            if neg_ids is not None:
                iv = out_dict["emile_interest_vectors"]; w = out_dict["emile_w"]
                pos_ids = out_dict["emile_pos_ids"]; ie = self.interest_extractor
                pos_v = ie.get_item_emb(pos_ids) if self.emile_use_fused_itememb else ie.get_cf_emb(pos_ids)
                neg_v = ie.get_item_emb(neg_ids) if self.emile_use_fused_itememb else ie.get_cf_emb(neg_ids)
                H_vec = (iv * w[:, :, None]).sum(dim=1)
                d_pos_h = self._cos_dist(pos_v[:, None, :], iv)
                L_ipd = (self._bpr_dist(self._cos_dist(pos_v, H_vec), d_pos_h.min(dim=1).values, margin=self.ipd_margin) +
                         self._bpr_dist(self._cos_dist(pos_v, H_vec), self._cos_dist(neg_v, H_vec), margin=self.ipd_margin) +
                         self._bpr_dist(d_pos_h.min(dim=1).values, self._cos_dist(neg_v, H_vec), margin=self.ipd_margin))
                loss = loss + self._emile_w() * (self.lambda_ipd * L_ipd); out_dict["loss_ipd"] = L_ipd.detach()

        if self.use_shnc:
            loss_shnc, aux = self.compute_shnc_loss(out_dict)
            loss = loss + self._shnc_w() * self.lambda_shnc * loss_shnc
            out_dict["loss_shnc"] = loss_shnc.detach()
            if self.training and self.global_step % 1000 == 0:
                score_gap = aux["pos_score_mean"] - aux["hard_neg_score_mean"]
                logging.info(
                    f"[SHNC] step={self.global_step} w={self._shnc_w():.4f} lambda={self.lambda_shnc:.4f} "
                    f"loss={loss_shnc.item():.6f} pos_mean={aux['pos_score_mean'].item():.4f} "
                    f"hn_mean={aux['hard_neg_score_mean'].item():.4f} gap={score_gap.item():.4f} "
                    f"filter_history={self.shnc_filter_history} "
                    f"rank_start={self.shnc_rank_start} rank_end={self.shnc_rank_end} "
                    f"valid_neg_rate={aux['valid_neg_rate']:.4f} filtered_rate={aux['filtered_rate']:.4f} "
                    f"sample_mode={self.shnc_sample_mode} shnc_num={self.shnc_num}")
        return loss
