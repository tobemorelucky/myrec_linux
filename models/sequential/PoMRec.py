# -*- coding: UTF-8 -*-
import logging
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import SequentialModel


# =========================
#  Utils: load embedding tables
# =========================
def _ensure_2d_np(x):
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"embedding must be 2D array, got shape {x.shape}")
    return x


def _load_llm_table_pkl(path: str, expected_num_items_plus1: int) -> torch.Tensor:
    """
    Read LLM embedding pkl, return table (N1, d).
    Supports:
      - (N1, d) already has row0 padding
      - (N1-1, d) no row0 -> prepend zeros
      - otherwise -> truncate/pad to N1 (safe but not recommended)
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
    Load stage0 collaborative item emb, return (N1, emb), with row0=0.
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
#  Alignment loss (stable)
# =========================
class InfoNCEAlign(nn.Module):
    """
    Symmetric InfoNCE alignment (hard diagonal labels).
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
        loss1 = F.cross_entropy(logits, labels)
        loss2 = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss1 + loss2)


# =========================
#  MultiInterestExtractor (PoMRec backbone) + LLM branch
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
        freeze_llm: bool = True,   # kept for CLI compatibility; llm_table is buffer anyway
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

        # --- Collaborative embedding (PoMRec core, trainable) ---
        self.i_embeddings = nn.Embedding(item_num, emb_size)

        # --- Position embedding ---
        self.p_embeddings = nn.Embedding(max_his + 1, emb_size)

        # --- LLM table + adapter (for fusion and/or alignment) ---

        if self.use_llmemb:
            if not llm_emb_path:
                raise ValueError("use_llmemb=1 but llm_emb_path is empty")

            llm_table = _load_llm_table_pkl(llm_emb_path, expected_num_items_plus1=item_num)
            # 修改1
            # self.register_buffer("llm_table", llm_table)  # frozen buffer (N1, d_llm)
            self.register_buffer("llm_table", llm_table, persistent=False)
            d_llm = llm_table.size(1)

            # adapter: (d_llm -> emb_size)
            self.adapter = nn.Sequential(
                nn.Linear(d_llm, d_llm // 2),
                nn.GELU(),
                nn.Linear(d_llm // 2, emb_size),
                nn.LayerNorm(emb_size),
            )

            # gamma for fusion
            if self.gamma_trainable:
                self.log_gamma = nn.Parameter(
                    torch.log(torch.exp(torch.tensor(gamma_init)) - 1.0))  # softplus^{-1}(gamma_init)

                def gamma_value(self):
                    return F.softplus(self.log_gamma)  # >0
            else:
                self.register_buffer("gamma", torch.tensor(float(gamma_init)))

            # optional fixed anchor table from stage0 (recommended)
            if srs_emb_path:
                srs_table = _load_srs_emb_pkl(srs_emb_path, expected_num_items_plus1=item_num)
                self.srs_emb = nn.Embedding.from_pretrained(srs_table, freeze=True)

        # --- Prompts ---
        self.max_prompt = 5
        pad_len = max(0, self.max_prompt - self.prompt_num)
        self.register_buffer("prompt_pad", torch.ones(pad_len, emb_size))

        self.prompt1 = nn.Embedding(self.prompt_num, emb_size)
        self.prompt2 = nn.Embedding(self.prompt_num, emb_size)

        # --- Attention blocks (as original PoMRec) ---
        self.W1 = nn.Linear(emb_size, attn_size)
        self.W2 = nn.Linear(attn_size, self.K)

        self.W3 = nn.Linear(emb_size, attn_size)
        self.W4 = nn.Linear(attn_size, 1)

        self.map = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(2, 1), dtype=torch.float32, requires_grad=True)
        ))

    # ----- embedding getters -----
    def get_cf_emb(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.i_embeddings(item_ids)

    def get_llm_emb(self, item_ids: torch.Tensor) -> torch.Tensor:
        if not self.use_llmemb:
            raise RuntimeError("get_llm_emb called but use_llmemb=0")
        z = self.llm_table[item_ids]  # (..., d_llm)
        return self.adapter(z)        # (..., emb)

    def get_anchor_emb(self, item_ids: torch.Tensor) -> torch.Tensor:
        # if stage0 anchor provided -> use it; else use current CF embedding
        if self.use_llmemb and (self.srs_emb is not None):
            return self.srs_emb(item_ids)
        return self.get_cf_emb(item_ids)

    def get_item_emb(self, item_ids: torch.Tensor) -> torch.Tensor:
        e_cf = self.get_cf_emb(item_ids)
        if (not self.use_llmemb) or (not self.llm_fuse):
            return e_cf

        e_llm = self.get_llm_emb(item_ids)

        if hasattr(self, "log_gamma"):
            g = F.softplus(self.log_gamma)
            # g = torch.clamp(g, 0.0, 0.2)  # 可选
        else:
            g = self.gamma  # buffer or Parameter

        return e_cf + g * e_llm

    # ----- attention helper -----
    @staticmethod
    def value2attn(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = values.masked_fill(mask.unsqueeze(-1) == 0, -np.inf)
        values = values.transpose(-1, -2)  # (bsz, K, L) or (bsz, 1, L)
        attn = (values - values.max()).softmax(dim=-1)
        attn = attn.masked_fill(torch.isnan(attn), 0)
        return attn

    # ----- REQUIRED forward -----
    def forward(self, history: torch.Tensor, lengths: torch.Tensor):
        """
        Return:
          interest_vectors: (bsz, K, emb)
          distri_vectors:   (bsz, emb)
        """
        batch_size, seq_len = history.shape
        device = history.device

        valid_his = (history > 0).long()
        his_vectors = self.get_item_emb(history)  # (bsz, his_max, emb)  # NOTE: uses fused embedding if enabled

        # position encoding
        len_range = torch.arange(self.max_his, device=device)
        position = (lengths[:, None] - len_range[None, :seq_len]) * valid_his
        pos_vectors = self.p_embeddings(position)
        his_vectors = his_vectors + pos_vectors

        # extend mask for prompts
        valid_his = torch.cat([valid_his, torch.ones([batch_size, self.max_prompt], device=device)], dim=1)

        # ---- Multi-Interest Extraction ----
        prompt1 = torch.cat([self.prompt_pad.to(device), self.prompt1.weight], dim=0)
        prompt1 = prompt1.unsqueeze(0).expand(batch_size, -1, -1)
        his_vectors_prompt1 = torch.cat([his_vectors, prompt1], dim=1)

        attn_score = self.W2(self.W1(his_vectors_prompt1).tanh())  # (bsz, L, K)
        attn_score = self.value2attn(attn_score, valid_his)        # (bsz, K, L)

        interest_vectors = (his_vectors_prompt1[:, None, :, :] * attn_score[:, :, :, None]).sum(-2)  # (bsz, K, emb)

        # variance
        var = []
        for kk in range(self.K):
            x_mean_2 = (his_vectors_prompt1 - interest_vectors[:, kk:kk + 1, :]) ** 2
            var_k = torch.matmul(attn_score[:, kk:kk + 1, :], x_mean_2)
            var_k = torch.sqrt(var_k)
            var.append(var_k)
        variance = torch.cat(var, 1)
        interest_vectors = interest_vectors + self.lamb * variance

        # ---- Interest Distribution Predict ----
        prompt2 = torch.cat([self.prompt_pad.to(device), self.prompt2.weight], dim=0)
        prompt2 = prompt2.unsqueeze(0).expand(batch_size, -1, -1)
        his_vectors_prompt2 = torch.cat([his_vectors, prompt2], dim=1)

        distri_pred = self.W4(self.W3(his_vectors_prompt2).tanh())  # (bsz, L, 1)
        distri_pred = self.value2attn(distri_pred, valid_his)       # (bsz, 1, L)
        distri_vectors = torch.matmul(distri_pred, his_vectors_prompt2).squeeze(1)  # (bsz, emb)

        return interest_vectors, distri_vectors


# =========================
#  PoMRec
# =========================
class PoMRec(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'

    extra_log_args = [
        "K", "prompt_num", "lamb", "random_seed",
        "use_llmemb", "freeze_emb", "alpha", "tau",
        "rat_alpha_warmup_steps", "align_on", "align_sample_k",
        "llm_fuse", "gamma_init", "gamma_trainable",
    ]

    @staticmethod
    def parse_model_args(parser):
        # ---- original PoMRec args ----
        parser.add_argument('--emb_size', type=int, default=64)
        parser.add_argument('--attn_size', type=int, default=8)
        parser.add_argument('--K', type=int, default=3)
        parser.add_argument('--prompt_num', type=int, default=4)
        parser.add_argument('--n_layers', type=int, default=1)
        parser.add_argument('--lamb', type=float, default=3)

        # ---- LLMEmb-inspired alignment (regularization) ----
        parser.add_argument('--use_llmemb', type=int, default=0,
                            help='0: plain PoMRec; 1: add alignment regularizer (and optional fusion)')
        parser.add_argument('--llm_emb_path', type=str, default='',
                            help='LLM embedding table pkl, recommended PCA table (N or N+1, d)')
        parser.add_argument('--freeze_emb', action='store_true',
                            help='Kept for CLI compatibility (llm_table is buffer anyway)')
        parser.add_argument('--srs_emb_path', type=str, default='',
                            help='Stage0 exported collab item emb pkl (N or N+1, emb). '
                                 'If provided, align to it; else align to current i_embeddings.')

        parser.add_argument('--alpha', type=float, default=0.01, help='Alignment loss weight')
        parser.add_argument('--tau', type=float, default=0.2, help='InfoNCE temperature')

        parser.add_argument('--init_ckpt', type=str, default='',
                            help='Warm-start from baseline PoMRec checkpoint (optional)')
        parser.add_argument('--init_strict', type=int, default=0,
                            help='1 strict load; 0 partial load (recommended)')

        parser.add_argument('--rat_alpha_warmup_steps', type=int, default=0,
                            help='Warmup alpha linearly for first N steps. 0 = no warmup.')

        parser.add_argument('--align_on', type=str, default='pos',
                            choices=['pos', 'pos+his'],
                            help="Align on positive items only (pos) or also history items (pos+his).")
        parser.add_argument('--align_sample_k', type=int, default=0,
                            help="If align_on includes history, optionally sample k history items per user (0=use all nonzero).")

        # ---- fusion (cf + gamma * llm) ----
        parser.add_argument('--llm_fuse', type=int, default=1,
                            help='1: item_emb = cf + gamma*llm; 0: only cf (align-only)')
        parser.add_argument('--gamma_init', type=float, default=0.05, help='init gamma for fusion')
        parser.add_argument('--gamma_trainable', type=int, default=1, help='1: learnable gamma; 0: fixed gamma')

        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)

        # ---- base PoMRec params ----
        self.emb_size = args.emb_size
        self.attn_size = args.attn_size
        self.K = args.K
        self.prompt_num = args.prompt_num
        self.n_layers = args.n_layers
        self.lamb = args.lamb
        self.max_his = args.history_max

        # ---- llm params ----
        self.use_llmemb = int(getattr(args, "use_llmemb", 0))
        self.llm_emb_path = getattr(args, "llm_emb_path", "")
        self.freeze_emb = bool(getattr(args, "freeze_emb", False))
        self.srs_emb_path = getattr(args, "srs_emb_path", "")

        self.alpha = float(getattr(args, "alpha", 0.01))
        self.tau = float(getattr(args, "tau", 0.2))
        self.rat_alpha_warmup_steps = int(getattr(args, "rat_alpha_warmup_steps", 0))

        self.align_on = getattr(args, "align_on", "pos")
        self.align_sample_k = int(getattr(args, "align_sample_k", 0))

        self.init_ckpt = getattr(args, "init_ckpt", "")
        self.init_strict = int(getattr(args, "init_strict", 0))

        # IMPORTANT: set these BEFORE _define_params()
        self.llm_fuse = int(getattr(args, "llm_fuse", 1))
        self.gamma_init = float(getattr(args, "gamma_init", 0.05))
        self.gamma_trainable = int(getattr(args, "gamma_trainable", 1))

        # build modules
        self._define_params()
        self.apply(self.init_weights)

        if self.use_llmemb:
            self.align_loss_func = InfoNCEAlign(tau=self.tau)

        # warm-start if provided
        if self.use_llmemb and self.init_ckpt:
            self.load_model(self.init_ckpt, strict=bool(self.init_strict))
            logging.info(f"[PoMRec] Warm-start from {self.init_ckpt} (strict={bool(self.init_strict)})")
        else:
            logging.info("[PoMRec] Train from scratch")

        self.global_step = 0  # for alpha warmup

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
            freeze_llm=self.freeze_emb,
            srs_emb_path=self.srs_emb_path,
            llm_fuse=self.llm_fuse,
            gamma_init=self.gamma_init,
            gamma_trainable=self.gamma_trainable,
        )

        self.proj = nn.Sequential()
        for i in range(max(0, self.n_layers - 1)):
            self.proj.add_module(f'proj_{i}', nn.Linear(self.emb_size, self.emb_size))
            self.proj.add_module(f'dropout_{i}', nn.Dropout(p=0.5))
            self.proj.add_module(f'relu_{i}', nn.ReLU(inplace=True))
        self.proj.add_module('proj_final', nn.Linear(self.emb_size, self.K))

    def load_model(self, model_path=None, strict: bool = False):
        """
        Safe partial load with shape check (strict=False).
        """
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

        logging.info('Load model from ' + model_path)

    def _alpha_t(self) -> float:
        if self.rat_alpha_warmup_steps <= 0:
            return self.alpha
        t = min(self.global_step, self.rat_alpha_warmup_steps)
        return self.alpha * (t / float(self.rat_alpha_warmup_steps))

    @torch.no_grad()
    def _sample_history_ids(self, history_row: torch.Tensor, k: int) -> torch.Tensor:
        """
        history_row: (H,) with zeros padded
        return: (k,) sampled non-zero ids (with replacement if needed)
        """
        device = history_row.device
        his = history_row[history_row != 0]
        if his.numel() == 0:
            return torch.zeros((k,), dtype=history_row.dtype, device=device)
        if his.numel() >= k:
            idx = torch.randperm(his.numel(), device=device)[:k]
            return his[idx]
        # with replacement
        ridx = torch.randint(0, his.numel(), (k,), device=device)
        return his[ridx]

    def forward(self, feed_dict):
        self.global_step += 1

        import torch.nn.functional as F

        # ---- debug print gamma ----
        if self.use_llmemb and getattr(self, "llm_fuse", 0) and (self.global_step % 200 == 0):
            ie = self.interest_extractor
            with torch.no_grad():
                if hasattr(ie, "log_gamma"):
                    g = F.softplus(ie.log_gamma).item()
                elif hasattr(ie, "gamma"):
                    g = float(ie.gamma.detach().item())
                else:
                    g = None
            print(f"[step {self.global_step}] gamma={g}")

        if self.use_llmemb and getattr(self, "llm_fuse", 0) and (self.global_step % 200 == 0):
            ie = self.interest_extractor
            with torch.no_grad():
                # 取一个小 batch 的 pos item 来估计量级
                pos_ids = feed_dict["item_id"][:, 0]
                pos_ids = pos_ids[pos_ids != 0][:128]
                if pos_ids.numel() > 0:
                    e_cf = ie.get_cf_emb(pos_ids)
                    e_llm = ie.get_llm_emb(pos_ids)
                    if hasattr(ie, "log_gamma"):
                        g = F.softplus(ie.log_gamma)
                    else:
                        g = ie.gamma
                    ratio = (g * e_llm).norm(dim=-1).mean() / (e_cf.norm(dim=-1).mean() + 1e-12)
                    print(
                        f"[step {self.global_step}] gamma={float(g.detach().item()):.6f}  llm/cf_norm_ratio={float(ratio):.4f}")

        i_ids = feed_dict['item_id']          # (bsz, 1+neg)
        history = feed_dict['history_items']  # (bsz, max_his)
        lengths = feed_dict['lengths']        # (bsz,)

        interest_vectors, distri_vectors = self.interest_extractor(history, lengths)

        # main scoring: uses get_item_emb (cf or fused)
        i_vectors = self.interest_extractor.get_item_emb(i_ids)  # (bsz, cand, emb)
        pred_intent = self.proj(distri_vectors)                  # (bsz, K)
        user_vector = (interest_vectors * pred_intent.softmax(-1)[:, :, None]).sum(-2)  # (bsz, emb)
        prediction = (user_vector[:, None, :] * i_vectors).sum(-1)                      # (bsz, cand)

        out_dict = {'prediction': prediction}

        # alignment regularizer
        if self.use_llmemb:
            pos_ids = i_ids[:, 0]  # (bsz,)

            if self.align_on == "pos":
                align_ids = pos_ids
            else:
                # pos + history
                if self.align_sample_k > 0:
                    bsz = history.size(0)
                    device = history.device
                    sampled = torch.zeros((bsz, self.align_sample_k), dtype=history.dtype, device=device)
                    # sampling is small cost; keep it simple
                    for b in range(bsz):
                        sampled[b] = self._sample_history_ids(history[b], self.align_sample_k)
                    align_ids = torch.cat([pos_ids.unsqueeze(1), sampled], dim=1).reshape(-1)
                else:
                    align_ids = torch.cat([pos_ids.unsqueeze(1), history], dim=1).reshape(-1)

            mask = (align_ids != 0)
            if mask.any():
                ids = align_ids[mask]
                srs = self.interest_extractor.get_anchor_emb(ids)  # (M, emb)
                llm = self.interest_extractor.get_llm_emb(ids)     # (M, emb)
                out_dict['align_loss'] = self.align_loss_func(srs, llm)
            else:
                out_dict['align_loss'] = torch.zeros([], device=prediction.device)

        return out_dict

    def loss(self, out_dict: dict):
        loss = super().loss(out_dict)
        if self.use_llmemb and 'align_loss' in out_dict:
            loss = loss + self._alpha_t() * out_dict['align_loss']
        return loss
