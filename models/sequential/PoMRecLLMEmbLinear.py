# -*- coding: UTF-8 -*-
"""
PoMRecLLMEmbLinear — Fixed LLMEmb-style naive migration to PoMRec.

Paper baseline: PoMRec-LLMEmb-Linear
- Adapter: Linear(d_llm, d_llm//2) -> Linear(d_llm//2, emb_size)  (no GELU, no LN)
- Injection: both history encoding and candidate scoring use adapted LLM embedding
- Mode: replace (no CF item embedding, no residual)
- No alignment, no TIC, no MVTC, no ablation params.

Usage:
  python main.py --model_name PoMRecLLMEmbLinear --dataset beauty --random_seed 42
"""

import logging
import os
import pickle

import numpy as np
import torch
import torch.nn as nn

from models.sequential.PoMRec import PoMRec


# =========================
#  Utility: safe embedding loader
# =========================
def _safe_load_emb_table(path, expected_rows, name="embedding"):
    if path.endswith(".pt"):
        data = torch.load(path, map_location="cpu")
    else:
        data = pickle.load(open(path, "rb"))
    arr = np.asarray(data, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"[PoMRecLLMEmbLinear] Expected 2D {name}, got {arr.shape}")

    if arr.shape[0] == expected_rows:
        pass  # already has padding
    elif arr.shape[0] == expected_rows - 1:
        arr = np.vstack([np.zeros((1, arr.shape[1]), dtype=np.float32), arr])
    else:
        raise ValueError(
            f"[PoMRecLLMEmbLinear] Unexpected {name} shape {arr.shape}, "
            f"expected ({expected_rows - 1}, D) or ({expected_rows}, D)"
        )
    return torch.tensor(arr, dtype=torch.float32)


def _find_llm_emb_path(dataset):
    """Auto-discover LLM embedding file under data/<dataset>/handled/."""
    candidates = [
        f"./data/{dataset}/handled/llm_table_pca1536.pkl",
        f"./data/{dataset}/handled/my_llm_pca1536.pkl",
        f"./data/{dataset}/handled/llm_table_pca64.pkl",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"[PoMRecLLMEmbLinear] No LLM embedding found for dataset '{dataset}'. "
        f"Tried: {candidates}"
    )


# =========================
#  PoMRecLLMEmbLinear
# =========================
class PoMRecLLMEmbLinear(PoMRec):
    """PoMRec + LLMEmb-style naive linear adapter (replace mode, both scope)."""

    reader = PoMRec.reader
    runner = PoMRec.runner

    extra_log_args = [
        "K", "prompt_num", "lamb", "random_seed",
    ]

    @staticmethod
    def init_weights(m):
        """Safe init_weights using isinstance (parent uses str(type) which matches our name)."""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)

    @staticmethod
    def parse_model_args(parser):
        # Register only PoMRec base args — NO LLM/fusion/ablation args.
        parser = PoMRec.parse_model_args(parser)
        parser.add_argument('--llm_scale', type=float, default=1.0,
                            help='Scale factor on adapter output. 1.0 = no scaling.')
        return parser

    # ------------------------------------------------------------------
    #  __init__
    # ------------------------------------------------------------------
    def __init__(self, args, corpus):
        # Force PoMRec's internal LLM off
        args.use_llmemb = 0
        args.llm_fuse = 0

        self._llm_scale = float(getattr(args, 'llm_scale', 1.0))

        logging.info(f"[PoMRecLLMEmbLinear] initialized (LLMEmb-style naive migration), llm_scale={self._llm_scale}")

        super().__init__(args, corpus)

        # ---- Auto-load LLM embedding ----
        llm_emb_path = _find_llm_emb_path(args.dataset)
        logging.info(f"[PoMRecLLMEmbLinear] LLM emb path: {llm_emb_path}")
        llm_table = _safe_load_emb_table(llm_emb_path, self.item_num, name="LLM emb")
        logging.info(f"[PoMRecLLMEmbLinear]   shape: {llm_table.shape}")

        d_llm = llm_table.size(1)
        self.register_buffer("llm_table", llm_table, persistent=False)

        # ---- LLMEmb-style adapter (Linear -> Linear, no GELU, no LN) ----
        self.llm_adapter = nn.Sequential(
            nn.Linear(d_llm, d_llm // 2),
            nn.Linear(d_llm // 2, self.emb_size),
        )
        logging.info(
            f"[PoMRecLLMEmbLinear] Adapter built: "
            f"{d_llm} -> {d_llm // 2} -> {self.emb_size} (LLMEmb-style, no GELU, no LN)"
        )

        # ---- Patch extractor: replace mode, both scope ----
        self.interest_extractor.get_item_emb = self._get_item_emb

        self._step = 0

    # ------------------------------------------------------------------
    #  Item embedding: replace with adapted LLM
    # ------------------------------------------------------------------
    def _get_adapted_llm_emb(self, item_ids):
        return self.llm_adapter(self.llm_table[item_ids]) * self._llm_scale

    def _get_item_emb(self, item_ids):
        """Replace: e_final = adapter(e_llm).  No CF."""
        return self._get_adapted_llm_emb(item_ids)

    # ------------------------------------------------------------------
    #  Diagnostics
    # ------------------------------------------------------------------
    @staticmethod
    def _diag(name, tensor, step, force=False):
        """Log tensor stats at step 1, every 1000 steps, or when NaN/Inf detected."""
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        if not (force or step == 1 or step % 1000 == 0 or has_nan or has_inf):
            return has_nan, has_inf
        with torch.no_grad():
            t = tensor.float()
            logging.info(
                f"[DIAG step {step}] {name}: shape={tuple(tensor.shape)} "
                f"mean={t.mean().item():.6f} std={t.std().item():.6f} "
                f"max_abs={t.abs().max().item():.6f} "
                f"nan={has_nan} inf={has_inf}"
            )
        return has_nan, has_inf

    # ------------------------------------------------------------------
    #  Forward (identical to PoMRec scoring path)
    # ------------------------------------------------------------------
    def forward(self, feed_dict):
        self._step += 1

        i_ids = feed_dict["item_id"]
        history = feed_dict["history_items"]
        lengths = feed_dict["lengths"]

        interest_vectors, distri_vectors = self.interest_extractor(history, lengths)

        i_vectors = self._get_item_emb(i_ids)
        pred_intent = self.proj(distri_vectors)
        q = pred_intent.softmax(dim=-1)
        user_vector = (interest_vectors * q[:, :, None]).sum(-2)
        prediction = (user_vector[:, None, :] * i_vectors).sum(-1)

        out_dict = {"prediction": prediction}

        # ---- NaN/Inf diagnostics ----
        step = self._step
        force_diag = (step == 1 or step % 1000 == 0)

        # 1. LLM raw table (one-time at step 1)
        if step == 1:
            self._diag("llm_table_raw", self.llm_table, step, force=True)

        # 2. Adapter output
        nan_adapt, inf_adapt = self._diag("i_vectors(adapter_out)", i_vectors, step, force=force_diag)

        # 3. Extractor outputs
        self._diag("interest_vectors", interest_vectors, step, force=force_diag)
        self._diag("distri_vectors(raw)", distri_vectors, step, force=force_diag)
        self._diag("pred_intent(softmax)", q, step, force=force_diag)

        # 4. Candidate scores
        nan_pred, inf_pred = self._diag("prediction(scores)", prediction, step, force=force_diag)

        # 5. Force print on NaN/Inf
        if nan_adapt or inf_adapt or nan_pred or inf_pred:
            logging.warning(
                f"[DIAG step {step}] NaN/Inf DETECTED! "
                f"adapter_nan={nan_adapt} adapter_inf={inf_adapt} "
                f"pred_nan={nan_pred} pred_inf={inf_pred}"
            )
            # Also print i_ids range
            with torch.no_grad():
                logging.warning(
                    f"[DIAG step {step}] i_ids: min={i_ids.min().item()} max={i_ids.max().item()} "
                    f"history: min={history[history>0].min().item() if (history>0).any() else -1} "
                    f"max={history.max().item()}"
                )

        return out_dict

    # ------------------------------------------------------------------
    #  Loss
    # ------------------------------------------------------------------
    def loss(self, out_dict):
        loss = super().loss(out_dict)
        self._diag("loss", loss, self._step, force=torch.isnan(loss).any().item() or torch.isinf(loss).any().item())
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logging.warning(f"[DIAG step {self._step}] LOSS is NaN/Inf!")
        return loss
