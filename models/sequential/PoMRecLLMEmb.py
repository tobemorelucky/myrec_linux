# -*- coding: UTF-8 -*-
"""
PoMRecLLMEmb — PoMRec baseline with LLM semantic enhancement ablations.

Supports three fusion modes:
  - none:     Pure PoMRec, e_final = e_cf
  - replace:  LLMEmb-style, e_final = adapter(e_llm)  (PoMRec-LLMReplace)
  - residual: Semantic residual, e_final = e_cf + gamma * adapter(e_llm)  (PoMRec-LLMResidual)

Does NOT modify original PoMRec.py. All new logic is self-contained here.
"""

import logging
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sequential.PoMRec import PoMRec, InfoNCEAlign


# =========================
#  Utility: safe embedding table loader
# =========================
def _safe_load_emb_table(path, expected_rows, name="embedding"):
    """Load a 2D embedding table from .pkl or .pt, auto-pad row 0 if needed.

    Args:
        path: str, file path
        expected_rows: int, number of rows INCLUDING padding row 0 (i.e. item_num)
        name: str, label for log messages

    Returns:
        torch.Tensor of shape (expected_rows, dim), dtype=float32
    """
    if path.endswith(".pt"):
        data = torch.load(path, map_location="cpu")
    else:
        data = pickle.load(open(path, "rb"))

    arr = np.asarray(data, dtype=np.float32)

    logging.info(f"[PoMRecLLMEmb] Loading {name}: {path}")
    logging.info(f"[PoMRecLLMEmb]   original shape: {arr.shape}")

    if arr.ndim != 2:
        raise ValueError(f"[PoMRecLLMEmb] Expected 2D {name}, got shape {arr.shape}")

    if arr.shape[0] == expected_rows:
        logging.info(f"[PoMRecLLMEmb]   shape[0]=={expected_rows} (item_num), already has padding row 0")
    elif arr.shape[0] == expected_rows - 1:
        logging.info(f"[PoMRecLLMEmb]   shape[0]=={expected_rows - 1} (item_num-1), prepending padding row 0")
        arr = np.vstack([np.zeros((1, arr.shape[1]), dtype=np.float32), arr])
    else:
        raise ValueError(
            f"[PoMRecLLMEmb] Unexpected {name} shape: {arr.shape}. "
            f"Expected ({expected_rows - 1}, D) or ({expected_rows}, D)."
        )

    tensor = torch.tensor(arr, dtype=torch.float32)
    logging.info(f"[PoMRecLLMEmb]   final shape: {tensor.shape}, dtype: {tensor.dtype}")
    return tensor


# =========================
#  PoMRecLLMEmb
# =========================
class PoMRecLLMEmb(PoMRec):
    """PoMRec + LLM semantic fusion ablations (none / replace / residual)."""

    reader = PoMRec.reader
    runner = PoMRec.runner

    extra_log_args = [
        "K", "prompt_num", "lamb", "random_seed",
        "llm_fuse_mode", "use_llm_align",
        "gamma_init", "gamma_trainable", "align_weight",
    ]

    @staticmethod
    def parse_model_args(parser):
        # Register all PoMRec base args (emb_size, K, lamb, llm_emb_path, srs_emb_path, gamma_init, ...)
        parser = PoMRec.parse_model_args(parser)

        # ---- PoMRecLLMEmb-specific args (no conflict with PoMRec) ----
        parser.add_argument("--llm_fuse_mode", type=str, default="none",
                            choices=["none", "replace", "residual"],
                            help="none: pure PoMRec | replace: LLMEmb-style e=adapter(llm) | residual: e=cf+gamma*adapter(llm)")
        parser.add_argument("--freeze_llm_emb", type=int, default=1,
                            help="1: freeze LLM table (buffer); 0: trainable Parameter")
        parser.add_argument("--use_llm_align", type=int, default=0,
                            help="1: add InfoNCE alignment loss between adapter(llm) and frozen SRS emb")
        parser.add_argument("--align_weight", type=float, default=0.001,
                            help="Weight of alignment loss (analogous to PoMRec --alpha)")
        parser.add_argument("--align_tau", type=float, default=0.2,
                            help="InfoNCE temperature for alignment (analogous to PoMRec --tau)")

        return parser  # PoMRec.parse_model_args already called SequentialModel.parse_model_args

    # ------------------------------------------------------------------
    #  __init__
    # ------------------------------------------------------------------
    def __init__(self, args, corpus):
        # ---- Parse fusion / alignment args BEFORE super().__init__() ----
        self.llm_fuse_mode = getattr(args, "llm_fuse_mode", "none")
        self._llm_emb_path = getattr(args, "llm_emb_path", "")
        self._freeze_llm_emb = int(getattr(args, "freeze_llm_emb", 1))
        self._gamma_init = float(getattr(args, "gamma_init", 0.1))
        self._gamma_trainable = int(getattr(args, "gamma_trainable", 0))

        self._use_llm_align = int(getattr(args, "use_llm_align", 0))
        self._srs_emb_path = getattr(args, "srs_emb_path", "")
        self._align_weight = float(getattr(args, "align_weight", 0.001))
        self._align_tau = float(getattr(args, "align_tau", 0.2))

        # Force PoMRec's own LLM logic off — we handle everything ourselves
        args.use_llmemb = 0
        args.llm_fuse = 0

        logging.info("[PoMRecLLMEmb] initialized from PoMRec baseline")
        logging.info(f"[PoMRecLLMEmb] llm_fuse_mode={self.llm_fuse_mode}")

        super().__init__(args, corpus)

        # ---- Build LLM adapter if fusion is enabled ----
        if self.llm_fuse_mode != "none":
            if not self._llm_emb_path:
                raise ValueError(
                    "llm_fuse_mode={} requires --llm_emb_path".format(self.llm_fuse_mode)
                )
            self._build_llm_adapter()
            # Patch the extractor so its internal get_item_emb uses our fusion
            self.interest_extractor.get_item_emb = self._fused_get_item_emb

        # ---- Build alignment module ----
        if self._use_llm_align:
            if self.llm_fuse_mode == "none":
                raise ValueError("use_llm_align=1 requires llm_fuse_mode != none")
            self._build_alignment()

        # Step counter for debug logging
        self._step = 0

    # ------------------------------------------------------------------
    #  LLM embedding loading + adapter
    # ------------------------------------------------------------------
    def _load_llm_table(self):
        return _safe_load_emb_table(self._llm_emb_path, self.item_num, name="LLM emb")

    def _build_llm_adapter(self):
        llm_table = self._load_llm_table()
        d_llm = llm_table.size(1)

        # Store LLM table
        if self._freeze_llm_emb:
            self.register_buffer("llm_table", llm_table, persistent=False)
        else:
            self.llm_table = nn.Parameter(llm_table)

        # Adapter: d_llm -> emb_size (same architecture as LLMEmb / PoMRec)
        self.llm_adapter = nn.Sequential(
            nn.Linear(d_llm, d_llm // 2),
            nn.GELU(),
            nn.Linear(d_llm // 2, self.emb_size),
            nn.LayerNorm(self.emb_size),
        )

        # Gamma for residual mode
        if self.llm_fuse_mode == "residual":
            if self._gamma_trainable:
                self.log_gamma = nn.Parameter(
                    torch.log(torch.exp(torch.tensor(self._gamma_init)) - 1.0)
                )
            else:
                self.register_buffer("gamma", torch.tensor(float(self._gamma_init)))

        logging.info(
            f"[PoMRecLLMEmb] LLM adapter built: "
            f"d_llm={d_llm} -> emb_size={self.emb_size}, "
            f"freeze_llm_emb={self._freeze_llm_emb}"
        )

    def _get_adapted_llm_emb(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Return adapter(e_llm) for given item IDs."""
        return self.llm_adapter(self.llm_table[item_ids])

    def _gamma_value(self):
        """Return current gamma (scalar float tensor, no grad when fixed)."""
        if hasattr(self, "log_gamma"):
            return F.softplus(self.log_gamma)
        return self.gamma

    # ------------------------------------------------------------------
    #  Fused item embedding (replaces MultiInterestExtractor.get_item_emb)
    # ------------------------------------------------------------------
    def _fused_get_item_emb(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Apply the selected fusion mode.

        This method is patched onto self.interest_extractor so both
        history encoding and candidate scoring use the same fusion.
        """
        e_cf = self.interest_extractor.i_embeddings(item_ids)  # PoMRec CF embedding

        if self.llm_fuse_mode == "none":
            return e_cf

        e_llm = self._get_adapted_llm_emb(item_ids)

        if self.llm_fuse_mode == "replace":
            return e_llm
        elif self.llm_fuse_mode == "residual":
            return e_cf + self._gamma_value() * e_llm
        else:
            return e_cf  # fallback

    # ------------------------------------------------------------------
    #  Alignment
    # ------------------------------------------------------------------
    def _build_alignment(self):
        if not self._srs_emb_path:
            raise ValueError("use_llm_align=1 but srs_emb_path is empty")

        srs_table = _safe_load_emb_table(self._srs_emb_path, self.item_num, name="SRS emb")
        self.srs_emb = nn.Embedding.from_pretrained(srs_table, freeze=True)
        self.align_loss_fn = InfoNCEAlign(tau=self._align_tau)

        logging.info(
            f"[PoMRecLLMEmb] Alignment built: "
            f"align_weight={self._align_weight}, tau={self._align_tau}"
        )

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, feed_dict):
        self._step += 1

        i_ids = feed_dict["item_id"]          # (bsz, 1+neg)
        history = feed_dict["history_items"]  # (bsz, max_his)
        lengths = feed_dict["lengths"]        # (bsz,)

        # Multi-interest extraction (uses patched get_item_emb internally)
        interest_vectors, distri_vectors = self.interest_extractor(history, lengths)

        # Candidate scoring with fused embeddings
        i_vectors = self._fused_get_item_emb(i_ids)              # (bsz, cand, emb)
        pred_intent = self.proj(distri_vectors)                   # (bsz, K)
        user_vector = (interest_vectors * pred_intent.softmax(-1)[:, :, None]).sum(-2)  # (bsz, emb)
        prediction = (user_vector[:, None, :] * i_vectors).sum(-1)                      # (bsz, cand)

        out_dict = {"prediction": prediction}

        # ---- Alignment loss (on positive items only) ----
        if self._use_llm_align:
            pos_ids = i_ids[:, 0]        # (bsz,)
            mask = (pos_ids != 0)
            if mask.any():
                ids = pos_ids[mask]
                srs = self.srs_emb(ids)                    # (M, emb)
                llm = self._get_adapted_llm_emb(ids)       # (M, emb)
                out_dict["align_loss"] = self.align_loss_fn(srs, llm)
            else:
                out_dict["align_loss"] = torch.zeros([], device=prediction.device)

        # ---- Debug: print gamma every 200 steps ----
        if self.llm_fuse_mode == "residual" and self._step % 200 == 0:
            with torch.no_grad():
                g = self._gamma_value().item()
            logging.info(f"[PoMRecLLMEmb step {self._step}] gamma={g:.6f}")

        return out_dict

    # ------------------------------------------------------------------
    #  Loss
    # ------------------------------------------------------------------
    def loss(self, out_dict: dict):
        # Base BPR loss from SequentialModel / GeneralModel
        loss = super().loss(out_dict)
        if self._use_llm_align and "align_loss" in out_dict:
            loss = loss + self._align_weight * out_dict["align_loss"]
        return loss
