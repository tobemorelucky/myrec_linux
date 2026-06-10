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
        "llm_fuse_mode", "llm_adapter_arch", "llm_inject_scope",
        "use_llm_align", "use_tic", "tic_score_mode", "use_mvtc",
        "gamma_init", "gamma_trainable",
        "align_weight", "tic_weight", "mvtc_weight",
        "tic_score_lambda", "tic_score_eta",
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

        # ---- LLM adapter architecture & injection scope ablations ----
        parser.add_argument("--llm_adapter_arch", type=str, default="ours",
                            choices=["ours", "llmemb", "noln", "linear"],
                            help="ours: L->GELU->L->LN | llmemb: L->L | noln: L->GELU->L | linear: single L")
        parser.add_argument("--llm_inject_scope", type=str, default="both",
                            choices=["both", "history_only", "candidate_only"],
                            help="both: history+candidate | history_only: only history fused | candidate_only: only candidate fused")
        parser.add_argument("--use_llm_align", type=int, default=0,
                            help="1: add InfoNCE alignment loss between adapter(llm) and frozen SRS emb")
        parser.add_argument("--align_weight", type=float, default=0.001,
                            help="Weight of alignment loss (analogous to PoMRec --alpha)")
        parser.add_argument("--align_tau", type=float, default=0.2,
                            help="InfoNCE temperature for alignment (analogous to PoMRec --tau)")

        # ---- Target-Interest Consistency (TIC) ----
        parser.add_argument("--use_tic", type=int, default=0,
                            help="1: add target-interest consistency loss (cosine-distance BPR)")
        parser.add_argument("--tic_weight", type=float, default=0.001,
                            help="Weight of TIC loss")
        parser.add_argument("--tic_tau", type=float, default=0.2,
                            help="TIC margin for BPR softplus")

        # ---- Candidate-aware scoring ----
        parser.add_argument("--tic_score_mode", type=str, default="none",
                            choices=["none", "candidate", "residual", "mix"],
                            help="none: base PoMRec | candidate: pure cand-weight | residual: base+lambda*cand | mix: (1-eta)*q+eta*cand")
        parser.add_argument("--tic_score_tau", type=float, default=0.2,
                            help="Temperature for candidate-aware interest softmax")
        parser.add_argument("--tic_score_lambda", type=float, default=0.1,
                            help="Weight for residual cand_score (base + lambda * cand)")
        parser.add_argument("--tic_score_eta", type=float, default=0.1,
                            help="Interpolation weight for mix mode (1-eta)*q + eta*cand")

        # ---- MVTC: Multi-View Target Consistency (CF-teacher -> intent KL) ----
        parser.add_argument("--use_mvtc", type=int, default=0,
                            help="1: add MVTC KL loss (stopgrad CF-teacher -> intent distribution)")
        parser.add_argument("--mvtc_weight", type=float, default=0.001,
                            help="Weight of MVTC loss")
        parser.add_argument("--mvtc_tau_cf", type=float, default=0.2,
                            help="Temperature for CF-teacher distribution")

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

        # ---- Adapter architecture & injection scope ablations ----
        self._llm_adapter_arch = getattr(args, "llm_adapter_arch", "ours")
        self._llm_inject_scope = getattr(args, "llm_inject_scope", "both")

        # ---- TIC args ----
        self._use_tic = int(getattr(args, "use_tic", 0))
        self._tic_weight = float(getattr(args, "tic_weight", 0.001))
        self._tic_tau = float(getattr(args, "tic_tau", 0.2))

        # ---- Candidate-aware scoring args ----
        self._tic_score_mode = getattr(args, "tic_score_mode", "none")
        self._tic_score_tau = float(getattr(args, "tic_score_tau", 0.2))
        self._tic_score_lambda = float(getattr(args, "tic_score_lambda", 0.1))
        self._tic_score_eta = float(getattr(args, "tic_score_eta", 0.1))

        # ---- MVTC args ----
        self._use_mvtc = int(getattr(args, "use_mvtc", 0))
        self._mvtc_weight = float(getattr(args, "mvtc_weight", 0.001))
        self._mvtc_tau_cf = float(getattr(args, "mvtc_tau_cf", 0.2))

        # Force PoMRec's own LLM logic off — we handle everything ourselves
        args.use_llmemb = 0
        args.llm_fuse = 0

        logging.info("[PoMRecLLMEmb] initialized from PoMRec baseline")
        logging.info(f"[PoMRecLLMEmb] llm_fuse_mode={self.llm_fuse_mode} "
                     f"adapter_arch={self._llm_adapter_arch} inject_scope={self._llm_inject_scope} "
                     f"use_llm_align={self._use_llm_align} "
                     f"use_tic={self._use_tic} tic_score_mode={self._tic_score_mode} "
                     f"use_mvtc={self._use_mvtc} "
                     f"tic_weight={self._tic_weight} tic_score_lambda={self._tic_score_lambda} "
                     f"tic_score_eta={self._tic_score_eta} mvtc_weight={self._mvtc_weight}")

        super().__init__(args, corpus)

        # ---- Build LLM adapter if fusion is enabled ----
        if self.llm_fuse_mode != "none":
            if not self._llm_emb_path:
                raise ValueError(
                    "llm_fuse_mode={} requires --llm_emb_path".format(self.llm_fuse_mode)
                )
            self._build_llm_adapter()
            # Patch extractor based on injection scope
            if self._llm_inject_scope in ("both", "history_only"):
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

        # Adapter: d_llm -> emb_size
        arch = self._llm_adapter_arch
        if arch == "ours":
            self.llm_adapter = nn.Sequential(
                nn.Linear(d_llm, d_llm // 2),
                nn.GELU(),
                nn.Linear(d_llm // 2, self.emb_size),
                nn.LayerNorm(self.emb_size),
            )
        elif arch == "llmemb":
            self.llm_adapter = nn.Sequential(
                nn.Linear(d_llm, d_llm // 2),
                nn.Linear(d_llm // 2, self.emb_size),
            )
        elif arch == "noln":
            self.llm_adapter = nn.Sequential(
                nn.Linear(d_llm, d_llm // 2),
                nn.GELU(),
                nn.Linear(d_llm // 2, self.emb_size),
            )
        elif arch == "linear":
            self.llm_adapter = nn.Sequential(
                nn.Linear(d_llm, self.emb_size),
            )
        else:
            raise ValueError(f"Unknown llm_adapter_arch: {arch}")

        # Gamma for residual mode
        if self.llm_fuse_mode == "residual":
            if self._gamma_trainable:
                self.log_gamma = nn.Parameter(
                    torch.log(torch.exp(torch.tensor(self._gamma_init)) - 1.0)
                )
            else:
                self.register_buffer("gamma", torch.tensor(float(self._gamma_init)))

        logging.info(
            f"[PoMRecLLMEmb] LLM adapter built: arch={arch} "
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
    #  TIC: Target-Interest Consistency (cosine-distance BPR)
    # ------------------------------------------------------------------
    @staticmethod
    def _cos_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Cosine similarity along last dim."""
        a = F.normalize(a, dim=-1, eps=eps)
        b = F.normalize(b, dim=-1, eps=eps)
        return (a * b).sum(dim=-1)

    @staticmethod
    def _cos_dist(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Cosine distance along last dim."""
        return 1.0 - PoMRecLLMEmb._cos_sim(a, b, eps=eps)

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

        # Candidate scoring — always compute base_score (PoMRec original path)
        # Inject scope: candidate embedding source
        if self._llm_inject_scope == "history_only":
            i_vectors = self.interest_extractor.i_embeddings(i_ids)  # pure CF for candidates
        else:
            i_vectors = self._fused_get_item_emb(i_ids)              # fused (both / candidate_only)
        pred_intent = self.proj(distri_vectors)                   # (B, K)
        q = pred_intent.softmax(dim=-1)                           # (B, K)
        user_vector = (interest_vectors * q[:, :, None]).sum(-2)   # (B, D)
        base_score = (user_vector[:, None, :] * i_vectors).sum(-1) # (B, C) — PoMRec original

        # Raw dot-product sim (no normalize — keeps scale)
        raw_sim = torch.einsum("bke,bce->bck", interest_vectors, i_vectors)  # (B, C, K)

        mode = self._tic_score_mode
        if mode == "none":
            prediction = base_score

        elif mode == "candidate":
            cand_weight = torch.softmax(raw_sim / self._tic_score_tau, dim=-1)  # (B, C, K)
            prediction = (cand_weight * raw_sim).sum(dim=-1)                     # (B, C)

        elif mode == "residual":
            cand_weight = torch.softmax(raw_sim / self._tic_score_tau, dim=-1)
            cand_score = (cand_weight * raw_sim).sum(dim=-1)
            prediction = base_score + self._tic_score_lambda * cand_score

        elif mode == "mix":
            cand_weight = torch.softmax(raw_sim / self._tic_score_tau, dim=-1)   # (B, C, K)
            final_weight = (1.0 - self._tic_score_eta) * q[:, None, :] + self._tic_score_eta * cand_weight
            prediction = (final_weight * raw_sim).sum(dim=-1)

        else:
            prediction = base_score  # fallback

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

        # ---- TIC stash (used in loss()) ----
        if self._use_tic:
            out_dict["tic_interest_vectors"] = interest_vectors       # (B, K, D)
            out_dict["tic_user_vector"] = user_vector                 # (B, D)
            out_dict["tic_pos_ids"] = i_ids[:, 0]                     # (B,)

        # ---- MVTC stash (used in loss()) ----
        if self._use_mvtc:
            out_dict["mvtc_interest_vectors"] = interest_vectors      # (B, K, D)
            out_dict["mvtc_pred_intent"] = pred_intent                # (B, K)
            out_dict["mvtc_pos_ids"] = i_ids[:, 0]                     # (B,)

        # ---- Debug prints ----
        if self._step % 200 == 0:
            if self.llm_fuse_mode == "residual":
                with torch.no_grad():
                    g = self._gamma_value().item()
                logging.info(f"[PoMRecLLMEmb step {self._step}] gamma={g:.6f}")

        return out_dict

    # ------------------------------------------------------------------
    #  Loss
    # ------------------------------------------------------------------
    def loss(self, out_dict: dict):
        # Base BPR loss from SequentialModel / GeneralModel
        rec_loss = super().loss(out_dict)
        loss = rec_loss

        # Alignment
        if self._use_llm_align and "align_loss" in out_dict:
            loss = loss + self._align_weight * out_dict["align_loss"]

        # TIC: Target-Interest Consistency
        tic_loss = None
        if self._use_tic and "tic_interest_vectors" in out_dict:
            iv = out_dict["tic_interest_vectors"]      # (B, K, D)
            uv = out_dict["tic_user_vector"]            # (B, D)
            pos_ids = out_dict["tic_pos_ids"]            # (B,)
            mask = (pos_ids != 0)

            if mask.any():
                pos_ids = pos_ids[mask]
                iv = iv[mask]
                uv = uv[mask]

                # Positive item embedding (respecting current fusion mode)
                pos_v = self._fused_get_item_emb(pos_ids)  # (B', D)

                # Cosine distances
                d_pos_H = self._cos_dist(pos_v, uv)                       # (B',)
                d_pos_h = self._cos_dist(pos_v[:, None, :], iv)           # (B', K)
                d_pos_h_best = d_pos_h.min(dim=-1).values                 # (B',)

                # BPR: d(pos, H) < d(pos, h_best) + margin
                tic_loss = F.softplus(d_pos_H - d_pos_h_best + self._tic_tau).mean()
                loss = loss + self._tic_weight * tic_loss
                out_dict["loss_tic"] = tic_loss.detach()
            else:
                tic_loss = torch.zeros([], device=loss.device)

        # MVTC: Multi-View Target Consistency (CF-teacher KL -> intent)
        mvtc_loss = None
        if self._use_mvtc and "mvtc_interest_vectors" in out_dict:
            iv = out_dict["mvtc_interest_vectors"]       # (B, K, D)
            pred_intent = out_dict["mvtc_pred_intent"]    # (B, K)
            pos_ids = out_dict["mvtc_pos_ids"]             # (B,)
            mask = (pos_ids != 0)

            if mask.any():
                pos_ids = pos_ids[mask]
                iv = iv[mask]
                pred_intent = pred_intent[mask]

                # Student: intent distribution q = softmax(pred_intent)
                log_q = F.log_softmax(pred_intent, dim=-1)               # (B', K)

                # Teacher: p_cf = softmax(cos(interest, pos) / tau)
                pos_v = self._fused_get_item_emb(pos_ids)                # (B', D)
                cos_sim = self._cos_sim(iv, pos_v[:, None, :])           # (B', K)
                p_teacher = F.softmax(cos_sim.detach() / self._mvtc_tau_cf, dim=-1)  # stopgrad

                mvtc_loss = F.kl_div(log_q, p_teacher, reduction="batchmean")
                loss = loss + self._mvtc_weight * mvtc_loss
                out_dict["loss_mvtc"] = mvtc_loss.detach()
            else:
                mvtc_loss = torch.zeros([], device=loss.device)

        # Periodic logging
        if self._step % 200 == 0:
            parts = [f"rec={rec_loss.item():.4f}"]
            if self._use_llm_align and "align_loss" in out_dict:
                parts.append(f"align={out_dict['align_loss'].item():.4f}")
            if tic_loss is not None:
                parts.append(f"tic={tic_loss.item():.6f}")
            if mvtc_loss is not None:
                parts.append(f"mvtc={mvtc_loss.item():.6f}")
            parts.append(f"total={loss.item():.4f}")
            logging.info(f"[PoMRecLLMEmb step {self._step}] " + " ".join(parts))

        return loss
