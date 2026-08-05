# -*- coding: UTF-8 -*-
"""
Utility functions for LLMMIRec.

- load_llm_table: safe LLM embedding loader with strict shape validation
- check_nan_inf: NaN/Inf diagnostic that raises RuntimeError
- get_activation: returns nn.Module for a named activation
"""

import logging
import pickle

import numpy as np
import torch
import torch.nn as nn


# =========================
#  LLM embedding loader
# =========================

def load_llm_table(path: str, expected_rows: int) -> torch.Tensor:
    """Load LLM embedding table from .pkl with strict shape validation.

    Args:
        path: path to .pkl file
        expected_rows: expected number of rows (corpus.n_items), INCLUDING row 0 padding

    Returns:
        torch.Tensor of shape (expected_rows, D), dtype=float32

    Raises:
        FileNotFoundError: if path does not exist
        ValueError: if array is not 2D, or shape[0] does not match expected_rows
            after auto-padding
    """
    if not path:
        raise ValueError("llm_emb_path is empty")

    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"LLM embedding file not found: {path}")

    arr = pickle.load(open(path, "rb"))
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(
            f"LLM embedding must be 2D, got shape {arr.shape} from {path}"
        )

    logging.info(f"[LLMMIRec] Loading LLM table: {path}")
    logging.info(f"[LLMMIRec]   original shape: {arr.shape}, dtype: {arr.dtype}")

    # Auto-pad row 0 if needed
    if arr.shape[0] == expected_rows:
        # Already has padding row
        logging.info(f"[LLMMIRec]   shape[0]=={expected_rows} (== n_items), row0 exists")
        table = arr
    elif arr.shape[0] == expected_rows - 1:
        # Missing row 0 — prepend zeros
        logging.info(
            f"[LLMMIRec]   shape[0]=={expected_rows - 1} (== n_items - 1), "
            f"prepending row 0"
        )
        table = np.vstack([np.zeros((1, arr.shape[1]), dtype=np.float32), arr])
    else:
        # Shape mismatch that cannot be resolved
        raise ValueError(
            f"[LLMMIRec] LLM table shape mismatch: got {arr.shape}, "
            f"expected ({expected_rows}, D) or ({expected_rows - 1}, D). "
            f"Corpus n_items = {expected_rows}. "
            f"Auto-truncation/padding is NOT allowed for safety."
        )

    tensor = torch.tensor(table, dtype=torch.float32)

    # Statistics
    row0_norm = float(torch.norm(tensor[0]).item())
    other_norms = torch.norm(tensor[1:], dim=1)
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()

    logging.info(
        f"[LLMMIRec]   final shape: {tensor.shape}, dtype: {tensor.dtype}"
    )
    logging.info(
        f"[LLMMIRec]   row0 norm: {row0_norm:.6f}"
    )
    logging.info(
        f"[LLMMIRec]   other rows norm: "
        f"min={float(other_norms.min()):.4f} "
        f"mean={float(other_norms.mean()):.4f} "
        f"max={float(other_norms.max()):.4f} "
        f"std={float(other_norms.std()):.4f}"
    )
    logging.info(
        f"[LLMMIRec]   NaN: {nan_count}, Inf: {inf_count}"
    )

    if nan_count > 0 or inf_count > 0:
        raise ValueError(
            f"[LLMMIRec] LLM table contains NaN={nan_count} or Inf={inf_count}!"
        )

    return tensor


# =========================
#  NaN / Inf checker
# =========================

def check_nan_inf(tensor: torch.Tensor, name: str) -> None:
    """Check tensor for NaN or Inf and raise RuntimeError if found.

    Args:
        tensor: tensor to check
        name: human-readable name for error messages

    Raises:
        RuntimeError: if NaN or Inf is detected
    """
    nan_mask = torch.isnan(tensor)
    inf_mask = torch.isinf(tensor)

    has_nan = nan_mask.any().item()
    has_inf = inf_mask.any().item()

    if has_nan or has_inf:
        nan_count = nan_mask.sum().item()
        inf_count = inf_mask.sum().item()
        raise RuntimeError(
            f"[LLMMIRec] NaN/Inf detected in '{name}'! "
            f"shape={tuple(tensor.shape)} "
            f"NaN={nan_count}, Inf={inf_count} "
            f"mean={float(tensor[~nan_mask & ~inf_mask].mean()) if (~nan_mask & ~inf_mask).any() else 'N/A'}"
        )


# =========================
#  Activation helper
# =========================

def get_activation(name: str) -> nn.Module:
    """Return an activation module by name.

    Args:
        name: 'gelu' or 'relu' (case-insensitive)

    Returns:
        nn.Module activation

    Raises:
        ValueError: for unknown activation names
    """
    name = name.lower().strip()
    if name == "gelu":
        return nn.GELU()
    elif name == "relu":
        return nn.ReLU()
    else:
        raise ValueError(
            f"Unknown activation: '{name}'. Supported: gelu, relu"
        )
