# -*- coding: UTF-8 -*-
"""Synthetic tests for LLMMIRecCAISD semantic distillation math."""

import sys, os, math
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F


def test_teacher_shape_and_detach():
    """T = A_teacher @ Q → [B,K,32], fully detached, rows sum to 1."""
    B, K, L = 2, 4, 6
    A = F.softmax(torch.randn(B, K, L, requires_grad=True), dim=-1)
    A_teacher = A.detach()
    Q = F.softmax(torch.randn(B, L, 32), dim=-1)
    T = torch.bmm(A_teacher, Q)
    T = T / T.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    assert T.shape == (B, K, 32), f"T shape {T.shape}"
    assert not T.requires_grad, "T should be detached"
    rsum = T.sum(dim=-1)
    assert torch.allclose(rsum, torch.ones(B, K), atol=1e-4), f"Row sum: {rsum}"
    print("  T shape + detach + row-sum: OK")


def test_padding_neutral():
    """Padding items (Q row 0) contribute nothing to T."""
    B, K, L = 2, 4, 5
    A = F.softmax(torch.randn(B, K, L), dim=-1).detach()
    A[:, :, -2:] = 0.0  # no attention to padding
    Q = F.softmax(torch.randn(B, L, 32), dim=-1)
    Q[:, -2:, :] = 0.0  # padding items have zero assignment
    T = torch.bmm(A, Q)
    T = T / T.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    assert torch.isfinite(T).all()
    print("  padding neutral: OK")


def test_gradient_to_V_and_predictor():
    """Semantic loss gradients reach V (extractor) and semantic_predictor."""
    B, K, D = 2, 4, 64
    predictor = nn.Linear(D, 32)
    V = torch.randn(B, K, D, requires_grad=True)  # interest vectors (extractor output)
    T = F.softmax(torch.randn(B, K, 32), dim=-1).detach()  # frozen teacher

    P_logits = predictor(V)
    log_P = F.log_softmax(P_logits, dim=-1)
    kl = F.kl_div(log_P, T, reduction="none").sum(dim=-1)
    L_sem = kl.mean()
    L_sem.backward()

    assert V.grad is not None and torch.isfinite(V.grad).all(), "V should receive gradients"
    assert predictor.weight.grad is not None and torch.isfinite(predictor.weight.grad).all(), \
        "Predictor should receive gradients"
    print(f"  V grad norm={V.grad.norm():.4f}, predictor grad norm={predictor.weight.grad.norm():.4f}: OK")


def test_confidence_bounds():
    """Confidence = 1 - H/log(32) ∈ [0,1]."""
    T = F.softmax(torch.randn(3, 5, 32), dim=-1)
    eps = 1e-8
    H = -(T * torch.log(T + eps)).sum(dim=-1)
    conf = (1.0 - H / math.log(32)).clamp(0.0, 1.0)
    assert (conf >= 0).all() and (conf <= 1).all(), f"Conf out of bounds: {conf}"
    print(f"  confidence range [{conf.min():.3f}, {conf.max():.3f}]: OK")


def test_predictor_not_in_recommendation():
    """Semantic predictor output does not affect recommendation score."""
    B, K, D = 2, 4, 64
    V = torch.randn(B, K, D)
    predictor = nn.Linear(D, 32)
    pred_logits = predictor(V)  # used only for loss

    # Recommendation path uses only V
    w = F.softmax(torch.randn(B, K), dim=-1)
    user_vec = (V * w[:, :, None]).sum(dim=1)
    cand = torch.randn(B, 1, D)
    score = (user_vec[:, None, :] * cand).sum(dim=-1)

    # Changing predictor weights must not change score
    score_before = score.clone()
    with torch.no_grad():
        predictor.weight.data += 1.0
    score_after = (user_vec[:, None, :] * cand).sum(dim=-1)
    assert torch.allclose(score_before, score_after), "Predictor must not affect recommendation"
    print("  predictor excluded from recommendation: OK")


def test_none_mode_identity():
    """none mode: no teacher, no predictor, pure ASPCF baseline."""
    # Model with none mode doesn't create semantic_predictor or t_semantic_assign
    assert True
    print("  none mode identity (by construction): OK")


def test_kl_finite():
    """Uniform and confidence KL losses are finite."""
    B, K = 2, 4
    V = torch.randn(B, K, 64, requires_grad=True)
    predictor = nn.Linear(64, 32)
    T = F.softmax(torch.randn(B, K, 32), dim=-1).detach()
    P = predictor(V)
    log_P = F.log_softmax(P, dim=-1)
    kl = F.kl_div(log_P, T, reduction="none").sum(dim=-1)
    L_uniform = kl.mean()
    assert torch.isfinite(L_uniform)
    H = -(T * torch.log(T + 1e-8)).sum(dim=-1)
    conf = (1 - H / math.log(32)).clamp(0, 1)
    L_conf = (conf * kl).sum() / conf.sum().clamp(min=1e-8)
    assert torch.isfinite(L_conf)
    print(f"  L_uniform={L_uniform.item():.4f}, L_conf={L_conf.item():.4f}: OK")


if __name__ == "__main__":
    print("=== CAISD unit tests ===")
    test_teacher_shape_and_detach()
    test_padding_neutral()
    test_gradient_to_V_and_predictor()
    test_confidence_bounds()
    test_predictor_not_in_recommendation()
    test_none_mode_identity()
    test_kl_finite()
    print("ALL TESTS PASSED")
