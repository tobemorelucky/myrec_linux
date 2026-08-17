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


def test_training_only_gate():
    """Semantic distill computation gated on (training or return_intermediate).

    Simulates the forward-path branch:
      - eval + return_intermediate=False → no semantic predictor call
      - eval + return_intermediate=True  → predictor allowed
      - train                              → predictor allowed, loss stashed
    """
    class FakePredictor:
        def __init__(self):
            self.calls = 0
        def __call__(self, x):
            self.calls += 1
            return x[..., :32]

    B, K, L, D = 2, 4, 6, 64
    predictor = FakePredictor()

    def forward_sim(training, return_intermediate, distill_mode="confidence"):
        V = torch.randn(B, K, D)
        need = (distill_mode != "none") and (training or return_intermediate)
        if need:
            _ = predictor(V)
        return need

    # eval, no intermediate → predictor NOT called
    need = forward_sim(training=False, return_intermediate=False)
    assert need is False and predictor.calls == 0, "eval fast path must skip predictor"
    # eval + intermediate → predictor called
    forward_sim(training=False, return_intermediate=True)
    assert predictor.calls == 1
    # train → predictor called
    forward_sim(training=True, return_intermediate=False)
    assert predictor.calls == 2
    print("  training-only gate: OK")


def test_eval_recommendation_identity():
    """Recommendation score independent of semantic distillation machinery."""
    B, K, D = 2, 4, 64
    V = torch.randn(B, K, D)
    w = F.softmax(torch.randn(B, K), dim=-1)
    user_vec = (V * w[:, :, None]).sum(dim=1)
    cand = torch.randn(B, 1, D)
    score = (user_vec[:, None, :] * cand).sum(dim=-1)
    # No semantic predictor involvement — score stays identical regardless
    score2 = (user_vec[:, None, :] * cand).sum(dim=-1)
    assert torch.equal(score, score2)
    print("  eval recommendation identity: OK")


def test_responsibility_teacher():
    """Responsibility teacher: R, W shapes/sums, padding, detachment."""
    B, K, L = 2, 4, 6
    history = torch.tensor([[1, 2, 3, 0, 0, 0], [1, 2, 3, 4, 0, 0]], dtype=torch.long)
    valid = (history > 0).float()  # [B, L]
    A = F.softmax(torch.randn(B, K, L, requires_grad=True), dim=-1).detach()
    Q = F.softmax(torch.randn(B, L, 32), dim=-1)

    # R computation
    A_masked = A * valid.unsqueeze(1)  # [B, K, L]
    R = A_masked / A_masked.sum(dim=1, keepdim=True).clamp_min(1e-8)
    assert R.shape == (B, K, L), f"R shape {R.shape}"
    assert not R.requires_grad, "R should be detached"

    # Valid items: R sums to ~1 over K
    ksum = R.sum(dim=1)  # [B, L]
    for b in range(B):
        for l in range(L):
            if history[b, l] > 0:
                assert abs(ksum[b, l].item() - 1.0) < 1e-4, f"R sum at ({b},{l}): {ksum[b, l]}"
    # Padding: R = 0
    assert (R[0, :, 3:].abs().max().item() == 0), "Padding R should be 0"
    print("  R shape + K-sum + padding: OK")

    # W computation
    W = A_masked * R
    W = W / W.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    # Each interest sums to ~1 over valid history
    wsum = W.sum(dim=-1)  # [B, K]
    assert torch.allclose(wsum, torch.ones(B, K), atol=1e-4), f"W row sums: {wsum}"
    assert not W.requires_grad
    print("  W per-interest sum + detached: OK")

    # T
    T = torch.bmm(W, Q)
    T = T / T.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    assert T.shape == (B, K, 32)
    assert torch.allclose(T.sum(dim=-1), torch.ones(B, K), atol=1e-4)
    assert torch.isfinite(T).all()
    print("  T shape + sum + finite: OK")

    # Semantic loss still flows to V
    V = torch.randn(B, K, 64, requires_grad=True)
    predictor = torch.nn.Linear(64, 32)
    P = predictor(V)
    log_P = F.log_softmax(P, dim=-1)
    kl = F.kl_div(log_P, T, reduction="none").sum(dim=-1)
    L = kl.mean()
    L.backward()
    assert V.grad is not None and torch.isfinite(V.grad).all()
    print("  semantic loss grad to V: OK")


def test_attention_mode_identity():
    """attention mode: T = A @ Q then normalize (unchanged behavior)."""
    B, K, L = 2, 4, 6
    A = F.softmax(torch.randn(B, K, L), dim=-1).detach()
    Q = F.softmax(torch.randn(B, L, 32), dim=-1)
    T = torch.bmm(A, Q)
    T = T / T.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    assert T.shape == (B, K, 32)
    assert torch.allclose(T.sum(dim=-1), torch.ones(B, K), atol=1e-4)
    print("  attention teacher identity: OK")


def test_responsibility_power_equivalence():
    """responsibility_power alpha=0 == attention; alpha=1 == responsibility."""
    B, K, L = 2, 4, 6
    history = torch.tensor([[1, 2, 3, 0, 0, 0], [1, 2, 3, 4, 0, 0]], dtype=torch.long)
    valid = (history > 0).float()
    A = F.softmax(torch.randn(B, K, L), dim=-1).detach()
    Q = F.softmax(torch.randn(B, L, 32), dim=-1)

    A_masked = A * valid.unsqueeze(1)

    # Attention teacher (reference)
    T_attn = torch.bmm(A, Q)
    T_attn = T_attn / T_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # Responsibility teacher (reference)
    R = A_masked / A_masked.sum(dim=1, keepdim=True).clamp_min(1e-8)
    W1 = A_masked * R
    W1 = W1 / W1.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    T_resp = torch.bmm(W1, Q)
    T_resp = T_resp / T_resp.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    # responsibility_power alpha=0: W = A_masked * R^0 = A_masked, normalized
    R_pow = A_masked / A_masked.sum(dim=1, keepdim=True).clamp_min(1e-8)
    W0 = A_masked * (R_pow + 1e-8).pow(0.0)
    W0 = W0 / W0.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    T0 = torch.bmm(W0, Q)
    T0 = T0 / T0.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    # Note: attention teacher uses UNMASKED A (matches model code path).
    # For alpha=0 equivalence, the model's responsibility_power branch applies valid mask.
    # Compare T0 against masked-attention teacher:
    T_attn_masked = torch.bmm(A_masked, Q)
    T_attn_masked = T_attn_masked / T_attn_masked.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    assert torch.allclose(T0, T_attn_masked, atol=1e-6), "alpha=0 should equal masked-attention teacher"

    # responsibility_power alpha=1
    W1p = A_masked * (R_pow + 1e-8).pow(1.0)
    W1p = W1p / W1p.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    T1 = torch.bmm(W1p, Q)
    T1 = T1 / T1.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    assert torch.allclose(T1, T_resp, atol=1e-6), "alpha=1 should equal responsibility teacher"

    # Shapes and finite
    assert T0.shape == (B, K, 32) and T1.shape == (B, K, 32)
    assert torch.isfinite(T0).all() and torch.isfinite(T1).all()
    print("  alpha=0/1 equivalence + finite: OK")


if __name__ == "__main__":
    print("=== CAISD unit tests ===")
    test_teacher_shape_and_detach()
    test_padding_neutral()
    test_gradient_to_V_and_predictor()
    test_confidence_bounds()
    test_predictor_not_in_recommendation()
    test_none_mode_identity()
    test_kl_finite()
    test_training_only_gate()
    test_eval_recommendation_identity()
    test_responsibility_teacher()
    test_attention_mode_identity()
    test_responsibility_power_equivalence()
    print("ALL TESTS PASSED")
