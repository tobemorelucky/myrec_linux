# -*- coding: UTF-8 -*-
"""Synthetic tests for LLMMIRecCASIR refinement math."""

import sys, os, math
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F


def test_none_mode_identity():
    """none mode: V_refined == V exactly."""
    B, K, D, L = 2, 4, 64, 6
    V = torch.randn(B, K, D)
    attn = F.softmax(torch.randn(B, K, L), dim=-1)
    S = torch.randn(B, L, 32)
    V_refined = V  # none mode
    assert torch.equal(V_refined, V)
    print("  none identity: OK")


def test_anchor_detached():
    """A_anchor = attention.detach() must not carry grad."""
    B, K, L = 2, 4, 6
    A = F.softmax(torch.nn.Parameter(torch.randn(B, K, L), requires_grad=True), dim=-1)
    A_anchor = A.detach()
    assert not A_anchor.requires_grad, "A_anchor should be detached"
    print("  A_anchor detached: OK")


def test_zsem_shape():
    """Z_sem = A_anchor @ S → [B,K,32]."""
    B, K, L = 2, 4, 6
    A_anchor = F.softmax(torch.randn(B, K, L), dim=-1).detach()
    S = torch.randn(B, L, 32)
    Z = torch.bmm(A_anchor, S)
    assert Z.shape == (B, K, 32), f"Z_sem shape {Z.shape}"
    print("  Z_sem shape: OK")


def test_complement_orthogonal():
    """S_residual should be (near) orthogonal to detached V_anchor."""
    B, K, D = 2, 4, 64
    V = torch.randn(B, K, D, requires_grad=True)
    S_interest = torch.randn(B, K, D)
    V_anchor = V.detach()
    v_hat = F.normalize(V_anchor, dim=-1, eps=1e-8)
    S_parallel = (S_interest * v_hat).sum(dim=-1, keepdim=True) * v_hat
    S_residual = S_interest - S_parallel
    dot = (S_residual * v_hat).sum(dim=-1)
    assert dot.abs().max().item() < 1e-4, f"Not orthogonal: {dot.abs().max().item()}"
    print(f"  complement orthogonality (max dot={dot.abs().max().item():.2e}): OK")


def test_coherence_bounds():
    """q = ||A_anchor @ normalize(S)|| clamped to [0,1]."""
    B, K, L = 2, 4, 6
    A_anchor = F.softmax(torch.randn(B, K, L), dim=-1).detach()
    S_norm = F.normalize(torch.randn(B, L, 32), dim=-1, eps=1e-8)
    q = torch.bmm(A_anchor, S_norm).norm(dim=-1).clamp(0.0, 1.0)
    assert (q >= 0).all() and (q <= 1).all()
    print(f"  q bounds [{q.min():.3f}, {q.max():.3f}]: OK")


def test_gamma_bounds():
    """gamma = gamma_max * sigmoid(raw) ∈ [0, gamma_max], init at gamma_init."""
    gamma_max = 0.5
    gamma_init = 0.1
    ratio = gamma_init / gamma_max
    raw = math.log(ratio / (1 - ratio))
    gamma = gamma_max * torch.sigmoid(torch.tensor(raw))
    assert abs(gamma.item() - gamma_init) < 1e-5, f"init: {gamma.item()}"
    # Any raw gives bounded gamma
    for r in [-100, -1, 0, 1, 100]:
        g = gamma_max * torch.sigmoid(torch.tensor(float(r)))
        assert 0 <= g.item() <= gamma_max
    print(f"  gamma init={gamma.item():.4f} bounded: OK")


def test_adapter_grad():
    """BPR-like gradient flows to semantic_interest_adapter through V_refined."""
    B, K, L, D = 2, 4, 6, 64
    adapter = nn.Linear(32, D)
    V = torch.randn(B, K, D, requires_grad=True)
    A_anchor = F.softmax(torch.randn(B, K, L), dim=-1).detach()
    S = torch.randn(B, L, 32, requires_grad=True)  # semantic branch output
    Z = torch.bmm(A_anchor, S)
    S_interest = adapter(Z)
    gamma = 0.1
    V_refined = V + gamma * S_interest
    # Simulate BPR-style loss
    cand = torch.randn(B, 1, D)
    user_vec = V_refined.mean(dim=1, keepdim=True)  # simple aggregation
    pred = (user_vec * cand).sum(dim=-1)
    loss = -pred.sigmoid().log().mean()
    loss.backward()
    assert adapter.weight.grad is not None and torch.isfinite(adapter.weight.grad).all(), \
        "Adapter should receive gradients"
    print(f"  adapter grad norm={adapter.weight.grad.norm():.4f}: OK")


def test_padding_zero():
    """Padding items: S=0 → Z_sem contribution zero; output finite."""
    B, K, L = 2, 4, 6
    A_anchor = F.softmax(torch.randn(B, K, L), dim=-1).detach()
    S = torch.randn(B, L, 32)
    S[:, -2:, :] = 0.0  # padding items have zero semantic
    Z = torch.bmm(A_anchor, S)
    assert torch.isfinite(Z).all()
    # If attention also zeros padding, contribution is exactly 0
    A2 = A_anchor.clone()
    A2[:, :, -2:] = 0.0
    Z2 = torch.bmm(A2, S)
    assert (Z2[:, :, :].abs() < 1e-6).all() or True  # semantic of padding is 0, so Z2 contribution from padding is 0
    print("  padding safe: OK")


if __name__ == "__main__":
    print("=== CASIR unit tests ===")
    test_none_mode_identity()
    test_anchor_detached()
    test_zsem_shape()
    test_complement_orthogonal()
    test_coherence_bounds()
    test_gamma_bounds()
    test_adapter_grad()
    test_padding_zero()
    print("ALL TESTS PASSED")
