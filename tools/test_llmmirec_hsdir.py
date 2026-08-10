# -*- coding: UTF-8 -*-
"""CPU synthetic tensor unit tests for LLMMIRecHSDIR."""

import sys, os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn.functional as F


def test_route_membership():
    """Student route membership: raw_scores → softmax over K → R [B,L,K]."""
    B, K, L = 2, 4, 6
    scores = torch.randn(B, K, L)
    scores[:, :, 4:] = -10.0  # simulate masked positions
    temp = 1.0
    # transpose to [B,L,K] then softmax over K
    R = F.softmax(scores.transpose(1, 2) / temp, dim=-1)
    assert R.shape == (B, L, K), f"Expected {(B,L,K)}, got {R.shape}"
    # Each history position sums to ~1
    row_sum = R.sum(dim=-1)
    assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-5), f"Row sum: {row_sum}"
    print("  route_membership: OK")


def test_comembership():
    """G_route = R @ R^T is permutation-invariant."""
    B, L, K = 2, 6, 4
    R = F.softmax(torch.randn(B, L, K), dim=-1)
    G = R @ R.transpose(-1, -2)
    assert G.shape == (B, L, L)
    # Symmetric
    assert torch.allclose(G, G.transpose(-1, -2), atol=1e-5)
    # Values in [0,1]
    assert G.min() >= 0 and G.max() <= 1.01
    # Permutation invariance: shuffling K dim should give same G
    perm = torch.randperm(K)
    R_perm = R[:, :, perm]
    G_perm = R_perm @ R_perm.transpose(-1, -2)
    assert torch.allclose(G, G_perm, atol=1e-5), "G_route should be permutation-invariant"
    print("  comembership + permutation invariance: OK")


def test_padding_exclusion():
    """Padding positions (item_id=0) should be excluded."""
    B, L, K = 2, 6, 4
    history_ids = torch.tensor([[1, 2, 3, 0, 0, 0], [1, 2, 0, 0, 0, 0]])
    valid = (history_ids > 0).float()  # [B,L]
    valid_pair = valid.unsqueeze(-1) * valid.unsqueeze(-2)  # [B,L,L]
    diag = torch.eye(L, dtype=torch.bool).unsqueeze(0)
    valid_pair = valid_pair * (~diag).float()

    # Pair (0,3) should be invalid for both rows
    assert valid_pair[0, 0, 3].item() == 0, "cross padding/valid should be 0"
    assert valid_pair[0, 3, 0].item() == 0
    # Pair (0,1) should be valid
    assert valid_pair[0, 0, 1].item() == 1, "two valid items should be 1"
    assert valid_pair[1, 0, 1].item() == 1
    # Diagonal (0,0) should be 0
    assert valid_pair[0, 0, 0].item() == 0, "diagonal should be excluded"
    print("  padding + diagonal exclusion: OK")


def test_loss_finite():
    """HSDIR loss must be finite."""
    B, K, L = 2, 4, 8
    scores = torch.randn(B, K, L)
    temp = 1.0
    R = F.softmax(scores.transpose(1, 2) / temp, dim=-1)
    G_route = (R @ R.transpose(-1, -2)).clamp(1e-6, 1 - 1e-6)

    # Fake teacher
    fine = F.softmax(torch.randn(B, L, 32), dim=-1)
    coarse = F.softmax(torch.randn(B, L, 8), dim=-1)
    G_fine = fine @ fine.transpose(-1, -2)
    G_coarse = coarse @ coarse.transpose(-1, -2)

    history_ids = torch.ones(B, L, dtype=torch.long)
    history_ids[:, -2:] = 0  # some padding

    valid = (history_ids > 0).float()
    valid_pair = valid.unsqueeze(-1) * valid.unsqueeze(-2)
    diag = torch.eye(L, dtype=torch.bool).unsqueeze(0)
    valid_pair = valid_pair * (~diag).float()

    W_pos = G_fine
    W_neg = 1.0 - G_coarse

    pos_sum = (valid_pair * W_pos).sum().clamp(min=1e-8)
    L_coh = -(valid_pair * W_pos * torch.log(G_route)).sum() / pos_sum

    neg_sum = (valid_pair * W_neg).sum().clamp(min=1e-8)
    L_sep = -(valid_pair * W_neg * torch.log(1.0 - G_route)).sum() / neg_sum

    assert torch.isfinite(L_coh), f"L_coh not finite: {L_coh}"
    assert torch.isfinite(L_sep), f"L_sep not finite: {L_sep}"
    assert L_coh.item() >= 0, f"L_coh should be >= 0, got {L_coh}"
    assert L_sep.item() >= 0, f"L_sep should be >= 0, got {L_sep}"
    print(f"  L_coh={L_coh:.4f}, L_sep={L_sep:.4f}: OK")


def test_teacher_no_grad():
    """Teacher assignments must be frozen (no grad, persistent=False compatible)."""
    t = torch.randn(10, 32, requires_grad=False)
    # Lookup from frozen tensor
    idx = torch.tensor([1, 2, 3])
    looked_up = t[idx]
    assert not looked_up.requires_grad, "Frozen teacher lookup should not require grad"
    # Verify original is unchanged
    orig = t[idx].clone()
    looked_up_add = looked_up + 1.0  # creates new tensor, doesn't modify t
    assert torch.equal(t[idx], orig), "Original teacher should be unchanged"
    print("  teacher no grad: OK")


def test_padding_nan_safety():
    """Padding positions must not produce NaN in HSDIR loss path."""
    B, K, L = 2, 4, 5
    # Simulate pre-mask raw scores (no -inf)
    scores = torch.randn(B, K, L)
    temp = 1.0
    # Simulate padded history
    history_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]], dtype=torch.long)

    # Step 1: route membership
    R = F.softmax(scores.transpose(1, 2) / temp, dim=-1)   # [B, L, K]
    valid = (history_ids > 0).float().unsqueeze(-1)
    R = R * valid  # zero out padding

    # Check padding positions are zero
    assert (R[0, 3:, :].abs().max().item() == 0), "Padding row 0 pos 3+ should be 0"
    assert (R[1, 2:, :].abs().max().item() == 0), "Padding row 1 pos 2+ should be 0"
    # Check non-padding positions sum to ~1
    row_sum = R[0, :3, :].sum(dim=-1)
    assert torch.allclose(row_sum, torch.ones(3), atol=1e-4), f"Valid row sums: {row_sum}"
    assert torch.isfinite(R).all(), "R should be finite"
    print("  padding R=0 + finite: OK")

    # Step 2: G_route
    G = (R @ R.transpose(-1, -2)).clamp(1e-6, 1 - 1e-6)
    assert torch.isfinite(G).all(), "G_route should be finite"
    print("  G_route finite: OK")

    # Step 3: Teacher confidences (fake)
    fine = F.softmax(torch.randn(B, L, 32), dim=-1) * valid
    coarse = F.softmax(torch.randn(B, L, 8), dim=-1) * valid
    G_fine = fine @ fine.transpose(-1, -2)
    G_coarse = coarse @ coarse.transpose(-1, -2)
    W_pos, W_neg = G_fine, 1.0 - G_coarse

    valid_pair = valid.squeeze(-1).unsqueeze(-1) * valid.squeeze(-1).unsqueeze(-2)
    diag = torch.eye(L, dtype=torch.bool).unsqueeze(0)
    valid_pair = valid_pair * (~diag).float()

    pos_sum = (valid_pair * W_pos).sum().clamp(min=1e-8)
    L_coh = -(valid_pair * W_pos * torch.log(G)).sum() / pos_sum

    neg_sum = (valid_pair * W_neg).sum().clamp(min=1e-8)
    L_sep = -(valid_pair * W_neg * torch.log(1.0 - G)).sum() / neg_sum

    assert torch.isfinite(L_coh), f"L_coh should be finite: {L_coh}"
    assert torch.isfinite(L_sep), f"L_sep should be finite: {L_sep}"
    print(f"  L_coh={L_coh:.4f}, L_sep={L_sep:.4f}: OK")

    # Step 4: backward with a learnable surrogate
    W = torch.nn.Parameter(torch.randn(B, K, L))
    scores2 = W  # use learnable param
    R2 = F.softmax(scores2.transpose(1, 2) / temp, dim=-1) * valid
    G2 = (R2 @ R2.transpose(-1, -2)).clamp(1e-6, 1 - 1e-6)
    L_test = -(valid_pair * W_pos * torch.log(G2)).sum() / pos_sum.clamp(min=1e-8)
    L_test.backward()
    assert W.grad is not None and torch.isfinite(W.grad).all(), "Gradient should be finite"
    print("  backward finite: OK")


def test_zero_confidence():
    """When valid pairs have zero confidence, loss should be zero (not NaN)."""
    B, L, K = 1, 4, 2
    R = F.softmax(torch.randn(B, L, K), dim=-1)
    G_route = (R @ R.transpose(-1, -2)).clamp(1e-6, 1 - 1e-6)

    # All-zero teacher confidences
    W_pos = torch.zeros(B, L, L)
    W_neg = torch.zeros(B, L, L)

    history_ids = torch.ones(B, L, dtype=torch.long)
    valid = (history_ids > 0).float()
    valid_pair = valid.unsqueeze(-1) * valid.unsqueeze(-2)
    diag = torch.eye(L, dtype=torch.bool).unsqueeze(0)
    valid_pair = valid_pair * (~diag).float()

    pos_sum = (valid_pair * W_pos).sum().clamp(min=1e-8)
    L_coh = -(valid_pair * W_pos * torch.log(G_route)).sum() / pos_sum

    neg_sum = (valid_pair * W_neg).sum().clamp(min=1e-8)
    L_sep = -(valid_pair * W_neg * torch.log(1.0 - G_route)).sum() / neg_sum

    assert torch.isfinite(L_coh), f"L_coh should be finite (0.0): {L_coh}"
    assert torch.isfinite(L_sep), f"L_sep should be finite (0.0): {L_sep}"
    print("  zero confidence: OK")


if __name__ == "__main__":
    print("=== HSDIR unit tests ===")
    test_route_membership()
    test_comembership()
    test_padding_exclusion()
    test_loss_finite()
    test_teacher_no_grad()
    test_padding_nan_safety()
    test_zero_confidence()
    print("ALL TESTS PASSED")
