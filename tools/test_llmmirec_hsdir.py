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


def test_support_confidence_calibration():
    """Support-confidence calibration: numerical safety and invariants."""
    B, L, K = 2, 5, 4
    base_w = F.softmax(torch.randn(B, K), dim=-1)
    R = F.softmax(torch.randn(B, L, K), dim=-1)
    # Simulate padding
    valid = torch.tensor([[1,1,1,0,0],[1,1,1,1,0]], dtype=torch.float32).unsqueeze(-1)
    R = R * valid
    history_ids = torch.tensor([[1,2,3,0,0],[1,2,3,4,0]], dtype=torch.long)

    # Support
    support = R.sum(dim=1)
    support = support / support.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    assert torch.allclose(support.sum(dim=-1), torch.ones(B), atol=1e-4)

    # Confidence
    eps = 1e-8
    ent = -(R * torch.log(R + eps)).sum(dim=-1)
    norm_ent = (ent / torch.log(torch.tensor(K, dtype=torch.float32))) * valid.squeeze(-1)
    count = valid.squeeze(-1).sum(dim=-1).clamp(min=1)
    mean_ent = norm_ent.sum(dim=-1) / count
    conf = 1.0 - mean_ent
    assert (conf >= 0).all() and (conf <= 1.01).all(), f"confidence out of range: {conf}"

    # Calibrate
    beta = 1.0
    calibrated = base_w * (support + eps).pow(beta * conf.unsqueeze(-1))
    final_w = calibrated / calibrated.sum(dim=-1, keepdim=True).clamp_min(eps)
    assert torch.allclose(final_w.sum(dim=-1), torch.ones(B), atol=1e-4), f"final weights don't sum to 1: {final_w.sum(dim=-1)}"
    assert torch.isfinite(final_w).all(), f"final weights not finite"
    print(f"  base_w={base_w[0].tolist()}")
    print(f"  support={support[0].tolist()}")
    print(f"  conf={conf.tolist()}")
    print(f"  final_w={final_w[0].tolist()}")
    print("  support_confidence: OK")

    # Edge case: length 1
    R1 = F.softmax(torch.randn(1, 1, K), dim=-1)
    valid1 = torch.ones(1, 1, 1)
    R1 = R1 * valid1
    sup1 = R1.sum(dim=1); sup1 = sup1 / sup1.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    ent1 = -(R1 * torch.log(R1 + eps)).sum(dim=-1)
    nent1 = (ent1 / torch.log(torch.tensor(K, dtype=torch.float32))) * valid1.squeeze(-1)
    c1 = 1.0 - nent1.sum(dim=-1) / valid1.squeeze(-1).sum(dim=-1).clamp(min=1)
    assert torch.isfinite(c1).all(), f"L=1 confidence should be finite: {c1}"
    print("  length=1 safe: OK")


def test_pair_selective():
    """pair_selective HSR loss: padding exclusion, p≠n≠anchor, length<3 safe, gradient"""
    B, L, K = 2, 5, 4
    route_scores = torch.nn.Parameter(torch.randn(B, K, L), requires_grad=True)
    history_ids = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 3, 4, 0]], dtype=torch.long)
    # Fake teacher
    fine = F.softmax(torch.randn(B, L, 32), dim=-1)
    coarse = F.softmax(torch.randn(B, L, 8), dim=-1)
    Gf = fine @ fine.transpose(-1, -2); Gc = coarse @ coarse.transpose(-1, -2)

    temp, margin = 1.0, 0.1
    valid = (history_ids > 0).float()
    vp = valid.unsqueeze(-1) * valid.unsqueeze(-2)
    diag = torch.eye(L, dtype=torch.bool).unsqueeze(0)
    vp = vp * (~diag).float()

    # Student
    R = F.softmax(route_scores.transpose(1, 2) / temp, dim=-1) * valid.unsqueeze(-1)
    G = (R @ R.transpose(-1, -2)).clamp(1e-6, 1 - 1e-6)

    valid_anchor = (vp.sum(dim=-1) > 0)  # [B, L] bool
    pos_scores = Gf.clone(); pos_scores[vp == 0] = float("-inf")
    p_idx = pos_scores.argmax(dim=-1)
    b_idx = torch.arange(B).unsqueeze(-1).expand(-1, L)
    l_idx = torch.arange(L).unsqueeze(0)
    neg_scores = Gc.clone(); neg_scores[vp == 0] = float("inf")
    neg_scores[b_idx, l_idx, p_idx] = float("inf")
    n_idx = neg_scores.argmin(dim=-1)
    pos_ok = vp[b_idx, l_idx, p_idx].bool()
    neg_ok = vp[b_idx, l_idx, n_idx].bool()
    ok = valid_anchor & pos_ok & neg_ok & (p_idx != n_idx)

    # Verify padding never selected
    for b in range(B):
        for i in range(L):
            if ok[b, i]:
                assert p_idx[b, i].item() != 0 or history_ids[b, p_idx[b, i]].item() != 0, "padding selected as positive"
                assert n_idx[b, i].item() != 0 or history_ids[b, n_idx[b, i]].item() != 0, "padding selected as negative"
    print("  padding exclusion: OK")

    # Verify length<3 safe
    history_short = torch.tensor([[1, 2, 0, 0, 0]], dtype=torch.long)
    v_short = (history_short > 0).float(); vp_s = v_short.unsqueeze(-1) * v_short.unsqueeze(-2)
    vp_s = vp_s * (~diag[:, :5, :5]).float()
    assert vp_s[:, 0, 1].item() == 1 and vp_s[:, 1, 0].item() == 1, "pair (0,1) should be valid"
    assert vp_s.sum(dim=-1)[0, 0].item() >= 1, "anchor 0 should have a valid pair"
    print("  length<3 safe: OK")

    # Loss finite
    c_i = Gf[b_idx, l_idx, p_idx] * (1.0 - Gc[b_idx, l_idx, n_idx])
    c_i = c_i.detach()
    r_pos = G[b_idx, l_idx, p_idx]; r_neg = G[b_idx, l_idx, n_idx]
    loss_per = c_i * F.relu(margin - r_pos + r_neg)
    total_c = (c_i * ok.float()).sum().clamp(min=1e-8)
    L = (loss_per * ok.float()).sum() / total_c
    assert torch.isfinite(L).all(), f"Loss not finite: {L}"
    print(f"  pair_selective loss={L.item():.4f}: OK")

    # Gradient: G_route must have grad
    L.backward()
    assert route_scores.grad is not None and torch.isfinite(route_scores.grad).all(), "Gradient should be finite"
    print("  G_route gradient: OK")

    # Teacher confidence has no gradient (c_i is detached)
    assert not c_i.requires_grad, "Teacher confidence should be detached"
    print("  teacher no grad: OK")


def test_attention_contribution():
    """attention_contribution: C from A, valid sums to 1, padding=0, gradient from HSR back to extractor."""
    B, K, L = 2, 4, 5
    # Simulate attention_maps from extractor (with gradients)
    A = torch.nn.Parameter(torch.randn(B, K, L), requires_grad=True)
    # Apply softmax (as extractor does)
    am = F.softmax(A, dim=-1)  # [B, K, L]
    history_ids = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]], dtype=torch.long)
    valid = (history_ids > 0).float().unsqueeze(-1)  # [B, L, 1]

    # C construction
    C = am.transpose(1, 2)  # [B, L, K]
    C = C * valid
    C = C / C.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    assert C.shape == (B, L, K), f"Shape: {C.shape}"
    # Valid items sum to 1
    rsum = C[0, :3, :].sum(dim=-1)
    assert torch.allclose(rsum, torch.ones(3), atol=1e-4), f"Valid row sum: {rsum}"
    # Padding is 0
    assert (C[0, 3:, :].abs().max().item() == 0), "Padding C should be 0"
    print("  C shape + valid sum + padding=0: OK")

    # G_student finite
    G = (C @ C.transpose(-1, -2)).clamp(1e-6, 1 - 1e-6)
    assert torch.isfinite(G).all(), "G_student not finite"
    print("  G_student finite: OK")

    # Gradient from HSR loss back to A
    Gf = F.softmax(torch.randn(B, L, 32), dim=-1) @ F.softmax(torch.randn(B, L, 32), dim=-1).transpose(-1, -2)
    Gc = F.softmax(torch.randn(B, L, 8), dim=-1) @ F.softmax(torch.randn(B, L, 8), dim=-1).transpose(-1, -2)
    vp = valid.squeeze(-1).unsqueeze(-1) * valid.squeeze(-1).unsqueeze(-2)
    diag = torch.eye(L, dtype=torch.bool).unsqueeze(0)
    vp = vp * (~diag).float()
    W_pos, W_neg = Gf, 1.0 - Gc
    pos_sum = (vp * W_pos).sum().clamp(min=1e-8)
    L_coh = -(vp * W_pos * torch.log(G)).sum() / pos_sum
    neg_sum = (vp * W_neg).sum().clamp(min=1e-8)
    L_sep = -(vp * W_neg * torch.log(1.0 - G)).sum() / neg_sum
    L = L_coh + L_sep
    assert torch.isfinite(L), f"Loss not finite: {L}"
    L.backward()
    assert A.grad is not None and torch.isfinite(A.grad).all(), "Gradient should flow back to A"
    print(f"  loss={L.item():.4f}, grad finite: OK")


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
    test_support_confidence_calibration()
    test_pair_selective()
    test_attention_contribution()
    test_zero_confidence()
    print("ALL TESTS PASSED")
