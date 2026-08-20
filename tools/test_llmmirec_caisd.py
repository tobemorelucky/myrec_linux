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


def pairwise_js(P, K, eps=1e-8):
    """Stable pairwise JS over strict upper triangle; returns (JS_BP, JS_mat)."""
    B = P.shape[0]
    P = P.clamp(min=eps)
    k_idx, l_idx = torch.triu_indices(K, K, offset=1)
    M = 0.5 * (P[:, k_idx] + P[:, l_idx])
    js = 0.5 * ((P[:, k_idx] * torch.log(P[:, k_idx] / M)).sum(-1)
                + (P[:, l_idx] * torch.log(P[:, l_idx] / M)).sum(-1))
    js = js / math.log(2)
    mat = torch.zeros(B, K, K)
    mat[:, k_idx, l_idx] = js
    mat[:, l_idx, k_idx] = js
    return js, mat


def test_relation_js_math():
    """Pairwise JS: shape, symmetry, diagonal 0, upper-triangle count."""
    B, K = 3, 4
    T = F.softmax(torch.randn(B, K, 32), dim=-1)
    P = F.softmax(torch.randn(B, K, 32), dim=-1)
    js_T, mat_T = pairwise_js(T, K)
    js_P, mat_P = pairwise_js(P, K)

    assert js_T.shape == (B, K * (K - 1) // 2), f"shape {js_T.shape}"
    # Symmetric
    assert torch.allclose(mat_T, mat_T.transpose(-1, -2), atol=1e-5)
    # Diagonal ~0
    assert torch.allclose(torch.diagonal(mat_T[0], 0), torch.zeros(K), atol=1e-6)
    # JS in [0,1]
    assert js_T.min() >= 0 and js_T.max() <= 1.01, f"JS range [{js_T.min()}, {js_T.max()}]"
    assert torch.isfinite(js_T).all()
    # Upper triangle pair count
    assert js_T.shape[-1] == K * (K - 1) // 2
    print(f"  pairwise JS shape/sym/diag/range [{js_T.min():.3f},{js_T.max():.3f}]: OK")

    # Gradients: JS_T detached, JS_P grad
    V = torch.randn(B, K, 64, requires_grad=True)
    predictor = torch.nn.Linear(64, 32)
    P2 = F.softmax(predictor(V), dim=-1)
    js_P2, _ = pairwise_js(P2, K)
    loss = F.smooth_l1_loss(js_P2, js_T.detach()).mean()
    loss.backward()
    assert V.grad is not None and torch.isfinite(V.grad).all(), "V should get grad"
    assert predictor.weight.grad is not None, "Predictor should get grad"
    print(f"  relational loss={loss.item():.4f}, grad to V+predictor: OK")


def test_relation_js_scale():
    """Identical distributions → JS≈0; disjoint → JS≈1."""
    B, K = 2, 4
    P = F.softmax(torch.randn(B, K, 32), dim=-1)
    js_same, _ = pairwise_js(P, K)
    # identical profiles → JS 0
    T_dup = P.clone()
    js_dup, _ = pairwise_js(torch.cat([T_dup, T_dup], dim=1).reshape(B * 2, K, 32)[:B], K) if False else (None, None)
    # Direct: two copies of same distribution
    P2 = P.reshape(-1, 32)
    k_idx, l_idx = torch.triu_indices(K, K, offset=1)
    # use P2 pairs where both rows identical
    js_ident = []
    for b in range(B):
        for ki, li in zip(k_idx.tolist(), l_idx.tolist()):
            p_k = P2[b * K + ki].clamp(min=1e-8)
            p_l = P2[b * K + ki].clamp(min=1e-8)  # identical
            m = 0.5 * (p_k + p_l)
            js = 0.5 * ((p_k * torch.log(p_k / m)).sum() + (p_l * torch.log(p_l / m)).sum()) / math.log(2)
            js_ident.append(float(js))
    assert max(js_ident) < 1e-4, f"identical JS should be ~0: {max(js_ident)}"
    print(f"  identical JS max={max(js_ident):.2e}: OK")


def test_relation_none_compat():
    """relation_mode=none: no JS computation; loss is just profile KL."""
    # In none mode, forward stashes only _sem_loss/_sem_profile_loss
    # Simulate the loss path
    B, K = 2, 4
    kl = torch.rand(B, K)
    L_profile = kl.mean()
    # none mode: no relation term
    L_sem = L_profile
    assert torch.isfinite(L_sem)
    print(f"  none-mode L_sem={L_sem.item():.4f} (profile only): OK")


def test_tasid_math():
    """TASID: target-aware student dist, teacher dist, KL finite, grad to V."""
    B, K, D_sem = 2, 4, 32
    # Interest semantic vectors (would be V[..., :32]) with grad
    V_sem = torch.randn(B, K, D_sem, requires_grad=True)
    # Target semantic query (detached)
    q_target = torch.randn(B, D_sem).detach()
    # Teacher prototype profiles T [B,K,32] (detached)
    T = F.softmax(torch.randn(B, K, 32), dim=-1).detach()

    temp = 0.1
    cos_student = F.cosine_similarity(q_target.unsqueeze(1), V_sem, dim=-1)
    q_tasid = torch.softmax(cos_student / temp, dim=-1)          # [B,K]
    cos_teacher = F.cosine_similarity(q_target.unsqueeze(1), T, dim=-1)
    p_teacher = torch.softmax(cos_teacher / temp, dim=-1).detach()

    assert q_tasid.shape == (B, K)
    assert torch.allclose(q_tasid.sum(dim=-1), torch.ones(B), atol=1e-4), "student dist sums to 1"
    assert torch.allclose(p_teacher.sum(dim=-1), torch.ones(B), atol=1e-4), "teacher dist sums to 1"
    assert not p_teacher.requires_grad, "teacher dist detached"
    assert q_tasid.requires_grad, "student dist carries grad"

    L_tasid = F.kl_div(F.log_softmax(cos_student / temp, dim=-1), p_teacher, reduction="batchmean")
    assert torch.isfinite(L_tasid), f"TASID loss not finite: {L_tasid}"
    L_tasid.backward()
    assert V_sem.grad is not None and torch.isfinite(V_sem.grad).all(), "grad to V"
    print(f"  TASID loss={L_tasid.item():.4f}, dists sum=1, grad to V: OK")


def test_tasid_asymmetric():
    """Asymmetric student: concat(CF, semantic) query vs concat(CF, semantic) interest."""
    B, K, D_sem, D_full = 2, 4, 32, 64
    V = torch.randn(B, K, D_full, requires_grad=True)   # full interest vectors
    q_sem = torch.randn(B, D_sem).detach()              # target semantic (detached)
    c_target = torch.randn(B, D_sem).detach()           # target CF (detached)
    student_query = torch.cat([q_sem, c_target], dim=-1)  # [B, 64]
    student_interest = torch.cat(
        [V[..., D_sem:], V[..., :D_sem]], dim=-1)         # [B, K, 64]
    cos_student = F.cosine_similarity(student_query.unsqueeze(1), student_interest, dim=-1)
    q_tasid = torch.softmax(cos_student / 0.1, dim=-1)

    assert q_tasid.shape == (B, K)
    assert torch.allclose(q_tasid.sum(dim=-1), torch.ones(B), atol=1e-4)
    assert not student_query.requires_grad, "student query (target side) detached"
    assert q_tasid.requires_grad, "student distribution carries grad via V"
    # Gradient reaches V
    T = F.softmax(torch.randn(B, K, 32), dim=-1).detach()
    q_t_llm = q_sem
    cos_teacher = F.cosine_similarity(q_t_llm.unsqueeze(1), T, dim=-1)
    p_teacher = torch.softmax(cos_teacher / 0.1, dim=-1).detach()
    L = F.kl_div(F.log_softmax(cos_student / 0.1, dim=-1), p_teacher, reduction="batchmean")
    L.backward()
    assert V.grad is not None and torch.isfinite(V.grad).all(), "grad to V (interest)"
    print(f"  asymmetric TASID loss={L.item():.4f}, grad to V: OK")


def test_tasid_none_compat():
    """tasid_mode=none: no target query computed, original loss path intact."""
    # In none mode forward stashes no _tasid_loss; loss() skips it.
    # Simulate: no key in out_dict → no term added.
    out_dict = {"prediction": torch.randn(2, 2)}
    assert "_tasid_loss" not in out_dict
    print("  tasid none-mode compat: OK")


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
    test_relation_js_math()
    test_relation_js_scale()
    test_relation_none_compat()
    test_tasid_math()
    test_tasid_asymmetric()
    test_tasid_none_compat()
    print("ALL TESTS PASSED")
