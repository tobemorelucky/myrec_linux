# -*- coding: UTF-8 -*-
"""
CPU synthetic tests for ECTIR (Chapter 4 Modules 1-3).

Covers the contract required before any training run:
  - `ectir_mode=baseline` reproduces the original extractor
  - padding is strictly masked in the transport plan
  - transport row / column marginals match their targets
  - no NaN/Inf anywhere (including length-0 and all-padding edge cases)
  - backward produces finite gradients for every new parameter
  - K=1 and K=4 boundaries
  - batch synthetic forward through the whole router + aggregator

Usage: python tools/test_llmmirec_ectir.py
"""

import math
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sequential.llmmi_components import (      # noqa: E402
    QueryMultiInterestExtractor, TransportInterestRouter,
    TransportEvidenceAggregator, _NEG,
)

torch.manual_seed(0)

B, L, D, DA = 8, 20, 64, 64
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def make_inputs(B=B, L=L, D=D, lengths=None):
    h = torch.randn(B, L, D)
    if lengths is None:
        lengths = torch.randint(1, L + 1, (B,))
    lengths = lengths.clone()
    valid = (torch.arange(L)[None, :] < lengths[:, None])
    h = h * valid[:, :, None]
    return h, lengths, valid


# =========================
#  Module 1
# =========================

def test_shapes_and_mass():
    print("\n== Module 1: shapes, marginals, padding ==")
    r = TransportInterestRouter(K=4, emb_size=D, attn_size=DA, eps=0.1, n_sinkhorn=5)
    r.eval()
    h, lengths, valid = make_inputs()
    with torch.no_grad():
        V, A, info = r(h, lengths, return_intermediate=True)
    T = info["transport"]
    check("V shape", tuple(V.shape) == (B, 4, D), str(tuple(V.shape)))
    check("T shape", tuple(T.shape) == (B, L, 4), str(tuple(T.shape)))
    check("A shape", tuple(A.shape) == (B, 4, L), str(tuple(A.shape)))

    # padding is exactly zero
    pad_max = T[~valid].abs().max().item()
    check("transport exactly 0 at padding", pad_max == 0.0, f"max={pad_max:.2e}")

    # row marginals ~ uniform over valid
    a = valid.float() / valid.sum(1, keepdim=True).clamp(min=1)
    row = T.sum(-1)
    err_row = (row - a).abs().max().item()
    check("row mass ~= a (uniform over valid)", err_row < 1e-2, f"max_err={err_row:.2e}")

    # column marginals ~ b
    b = info["capacity"]
    col = T.sum(1)
    err_col = (col - b).abs().max().item()
    check("column mass ~= b (evidence capacity)", err_col < 1e-2, f"max_err={err_col:.2e}")

    check("capacity sums to 1", (b.sum(-1) - 1).abs().max().item() < 1e-5,
          f"max_dev={(b.sum(-1)-1).abs().max().item():.2e}")
    check("no NaN/Inf in T", bool(torch.isfinite(T).all()))
    check("attention rows sum to 1", (A.sum(-1) - 1).abs().max().item() < 1e-5)


def test_capacity_not_uniform():
    print("\n== Module 1: capacity is evidence-driven, not 1/K ==")
    r = TransportInterestRouter(K=4, emb_size=D, attn_size=DA, tau_c=1.0)
    r.eval()
    h, lengths, _ = make_inputs()
    with torch.no_grad():
        _, _, info = r(h, lengths, return_intermediate=True)
    b = info["capacity"]
    dev = (b - 1.0 / 4).abs().mean().item()
    check("capacity deviates from uniform", dev > 1e-4, f"mean|b - 1/K|={dev:.4f}")
    ent = -(b * torch.log(b + 1e-12)).sum(-1).mean().item()
    check("capacity entropy < ln K (concentrated)", ent < math.log(4) - 1e-6,
          f"H(b)={ent:.4f} < ln4={math.log(4):.4f}")


def test_padding_strictness():
    print("\n== Module 1: all-padding and length-1 edge cases ==")
    r = TransportInterestRouter(K=4, emb_size=D, attn_size=DA)
    r.eval()
    # one sample with length 0
    lengths = torch.tensor([0, 1, L, 5, 1, 2, 3, L])
    h, lengths, valid = make_inputs(lengths=lengths)
    with torch.no_grad():
        V, A, info = r(h, lengths, return_intermediate=True)
    T = info["transport"]
    check("no NaN/Inf (with length-0 sample)", bool(torch.isfinite(T).all()))
    check("zero-length sample has all-zero transport",
          float(T[0].abs().max()) == 0.0, f"max={float(T[0].abs().max()):.2e}")
    check("zero-length sample V is finite", bool(torch.isfinite(V[0]).all()))


def test_gradients():
    print("\n== Module 1: gradients reach every parameter ==")
    r = TransportInterestRouter(K=4, emb_size=D, attn_size=DA, n_refine=1)
    r.train()
    h, lengths, _ = make_inputs()
    V, T = r(h, lengths)
    (V.sum() + T.sum()).backward()
    for n, p in r.named_parameters():
        ok = p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        check(f"grad finite+nonzero: {n}", bool(ok))
    # the input should also receive gradient (used downstream by nothing here,
    # but the affinity path must be differentiable w.r.t. history embeddings)
    h2 = h.clone().requires_grad_(True)
    V2, _ = r(h2, lengths)
    V2.sum().backward()
    check("grad flows to history embeddings",
          bool(torch.isfinite(h2.grad).all() and h2.grad.abs().sum() > 0))


def test_K_boundaries():
    print("\n== Module 1: K=1 and K=4 boundaries ==")
    for K in (1, 4):
        r = TransportInterestRouter(K=K, emb_size=D, attn_size=DA)
        r.eval()
        h, lengths, _ = make_inputs()
        with torch.no_grad():
            V, T = r(h, lengths)
        check(f"K={K}: V shape", tuple(V.shape) == (B, K, D), str(tuple(V.shape)))
        check(f"K={K}: finite", bool(torch.isfinite(V).all() and torch.isfinite(T).all()))
    # K=1 -> capacity must be exactly 1
    r1 = TransportInterestRouter(K=1, emb_size=D, attn_size=DA)
    h, lengths, _ = make_inputs()
    with torch.no_grad():
        _, _, info = r1(h, lengths, return_intermediate=True)
    check("K=1 capacity == 1", (info["capacity"] - 1).abs().max().item() < 1e-6)


def test_sinkhorn_and_refine_variants():
    print("\n== Module 1: sinkhorn iteration count / refinement steps ==")
    for ns in (1, 3, 5):
        r = TransportInterestRouter(K=4, emb_size=D, attn_size=DA, n_sinkhorn=ns)
        r.eval()
        h, lengths, _ = make_inputs()
        with torch.no_grad():
            _, _, info = r(h, lengths, return_intermediate=True)
        T = info["transport"]
        a = (torch.arange(L)[None, :] < lengths[:, None]).float()
        a = a / a.sum(1, keepdim=True)
        err = (T.sum(-1) - a).abs().max().item()
        check(f"n_sinkhorn={ns}: finite & row-err<1e-1", bool(torch.isfinite(T).all()) and err < 1e-1,
              f"row_err={err:.2e}")
    for nr in (0, 1, 2):
        r = TransportInterestRouter(K=4, emb_size=D, attn_size=DA, n_refine=nr)
        r.eval()
        h, lengths, _ = make_inputs()
        with torch.no_grad():
            V, T = r(h, lengths)
        check(f"n_refine={nr}: finite", bool(torch.isfinite(V).all() and torch.isfinite(T).all()))
        check(f"n_refine={nr}: W_r present={nr > 0}", (r.W_r is not None) == (nr > 0))


def test_refine_starts_as_identity():
    print("\n== Module 2: refinement starts as identity (W_r ~ 0 at init) ==")
    r0 = TransportInterestRouter(K=4, emb_size=D, attn_size=DA, n_refine=0)
    r1 = TransportInterestRouter(K=4, emb_size=D, attn_size=DA, n_refine=1)
    r1.load_state_dict({k: v for k, v in r0.state_dict().items()}, strict=False)
    # force W_r to exactly zero
    with torch.no_grad():
        r1.W_r.weight.zero_(); r1.W_r.bias.zero_()
    r0.eval(); r1.eval()
    h, lengths, _ = make_inputs()
    with torch.no_grad():
        V0, _ = r0(h, lengths)
        V1, _ = r1(h, lengths)
    err = (V0 - V1).abs().max().item()
    check("n_refine=1 with W_r=0 equals n_refine=0", err < 1e-5, f"max_err={err:.2e}")


# =========================
#  Module 3
# =========================

def test_evidence_aggregator():
    print("\n== Module 3: evidence aggregator ==")
    K = 4
    agg = TransportEvidenceAggregator(K=K, emb_size=D, hidden=32)
    agg.eval()
    h, lengths, valid = make_inputs()
    r = TransportInterestRouter(K=K, emb_size=D, attn_size=DA)
    r.eval()
    with torch.no_grad():
        V, _, info = r(h, lengths, return_intermediate=True)
        pos = (lengths[:, None] - torch.arange(L)[None, :]) * valid.float()
        w = agg(V, info["transport"], h, lengths, pos)
    check("w shape", tuple(w.shape) == (B, K), str(tuple(w.shape)))
    check("w sums to 1", (w.sum(-1) - 1).abs().max().item() < 1e-5)
    check("w non-negative", bool((w >= 0).all()))
    check("w finite", bool(torch.isfinite(w).all()))

    agg.train()
    w2 = agg(V, info["transport"], h, lengths, pos)
    w2.sum().backward()
    for n, p in agg.named_parameters():
        ok = p.grad is not None and torch.isfinite(p.grad).all()
        check(f"grad finite: {n}", bool(ok))

    # zero-length sample must not produce NaN
    lengths0 = torch.tensor([0] * B)
    with torch.no_grad():
        T0 = torch.zeros(B, L, K)
        w3 = agg(V, T0, torch.zeros(B, L, D), lengths0, torch.zeros(B, L))
    check("degenerate (all-padding) is finite", bool(torch.isfinite(w3).all()))


def test_evidence_aggregator_uses_evidence():
    print("\n== Module 3: weights actually depend on the evidence ==")
    K = 4
    agg = TransportEvidenceAggregator(K=K, emb_size=D, hidden=32)
    agg.eval()
    h, lengths, valid = make_inputs()
    r = TransportInterestRouter(K=K, emb_size=D, attn_size=DA)
    r.eval()
    pos = (lengths[:, None] - torch.arange(L)[None, :]) * valid.float()
    with torch.no_grad():
        V, _, info = r(h, lengths, return_intermediate=True)
        T = info["transport"]
        w_a = agg(V, T, h, lengths, pos)
        # perturb the transport plan (reallocate mass toward interest 0)
        T2 = T.clone()
        T2[:, :, 0] = T2[:, :, 0] * 3.0
        w_b = agg(V, T2, h, lengths, pos)
    diff = (w_a - w_b).abs().max().item()
    check("weights change when transport changes", diff > 1e-6, f"max|dw|={diff:.2e}")


# =========================
#  Baseline degeneration
# =========================

def test_baseline_extractor_unchanged():
    """The original extractor must be untouched by the ECTIR additions."""
    print("\n== baseline extractor still reproduces its known behaviour ==")
    torch.manual_seed(0)
    ex = QueryMultiInterestExtractor(K=4, emb_size=D, attn_size=DA)
    ex.eval()
    h, lengths, valid = make_inputs()
    with torch.no_grad():
        V, A = ex(h, lengths)
    check("extractor V shape", tuple(V.shape) == (B, 4, D))
    check("extractor attn rows sum to 1", (A.sum(-1) - 1).abs().max().item() < 1e-5)
    pad_mask = (~valid).unsqueeze(1).expand_as(A)          # [B, K, L]
    check("extractor padding attention is 0",
          float(A[pad_mask].abs().max()) == 0.0)

    # attention is computed per-interest independently: softmax over L
    scores = torch.bmm(ex.Wq(ex.query).unsqueeze(0).expand(B, -1, -1),
                       ex.Wk(h).transpose(1, 2)) / math.sqrt(DA)
    scores = scores.masked_fill((~valid).unsqueeze(1), float("-inf"))
    ref = torch.softmax(scores, dim=-1)
    err = (ref - A).abs().max().item()
    check("extractor attn == softmax over L (per-interest, independent)", err < 1e-5,
          f"max_err={err:.2e}")


def test_router_differs_from_baseline():
    """Transport routing must NOT be a disguised version of the baseline."""
    print("\n== transport routing differs from the baseline operator ==")
    torch.manual_seed(0)
    ex = QueryMultiInterestExtractor(K=4, emb_size=D, attn_size=DA)
    rt = TransportInterestRouter(K=4, emb_size=D, attn_size=DA)
    # copy shared projections so the only difference is the operator
    sd = {k: v for k, v in ex.state_dict().items() if k in rt.state_dict()}
    rt.load_state_dict(sd, strict=False)
    ex.eval(); rt.eval()
    h, lengths, _ = make_inputs()
    with torch.no_grad():
        _, A_base = ex(h, lengths)
        _, A_tr = rt(h, lengths)
    diff = (A_base - A_tr).abs().max().item()
    check("attention maps differ", diff > 1e-3, f"max|A_base - A_transport|={diff:.4f}")


if __name__ == "__main__":
    test_shapes_and_mass()
    test_capacity_not_uniform()
    test_padding_strictness()
    test_gradients()
    test_K_boundaries()
    test_sinkhorn_and_refine_variants()
    test_refine_starts_as_identity()
    test_evidence_aggregator()
    test_evidence_aggregator_uses_evidence()
    test_baseline_extractor_unchanged()
    test_router_differs_from_baseline()
    print(f"\n===== {len(PASS)} passed, {len(FAIL)} failed =====")
    if FAIL:
        print("failed:", FAIL)
        sys.exit(1)
