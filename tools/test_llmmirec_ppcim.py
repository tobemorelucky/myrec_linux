# -*- coding: UTF-8 -*-
"""
CPU tests for PPCIM (Chapter 4 Round 1).

Covers the contract required before training:
  A. shapes                          V[B,K,D] E[B,C,D] m/pi[B,K,C] pred[B,C]
  B. baseline exact reproduction     real ASPCF checkpoint, tensor-level <= 1e-7
  C. uniform prior                   p_k = 1/K and independent of history
  D. posterior normalisation         sum_k pi = 1
  E. train/test candidate count      C=2 and C=1001 use the identical formula
  F. permutation independence        score is pointwise in each candidate
  G. no leakage                      positive and negatives share one formula
  H. numerical stability             no NaN/Inf, no candidate-set coupling
  I. backward                        grads reach item encoder / extractor / aggregator
  J. large-tau consistency           sqrt(D)*tau*lse -> (sum_k p_k V_k)^T e_j

Usage: python tools/test_llmmirec_ppcim.py
"""

import math
import os
import pickle
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader                       # noqa: E402
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF     # noqa: E402
from models.sequential.LLMMIRecPPCIM import LLMMIRecPPCIM     # noqa: E402

torch.manual_seed(0)

B, K, D = 8, 4, 64
PASS, FAIL = [], []
ASPCF_CKPT = "new_model/llmmirec_aspcf_phase2/beauty/seed42/LLMMIRecASPCF_seed42.pt"
CORPUS = "./data/beauty/SeqReader.pkl"


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


class Args:
    pass


def mk_args(mode="marginal", prior="history", tau=1.0):
    a = Args()
    a.device = torch.device("cpu"); a.model_path = ""; a.buffer = 1; a.history_max = 20
    a.num_neg = 1; a.test_all = 0; a.emb_size = D; a.attn_size = 64; a.K = K
    a.dropout = 0.1; a.item_encoder = "aspcf"
    a.llm_emb_path = "./data/beauty/handled/llm_table_pca1536.pkl"
    a.semantic_rank = 512; a.semantic_dim = 32; a.semantic_hidden = 128
    a.complement_dim = 32; a.tail_hidden = 64; a.complement_hidden = 64
    a.gate_hidden = 64; a.aspcf_gate_mode = "basic"
    a.adapter_hidden = 256; a.adapter_activation = "gelu"; a.adapter_use_ln = 0
    a.gamma_init = 0.1; a.gamma_trainable = 0
    a.lambda_relation = 0.01; a.relation_sample_size = 128
    a.relation_teacher_temp = 0.1; a.relation_student_temp = 0.1
    a.ppcim_mode = mode; a.ppcim_prior = prior; a.ppcim_tau = tau; a.ppcim_eps = 1e-8
    return a


def load_corpus():
    return pickle.load(open(CORPUS, "rb"))


# =========================
#  A. shapes + C/D/H
# =========================

def test_shapes_and_numerics():
    print("\n== A/D/H: shapes, posterior normalisation, stability ==")
    corpus = load_corpus()
    m = LLMMIRecPPCIM(mk_args("marginal"), corpus)
    m.eval()
    C = 1001
    iv = torch.randn(B, K, D)
    ce = torch.randn(B, C, D)
    h = torch.randn(B, 20, D)
    ln = torch.randint(1, 21, (B,))
    with torch.no_grad():
        pred, u, extra = m._score(iv, ce, h, ln, need_posterior=True)
    check("prediction [B,C]", tuple(pred.shape) == (B, C), str(tuple(pred.shape)))
    check("user_vector [B,D]", tuple(u.shape) == (B, D))
    check("m [B,K,C]", tuple(extra["candidate_interest_match"].shape) == (B, K, C),
          str(tuple(extra["candidate_interest_match"].shape)))
    pi = extra["candidate_interest_posterior"]
    check("pi [B,K,C]", tuple(pi.shape) == (B, K, C))
    check("D: sum_k pi == 1", (pi.sum(1) - 1).abs().max().item() < 1e-5,
          f"max_dev={(pi.sum(1)-1).abs().max().item():.2e}")
    check("pi non-negative", bool((pi >= 0).all()))
    check("H: finite", bool(torch.isfinite(pred).all() and torch.isfinite(pi).all()))
    check("prior [B,K]", tuple(extra["history_interest_prior"].shape) == (B, K))
    check("no trainable params added",
          all(not p.requires_grad for n, p in m.named_parameters()
              if "ppcim" in n.lower()))


def test_posterior_mean_shape():
    print("\n== A: posterior_mean mode ==")
    corpus = load_corpus()
    m = LLMMIRecPPCIM(mk_args("posterior_mean"), corpus)
    m.eval()
    C = 1001
    with torch.no_grad():
        pred, u, extra = m._score(torch.randn(B, K, D), torch.randn(B, C, D),
                                  torch.randn(B, 20, D), torch.randint(1, 21, (B,)),
                                  need_posterior=True)
    check("prediction [B,C]", tuple(pred.shape) == (B, C))
    check("pi present", "candidate_interest_posterior" in extra)
    check("finite", bool(torch.isfinite(pred).all()))


# =========================
#  C. uniform prior
# =========================

def test_uniform_prior():
    print("\n== C: uniform prior is 1/K and history-independent ==")
    corpus = load_corpus()
    m = LLMMIRecPPCIM(mk_args("marginal", prior="uniform"), corpus)
    m.eval()
    h1, h2 = torch.randn(B, 20, D), torch.randn(B, 20, D) * 3.0
    ln = torch.randint(1, 21, (B,))
    with torch.no_grad():
        _, _, e1 = m._score(torch.randn(B, K, D), torch.randn(B, 5, D), h1, ln)
        _, _, e2 = m._score(torch.randn(B, K, D), torch.randn(B, 5, D), h2, ln)
    p1, p2 = e1["history_interest_prior"], e2["history_interest_prior"]
    check("uniform prior == 1/K", (p1 - 1.0 / K).abs().max().item() < 1e-7,
          f"max_dev={(p1 - 1.0/K).abs().max().item():.2e}")
    check("uniform prior independent of history",
          (p1 - p2).abs().max().item() == 0.0)
    # history prior should differ across histories
    mh = LLMMIRecPPCIM(mk_args("marginal", prior="history"), corpus)
    mh.eval()
    with torch.no_grad():
        _, _, f1 = mh._score(torch.randn(B, K, D), torch.randn(B, 5, D), h1, ln)
        _, _, f2 = mh._score(torch.randn(B, K, D), torch.randn(B, 5, D), h2, ln)
    check("history prior DOES depend on history",
          (f1["history_interest_prior"] - f2["history_interest_prior"]).abs().max() > 1e-6)


# =========================
#  E/F. candidate-set independence
# =========================

def test_candidate_count_and_permutation():
    print("\n== E/F: candidate count C=2 vs 1001, permutation independence ==")
    corpus = load_corpus()
    m = LLMMIRecPPCIM(mk_args("marginal"), corpus)
    m.eval()
    iv = torch.randn(B, K, D); h = torch.randn(B, 20, D); ln = torch.randint(1, 21, (B,))
    e_big = torch.randn(B, 1001, D)
    with torch.no_grad():
        s_big, _, _ = m._score(iv, e_big, h, ln)
        s_two, _, _ = m._score(iv, e_big[:, :2], h, ln)

    # A candidate's score must not depend on the other candidates.
    # Compare relatively: bmm uses different kernels for C=2 vs C=1001, so
    # float32 reduction order gives O(eps * |score|) noise, not a real coupling.
    d_E = (s_two - s_big[:, :2]).abs().max().item()
    scale_E = s_big.abs().max().item()
    check("E: C=2 scores == first 2 of C=1001 (relative)",
          d_E / scale_E < 1e-5,
          f"abs={d_E:.2e} rel={d_E/scale_E:.2e} (score scale={scale_E:.1f})")

    # permutation independence
    perm = torch.randperm(1001)
    with torch.no_grad():
        s_perm, _, _ = m._score(iv, e_big[:, perm], h, ln)
    check("F: permuting candidates permutes scores",
          (s_perm - s_big[:, perm]).abs().max().item() < 1e-6,
          f"max_diff={(s_perm - s_big[:, perm]).abs().max().item():.2e}")


# =========================
#  G. no leakage
# =========================

def test_no_leakage():
    print("\n== G: positives and negatives use the identical formula ==")
    corpus = load_corpus()
    m = LLMMIRecPPCIM(mk_args("marginal"), corpus)
    m.eval()
    iv = torch.randn(B, K, D); h = torch.randn(B, 20, D); ln = torch.randint(1, 21, (B,))
    e = torch.randn(B, 5, D)
    with torch.no_grad():
        s_all, _, _ = m._score(iv, e, h, ln)
        # scoring candidate 0 alone (as if it were the only candidate) must match
        s_only0, _, _ = m._score(iv, e[:, :1], h, ln)
    check("G: score of candidate 0 identical in both sets",
          (s_all[:, :1] - s_only0).abs().max().item() < 1e-6,
          f"max_diff={(s_all[:, :1]-s_only0).abs().max().item():.2e}")
    # shuffling which candidate is "positive" must not change any score
    e2 = e.clone(); e2[:, [0, 3]] = e2[:, [3, 0]]
    with torch.no_grad():
        s2, _, _ = m._score(iv, e2, h, ln)
    check("G: swapping positive position only permutes scores",
          (s2[:, 0] - s_all[:, 3]).abs().max().item() < 1e-6 and
          (s2[:, 3] - s_all[:, 0]).abs().max().item() < 1e-6)


# =========================
#  H. numerical stability
# =========================

def test_numerical_stability():
    print("\n== H: numerical stability across tau ==")
    corpus = load_corpus()
    iv = torch.randn(B, K, D) * 5; h = torch.randn(B, 20, D); ln = torch.randint(1, 21, (B,))
    e = torch.randn(B, 17, D) * 5
    for tau in (0.01, 0.1, 1.0, 10.0, 1000.0):
        m = LLMMIRecPPCIM(mk_args("marginal", tau=tau), corpus); m.eval()
        with torch.no_grad():
            s, _, _ = m._score(iv, e, h, ln)
        check(f"tau={tau:>7}: finite", bool(torch.isfinite(s).all()),
              f"range=[{float(s.min()):.3g},{float(s.max()):.3g}]")
    try:
        LLMMIRecPPCIM(mk_args("marginal", tau=0.0), corpus)
        check("tau<=0 raises", False, "no error")
    except ValueError:
        check("tau<=0 raises", True)
    try:
        LLMMIRecPPCIM(mk_args("marginal", tau=-1.0), corpus)
        check("tau<0 raises", False, "no error")
    except ValueError:
        check("tau<0 raises", True)


# =========================
#  J. large-tau consistency
# =========================

def test_large_tau_consistency():
    print("\n== J: sqrt(D)*tau*logsumexp -> (sum_k p_k V_k)^T e_j  as tau grows ==")
    corpus = load_corpus()
    iv = torch.randn(B, K, D, dtype=torch.float64)
    e = torch.randn(B, 9, D, dtype=torch.float64)
    h = torch.randn(B, 20, D); ln = torch.randint(1, 21, (B,))
    m = LLMMIRecPPCIM(mk_args("marginal"), corpus); m.eval()
    with torch.no_grad():
        p = m.aggregator(h, ln).double()                       # [B,K]
    sqrtD = math.sqrt(D)
    mmat = torch.bmm(iv, e.transpose(1, 2)) / sqrtD            # [B,K,C]
    ref = (torch.bmm(p.unsqueeze(1), iv).squeeze(1)[:, None, :] * e).sum(-1)  # [B,C]
    prev = None
    for tau in (1.0, 10.0, 100.0, 1000.0):
        z = torch.log(p + 1e-8).unsqueeze(-1) + mmat / tau
        pred = sqrtD * tau * torch.logsumexp(z, dim=1)
        err = (pred - ref).abs().max().item()
        rel = err / ref.abs().max().item()
        if prev is not None:
            check(f"tau={tau:>7}: error shrinks", err < prev + 1e-12,
                  f"abs_err={err:.3e} (prev {prev:.3e})")
        prev = err
    check("J: tau=1000 relative error < 1e-3", rel < 1e-3, f"rel_err={rel:.3e}")


# =========================
#  I. backward
# =========================

def test_backward():
    print("\n== I: gradients reach item encoder / extractor / aggregator ==")
    corpus = load_corpus()
    m = LLMMIRecPPCIM(mk_args("marginal"), corpus)
    m.train()
    Bb, L = 4, 20
    # Build a REALISTIC batch: valid items at the FRONT, right-padded with 0
    # (as SeqReader/collate produce). Without the zeros, valid_his is all-ones
    # and position = lengths - i goes negative for short histories.
    lengths = torch.randint(1, L + 1, (Bb,))
    hist = torch.zeros(Bb, L, dtype=torch.long)
    for i in range(Bb):
        hist[i, :lengths[i]] = torch.randint(1, 100, (int(lengths[i]),))
    fd = {
        "history_items": hist,
        "lengths": lengths,
        "item_id": torch.randint(1, 100, (Bb, 3)),
    }
    out = m(fd)
    check("marginal: prediction shape", tuple(out["prediction"].shape) == (Bb, 3))
    out["prediction"].sum().backward()
    grp = {"item_encoder": [], "extractor": [], "aggregator": []}
    for n, p in m.named_parameters():
        for k in grp:
            if n.startswith(k):
                grp[k].append(p.grad is not None and torch.isfinite(p.grad).all()
                              and p.grad.abs().sum() > 0)
    for k, v in grp.items():
        check(f"grad finite+nonzero in {k}", len(v) > 0 and all(v))

    # posterior_mean must also be differentiable
    m2 = LLMMIRecPPCIM(mk_args("posterior_mean"), corpus); m2.train()
    out2 = m2(fd)
    out2["prediction"].sum().backward()
    ok = all((p.grad is not None and torch.isfinite(p.grad).all())
             for n, p in m2.named_parameters() if p.requires_grad)
    check("posterior_mean: all grads finite", ok)


# =========================
#  B. baseline exact reproduction (real checkpoint)
# =========================

def test_baseline_reproduction():
    print("\n== B: baseline reproduces real ASPCF checkpoint (tensor level) ==")
    if not os.path.exists(ASPCF_CKPT) or not os.path.exists(CORPUS):
        print("  [skip] checkpoint or corpus missing")
        return
    corpus = load_corpus()
    st = torch.load(ASPCF_CKPT, map_location="cpu")
    ma = LLMMIRecASPCF(mk_args(), corpus); ma.load_state_dict(st, strict=False)
    mb = LLMMIRecPPCIM(mk_args("baseline"), corpus); mb.load_state_dict(st, strict=False)
    check("param count identical", ma.count_variables() == mb.count_variables(),
          f"{ma.count_variables()} vs {mb.count_variables()}")
    check("state_dict keys identical",
          set(ma.state_dict().keys()) == set(mb.state_dict().keys()))
    ma.eval(); mb.eval(); ma.device = torch.device("cpu"); mb.device = torch.device("cpu")

    ds = ma.Dataset(ma, corpus, "test"); ds.prepare()
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0,
                    collate_fn=ds.collate_batch)
    maxdiff = 0.0
    with torch.inference_mode():
        for i, b in enumerate(dl):
            d = (ma(b)["prediction"] - mb(b)["prediction"]).abs().max().item()
            maxdiff = max(maxdiff, d)
            if i >= 20:
                break
    check("B: per-batch prediction max|diff| <= 1e-7", maxdiff <= 1e-7, f"max={maxdiff:.3e}")


if __name__ == "__main__":
    test_shapes_and_numerics()
    test_posterior_mean_shape()
    test_uniform_prior()
    test_candidate_count_and_permutation()
    test_no_leakage()
    test_numerical_stability()
    test_large_tau_consistency()
    test_backward()
    test_baseline_reproduction()
    print(f"\n===== {len(PASS)} passed, {len(FAIL)} failed =====")
    if FAIL:
        print("failed:", FAIL)
        sys.exit(1)
