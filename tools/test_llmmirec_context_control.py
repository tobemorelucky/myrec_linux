# -*- coding: UTF-8 -*-
"""
Unit tests for LLMMIRecContextControl (Chapter 4 Phase 3).

Covers the 9 required checks:
  1. baseline mode loads the real ASPCF checkpoint -> prediction bit-identical
  2. shape [B,L,64] -> [B,L,64]
  3. padding: padded output positions are exactly zero
  4. self-attn: padded positions are never attended to (effective keys exclude them)
  5. ffn_control: perturbing one position changes NO other position's output
  6. self_attn: perturbing one valid position DOES change other valid outputs
  7. no NaN / Inf
  8. backward: contextualizer / ItemEncoder / extractor all receive gradients
  9. length = 1 and length = 20 boundaries

Run:
  python tools/test_llmmirec_context_control.py
"""

import math
import os
import pickle
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF                    # noqa: E402
from models.sequential.LLMMIRecContextControl import LLMMIRecContextControl  # noqa: E402

DS = "beauty"
CKPT = f"new_model/llmmirec_aspcf_phase2/{DS}/seed42/LLMMIRecASPCF_seed42.pt"
DEV = torch.device("cpu")

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def make_args(mode, hidden_ctl=256):
    import argparse
    a = argparse.Namespace()
    a.emb_size = 64; a.attn_size = 64; a.K = 4; a.history_max = 20
    a.num_neg = 1; a.test_all = 0; a.buffer = 0; a.dropout = 0.1
    a.item_encoder = "aspcf"
    a.llm_emb_path = f"./data/{DS}/handled/llm_table_pca1536.pkl"
    a.semantic_rank = 512; a.semantic_dim = 32; a.semantic_hidden = 128
    a.complement_dim = 32; a.tail_hidden = 64; a.complement_hidden = 64
    a.gate_hidden = 64; a.aspcf_gate_mode = "basic"
    a.adapter_hidden = 256; a.adapter_activation = "gelu"; a.adapter_use_ln = 0
    a.gamma_init = 0.1; a.gamma_trainable = 0
    a.lambda_relation = 0.01; a.relation_sample_size = 128
    a.relation_teacher_temp = 0.1; a.relation_student_temp = 0.1
    a.context_mode = mode; a.context_dropout = 0.1; a.context_heads = 4
    a.context_ffn_hidden = 128; a.context_ffn_control_hidden = hidden_ctl
    a.device = DEV; a.path = "./data/"; a.model_path = CKPT
    return a


def build(mode, corpus, ckpt=None):
    m = LLMMIRecContextControl(make_args(mode), corpus)
    if ckpt:
        missing, unexpected = m.load_state_dict(
            torch.load(ckpt, map_location="cpu"), strict=False)
        return m, missing, unexpected
    return m, None, None


def synth_batch(K_hist=None, lengths=None, seed=0):
    """Right-padded history (zeros at the tail) + 1 pos + 3 neg candidates."""
    g = torch.Generator().manual_seed(seed)
    B = len(lengths)
    L = K_hist
    history = torch.zeros(B, L, dtype=torch.long)
    for b, ln in enumerate(lengths):
        history[b, :ln] = torch.randint(1, 1000, (ln,), generator=g)
    return {
        "history_items": history,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "item_id": torch.randint(1, 1000, (B, 4), generator=g),
        "batch_size": B,
        "phase": "train",
    }


def main():
    torch.manual_seed(0)
    corpus = pickle.load(open(f"./data/{DS}/SeqReader.pkl", "rb"))

    print("=" * 90)
    print("LLMMIRecContextControl unit tests")
    print("=" * 90)

    # ---------------------------------------------------------------- 1
    print("\n[1] baseline mode == ASPCF (real checkpoint)")
    ref = LLMMIRecASPCF(make_args("baseline"), corpus)
    ref.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)
    ref.eval()

    base, missing, unexpected = build("baseline", corpus, CKPT)
    base.eval()
    check("1a. baseline has NO contextualizer built",
          base.contextualizer is None)
    check("1b. baseline param count == ASPCF param count",
          base.count_variables() == ref.count_variables(),
          f"ctx={base.count_variables()} aspcf={ref.count_variables()}")
    check("1c. checkpoint loads with no missing/unexpected keys",
          len(missing) == 0 and len(unexpected) == 0,
          f"missing={len(missing)} unexpected={len(unexpected)}")

    batch = synth_batch(K_hist=20, lengths=[20, 7, 3, 1], seed=1)
    with torch.no_grad():
        o_ref = ref(batch, return_intermediate=True)
        o_base = base(batch, return_intermediate=True)
    d = float((o_ref["prediction"] - o_base["prediction"]).abs().max())
    check("1d. prediction bit-identical to LLMMIRecASPCF", d == 0.0, f"max|diff|={d:.3e}")

    # ---------------------------------------------------------------- 2
    print("\n[2] shape [B,L,64] -> [B,L,64]")
    for mode in ("ffn_control", "self_attn"):
        m, _, _ = build(mode, corpus, CKPT)
        m.eval()
        with torch.no_grad():
            o = m(batch, return_intermediate=True)
        pre, post = o["history_pre_context"], o["history_post_context"]
        ok = pre.shape == post.shape == (4, 20, 64)
        check(f"2. {mode}: {tuple(pre.shape)} -> {tuple(post.shape)}", ok)

    # ---------------------------------------------------------------- 3
    print("\n[3] padded output positions are exactly zero")
    for mode in ("ffn_control", "self_attn"):
        m, _, _ = build(mode, corpus, CKPT)
        m.eval()
        with torch.no_grad():
            o = m(batch, return_intermediate=True)
        post = o["history_post_context"]
        L = post.size(1)
        pad = torch.arange(L)[None, :] >= batch["lengths"][:, None]
        mx = float(post[pad].abs().max()) if pad.any() else 0.0
        check(f"3. {mode}: max|output| at padded positions == 0", mx == 0.0,
              f"max={mx:.3e}")

    # ---------------------------------------------------------------- 4
    print("\n[4] self-attn never attends to padded keys")
    m, _, _ = build("self_attn", corpus, CKPT)
    m.eval()
    with torch.no_grad():
        o = m(batch, return_intermediate=True)
    A = o["context_attention_maps"]              # [B, h, L_q, L_k]
    L = A.size(-1)
    # NOTE on semantics: only KEYS are masked (per spec). A padded QUERY row
    # still attends legally to valid keys; its OUTPUT is discarded by the
    # explicit zero-out (test 3). So the correct check is over the KEY axis.
    pad_key = torch.arange(L)[None, :] >= batch["lengths"][:, None]  # [B,L] True=padded KEY
    mass_on_pad = A * pad_key[:, None, None, :].to(A.dtype)          # zero out valid keys
    mx = float(mass_on_pad.max())
    check("4a. attention mass placed on padded KEYS == 0", mx == 0.0, f"max={mx:.3e}")

    # every VALID query row (one that has at least one valid key) sums to 1
    row_sum = A.sum(-1)                                              # [B,h,L_q]
    has_key = (batch["lengths"][:, None] > 0).expand(-1, L)          # [B,L_q]
    rs = row_sum.permute(0, 2, 1)[has_key]
    s_mx = float((rs - 1.0).abs().max())
    check("4b. valid query rows sum to 1.0", s_mx < 1e-5, f"max|sum-1|={s_mx:.3e}")
    # a query row with NO valid key at all must be exactly 0 (not NaN)
    zero_len = synth_batch(K_hist=20, lengths=[0, 5], seed=3)
    with torch.no_grad():
        A0 = m(zero_len, return_intermediate=True)["context_attention_maps"]
    row0 = A0[0].sum(-1)                                             # batch elem with length 0
    check("4c. all-masked query row is exactly 0, not NaN",
          float(row0.abs().max()) == 0.0 and torch.isfinite(A0).all().item(),
          f"max={float(row0.abs().max()):.3e}")

    # ---------------------------------------------------------------- 5
    print("\n[5] ffn_control: NO cross-position mixing")
    m, _, _ = build("ffn_control", corpus, CKPT)
    m.eval()
    b2 = synth_batch(K_hist=20, lengths=[20, 20], seed=2)
    with torch.no_grad():
        out_a = m(b2, return_intermediate=True)["history_post_context"]
        b2_mod = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                  for k, v in b2.items()}
        b2_mod["history_items"][0, 5] = 777            # perturb ONE position
        out_b = m(b2_mod, return_intermediate=True)["history_post_context"]
    diff = (out_a - out_b).abs().amax(dim=(0, 2))       # [L]
    changed = (diff > 0).nonzero().flatten().tolist()
    check("5. perturbing position 5 changes ONLY position 5",
          changed == [5], f"changed positions={changed}")

    # ---------------------------------------------------------------- 6
    print("\n[6] self_attn: cross-position interaction IS real")
    m, _, _ = build("self_attn", corpus, CKPT)
    m.eval()
    with torch.no_grad():
        out_a = m(b2, return_intermediate=True)["history_post_context"]
        out_b = m(b2_mod, return_intermediate=True)["history_post_context"]
    diff = (out_a - out_b).abs().amax(dim=(0, 2))
    changed = (diff > 0).nonzero().flatten().tolist()
    others = [i for i in changed if i != 5]
    check("6. perturbing position 5 changes OTHER valid positions too",
          len(others) > 0, f"changed positions={changed}")

    # ---------------------------------------------------------------- 7
    print("\n[7] no NaN / Inf")
    for mode in ("baseline", "ffn_control", "self_attn"):
        m, _, _ = build(mode, corpus, CKPT)
        m.eval()
        with torch.no_grad():
            o = m(batch, return_intermediate=True)
        ok = all(torch.isfinite(o[k]).all().item()
                 for k in ("prediction", "history_post_context", "interest_vectors"))
        check(f"7. {mode}: all outputs finite", ok)

    # ---------------------------------------------------------------- 8
    print("\n[8] backward: gradients reach contextualizer / ItemEncoder / extractor")
    for mode in ("ffn_control", "self_attn"):
        m, _, _ = build(mode, corpus, CKPT)
        m.train()
        o = m(batch)
        o["prediction"].sum().backward()
        g_ctx = sum(p.grad.abs().sum().item() for p in m.contextualizer.parameters()
                    if p.grad is not None)
        g_enc = sum(p.grad.abs().sum().item() for p in m.item_encoder.parameters()
                    if p.grad is not None)
        g_ext = sum(p.grad.abs().sum().item() for p in m.extractor.parameters()
                    if p.grad is not None)
        check(f"8. {mode}: ctx={g_ctx:.3e} enc={g_enc:.3e} ext={g_ext:.3e}",
              g_ctx > 0 and g_enc > 0 and g_ext > 0)

    # ---------------------------------------------------------------- 9
    print("\n[9] length = 1 and length = 20 boundaries")
    for mode in ("ffn_control", "self_attn"):
        m, _, _ = build(mode, corpus, CKPT)
        m.eval()
        for ln in (1, 20):
            bb = synth_batch(K_hist=ln, lengths=[ln] * 3, seed=ln)
            with torch.no_grad():
                o = m(bb, return_intermediate=True)
            ok = torch.isfinite(o["prediction"]).all().item()
            if mode == "self_attn":
                A = o["context_attention_maps"]
                s = float((A.sum(-1) - 1.0).abs().max())
                ok = ok and s < 1e-5
            check(f"9. {mode} len={ln}: finite + valid attention", ok)

    # ---------------------------------------------------------------- params
    print("\n[+] parameter counts")
    _, _, _ = 0, 0, 0
    pb = build("baseline", corpus, CKPT)[0].count_variables()
    pf = build("ffn_control", corpus, CKPT)[0]
    ps = build("self_attn", corpus, CKPT)[0]
    cf, cs = pf.count_context_params(), ps.count_context_params()
    print(f"    ASPCF / baseline      : {pb}")
    print(f"    FFN control           : {pf.count_variables()}  (+{cf})")
    print(f"    Self-attn             : {ps.count_variables()}  (+{cs})")
    print(f"    control-block gap     : {abs(cs - cf)} "
          f"({abs(cs - cf) / max(cf, 1) * 100:.2f}% of the block)")

    print("\n" + "=" * 90)
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
    print("=" * 90)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
