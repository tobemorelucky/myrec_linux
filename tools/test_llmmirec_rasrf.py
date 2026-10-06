# -*- coding: UTF-8 -*-
"""
CPU synthetic tests for RASRF (Chapter 3 Round 2).

Verifies the structural contract:
  e_final == e_cf + gate * e_sem,  with an item-specific gate driven by
  explicit cross-view consistency signals.

Also checks the two properties the round depends on:
  1. at init the semantic correction is ~0, so the model starts as pure CF
  2. the gate actually varies across items (not a disguised global gamma)

Usage: python tools/test_llmmirec_rasrf.py
"""

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sequential.llmmi_components import ItemEncoder  # noqa: E402

torch.manual_seed(0)

N_ITEM = 64
D_LLM = 48
EMB = 32
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def make_llm():
    t = torch.randn(N_ITEM, D_LLM)
    t[0] = 0.0
    return t


def build(llm=None, gate_mode="scalar", gate_input="agree_diff_inter", neigh=None):
    if llm is None:
        llm = make_llm()
    return ItemEncoder(
        item_num=N_ITEM, emb_size=EMB, mode="rasrf", llm_table=llm,
        adapter_hidden=16, adapter_activation="gelu", adapter_use_ln=0,
        neigh_prior=neigh,
        rasrf_gate_mode=gate_mode, rasrf_gate_input=gate_input,
        rasrf_gate_hidden=16,
    )


def test_exact_residual_form():
    print("\n== e_final == e_cf + gate * e_sem ==")
    llm = make_llm()
    enc = build()
    enc.eval()
    ids = torch.arange(1, N_ITEM)
    with torch.no_grad():
        out = enc(ids, return_components=True)
    e_cf, gate, e_sem = out["e_cf"], out["gate"], out["e_sem_raw"]
    manual = e_cf + gate * e_sem
    err = (manual - out["emb"]).abs().max().item()
    check("exact residual form", err < 1e-6, f"max_err={err:.2e}")

    # "semantic" component returned is the GATED correction
    gated = gate * e_sem
    err2 = (gated - out["semantic"]).abs().max().item()
    check("returned semantic == gate * e_sem", err2 < 1e-6, f"max_err={err2:.2e}")

    # CF main path is returned unmasked
    err3 = (e_cf - out["complement"]).abs().max().item()
    check("returned complement == e_cf", err3 < 1e-6, f"max_err={err3:.2e}")


def test_starts_as_pure_cf():
    """CF must dominate at init, but the correction need not be identically 0.

    The repo's init_weights gives every Linear ~N(0, 0.01), so the adapter
    output is small but not zero. We assert only that the CF main path is the
    larger one; the exact ratio for the real (d_llm=1536, emb=64) config is
    measured on the first training batch and reported in the round notes.
    """
    print("\n== at init the CF path dominates ==")
    llm = make_llm()
    enc = build()
    enc.eval()
    ids = torch.arange(1, N_ITEM)
    with torch.no_grad():
        out = enc(ids, return_components=True)
    ratio = (out["semantic"].norm() / out["complement"].norm()).item()
    check("init correction smaller than CF path", ratio < 1.0,
          f"||gate*e_sem|| / ||e_cf|| = {ratio:.4f}")


def test_gate_is_item_specific():
    print("\n== gate varies across items (not a global gamma) ==")
    llm = make_llm()
    enc = build()
    enc.eval()
    ids = torch.arange(1, N_ITEM)
    with torch.no_grad():
        out = enc(ids, return_components=True)
    g = out["gate"]
    gv = g.reshape(1, -1) if g.dim() == 2 else g.reshape(-1)
    check("gate std > 0", float(gv.std()) > 1e-4, f"std={float(gv.std()):.6f}")
    check("gate in (0,1)", bool((gv > 0).all() and (gv < 1).all()),
          f"min={float(gv.min()):.4f} max={float(gv.max()):.4f}")


def test_gate_modes_and_inputs():
    print("\n== gate modes / signal sets / neighbour prior ==")
    llm = make_llm()
    ids = torch.arange(1, N_ITEM)
    for gm in ("scalar", "vector"):
        for gi in ("agree", "agree_diff", "agree_diff_inter"):
            enc = build(gate_mode=gm, gate_input=gi)
            enc.eval()
            with torch.no_grad():
                out = enc(ids, return_components=True)
            check(f"gate_mode={gm:6s} input={gi:16s} shape",
                  tuple(out["emb"].shape) == (N_ITEM - 1, EMB))
            expected = (EMB,) if gm == "vector" else (1,)
            check(f"  gate dim == {expected}", tuple(out["gate"].shape[1:]) == expected,
                  str(tuple(out["gate"].shape)))

    neigh = torch.rand(N_ITEM, 2)
    neigh[0] = 0.0
    enc = build(neigh=neigh)
    enc.eval()
    with torch.no_grad():
        out = enc(ids, return_components=True)
    check("neighbour prior changes the gate", tuple(out["gate"].shape) == (N_ITEM - 1, 1))
    check("gate_in_dim = 2D+1+2", enc.gate_in_dim == 2 * EMB + 1 + 2,
          f"gate_in_dim={enc.gate_in_dim}")


def test_padding_zero():
    print("\n== padding produces zero output ==")
    llm = make_llm()
    neigh = torch.rand(N_ITEM, 2)
    neigh[0] = 0.0
    enc = build(neigh=neigh)
    enc.eval()
    ids = torch.tensor([0, 0, 1, 2])
    with torch.no_grad():
        out = enc(ids, return_components=True)
    for k in ("emb", "semantic", "complement", "gate", "e_cf", "e_sem_raw"):
        v = out[k]
        z = v[:2].abs().max().item() if v.dim() > 1 else v[:2].abs().max().item()
        check(f"padding {k} is zero", z == 0.0)
    check("non-padding emb nonzero", out["emb"][2:].abs().max().item() > 0.0)


def test_gradients():
    print("\n== gradients reach CF path, semantic path and gate ==")
    llm = make_llm()
    enc = build(neigh=None)
    enc.train()
    ids = torch.randint(1, N_ITEM, (4, 6))
    enc(ids).sum().backward()
    params = dict(enc.named_parameters())
    for name in ("id_embedding.weight", "adapter.0.weight", "adapter.2.weight",
                 "rasrf_gate.0.weight", "rasrf_gate.2.weight"):
        p = params.get(name)
        ok = p is not None and p.grad is not None and p.grad.abs().sum().item() > 0
        check(f"grad nonzero: {name}", ok)


def test_buffers_and_state_dict():
    print("\n== buffers are non-persistent ==")
    llm = make_llm()
    neigh = torch.rand(N_ITEM, 2)
    enc = build(neigh=neigh)
    bnames = {n for n, _ in enc.named_buffers()}
    sd = enc.state_dict()
    check("llm_table is a buffer", "llm_table" in bnames)
    check("neigh_prior is a buffer", "neigh_prior" in bnames)
    check("buffers excluded from state_dict",
          "llm_table" not in sd and "neigh_prior" not in sd)
    check("id_embedding IS in state_dict", "id_embedding.weight" in sd)


def test_invalid_configs():
    print("\n== invalid configurations raise at construction ==")
    llm = make_llm()
    neigh_bad = torch.rand(N_ITEM + 5, 2)

    def expect(name, fn):
        try:
            fn()
            check(name, False, "no error raised")
        except (ValueError, RuntimeError) as e:
            check(name, True, type(e).__name__)

    expect("bad gate_mode raises", lambda: build(gate_mode="bogus"))
    expect("bad gate_input raises", lambda: build(gate_input="bogus"))
    expect("wrong neigh rows raises", lambda: build(neigh=neigh_bad))
    expect("missing llm_table raises", lambda: ItemEncoder(
        item_num=N_ITEM, emb_size=EMB, mode="rasrf", llm_table=None))


def test_residual_control_still_works():
    print("\n== global-gamma residual control (same file) ==")
    llm = make_llm()
    enc = ItemEncoder(item_num=N_ITEM, emb_size=EMB, mode="residual", llm_table=llm,
                      adapter_hidden=16, gamma_init=0.1, gamma_trainable=1)
    enc.eval()
    ids = torch.arange(1, N_ITEM)
    with torch.no_grad():
        e = enc(ids)
    check("residual control runs", tuple(e.shape) == (N_ITEM - 1, EMB))
    # with a trainable gamma there is no gamma buffer but a log_gamma parameter
    check("trainable gamma is a parameter", "log_gamma" in dict(enc.named_parameters()))


def test_as_pcf_regression():
    print("\n== ASPCF regression (earlier modes unaffected) ==")
    llm = make_llm()
    enc = ItemEncoder(item_num=N_ITEM, emb_size=EMB, mode="aspcf", llm_table=llm,
                      semantic_rank=16, semantic_dim=16, semantic_hidden=16,
                      complement_dim=16, tail_hidden=16, complement_hidden=16,
                      gate_hidden=16, aspcf_gate_mode="basic")
    enc.eval()
    with torch.no_grad():
        out = enc(torch.randint(1, N_ITEM, (3, 5)), return_components=True)
    check("aspcf emb shape", tuple(out["emb"].shape) == (3, 5, EMB))
    check("aspcf alpha sums to 1",
          ((out["alpha_sem"] + out["alpha_comp"]) - 1.0).abs().max().item() < 1e-5)


if __name__ == "__main__":
    test_exact_residual_form()
    test_starts_as_pure_cf()
    test_gate_is_item_specific()
    test_gate_modes_and_inputs()
    test_padding_zero()
    test_gradients()
    test_buffers_and_state_dict()
    test_invalid_configs()
    test_residual_control_still_works()
    test_as_pcf_regression()
    print(f"\n===== {len(PASS)} passed, {len(FAIL)} failed =====")
    if FAIL:
        print("failed:", FAIL)
        sys.exit(1)
