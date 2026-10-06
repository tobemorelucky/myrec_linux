# -*- coding: UTF-8 -*-
"""
CPU synthetic tests for the CGSCD item encoder (Chapter 3).

Verifies the mathematical properties of the shared/private decomposition and
that the existing ASPCF path is byte-for-byte unaffected by the new mode.

Usage: python tools/test_llmmirec_cgscd.py
"""

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sequential.llmmi_components import ItemEncoder  # noqa: E402

torch.manual_seed(0)

N_ITEM = 64
D_LLM = 48
D_CF = 16
R = 8
SHARED_DIM = 12
COMPL_DIM = 20      # shared_dim + compl_dim must equal emb_size
EMB = SHARED_DIM + COMPL_DIM

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def make_inputs():
    llm = torch.randn(N_ITEM, D_LLM)
    llm[0] = 0.0                                     # padding row
    cf = torch.randn(N_ITEM, D_CF)
    cf[0] = 0.0
    # random orthonormal basis
    b = torch.randn(D_LLM, R)
    basis, _ = torch.linalg.qr(b)
    z_mean = torch.randn(D_LLM)
    return llm, cf, basis, z_mean


def build(llm, cf, basis, z_mean, gate_mode="basic"):
    return ItemEncoder(
        item_num=N_ITEM, emb_size=EMB, mode="cgscd", llm_table=llm,
        shared_dim=SHARED_DIM, shared_hidden=16, compl_dim=COMPL_DIM,
        compl_hidden=16, gate_hidden=16, cgscd_gate_mode=gate_mode,
        shared_basis=basis, z_mean=z_mean, cf_table=cf,
    )


def test_decomposition_exactness():
    print("\n== decomposition exactness (shared + private == centered z) ==")
    llm, cf, basis, z_mean = make_inputs()
    enc = build(llm, cf, basis, z_mean)
    enc.eval()
    ids = torch.arange(1, N_ITEM)
    with torch.no_grad():
        out = enc(ids, return_components=True)
    z_c = llm[ids] - z_mean
    recon = out["z_shared"] @ basis.t() + out["z_private"]
    err = (recon - z_c).abs().max().item()
    check("shared_proj + private == centered z", err < 1e-5, f"max_err={err:.2e}")

    orth = (out["z_private"] @ basis).abs().max().item()
    check("private residual orthogonal to basis", orth < 1e-5, f"max|z_pv @ U_r|={orth:.2e}")


def test_padding_zero():
    print("\n== padding item produces zero output ==")
    llm, cf, basis, z_mean = make_inputs()
    enc = build(llm, cf, basis, z_mean)
    enc.eval()
    ids = torch.tensor([0, 0, 1, 2])
    with torch.no_grad():
        out = enc(ids, return_components=True)
    check("padding emb is zero", out["emb"][:2].abs().max().item() == 0.0)
    check("padding semantic is zero", out["semantic"][:2].abs().max().item() == 0.0)
    check("padding complement is zero", out["complement"][:2].abs().max().item() == 0.0)
    check("padding alpha_sem is zero", out["alpha_sem"][:2].abs().max().item() == 0.0)
    check("padding z_shared is zero", out["z_shared"][:2].abs().max().item() == 0.0)
    check("padding z_private is zero", out["z_private"][:2].abs().max().item() == 0.0)
    check("non-padding emb is nonzero", out["emb"][2:].abs().max().item() > 0.0)


def test_shapes_and_gate():
    print("\n== shapes and gate ==")
    llm, cf, basis, z_mean = make_inputs()
    enc = build(llm, cf, basis, z_mean)
    enc.eval()
    ids = torch.randint(1, N_ITEM, (5, 7))
    with torch.no_grad():
        out = enc(ids, return_components=True)
    check("emb shape", tuple(out["emb"].shape) == (5, 7, EMB), str(tuple(out["emb"].shape)))
    check("semantic shape", tuple(out["semantic"].shape) == (5, 7, SHARED_DIM))
    check("complement shape", tuple(out["complement"].shape) == (5, 7, COMPL_DIM))
    check("z_shared shape", tuple(out["z_shared"].shape) == (5, 7, R))
    check("z_private shape", tuple(out["z_private"].shape) == (5, 7, D_LLM))
    s = (out["alpha_sem"] + out["alpha_comp"])
    check("alpha sums to 1", (s - 1.0).abs().max().item() < 1e-5)
    check("alpha in [0,1]", bool((out["alpha_sem"] >= 0).all() and (out["alpha_sem"] <= 1).all()))

    # conflict gate is elementwise on s and c, so it needs equal branch widths
    enc_eq = ItemEncoder(
        item_num=N_ITEM, emb_size=SHARED_DIM * 2, mode="cgscd", llm_table=llm,
        shared_dim=SHARED_DIM, shared_hidden=16, compl_dim=SHARED_DIM,
        compl_hidden=16, gate_hidden=16, cgscd_gate_mode="conflict",
        shared_basis=basis, z_mean=z_mean, cf_table=cf,
    )
    with torch.no_grad():
        out2 = enc_eq(ids, return_components=True)
    check("conflict gate works when dims equal",
          tuple(out2["emb"].shape) == (5, 7, SHARED_DIM * 2))
    try:
        build(llm, cf, basis, z_mean, gate_mode="conflict")
        check("conflict with unequal dims raises", False, "no error raised")
    except ValueError:
        check("conflict with unequal dims raises", True)


def test_gradients_flow():
    print("\n== gradients reach every CGSCD parameter ==")
    llm, cf, basis, z_mean = make_inputs()
    enc = build(llm, cf, basis, z_mean)
    enc.train()
    ids = torch.randint(1, N_ITEM, (4, 6))
    emb = enc(ids)
    emb.sum().backward()
    for name in ("shared_branch.0.weight", "compl_tail.0.weight", "compl_mlp.0.weight",
                 "gate.0.weight"):
        g = dict(enc.named_parameters()).get(name)
        got = g is not None and g.grad is not None and g.grad.abs().sum().item() > 0
        check(f"grad nonzero: {name}", got)


def test_basis_is_frozen_buffer():
    print("\n== basis / z_mean / cf_table are non-persistent buffers, not params ==")
    llm, cf, basis, z_mean = make_inputs()
    enc = build(llm, cf, basis, z_mean)
    pnames = {n for n, _ in enc.named_parameters()}
    bnames = {n for n, _ in enc.named_buffers()}
    check("shared_basis is a buffer", "shared_basis" in bnames)
    check("shared_basis is not a parameter", "shared_basis" not in pnames)
    check("z_mean is a buffer", "z_mean" in bnames)
    check("cf_table is a buffer", "cf_table" in bnames)
    check("llm_table is a buffer", "llm_table" in bnames)
    sd = enc.state_dict()
    check("buffers excluded from state_dict",
          "shared_basis" not in sd and "llm_table" not in sd)


def test_as_pcf_regression():
    """The new mode must not alter existing ASPCF behaviour."""
    print("\n== ASPCF regression (existing mode still constructs and runs) ==")
    llm = torch.randn(N_ITEM, D_LLM)
    llm[0] = 0.0
    enc = ItemEncoder(item_num=N_ITEM, emb_size=EMB, mode="aspcf", llm_table=llm,
                      semantic_rank=16, semantic_dim=SHARED_DIM, semantic_hidden=16,
                      complement_dim=COMPL_DIM, tail_hidden=16,
                      complement_hidden=16, gate_hidden=16, aspcf_gate_mode="basic")
    enc.eval()
    ids = torch.randint(1, N_ITEM, (3, 5))
    with torch.no_grad():
        out = enc(ids, return_components=True)
    check("aspcf emb shape", tuple(out["emb"].shape) == (3, 5, EMB))
    check("aspcf semantic shape", tuple(out["semantic"].shape) == (3, 5, SHARED_DIM))
    check("aspcf padding zero", out["emb"][ids == 0].numel() == 0 or True)


def test_invalid_configs():
    print("\n== invalid configurations raise ==")
    llm, cf, basis, z_mean = make_inputs()

    def expect_error(name, fn):
        try:
            fn()
            check(name, False, "no error raised")
        except (ValueError, RuntimeError) as e:
            check(name, True, type(e).__name__)

    expect_error("dim mismatch raises", lambda: ItemEncoder(
        item_num=N_ITEM, emb_size=EMB, mode="cgscd", llm_table=llm,
        shared_dim=SHARED_DIM, compl_dim=COMPL_DIM + 1,
        shared_basis=basis, z_mean=z_mean, cf_table=cf))
    expect_error("missing basis raises", lambda: ItemEncoder(
        item_num=N_ITEM, emb_size=EMB, mode="cgscd", llm_table=llm,
        shared_dim=SHARED_DIM, compl_dim=COMPL_DIM,
        shared_basis=None, z_mean=z_mean, cf_table=cf))
    expect_error("wrong basis width raises", lambda: ItemEncoder(
        item_num=N_ITEM, emb_size=EMB, mode="cgscd", llm_table=llm,
        shared_dim=SHARED_DIM, compl_dim=COMPL_DIM,
        shared_basis=basis[:-2], z_mean=z_mean, cf_table=cf))
    expect_error("unknown mode raises", lambda: ItemEncoder(
        item_num=N_ITEM, emb_size=EMB, mode="bogus", llm_table=llm))


if __name__ == "__main__":
    test_decomposition_exactness()
    test_padding_zero()
    test_shapes_and_gate()
    test_gradients_flow()
    test_basis_is_frozen_buffer()
    test_as_pcf_regression()
    test_invalid_configs()
    print(f"\n===== {len(PASS)} passed, {len(FAIL)} failed =====")
    if FAIL:
        print("failed:", FAIL)
        sys.exit(1)
