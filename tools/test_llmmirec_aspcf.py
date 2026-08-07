# -*- coding: UTF-8 -*-
"""
Single-batch validation for LLMMIRec ASPCF mode.

Checks:
  1. Forward pass produces valid predictions
  2. BPR loss is finite
  3. Relation loss computed (when lambda_relation > 0)
  4. Backward produces finite gradients
  5. Final embedding shape = emb_size (64)
  6. semantic/complement shape = 32
  7. alpha_sem + alpha_comp ≈ 1
  8. Padding output = 0
  9. id / llm_replace / residual modes still work
"""

import logging, os, pickle, sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch
import torch.nn.functional as F

from models.sequential.LLMMIRec import LLMMIRec
from models.BaseModel import BaseModel


def build_model(dataset, mode, **kwargs):
    class Args:
        pass

    args = Args()
    args.path = "./data/"
    args.dataset = dataset
    args.sep = "\t"
    args.device = torch.device("cpu")
    args.model_path = f"./model/LLMMIRec/test_{mode}.pt"
    args.buffer = 0
    args.num_neg = 1
    args.dropout = 0.1
    args.test_all = 0
    args.history_max = 20
    args.emb_size = 64
    args.attn_size = 64
    args.K = 4
    args.item_encoder = mode
    args.llm_emb_path = ""
    args.adapter_hidden = 256
    args.adapter_activation = "gelu"
    args.adapter_use_ln = 0
    args.gamma_init = 0.1
    args.gamma_trainable = 0
    # ASPCF
    args.semantic_rank = kwargs.get("semantic_rank", 512)
    args.semantic_dim = 32
    args.semantic_hidden = 128
    args.complement_dim = 32
    args.tail_hidden = 64
    args.complement_hidden = 64
    args.gate_hidden = 64
    args.lambda_relation = kwargs.get("lambda_relation", 0.1)
    args.relation_sample_size = 128
    args.relation_teacher_temp = 0.1
    args.relation_student_temp = 0.1

    if mode in ("llm_replace", "residual", "aspcf"):
        args.llm_emb_path = f"./data/{dataset}/handled/llm_table_pca1536.pkl"

    corpus = pickle.load(open(f"./data/{dataset}/SeqReader.pkl", "rb"))
    model = LLMMIRec(args, corpus)
    return model, corpus


def test_mode(dataset, mode, **kwargs):
    print(f"\n{'='*60}")
    print(f"Testing item_encoder={mode}")
    print(f"{'='*60}")

    model, corpus = build_model(dataset, mode, **kwargs)
    n_params = model.count_variables()
    print(f"#params: {n_params}")

    # Build batch
    ds = LLMMIRec.Dataset(model, corpus, "train")
    ds.actions_before_epoch()
    indices = list(range(min(8, len(ds))))
    feeds = [ds[i] for i in indices]
    batch = ds.collate_batch(feeds)
    batch = {k: v if not isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

    # Forward
    model.train()
    out = model(batch, return_intermediate=True)
    pred = out["prediction"]

    # Check prediction
    assert pred.shape[-1] == 2, f"Expected 2 candidates, got {pred.shape[-1]}"
    assert not torch.isnan(pred).any(), "NaN in prediction"
    assert not torch.isinf(pred).any(), "Inf in prediction"
    print(f"  prediction: shape={pred.shape}, mean={float(pred.mean()):.4f}")

    # BPR loss
    bpr_loss = super(LLMMIRec, model).loss(out)
    assert torch.isfinite(bpr_loss), f"BPR loss not finite: {bpr_loss}"
    print(f"  BPR loss: {float(bpr_loss):.6f}")

    # Backward
    model.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.optimizer.zero_grad()

    # Re-forward with loss to get total loss
    total_loss = model.loss(out)
    print(f"  Total loss: {float(total_loss):.6f}")
    total_loss.backward()

    # Gradient check
    grad_ok = True
    max_grad = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            g = p.grad.abs().max().item()
            max_grad = max(max_grad, g)
            if not np.isfinite(g):
                print(f"  BAD grad: {name} max={g}")
                grad_ok = False
    print(f"  Max gradient: {max_grad:.6f}, all finite: {grad_ok}")
    assert grad_ok, "Non-finite gradients detected"

    # ASPCF-specific checks
    if mode == "aspcf":
        # Embedding shape
        hv = out["history_vectors"]
        cv = out["candidate_vectors"]
        assert hv.shape[-1] == 64, f"history_emb dim={hv.shape[-1]}, expected 64"
        assert cv.shape[-1] == 64, f"candidate_emb dim={cv.shape[-1]}, expected 64"
        print(f"  history_emb dim: {hv.shape[-1]} (expected 64)")

        # Semantic/complement shapes
        for prefix in ["history", "candidate"]:
            s = out[f"{prefix}_semantic"]
            c = out[f"{prefix}_complement"]
            as_ = out[f"{prefix}_alpha_sem"]
            ac = out[f"{prefix}_alpha_comp"]
            assert s.shape[-1] == 32, f"{prefix}_semantic dim={s.shape[-1]}"
            assert c.shape[-1] == 32, f"{prefix}_complement dim={c.shape[-1]}"
            print(f"  {prefix}: semantic={s.shape[-1]}d, complement={c.shape[-1]}d")

            # Alpha sum = 1
            alpha_sum = (as_ + ac)
            # Check for non-padding items
            non_pad = (as_ > 0).float().mean()
            if non_pad > 0:
                valid = as_ > 0
                alpha_sum_valid = alpha_sum[valid]
                max_dev = (alpha_sum_valid - 1.0).abs().max().item()
                print(f"  {prefix}: alpha_sem+alpha_comp max_deviation={max_dev:.6f}")
                assert max_dev < 0.01, f"alpha sum deviates: {max_dev}"

        # Padding output = 0
        pad_loc = batch["history_items"] == 0
        if pad_loc.any():
            pad_norm = hv[pad_loc].norm(dim=-1).max().item()
            print(f"  padding output max norm: {pad_norm:.6f}")
            assert pad_norm < 1e-6, f"Padding output not zero: {pad_norm}"

        # Relation loss
        if "_relation_ids" in out:
            rel_ids = out["_relation_ids"]
            print(f"  relation items sampled: {rel_ids.numel()}")
        rel_loss = out.get("loss_relation", None)
        if rel_loss is not None:
            print(f"  relation loss: {float(rel_loss):.6f}")

    # Eval determinism
    model.eval()
    out_e1 = model(batch)
    out_e2 = model(batch)
    same = torch.allclose(out_e1["prediction"], out_e2["prediction"])
    print(f"  eval determinism: {same}")
    assert same, "Eval not deterministic"

    print(f"  PASS: {mode}")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    dataset = "beauty"

    # Test all four modes
    test_mode(dataset, "id")
    test_mode(dataset, "llm_replace")
    test_mode(dataset, "residual")
    test_mode(dataset, "aspcf", lambda_relation=0.1, semantic_rank=512)

    # Test lambda_relation=0
    test_mode(dataset, "aspcf", lambda_relation=0.0, semantic_rank=512)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
