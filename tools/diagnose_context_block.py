# -*- coding: UTF-8 -*-
"""
Chapter 4 Phase 3 — contextualization diagnostics (inference only).

Purpose: prove the self-attention block is NOT learning an identity / self-copy
map, i.e. that genuine cross-position information flow happens.

Reports:
  A. contextual shift        ||H_ctx − H|| / ||H||
  B. self-attention entropy  H(A) per (query, head), averaged
  C. diagonal vs off-diagonal attention mass
  D. head-to-head attention similarity
  E. contextual shift grouped by history length

Usage:
  python tools/diagnose_context_block.py --config self_attn
  python tools/diagnose_context_block.py --config ffn_control
"""

import argparse
import json
import math
import os
import pickle
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader                                    # noqa: E402
from models.sequential.LLMMIRecContextControl import LLMMIRecContextControl  # noqa: E402


def build(ckpt, mode, corpus, device):
    a = argparse.Namespace()
    a.emb_size = 64; a.attn_size = 64; a.K = 4; a.history_max = 20
    a.num_neg = 1; a.test_all = 0; a.buffer = 1; a.dropout = 0.1
    a.item_encoder = "aspcf"
    a.llm_emb_path = "./data/beauty/handled/llm_table_pca1536.pkl"
    a.semantic_rank = 512; a.semantic_dim = 32; a.semantic_hidden = 128
    a.complement_dim = 32; a.tail_hidden = 64; a.complement_hidden = 64
    a.gate_hidden = 64; a.aspcf_gate_mode = "basic"
    a.adapter_hidden = 256; a.adapter_activation = "gelu"; a.adapter_use_ln = 0
    a.gamma_init = 0.1; a.gamma_trainable = 0
    a.lambda_relation = 0.01; a.relation_sample_size = 128
    a.relation_teacher_temp = 0.1; a.relation_student_temp = 0.1
    a.context_mode = mode; a.context_dropout = 0.1; a.context_heads = 4
    a.context_ffn_hidden = 128; a.context_ffn_control_hidden = 256
    a.device = device; a.path = "./data/"; a.model_path = ckpt
    m = LLMMIRecContextControl(a, corpus)
    miss, unexp = m.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    if miss or unexp:
        print(f"[warn] missing={len(miss)} unexpected={len(unexp)}")
    m = m.to(device); m.device = device; m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True,
                    choices=["self_attn", "ffn_control", "baseline"])
    ap.add_argument("--dataset", type=str, default="beauty")
    ap.add_argument("--phase", type=str, default="test")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--output_dir", type=str, default="./diagnostics_context_block")
    args = ap.parse_args()

    ck = args.ckpt or (f"new_model/llmmirec_context/{args.dataset}/{args.config}/"
                       f"seed42/LLMMIRecContextControl_seed42.pt")
    device = torch.device(args.device)
    corpus = pickle.load(open(f"./data/{args.dataset}/SeqReader.pkl", "rb"))
    model = build(ck, args.config, corpus, device)

    dset = model.Dataset(model, corpus, args.phase); dset.prepare()
    dl = DataLoader(dset, batch_size=args.batch_size, shuffle=False,
                    num_workers=0, collate_fn=dset.collate_batch)

    shift, ent, diag_m, off_m, hsim = [], [], [], [], []
    shift_by_len = {}
    n = 0
    with torch.inference_mode():
        for batch in dl:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            out = model(batch, return_intermediate=True)
            pre = out["history_pre_context"]                 # [B,L,D]
            post = out["history_post_context"]
            lengths = batch["lengths"]
            L = pre.size(1)
            valid = (torch.arange(L, device=device)[None, :] < lengths[:, None])

            # ---- A. contextual shift (valid positions only) ----
            num = (post - pre).norm(dim=-1)                  # [B,L]
            den = pre.norm(dim=-1).clamp(min=1e-12)
            sh = (num / den)[valid]                          # [n_valid]
            shift.append(sh.cpu().numpy())

            # ---- E. shift by history length ----
            for b in range(pre.size(0)):
                ln = int(lengths[b])
                if ln == 0:
                    continue
                vb = valid[b]
                shift_by_len.setdefault(ln, []).append(
                    float((num[b][vb] / den[b][vb]).mean()))

            A = out["context_attention_maps"]
            if A is not None:
                # A: [B, h, Lq, Lk]
                # ---- B. attention entropy over VALID keys ----
                eps = 1e-12
                a_clamped = A.clamp(min=eps)
                ent_all = -(A * a_clamped.log()).sum(dim=-1)         # [B,h,Lq]
                # only rows that have >=2 valid keys
                has2 = (lengths[:, None] >= 2).expand(-1, L)
                ent.append(ent_all.permute(0, 2, 1)[has2].cpu().numpy().ravel())

                # ---- C. diagonal vs off-diagonal mass ----
                # diagonal = query i attends to key i (self-copy)
                idx = torch.arange(L, device=device)
                eye = (idx[None, :, None] == idx[None, None, :])[0]  # [Lq,Lk]
                d_mass = (A * eye[None, None].to(A.dtype)).sum(dim=-1)  # [B,h,Lq]
                # off-diagonal, restricted to valid keys
                vk = valid[:, None, None, :].to(A.dtype)
                off = (A * vk).sum(dim=-1) - d_mass                  # valid mass minus diag
                diag_m.append(d_mass.permute(0, 2, 1)[has2].cpu().numpy().ravel())
                off_m.append(off.permute(0, 2, 1)[has2].cpu().numpy().ravel())

                # ---- D. head-to-head attention similarity ----
                # per (batch, query) flatten head attention -> cosine between heads
                Af = A.reshape(A.size(0), A.size(1), -1)              # [B,h,Lq*Lk]
                An = Af / Af.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                sim = torch.einsum("bhp,bgp->bhg", An, An)            # [B,h,g]
                iu = torch.triu_indices(A.size(1), A.size(1), offset=1)
                hsim.append(sim[:, iu[0], iu[1]].cpu().numpy().ravel())

            n += 1
            if args.max_batches and n >= args.max_batches:
                break

    cat = lambda L_: np.concatenate(L_) if L_ else np.array([0.0])
    sh = cat(shift)
    res = {
        "config": args.config, "checkpoint": ck, "phase": args.phase,
        "n_users": int(sh.size),
        "A_contextual_shift": {
            "mean": float(sh.mean()), "std": float(sh.std()),
            "p05": float(np.percentile(sh, 5)), "p50": float(np.percentile(sh, 50)),
            "p95": float(np.percentile(sh, 95)),
        },
    }

    if ent:
        e = cat(ent); dm = cat(diag_m); om = cat(off_m); hs = cat(hsim)
        res["B_attention_entropy"] = {
            "mean": float(e.mean()), "p05": float(np.percentile(e, 5)),
            "p50": float(np.percentile(e, 50)), "p95": float(np.percentile(e, 95)),
            "max_possible_ln_L": float(math.log(20)),
        }
        res["C_diagonal_vs_offdiagonal"] = {
            "diag_mass_mean": float(dm.mean()),
            "offdiag_mass_mean": float(om.mean()),
            "diag_share": float(dm.mean() / (dm.mean() + om.mean() + 1e-12)),
        }
        res["D_head_similarity"] = {
            "mean": float(hs.mean()), "p05": float(np.percentile(hs, 5)),
            "p50": float(np.percentile(hs, 50)), "p95": float(np.percentile(hs, 95)),
        }
    res["E_shift_by_history_length"] = {
        str(k): float(np.mean(v)) for k, v in sorted(shift_by_len.items())}

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f"{args.dataset}_{args.config}.json")
    json.dump(res, open(path, "w"), indent=2)

    print("=" * 96)
    print(f"Contextualization diagnostics — {args.config} ({args.phase})")
    print("=" * 96)
    a_ = res["A_contextual_shift"]
    print(f"A. contextual shift ||H_ctx−H||/||H|| : mean={a_['mean']:.4f} "
          f"p05={a_['p05']:.4f} p50={a_['p50']:.4f} p95={a_['p95']:.4f}")
    if ent:
        b_ = res["B_attention_entropy"]
        print(f"B. attention entropy                  : mean={b_['mean']:.4f} "
              f"p05={b_['p05']:.4f} p50={b_['p50']:.4f} p95={b_['p95']:.4f} "
              f"(ln20={b_['max_possible_ln_L']:.4f})")
        c_ = res["C_diagonal_vs_offdiagonal"]
        print(f"C. diagonal mass={c_['diag_mass_mean']:.4f}  "
              f"off-diagonal mass={c_['offdiag_mass_mean']:.4f}  "
              f"diag share={c_['diag_share']:.4f}")
        d_ = res["D_head_similarity"]
        print(f"D. head-to-head attention similarity  : mean={d_['mean']:.4f} "
              f"p05={d_['p05']:.4f} p50={d_['p50']:.4f} p95={d_['p95']:.4f}")
    print("E. contextual shift by history length:")
    for k, v in res["E_shift_by_history_length"].items():
        print(f"     len={k:>3s}  shift={v:.4f}")
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
