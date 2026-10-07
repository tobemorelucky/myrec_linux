# -*- coding: UTF-8 -*-
"""
Diagnostics for PPCIM (Chapter 4 Round 1).

Purpose: verify that different candidates really do activate DIFFERENT interests.
This is NOT a "sharper posterior is better" check — a degenerate solution where
every candidate gets the same peaked posterior would look sharp but carry no
candidate-specific information. So the tool reports both the level (entropy) and
the spread ACROSS candidates.

Reports:
  1. positive candidate posterior entropy
  2. negative candidate posterior entropy
  3. positive candidate top-interest histogram
  4. positive vs negative top-interest agreement rate
  5. mean pairwise JS / cosine difference between candidate posteriors
  6. history prior entropy vs candidate posterior entropy

Usage:
  python tools/analyze_ppcim_posterior.py \
      --checkpoint new_model/llmmirec_ppcim/beauty/ppcim_history_marginal/seed42/LLMMIRecPPCIM_seed42.pt \
      --dataset beauty --ppcim_mode marginal --ppcim_prior history
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader                       # noqa: E402
from models.sequential.LLMMIRecPPCIM import LLMMIRecPPCIM     # noqa: E402


def ent(p, axis=-1, eps=1e-12):
    return -(p * np.log(p + eps)).sum(axis=axis)


def js_divergence(p, q, eps=1e-12):
    """JS divergence between two probability vectors."""
    m = 0.5 * (p + q)
    return 0.5 * ((p * np.log((p + eps) / (m + eps))).sum()
                  + (q * np.log((q + eps) / (m + eps))).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--dataset", type=str, default="beauty")
    ap.add_argument("--phase", type=str, default="test", choices=["train", "dev", "test"])
    ap.add_argument("--max_batches", type=int, default=30)
    ap.add_argument("--pair_samples", type=int, default=64,
                    help="candidate pairs sampled per batch for the JS estimate")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output_dir", type=str, default="./diagnostics_ppcim")
    # model config (defaults match the frozen Beauty ASPCF config)
    ap.add_argument("--emb_size", type=int, default=64)
    ap.add_argument("--attn_size", type=int, default=64)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--history_max", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--semantic_rank", type=int, default=512)
    ap.add_argument("--semantic_dim", type=int, default=32)
    ap.add_argument("--complement_dim", type=int, default=32)
    ap.add_argument("--ppcim_mode", type=str, default="marginal")
    ap.add_argument("--ppcim_prior", type=str, default="history")
    ap.add_argument("--ppcim_tau", type=float, default=1.0)
    ap.add_argument("--llm_emb_path", type=str, default="")
    args = ap.parse_args()

    ds_name = args.dataset
    a = args
    a.llm_emb_path = a.llm_emb_path or f"./data/{ds_name}/handled/llm_table_pca1536.pkl"
    a.device = torch.device(a.device)
    a.model_path = a.checkpoint
    a.buffer = 1; a.num_neg = 1; a.test_all = 0; a.dropout = 0.1
    a.item_encoder = "aspcf"
    a.adapter_hidden = 256; a.adapter_activation = "gelu"; a.adapter_use_ln = 0
    a.gamma_init = 0.1; a.gamma_trainable = 0
    a.semantic_hidden = 128; a.tail_hidden = 64
    a.complement_hidden = 64; a.gate_hidden = 64; a.aspcf_gate_mode = "basic"
    a.lambda_relation = 0.01; a.relation_sample_size = 128
    a.relation_teacher_temp = 0.1; a.relation_student_temp = 0.1
    a.ppcim_eps = 1e-8

    corpus = pickle.load(open(f"./data/{ds_name}/SeqReader.pkl", "rb"))
    model = LLMMIRecPPCIM(a, corpus)
    st = torch.load(a.checkpoint, map_location="cpu")
    miss, unexp = model.load_state_dict(st, strict=False)
    if miss or unexp:
        print(f"[warn] missing={len(miss)} unexpected={len(unexp)}")
    model = model.to(a.device); model.device = a.device; model.eval()

    dset = model.Dataset(model, corpus, a.phase); dset.prepare()
    dl = DataLoader(dset, batch_size=a.batch_size, shuffle=False, num_workers=0,
                    collate_fn=dset.collate_batch)

    pos_ent, neg_ent, pri_ent = [], [], []
    top_pos = np.zeros(a.K, dtype=np.int64)
    top_neg = np.zeros(a.K, dtype=np.int64)
    agree, disagree = 0, 0
    js_vals, cos_vals = [], []
    rng = np.random.default_rng(0)

    n = 0
    with torch.inference_mode():
        for batch in dl:
            batch = {k: v.to(a.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model(batch, return_intermediate=True)
            pi = out.get("candidate_interest_posterior")
            if pi is None:
                raise RuntimeError(
                    "no posterior returned; run with ppcim_mode != baseline")
            p_prior = out["history_interest_prior"]
            # pi: [B, K, C] -> [B, C, K]
            pi_ck = pi.detach().float().cpu().numpy().transpose(0, 2, 1)
            pri = p_prior.detach().float().cpu().numpy()               # [B,K]

            for b in range(pi_ck.shape[0]):
                P = pi_ck[b]                                           # [C,K]
                pos_ent.append(ent(P[0]))
                neg_ent.append(ent(P[1:]))
                pri_ent.append(ent(pri[b]))
                top_pos[int(P[0].argmax())] += 1
                tn = P[1:].argmax(axis=1)
                for t in tn:
                    top_neg[int(t)] += 1
                agree += int((tn == int(P[0].argmax())).sum())
                disagree += int((tn != int(P[0].argmax())).sum())

                C = P.shape[0]
                if C >= 2:
                    idx = rng.choice(C, size=min(args.pair_samples, C), replace=False)
                    Q = P[idx]
                    for i1 in range(0, len(idx), 8):
                        for i2 in range(i1 + 1, min(i1 + 8, len(idx))):
                            js_vals.append(js_divergence(Q[i1], Q[i2]))
                            cos_vals.append(
                                float(Q[i1] @ Q[i2] /
                                      (np.linalg.norm(Q[i1]) * np.linalg.norm(Q[i2]) + 1e-12)))
            n += 1
            if args.max_batches and n >= args.max_batches:
                break

    neg_ent_flat = np.concatenate(neg_ent) if neg_ent else np.array([0.0])
    stats = {
        "num_users": int(len(pos_ent)),
        "K": int(a.K),
        "max_entropy": float(np.log(a.K)),
        "pos_posterior_entropy_mean": float(np.mean(pos_ent)),
        "neg_posterior_entropy_mean": float(neg_ent_flat.mean()),
        "neg_posterior_entropy_std": float(neg_ent_flat.std()),
        "prior_entropy_mean": float(np.mean(pri_ent)),
        "pos_top_interest_hist": top_pos.tolist(),
        "neg_top_interest_hist": top_neg.tolist(),
        "pos_top_interest_frac": (top_pos / max(pos_ent.__len__(), 1)).tolist(),
        "neg_top_interest_frac": (top_neg / max(neg_ent_flat.size, 1)).tolist(),
        "pos_neg_top_agree_frac": float(agree / max(agree + disagree, 1)),
        "candidate_pair_js_mean": float(np.mean(js_vals)) if js_vals else float("nan"),
        "candidate_pair_cos_mean": float(np.mean(cos_vals)) if cos_vals else float("nan"),
        "n_pairs": int(len(js_vals)),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"{ds_name}_{a.ppcim_mode}_{a.ppcim_prior}_tau{a.ppcim_tau}"
    path = os.path.join(args.output_dir, f"{tag}.json")
    json.dump(stats, open(path, "w"), indent=2)

    print(json.dumps(stats, indent=2))
    print(f"\n[saved] {path}")
    print("\n判读要点：")
    print("  - candidate_pair_js_mean 明显 > 0 且 candidate_pair_cos_mean < 1")
    print("    => 不同 candidate 确实激活不同 interest（这是本轮要验证的）")
    print("  - neg_posterior_entropy_mean 远低于 prior_entropy_mean")
    print("    => posterior 比 prior 更尖锐，说明 matching 提供了信息")
    print("  - pos_neg_top_agree_frac ≈ 1/K 说明正负样本无系统性偏好")
    print("    （若远大于 1/K，需检查是否引入了 leakage）")


if __name__ == "__main__":
    main()
