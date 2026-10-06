# -*- coding: UTF-8 -*-
"""
Build the offline local-neighborhood agreement prior for Chapter 3 Round 2
(Reliability-Aware Semantic Residual Fusion).

Motivation: a global consistency scalar (cosine between the two views of the
SAME item) says nothing about whether the two views organise the item's
NEIGHBOURHOOD the same way. An item can have a high self-cosine yet sit in a
region where the LLM and collaborative views disagree about who its neighbours
are. The gate needs that local signal.

Two per-item scalars, both in [-1, 1] before rescaling:

  agree_llm_to_cf[i] = mean over j in N_llm(i) of  cos_cf(i, j)
      "do this item's LLM-neighbours also look similar in collaborative space?"
  agree_cf_to_llm[i] = mean over j in N_cf(i) of  cos_llm(i, j)
      "do this item's collaborative-neighbours also look similar in LLM space?"

N_llm is taken from the ALREADY-BUILT `semantic_hardneg_top100.pkl`
(row i = top-100 LLM-space neighbours, self excluded, row 0 = padding) so no
new LLM-space search is needed. Only the collaborative-space top-k search is
computed here (d_cf = 64, cheap).

Output: data/<dataset>/handled/semantic_neighborhood_agreement_k<k>.pkl
  {'agreement': float32 [n_items, 2],   # row 0 = zeros (padding)
   'k': int, 'dataset': str,
   'source': {'llm_path','llm_sha1','cf_path','cf_sha1','bank_path','bank_sha1'},
   'stats': {...}}

Usage:
  python tools/build_semantic_neighborhood_agreement.py --dataset beauty --k 20
"""

import argparse
import hashlib
import os
import pickle

import numpy as np


def sha1_of(path, chunk=1 << 22):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def l2n(x, eps=1e-12):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def cf_topk_neighbors(cf, k, chunk=2048):
    """Top-k collaborative-space neighbours per item (self excluded).

    Returns int64 [N, k] of indices into `cf`.
    """
    N = cf.shape[0]
    k = min(k, N - 1)
    out = np.zeros((N, k), dtype=np.int64)
    for i in range(0, N, chunk):
        end = min(i + chunk, N)
        sim = cf[i:end] @ cf.T                      # (C, N)
        for j in range(end - i):
            sim[j, i + j] = -np.inf                 # exclude self
        out[i:end] = np.argpartition(-sim, k, axis=1)[:, :k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--k", type=int, default=20, help="neighbourhood size")
    ap.add_argument("--llm_emb_path", type=str, default="")
    ap.add_argument("--cf_emb_path", type=str, default="")
    ap.add_argument("--bank_path", type=str, default="")
    ap.add_argument("--chunk", type=int, default=2048)
    args = ap.parse_args()

    ds = args.dataset
    llm_path = args.llm_emb_path or f"./data/{ds}/handled/llm_table_pca1536.pkl"
    cf_path = args.cf_emb_path or f"./data/{ds}/handled/itm_emb_pomrec.pkl"
    bank_path = args.bank_path or f"./data/{ds}/handled/semantic_hardneg_top100.pkl"
    out_path = f"./data/{ds}/handled/semantic_neighborhood_agreement_k{args.k}.pkl"

    for p in (llm_path, cf_path, bank_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing input: {p}")

    Z = np.asarray(pickle.load(open(llm_path, "rb")), dtype=np.float32)
    C = np.asarray(pickle.load(open(cf_path, "rb")), dtype=np.float32)
    bank = np.asarray(pickle.load(open(bank_path, "rb")), dtype=np.int64)

    # Align: Z row 0 is padding; C has no padding row.
    if Z.shape[0] == C.shape[0] + 1:
        if not np.allclose(Z[0], 0.0):
            raise ValueError("Z has a leading row but it is not zero padding")
        Zv = Z[1:]
    elif Z.shape[0] == C.shape[0]:
        Zv = Z
    else:
        raise ValueError(f"row mismatch: Z {Z.shape[0]} vs C {C.shape[0]}")

    n_items = Z.shape[0]                 # includes padding row 0
    N = Zv.shape[0]
    k = min(args.k, bank.shape[1], N - 1)
    print(f"[nbr-agree] {ds}: Z{Z.shape} C{C.shape} bank{bank.shape}  k={k}")

    Zn, Cn = l2n(Zv), l2n(C)
    Zn_t = np.zeros((n_items, Zn.shape[1]), dtype=np.float32)
    Zn_t[1:] = Zn
    Cn_t = np.zeros((n_items, Cn.shape[1]), dtype=np.float32)
    Cn_t[1:] = Cn

    # ---- direction 1: LLM neighbours, measured in collaborative space ----
    # bank rows are item ids (1-based, row 0 unused); take the first k columns
    ids = np.arange(1, n_items)
    llm_nbr = bank[1:, :k]                                  # [N, k] item ids
    a1 = (Cn_t[ids][:, None, :] * Cn_t[llm_nbr]).sum(-1).mean(-1)   # [N]

    # ---- direction 2: collaborative neighbours, measured in LLM space ----
    print(f"[nbr-agree] searching CF-space neighbours (N={N}) ...")
    cf_nbr = cf_topk_neighbors(Cn, k, chunk=args.chunk) + 1  # -> item ids
    a2 = (Zn_t[ids][:, None, :] * Zn_t[cf_nbr]).sum(-1).mean(-1)   # [N]

    agreement = np.zeros((n_items, 2), dtype=np.float32)
    agreement[1:, 0] = a1
    agreement[1:, 1] = a2

    stats = {
        "a1_llm2cf_mean": float(a1.mean()), "a1_llm2cf_std": float(a1.std()),
        "a1_p05": float(np.percentile(a1, 5)), "a1_p50": float(np.percentile(a1, 50)),
        "a1_p95": float(np.percentile(a1, 95)),
        "a2_cf2llm_mean": float(a2.mean()), "a2_cf2llm_std": float(a2.std()),
        "a2_p05": float(np.percentile(a2, 5)), "a2_p50": float(np.percentile(a2, 50)),
        "a2_p95": float(np.percentile(a2, 95)),
        "corr_a1_a2": float(np.corrcoef(a1, a2)[0, 1]),
        "n_items": int(n_items), "k": int(k),
    }

    payload = {
        "agreement": agreement,
        "k": int(k),
        "dataset": ds,
        "source": {
            "llm_path": llm_path, "llm_sha1": sha1_of(llm_path),
            "cf_path": cf_path, "cf_sha1": sha1_of(cf_path),
            "bank_path": bank_path, "bank_sha1": sha1_of(bank_path),
        },
        "stats": stats,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pickle.dump(payload, open(out_path, "wb"))

    print(f"[nbr-agree] saved: {out_path}  agreement{agreement.shape}")
    for kk, vv in stats.items():
        print(f"    {kk:18s} = {vv}")


if __name__ == "__main__":
    main()
