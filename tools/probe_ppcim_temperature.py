# -*- coding: UTF-8 -*-
"""
PPCIM Round 1.5 — inference-only temperature probe.

NO TRAINING. Uses the already-trained `ppcim_history_marginal` (tau=1.0, seed 42)
checkpoint and re-scores the Beauty test split under several tau values, purely
at inference time.

Why this is legitimate: `m = <V_k,e_j>/sqrt(D)` and the prior `p` do NOT depend on
tau. tau only enters the softmax/logsumexp that turns `m` into a posterior. So a
single forward pass per batch yields everything needed for every tau.

Two scoring rules are probed:
  marginal        sqrt(D)*tau*logsumexp_k(log p_k + m_k/tau)
  posterior_mean  sqrt(D)*sum_k pi_k(tau)*m_k
plus the tau -> 0 limit of marginal (= hard max):
  hardmax         max_k <V_k, e_j> == sqrt(D)*max_k m_k

This is a MECHANISM probe, not a tuning experiment. The goal is to locate the tau
range where the posterior actually becomes candidate-specific. Do NOT pick the
best-scoring tau and report it as the method.

Usage:
  python tools/probe_ppcim_temperature.py --dataset beauty
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

from torch.utils.data import DataLoader                       # noqa: E402
from models.sequential.LLMMIRecPPCIM import LLMMIRecPPCIM     # noqa: E402

TAUS = [1.0, 0.5, 0.2, 0.1, 0.05]


def ndcg_at(rank, k):
    return (1.0 / math.log2(rank + 1)) if rank <= k else 0.0


def entropy(p, axis=-1, eps=1e-12):
    return -(p * np.log(p + eps)).sum(axis=axis)


def js_div(p, q, eps=1e-12):
    m = 0.5 * (p + q)
    return 0.5 * ((p * np.log((p + eps) / (m + eps))).sum()
                  + (q * np.log((q + eps) / (m + eps))).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="beauty")
    ap.add_argument("--ckpt", type=str,
                    default="new_model/llmmirec_ppcim/beauty/"
                            "ppcim_history_marginal/seed42/LLMMIRecPPCIM_seed42.pt")
    ap.add_argument("--phase", type=str, default="test")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_batches", type=int, default=0, help="0 = all")
    ap.add_argument("--pair_samples", type=int, default=48)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output_dir", type=str, default="./diagnostics_ppcim_probe")
    args = ap.parse_args()

    ds_name = args.dataset
    a = args
    a.emb_size = 64; a.attn_size = 64; a.K = 4; a.history_max = 20
    a.num_neg = 1; a.test_all = 0; a.buffer = 1; a.dropout = 0.1
    a.item_encoder = "aspcf"
    a.llm_emb_path = f"./data/{ds_name}/handled/llm_table_pca1536.pkl"
    a.semantic_rank = 512; a.semantic_dim = 32; a.semantic_hidden = 128
    a.complement_dim = 32; a.tail_hidden = 64; a.complement_hidden = 64
    a.gate_hidden = 64; a.aspcf_gate_mode = "basic"
    a.adapter_hidden = 256; a.adapter_activation = "gelu"; a.adapter_use_ln = 0
    a.gamma_init = 0.1; a.gamma_trainable = 0
    a.lambda_relation = 0.01; a.relation_sample_size = 128
    a.relation_teacher_temp = 0.1; a.relation_student_temp = 0.1
    a.ppcim_mode = "marginal"; a.ppcim_prior = "history"
    a.ppcim_tau = 1.0; a.ppcim_eps = 1e-8
    a.device = torch.device(args.device)
    a.model_path = args.ckpt

    corpus = pickle.load(open(f"./data/{ds_name}/SeqReader.pkl", "rb"))
    model = LLMMIRecPPCIM(a, corpus)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=False)
    model = model.to(args.device); model.device = args.device; model.eval()
    sqrtD = math.sqrt(a.emb_size)
    eps = a.ppcim_eps
    print(f"[probe] ckpt={args.ckpt}")
    print(f"[probe] dataset={ds_name} K={a.K} D={a.emb_size} C(test)=1+1000")

    dset = model.Dataset(model, corpus, args.phase); dset.prepare()
    dl = DataLoader(dset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                    collate_fn=dset.collate_batch)

    names = [f"tau={t}" for t in TAUS] + ["hardmax(tau->0)"]
    rank_hist = {n: [] for n in names}
    diag = {n: {"prior_H": [], "post_H": [], "pos_maxp": [], "neg_maxp": [],
                "z_std": [], "mtau_std": [], "cos": [], "js": [],
                "agree": 0, "disagree": 0, "top_pos": np.zeros(a.K, dtype=np.int64),
                "top_neg": np.zeros(a.K, dtype=np.int64)} for n in names}
    rng = np.random.default_rng(0)

    n = 0
    with torch.inference_mode():
        for batch in dl:
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model(batch, return_intermediate=True)
            m = out["candidate_interest_match"]          # [B,K,C]  tau-independent
            p = out["history_interest_prior"]            # [B,K]    tau-independent
            logp = torch.log(p + eps).unsqueeze(-1)      # [B,K,1]

            scores = {}
            for t in TAUS:
                z = logp + m / t
                scores[f"tau={t}"] = sqrtD * t * torch.logsumexp(z, dim=1)
            # hard max == tau -> 0 limit of marginal
            scores["hardmax(tau->0)"] = sqrtD * m.max(dim=1).values

            for name, sc in scores.items():
                r = (sc > sc[:, 0:1]).sum(dim=1).cpu().numpy() + 1
                rank_hist[name].append(r)

            # ---- diagnostics per tau (fully vectorised) ----
            B, K, C = m.shape
            idx = torch.from_numpy(
                rng.choice(C, size=min(args.pair_samples, C), replace=False)
            ).to(m.device)
            iu = torch.triu_indices(len(idx), len(idx), offset=1)
            prior_H = entropy(p.detach().cpu().numpy(), axis=-1)          # [B]
            for t in TAUS:
                name = f"tau={t}"
                z = logp + m / t                                          # [B,K,C]
                pi = torch.softmax(z, dim=1)                              # [B,K,C]
                D = diag[name]
                pih = pi.detach().cpu().numpy()
                D["prior_H"].append(prior_H)
                D["post_H"].append(entropy(pih, axis=1))                  # [B,C]
                D["pos_maxp"].append(pi[:, :, 0].max(dim=1).values.cpu().numpy())
                D["neg_maxp"].append(pi[:, :, 1:].max(dim=1).values.mean(1).cpu().numpy())
                D["z_std"].append(z.std(dim=1).mean(1).cpu().numpy())
                D["mtau_std"].append((m / t).std(dim=1).mean(1).cpu().numpy())
                top = pi.argmax(dim=1)                                    # [B,C]
                tp = top[:, 0]
                tn = top[:, 1:]
                D["top_pos"] += torch.bincount(tp.cpu(), minlength=K).numpy()
                D["top_neg"] += torch.bincount(tn.reshape(-1).cpu(), minlength=K).numpy()
                D["agree"] += int((tn == tp[:, None]).sum().item())
                D["disagree"] += int((tn != tp[:, None]).sum().item())

                Q = pi[:, :, idx]                                          # [B,K,S]
                Qn = Q / Q.norm(dim=1, keepdim=True).clamp(min=1e-12)
                cos = torch.einsum("bks,bkt->bst", Qn, Qn)[:, iu[0], iu[1]]  # [B,P]
                Qi = Q[:, :, iu[0]]                                        # [B,K,P]
                Qj = Q[:, :, iu[1]]
                Mjs = 0.5 * (Qi + Qj)
                js = 0.5 * ((Qi * torch.log((Qi + eps) / (Mjs + eps))).sum(1)
                            + (Qj * torch.log((Qj + eps) / (Mjs + eps))).sum(1))
                D["cos"].append(cos.mean(1).cpu().numpy())
                D["js"].append(js.mean(1).cpu().numpy())
            n += 1
            if args.max_batches and n >= args.max_batches:
                break

    # ---- aggregate ----
    out = {}
    for name in names:
        r = np.concatenate(rank_hist[name])
        res = {f"HR@{k}": float((r <= k).mean()) for k in (5, 10, 20)}
        res.update({f"NDCG@{k}": float(np.mean([ndcg_at(int(x), k) for x in r]))
                    for k in (5, 10, 20)})
        d = diag[name]

        def cat(lst):
            return np.concatenate([np.asarray(x).ravel() for x in lst]) if lst else np.array([])

        if d["prior_H"]:
            ph = float(cat(d["prior_H"]).mean()); po = float(cat(d["post_H"]).mean())
            res.update({
                "prior_H": ph, "posterior_H": po,
                "entropy_reduction_pct": (ph - po) / ph * 100,
                "pos_max_prob": float(cat(d["pos_maxp"]).mean()),
                "neg_max_prob": float(cat(d["neg_maxp"]).mean()),
                "z_cross_interest_std": float(cat(d["z_std"]).mean()),
                "m_over_tau_cross_interest_std": float(cat(d["mtau_std"]).mean()),
                "cand_pairwise_cos": float(cat(d["cos"]).mean()),
                "cand_pairwise_js": float(cat(d["js"]).mean()),
                "pos_neg_top_agree": d["agree"] / max(d["agree"] + d["disagree"], 1),
                "n_users_mechanism": int(cat(d["prior_H"]).size),
                "top_pos_frac": (d["top_pos"] / max(d["top_pos"].sum(), 1)).tolist(),
                "top_neg_frac": (d["top_neg"] / max(d["top_neg"].sum(), 1)).tolist(),
            })
        out[name] = res

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f"{ds_name}_temperature_probe.json")
    json.dump(out, open(path, "w"), indent=2)

    # ---- report ----
    print("\n" + "=" * 112)
    print("温度探针：Beauty test，同一 checkpoint（ppcim_history_marginal τ=1.0 seed42），仅推理覆盖 τ")
    print("=" * 112)
    print(f"{'setting':18s}{'HR@5':>9s}{'HR@10':>9s}{'HR@20':>9s}{'NDCG@5':>9s}{'NDCG@10':>9s}{'NDCG@20':>9s}")
    print(f"{'ASPCF(参考)':18s}{0.1592:9.4f}{0.2292:9.4f}{0.3171:9.4f}{0.1088:9.4f}{0.1313:9.4f}{0.1535:9.4f}")
    for name in names:
        r = out[name]
        print(f"{name:18s}{r['HR@5']:9.4f}{r['HR@10']:9.4f}{r['HR@20']:9.4f}"
              f"{r['NDCG@5']:9.4f}{r['NDCG@10']:9.4f}{r['NDCG@20']:9.4f}")
    print("\n" + "=" * 112)
    print("机制诊断（posterior 是否真正 candidate-specific）")
    print("=" * 112)
    hdr = f"{'setting':18s}{'priorH':>8s}{'postH':>8s}{'H↓%':>7s}{'cos':>8s}{'JS':>9s}" \
          f"{'posMaxP':>9s}{'negMaxP':>9s}{'z_std':>8s}{'agree':>7s}"
    print(hdr)
    for name in names:
        r = out[name]
        if "posterior_H" not in r:
            continue
        print(f"{name:18s}{r['prior_H']:8.4f}{r['posterior_H']:8.4f}"
              f"{r['entropy_reduction_pct']:6.2f}%{r['cand_pairwise_cos']:8.4f}"
              f"{r['cand_pairwise_js']:9.5f}{r['pos_max_prob']:9.4f}{r['neg_max_prob']:9.4f}"
              f"{r['z_cross_interest_std']:8.4f}{r['pos_neg_top_agree']:7.3f}")
    print(f"\n参考：ln K = {math.log(a.K):.4f}；top-interest 随机一致率 = {1.0/a.K:.3f}")
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
