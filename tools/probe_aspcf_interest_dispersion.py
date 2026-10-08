# -*- coding: UTF-8 -*-
"""
Chapter 4 Phase 2 — ASPCF interest-dispersion probe (inference only).

QUESTION
    Does the attention-weighted dispersion of the ASPCF history value space
    (X = extractor.Wv(history_emb_pos)) carry ranking signal that the
    attention-weighted mean does not?

NOT a model. NOT a contribution. NOT a training run.
`LLMMIRecASPCF.py` is NOT modified — everything is recomputed from the
checkpoint's own eval-mode forward passes.

KEY ALGEBRA (why one forward pass covers every lambda)
    u_mu     = Σ_k p_k · mu_k          (mu_k = Σ_l A[k,l] X_l)
    u_std    = Σ_k p_k · std_k         (std_k = sqrt(Σ_l A[k,l] (X_l−mu_k)²))
    u_lambda = u_mu + λ · u_std
    score_λ  = <u_lambda, e_j> = <u_mu, e_j> + λ · <u_std, e_j>
    ⇒ score_λ is AFFINE in λ, and both terms are λ-independent.
      So one forward pass yields every λ exactly.

SELF-CHECKS (abort on failure)
    1. max|mu_recomputed − model_interest_vectors| <= 1e-6
    2. max|score_mu − model_prediction|            <= 1e-6
    3. λ=0 must reproduce the frozen ASPCF numbers

λ IS SELECTED ON DEV ONLY. Test is evaluated once with the dev-selected λ
(plus λ=0 as a self-check). Never select on test.

Usage:
  python tools/probe_aspcf_interest_dispersion.py --dataset beauty
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

from torch.utils.data import DataLoader                        # noqa: E402
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF      # noqa: E402

LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
TOL = 1e-6


# ---------------------------------------------------------------- metrics

def ndcg_at(rank, k):
    return (1.0 / math.log2(rank + 1)) if rank <= k else 0.0


def rank_metrics(rank):
    """rank: int array [N]; 1-based rank of the positive candidate."""
    out = {}
    for k in (5, 10, 20):
        out[f"HR@{k}"] = float((rank <= k).mean())
        out[f"NDCG@{k}"] = float(np.mean([ndcg_at(int(r), k) for r in rank]))
    return out


def spearman(x, y):
    """Dependency-free Spearman: Pearson on average-tied ranks."""
    def _rank(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), dtype=np.float64)
        r[order] = np.arange(len(a), dtype=np.float64)
        # average ties
        s = a[order]
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j + 1] == s[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = _rank(np.asarray(x, dtype=np.float64)), _rank(np.asarray(y, dtype=np.float64))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    den = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(); y = y - y.mean()
    den = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    return float((x * y).sum() / den) if den > 0 else float("nan")


# ---------------------------------------------------------------- model io

def build_model(args, ds_name, device):
    a = argparse.Namespace(**vars(args))
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
    a.device = device
    a.path = "./data/"
    a.model_path = args.ckpt

    corpus = pickle.load(open(f"./data/{ds_name}/SeqReader.pkl", "rb"))
    model = LLMMIRecASPCF(a, corpus)
    missing, unexpected = model.load_state_dict(
        torch.load(args.ckpt, map_location="cpu"), strict=False)
    if missing or unexpected:
        print(f"[warn] missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device); model.device = device; model.eval()
    return model, corpus, a


def recompute_mu_std(model, batch, out, K, eps):
    """Recompute mu/std of the ATTENTION-WEIGHTED history value space.

    X = extractor.Wv(history_emb_pos); dropout is identity in eval mode.
    """
    history = batch["history_items"]
    lengths = batch["lengths"]
    B, L = history.shape
    device = history.device

    A = out["attention_maps"]                 # [B,K,L]
    hraw = out["history_vectors"]             # [B,L,D] = item_encoder(history, raw)
    iv = out["interest_vectors"]              # [B,K,D] model's own output

    # identical position construction to LLMMIRecASPCF.forward
    valid_his = (history > 0).long()
    len_range = torch.arange(model.max_his, device=device)
    position = (lengths[:, None] - len_range[None, :L]) * valid_his
    hpos = hraw + model.position_emb(position)          # dropout is eval-identity

    X = model.extractor.Wv(hpos)              # [B,L,D]
    mu = torch.bmm(A, X)                      # [B,K,D]  attention-weighted mean

    # self-check 1: mu must equal the model's own interest vectors
    err_mu = float((mu - iv).abs().max().item())

    # attention-weighted per-dim RMS deviation
    dev2 = (X[:, None, :, :] - mu[:, :, None, :]) ** 2   # [B,K,L,D]
    var = torch.einsum("bkl,bkld->bkd", A, dev2)         # [B,K,D]
    std = torch.sqrt(var + eps)

    return mu, std, X, err_mu


# ---------------------------------------------------------------- one pass

def run_pass(model, corpus, a, phase, lambdas, device,
             eps=1e-8, max_batches=0, corr_users=32, collect_diag=False):
    dset = model.Dataset(model, corpus, phase)
    dset.prepare()
    dl = DataLoader(dset, batch_size=a.batch_size, shuffle=False,
                    num_workers=0, collate_fn=dset.collate_batch)

    lambdas = sorted(set(float(x) for x in lambdas))   # dedupe (best_lam may be 0.0)
    ranks = {lam: [] for lam in lambdas}
    diag = {
        "std_norm": [], "std_norm_spread": [], "std_frac_of_mu": [],
        "cent": [], "disp": [],
        "pw_n": 0, "pw_win": 0, "pw_margin_sum": 0.0,
        "pw_n_c": 0, "pw_win_c": 0, "pw_margin_sum_c": 0.0,
        "pop": [], "hlen": [], "rank0": [], "rankbest": [],
        "err_mu": 0.0, "err_score0": 0.0,
    }
    rng = np.random.default_rng(0)
    n = 0

    with torch.inference_mode():
        for batch in dl:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            i_ids = batch["item_id"]
            out = model(batch, return_intermediate=True)

            mu, std, X, err_mu = recompute_mu_std(model, batch, out, a.K, eps)
            diag["err_mu"] = max(diag["err_mu"], err_mu)

            p = out["interest_weights"]                     # [B,K]
            cand = out["candidate_vectors"]                 # [B,C,D]

            u_mu = (mu * p[:, :, None]).sum(dim=1)          # [B,D]
            u_std = (std * p[:, :, None]).sum(dim=1)        # [B,D]

            score_mu = (u_mu[:, None, :] * cand).sum(dim=-1)     # [B,C]
            score_disp = (u_std[:, None, :] * cand).sum(dim=-1)  # [B,C]

            # self-check 2: lambda=0 must reproduce the model's prediction
            diag["err_score0"] = max(
                diag["err_score0"],
                float((score_mu - out["prediction"]).abs().max().item()))

            for lam in lambdas:
                sc = score_mu + lam * score_disp
                r = (sc > sc[:, 0:1]).sum(dim=1).cpu().numpy() + 1
                ranks[lam].append(r)

            if collect_diag:
                # ---- A. dispersion statistics ----
                sn = std.norm(dim=-1)                        # [B,K] ||std_k||
                diag["std_norm"].append(sn.reshape(-1).cpu().numpy())
                # spread of ||std_k|| ACROSS the K interests of one user
                diag["std_norm_spread"].append(
                    (sn.std(dim=1) / (sn.mean(dim=1) + 1e-12)).cpu().numpy())
                mn = mu.norm(dim=-1)                         # [B,K]
                diag["std_frac_of_mu"].append(
                    (sn / (mn + 1e-12)).reshape(-1).cpu().numpy())

                # ---- B. centrality vs dispersion (candidate level) ----
                bs = min(corr_users, score_mu.size(0))
                sel = torch.from_numpy(
                    rng.choice(score_mu.size(0), size=bs, replace=False)).to(device)
                diag["cent"].append(score_mu[sel].reshape(-1).cpu().numpy())
                diag["disp"].append(score_disp[sel].reshape(-1).cpu().numpy())

                # ---- C. pairwise ranking signal (pos vs all negatives) ----
                dp = score_disp[:, 0:1] - score_disp[:, 1:]        # [B,C-1]
                cp = score_mu[:, 0:1] - score_mu[:, 1:]
                diag["pw_n"] += int(dp.numel())
                diag["pw_win"] += int((dp > 0).sum().item())
                diag["pw_margin_sum"] += float(dp.sum().item())
                diag["pw_n_c"] += int(cp.numel())
                diag["pw_win_c"] += int((cp > 0).sum().item())
                diag["pw_margin_sum_c"] += float(cp.sum().item())

                # ---- D/E. buckets ----
                diag["pop"].append(i_ids[:, 0].cpu().numpy())
                diag["hlen"].append(batch["lengths"].cpu().numpy())

            n += 1
            if max_batches and n >= max_batches:
                break

    res = {"metrics_by_lambda": {}}
    for lam in lambdas:
        res["metrics_by_lambda"][str(lam)] = rank_metrics(np.concatenate(ranks[lam]))
    res["_selfcheck"] = {"max_err_mu": diag["err_mu"],
                         "max_err_score_lambda0": diag["err_score0"]}
    res["_ranks"] = {str(lam): np.concatenate(ranks[lam]) for lam in lambdas}

    if collect_diag:
        cat = lambda lst: np.concatenate([np.asarray(x).ravel() for x in lst])
        sn = cat(diag["std_norm"])
        res["dispersion"] = {
            "std_norm_mean": float(sn.mean()), "std_norm_std": float(sn.std()),
            "std_norm_p05": float(np.percentile(sn, 5)),
            "std_norm_p50": float(np.percentile(sn, 50)),
            "std_norm_p95": float(np.percentile(sn, 95)),
            "std_norm_across_k_relsd_mean": float(cat(diag["std_norm_spread"]).mean()),
            "std_norm_across_k_relsd_p50": float(np.percentile(cat(diag["std_norm_spread"]), 50)),
            "std_over_mu_ratio_mean": float(cat(diag["std_frac_of_mu"]).mean()),
        }
        cent = cat(diag["cent"]); disp = cat(diag["disp"])
        res["centrality_vs_dispersion"] = {
            "n_pairs": int(cent.size),
            "pearson": pearson(cent, disp),
            "spearman": spearman(cent, disp),
            "cent_std": float(cent.std()), "disp_std": float(disp.std()),
        }
        res["pairwise"] = {
            "dispersion": {
                "p_pos_gt_neg": diag["pw_win"] / max(diag["pw_n"], 1),
                "mean_margin": diag["pw_margin_sum"] / max(diag["pw_n"], 1),
            },
            "centrality": {
                "p_pos_gt_neg": diag["pw_win_c"] / max(diag["pw_n_c"], 1),
                "mean_margin": diag["pw_margin_sum_c"] / max(diag["pw_n_c"], 1),
            },
        }
        res["_buckets_raw"] = {
            "pop_item": cat(diag["pop"]).tolist(),
            "hlen": cat(diag["hlen"]).tolist(),
        }
    return res


# ---------------------------------------------------------------- reporting

def print_metrics(title, m):
    print(f"\n{title}")
    print(f"{'lambda':>8s}{'HR@5':>9s}{'HR@10':>9s}{'HR@20':>9s}"
          f"{'NDCG@5':>9s}{'NDCG@10':>9s}{'NDCG@20':>9s}")
    for lam, r in m.items():
        print(f"{lam:>8s}{r['HR@5']:9.4f}{r['HR@10']:9.4f}{r['HR@20']:9.4f}"
              f"{r['NDCG@5']:9.4f}{r['NDCG@10']:9.4f}{r['NDCG@20']:9.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="beauty")
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--corr_users", type=int, default=32)
    ap.add_argument("--max_batches", type=int, default=0, help="0 = all (debug only)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output_dir", type=str, default="./diagnostics_aspcf_dispersion")
    args = ap.parse_args()

    ds = args.dataset
    if not args.ckpt:
        args.ckpt = (f"new_model/llmmirec_aspcf_phase2/{ds}/seed42/"
                     f"LLMMIRecASPCF_seed42.pt")
    device = torch.device(args.device)
    model, corpus, a = build_model(args, ds, device)

    print("=" * 108)
    print(f"ASPCF interest-dispersion probe — {ds} (inference only, no training)")
    print(f"ckpt={args.ckpt}")
    print(f"K={a.K} D={a.emb_size} attn_size={a.attn_size} history_max={a.history_max}")
    print("=" * 108)

    # ---------------- dev: select lambda ----------------
    dev = run_pass(model, corpus, a, "dev", LAMBDAS, device,
                   max_batches=args.max_batches)
    print_metrics("DEV — lambda scan (selection happens HERE, never on test)",
                  dev["metrics_by_lambda"])
    sc = dev["_selfcheck"]
    print(f"\n[selfcheck dev] max|mu − model_iv| = {sc['max_err_mu']:.3e}   "
          f"max|score(λ=0) − model_pred| = {sc['max_err_score_lambda0']:.3e}")
    if sc["max_err_mu"] > TOL or sc["max_err_score_lambda0"] > TOL:
        raise SystemExit("SELF-CHECK FAILED — aborting dispersion probe.")

    # selection rule: NDCG@5 primary, HR@5 secondary
    ranked = sorted(dev["metrics_by_lambda"].items(),
                    key=lambda kv: (-kv[1]["NDCG@5"], -kv[1]["HR@5"]))
    best_lam = float(ranked[0][0])
    print(f"\n[select] best dev lambda = {best_lam} "
          f"(NDCG@5={ranked[0][1]['NDCG@5']:.4f}, HR@5={ranked[0][1]['HR@5']:.4f})")
    print(f"[select] dev lambda=0        "
          f"(NDCG@5={dev['metrics_by_lambda']['0.0']['NDCG@5']:.4f}, "
          f"HR@5={dev['metrics_by_lambda']['0.0']['HR@5']:.4f})")

    # ---------------- test: lambda=0 (self-check) + best lambda ----------
    # The FULL lambda grid is also run on test, but ONLY to profile where the
    # degradation lands (buckets). Selection already happened on dev; nothing
    # below re-selects a lambda on test.
    test = run_pass(model, corpus, a, "test", sorted(set(LAMBDAS + [best_lam])), device,
                    max_batches=args.max_batches, corr_users=args.corr_users,
                    collect_diag=True)
    print_metrics(f"TEST — λ=0 (self-check) + dev-selected λ*={best_lam}; "
                  f"remaining rows are the DEGRADATION PROFILE only (not selected here)",
                  test["metrics_by_lambda"])
    sc = test["_selfcheck"]
    print(f"\n[selfcheck test] max|mu − model_iv| = {sc['max_err_mu']:.3e}   "
          f"max|score(λ=0) − model_pred| = {sc['max_err_score_lambda0']:.3e}")
    if sc["max_err_mu"] > TOL or sc["max_err_score_lambda0"] > TOL:
        raise SystemExit("SELF-CHECK FAILED on test.")

    # ---------------- diagnostics ----------------
    print("\n" + "=" * 108)
    print("A. interest dispersion statistics ( ||std_k|| , std_k = sqrt(Σ_l A_kl (X_l−mu_k)²) )")
    print("=" * 108)
    d = test["dispersion"]
    print(f"  ||std_k||  mean={d['std_norm_mean']:.4f}  std={d['std_norm_std']:.4f}  "
          f"p05={d['std_norm_p05']:.4f}  p50={d['std_norm_p50']:.4f}  p95={d['std_norm_p95']:.4f}")
    print(f"  ||std_k|| 跨 K 的相对标准差 (per-user): "
          f"mean={d['std_norm_across_k_relsd_mean']:.4f}  p50={d['std_norm_across_k_relsd_p50']:.4f}")
    print(f"  ||std_k|| / ||mu_k|| 均值 = {d['std_over_mu_ratio_mean']:.4f}")

    print("\n" + "=" * 108)
    print("B. centrality vs dispersion (candidate level)")
    print("=" * 108)
    c = test["centrality_vs_dispersion"]
    print(f"  n_pairs={c['n_pairs']}  pearson={c['pearson']:+.4f}  spearman={c['spearman']:+.4f}")
    print(f"  score_centrality std={c['cent_std']:.4f}   score_dispersion std={c['disp_std']:.4f}")

    print("\n" + "=" * 108)
    print("C. pairwise ranking signal (positive vs all 1000 negatives, test)")
    print("=" * 108)
    pw = test["pairwise"]
    print(f"  dispersion : P(s_disp(pos) > s_disp(neg)) = {pw['dispersion']['p_pos_gt_neg']:.4f}   "
          f"mean margin = {pw['dispersion']['mean_margin']:+.4f}")
    print(f"  centrality : P(s_cent(pos) > s_cent(neg)) = {pw['centrality']['p_pos_gt_neg']:.4f}   "
          f"mean margin = {pw['centrality']['mean_margin']:+.4f}")
    print("  (raw margin 只在同一个 dispersion score 内解释，不跨模型比较绝对尺度)")

    # ---------------- D/E buckets ----------------
    raw = test["_buckets_raw"]
    pop_item = np.asarray(raw["pop_item"], dtype=np.int64)
    hlen = np.asarray(raw["hlen"], dtype=np.float64)
    rank0 = test["_ranks"]["0.0"]
    rankb = test["_ranks"][str(best_lam)]

    pop = np.zeros(corpus.n_items, dtype=np.float64)
    cnt = corpus.data_df["train"].groupby("item_id").size()
    pop[cnt.index.to_numpy()] = cnt.to_numpy()
    pop_val = pop[pop_item]

    pop_edges = np.unique(np.percentile(pop[pop > 0], [0, 20, 40, 60, 80, 100]).astype(np.float64))
    hlen_edges = np.array([0, 5, 10, 15, 20, 10 ** 9], dtype=np.float64)

    def bucket_table(name, edges, values, note=""):
        """NDCG@5 profile across the FULL lambda grid, per bucket.

        λ* was chosen on dev and happens to be 0.0, so a "λ=0 vs λ*" column would
        be trivially identical. Instead we show where each λ>0 lands, which is the
        diagnostic question D/E actually ask (is the degradation uniform?).
        """
        print("\n" + "=" * 108)
        print(f"{name}{note}")
        print("=" * 108)
        q = np.digitize(values, edges[1:-1])
        grid = [lam for lam in sorted(test["_ranks"]) ]
        hdr = f"{'bin':>16s}{'n':>8s}" + "".join(f"{'l=' + lam:>11s}" for lam in grid)
        print(hdr + "   (NDCG@5; λ*=" + str(best_lam) + " selected on DEV)")
        rows = []
        for i in range(len(edges) - 1):
            m = q == i
            if m.sum() == 0:
                continue
            lab = f"[{edges[i]:.0f},{edges[i+1]:.0f})"
            vals = {}
            line = f"{lab:>16s}{int(m.sum()):8d}"
            for lam in grid:
                v = float(np.mean([ndcg_at(int(r), 5)
                                   for r in test["_ranks"][lam][m]]))
                vals[lam] = v
                line += f"{v:11.4f}"
            print(line)
            rows.append(dict(bin=lab, n=int(m.sum()),
                             ndcg5_by_lambda={lam: vals[lam] for lam in grid},
                             delta_ndcg5_vs_l0={lam: vals[lam] - vals["0.0"] for lam in grid}))
        return rows

    by_pop = bucket_table("D. popularity buckets (target item, train frequency quintiles)",
                          pop_edges, pop_val.astype(np.float64))
    by_hlen = bucket_table("E. history-length buckets",
                           hlen_edges, hlen,
                           note="  (观察：长历史中 dispersion 是否更可靠)")

    # ---------------- save ----------------
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f"{ds}_dispersion_probe.json")
    payload = {
        "checkpoint": args.ckpt, "dataset": ds, "K": a.K, "emb_size": a.emb_size,
        "dev_metrics_by_lambda": dev["metrics_by_lambda"],
        "dev_selected_lambda": best_lam,
        "test_metrics_by_lambda": test["metrics_by_lambda"],
        "test_selfcheck": test["_selfcheck"],
        "dispersion_stats": d,
        "centrality_vs_dispersion": c,
        "pairwise": pw,
        "by_popularity": by_pop,
        "by_history_length": by_hlen,
        "lambdas_scanned": LAMBDAS,
    }
    json.dump(payload, open(path, "w"), indent=2)
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
