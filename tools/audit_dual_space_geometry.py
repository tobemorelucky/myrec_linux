# -*- coding: UTF-8 -*-
"""
Chapter 4 Phase 4 — Dual-Space Ranking Geometry Audit  (inference only, NO training).

F8 (to be tested, NOT assumed to be a bottleneck):
    ASPCF builds an explicit semantic/complement split at the ITEM level
    (e = concat[sqrt(a_s)*s, sqrt(a_c)*c], 32+32 = 64), but there is currently
    no evidence that the downstream multi-interest projection and the final
    dot product actually exploit those two subspaces.

Sections
  2  exact decomposition of the final score, verified against the model
  3  ranking signal of each coordinate block alone
  4  branch correlation / conflict (quadrants of the pos-neg margin)
  5  ItemEncoder energy: alpha alone is NOT the semantic contribution
  6  Wv / Wk cross-subspace mixing (weight-level, static)
  7  dev-only beta calibration  score_beta = score_comp + beta*score_sem
  8  fine-grained history->candidate evidence (max / last cosine)
  9  popularity & history-length buckets

TERMINOLOGY (enforced): u[:32] / u[32:] are "semantic-coordinate" /
"complement-coordinate" user CONTRIBUTIONS. They are NOT "pure semantic
interest" / "pure collaborative interest", because Wv is a full dense 64x64
projection, so the user-side halves may already mix the input subspaces.
On the CANDIDATE side only, e[:32]/e[32:] correspond exactly to the two
final output blocks of the ASPCF ItemEncoder.

Usage:
  python tools/audit_dual_space_geometry.py --dataset beauty
  python tools/audit_dual_space_geometry.py --dataset ml-1m
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

BETAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
HALF = 32


# ---------------------------------------------------------------- helpers

def ndcg_at(rank, k):
    return (1.0 / math.log2(rank + 1)) if rank <= k else 0.0


def rank_metrics(rank):
    o = {}
    for k in (5, 10, 20):
        o[f"HR@{k}"] = float((rank <= k).mean())
        o[f"NDCG@{k}"] = float(np.mean([ndcg_at(int(r), k) for r in rank]))
    return o


def _avg_rank(a):
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), dtype=np.float64)
    r[order] = np.arange(len(a), dtype=np.float64)
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


def corr(x, y, kind="pearson"):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if kind == "spearman":
        x, y = _avg_rank(x), _avg_rank(y)
    x = x - x.mean(); y = y - y.mean()
    den = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    return float((x * y).sum() / den) if den > 0 else float("nan")


def pct_stats(v):
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return {}
    return {"mean": float(v.mean()), "std": float(v.std()),
            "p05": float(np.percentile(v, 5)), "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95))}


# ---------------------------------------------------------------- model

def build(ds, ckpt, device, batch_size=256):
    a = argparse.Namespace()
    a.emb_size = 64; a.attn_size = 64; a.K = 4; a.history_max = 20
    a.num_neg = 1; a.test_all = 0; a.buffer = 1; a.dropout = 0.1
    a.batch_size = int(batch_size)
    a.item_encoder = "aspcf"
    a.llm_emb_path = f"./data/{ds}/handled/llm_table_pca1536.pkl"
    a.semantic_rank = 512; a.semantic_dim = 32; a.semantic_hidden = 128
    a.complement_dim = 32; a.tail_hidden = 64; a.complement_hidden = 64
    a.gate_hidden = 64; a.aspcf_gate_mode = "basic"
    a.adapter_hidden = 256; a.adapter_activation = "gelu"; a.adapter_use_ln = 0
    a.gamma_init = 0.1; a.gamma_trainable = 0
    a.lambda_relation = 0.01; a.relation_sample_size = 128
    a.relation_teacher_temp = 0.1; a.relation_student_temp = 0.1
    a.device = device; a.path = "./data/"; a.model_path = ckpt
    corpus = pickle.load(open(f"./data/{ds}/SeqReader.pkl", "rb"))
    m = LLMMIRecASPCF(a, corpus)
    miss, unexp = m.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    if miss or unexp:
        print(f"[warn] missing={len(miss)} unexpected={len(unexp)}")
    m = m.to(device); m.device = device; m.eval()
    return m, corpus, a


# ---------------------------------------------------------------- one pass

def run_pass(model, corpus, a, phase, betas, device,
             max_batches=0, corr_users=24, need_beta=True):
    dset = model.Dataset(model, corpus, phase); dset.prepare()
    dl = DataLoader(dset, batch_size=a.batch_size, shuffle=False,
                    num_workers=0, collate_fn=dset.collate_batch)

    R = {}
    def add(k):
        R[k] = []

    for k in ("full", "sem", "comp", "g_sem", "g_comp", "g_full",
              "last_sem", "last_comp", "last_full"):
        add(k)
    if need_beta:
        for b in betas:
            add(f"beta={b}")

    acc = {k: {"pw_n": 0, "pw_win": 0, "pw_margin": 0.0} for k in
           ("full", "sem", "comp", "g_sem", "g_comp", "g_full",
            "last_sem", "last_comp", "last_full")}
    quad = {"q1": 0, "q2": 0, "q3": 0, "q4": 0}
    E = {"alpha_sem": [], "alpha_comp": [],
         "rho_sem_pos": [], "rho_sem_all": [], "rho_sem_hist": [],
         "alpha_sem_hist": [], "alpha_comp_hist": [],
         "sn_pos": [], "sc_pos": []}
    CORR = {"full": [], "g_sem": [], "g_comp": [], "g_full": [],
            "sem": [], "comp": [],
            "last_sem": [], "last_comp": [], "last_full": []}
    META = {"pop_item": [], "hlen": []}
    err_full = 0.0
    err32 = 0.0
    rng = np.random.default_rng(0)
    n = 0

    with torch.inference_mode():
        for batch in dl:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            i_ids = batch["item_id"]
            history = batch["history_items"]
            lengths = batch["lengths"]
            out = model(batch, return_intermediate=True)
            B, L = history.shape
            C = i_ids.size(1)

            u = out["user_vector"]                       # [B,64]
            e = out["candidate_vectors"]                 # [B,C,64]
            pred = out["prediction"]                     # [B,C]

            # ---------------- 2. exact decomposition ----------------
            u_sem, u_comp = u[:, :HALF], u[:, HALF:]
            e_sem, e_comp = e[..., :HALF], e[..., HALF:]
            s_sem = (u_sem[:, None, :] * e_sem).sum(-1)          # [B,C]
            s_comp = (u_comp[:, None, :] * e_comp).sum(-1)
            s_full = s_sem + s_comp
            # float32 error is informational only: splitting one 64-term dot
            # product into two 32-term partial sums reorders the accumulation,
            # so ~1e-6 absolute is expected float32 associativity noise.
            err32 = max(err32, float((s_full - pred).abs().max().item()))
            # the real gate: the identity must be exact in float64.
            # Run ONCE per pass: this upcasts [B,C,64] to float64, which is
            # ~130 MB per batch at C=1001 -- repeating it every batch leaks
            # host RSS at ~0.6 GB/batch and OOMs the full Beauty run.
            if n == 0:
                u64, e64 = u.double(), e.double()
                ref64 = (u64[:, None, :] * e64).sum(-1)
                rec64 = ((u64[..., :HALF][:, None, :] * e64[..., :HALF]).sum(-1)
                         + (u64[..., HALF:][:, None, :] * e64[..., HALF:]).sum(-1))
                err_full = max(err_full, float((rec64 - ref64).abs().max().item()))
                del u64, e64, ref64, rec64

            # ---------------- 8. fine-grained evidence ----------------
            hsem = out["history_semantic"]               # [B,L,32] un-gated s
            hcomp = out["history_complement"]            # [B,L,32]
            hitem = out["history_vectors"]               # [B,L,64] raw item emb
            csem = out["candidate_semantic"]
            ccomp = out["candidate_complement"]
            cfull = e

            lmask = (torch.arange(L, device=device)[None, :]
                     < lengths[:, None])                 # [B,L] True=valid

            def maxcos(h, c):
                hn = h / h.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                cn = c / c.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                sim = torch.bmm(hn, cn.transpose(1, 2))          # [B,L,C]
                sim = sim.masked_fill(~lmask[:, :, None], -1e4)
                return sim.max(dim=1).values

            g_sem = maxcos(hsem, csem)
            g_comp = maxcos(hcomp, ccomp)
            g_full = maxcos(hitem, cfull)

            last_idx = (lengths - 1).clamp(min=0).long()
            bidx = torch.arange(B, device=device)
            def lastcos(h, c):
                hn = h[bidx, last_idx] / h[bidx, last_idx].norm(dim=-1, keepdim=True).clamp(min=1e-12)
                cn = c / c.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                return torch.bmm(hn[:, None, :], cn.transpose(1, 2)).squeeze(1)
            l_sem = lastcos(hsem, csem)
            l_comp = lastcos(hcomp, ccomp)
            l_full = lastcos(hitem, cfull)
            for nm, sc in (("last_sem", l_sem), ("last_comp", l_comp),
                           ("last_full", l_full)):
                sc = sc.clone()
                sc[lengths == 0] = 0.0
                if nm == "last_sem":
                    l_sem = sc
                elif nm == "last_comp":
                    l_comp = sc
                else:
                    l_full = sc

            scores = {"full": s_full, "sem": s_sem, "comp": s_comp,
                      "g_sem": g_sem, "g_comp": g_comp, "g_full": g_full,
                      "last_sem": l_sem, "last_comp": l_comp, "last_full": l_full}

            # ---------------- ranks ----------------
            for nm, sc in scores.items():
                r = (sc > sc[:, 0:1]).sum(dim=1).cpu().numpy() + 1
                R[nm].append(r)
                d = sc[:, 0:1] - sc[:, 1:]
                a_ = acc[nm]
                a_["pw_n"] += int(d.numel())
                a_["pw_win"] += int((d > 0).sum().item())
                a_["pw_margin"] += float(d.sum().item())

            if need_beta:
                for b in betas:
                    sc = s_comp + b * s_sem
                    R[f"beta={b}"].append(
                        (sc > sc[:, 0:1]).sum(dim=1).cpu().numpy() + 1)

            # ---------------- 4. quadrants ----------------
            ds = (s_sem[:, 0:1] - s_sem[:, 1:])
            dc = (s_comp[:, 0:1] - s_comp[:, 1:])
            quad["q1"] += int(((ds > 0) & (dc > 0)).sum().item())
            quad["q2"] += int(((ds > 0) & (dc <= 0)).sum().item())
            quad["q3"] += int(((ds <= 0) & (dc > 0)).sum().item())
            quad["q4"] += int(((ds <= 0) & (dc <= 0)).sum().item())

            # ---------------- 5. energy ----------------
            a_s = out["candidate_alpha_sem"]             # [B,C]
            a_c = out["candidate_alpha_comp"]
            E["alpha_sem"].append(a_s[:, 0].cpu().numpy())
            E["alpha_comp"].append(a_c[:, 0].cpu().numpy())
            E["sn_pos"].append(csem[:, 0].norm(dim=-1).cpu().numpy())
            E["sc_pos"].append(ccomp[:, 0].norm(dim=-1).cpu().numpy())
            # E = alpha * ||branch||^2 must use the RAW branch output (s, c),
            # NOT the already-scaled embedding block sqrt(alpha)*s.
            Es = a_s * (csem ** 2).sum(-1)
            Ec = a_c * (ccomp ** 2).sum(-1)
            E["rho_sem_pos"].append((Es[:, 0] / (Es[:, 0] + Ec[:, 0] + 1e-12)).cpu().numpy())
            E["rho_sem_all"].append(
                (Es / (Es + Ec + 1e-12)).reshape(-1).cpu().numpy())
            hs = out["history_alpha_sem"]; hc = out["history_alpha_comp"]
            hEs = hs * (hsem ** 2).sum(-1); hEc = hc * (hcomp ** 2).sum(-1)
            hv = lmask.reshape(-1)
            E["rho_sem_hist"].append(
                (hEs / (hEs + hEc + 1e-12)).reshape(-1)[hv].cpu().numpy())
            E["alpha_sem_hist"].append(hs.reshape(-1)[hv].cpu().numpy())
            E["alpha_comp_hist"].append(hc.reshape(-1)[hv].cpu().numpy())

            # ---------------- correlations (sampled candidates) ----------------
            bs = min(corr_users, B)
            sel = torch.from_numpy(rng.choice(B, size=bs, replace=False)).to(device)
            sl = sel[:, None].expand(-1, C)
            CORR["full"].append(pred[sl].cpu().numpy())
            CORR["sem"].append(s_sem[sl].cpu().numpy())
            CORR["comp"].append(s_comp[sl].cpu().numpy())
            CORR["g_sem"].append(g_sem[sl].cpu().numpy())
            CORR["g_comp"].append(g_comp[sl].cpu().numpy())
            CORR["g_full"].append(g_full[sl].cpu().numpy())
            CORR["last_sem"].append(l_sem[sl].cpu().numpy())
            CORR["last_comp"].append(l_comp[sl].cpu().numpy())
            CORR["last_full"].append(l_full[sl].cpu().numpy())

            META["pop_item"].append(i_ids[:, 0].cpu().numpy())
            META["hlen"].append(lengths.cpu().numpy())

            n += 1
            if max_batches and n >= max_batches:
                break

    cat = lambda L_: np.concatenate([np.asarray(x).ravel() for x in L_])
    res = {"phase": phase, "n_users": int(cat(R["full"]).size),
           "recompose_max_abs_err_f64": err_full,
           "recompose_max_abs_err_f32": err32,
           "ranking": {k: rank_metrics(cat(v)) for k, v in R.items()},
           "pairwise": {k: {"p_pos_gt_neg": v["pw_win"] / max(v["pw_n"], 1),
                            "mean_margin": v["pw_margin"] / max(v["pw_n"], 1),
                            "n_pairs": v["pw_n"]} for k, v in acc.items()},
           "quadrants": {k: v / max(sum(quad.values()), 1) for k, v in quad.items()},
           "quadrant_counts": quad,
           "energy": {k: pct_stats(cat(v)) for k, v in E.items()},
           "_acc": acc, "_R": {k: cat(v) for k, v in R.items()},
           "_meta": {k: cat(v) for k, v in META.items()},
           "_corr": {k: cat(v) for k, v in CORR.items()},
           }
    return res


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="beauty")
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--corr_users", type=int, default=24)
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output_dir", type=str, default="./diagnostics_ch4_phase4")
    args = ap.parse_args()
    ds = args.dataset
    if not args.ckpt:
        args.ckpt = (f"new_model/llmmirec_aspcf_phase2/{ds}/seed42/"
                     f"LLMMIRecASPCF_seed42.pt")
    device = torch.device(args.device)
    model, corpus, a = build(ds, args.ckpt, device, args.batch_size)

    print("=" * 104)
    print(f"Chapter 4 Phase 4 — Dual-Space Ranking Geometry Audit — {ds} (inference only)")
    print(f"ckpt={args.ckpt}")
    print("=" * 104)

    dev = run_pass(model, corpus, a, "dev", BETAS, device,
                   max_batches=args.max_batches, corr_users=args.corr_users)
    print(f"\n[dev] recompose  float64 err = {dev['recompose_max_abs_err_f64']:.3e}  "
          f"(gate, must be <1e-9)   float32 err = {dev['recompose_max_abs_err_f32']:.3e} "
          f"(associativity noise, informational)")
    if dev["recompose_max_abs_err_f64"] > 1e-9:
        raise SystemExit("DECOMPOSITION SELF-CHECK FAILED — aborting.")
    print("\nDEV beta scan (score_beta = score_comp + beta*score_sem)")
    print(f"{'beta':>8s}{'HR@5':>9s}{'HR@10':>9s}{'HR@20':>9s}"
          f"{'NDCG@5':>9s}{'NDCG@10':>9s}{'NDCG@20':>9s}")
    for b in BETAS:
        m = dev["ranking"][f"beta={b}"]
        print(f"{b:>8}{m['HR@5']:9.4f}{m['HR@10']:9.4f}{m['HR@20']:9.4f}"
              f"{m['NDCG@5']:9.4f}{m['NDCG@10']:9.4f}{m['NDCG@20']:9.4f}")
    ranked = sorted(((f"beta={b}", dev["ranking"][f"beta={b}"]) for b in BETAS),
                    key=lambda kv: (-kv[1]["NDCG@5"], -kv[1]["HR@5"]))
    best_beta = float(ranked[0][0].split("=")[1])
    print(f"[select] dev-best beta = {best_beta} "
          f"(HR@5={ranked[0][1]['HR@5']:.4f}, NDCG@5={ranked[0][1]['NDCG@5']:.4f})")
    print(f"[select] beta=1 (== ASPCF): HR@5="
          f"{dev['ranking']['beta=1.0']['HR@5']:.4f}, "
          f"NDCG@5={dev['ranking']['beta=1.0']['NDCG@5']:.4f}")

    test = run_pass(model, corpus, a, "test", BETAS, device,
                    max_batches=args.max_batches, corr_users=args.corr_users)
    print(f"\n[test] recompose  float64 err = {test['recompose_max_abs_err_f64']:.3e}  "
          f"(gate, must be <1e-9)   float32 err = {test['recompose_max_abs_err_f32']:.3e} "
          f"(associativity noise, informational)")
    if test["recompose_max_abs_err_f64"] > 1e-9:
        raise SystemExit("DECOMPOSITION SELF-CHECK FAILED on test.")

    print("\n" + "=" * 104)
    print("2/3. Coordinate-block ranking (TEST)")
    print("=" * 104)
    print(f"{'score':>12s}{'HR@5':>9s}{'HR@10':>9s}{'HR@20':>9s}"
          f"{'NDCG@5':>9s}{'NDCG@10':>9s}{'NDCG@20':>9s}"
          f"{'P(pos>neg)':>12s}{'margin':>10s}")
    for k in ("full", "sem", "comp"):
        m = test["ranking"][k]; p = test["pairwise"][k]
        print(f"{k:>12s}{m['HR@5']:9.4f}{m['HR@10']:9.4f}{m['HR@20']:9.4f}"
              f"{m['NDCG@5']:9.4f}{m['NDCG@10']:9.4f}{m['NDCG@20']:9.4f}"
              f"{p['p_pos_gt_neg']:12.4f}{p['mean_margin']:10.4f}")

    print("\n" + "=" * 104)
    print("4. branch correlation / conflict (TEST)")
    print("=" * 104)
    cs, cc = test["_corr"]["sem"].ravel(), test["_corr"]["comp"].ravel()
    print(f"  candidate-level Pearson(sem,comp)  = {corr(cs, cc):+.4f}")
    print(f"  candidate-level Spearman(sem,comp) = {corr(cs, cc, 'spearman'):+.4f}")
    q = test["quadrants"]; qc = test["quadrant_counts"]
    print(f"  quadrant share of (pos−neg) pairs:")
    print(f"    Q1 Δsem>0 & Δcomp>0 : {q['q1']:.4f}  (n={qc['q1']})")
    print(f"    Q2 Δsem>0 & Δcomp<=0: {q['q2']:.4f}  (n={qc['q2']})   <- semantic right, complement wrong")
    print(f"    Q3 Δsem<=0 & Δcomp>0: {q['q3']:.4f}  (n={qc['q3']})   <- complement right, semantic wrong")
    print(f"    Q4 both<=0          : {q['q4']:.4f}  (n={qc['q4']})")
    print(f"  Q2 − Q3 = {q['q2'] - q['q3']:+.4f}")

    print("\n" + "=" * 104)
    print("5. ItemEncoder energy  (E = alpha * ||branch||^2,  rho_sem = E_sem/(E_sem+E_comp))")
    print("=" * 104)
    e_ = test["energy"]
    for k in ("alpha_sem", "alpha_comp", "sn_pos", "sc_pos",
              "rho_sem_pos", "rho_sem_all", "rho_sem_hist"):
        s = e_.get(k, {})
        if s:
            print(f"  {k:16s} mean={s['mean']:.4f} std={s['std']:.4f} "
                  f"p05={s['p05']:.4f} p50={s['p50']:.4f} p95={s['p95']:.4f}")

    print("\n" + "=" * 104)
    print("6. Wv / Wk cross-subspace mixing (weights, static)")
    print("=" * 104)
    Wv = model.extractor.Wv.weight.detach().cpu().numpy()
    Wk = model.extractor.Wk.weight.detach().cpu().numpy()
    def fn(a_):
        return float(np.linalg.norm(a_, "fro"))
    Wss, Wsc = Wv[:32, :32], Wv[:32, 32:]
    Wcs, Wcc = Wv[32:, :32], Wv[32:, 32:]
    tot = fn(Wv) ** 2
    r_cross = (fn(Wsc) ** 2 + fn(Wcs) ** 2) / tot
    print(f"  ||Wv||_F={math.sqrt(tot):.4f}   ||W_ss||={fn(Wss):.4f} "
          f"||W_sc||={fn(Wsc):.4f} ||W_cs||={fn(Wcs):.4f} ||W_cc||={fn(Wcc):.4f}")
    print(f"  blocks (share of ||Wv||_F^2): ss={fn(Wss)**2/tot:.4f} "
          f"sc={fn(Wsc)**2/tot:.4f} cs={fn(Wcs)**2/tot:.4f} cc={fn(Wcc)**2/tot:.4f}")
    print(f"  >>> Wv cross mixing ratio r_cross = {r_cross:.4f}")
    print(f"  ||Wk[:, :32]||_F={fn(Wk[:, :32]):.4f}   "
          f"||Wk[:, 32:]||_F={fn(Wk[:, 32:]):.4f}   "
          f"ratio(in32/in64)={fn(Wk[:, :32])**2/(fn(Wk)**2):.4f}")
    print(f"  ||Wv[:, :32]||_F={fn(Wv[:, :32]):.4f}   "
          f"||Wv[:, 32:]||_F={fn(Wv[:, 32:]):.4f}   "
          f"ratio(in32/in64)={fn(Wv[:, :32])**2/(fn(Wv)**2):.4f}")

    print("\n" + "=" * 104)
    print("7. TEST with dev-selected beta (evaluated ONCE)")
    print("=" * 104)
    print(f"{'beta':>8s}{'HR@5':>9s}{'HR@10':>9s}{'HR@20':>9s}"
          f"{'NDCG@5':>9s}{'NDCG@10':>9s}{'NDCG@20':>9s}")
    for b in BETAS:
        m = test["ranking"][f"beta={b}"]
        mark = "  <<< dev-best" if abs(b - best_beta) < 1e-9 else ""
        print(f"{b:>8}{m['HR@5']:9.4f}{m['HR@10']:9.4f}{m['HR@20']:9.4f}"
              f"{m['NDCG@5']:9.4f}{m['NDCG@10']:9.4f}{m['NDCG@20']:9.4f}{mark}")
    aspcf = test["ranking"]["beta=1.0"]
    sel = test["ranking"][f"beta={best_beta}"]
    print(f"\n  beta=1 (ASPCF) NDCG@5 = {aspcf['NDCG@5']:.4f}")
    print(f"  dev-best beta={best_beta} NDCG@5 = {sel['NDCG@5']:.4f}  "
          f"Δ={(sel['NDCG@5']-aspcf['NDCG@5'])/aspcf['NDCG@5']*100:+.2f}%")

    print("\n" + "=" * 104)
    print("8. fine-grained history->candidate evidence (TEST)")
    print("=" * 104)
    print(f"{'score':>12s}{'HR@5':>9s}{'NDCG@5':>9s}{'P(pos>neg)':>12s}"
          f"{'margin':>10s}{'r vs ASPCF':>13s}{'rho vs ASPCF':>14s}")
    for k in ("g_sem", "g_comp", "g_full", "last_sem", "last_comp", "last_full"):
        m = test["ranking"][k]; p = test["pairwise"][k]
        x = test["_corr"][k].ravel(); y = test["_corr"]["full"].ravel()
        print(f"{k:>12s}{m['HR@5']:9.4f}{m['NDCG@5']:9.4f}"
              f"{p['p_pos_gt_neg']:12.4f}{p['mean_margin']:10.4f}"
              f"{corr(x, y):13.4f}{corr(x, y, 'spearman'):14.4f}")

    # ---------------- 9. buckets ----------------
    pop_item = test["_meta"]["pop_item"].astype(np.int64)
    hlen = test["_meta"]["hlen"].astype(np.float64)
    pop = np.zeros(corpus.n_items, dtype=np.float64)
    cnt = corpus.data_df["train"].groupby("item_id").size()
    pop[cnt.index.to_numpy()] = cnt.to_numpy()
    pop_val = pop[pop_item]
    pop_edges = np.unique(np.percentile(pop[pop > 0], [0, 20, 40, 60, 80, 100]).astype(np.float64))
    hlen_edges = np.array([0, 5, 10, 15, 20, 10 ** 9], dtype=np.float64)

    def bucket(name, edges, values):
        print("\n" + "=" * 104)
        print(f"9. {name}   (HR@5 / NDCG@5)")
        print("=" * 104)
        qb = np.digitize(values, edges[1:-1])
        keys = ("full", "sem", "comp", "g_sem", "g_comp")
        print(f"{'bin':>16s}{'n':>8s}" + "".join(f"{k:>18s}" for k in keys))
        rows = []
        for i in range(len(edges) - 1):
            m_ = qb == i
            if m_.sum() == 0:
                continue
            line = f"{'[' + str(int(edges[i])) + ',' + str(int(edges[i+1])) + ')':>16s}{int(m_.sum()):8d}"
            row = {"bin": f"[{edges[i]:.0f},{edges[i+1]:.0f})", "n": int(m_.sum())}
            for k in keys:
                r = test["_R"][k][m_]
                h = float((r <= 5).mean())
                nd = float(np.mean([ndcg_at(int(x), 5) for x in r]))
                line += f"{h:8.4f}/{nd:.4f}  "
                row[f"{k}_hr5"] = h; row[f"{k}_ndcg5"] = nd
            print(line)
            rows.append(row)
        return rows

    by_pop = bucket("popularity buckets (target item, train freq quintiles)",
                    pop_edges, pop_val)
    by_hlen = bucket("history-length buckets", hlen_edges, hlen)

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f"{ds}_geometry.json")
    json.dump({
        "dataset": ds, "checkpoint": args.ckpt,
        "dev_ranking": dev["ranking"],
        "dev_recompose_err_f64": dev["recompose_max_abs_err_f64"],
        "dev_recompose_err_f32": dev["recompose_max_abs_err_f32"],
        "dev_selected_beta": best_beta,
        "test_recompose_err_f64": test["recompose_max_abs_err_f64"],
        "test_recompose_err_f32": test["recompose_max_abs_err_f32"],
        "test_ranking": test["ranking"],
        "test_pairwise": test["pairwise"],
        "quadrants": test["quadrants"], "quadrant_counts": test["quadrant_counts"],
        "energy": test["energy"],
        "wv_cross_mixing_ratio": r_cross,
        "wv_block_shares": {"ss": fn(Wss) ** 2 / tot, "sc": fn(Wsc) ** 2 / tot,
                            "cs": fn(Wcs) ** 2 / tot, "cc": fn(Wcc) ** 2 / tot},
        "by_popularity": by_pop, "by_history_length": by_hlen,
        "betas": BETAS,
    }, open(path, "w"), indent=2)
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
