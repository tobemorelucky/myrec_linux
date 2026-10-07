# -*- coding: UTF-8 -*-
"""
Chapter 4 Phase 0 diagnostic — where does the remaining gap come from?

NO TRAINING. Loads existing checkpoints and analyses the test split along six
axes, comparing PoMRec / LLMMIRec-ID / LLMMIRecASPCF (+ HSDIR where the
checkpoint exists).

  1. HR/NDCG bucketed by target-item popularity (train frequency)
  2. HR/NDCG bucketed by user history length
  3. interest pairwise cosine / effective rank
  4. attention entropy
  5. interest aggregation weight entropy / active interests
  6. positive-vs-negative score margin

Fairness note: the checkpoints come from different training configurations
(PoMRec bs=256 lr=0.002/0.001; LLMMIRec-ID bs=256 lr=0.001; ASPCF bs=1024
lr=0.004/0.001). They are the best available per-model checkpoints, not a
controlled comparison; configs are recorded in the output.

Usage:
  python tools/diagnose_ch4_phase0.py --dataset beauty --device cuda
  python tools/diagnose_ch4_phase0.py --dataset ml-1m  --device cuda
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
from models.sequential.PoMRec import PoMRec                   # noqa: E402
from models.sequential.LLMMIRec import LLMMIRec               # noqa: E402
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF     # noqa: E402
from models.sequential.LLMMIRecHSDIR import LLMMIRecHSDIR     # noqa: E402

LLM = "./data/{ds}/handled/llm_table_pca1536.pkl"


def spec_table(ds):
    """Per-dataset model specs: class, checkpoint, and the args needed to build."""
    base_common = dict(emb_size=64, attn_size=64, K=4, history_max=20, num_neg=1,
                       test_all=0, buffer=1, dropout=0.1, llm_emb_path=LLM.format(ds=ds))
    specs = {}

    # ---- PoMRec (own backbone) ----
    # NOTE: PoMRec's standard config is dataset-specific: beauty uses K=4/lamb=4.0,
    # ml-1m uses K=2/lamb=1.0. The LLMMIRec family uses K=4 everywhere, so the
    # number of interests is NOT matched between PoMRec and the others on ml-1m.
    pom = {"beauty": dict(K=4, lamb=4.0, lr=0.002),
           "ml-1m":  dict(K=2, lamb=1.0, lr=0.001)}[ds]
    specs["PoMRec"] = dict(
        cls=PoMRec,
        ckpt=f"new_model/pomrec_standard/{ds}/PoMRec_seed42.pt",
        cfg=f"lr={pom['lr']}, bs=256, attn_size=8, n_layers=2, prompt_num=3, "
            f"K={pom['K']}, lamb={pom['lamb']}, dropout=0.0",
        args=dict(emb_size=64, attn_size=8, K=pom["K"], history_max=20, num_neg=1,
                  test_all=0, buffer=1, dropout=0.0, prompt_num=3, n_layers=2,
                  lamb=pom["lamb"],
                  use_llmemb=0, llm_emb_path="", freeze_emb=False, llm_fuse=0,
                  alpha=0.01, tau=0.2, rat_alpha_warmup_steps=0, align_on="pos",
                  align_sample_k=0, init_ckpt="", init_strict=0,
                  gamma_init=0.05, gamma_trainable=1, srs_emb_path=""),
    )

    # ---- LLMMIRec-ID ----
    a = dict(base_common)
    specs["LLMMIRec-ID"] = dict(
        cls=LLMMIRec, ckpt=f"new_model/llmmirec_phase0/{ds}/id/LLMMIRec_id_seed42.pt",
        cfg="lr=0.001, bs=256, dropout=0.1",
        args=dict(a, item_encoder="id", adapter_hidden=256, adapter_activation="gelu",
                  adapter_use_ln=0, gamma_init=0.1, gamma_trainable=0,
                  semantic_rank=512, semantic_dim=32, semantic_hidden=128, complement_dim=32,
                  tail_hidden=64, complement_hidden=64, gate_hidden=64, aspcf_gate_mode="basic",
                  interest_query_mode="learnable", prototype_path="",
                  lambda_relation=0.0, relation_sample_size=128,
                  relation_teacher_temp=0.1, relation_student_temp=0.1),
    )

    # ---- ASPCF ----
    specs["ASPCF"] = dict(
        cls=LLMMIRecASPCF,
        ckpt=f"new_model/llmmirec_aspcf_phase2/{ds}/seed42/LLMMIRecASPCF_seed42.pt",
        cfg=f"lr={'0.004' if ds=='beauty' else '0.001'}, bs=1024, dropout=0.1",
        args=dict(base_common, item_encoder="aspcf", adapter_hidden=256,
                  adapter_activation="gelu", adapter_use_ln=0, gamma_init=0.1,
                  gamma_trainable=0, semantic_rank=512, semantic_dim=32, semantic_hidden=128,
                  complement_dim=32, tail_hidden=64, complement_hidden=64, gate_hidden=64,
                  aspcf_gate_mode="basic", lambda_relation=0.01, relation_sample_size=128,
                  relation_teacher_temp=0.1, relation_student_temp=0.1),
    )

    # ---- HSDIR (different config; only if a matching checkpoint exists) ----
    hs = f"new_model/llmmirec_hsdir_phase1/{ds}"
    cand = None
    if os.path.isdir(hs):
        for lbl in sorted(os.listdir(hs)):
            p = os.path.join(hs, lbl, f"LLMMIRecHSDIR_{lbl}_seed42.pt")
            if os.path.exists(p):
                cand = (lbl, p)          # keep the last one found; label recorded
    if cand:
        lbl, p = cand
        specs["HSDIR"] = dict(
            cls=LLMMIRecHSDIR, ckpt=p,
            cfg=f"lr={'0.008' if ds=='beauty' else '0.002'}, bs=1024 (DIFFERENT CONFIG)",
            args=dict(base_common, item_encoder="aspcf", adapter_hidden=256,
                      adapter_activation="gelu", adapter_use_ln=0, gamma_init=0.1,
                      gamma_trainable=0, semantic_rank=512, semantic_dim=32,
                      semantic_hidden=128, complement_dim=32, tail_hidden=64,
                      complement_hidden=64, gate_hidden=64, aspcf_gate_mode="basic",
                      lambda_relation=0.01, relation_sample_size=128,
                      relation_teacher_temp=0.1, relation_student_temp=0.1,
                      lambda_hsr=0.0, hsr_teacher_mode="hierarchical", hsr_student_temp=1.0,
                      hsr_loss_mode="absolute", hsr_margin=0.1, hsr_pair_margin=0.1,
                      hsr_confidence_mode="semantic", hsr_route_source="raw",
                      teacher_path="", aggregation_mode="base", support_beta=1.0),
        )
    return specs


# =========================
#  PoMRec internals (replicated, with a runtime self-check)
# =========================

def pomrec_attention(model, history, lengths):
    """Replicate PoMRec's MultiInterestExtractor attention.

    Returned attention is verified against the extractor's own output in
    check_pomrec_replication(); callers must not trust it silently.
    """
    ie = model.interest_extractor
    B, L = history.shape
    dev = history.device
    valid_his = (history > 0).long()
    his = ie.get_item_emb(history)
    len_range = torch.arange(ie.max_his, device=dev)
    position = (lengths[:, None] - len_range[None, :L]) * valid_his
    his = his + ie.p_embeddings(position)
    # NOTE: PoMRec hardcodes self.max_prompt = 5 (not prompt_num); the padded
    # prompt block is max_prompt wide and is always fully valid.
    valid_cat = torch.cat([valid_his, torch.ones([B, ie.max_prompt], device=dev)], dim=1)
    p1 = torch.cat([ie.prompt_pad.to(dev), ie.prompt1.weight], dim=0).unsqueeze(0).expand(B, -1, -1)
    hvp = torch.cat([his, p1], dim=1)
    attn = ie.value2attn(ie.W2(ie.W1(hvp).tanh()), valid_cat)     # [B, K, L+p]
    return attn, hvp, valid_cat


def check_pomrec_replication(model, history, lengths, interest_vectors):
    """Verify the replicated attention reproduces the extractor's output."""
    attn, hvp, _ = pomrec_attention(model, history, lengths)
    isum = (hvp[:, None, :, :] * attn[:, :, :, None]).sum(-2)
    var = []
    for kk in range(model.interest_extractor.K):
        d = (hvp - isum[:, kk:kk + 1, :]) ** 2
        var.append(torch.sqrt(torch.matmul(attn[:, kk:kk + 1, :], d)))
    full = isum + model.interest_extractor.lamb * torch.cat(var, 1)
    return float((full - interest_vectors).abs().max().item()), attn


# =========================
#  Metrics
# =========================

def ndcg_at_k(rank, k):
    return (1.0 / math.log2(rank + 1)) if rank <= k else 0.0


def entropy(p, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > eps]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def eff_rank_mat(vectors):
    """Entropy-based effective rank of a [K, D] matrix (centered)."""
    X = np.asarray(vectors, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    p = s ** 2
    tot = p.sum()
    if tot <= 0:
        return float("nan")
    p = p / tot
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


# =========================
#  Main
# =========================

def run_model(ds, name, spec, corpus, pop, device, verbose=True):
    cls = spec["cls"]
    class A:
        pass
    a = A()
    for k, v in spec["args"].items():
        setattr(a, k, v)
    a.device = torch.device(device)
    a.model_path = spec["ckpt"]
    a.batch_size = 256
    a.eval_batch_size = 256
    a.num_workers = 0

    model = cls(a, corpus)
    state = torch.load(spec["ckpt"], map_location="cpu")
    if isinstance(state, dict):
        for w in ("state_dict", "model", "net"):
            if w in state and isinstance(state[w], dict):
                state = state[w]
                break
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.device = torch.device(device)
    model.eval()

    dset = model.Dataset(model, corpus, "test")
    dset.prepare()
    dl = DataLoader(dset, batch_size=256, shuffle=False, num_workers=0,
                    pin_memory=False, collate_fn=dset.collate_batch)

    is_pomrec = isinstance(model, PoMRec)
    rec = {k: [] for k in ("pop", "hlen", "rank", "hr5", "hr10", "hr20",
                           "ndcg5", "ndcg10", "ndcg20", "margin",
                           "icos", "ier", "attn_ent", "w_ent", "active_k", "w_top1")}

    checked, repl_err = False, float("nan")
    n = 0
    with torch.inference_mode():
        for batch in dl:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            hist = batch["history_items"]
            lengths = batch["lengths"]
            tgt = batch["item_id"][:, 0]

            if is_pomrec:
                iv, dv = model.interest_extractor(hist, lengths)
                i_vecs = model.interest_extractor.get_item_emb(batch["item_id"])
                w = model.proj(dv).softmax(-1)
                pred = ((iv * w[:, :, None]).sum(-2)[:, None, :] * i_vecs).sum(-1)
                if not checked:
                    repl_err, attn = check_pomrec_replication(model, hist, lengths, iv)
                    checked = True
                else:
                    attn, _, _ = pomrec_attention(model, hist, lengths)
            else:
                out = model(batch, return_intermediate=True)
                pred = out["prediction"]
                iv = out["interest_vectors"]
                w = out["interest_weights"]
                attn = out["attention_maps"]

            pos = pred[:, 0]
            neg = pred[:, 1:]
            margin = (pos - neg.max(dim=1).values).cpu().numpy()
            rank = (pred > pred[:, 0:1]).sum(dim=1).cpu().numpy() + 1

            ivn = iv.detach().float().cpu().numpy()
            an = attn.detach().float().cpu().numpy()
            wn = w.detach().float().cpu().numpy()
            hn = hist.cpu().numpy()
            ln = lengths.cpu().numpy()
            pn = pop[tgt.cpu().numpy()]

            for b in range(len(rank)):
                K = ivn.shape[1]
                sub = ivn[b]
                cn = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-12)
                off = cn @ cn.T
                iu = np.triu_indices(K, 1)
                rec["icos"].append(float(off[iu].mean()))
                rec["ier"].append(eff_rank_mat(sub))

                Lv = int(ln[b])
                a_row = an[b, :, :Lv] if Lv > 0 else an[b, :, :1]
                rec["attn_ent"].append(float(np.mean([entropy(a_row[k]) for k in range(K)])))

                wb = wn[b]
                wb = wb / (wb.sum() + 1e-12)
                rec["w_ent"].append(entropy(wb))
                rec["active_k"].append(float(1.0 / (np.square(wb).sum() + 1e-12)))
                rec["w_top1"].append(float(wb.max()))

                r = int(rank[b])
                rec["rank"].append(r)
                rec["hr5"].append(1.0 if r <= 5 else 0.0)
                rec["hr10"].append(1.0 if r <= 10 else 0.0)
                rec["hr20"].append(1.0 if r <= 20 else 0.0)
                rec["ndcg5"].append(ndcg_at_k(r, 5))
                rec["ndcg10"].append(ndcg_at_k(r, 10))
                rec["ndcg20"].append(ndcg_at_k(r, 20))
                rec["margin"].append(float(margin[b]))
                rec["pop"].append(float(pn[b]))
                rec["hlen"].append(Lv)
            n += 1

    if verbose:
        if is_pomrec:
            print(f"    [self-check] PoMRec attention replication max_err = {repl_err:.2e}"
                  f"  {'OK' if repl_err < 1e-3 else '*** MISMATCH ***'}")
    return {k: np.asarray(v) for k, v in rec.items()}, repl_err


def summarize(d, pop_edges=None, hlen_edges=None):
    out = {}
    out["n"] = int(len(d["rank"]))
    for m in ("hr5", "hr10", "hr20", "ndcg5", "ndcg10", "ndcg20", "margin"):
        out[m] = float(d[m].mean())
    out["pop_median"] = float(np.median(d["pop"]))
    out["hlen_mean"] = float(d["hlen"].mean())
    for m in ("icos", "ier", "attn_ent", "w_ent", "active_k", "w_top1"):
        v = d[m][np.isfinite(d[m])]
        out[m] = float(v.mean()) if v.size else float("nan")
        # percentiles matter here: a collapsed multi-interest head is bimodal,
        # and the mean alone hides it
        out[m + "_p05"] = float(np.percentile(v, 5)) if v.size else float("nan")
        out[m + "_p50"] = float(np.percentile(v, 50)) if v.size else float("nan")
        out[m + "_p95"] = float(np.percentile(v, 95)) if v.size else float("nan")

    # popularity quintiles
    if pop_edges is not None:
        q = np.digitize(d["pop"], pop_edges[1:-1])
        out["by_pop"] = []
        for i in range(len(pop_edges) - 1):
            m = q == i
            out["by_pop"].append(dict(
                bin=f"[{pop_edges[i]:.0f},{pop_edges[i+1]:.0f})",
                n=int(m.sum()),
                hr5=float(d["hr5"][m].mean()) if m.sum() else float("nan"),
                ndcg5=float(d["ndcg5"][m].mean()) if m.sum() else float("nan"),
                ndcg10=float(d["ndcg10"][m].mean()) if m.sum() else float("nan"),
                ndcg20=float(d["ndcg20"][m].mean()) if m.sum() else float("nan"),
            ))
    # history-length bins
    if hlen_edges is not None:
        q = np.digitize(d["hlen"], hlen_edges[1:-1])
        out["by_hlen"] = []
        for i in range(len(hlen_edges) - 1):
            m = q == i
            out["by_hlen"].append(dict(
                bin=f"[{hlen_edges[i]:.0f},{hlen_edges[i+1]:.0f})",
                n=int(m.sum()),
                hr5=float(d["hr5"][m].mean()) if m.sum() else float("nan"),
                ndcg5=float(d["ndcg5"][m].mean()) if m.sum() else float("nan"),
                ndcg10=float(d["ndcg10"][m].mean()) if m.sum() else float("nan"),
                ndcg20=float(d["ndcg20"][m].mean()) if m.sum() else float("nan"),
            ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["beauty", "ml-1m"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--models", type=str, default="PoMRec,LLMMIRec-ID,ASPCF,HSDIR")
    ap.add_argument("--output_dir", type=str, default="./diagnostics_ch4_phase0")
    args = ap.parse_args()

    ds = args.dataset
    os.makedirs(args.output_dir, exist_ok=True)
    corpus = pickle.load(open(f"./data/{ds}/SeqReader.pkl", "rb"))

    # positive-item popularity from TRAIN only (no leakage)
    df = corpus.data_df["train"]
    cnt = df.groupby("item_id").size()
    pop = np.zeros(corpus.n_items, dtype=np.float64)
    pop[cnt.index.to_numpy()] = cnt.to_numpy()

    specs = spec_table(ds)
    want = [m.strip() for m in args.models.split(",") if m.strip()]

    print("=" * 100)
    print(f"Chapter 4 Phase 0 diagnostic — {ds} (test split, no training)")
    print("=" * 100)

    pop_edges = np.percentile(pop[pop > 0], [0, 20, 40, 60, 80, 100])
    pop_edges = np.unique(pop_edges.astype(np.float64))
    hlen_edges = np.array([0, 5, 10, 15, 20, 10**9], dtype=np.float64)

    results = {}
    for name in want:
        if name not in specs:
            print(f"\n--- {name}: 无可用 checkpoint，跳过")
            continue
        spec = specs[name]
        print(f"\n--- {name}   [{spec['cfg']}]")
        print(f"    ckpt: {spec['ckpt']}")
        if not os.path.exists(spec["ckpt"]):
            print("    *** checkpoint 缺失，跳过 ***")
            continue
        d, err = run_model(ds, name, spec, corpus, pop, args.device)
        s = summarize(d, pop_edges, hlen_edges)
        if isinstance(spec["cls"], type) and spec["cls"] is PoMRec:
            s["pomrec_replication_max_err"] = err
        results[name] = dict(cfg=spec["cfg"], ckpt=spec["ckpt"], summary=s)
        print(f"    n={s['n']}  HR@5={s['hr5']:.4f}  NDCG@5={s['ndcg5']:.4f}  NDCG@10={s['ndcg10']:.4f}")
        print(f"    interest_cos={s['icos']:.4f} [p05={s['icos_p05']:.3f} p50={s['icos_p50']:.3f} "
              f"p95={s['icos_p95']:.3f}]  eff_rank={s['ier']:.3f} "
              f"[p05={s['ier_p05']:.2f} p50={s['ier_p50']:.2f} p95={s['ier_p95']:.2f}]")
        print(f"    attn_ent={s['attn_ent']:.4f}  w_ent={s['w_ent']:.4f} "
              f"[p05={s['w_ent_p05']:.3f} p50={s['w_ent_p50']:.3f} p95={s['w_ent_p95']:.3f}]  "
              f"active_k={s['active_k']:.3f} [p50={s['active_k_p50']:.2f} "
              f"p95={s['active_k_p95']:.2f}]  margin={s['margin']:.4f}")

    out = os.path.join(args.output_dir, f"{ds}.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\n[saved] {out}")

    # ---- readable comparison ----
    print("\n" + "=" * 100)
    print(f"{ds}: 总体指标")
    print("=" * 100)
    hdr = f"{'model':14s}{'HR@5':>9s}{'NDCG@5':>9s}{'NDCG@10':>9s}{'NDCG@20':>9s}" \
          f"{'icos':>8s}{'effR':>7s}{'attnH':>8s}{'wH':>8s}{'actK':>7s}{'margin':>9s}"
    print(hdr)
    for name, r in results.items():
        s = r["summary"]
        print(f"{name:14s}{s['hr5']:9.4f}{s['ndcg5']:9.4f}{s['ndcg10']:9.4f}{s['ndcg20']:9.4f}"
              f"{s['icos']:8.4f}{s['ier']:7.3f}{s['attn_ent']:8.4f}{s['w_ent']:8.4f}"
              f"{s['active_k']:7.3f}{s['margin']:9.4f}")

    # structural metrics: percentiles expose a collapsed multi-interest head
    print("\n结构指标分位数（判断多兴趣头是否塌缩）")
    print(f"{'model':14s}{'icos p05/p50/p95':>26s}{'effR p05/p50/p95':>24s}"
          f"{'wEnt p05/p50/p95':>26s}{'actK p50/p95':>16s}")
    for name, r in results.items():
        s = r["summary"]

        def trio(key):
            return f"{s[key + '_p05']:.2f}/{s[key + '_p50']:.2f}/{s[key + '_p95']:.2f}"

        print(f"{name:14s}{trio('icos'):>26s}{trio('ier'):>24s}"
              f"{trio('w_ent'):>26s}"
              f"{s['active_k_p50']:.2f}/{s['active_k_p95']:.2f}".rjust(16))

    for key, title in (("by_pop", "按 positive item 流行度分桶（train 频次五分位）"),
                       ("by_hlen", "按 user history 长度分桶")):
        print("\n" + "=" * 100)
        print(f"{ds}: {title}")
        print("=" * 100)
        print(f"{'model':14s}{'bin':>16s}{'n':>8s}{'HR@5':>9s}{'NDCG@5':>9s}{'NDCG@10':>9s}{'NDCG@20':>9s}")
        for name, r in results.items():
            for row in r["summary"].get(key, []):
                print(f"{name:14s}{row['bin']:>16s}{row['n']:8d}{row['hr5']:9.4f}"
                      f"{row['ndcg5']:9.4f}{row['ndcg10']:9.4f}{row['ndcg20']:9.4f}")
            print()


if __name__ == "__main__":
    main()
