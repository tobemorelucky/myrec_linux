# -*- coding: UTF-8 -*-
"""
Paired benefit analysis: attention-teacher vs responsibility-teacher checkpoints.

Uses the SAME dev samples and candidate order for both checkpoints.
ALL structural features are computed from the attention checkpoint only
(avoids post-treatment bias from responsibility training).

Usage:
  python tools/analyze_caisd_teacher_benefit.py \
    --attention_checkpoint <attn_ckpt.pt> \
    --responsibility_checkpoint <resp_ckpt.pt> \
    --teacher_path ./data/beauty/handled/llmmi_proto32_sr512.pkl \
    --dataset beauty --phase dev \
    --max_batches 50 --output_dir <dir>
"""

import argparse, json, logging, math, os, pickle, sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch
import torch.nn.functional as F

from models.sequential.LLMMIRecCAISD import LLMMIRecCAISD


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attention_checkpoint", type=str, required=True)
    p.add_argument("--responsibility_checkpoint", type=str, required=True)
    p.add_argument("--teacher_path", type=str, required=True)
    p.add_argument("--dataset", type=str, default="beauty")
    p.add_argument("--phase", type=str, default="dev", choices=["train", "dev", "test"])
    p.add_argument("--max_batches", type=int, default=50)
    p.add_argument("--output_dir", type=str, default="./caisd_benefit")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def normalized_entropy(probs, dim=-1, eps=1e-8):
    n = probs.shape[dim]
    if n <= 1:
        return torch.zeros_like(probs.sum(dim=dim))
    H = -(probs * torch.log(probs + eps)).sum(dim=dim)
    return H / math.log(n)


def per_user_k_cos_batch(x):
    B, kk = x.shape[:2]
    if kk <= 1:
        return torch.zeros(B)
    vals = []
    for b in range(B):
        v = x[b].reshape(kk, -1)
        vn = F.normalize(v, dim=-1, eps=1e-8)
        cm = vn @ vn.t()
        eye = torch.eye(kk, device=cm.device, dtype=cm.dtype)
        off = (cm * (1 - eye)).sum() / max(kk * (kk - 1), 1)
        vals.append(float(off))
    return torch.tensor(vals)


def js_divergence(P, Q, eps=1e-8):
    """Per-sample (B,K) Jensen-Shannon, normalized by log(2)."""
    P = P.clamp(min=eps)
    Q = Q.clamp(min=eps)
    M = 0.5 * (P + Q)
    kl_pm = (P * torch.log(P / M)).sum(dim=-1)
    kl_qm = (Q * torch.log(Q / M)).sum(dim=-1)
    js = 0.5 * (kl_pm + kl_qm)
    return js / math.log(2)


def build_model(checkpoint, dataset, teacher_path, device):
    state = torch.load(checkpoint, map_location="cpu")
    emb_size = state["position_emb.weight"].shape[1]
    K_emb = state["extractor.query"].shape[0]
    attn_size = state["extractor.Wq.weight"].shape[0]

    class D: pass
    ma = D()
    ma.device = torch.device(device)
    ma.model_path = checkpoint
    ma.buffer = 1; ma.history_max = 20; ma.num_neg = 1; ma.test_all = 0
    ma.emb_size = emb_size; ma.attn_size = attn_size; ma.K = K_emb
    ma.item_encoder = "aspcf"
    ma.llm_emb_path = f"./data/{dataset}/handled/llm_table_pca1536.pkl"
    ma.adapter_hidden = 256; ma.adapter_activation = "gelu"; ma.adapter_use_ln = 0
    ma.gamma_init = 0.1; ma.gamma_trainable = 0
    ma.semantic_rank = 512
    ma.semantic_dim = 32; ma.semantic_hidden = 128
    ma.complement_dim = 32; ma.tail_hidden = 64
    ma.complement_hidden = 64; ma.gate_hidden = 64
    ma.aspcf_gate_mode = "basic"
    ma.lambda_relation = 0.01
    ma.relation_sample_size = 128
    ma.relation_teacher_temp = 0.1; ma.relation_student_temp = 0.1
    ma.semantic_teacher_path = teacher_path
    ma.semantic_distill_mode = "uniform"
    ma.semantic_teacher_mode = "attention"
    ma.semantic_responsibility_alpha = 0.5
    ma.lambda_interest_semantic = 0.01
    ma.dropout = 0.1

    corpus = pickle.load(open(f"./data/{dataset}/SeqReader.pkl", "rb"))
    model = LLMMIRecCAISD(ma, corpus).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, corpus


def compute_ndcg(ranks, k):
    """ranks: [B] 1-indexed. NDCG@k with single relevant item."""
    hit = ranks <= k
    return hit.float() / torch.log2(ranks.float() + 1.0)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    logging.info("Loading attention checkpoint...")
    model_attn, corpus = build_model(args.attention_checkpoint, args.dataset,
                                      args.teacher_path, device)
    logging.info("Loading responsibility checkpoint...")
    model_resp, _ = build_model(args.responsibility_checkpoint, args.dataset,
                                 args.teacher_path, device)

    tdata = pickle.load(open(args.teacher_path, "rb"))
    t_assign = torch.tensor(tdata["soft_assignments"], dtype=torch.float32).to(device)

    ds = model_attn.Dataset(model_attn, corpus, args.phase)
    ds.prepare()
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0,
                    pin_memory=False, collate_fn=ds.collate_batch)

    # Per-user accumulators
    user_rows = []
    K = model_attn.K

    batch_count = 0
    with torch.inference_mode():
        for batch in dl:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            history = batch["history_items"]
            lengths = batch["lengths"]
            B, L = history.shape

            # Both models on identical samples/candidates
            out_attn = model_attn(batch, return_intermediate=True)
            out_resp = model_resp(batch, return_intermediate=False)

            # Ranks
            pos_score_a = out_attn["prediction"][:, 0:1]
            rank_a = (out_attn["prediction"] > pos_score_a).sum(dim=1) + 1
            pos_score_r = out_resp["prediction"][:, 0:1]
            rank_r = (out_resp["prediction"] > pos_score_r).sum(dim=1) + 1

            ndcg5_a = compute_ndcg(rank_a, 5)
            ndcg10_a = compute_ndcg(rank_a, 10)
            ndcg5_r = compute_ndcg(rank_r, 5)
            ndcg10_r = compute_ndcg(rank_r, 10)

            delta_rank = rank_a.float() - rank_r.float()       # >0 resp better
            delta_ndcg5 = ndcg5_r - ndcg5_a
            delta_ndcg10 = ndcg10_r - ndcg10_a

            # ---- Features from ATTENTION checkpoint only ----
            A = out_attn["attention_maps"].detach()            # [B, K, L]
            valid = (history > 0).float()                      # [B, L]

            attn_cos = per_user_k_cos_batch(A)                 # [B]

            A_masked = A * valid.unsqueeze(1)
            R = A_masked / A_masked.sum(dim=1, keepdim=True).clamp_min(1e-8)

            resp_ent_user = []
            for b in range(B):
                vl = int(lengths[b].item())
                if vl <= 0:
                    resp_ent_user.append(0.0)
                    continue
                R_item = R[b, :, :vl].t()                      # [vl, K]
                ents = normalized_entropy(R_item.clamp(min=1e-8), dim=-1)
                resp_ent_user.append(float(ents.mean()))

            # Teachers from same anchor attention
            Q = t_assign[history]                              # [B, L, 32]
            T_attn = torch.bmm(A, Q)
            T_attn = T_attn / T_attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            T_attn = T_attn.detach()

            W_resp = A_masked * R
            W_resp = W_resp / W_resp.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            T_resp = torch.bmm(W_resp, Q)
            T_resp = T_resp / T_resp.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            T_resp = T_resp.detach()

            js_shift = js_divergence(T_attn, T_resp).mean(dim=-1)       # [B]
            cos_shift = 1.0 - F.cosine_similarity(
                T_attn.reshape(B * K, -1), T_resp.reshape(B * K, -1), dim=-1
            ).reshape(B, K).mean(dim=-1)                                 # [B]

            t_attn_cos = per_user_k_cos_batch(T_attn)                    # [B]
            t_resp_cos = per_user_k_cos_batch(T_resp)
            cos_reduction = t_attn_cos - t_resp_cos

            ent_attn_t = normalized_entropy(T_attn.clamp(min=1e-8), dim=-1)
            ent_resp_t = normalized_entropy(T_resp.clamp(min=1e-8), dim=-1)

            for b in range(B):
                user_rows.append({
                    "history_length": int(lengths[b].item()),
                    "attention_inter_interest_cos": float(attn_cos[b].item()),
                    "responsibility_entropy": float(resp_ent_user[b]),
                    "teacher_js_shift": float(js_shift[b].item()),
                    "teacher_cos_shift": float(cos_shift[b].item()),
                    "teacher_cos_reduction": float(cos_reduction[b].item()),
                    "attention_teacher_entropy": float(ent_attn_t[b].mean().item()),
                    "responsibility_teacher_entropy": float(ent_resp_t[b].mean().item()),
                    "delta_ndcg5": float(delta_ndcg5[b].item()),
                    "delta_ndcg10": float(delta_ndcg10[b].item()),
                    "delta_rank": float(delta_rank[b].item()),
                    "rank_attention": float(rank_a[b].item()),
                    "rank_responsibility": float(rank_r[b].item()),
                })

            batch_count += 1
            if args.max_batches > 0 and batch_count >= args.max_batches:
                break

    n = len(user_rows)
    logging.info(f"Total users analyzed: {n}")

    # ---- TSV ----
    tsv_path = os.path.join(args.output_dir, "user_factors.tsv")
    keys = list(user_rows[0].keys())
    with open(tsv_path, "w") as f:
        f.write("\t".join(keys) + "\n")
        for r in user_rows:
            f.write("\t".join(str(r[k]) for k in keys) + "\n")

    # ---- Factor arrays ----
    factors = ["history_length", "attention_inter_interest_cos",
               "responsibility_entropy", "teacher_js_shift", "teacher_cos_shift",
               "teacher_cos_reduction", "attention_teacher_entropy"]
    targets = ["delta_ndcg5", "delta_ndcg10", "delta_rank"]

    data = {k: np.array([r[k] for r in user_rows], dtype=np.float64) for k in keys}

    # ---- Summary ----
    summary = {
        "num_users": n,
        "mean_delta_ndcg5": round(float(data["delta_ndcg5"].mean()), 6),
        "mean_delta_ndcg10": round(float(data["delta_ndcg10"].mean()), 6),
        "mean_delta_rank": round(float(data["delta_rank"].mean()), 4),
        "responsibility_win_rate": round(float((data["delta_ndcg5"] > 0).mean()), 4),
    }
    for f in factors:
        a = data[f]
        a = a[np.isfinite(a)]
        summary[f] = {
            "mean": round(float(a.mean()), 6),
            "std": round(float(a.std()), 6),
            "p25": round(float(np.percentile(a, 25)), 6),
            "median": round(float(np.percentile(a, 50)), 6),
            "p75": round(float(np.percentile(a, 75)), 6),
        }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Spearman correlations ----
    from scipy.stats import spearmanr
    corr_out = {}
    for f in factors:
        corr_out[f] = {}
        for t in targets:
            valid = np.isfinite(data[f]) & np.isfinite(data[t])
            if valid.sum() < 3:
                corr_out[f][t] = None
            else:
                rho, p = spearmanr(data[f][valid], data[t][valid])
                corr_out[f][t] = {"rho": round(float(rho), 4), "p": round(float(p), 6)}
    with open(os.path.join(args.output_dir, "correlations.json"), "w") as f:
        json.dump(corr_out, f, indent=2)

    # ---- Quartile analysis ----
    quartiles = {}
    for f in factors:
        a = data[f]
        finite = np.isfinite(a)
        a_f = a[finite]
        delta5 = data["delta_ndcg5"][finite]
        delta10 = data["delta_ndcg10"][finite]
        delta_r = data["delta_rank"][finite]
        if a_f.size == 0:
            continue
        qs = np.percentile(a_f, [25, 50, 75])
        bounds = [(-np.inf, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], np.inf)]
        quartiles[f] = []
        for qi, (lo, hi) in enumerate(bounds):
            mask = (a_f > lo) & (a_f <= hi) if qi > 0 else (a_f <= hi)
            if qi == 3:
                mask = a_f > lo
            cnt = mask.sum()
            if cnt == 0:
                continue
            quartiles[f].append({
                "quartile": f"Q{qi + 1}",
                "user_count": int(cnt),
                "mean_delta_ndcg5": round(float(delta5[mask].mean()), 6),
                "mean_delta_ndcg10": round(float(delta10[mask].mean()), 6),
                "mean_delta_rank": round(float(delta_r[mask].mean()), 4),
                "responsibility_win_rate": round(float((delta5[mask] > 0).mean()), 4),
            })
    with open(os.path.join(args.output_dir, "quartiles.json"), "w") as f:
        json.dump(quartiles, f, indent=2)

    logging.info(f"Saved: {tsv_path}, summary.json, correlations.json, quartiles.json")
    print(f"Summary: {json.dumps(summary, indent=2)}")
    print(f"Correlations: {json.dumps(corr_out, indent=2)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
