# -*- coding: UTF-8 -*-
"""
Analyze which samples benefit from HSDIR vs baseline on the dev set.

Per-sample metrics + quartile group analysis + Spearman correlations.

Usage (Beauty):
  python tools/analyze_hsdir_benefit_factors.py \
    --baseline_checkpoint <baseline.pt> \
    --hsdir_checkpoint <hsdir.pt> \
    --teacher_path ./data/beauty/handled/llmmi_hier_proto32_8_sr512.pkl \
    --dataset beauty --output_dir <dir>

Usage (ML-1M):
  python tools/analyze_hsdir_benefit_factors.py \
    --baseline_checkpoint <baseline.pt> \
    --hsdir_checkpoint <hsdir.pt> \
    --teacher_path ./data/ml-1m/handled/llmmi_hier_proto32_8_sr512.pkl \
    --dataset ml-1m --output_dir <dir>
"""

import argparse, json, logging, math, os, pickle, sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch
import torch.nn.functional as F

from models.sequential.LLMMIRecHSDIR import LLMMIRecHSDIR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_checkpoint", type=str, required=True)
    p.add_argument("--hsdir_checkpoint", type=str, required=True)
    p.add_argument("--teacher_path", type=str, required=True)
    p.add_argument("--dataset", type=str, default="beauty")
    p.add_argument("--max_batches", type=int, default=0, help="0 = all dev")
    p.add_argument("--output_dir", type=str, default="./hsdir_benefit")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def normalized_entropy(probs, dim=-1, eps=1e-8):
    K = probs.shape[dim]
    if K <= 1:
        return torch.zeros_like(probs.sum(dim=dim))
    H = -(probs * torch.log(probs + eps)).sum(dim=dim)
    return H / math.log(K)


def per_user_k_cos(x):
    B, kk = x.shape[:2]
    if kk <= 1:
        return torch.zeros(B)
    vals = []
    for b in range(B):
        v = x[b].reshape(kk, -1)
        vn = F.normalize(v, dim=-1, eps=1e-8)
        cm = vn @ vn.t()
        off = (cm * (1 - torch.eye(kk))).sum() / max(kk * (kk - 1), 1)
        vals.append(float(off))
    return torch.tensor(vals)


def build_model(checkpoint, dataset, teacher_path, device):
    state = torch.load(checkpoint, map_location="cpu")
    emb_size = state["position_emb.weight"].shape[1]
    K_emb = state["extractor.query"].shape[0]
    attn_size = state["extractor.Wq.weight"].shape[0]
    has_log_gamma = "item_encoder.log_gamma" in state

    class D: pass
    ma = D()
    ma.device = torch.device(device)
    ma.model_path = checkpoint
    ma.buffer = 1; ma.history_max = 20; ma.num_neg = 1; ma.test_all = 0
    ma.emb_size = emb_size; ma.attn_size = attn_size; ma.K = K_emb
    ma.item_encoder = "aspcf"
    ma.llm_emb_path = f"./data/{dataset}/handled/llm_table_pca1536.pkl"
    ma.adapter_hidden = 256; ma.adapter_activation = "gelu"; ma.adapter_use_ln = 0
    ma.gamma_init = 0.1; ma.gamma_trainable = int(has_log_gamma)
    ma.semantic_rank = 512
    ma.semantic_dim = 32; ma.semantic_hidden = 128
    ma.complement_dim = 32; ma.tail_hidden = 64
    ma.complement_hidden = 64; ma.gate_hidden = 64
    ma.aspcf_gate_mode = "basic"
    ma.lambda_relation = 0.01
    ma.relation_sample_size = 128
    ma.relation_teacher_temp = 0.1; ma.relation_student_temp = 0.1
    ma.lambda_hsr = 0.01
    ma.hsr_teacher_mode = "hierarchical"
    ma.hsr_student_temp = 1.0
    ma.hsr_loss_mode = "absolute"; ma.hsr_margin = 0.1
    ma.hsr_confidence_mode = "semantic"
    ma.teacher_path = teacher_path
    ma.aggregation_mode = "base"; ma.support_beta = 1.0
    ma.dropout = 0.1

    corpus = pickle.load(open(f"./data/{dataset}/SeqReader.pkl", "rb"))
    model = LLMMIRecHSDIR(ma, corpus).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, corpus


def compute_rank(prediction, item_ids):
    """Return rank (1-indexed) of positive item (column 0)."""
    # prediction: [B, 1+N], first col = positive
    pos_score = prediction[:, 0:1]
    ranks = (prediction > pos_score).sum(dim=1) + 1  # [B]
    return ranks.float()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    logging.info("Loading baseline...")
    model_base, corpus = build_model(args.baseline_checkpoint, args.dataset, args.teacher_path, device)
    logging.info("Loading HSDIR...")
    model_hsdir, _ = build_model(args.hsdir_checkpoint, args.dataset, args.teacher_path, device)

    # Teacher for semantic diversity / agreement
    tdata = pickle.load(open(args.teacher_path, "rb"))
    fine_assign = torch.tensor(tdata["fine_assignments"], dtype=torch.float32).to(device)
    coarse_assign = torch.tensor(tdata["coarse_assignments"], dtype=torch.float32).to(device)

    # Dev dataset (use model_base's Dataset)
    ds = model_base.Dataset(model_base, corpus, "dev")
    ds.prepare()
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0,
                    pin_memory=False, collate_fn=ds.collate_batch)

    all_samples = []
    batch_count = 0

    with torch.inference_mode():
        for batch in dl:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            history = batch["history_items"]
            lengths = batch["lengths"]
            i_ids = batch["item_id"]
            B, L = history.shape

            # Forward both models
            out_base = model_base(batch, return_intermediate=True)
            out_hsdir = model_hsdir(batch, return_intermediate=True)

            rank_base = compute_rank(out_base["prediction"], i_ids)
            rank_hsdir = compute_rank(out_hsdir["prediction"], i_ids)
            delta_rank = rank_base - rank_hsdir  # >0 = HSDIR better

            # Baseline collapse: pairwise cosine of K attention maps
            attn_base = out_base["attention_maps"]  # [B, K, L]
            collapse = per_user_k_cos(attn_base)  # [B]

            # Routing entropy (HSDIR membership)
            if "route_membership" in out_hsdir:
                R = out_hsdir["route_membership"]  # [B, L, K]
                valid = (history > 0).float().unsqueeze(-1)
                R = R * valid
                route_ent = []
                for b in range(B):
                    vl = int(lengths[b].item())
                    if vl <= 0:
                        route_ent.append(0.0)
                        continue
                    ent_list = [float(normalized_entropy(R[b, j])) for j in range(vl)]
                    route_ent.append(float(np.mean(ent_list)) if ent_list else 0.0)
                route_ent = torch.tensor(route_ent, dtype=torch.float32)
            else:
                route_ent = torch.zeros(B)

            # Semantic diversity: normalized entropy of valid fine distribution mean
            fine = fine_assign[history]  # [B, L, 32]
            sem_div = []
            for b in range(B):
                vl = int(lengths[b].item())
                if vl <= 0:
                    sem_div.append(0.0)
                    continue
                valid_fine = fine[b, :vl]  # [vl, 32]
                q_bar = valid_fine.mean(dim=0)  # [32]
                sem_div.append(float(normalized_entropy(q_bar)))
            sem_div = torch.tensor(sem_div, dtype=torch.float32)

            # Semantic-collaborative agreement
            sem_col_agree = []
            he = out_base["history_vectors"].detach()  # [B, L, D]
            Gf = fine @ fine.transpose(-1, -2)  # [B, L, L]
            for b in range(B):
                vl = int(lengths[b].item())
                if vl <= 1:
                    sem_col_agree.append(0.0)
                    continue
                he_n = F.normalize(he[b, :vl], dim=-1, eps=1e-8)
                C = (he_n @ he_n.t() + 1.0) / 2.0
                gf_v = Gf[b, :vl, :vl]
                mask = ~torch.eye(vl, dtype=torch.bool, device=device)
                c_v = C[mask].cpu().numpy()
                gf_v_np = gf_v[mask].cpu().numpy()
                if c_v.std() > 1e-8 and gf_v_np.std() > 1e-8:
                    corr = np.corrcoef(c_v, gf_v_np)[0, 1]
                    sem_col_agree.append(float(corr) if not np.isnan(corr) else 0.0)
                else:
                    sem_col_agree.append(0.0)
            sem_col_agree = torch.tensor(sem_col_agree, dtype=torch.float32)

            for b in range(B):
                all_samples.append({
                    "delta_rank": float(delta_rank[b].item()),
                    "rank_base": float(rank_base[b].item()),
                    "rank_hsdir": float(rank_hsdir[b].item()),
                    "collapse": float(collapse[b].item()),
                    "routing_entropy": float(route_ent[b].item()),
                    "semantic_diversity": float(sem_div[b].item()),
                    "collapse_x_sem_div": float(collapse[b].item() * sem_div[b].item()),
                    "sem_col_agreement": float(sem_col_agree[b].item()),
                    "history_length": int(lengths[b].item()),
                })

            batch_count += 1
            if args.max_batches > 0 and batch_count >= args.max_batches:
                break

    logging.info(f"Total samples: {len(all_samples)}")

    # Build arrays
    fields = ["delta_rank", "rank_base", "rank_hsdir", "collapse",
              "routing_entropy", "semantic_diversity", "collapse_x_sem_div",
              "sem_col_agreement", "history_length"]
    data = {f: np.array([s[f] for s in all_samples]) for f in fields}

    # ---- Sample stats TSV ----
    tsv_path = os.path.join(args.output_dir, "sample_stats.tsv")
    with open(tsv_path, "w") as f:
        headers = fields + ["hsdir_win"]
        f.write("\t".join(headers) + "\n")
        for s in all_samples:
            vals = [str(s[f]) for f in fields] + [str(1 if s["delta_rank"] > 0 else 0)]
            f.write("\t".join(vals) + "\n")

    # ---- Factor quartile analysis ----
    factors = ["collapse", "routing_entropy", "semantic_diversity",
               "collapse_x_sem_div", "sem_col_agreement", "history_length"]
    summary_rows = []
    stats_out = {}

    for factor in factors:
        vals = data[factor]
        finite = np.isfinite(vals)
        vals_f = vals[finite]
        delta_f = data["delta_rank"][finite]

        qs = np.percentile(vals_f, [25, 50, 75])
        for qi, (lo, hi) in enumerate([(None, qs[0]), (qs[0], qs[1]),
                                        (qs[1], qs[2]), (qs[2], None)]):
            if lo is None:
                mask = vals_f <= hi
                label = f"Q1"
            elif hi is None:
                mask = vals_f > lo
                label = f"Q4"
            elif qi == 1:
                mask = (vals_f > lo) & (vals_f <= hi)
                label = f"Q2"
            else:
                mask = (vals_f > lo) & (vals_f <= hi)
                label = f"Q3"

            n = mask.sum()
            if n == 0:
                continue
            mean_dr = float(delta_f[mask].mean())
            win_rate = float((delta_f[mask] > 0).mean())
            mean_rb = float(data["rank_base"][finite][mask].mean())
            mean_rh = float(data["rank_hsdir"][finite][mask].mean())
            summary_rows.append({
                "factor": factor, "quartile": label, "n": int(n),
                "mean_delta_rank": round(mean_dr, 4),
                "hsdir_win_rate": round(win_rate, 4),
                "mean_rank_base": round(mean_rb, 2),
                "mean_rank_hsdir": round(mean_rh, 2),
            })

    # FS (factor summary) TSV
    fs_path = os.path.join(args.output_dir, "factor_summary.tsv")
    with open(fs_path, "w") as f:
        keys = ["factor", "quartile", "n", "mean_delta_rank",
                "hsdir_win_rate", "mean_rank_base", "mean_rank_hsdir"]
        f.write("\t".join(keys) + "\n")
        for r in summary_rows:
            f.write("\t".join(str(r[k]) for k in keys) + "\n")

    # ---- Spearman correlations ----
    from scipy.stats import spearmanr
    spearman = {}
    for factor in factors:
        valid = np.isfinite(data[factor]) & np.isfinite(data["delta_rank"])
        if valid.sum() < 3:
            spearman[factor] = None
        else:
            rho, p = spearmanr(data[factor][valid], data["delta_rank"][valid])
            spearman[factor] = {"rho": round(float(rho), 4), "p": round(float(p), 6)}

    stats_out["num_samples"] = len(all_samples)
    stats_out["overall_win_rate"] = round(float((data["delta_rank"] > 0).mean()), 4)
    stats_out["overall_mean_delta_rank"] = round(float(data["delta_rank"].mean()), 4)
    stats_out["spearman"] = spearman
    stats_out["factor_quartiles"] = summary_rows

    json_path = os.path.join(args.output_dir, "stats.json")
    with open(json_path, "w") as f:
        json.dump(stats_out, f, indent=2)

    logging.info(f"Saved: {tsv_path}, {fs_path}, {json_path}")
    logging.info(f"Overall win_rate={stats_out['overall_win_rate']}, "
                 f"mean_delta_rank={stats_out['overall_mean_delta_rank']}")
    for factor, sp in spearman.items():
        if sp:
            logging.info(f"  Spearman {factor}: rho={sp['rho']}, p={sp['p']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
