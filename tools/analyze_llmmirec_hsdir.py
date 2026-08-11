# -*- coding: UTF-8 -*-
"""
Dedicated HSDIR diagnostics: interest structure + routing + teacher-student metrics.

Usage:
  # baseline (lambda=0, teacher from file):
  python tools/analyze_llmmirec_hsdir.py \
    --checkpoint <ckpt.pt> \
    --teacher_path ./data/beauty/handled/llmmi_hier_proto32_8_sr512.pkl \
    --dataset beauty --max_batches 50 --output_dir <dir>

  # HSDIR (lambda>0, teacher from file):
  python tools/analyze_llmmirec_hsdir.py \
    --checkpoint <ckpt.pt> \
    --teacher_path ./data/beauty/handled/llmmi_hier_proto32_8_sr512.pkl \
    --dataset beauty --max_batches 50 --output_dir <dir>
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
    p = argparse.ArgumentParser(description="HSDIR diagnostics")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, default="beauty")
    p.add_argument("--teacher_path", type=str, required=True,
                   help="Path to llmmi_hier_proto32_8_sr512.pkl (required for both baseline and HSDIR)")
    p.add_argument("--aggregation_mode", type=str, default="base",
                   choices=["base", "support_confidence"])
    p.add_argument("--support_beta", type=float, default=1.0)
    p.add_argument("--hsr_loss_mode", type=str, default="absolute",
                   choices=["absolute", "relative", "pair_selective"])
    p.add_argument("--max_batches", type=int, default=50)
    p.add_argument("--output_dir", type=str, default="./diagnostics_hsdir")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def normalized_entropy(probs, dim=-1, eps=1e-8):
    K = probs.shape[dim]
    if K <= 1:
        return torch.zeros_like(probs.sum(dim=dim))
    H = -(probs * torch.log(probs + eps)).sum(dim=dim)
    return H / math.log(K)


def per_user_k_cos(x):
    """x: [B, K, *] -> average off-diagonal K-way cosine, per-user mean."""
    B, K = x.shape[:2]
    if K <= 1:
        return 0.0
    vals = []
    for b in range(min(B, 128)):
        v = x[b].reshape(K, -1)
        vn = F.normalize(v, dim=-1, eps=1e-8)
        cm = vn @ vn.t()
        off = (cm * (1 - torch.eye(K))).sum() / max(K * (K - 1), 1)
        vals.append(float(off))
    return float(np.mean(vals)) if vals else 0.0


def effective_rank_from_svd(x, eps=1e-8):
    s = torch.linalg.svdvals(x.float())
    s = s[s > eps]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    H = -(p * torch.log(p + eps)).sum()
    return float(torch.exp(H))


# ---- Padding helpers ----

def pad_history(h, max_L):
    """h: [B, L] -> [B, max_L], pad with 0."""
    if h.shape[1] < max_L:
        return F.pad(h, (0, max_L - h.shape[1]), value=0)
    return h


def pad_attn(a, max_L):
    """a: [B, K, L] -> [B, K, max_L], pad with 0."""
    if a.shape[-1] < max_L:
        return F.pad(a, (0, max_L - a.shape[-1]), value=0.0)
    return a


def pad_membership(R, max_L):
    """R: [B, L, K] -> [B, max_L, K], pad with 0."""
    if R.shape[1] < max_L:
        return F.pad(R, (0, 0, 0, max_L - R.shape[1]), value=0.0)
    return R


def pad_relation(G, max_L):
    """G: [B, L, L] -> [B, max_L, max_L], pad with 0."""
    if G.shape[1] < max_L:
        return F.pad(G, (0, max_L - G.shape[1], 0, max_L - G.shape[1]), value=0.0)
    return G


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    corpus = pickle.load(open(f"./data/{args.dataset}/SeqReader.pkl", "rb"))
    state = torch.load(args.checkpoint, map_location="cpu")

    # Infer config from state_dict
    emb_size = state["position_emb.weight"].shape[1]
    K_emb = state["extractor.query"].shape[0]
    has_adapter = "item_encoder.adapter.0.weight" in state
    adapter_hidden = state["item_encoder.adapter.0.weight"].shape[0] if has_adapter else 256
    adapter_use_ln = any("adapter.3" in k for k in state.keys())
    has_log_gamma = "item_encoder.log_gamma" in state
    has_semantic = "item_encoder.semantic_branch.0.weight" in state
    semantic_rank = state.get("item_encoder.semantic_branch.0.weight", torch.zeros(1)).shape[1] if has_semantic else 512
    attn_size = state["extractor.Wq.weight"].shape[0]

    class D:
        pass
    ma = D()
    ma.device = torch.device(args.device)
    ma.model_path = args.checkpoint
    ma.buffer = 1; ma.history_max = 20; ma.num_neg = 1; ma.test_all = 0
    ma.emb_size = emb_size; ma.attn_size = attn_size; ma.K = K_emb
    ma.item_encoder = "aspcf"
    ma.llm_emb_path = f"./data/{args.dataset}/handled/llm_table_pca1536.pkl"
    ma.adapter_hidden = adapter_hidden
    ma.adapter_activation = "gelu"; ma.adapter_use_ln = int(adapter_use_ln)
    ma.gamma_init = 0.1; ma.gamma_trainable = int(has_log_gamma)
    ma.semantic_rank = semantic_rank
    ma.semantic_dim = 32; ma.semantic_hidden = 128
    ma.complement_dim = 32; ma.tail_hidden = 64
    ma.complement_hidden = 64; ma.gate_hidden = 64
    ma.aspcf_gate_mode = "basic"
    ma.lambda_relation = 0.01
    ma.relation_sample_size = 128
    ma.relation_teacher_temp = 0.1; ma.relation_student_temp = 0.1
    ma.lambda_hsr = 0.01  # enables route_scores in forward
    ma.hsr_teacher_mode = "hierarchical"
    ma.hsr_student_temp = 1.0
    ma.teacher_path = args.teacher_path
    ma.aggregation_mode = args.aggregation_mode
    ma.support_beta = args.support_beta
    ma.hsr_loss_mode = args.hsr_loss_mode
    ma.dropout = 0.1

    model = LLMMIRecHSDIR(ma, corpus).to(args.device)
    model.load_state_dict(state, strict=False)
    model.eval()
    logging.info(f"Loaded: #params={model.count_variables()}")

    # Load teacher from file (always, for consistent diagnostics)
    tdata = pickle.load(open(args.teacher_path, "rb"))
    teacher = {
        "fine": torch.tensor(tdata["fine_assignments"], dtype=torch.float32).to(args.device),
        "coarse": torch.tensor(tdata["coarse_assignments"], dtype=torch.float32).to(args.device),
    }
    logging.info(f"Teacher loaded: fine={teacher['fine'].shape} coarse={teacher['coarse'].shape}")

    # Dataset
    ds = model.Dataset(model, corpus, "test")
    ds.prepare()
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0,
                    pin_memory=False, collate_fn=ds.collate_batch)

    K = K_emb

    # ---- Collect all batches WITHOUT concat ----
    batches_iv = []
    batches_attn = []
    batches_w = []
    batches_base_w = []
    batches_lengths = []
    batches_history = []
    batches_R = []
    batches_G_route = []
    batches_fine_rel = []
    batches_coarse_rel = []
    batches_support = []
    batches_confidence = []

    with torch.inference_mode():
        for batch in dl:
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model(batch, return_intermediate=True)

            batches_iv.append(out["interest_vectors"].cpu())
            batches_w.append(out["interest_weights"].cpu())
            if "base_interest_weights" in out:
                batches_base_w.append(out["base_interest_weights"].cpu())
            if "support_distribution" in out:
                batches_support.append(out["support_distribution"].cpu())
                batches_confidence.append(out["routing_confidence"].cpu())
            batches_lengths.append(batch["lengths"].cpu())
            batches_history.append(batch["history_items"].cpu())
            batches_attn.append(out["attention_maps"].cpu())

            if "route_membership" in out:
                batches_R.append(out["route_membership"].cpu())
                batches_G_route.append(out["route_comembership"].cpu())
            else:
                batches_R.append(None)
                batches_G_route.append(None)

            # Teacher relations: always compute from file
            h = batch["history_items"]
            fine = teacher["fine"][h]    # [B, L, 32]
            coarse = teacher["coarse"][h]  # [B, L, 8]
            batches_fine_rel.append((fine @ fine.transpose(-1, -2)).cpu())
            batches_coarse_rel.append((coarse @ coarse.transpose(-1, -2)).cpu())

            if args.max_batches > 0 and len(batches_iv) >= args.max_batches:
                break

    # ---- Compute global max_L ----
    max_L = 0
    for h in batches_history:
        max_L = max(max_L, h.shape[1])
    for a in batches_attn:
        max_L = max(max_L, a.shape[-1])
    for R in batches_R:
        if R is not None:
            max_L = max(max_L, R.shape[1])
    logging.info(f"Global max_L: {max_L}")

    # ---- Pad and concat ----
    iv = torch.cat(batches_iv, dim=0)
    w = torch.cat(batches_w, dim=0)
    lengths = torch.cat(batches_lengths, dim=0)
    hist = torch.cat([pad_history(h, max_L) for h in batches_history], dim=0)
    attn = torch.cat([pad_attn(a, max_L) for a in batches_attn], dim=0)

    N, _, D = iv.shape
    stats = {"num_samples": N, "K": K, "D": D, "max_L": max_L}

    has_routing = any(R is not None for R in batches_R)

    # ---- A. Interest structure ----
    stats["mean_pairwise_interest_cos"] = round(per_user_k_cos(iv), 6)

    eff_ranks = [effective_rank_from_svd(iv[i]) for i in range(min(N, 500))]
    stats["mean_effective_rank"] = round(float(np.mean(eff_ranks)), 3)

    ent_list = []
    for i in range(min(N, 500)):
        vl = int(lengths[i].item())
        if vl <= 0: continue
        a = attn[i, :, :vl]
        for k in range(K):
            ent_list.append(float(normalized_entropy(a[k])))
    stats["mean_attention_entropy"] = round(float(np.mean(ent_list)) if ent_list else 0.0, 6)

    stats["mean_query_attention_cos"] = round(per_user_k_cos(attn), 6)

    w_ent = normalized_entropy(w, dim=-1)
    stats["interest_weight_entropy"] = round(float(w_ent.mean()), 6)
    stats["mean_max_interest_weight"] = round(float(w.max(dim=-1).values.mean()), 6)

    # Calibration metrics
    if batches_base_w:
        base_w = torch.cat(batches_base_w, dim=0)
        base_w_ent = normalized_entropy(base_w, dim=-1)
        stats["base_weight_entropy"] = round(float(base_w_ent.mean()), 6)
        stats["final_weight_entropy"] = stats["interest_weight_entropy"]

    if batches_support:
        sup_cat = torch.cat(batches_support, dim=0)       # [N, K]
        conf_cat = torch.cat(batches_confidence, dim=0)    # [N]
        stats["mean_routing_confidence"] = round(float(conf_cat.mean()), 6)
        # Correlation between support and final weight
        sup_np = sup_cat[:500].numpy()
        w_np = w[:500].numpy()
        corr_list = []
        for i in range(min(sup_np.shape[0], w_np.shape[0])):
            if sup_np[i].std() > 1e-8 and w_np[i].std() > 1e-8:
                c = np.corrcoef(sup_np[i], w_np[i])[0, 1]
                if not np.isnan(c):
                    corr_list.append(c)
        stats["support_final_weight_corr"] = round(float(np.mean(corr_list)) if corr_list else 0.0, 6)

    # ---- B. Student routing ----
    if has_routing:
        R_padded = [pad_membership(R, max_L) if R is not None else
                     torch.zeros(1, max_L, K) for R in batches_R]
        R_full = torch.cat(R_padded, dim=0)  # [N, max_L, K]

        r_ent_list, r_max_list = [], []
        for i in range(min(N, 500)):
            vl = int(lengths[i].item())
            if vl <= 0: continue
            for j in range(vl):
                r_ent_list.append(float(normalized_entropy(R_full[i, j])))
                r_max_list.append(float(R_full[i, j].max()))
        stats["route_membership_entropy"] = round(float(np.mean(r_ent_list)) if r_ent_list else 0.0, 6)
        stats["route_membership_max"] = round(float(np.mean(r_max_list)) if r_max_list else 0.0, 6)

        supports = []
        for i in range(min(N, 500)):
            vl = int(lengths[i].item())
            if vl <= 0: continue
            sup = R_full[i, :vl, :].sum(dim=0)
            sup = sup / sup.sum().clamp(min=1e-8)
            supports.append(sup)
        if supports:
            sup_stack = torch.stack(supports, dim=0)
            sup_ent = normalized_entropy(sup_stack, dim=-1)
            stats["support_entropy"] = round(float(sup_ent.mean()), 6)
            eff_k = [float(torch.exp(-(s * torch.log(s + 1e-8)).sum())) for s in sup_stack]
            stats["effective_active_K"] = round(float(np.mean(eff_k)), 3)

    # ---- C. Teacher-Student (per-batch, no concat needed) ----
    if has_routing:
        same_fine_list, diff_coarse_list = [], []
        corr_rf_list, corr_rc_list = [], []

        for bi in range(len(batches_G_route)):
            G = batches_G_route[bi]
            Gf = batches_fine_rel[bi]
            Gc = batches_coarse_rel[bi]
            h = batches_history[bi]  # [B, L_bi]

            if G is None:
                continue

            B_bi, L_bi = h.shape
            valid = (h > 0).float()  # [B, L_bi]
            vp = valid.unsqueeze(-1) * valid.unsqueeze(-2)
            diag = torch.eye(L_bi, dtype=torch.bool).unsqueeze(0)
            vp = vp * (~diag).float()

            for b in range(min(B_bi, 32)):
                mask = vp[b].bool()
                if mask.sum() < 2:
                    continue
                gv = G[b][mask]
                gfv = Gf[b][mask]
                gcv = Gc[b][mask]
                w_sum = gfv.sum().clamp(min=1e-8)
                same_fine_list.append(float((gfv * gv).sum() / w_sum))
                w_neg = (1 - gcv).sum().clamp(min=1e-8)
                diff_coarse_list.append(float(((1 - gcv) * gv).sum() / w_neg))

                gv_np = gv.numpy(); gfv_np = gfv.numpy(); gcv_np = gcv.numpy()
                if gv_np.std() > 1e-8 and gfv_np.std() > 1e-8:
                    c = np.corrcoef(gv_np, gfv_np)[0, 1]
                    if not np.isnan(c):
                        corr_rf_list.append(c)
                if gv_np.std() > 1e-8 and gcv_np.std() > 1e-8:
                    c = np.corrcoef(gv_np, gcv_np)[0, 1]
                    if not np.isnan(c):
                        corr_rc_list.append(c)

        if same_fine_list:
            stats["same_fine_route_score"] = round(float(np.mean(same_fine_list)), 6)
            stats["diff_coarse_route_score"] = round(float(np.mean(diff_coarse_list)), 6)
            stats["corr_route_fine"] = round(float(np.mean(corr_rf_list)) if corr_rf_list else 0.0, 6)
            stats["corr_route_coarse"] = round(float(np.mean(corr_rc_list)) if corr_rc_list else 0.0, 6)

    # ---- D. Pair metrics (pair_selective diagnostics) ----
    if has_routing:
        pstat_anchor_ratio, pstat_conf, pstat_rpos, pstat_rneg, pstat_gap = [], [], [], [], []
        for bi in range(len(batches_G_route)):
            G = batches_G_route[bi]; Gf = batches_fine_rel[bi]; Gc = batches_coarse_rel[bi]
            h = batches_history[bi]
            if G is None: continue
            B_bi, L_bi = h.shape
            valid = (h > 0).float()
            vp = valid.unsqueeze(-1) * valid.unsqueeze(-2)
            diag = torch.eye(L_bi, dtype=torch.bool).unsqueeze(0)
            vp = vp * (~diag).float()
            valid_anchor = vp.sum(dim=-1) > 0  # [B_bi, L_bi]
            pos_scores = Gf.clone(); pos_scores[vp == 0] = float("-inf")
            p_idx = pos_scores.argmax(dim=-1)  # [B_bi, L_bi]
            neg_scores = Gc.clone(); neg_scores[vp == 0] = float("inf")
            bi_range = torch.arange(B_bi).unsqueeze(-1).expand(-1, L_bi)
            l_range = torch.arange(L_bi).unsqueeze(0)
            neg_scores[bi_range, l_range, p_idx] = float("inf")
            n_idx = neg_scores.argmin(dim=-1)
            pos_ok = vp[bi_range, l_range, p_idx]; neg_ok = vp[bi_range, l_range, n_idx]
            ok = valid_anchor & pos_ok & neg_ok & (p_idx != n_idx)
            if ok.sum() < 1: continue
            c_i = Gf[bi_range, l_range, p_idx] * (1.0 - Gc[bi_range, l_range, n_idx])
            rp = G[bi_range, l_range, p_idx]; rn = G[bi_range, l_range, n_idx]
            pstat_anchor_ratio.append(float(ok.float().mean()))
            pstat_conf.append(float(c_i[ok].mean()))
            pstat_rpos.append(float(rp[ok].mean()))
            pstat_rneg.append(float(rn[ok].mean()))
            pstat_gap.append(float((rp - rn)[ok].mean()))
        if pstat_gap:
            stats["pair_valid_anchor_ratio"] = round(float(np.mean(pstat_anchor_ratio)), 4)
            stats["pair_teacher_confidence"] = round(float(np.mean(pstat_conf)), 4)
            stats["pair_positive_route"] = round(float(np.mean(pstat_rpos)), 4)
            stats["pair_negative_route"] = round(float(np.mean(pstat_rneg)), 4)
            stats["pair_route_gap"] = round(float(np.mean(pstat_gap)), 4)

    # Save
    json_path = os.path.join(args.output_dir, "stats.json")
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    tsv_path = os.path.join(args.output_dir, "stats.tsv")
    with open(tsv_path, "w") as f:
        f.write("key\tvalue\n")
        for k, v in stats.items():
            f.write(f"{k}\t{v}\n")
    logging.info(f"Saved: {json_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
