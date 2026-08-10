# -*- coding: UTF-8 -*-
"""
Dedicated HSDIR diagnostics: interest structure + routing + teacher-student metrics.

Usage:
  # HSDIR (lambda>0, teacher loaded in model):
  python tools/analyze_llmmirec_hsdir.py \
    --checkpoint <hsdir_ckpt.pt> \
    --dataset beauty --max_batches 50 \
    --output_dir <dir>

  # baseline (lambda=0, teacher from separate file):
  python tools/analyze_llmmirec_hsdir.py \
    --checkpoint <baseline_ckpt.pt> \
    --teacher_path ./data/beauty/handled/llmmi_hier_proto32_8_sr512.pkl \
    --dataset beauty --max_batches 50 \
    --output_dir <dir>
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
    p.add_argument("--teacher_path", type=str, default="")
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


def per_user_k_cos(x, valid=None):
    """x: [B, K, *] → average off-diagonal K-way cosine, per-user mean."""
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
    ma.teacher_path = args.teacher_path or ""
    ma.dropout = 0.1

    model = LLMMIRecHSDIR(ma, corpus).to(args.device)
    model.load_state_dict(state, strict=False)
    model.eval()
    logging.info(f"Loaded: #params={model.count_variables()}")

    # External teacher for baseline
    ext_teacher = None
    if not hasattr(model, "t_fine_assign") and args.teacher_path:
        tdata = pickle.load(open(args.teacher_path, "rb"))
        ext_teacher = {
            "fine": torch.tensor(tdata["fine_assignments"], dtype=torch.float32).to(args.device),
            "coarse": torch.tensor(tdata["coarse_assignments"], dtype=torch.float32).to(args.device),
        }
        logging.info(f"External teacher loaded: fine={ext_teacher['fine'].shape}")

    # Dataset
    ds = model.Dataset(model, corpus, "test")
    ds.prepare()
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0,
                    pin_memory=False, collate_fn=ds.collate_batch)

    # Accumulators
    all_iv, all_attn, all_w, all_lengths, all_history = [], [], [], [], []
    all_R, all_G_route = [], []
    all_fine_rel, all_coarse_rel = [], []

    K = K_emb
    batch_count = 0
    with torch.inference_mode():
        for batch in dl:
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model(batch, return_intermediate=True)

            all_iv.append(out["interest_vectors"].cpu())
            all_w.append(out["interest_weights"].cpu())
            all_lengths.append(batch["lengths"].cpu())
            all_history.append(batch["history_items"].cpu())

            # Pad attention
            max_L = max(a.shape[-1] for a in [out["attention_maps"].cpu()])
            a = out["attention_maps"].cpu()
            if a.shape[-1] < max_L:
                a = F.pad(a, (0, max_L - a.shape[-1]))
            all_attn.append(a)

            if "route_membership" in out:
                R = out["route_membership"].cpu()
                all_R.append(R)
                G = out["route_comembership"].cpu()
                all_G_route.append(G)

            # Teacher relations
            if "teacher_fine_relation" in out:
                all_fine_rel.append(out["teacher_fine_relation"].cpu())
                all_coarse_rel.append(out["teacher_coarse_relation"].cpu())
            elif ext_teacher is not None:
                fine = ext_teacher["fine"][batch["history_items"]]
                coarse = ext_teacher["coarse"][batch["history_items"]]
                all_fine_rel.append((fine @ fine.transpose(-1, -2)).cpu())
                all_coarse_rel.append((coarse @ coarse.transpose(-1, -2)).cpu())

            batch_count += 1
            if args.max_batches > 0 and batch_count >= args.max_batches:
                break

    iv = torch.cat(all_iv, dim=0)
    w = torch.cat(all_w, dim=0)
    lengths = torch.cat(all_lengths, dim=0)
    hist = torch.cat(all_history, dim=0)
    N, _, D = iv.shape
    stats = {"num_samples": N, "K": K, "D": D}

    # Pad attention
    max_AL = max(a.shape[-1] for a in all_attn)
    attn = torch.cat([F.pad(a, (0, max_AL - a.shape[-1])) if a.shape[-1] < max_AL else a
                       for a in all_attn], dim=0)

    # ---- A. Interest structure ----
    stats["mean_pairwise_interest_cos"] = round(per_user_k_cos(iv), 6)

    eff_ranks = [effective_rank_from_svd(iv[i]) for i in range(min(N, 500))]
    stats["mean_effective_rank"] = round(float(np.mean(eff_ranks)), 3)

    # Attention entropy
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

    # ---- B. Student routing ----
    if all_R:
        R = torch.cat([r for r in all_R], dim=0)  # [N, Lmax, K]
        valid = (hist > 0).float()
        # Route membership entropy per valid history item
        r_ent_list = []
        r_max_list = []
        for i in range(min(N, 500)):
            vl = int(lengths[i].item())
            if vl <= 0: continue
            for j in range(vl):
                r_ent_list.append(float(normalized_entropy(R[i, j])))
                r_max_list.append(float(R[i, j].max()))
        stats["route_membership_entropy"] = round(float(np.mean(r_ent_list)) if r_ent_list else 0.0, 6)
        stats["route_membership_max"] = round(float(np.mean(r_max_list)) if r_max_list else 0.0, 6)

        # Support per interest
        supports = []
        for i in range(min(N, 500)):
            vl = int(lengths[i].item())
            if vl <= 0: continue
            sup = R[i, :vl, :].sum(dim=0)  # [K]
            sup = sup / sup.sum().clamp(min=1e-8)
            supports.append(sup)
        if supports:
            sup_stack = torch.stack(supports, dim=0)  # [N, K]
            sup_ent = normalized_entropy(sup_stack, dim=-1)
            stats["support_entropy"] = round(float(sup_ent.mean()), 6)
            eff_k = [float(torch.exp(-(s * torch.log(s + 1e-8)).sum())) for s in sup_stack]
            stats["effective_active_K"] = round(float(np.mean(eff_k)), 3)

    # ---- C. Teacher-Student ----
    if all_G_route and (all_fine_rel or ext_teacher is not None):
        has_pairs = False
        same_fine_list, diff_coarse_list = [], []
        corr_rf_list, corr_rc_list = [], []

        for bi in range(min(len(all_G_route), batch_count)):
            G = all_G_route[bi]  # [B, L, L]
            Gf = all_fine_rel[bi] if bi < len(all_fine_rel) else None
            Gc = all_coarse_rel[bi] if bi < len(all_coarse_rel) else None
            if Gf is None:
                continue
            h = hist[bi * 256:(bi + 1) * 256] if bi < len(all_history) else hist[:G.shape[0]]
            valid = (h > 0).float()
            vp = valid.unsqueeze(-1) * valid.unsqueeze(-2)  # [B, L, L]
            diag = torch.eye(vp.shape[-1], dtype=torch.bool).unsqueeze(0)
            vp = vp * (~diag).float()

            for b in range(min(G.shape[0], 32)):
                mask = vp[b].bool()
                if mask.sum() < 2:
                    continue
                has_pairs = True
                gv = G[b][mask]
                gfv = Gf[b][mask]
                gcv = Gc[b][mask]
                # Same-fine route score (weighted by Gf)
                w_sum = gfv.sum().clamp(min=1e-8)
                same_fine_list.append(float((gfv * gv).sum() / w_sum))
                # Diff-coarse route score (weighted by 1-Gc)
                w_neg = (1 - gcv).sum().clamp(min=1e-8)
                diff_coarse_list.append(float(((1 - gcv) * gv).sum() / w_neg))
                # Pearson correlations
                gv_np = gv.numpy(); gfv_np = gfv.numpy(); gcv_np = gcv.numpy()
                if gv_np.std() > 1e-8 and gfv_np.std() > 1e-8:
                    corr_rf_list.append(np.corrcoef(gv_np, gfv_np)[0, 1])
                if gv_np.std() > 1e-8 and gcv_np.std() > 1e-8:
                    corr_rc_list.append(np.corrcoef(gv_np, gcv_np)[0, 1])

        if has_pairs:
            stats["same_fine_route_score"] = round(float(np.mean(same_fine_list)), 6)
            stats["diff_coarse_route_score"] = round(float(np.mean(diff_coarse_list)), 6)
            stats["corr_route_fine"] = round(float(np.mean(corr_rf_list)) if corr_rf_list else 0.0, 6)
            stats["corr_route_coarse"] = round(float(np.mean(corr_rc_list)) if corr_rc_list else 0.0, 6)

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
