# -*- coding: UTF-8 -*-
"""
Analyze ASPCF-specific representations from a trained LLMMIRec checkpoint.

Reads a saved LLMMIRec model (aspcf mode), runs inference on the test set,
and computes diagnostics including alpha distributions and
semantic-complement disentanglement metrics.

Usage:
  python tools/analyze_llmmirec_aspcf.py \
    --checkpoint <path.pt> \
    --dataset beauty \
    [--max_batches 50] \
    [--output_dir ./diagnostics_aspcf]
"""

import argparse, json, logging, math, os, pickle, sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch
import torch.nn.functional as F

from models.sequential.LLMMIRec import LLMMIRec


def parse_args():
    p = argparse.ArgumentParser(description="LLMMIRec ASPCF diagnostics")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, default="beauty")
    p.add_argument("--max_batches", type=int, default=50)
    p.add_argument("--output_dir", type=str, default="./diagnostics_aspcf")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def normalized_entropy(probs, dim=-1, eps=1e-8):
    K = probs.shape[dim]
    if K <= 1:
        return torch.zeros(probs.shape[:dim] + probs.shape[dim+1:], device=probs.device)
    H = -(probs * torch.log(probs + eps)).sum(dim=dim)
    return H / math.log(K)


def effective_rank(s, eps=1e-8):
    s = s[s > eps]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    entropy = -(p * torch.log(p + eps)).sum()
    return float(torch.exp(entropy))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load checkpoint ----
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    corpus = pickle.load(open(f"./data/{args.dataset}/SeqReader.pkl", "rb"))
    state = torch.load(args.checkpoint, map_location="cpu")

    # Infer config
    has_adapter = "item_encoder.adapter.0.weight" in state
    has_log_gamma = "item_encoder.log_gamma" in state
    has_gamma_buf = "item_encoder.gamma" in state
    has_semantic = "item_encoder.semantic_branch.0.weight" in state

    if has_semantic:
        mode = "aspcf"
    elif has_adapter and (has_log_gamma or has_gamma_buf):
        mode = "residual"
    elif has_adapter:
        mode = "llm_replace"
    else:
        mode = "id"

    logging.info(f"Inferred mode: {mode}")

    # Build args
    class DummyArgs:
        pass

    ma = DummyArgs()
    ma.device = torch.device(args.device)
    ma.model_path = args.checkpoint
    ma.buffer = 1
    ma.history_max = 20
    ma.num_neg = 1
    ma.test_all = 0

    emb_size = state["position_emb.weight"].shape[1]
    K = state["extractor.query"].shape[0]
    has_adapter_bias = "item_encoder.adapter.0.bias" in state
    if has_adapter_bias:
        d_llm_in = state["item_encoder.adapter.0.weight"].shape[1]
        adapter_hidden = state["item_encoder.adapter.0.weight"].shape[0]
    else:
        adapter_hidden = 256

    adapter_use_ln = any("adapter.3" in k for k in state.keys())

    ma.emb_size = emb_size
    ma.attn_size = state["extractor.Wq.weight"].shape[0]
    ma.K = K
    ma.item_encoder = mode
    ma.llm_emb_path = f"./data/{args.dataset}/handled/llm_table_pca1536.pkl" if mode != "id" else ""
    ma.adapter_hidden = adapter_hidden
    ma.adapter_activation = "gelu"
    ma.adapter_use_ln = int(adapter_use_ln)
    ma.gamma_init = 0.1
    ma.gamma_trainable = int(has_log_gamma)
    ma.dropout = 0.1

    # ASPCF params from state
    ma.semantic_rank = state.get("item_encoder.semantic_branch.0.weight", torch.zeros(1)).shape[1] if has_semantic else 512
    ma.semantic_dim = state["item_encoder.semantic_branch.2.weight"].shape[0] if has_semantic else 32
    ma.semantic_hidden = state["item_encoder.semantic_branch.0.weight"].shape[0] if has_semantic else 128
    ma.complement_dim = state["item_encoder.complement_mlp.2.weight"].shape[0] if has_semantic else 32
    ma.tail_hidden = state["item_encoder.complement_tail.0.weight"].shape[0] if has_semantic else 64
    ma.complement_hidden = state["item_encoder.complement_mlp.0.weight"].shape[0] if has_semantic else 64
    ma.gate_hidden = state["item_encoder.gate.0.weight"].shape[0] if has_semantic else 64
    ma.lambda_relation = 0.0
    ma.relation_sample_size = 128
    ma.relation_teacher_temp = 0.1
    ma.relation_student_temp = 0.1

    model = LLMMIRec(ma, corpus).to(args.device)
    model.load_state_dict(state, strict=False)
    model.eval()
    logging.info(f"Loaded: #params={model.count_variables()}")

    # Build test dataset
    dataset = LLMMIRec.Dataset(model, corpus, "test")
    dataset.prepare()
    from torch.utils.data import DataLoader
    dl = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0,
                    pin_memory=False, collate_fn=dataset.collate_batch)

    # Accumulators
    all_iv, all_attn, all_w, all_lengths = [], [], [], []
    all_alpha_sem, all_alpha_comp = [], []
    all_h_semantic, all_h_complement = [], []
    all_proto_mass, all_proto_ids = [], []

    batch_count = 0
    with torch.inference_mode():
        for batch in dl:
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model(batch, return_intermediate=True)

            all_iv.append(out["interest_vectors"].cpu())
            all_attn.append(out["attention_maps"].cpu())
            all_w.append(out["interest_weights"].cpu())
            all_lengths.append(batch["lengths"].cpu())

            if mode == "aspcf" and "history_alpha_sem" in out:
                all_alpha_sem.append(out["history_alpha_sem"].cpu())
                all_alpha_comp.append(out["history_alpha_comp"].cpu())
                all_h_semantic.append(out["history_semantic"].cpu())
                all_h_complement.append(out["history_complement"].cpu())

            if "prototype_mass" in out:
                all_proto_mass.append(out["prototype_mass"].cpu())
                all_proto_ids.append(out["selected_prototype_ids"].cpu())

            batch_count += 1
            if args.max_batches > 0 and batch_count >= args.max_batches:
                break

    iv = torch.cat(all_iv, dim=0)        # [N, K, D]
    # Pad attention maps
    max_L = max(a.shape[-1] for a in all_attn)
    attn_padded = []
    for a in all_attn:
        if a.shape[-1] < max_L:
            a = F.pad(a, (0, max_L - a.shape[-1]), value=0.0)
        attn_padded.append(a)
    attn = torch.cat(attn_padded, dim=0)
    w = torch.cat(all_w, dim=0)
    lengths = torch.cat(all_lengths, dim=0)

    N, K, D = iv.shape
    logging.info(f"Samples: {N}, K={K}, D={D}")

    # --- Basic interest stats (same as Phase 0) ---
    iv_n = F.normalize(iv, dim=-1, eps=1e-8)
    sim_mat = iv_n @ iv_n.transpose(-1, -2)
    eye = torch.eye(K).unsqueeze(0)
    off_diag = sim_mat * (1 - eye)
    triu = torch.triu(torch.ones(K, K), diagonal=1).unsqueeze(0)
    mean_pairwise = float((off_diag * triu).sum(dim=(-1, -2)).mean() / triu.sum())

    max_inter = float(off_diag.max(dim=-1).values.max(dim=-1).values.mean())

    eff_ranks = []
    for i in range(min(N, 1000)):
        _, s, _ = torch.svd(iv[i])
        eff_ranks.append(effective_rank(s))
    mean_eff_rank = float(np.mean(eff_ranks))

    ent_attn = []
    for i in range(N):
        vl = int(lengths[i].item())
        if vl <= 0:
            continue
        a = attn[i, :, :vl]
        for k in range(K):
            ent_attn.append(float(normalized_entropy(a[k])))
    mean_attn_ent = float(np.mean(ent_attn)) if ent_attn else 0.0

    w_ent = normalized_entropy(w, dim=-1)
    mean_w_ent = float(w_ent.mean())

    q_attn_cos = []
    for i in range(N):
        vl = int(lengths[i].item())
        if vl <= 1:
            continue
        a = attn[i, :, :vl]
        an = F.normalize(a, dim=-1, eps=1e-8)
        cm = an @ an.t()
        off = (cm * (1 - eye)).sum() / (K * (K - 1))
        q_attn_cos.append(float(off))
    mean_q_attn_cos = float(np.mean(q_attn_cos)) if q_attn_cos else 0.0

    stats = {
        "checkpoint": args.checkpoint,
        "mode": mode,
        "num_samples": N, "K": K, "D": D,
        "mean_pairwise_cos": round(mean_pairwise, 6),
        "mean_max_inter_sim": round(max_inter, 6),
        "mean_effective_rank": round(mean_eff_rank, 3),
        "mean_attn_entropy": round(mean_attn_ent, 6),
        "mean_weight_entropy": round(mean_w_ent, 6),
        "mean_query_attn_cos": round(mean_q_attn_cos, 6),
    }

    # --- ASPCF-specific ---
    if mode == "aspcf" and len(all_alpha_sem) > 0:
        # Pad to uniform max_L (batches have different padded lengths)
        max_L2 = max(a.shape[-1] for a in all_alpha_sem)

        def pad2d(tensor_list, max_len):
            out = []
            for t in tensor_list:
                if t.shape[-1] < max_len:
                    t = F.pad(t, (0, max_len - t.shape[-1]), value=0.0)
                out.append(t)
            return torch.cat(out, dim=0)

        def pad3d(tensor_list, max_len):
            out = []
            for t in tensor_list:
                if t.shape[1] < max_len:
                    t = F.pad(t, (0, 0, 0, max_len - t.shape[1]), value=0.0)
                out.append(t)
            return torch.cat(out, dim=0)

        alpha_sem = pad2d(all_alpha_sem, max_L2)
        alpha_comp = pad2d(all_alpha_comp, max_L2)
        h_sem = pad3d(all_h_semantic, max_L2)
        h_comp = pad3d(all_h_complement, max_L2)

        # Flatten to [total_items] (only non-padding)
        a_sem_flat = alpha_sem.reshape(-1)
        a_comp_flat = alpha_comp.reshape(-1)
        non_pad = a_sem_flat > 0
        a_sem_val = a_sem_flat[non_pad]
        a_comp_val = a_comp_flat[non_pad]

        if a_sem_val.numel() > 0:
            def pct(x, p):
                return float(np.percentile(x.numpy(), p))

            for name, vals in [("alpha_sem", a_sem_val), ("alpha_comp", a_comp_val)]:
                stats[f"{name}_mean"] = round(float(vals.mean()), 6)
                stats[f"{name}_std"] = round(float(vals.std()), 6)
                stats[f"{name}_min"] = round(float(vals.min()), 6)
                stats[f"{name}_max"] = round(float(vals.max()), 6)
                stats[f"{name}_p10"] = round(pct(vals, 10), 6)
                stats[f"{name}_p50"] = round(pct(vals, 50), 6)
                stats[f"{name}_p90"] = round(pct(vals, 90), 6)

        # Semantic vs complement cosine similarity (non-padding items)
        s_flat = h_sem.reshape(-1, h_sem.shape[-1])[non_pad]  # [M, 32]
        c_flat = h_comp.reshape(-1, h_comp.shape[-1])[non_pad]
        if s_flat.shape[0] > 0:
            s_n = F.normalize(s_flat, dim=-1, eps=1e-8)
            c_n = F.normalize(c_flat, dim=-1, eps=1e-8)
            sc_cos = (s_n * c_n).sum(dim=-1)
            stats["semantic_complement_cos_mean"] = round(float(sc_cos.mean()), 6)
            stats["semantic_complement_cos_std"] = round(float(sc_cos.std()), 6)

    # --- Prototype-specific ---
    if len(all_proto_mass) > 0:
        proto_mass = torch.cat(all_proto_mass, dim=0)  # [N, proto_num]
        proto_ids = torch.cat(all_proto_ids, dim=0)     # [N, K]

        proto_num = proto_mass.shape[1]

        # Per-prototype usage frequency
        proto_usage = proto_mass.mean(dim=0)  # [proto_num]
        for p in range(proto_num):
            stats[f"proto{p}_usage"] = round(float(proto_usage[p]), 6)

        # Selection frequency (how often each prototype is in Top-K)
        select_flat = proto_ids.reshape(-1)
        for p in range(proto_num):
            freq = float((select_flat == p).float().mean())
            stats[f"proto{p}_select_freq"] = round(freq, 6)

        # Duplicate rate: fraction of users with repeated prototypes
        B = proto_ids.shape[0]
        dup_count = 0
        for i in range(B):
            unique = len(set(proto_ids[i].tolist()))
            if unique < K:
                dup_count += 1
        stats["proto_dup_rate"] = round(dup_count / max(B, 1), 6)

        # Prototype mass entropy per user
        pm = proto_mass.clamp(min=1e-8)
        H_proto = -(pm * torch.log(pm)).sum(dim=-1) / math.log(proto_num)  # [N]
        stats["proto_mass_entropy_mean"] = round(float(H_proto.mean()), 6)
        stats["proto_mass_entropy_std"] = round(float(H_proto.std()), 6)

    # Save
    json_path = os.path.join(args.output_dir, "stats.json")
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    logging.info(f"Saved: {json_path}")

    # TSV
    tsv_path = os.path.join(args.output_dir, "stats.tsv")
    with open(tsv_path, "w") as f:
        f.write("key\tvalue\n")
        for k, v in stats.items():
            f.write(f"{k}\t{v}\n")
    logging.info(f"Saved: {tsv_path}")

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
