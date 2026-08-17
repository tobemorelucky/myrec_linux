# -*- coding: UTF-8 -*-
"""
CAISD diagnostics: teacher diversity, teacher-student KL, responsibility entropy.

Usage:
  python tools/analyze_llmmirec_caisd.py \
    --checkpoint <ckpt.pt> \
    --teacher_path ./data/beauty/handled/llmmi_proto32_sr512.pkl \
    --dataset beauty \
    --semantic_teacher_mode responsibility_power \
    --semantic_responsibility_alpha 0.5 \
    --max_batches 50 \
    --output_dir <dir>
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
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--teacher_path", type=str, required=True)
    p.add_argument("--dataset", type=str, default="beauty")
    p.add_argument("--semantic_teacher_mode", type=str, default="attention",
                   choices=["attention", "responsibility", "responsibility_power"])
    p.add_argument("--semantic_responsibility_alpha", type=float, default=0.5)
    p.add_argument("--max_batches", type=int, default=50)
    p.add_argument("--output_dir", type=str, default="./diagnostics_caisd")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def normalized_entropy(probs, dim=-1, eps=1e-8):
    K = probs.shape[dim]
    if K <= 1:
        return torch.zeros_like(probs.sum(dim=dim))
    H = -(probs * torch.log(probs + eps)).sum(dim=dim)
    return H / math.log(K)


def per_user_k_cos(x):
    """x: [B, K, *] -> mean off-diagonal K-way cosine per user."""
    B, kk = x.shape[:2]
    if kk <= 1:
        return 0.0
    vals = []
    for b in range(min(B, 128)):
        v = x[b].reshape(kk, -1)
        vn = F.normalize(v, dim=-1, eps=1e-8)
        cm = vn @ vn.t()
        off = (cm * (1 - torch.eye(kk, device=cm.device))).sum() / max(kk * (kk - 1), 1)
        vals.append(float(off))
    return float(np.mean(vals)) if vals else 0.0


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    state = torch.load(args.checkpoint, map_location="cpu")
    emb_size = state["position_emb.weight"].shape[1]
    K_emb = state["extractor.query"].shape[0]
    attn_size = state["extractor.Wq.weight"].shape[0]

    class D: pass
    ma = D()
    ma.device = torch.device(args.device)
    ma.model_path = args.checkpoint
    ma.buffer = 1; ma.history_max = 20; ma.num_neg = 1; ma.test_all = 0
    ma.emb_size = emb_size; ma.attn_size = attn_size; ma.K = K_emb
    ma.item_encoder = "aspcf"
    ma.llm_emb_path = f"./data/{args.dataset}/handled/llm_table_pca1536.pkl"
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
    ma.semantic_teacher_path = args.teacher_path
    ma.semantic_distill_mode = "uniform"
    ma.semantic_teacher_mode = args.semantic_teacher_mode
    ma.semantic_responsibility_alpha = args.semantic_responsibility_alpha
    ma.lambda_interest_semantic = 0.01
    ma.dropout = 0.1

    corpus = pickle.load(open(f"./data/{args.dataset}/SeqReader.pkl", "rb"))
    model = LLMMIRecCAISD(ma, corpus).to(args.device)
    model.load_state_dict(state, strict=False)
    model.eval()

    ds = model.Dataset(model, corpus, "test")
    ds.prepare()
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0,
                    pin_memory=False, collate_fn=ds.collate_batch)

    all_T, all_W, all_KL = [], [], []
    all_attn = []
    batch_count = 0

    with torch.inference_mode():
        for batch in dl:
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model(batch, return_intermediate=True)
            all_attn.append(out["attention_maps"].cpu())
            if "interest_semantic_teacher" in out:
                all_T.append(out["interest_semantic_teacher"].cpu())
                all_KL.append(out["interest_semantic_kl"].cpu())
            if "interest_semantic_teacher_weight" in out:
                all_W.append(out["interest_semantic_teacher_weight"].cpu())
            batch_count += 1
            if args.max_batches > 0 and batch_count >= args.max_batches:
                break

    stats = {}

    # Attention diversity
    if all_attn:
        max_L = max(a.shape[-1] for a in all_attn)
        attn = torch.cat([F.pad(a, (0, max_L - a.shape[-1])) if a.shape[-1] < max_L else a
                          for a in all_attn], dim=0)
        stats["attention_inter_interest_cos"] = round(per_user_k_cos(attn), 6)

    # Teacher diversity
    if all_T:
        T = torch.cat(all_T, dim=0)
        stats["teacher_inter_interest_cos"] = round(per_user_k_cos(T), 6)
        T_ent = normalized_entropy(T.clamp(min=1e-8), dim=-1)
        stats["teacher_entropy"] = round(float(T_ent.mean()), 6)
        KL = torch.cat(all_KL, dim=0)
        stats["teacher_student_kl"] = round(float(KL.mean()), 6)

    # Responsibility entropy
    if all_W:
        W = torch.cat(all_W, dim=0)  # [N, K, L]
        W_ent = normalized_entropy(W.clamp(min=1e-8), dim=-1)
        stats["responsibility_entropy"] = round(float(W_ent.mean()), 6)

    json_path = os.path.join(args.output_dir, "stats.json")
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    logging.info(f"Saved: {json_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
