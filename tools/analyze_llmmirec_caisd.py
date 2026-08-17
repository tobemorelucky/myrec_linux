# -*- coding: UTF-8 -*-
"""
CAISD routing-redundancy diagnostics (dev by default).

Computes BOTH attention teacher and responsibility teacher from the same
batch of attention maps (independent of the checkpoint's teacher mode), and
reports routing redundancy, responsibility entropy, teacher differentiation
and teacher-student KL, each with mean/std/p25/median/p75.

Usage:
  python tools/analyze_llmmirec_caisd.py \
    --checkpoint <ckpt.pt> \
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
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--teacher_path", type=str, required=True)
    p.add_argument("--dataset", type=str, default="beauty")
    p.add_argument("--phase", type=str, default="dev",
                   choices=["train", "dev", "test"])
    p.add_argument("--max_batches", type=int, default=50)
    p.add_argument("--output_dir", type=str, default="./diagnostics_caisd")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def normalized_entropy(probs, dim=-1, eps=1e-8):
    n = probs.shape[dim]
    if n <= 1:
        return torch.zeros_like(probs.sum(dim=dim))
    H = -(probs * torch.log(probs + eps)).sum(dim=dim)
    return H / math.log(n)


def per_user_k_cos_batch(x, valid_lengths=None):
    """Per-user mean off-diagonal K-way cosine. x: [B, K, L]."""
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


def summarize(arr):
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": 0.0, "std": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0}
    return {
        "mean": round(float(a.mean()), 6),
        "std": round(float(a.std()), 6),
        "p25": round(float(np.percentile(a, 25)), 6),
        "median": round(float(np.percentile(a, 50)), 6),
        "p75": round(float(np.percentile(a, 75)), 6),
    }


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
    ma.semantic_teacher_mode = "attention"   # irrelevant for this analysis
    ma.semantic_responsibility_alpha = 0.5
    ma.lambda_interest_semantic = 0.01
    ma.dropout = 0.1

    corpus = pickle.load(open(f"./data/{args.dataset}/SeqReader.pkl", "rb"))
    model = LLMMIRecCAISD(ma, corpus).to(args.device)
    model.load_state_dict(state, strict=False)
    model.eval()

    ds = model.Dataset(model, corpus, args.phase)
    ds.prepare()
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0,
                    pin_memory=False, collate_fn=ds.collate_batch)

    # Per-user accumulators
    user_attn_cos, user_resp_ent, user_len = [], [], []
    user_attn_teacher_cos, user_resp_teacher_cos = [], []
    user_attn_teacher_ent, user_resp_teacher_ent = [], []
    user_attn_teacher_kl, user_resp_teacher_kl = [], []

    K = K_emb
    tdata = pickle.load(open(args.teacher_path, "rb"))
    t_assign = torch.tensor(tdata["soft_assignments"], dtype=torch.float32).to(args.device)

    batch_count = 0
    with torch.inference_mode():
        for batch in dl:
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            history = batch["history_items"]
            lengths = batch["lengths"]
            B, L = history.shape

            out = model(batch, return_intermediate=True)
            A = out["attention_maps"].detach()      # [B, K, L]
            V = out["interest_vectors"]             # [B, K, D]
            valid = (history > 0).float()           # [B, L]

            # ---- Routing redundancy (attention) ----
            attn_cos = per_user_k_cos_batch(A)      # [B]
            user_attn_cos.extend(attn_cos.tolist())

            # ---- Responsibility R over K dims ----
            A_masked = A * valid.unsqueeze(1)
            R = A_masked / A_masked.sum(dim=1, keepdim=True).clamp_min(1e-8)  # [B, K, L]
            R = R.detach()

            # Responsibility entropy: per valid item, entropy over K dims
            resp_ent_user = []
            for b in range(B):
                vl = int(lengths[b].item())
                if vl <= 0:
                    resp_ent_user.append(0.0)
                    continue
                R_item = R[b, :, :vl].t()           # [vl, K]  (K dim last for entropy)
                ents = normalized_entropy(R_item.clamp(min=1e-8), dim=-1)  # [vl]
                resp_ent_user.append(float(ents.mean()))
            user_resp_ent.extend(resp_ent_user)
            user_len.extend([int(lengths[b].item()) for b in range(B)])

            # ---- Both teachers from the SAME attention ----
            Q = t_assign[history]                   # [B, L, 32]

            # Attention teacher
            T_attn = torch.bmm(A, Q)                # [B, K, 32]
            T_attn = T_attn / T_attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            T_attn = T_attn.detach()

            # Responsibility teacher
            W_resp = A_masked * R
            W_resp = W_resp / W_resp.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            T_resp = torch.bmm(W_resp, Q)
            T_resp = T_resp / T_resp.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            T_resp = T_resp.detach()

            # Teacher diversity (per-user K-way cosine of T)
            user_attn_teacher_cos.extend(per_user_k_cos_batch(T_attn).tolist())
            user_resp_teacher_cos.extend(per_user_k_cos_batch(T_resp).tolist())

            # Teacher entropy
            ent_attn_t = normalized_entropy(T_attn.clamp(min=1e-8), dim=-1)
            ent_resp_t = normalized_entropy(T_resp.clamp(min=1e-8), dim=-1)
            user_attn_teacher_ent.extend(ent_attn_t.mean(dim=-1).tolist())
            user_resp_teacher_ent.extend(ent_resp_t.mean(dim=-1).tolist())

            # Teacher-student KL (student predictor from model)
            P_logits = model.semantic_predictor(V)
            log_P = F.log_softmax(P_logits, dim=-1)
            kl_attn = F.kl_div(log_P, T_attn, reduction="none").sum(dim=-1)  # [B, K]
            kl_resp = F.kl_div(log_P, T_resp, reduction="none").sum(dim=-1)
            user_attn_teacher_kl.extend(kl_attn.mean(dim=-1).tolist())
            user_resp_teacher_kl.extend(kl_resp.mean(dim=-1).tolist())

            batch_count += 1
            if args.max_batches > 0 and batch_count >= args.max_batches:
                break

    stats = {
        "phase": args.phase,
        "num_users": len(user_attn_cos),
        "attention_inter_interest_cos": summarize(user_attn_cos),
        "attention_inter_interest_cos_user": summarize(user_attn_cos),
        "responsibility_entropy": summarize(user_resp_ent),
        "responsibility_entropy_user": summarize(user_resp_ent),
        "attention_teacher_inter_interest_cos": summarize(user_attn_teacher_cos),
        "responsibility_teacher_inter_interest_cos": summarize(user_resp_teacher_cos),
        "teacher_cos_reduction": summarize(
            np.array(user_attn_teacher_cos) - np.array(user_resp_teacher_cos)),
        "attention_teacher_entropy": summarize(user_attn_teacher_ent),
        "responsibility_teacher_entropy": summarize(user_resp_teacher_ent),
        "attention_teacher_student_kl": summarize(user_attn_teacher_kl),
        "responsibility_teacher_student_kl": summarize(user_resp_teacher_kl),
        "history_length_mean": summarize(user_len),
    }

    json_path = os.path.join(args.output_dir, "stats.json")
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    logging.info(f"Saved: {json_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
