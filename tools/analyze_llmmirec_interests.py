# -*- coding: UTF-8 -*-
"""
Analyze multi-interest representations from a trained LLMMIRec checkpoint.

Reads a saved LLMMIRec model, runs inference on the test set with
return_intermediate=True, and computes 10 diagnostic statistics on
interest vectors, attention maps, and interest weights.

Usage:
  python tools/analyze_llmmirec_interests.py \
    --checkpoint <path.pt> \
    --dataset beauty \
    [--max_batches 20] \
    [--output_dir ./diagnostics]

Output:
  {output_dir}/stats.json   — all statistics as a dict
  {output_dir}/per_query.tsv — per-query statistics
"""

import argparse, json, logging, math, os, pickle, sys

# Ensure project root is on sys.path (needed when run from tools/ dir)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch
import torch.nn.functional as F

from models.sequential.LLMMIRec import LLMMIRec


def parse_args():
    p = argparse.ArgumentParser(description="LLMMIRec interest diagnostics")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to LLMMIRec .pt checkpoint")
    p.add_argument("--dataset", type=str, default="beauty")
    p.add_argument("--max_batches", type=int, default=0,
                   help="Max eval batches (0 = all)")
    p.add_argument("--output_dir", type=str, default="./diagnostics",
                   help="Output directory for stats files")
    p.add_argument("--device", type=str, default="cuda",
                   help="Device: cuda or cpu")
    return p.parse_args()


def safe_norm(x, dim=-1, eps=1e-8):
    return x.norm(dim=dim).clamp(min=eps)


def cos_sim_matrix(x):
    """x: [N, D] -> [N, N] cosine similarity matrix"""
    xn = F.normalize(x, dim=-1, eps=1e-8)
    return xn @ xn.t()


def effective_rank(s, eps=1e-8):
    """Effective rank (entropy-based) from singular values s."""
    s = s[s > eps]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    entropy = -(p * torch.log(p + eps)).sum()
    return float(torch.exp(entropy))


def normalized_entropy(probs, dim=-1, eps=1e-8):
    """Normalized entropy: H(p) / log(K), where K = probs.shape[dim]."""
    K = probs.shape[dim]
    if K <= 1:
        return torch.zeros(probs.shape[:dim] + probs.shape[dim+1:], device=probs.device)
    log_p = torch.log(probs + eps)
    H = -(probs * log_p).sum(dim=dim)
    return H / math.log(K)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load checkpoint ----
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    logging.info(f"Loading checkpoint: {args.checkpoint}")

    # Reconstruct a minimal model by loading the saved state_dict
    # We need corpus to get item_num
    corpus_path = f"./data/{args.dataset}/SeqReader.pkl"
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    corpus = pickle.load(open(corpus_path, "rb"))

    # Build minimal args for LLMMIRec
    class DummyArgs:
        pass

    model_args = DummyArgs()
    model_args.device = torch.device(args.device)
    model_args.model_path = args.checkpoint
    model_args.buffer = 1
    model_args.history_max = 20
    model_args.num_neg = 1         # required by GeneralModel
    model_args.test_all = 0         # required by GeneralModel

    state = torch.load(args.checkpoint, map_location="cpu")

    # Infer config from state_dict keys
    emb_size = state["item_encoder.id_embedding.weight"].shape[1]
    K = state["extractor.query"].shape[0]
    adapter_use_ln = any("adapter.3" in k for k in state.keys())

    # Detect mode: llm_table is persistent=False so NOT in state_dict.
    # Use adapter weights and gamma presence instead.
    has_adapter = "item_encoder.adapter.0.weight" in state
    has_log_gamma = "item_encoder.log_gamma" in state
    has_gamma_buf = "item_encoder.gamma" in state
    if has_adapter and (has_log_gamma or has_gamma_buf):
        mode = "residual"
    elif has_adapter:
        mode = "llm_replace"
    else:
        mode = "id"

    # Detect adapter params
    if has_adapter:
        llm_dim = state["item_encoder.adapter.0.weight"].shape[1]
        adapter_hidden = state["item_encoder.adapter.0.weight"].shape[0]
    else:
        llm_dim = 0
        adapter_hidden = 256

    gamma_init = 0.1
    gamma_trainable = int(has_log_gamma)

    model_args.emb_size = emb_size
    model_args.attn_size = state["extractor.Wq.weight"].shape[0]
    model_args.K = K
    model_args.item_encoder = mode
    model_args.llm_emb_path = (f"./data/{args.dataset}/handled/llm_table_pca1536.pkl"
                                if mode != "id" else "")
    model_args.adapter_hidden = adapter_hidden
    model_args.adapter_activation = "gelu"
    model_args.adapter_use_ln = int(adapter_use_ln)
    model_args.gamma_init = gamma_init
    model_args.gamma_trainable = gamma_trainable
    model_args.dropout = 0.1

    logging.info(f"Inferred config: mode={mode}, emb_size={emb_size}, "
                 f"K={K}, adapter_hidden={adapter_hidden}, "
                 f"gamma_trainable={gamma_trainable}")

    model = LLMMIRec(model_args, corpus).to(args.device)
    model.load_state_dict(state)
    model.eval()
    logging.info(f"Model loaded: #params={model.count_variables()}")

    # ---- Build test dataset ----
    dataset = LLMMIRec.Dataset(model, corpus, "test")
    dataset.prepare()  # populate buffer_dict for test set buffered access
    from torch.utils.data import DataLoader
    dl = DataLoader(
        dataset, batch_size=256, shuffle=False,
        num_workers=0, pin_memory=False,
        collate_fn=dataset.collate_batch,
    )

    # ---- Accumulators ----
    all_iv = []          # interest_vectors: [B,K,D]
    all_attn = []         # attention_maps: [B,K,L]
    all_w = []            # interest_weights: [B,K]
    all_lengths = []      # lengths: [B]

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

            batch_count += 1
            if args.max_batches > 0 and batch_count >= args.max_batches:
                break

    iv = torch.cat(all_iv, dim=0)      # [N, K, D]
    # Attention maps have variable L per batch — pad to max L
    from torch.nn.utils.rnn import pad_sequence as pad_seq
    max_L = max(a.shape[-1] for a in all_attn)
    attn_padded = []
    for a in all_attn:
        if a.shape[-1] < max_L:
            a = F.pad(a, (0, max_L - a.shape[-1]), value=0.0)
        attn_padded.append(a)
    attn = torch.cat(attn_padded, dim=0)  # [N, K, max_L]
    w = torch.cat(all_w, dim=0)          # [N, K]
    lengths = torch.cat(all_lengths, dim=0)  # [N]

    N, K, D = iv.shape
    L = attn.shape[-1]
    logging.info(f"Total samples: {N}, K={K}, D={D}, L={L}")

    # --- NaN/Inf check on all tensors ---
    for name, t in [("interest_vectors", iv), ("attention_maps", attn),
                     ("interest_weights", w)]:
        if torch.isnan(t).any() or torch.isinf(t).any():
            raise RuntimeError(f"NaN/Inf detected in aggregated {name}!")

    # ============================================================
    #  Stat 1: Average pairwise cosine similarity of interest vectors
    # ============================================================
    iv_norm = F.normalize(iv, dim=-1, eps=1e-8)   # [N, K, D]
    sim_mat = iv_norm @ iv_norm.transpose(-1, -2)   # [N, K, K]
    # Mask diagonal
    eye = torch.eye(K, device=sim_mat.device).unsqueeze(0)
    off_diag = sim_mat * (1 - eye)
    triu_mask = torch.triu(torch.ones(K, K, device=sim_mat.device), diagonal=1).unsqueeze(0)
    pairwise = (off_diag * triu_mask).sum(dim=(-1, -2)) / triu_mask.sum()
    mean_pairwise_cos = float(pairwise.mean())

    # ============================================================
    #  Stat 2: Max inter-interest cosine similarity per sample
    # ============================================================
    max_inter_sim = off_diag.max(dim=-1).values.max(dim=-1).values  # [N]
    mean_max_inter_sim = float(max_inter_sim.mean())

    # ============================================================
    #  Stat 3: Effective rank of interest matrix
    # ============================================================
    effective_ranks = []
    for i in range(N):
        _, s, _ = torch.svd(iv[i])  # [K, D] -> s: [K] (or min(K,D))
        effective_ranks.append(effective_rank(s))
    mean_eff_rank = float(np.mean(effective_ranks))

    # ============================================================
    #  Stat 4: Normalized entropy of attention maps
    #  (ignoring padding positions)
    # ============================================================
    entropies_attn = []
    for i in range(N):
        valid_len = int(lengths[i].item())
        if valid_len <= 0:
            continue
        a = attn[i, :, :valid_len]  # [K, valid_len]
        for k in range(K):
            entropies_attn.append(float(normalized_entropy(a[k])))
    mean_attn_entropy = float(np.mean(entropies_attn)) if entropies_attn else 0.0

    # ============================================================
    #  Stat 5: Average entropy of interest weights
    # ============================================================
    w_entropy = normalized_entropy(w, dim=-1)  # [N]
    mean_w_entropy = float(w_entropy.mean())

    # ============================================================
    #  Stat 6: Max interest weight per sample
    # ============================================================
    max_w = w.max(dim=-1).values  # [N]
    mean_max_w = float(max_w.mean())

    # ============================================================
    #  Stat 7: Average interest weight per query
    # ============================================================
    per_query_mean_w = w.mean(dim=0)  # [K]
    per_query_mean_w_list = [float(x) for x in per_query_mean_w]

    # ============================================================
    #  Stat 8: Fraction of samples where each query gets max weight
    # ============================================================
    argmax_w = w.argmax(dim=-1)  # [N]
    per_query_max_frac = []
    for k in range(K):
        frac = float((argmax_w == k).float().mean())
        per_query_max_frac.append(frac)

    # ============================================================
    #  Stat 9: Average historical attention mass per query
    #  (mass assigned to valid history positions)
    # ============================================================
    query_attn_mass = []
    for k in range(K):
        masses = []
        for i in range(N):
            valid_len = int(lengths[i].item())
            if valid_len <= 0:
                continue
            mass = float(attn[i, k, :valid_len].sum())
            masses.append(mass)
        query_attn_mass.append(float(np.mean(masses)) if masses else 0.0)

    # ============================================================
    #  Stat 10: Average cosine similarity between attention maps
    #  of different queries
    # ============================================================
    query_attn_cos = []
    for i in range(N):
        valid_len = int(lengths[i].item())
        if valid_len <= 1:
            continue
        a = attn[i, :, :valid_len]  # [K, valid_len]
        an = F.normalize(a, dim=-1, eps=1e-8)
        cm = an @ an.t()  # [K, K]
        eye = torch.eye(K, device=cm.device)
        off = (cm * (1 - eye)).sum() / (K * (K - 1))
        query_attn_cos.append(float(off))
    mean_query_attn_cos = float(np.mean(query_attn_cos)) if query_attn_cos else 0.0

    # ============================================================
    #  Compile results
    # ============================================================
    stats = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "num_samples": N,
        "K": K,
        "emb_size": D,
        "item_encoder": mode,
        "mean_pairwise_cos_sim": round(mean_pairwise_cos, 6),
        "mean_max_inter_sim": round(mean_max_inter_sim, 6),
        "mean_effective_rank": round(mean_eff_rank, 3),
        "mean_attn_entropy": round(mean_attn_entropy, 6),
        "mean_weight_entropy": round(mean_w_entropy, 6),
        "mean_max_weight": round(mean_max_w, 6),
        "per_query_mean_weight": [round(x, 6) for x in per_query_mean_w_list],
        "per_query_max_frac": [round(x, 6) for x in per_query_max_frac],
        "per_query_attn_mass": [round(x, 6) for x in query_attn_mass],
        "mean_query_attn_cos": round(mean_query_attn_cos, 6),
    }

    # Save JSON
    json_path = os.path.join(args.output_dir, "stats.json")
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    logging.info(f"Saved stats to {json_path}")

    # Save per-query TSV
    tsv_path = os.path.join(args.output_dir, "per_query.tsv")
    with open(tsv_path, "w") as f:
        f.write("query_idx\tmean_weight\tmax_weight_frac\tmean_attn_mass\n")
        for k in range(K):
            f.write(f"{k}\t{per_query_mean_w_list[k]:.6f}\t"
                    f"{per_query_max_frac[k]:.6f}\t"
                    f"{query_attn_mass[k]:.6f}\n")
    logging.info(f"Saved per-query stats to {tsv_path}")

    # Print summary
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
