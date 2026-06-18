#!/usr/bin/env python3
"""
Build semantic hard negative table from LLM item embeddings.

For each item, finds top-K semantically similar items (nearest neighbors
in L2-normalized LLM embedding space) to serve as hard negatives.

Usage:
    python scripts/build_semantic_hardneg.py \
        --input_llm_emb_path ./data/beauty/handled/llm_table_pca1536.pkl \
        --output_path ./data/beauty/handled/semantic_hardneg_top100.pkl \
        --topk 100 --chunk_size 1024
"""

import argparse, pickle, numpy as np, sys, math


def _ensure_2d(x):
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_llm_emb_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--chunk_size", type=int, default=1024)
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required.")
        sys.exit(1)

    data = pickle.load(open(args.input_llm_emb_path, "rb"))
    emb = _ensure_2d(data)
    print(f"Loaded embedding: {emb.shape}")

    # Skip padding row 0 if all zeros
    if emb.shape[0] > 1 and np.allclose(emb[0], 0.0):
        valid_emb = emb[1:]
        has_padding = True
        N = valid_emb.shape[0]
    else:
        valid_emb = emb
        has_padding = False
        N = valid_emb.shape[0]

    D = valid_emb.shape[1]
    topk = min(args.topk, N - 1)
    print(f"Valid items: {N}, dim: {D}, topk: {topk}")

    # L2 normalize
    norms = np.linalg.norm(valid_emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    emb_norm = valid_emb / norms
    emb_t = torch.tensor(emb_norm, dtype=torch.float32)

    # Initialize output table: (total_items, topk)
    total_items = emb.shape[0]
    if has_padding:
        hardneg_table = np.zeros((total_items, topk), dtype=np.int64)
        start_idx = 1
    else:
        hardneg_table = np.zeros((total_items, topk), dtype=np.int64)
        start_idx = 0

    # Chunked nearest neighbor search
    chunk_size = args.chunk_size
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        chunk = emb_t[i:end]                              # (C, D)
        sim = torch.matmul(chunk, emb_t.t())              # (C, N)
        # Exclude self
        for j in range(chunk.size(0)):
            global_idx = i + j
            sim[j, global_idx] = -float("inf")
        _, topk_idx = torch.topk(sim, k=topk, dim=1)     # (C, topk)
        topk_idx = topk_idx.cpu().numpy()
        # Map back: valid_emb index (0..N-1) → item_id (1..N if has_padding)
        if has_padding:
            topk_idx += 1  # shift by 1 for padding row
        hardneg_table[start_idx + i : start_idx + end, :] = topk_idx

        if (i // chunk_size) % 10 == 0:
            print(f"  chunk {i // chunk_size}: [{i}:{end}] done")

    # Save
    pickle.dump(hardneg_table, open(args.output_path, "wb"))
    print(f"Saved: {args.output_path}, shape={hardneg_table.shape}")

    # Show examples
    sample_items = [1, 2, 5] if has_padding else [0, 1, 4]
    for sid in sample_items:
        if sid < len(hardneg_table):
            print(f"  item {sid} hard negs: {hardneg_table[sid][:5].tolist()}...")


if __name__ == "__main__":
    main()
