# -*- coding: UTF-8 -*-
"""
Build offline semantic prototypes from frozen PCA LLM embeddings.

Uses MiniBatchKMeans on the first semantic_rank PCA dimensions,
then computes soft assignments via cosine similarity + softmax.

Usage:
  python tools/build_llmmi_semantic_prototypes.py \
    --dataset beauty \
    --semantic_rank 512 \
    --prototype_num 32 \
    --temperature 0.1

Output:
  data/<dataset>/handled/llmmi_proto32_sr512.pkl
    {
      'centers':          ndarray [prototype_num, semantic_rank],
      'soft_assignments': ndarray [n_items, prototype_num],  # row 0 = zeros
      'prototype_num':    int,
      'semantic_rank':    int,
      'temperature':      float,
    }
"""

import argparse, os, pickle, sys

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="beauty")
    p.add_argument("--semantic_rank", type=int, default=512)
    p.add_argument("--prototype_num", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    llm_path = f"./data/{args.dataset}/handled/llm_table_pca1536.pkl"
    out_path = (f"./data/{args.dataset}/handled/"
                f"llmmi_proto{args.prototype_num}_sr{args.semantic_rank}.pkl")

    if not os.path.exists(llm_path):
        raise FileNotFoundError(f"LLM table not found: {llm_path}")

    # Load
    table = np.asarray(pickle.load(open(llm_path, "rb")), dtype=np.float32)
    print(f"Loaded: {llm_path}  shape={table.shape}")
    n_items, d_full = table.shape
    assert args.semantic_rank <= d_full

    # Use only high-variance PCA dims, exclude row 0
    z_high = table[1:, :args.semantic_rank].copy()  # [N, semantic_rank]
    N = z_high.shape[0]
    print(f"Semantic subspace: {N} items x {args.semantic_rank} dims")

    # KMeans
    print(f"MiniBatchKMeans: n_clusters={args.prototype_num} ...")
    kmeans = MiniBatchKMeans(
        n_clusters=args.prototype_num,
        random_state=args.seed,
        batch_size=1024,
        n_init=3,
        max_iter=100,
    )
    kmeans.fit(z_high)
    centers = kmeans.cluster_centers_.astype(np.float32)  # [K, semantic_rank]
    print(f"Centers shape: {centers.shape}")

    # Soft assignment via cosine similarity
    z_norm = z_high / (np.linalg.norm(z_high, axis=1, keepdims=True) + 1e-8)
    c_norm = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
    cos_sim = z_norm @ c_norm.T  # [N, K]
    assignments = softmax(cos_sim / args.temperature, axis=1).astype(np.float32)

    # Pad row 0
    assignments_full = np.zeros((n_items, args.prototype_num), dtype=np.float32)
    assignments_full[1:] = assignments

    data = {
        "centers": centers,
        "soft_assignments": assignments_full,
        "prototype_num": args.prototype_num,
        "semantic_rank": args.semantic_rank,
        "temperature": args.temperature,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pickle.dump(data, open(out_path, "wb"))
    print(f"Saved: {out_path}")
    print(f"  centers: {centers.shape}")
    print(f"  soft_assignments: {assignments_full.shape}")


if __name__ == "__main__":
    main()
