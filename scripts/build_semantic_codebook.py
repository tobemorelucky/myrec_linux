#!/usr/bin/env python3
"""
Build semantic codebook from LLM item embeddings via KMeans clustering.

Usage:
    python scripts/build_semantic_codebook.py \
        --input ./data/beauty/handled/llm_table_pca1536.pkl \
        --output ./data/beauty/handled/semantic_codebook_C128.pkl \
        --num_clusters 128 --seed 42
"""

import argparse
import pickle
import numpy as np
import sys


def _ensure_2d(x):
    """Ensure input is 2D numpy array. Supports dict and ndarray/tensor."""
    if isinstance(x, dict):
        # dict: item_id -> embedding
        ids = sorted(x.keys())
        emb_list = [np.asarray(x[k]) for k in ids]
        arr = np.stack(emb_list, axis=0)
        print(f"  Loaded dict with {len(ids)} entries, dim={arr.shape[1]}")
        return arr
    elif isinstance(x, np.ndarray):
        return _ensure_2d_np(x)
    else:
        # try converting from torch tensor
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return _ensure_2d_np(x.cpu().numpy())
        except ImportError:
            pass
        raise TypeError(f"Unsupported pkl type: {type(x)}")


def _ensure_2d_np(arr):
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    return arr


def main():
    parser = argparse.ArgumentParser(description="Build semantic codebook from LLM embeddings")
    parser.add_argument("--input", type=str, required=True, help="Path to input pkl file")
    parser.add_argument("--output", type=str, required=True, help="Path to output pkl file")
    parser.add_argument("--num_clusters", type=int, default=128, help="Number of KMeans clusters")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Check sklearn
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("ERROR: scikit-learn is required. Install with: pip install scikit-learn")
        sys.exit(1)

    # Load
    data = pickle.load(open(args.input, "rb"))
    emb = _ensure_2d(data)
    print(f"  Embedding shape: {emb.shape}")

    # Drop padding row 0 if present (row 0 is all zeros in standard format)
    # Check: if row 0 is all zeros, it's padding
    if emb.shape[0] > 1 and np.allclose(emb[0], 0.0):
        valid_emb = emb[1:]  # skip padding
        has_padding = True
    else:
        valid_emb = emb
        has_padding = False

    N = valid_emb.shape[0]
    D = valid_emb.shape[1]
    C = min(args.num_clusters, N)
    print(f"  Valid items: {N}, dim: {D}, clusters: {C}")

    # L2 normalize
    norms = np.linalg.norm(valid_emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    valid_emb_norm = valid_emb / norms

    # KMeans
    kmeans = KMeans(n_clusters=C, random_state=args.seed, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(valid_emb_norm)

    # Map back: item_id -> cluster_id
    # item_ids are 1..N (or 0..N-1 depending on format)
    if has_padding:
        # row 0 was padding, items map to rows 1..N
        # item2cluster[i] = cluster for item_id i
        item2cluster = np.full(emb.shape[0], -1, dtype=np.int32)
        item2cluster[1:] = labels.astype(np.int32)
    else:
        item2cluster = labels.astype(np.int32)

    # Cluster stats
    sizes = np.bincount(labels, minlength=C)
    print(f"  Cluster sizes — min={sizes.min()}, max={sizes.max()}, mean={sizes.mean():.1f}")

    # Save
    output = {
        "num_clusters": C,
        "item2cluster": item2cluster,
        "cluster_centers": kmeans.cluster_centers_,
        "seed": args.seed,
        "source": args.input,
    }
    pickle.dump(output, open(args.output, "wb"))
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
