# -*- coding: UTF-8 -*-
"""
Build hierarchical semantic teacher from existing fine prototypes.

Input:  data/<dataset>/handled/llmmi_proto32_sr512.pkl
Output: data/<dataset>/handled/llmmi_hier_proto32_8_sr512.pkl

Layers:
  fine (32) → KMeans → coarse (8)
  coarse_assignment[item,c] = sum(fine_assignment[item,p] for p→c)
"""

import argparse, os, pickle
import numpy as np
from sklearn.cluster import KMeans


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="beauty")
    p.add_argument("--fine_num", type=int, default=32)
    p.add_argument("--coarse_num", type=int, default=8)
    p.add_argument("--semantic_rank", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    proto_path = (f"./data/{args.dataset}/handled/"
                  f"llmmi_proto{args.fine_num}_sr{args.semantic_rank}.pkl")
    out_path = (f"./data/{args.dataset}/handled/"
                f"llmmi_hier_proto{args.fine_num}_{args.coarse_num}_sr{args.semantic_rank}.pkl")

    if not os.path.exists(proto_path):
        raise FileNotFoundError(f"Fine prototype file not found: {proto_path}\n"
                                f"  Run tools/build_llmmi_semantic_prototypes.py first.")

    data = pickle.load(open(proto_path, "rb"))
    fine_centers = np.asarray(data["centers"], dtype=np.float32)       # [32, 512]
    fine_assignments = np.asarray(data["soft_assignments"], dtype=np.float32)  # [N, 32]
    n_items = fine_assignments.shape[0]

    print(f"Loaded fine prototypes: {fine_centers.shape[0]} centers x {fine_centers.shape[1]}d")
    print(f"Fine assignments: {n_items} items x 32")

    # Cluster fine centers into coarse
    kmeans = KMeans(n_clusters=args.coarse_num, random_state=args.seed, n_init=20)
    kmeans.fit(fine_centers)
    fine_to_coarse = kmeans.labels_.astype(np.int32)  # [32]
    coarse_centers = kmeans.cluster_centers_.astype(np.float32)  # [8, 512]

    # Coarse assignments: sum fine assignments per coarse cluster
    coarse_assignments = np.zeros((n_items, args.coarse_num), dtype=np.float32)
    for c in range(args.coarse_num):
        mask = (fine_to_coarse == c)
        coarse_assignments[:, c] = fine_assignments[:, mask].sum(axis=1)

    # Verify: each row should sum to ~1 (except padding row 0)
    row_sums = coarse_assignments[1:].sum(axis=1)
    max_dev = np.abs(row_sums - 1.0).max()
    print(f"Coarse assignment row-sum max deviation: {max_dev:.6f}")
    assert max_dev < 0.01, f"Coarse assignments don't sum to 1: max_dev={max_dev}"

    result = {
        "fine_centers": fine_centers,
        "fine_assignments": fine_assignments,
        "coarse_centers": coarse_centers,
        "coarse_assignments": coarse_assignments,
        "fine_to_coarse": fine_to_coarse,
        "fine_num": args.fine_num,
        "coarse_num": args.coarse_num,
        "semantic_rank": args.semantic_rank,
        "seed": args.seed,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pickle.dump(result, open(out_path, "wb"))
    print(f"\nSaved: {out_path}")
    print(f"  fine_centers:      {fine_centers.shape}")
    print(f"  fine_assignments:  {fine_assignments.shape}")
    print(f"  coarse_centers:    {coarse_centers.shape}")
    print(f"  coarse_assignments:{coarse_assignments.shape}")
    print(f"  fine_to_coarse:    {fine_to_coarse.shape}, labels: {np.unique(fine_to_coarse)}")


if __name__ == "__main__":
    main()
