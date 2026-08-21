# -*- coding: UTF-8 -*-
"""
TASID ablation summary: lambda / temperature sensitivity tables.

Reads the sensitivity sweep summary TSV (written by
run_llmmirec_caisd_tasid_sensitivity.sh), aggregates across seeds,
and outputs a comparison table of each config vs the tasid-off baseline.

Usage:
  python tools/summarize_llmmirec_caisd_tasid_ablation.py \
    --dataset beauty --sweep_type lambda --seeds 42
  python tools/summarize_llmmirec_caisd_tasid_ablation.py \
    --dataset ml-1m --sweep_type temp --seeds 0 1 42
"""

import argparse, json, logging, os, re

import numpy as np

METRICS = ["HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20"]
MAIN = "NDCG@5"


def parse_test_after(line):
    m = re.search(r"Test After Training: \((.*?)\)", line)
    if not m:
        return None
    out = {}
    for part in m.group(1).split(","):
        k, v = part.split(":")
        out[k.strip()] = float(v.strip())
    return out


def load_summary(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != len(header):
                continue
            rows.append(dict(zip(header, parts)))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, choices=["beauty", "ml-1m"])
    p.add_argument("--sweep_type", type=str, required=True, choices=["lambda", "temp"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = p.parse_args()

    summary_path = f"new_log/llmmirec_caisd_tasid_sweep/{args.dataset}/summary_{args.sweep_type}.tsv"
    rows = load_summary(summary_path)
    if not rows:
        logging.error(f"No rows in {summary_path}")
        return

    # Group by value (string), filter by requested seeds
    groups = {}
    for r in rows:
        if int(r["seed"]) not in args.seeds:
            continue
        v = r["value"]
        groups.setdefault(v, []).append(r)

    # Baseline = value "none"
    if "none" not in groups:
        logging.error("Baseline (value=none) missing — cannot compute relative changes.")
        return

    baseline = {}
    for m in METRICS:
        vals = [parse_test_after(r["test_after_training"])[m]
                for r in groups["none"]
                if parse_test_after(r["test_after_training"])]
        baseline[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    table = {"dataset": args.dataset, "sweep_type": args.sweep_type,
             "seeds": args.seeds, "baseline": baseline, "configs": {}}
    for v in sorted(groups.keys(), key=lambda x: (x == "none", float(x) if x != "none" else -1)):
        if v == "none":
            continue
        cfg = {}
        per_seed = {}
        for r in groups[v]:
            mets = parse_test_after(r["test_after_training"])
            if not mets:
                continue
            per_seed[str(r["seed"])] = {m: mets[m] for m in METRICS}
        if not per_seed:
            continue
        for m in METRICS:
            vals = [d[m] for d in per_seed.values()]
            mean = float(np.mean(vals)); std = float(np.std(vals))
            rel = (mean - baseline[m]["mean"]) / baseline[m]["mean"] * 100 if baseline[m]["mean"] else None
            cfg[m] = {"mean": mean, "std": std,
                      "rel_improvement_pct": round(rel, 4) if rel is not None else None}
        cfg["per_seed"] = per_seed
        cfg["n_seeds"] = len(per_seed)
        table["configs"][v] = cfg

    out_path = f"new_log/llmmirec_caisd_tasid_sweep/{args.dataset}/ablation_{args.sweep_type}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(table, f, indent=2)
    logging.info(f"Saved: {out_path}")

    # Console table
    print(f"\n=== TASID {args.sweep_type} sensitivity — {args.dataset} (seeds {args.seeds}) ===")
    print(f"{'value':<10}{'NDCG@5':>12}{'NDCG@10':>12}{'HR@5':>12}{'HR@10':>12}")
    b = baseline[MAIN]["mean"]
    print(f"{'baseline':<10}{b:>12.4f}{baseline['NDCG@10']['mean']:>12.4f}"
          f"{baseline['HR@5']['mean']:>12.4f}{baseline['HR@10']['mean']:>12.4f}")
    for v, cfg in table["configs"].items():
        rel = cfg[MAIN]["rel_improvement_pct"]
        rel_s = f"{rel:+.2f}%" if rel is not None else "n/a"
        print(f"{v:<10}{cfg[MAIN]['mean']:>12.4f}{cfg['NDCG@10']['mean']:>12.4f}"
              f"{cfg['HR@5']['mean']:>12.4f}{cfg['HR@10']['mean']:>12.4f}  ({rel_s})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
