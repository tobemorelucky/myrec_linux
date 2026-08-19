# -*- coding: UTF-8 -*-
"""
Multi-seed comparison for CAISD Phase 2 (ASPCF baseline vs CAISD).

Reads per-seed logs under new_log/llmmirec_{aspcf,caisd}_phase2/<dataset>/seed<seed>/
and computes per-metric mean/std and relative improvement.

Usage:
  python tools/summarize_llmmirec_caisd_phase2.py \
    --dataset beauty --seeds 0 1 42
"""

import argparse, json, logging, os, re, sys


METRICS = ["HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20"]


def parse_test_after(line):
    """Parse 'Test After Training: (HR@5:0.1,NDCG@5:0.2,...)' into dict."""
    m = re.search(r"Test After Training: \((.*?)\)", line)
    if not m:
        return None
    out = {}
    for part in m.group(1).split(","):
        k, v = part.split(":")
        out[k.strip()] = float(v.strip())
    return out


def load_log(path):
    """Read best-dev and test-after-training from a training log."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        lines = f.readlines()
    best_dev = None
    test_after = None
    for line in lines:
        if "Best Iter(dev)" in line and best_dev is None:
            best_dev = line.strip()
        if "Test After Training" in line:
            test_after = line.strip()
    if test_after is None:
        return None
    metrics = parse_test_after(test_after)
    return {"best_dev": best_dev, "test_after": test_after, "metrics": metrics}


def summarize(values):
    import numpy as np
    a = np.array(values, dtype=np.float64)
    if a.size == 0:
        return {"mean": None, "std": None}
    return {"mean": float(a.mean()), "std": float(a.std())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, choices=["beauty", "ml-1m"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 42])
    args = p.parse_args()

    families = {
        "ASPCF": ("new_log/llmmirec_aspcf_phase2", "LLMMIRecASPCF"),
        "CAISD": ("new_log/llmmirec_caisd_phase2", "LLMMIRecCAISD"),
    }

    result = {"dataset": args.dataset, "seeds": list(args.seeds), "per_seed": {}, "summary": {}}
    missing = {fam: [] for fam in families}

    for fam, (log_root, tag) in families.items():
        seed_results = {}
        for seed in args.seeds:
            log_path = os.path.join(log_root, args.dataset, f"seed{seed}", f"{tag}_seed{seed}.log")
            data = load_log(log_path)
            if data is None:
                missing[fam].append(seed)
                continue
            seed_results[seed] = data
        result["per_seed"][fam] = {str(s): d for s, d in seed_results.items()}

        if seed_results:
            # Per-metric mean/std across available seeds
            result["summary"][fam] = {}
            for metric in METRICS:
                vals = [d["metrics"][metric] for s, d in seed_results.items()
                        if d["metrics"] and metric in d["metrics"]]
                result["summary"][fam][metric] = summarize(vals)

    # Report missing seeds explicitly (never fill in)
    for fam in families:
        if missing[fam]:
            logging.warning(f"MISSING {fam} seeds: {missing[fam]} — not filled")

    # Relative improvement (only when BOTH families have all requested seeds)
    aspcf_seeds = set(result["per_seed"]["ASPCF"].keys())
    caisd_seeds = set(result["per_seed"]["CAISD"].keys())
    common = aspcf_seeds & caisd_seeds
    if len(common) == len(args.seeds):
        result["relative_improvement_percent"] = {}
        for metric in METRICS:
            a_m = result["summary"]["ASPCF"][metric]["mean"]
            c_m = result["summary"]["CAISD"][metric]["mean"]
            if a_m is not None and c_m is not None and a_m != 0:
                rel = (c_m - a_m) / a_m * 100.0
            else:
                rel = None
            result["relative_improvement_percent"][metric] = round(rel, 4) if rel is not None else None
    else:
        result["relative_improvement_percent"] = None
        logging.warning("Cannot compute relative improvement: seed sets differ")

    out_dir = f"new_log/llmmirec_caisd_phase2/{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "multi_seed_comparison.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logging.info(f"Saved: {out_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
