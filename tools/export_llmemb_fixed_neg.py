"""
tools/export_llmemb_fixed_neg.py

Export fixed negative samples from dev.csv / test.csv for LLMEmb fair evaluation.

Usage:
  python tools/export_llmemb_fixed_neg.py \
    --src_data ./data/beauty \
    --out_dir ../baselines/LLMEmb/data/beauty/handled
"""

import argparse
import csv
import json
import os
import pickle


def parse_neg_items(raw):
    """Parse neg_items string (Python list format like '[1,2,3]') into list of ints.

    Supports both JSON-compatible and Python-native list strings.
    """
    raw = raw.strip()
    if not raw or raw == "[]":
        return []
    # Try JSON first (safer)
    try:
        parsed = json.loads(raw)
        return [int(x) for x in parsed]
    except (json.JSONDecodeError, ValueError):
        pass
    # Fall back to eval for Python-native strings
    parsed = eval(raw)
    return [int(x) for x in parsed]


def read_csv_with_neg(path, sep):
    """Read CSV and return dicts: pos[user_id] = item_id, neg[user_id] = [item_id, ...]."""
    pos = {}
    neg = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter=sep)
        for row in reader:
            uid = int(row["user_id"])
            iid = int(row["item_id"])
            neg_list = parse_neg_items(row["neg_items"])
            pos[uid] = iid
            neg[uid] = neg_list
    return pos, neg


def main():
    parser = argparse.ArgumentParser(
        description="Export fixed negative samples for LLMEmb fair evaluation"
    )
    parser.add_argument("--src_data", type=str, required=True,
                        help="Path to source data dir, e.g. ./data/beauty")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output dir for pkl files, e.g. ../baselines/LLMEmb/data/beauty/handled")
    parser.add_argument("--sep", type=str, default="\t",
                        help="CSV separator (default: tab)")
    args = parser.parse_args()

    src_data = args.src_data
    out_dir = args.out_dir
    sep = args.sep

    dev_path = os.path.join(src_data, "dev.csv")
    test_path = os.path.join(src_data, "test.csv")

    for path, name in [(dev_path, "dev"), (test_path, "test")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name}.csv not found at {path}")

    # ------------------------------------------------------------------
    # 1. Read dev and test
    # ------------------------------------------------------------------
    print(f"Reading dev from:  {dev_path}")
    dev_pos, dev_neg = read_csv_with_neg(dev_path, sep)
    print(f"Reading test from: {test_path}")
    test_pos, test_neg = read_csv_with_neg(test_path, sep)

    print(f"  dev:  {len(dev_pos)} users")
    print(f"  test: {len(test_pos)} users")

    # ------------------------------------------------------------------
    # 2. Checks
    # ------------------------------------------------------------------
    errors = []

    for split_name, pos_dict, neg_dict in [
        ("dev", dev_pos, dev_neg),
        ("test", test_pos, test_neg),
    ]:
        # 2a. Check neg count consistency
        neg_counts = set(len(v) for v in neg_dict.values())
        if len(neg_counts) > 1:
            errors.append(
                f"{split_name}: inconsistent neg counts across users: {neg_counts}"
            )
        neg_count = list(neg_counts)[0] if len(neg_counts) == 1 else "inconsistent"
        print(f"  {split_name}: neg_per_user = {neg_count}")

        # 2b. Check no negative contains positive
        for uid in pos_dict:
            pos_iid = pos_dict[uid]
            neg_list = neg_dict.get(uid, [])
            if pos_iid in neg_list:
                errors.append(
                    f"{split_name} user {uid}: positive item {pos_iid} found in negative list"
                )

        # 2c. Check no item_id <= 0 in negative list
        for uid, neg_list in neg_dict.items():
            for neg_iid in neg_list:
                if neg_iid <= 0:
                    errors.append(
                        f"{split_name} user {uid}: negative item {neg_iid} <= 0"
                    )

        # 2d. Check positive item_id > 0
        for uid, pos_iid in pos_dict.items():
            if pos_iid <= 0:
                errors.append(
                    f"{split_name} user {uid}: positive item {pos_iid} <= 0"
                )

    # 2e. Compute max item id across all
    all_items = set()
    for d in [dev_pos, dev_neg, test_pos, test_neg]:
        for uid, v in d.items():
            if isinstance(v, list):
                all_items.update(v)
            else:
                all_items.add(v)
    max_item_id = max(all_items)

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        raise RuntimeError(f"Fixed neg export check FAILED: {len(errors)} errors.")

    check_passed = len(errors) == 0
    print(f"\nAll checks: {'PASSED' if check_passed else 'FAILED'}")

    # ------------------------------------------------------------------
    # 3. Write pkl files
    # ------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)

    outputs = {
        "valid_neg.pkl": dev_neg,
        "test_neg.pkl": test_neg,
        "valid_pos.pkl": dev_pos,
        "test_pos.pkl": test_pos,
    }

    for fname, data in outputs.items():
        fpath = os.path.join(out_dir, fname)
        with open(fpath, "wb") as f:
            pickle.dump(data, f)
        print(f"Written: {fpath} ({len(data)} entries)")

    # ------------------------------------------------------------------
    # 4. Stats JSON
    # ------------------------------------------------------------------
    dev_neg_count = set(len(v) for v in dev_neg.values())
    test_neg_count = set(len(v) for v in test_neg.values())

    stats = {
        "dev_users": len(dev_pos),
        "test_users": len(test_pos),
        "dev_neg_per_user": list(dev_neg_count)[0] if len(dev_neg_count) == 1 else str(dev_neg_count),
        "test_neg_per_user": list(test_neg_count)[0] if len(test_neg_count) == 1 else str(test_neg_count),
        "max_item_id": max_item_id,
        "check_passed": check_passed,
    }

    stats_path = os.path.join(out_dir, "fixed_neg_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Written: {stats_path}")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 50}")
    print(f"Output dir:      {out_dir}")
    print(f"Dev users:       {stats['dev_users']}")
    print(f"Test users:      {stats['test_users']}")
    print(f"Dev neg/user:    {stats['dev_neg_per_user']}")
    print(f"Test neg/user:   {stats['test_neg_per_user']}")
    print(f"Max item ID:     {stats['max_item_id']}")
    print(f"Check passed:    {stats['check_passed']}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
