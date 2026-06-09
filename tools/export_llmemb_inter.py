"""
tools/export_llmemb_inter.py

Export this repo's data/<dataset>/train.csv, dev.csv, test.csv
to LLMEmb-format inter.txt (user_id item_id per line, sorted by user & time).

Usage:
  python tools/export_llmemb_inter.py \
    --src_data ./data/beauty \
    --out_path ../baselines/LLMEmb/data/beauty/handled/inter.txt
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict


def read_csv(path, sep):
    """Read a CSV file and return list of (user_id, item_id, time)."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter=sep)
        for row in reader:
            user_id = int(row["user_id"])
            item_id = int(row["item_id"])
            time = int(row["time"])
            rows.append((user_id, item_id, time))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Export interactions to LLMEmb inter.txt format"
    )
    parser.add_argument("--src_data", type=str, required=True,
                        help="Path to source data dir, e.g. ./data/beauty")
    parser.add_argument("--out_path", type=str, required=True,
                        help="Output path for inter.txt, e.g. ../baselines/LLMEmb/data/beauty/handled/inter.txt")
    parser.add_argument("--sep", type=str, default="\t",
                        help="CSV separator (default: tab)")
    parser.add_argument("--check_only", type=int, default=0,
                        help="If 1, only run checks without writing output")
    args = parser.parse_args()

    src_data = args.src_data
    out_path = args.out_path
    sep = args.sep
    check_only = args.check_only

    # ------------------------------------------------------------------
    # 1. Read all three splits
    # ------------------------------------------------------------------
    train_path = os.path.join(src_data, "train.csv")
    dev_path = os.path.join(src_data, "dev.csv")
    test_path = os.path.join(src_data, "test.csv")

    for path, name in [(train_path, "train"), (dev_path, "dev"), (test_path, "test")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name}.csv not found at {path}")

    train_rows = read_csv(train_path, sep)
    dev_rows = read_csv(dev_path, sep)
    test_rows = read_csv(test_path, sep)

    print(f"Read train: {len(train_rows)} rows")
    print(f"Read dev:   {len(dev_rows)} rows")
    print(f"Read test:  {len(test_rows)} rows")

    # ------------------------------------------------------------------
    # 2. Merge and sort by (user_id, time), stable
    # ------------------------------------------------------------------
    all_rows = train_rows + dev_rows + test_rows
    # Python sort is stable, so sort by time first, then by user_id
    all_rows.sort(key=lambda x: x[2])          # sort by time
    all_rows.sort(key=lambda x: x[0])          # stable sort by user_id

    # Also record per-user which item is from dev / test for later checks
    dev_item_of = {uid: iid for uid, iid, _ in dev_rows}
    test_item_of = {uid: iid for uid, iid, _ in test_rows}

    # ------------------------------------------------------------------
    # 3. Build per-user sequences (already sorted globally)
    # ------------------------------------------------------------------
    user_seqs = defaultdict(list)
    for uid, iid, ts in all_rows:
        user_seqs[uid].append(iid)

    # ------------------------------------------------------------------
    # 4. Sanity checks
    # ------------------------------------------------------------------
    # 4a. user_id and item_id must be >= 1
    all_uids = [r[0] for r in all_rows]
    all_iids = [r[1] for r in all_rows]
    min_uid, max_uid = min(all_uids), max(all_uids)
    min_iid, max_iid = min(all_iids), max(all_iids)

    assert min_uid >= 1, f"Found user_id = {min_uid} < 1, padding 0 violation!"
    assert min_iid >= 1, f"Found item_id = {min_iid} < 1, padding 0 violation!"
    print(f"Padding check passed: min_user_id={min_uid}, min_item_id={min_iid}")

    # 4b. Check per-user constraints
    errors = []
    warnings = []

    for uid, seq in user_seqs.items():
        if len(seq) < 3:
            warnings.append(f"user {uid}: seq_len={len(seq)} < 3")

        # Check: last item must be test item
        last_item = seq[-1]
        expected_test = test_item_of.get(uid)
        if expected_test is not None and last_item != expected_test:
            errors.append(
                f"user {uid}: last item {last_item} != test item {expected_test}"
            )

        # Check: second-to-last item must be dev item
        if len(seq) >= 2:
            second_last = seq[-2]
            expected_dev = dev_item_of.get(uid)
            if expected_dev is not None and second_last != expected_dev:
                errors.append(
                    f"user {uid}: 2nd-last item {second_last} != dev item {expected_dev}"
                )

    # 4c. Check that every user in dev/test appears in our sequences
    for uid in dev_item_of:
        if uid not in user_seqs:
            errors.append(f"user {uid}: in dev.csv but not in merged sequence")
    for uid in test_item_of:
        if uid not in user_seqs:
            errors.append(f"user {uid}: in test.csv but not in merged sequence")

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings[:20]:
            print(f"  - {w}")
        if len(warnings) > 20:
            print(f"  ... and {len(warnings) - 20} more")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        raise RuntimeError(
            f"Sequence check FAILED: {len(errors)} users have mismatched dev/test items. "
            f"See details above."
        )

    check_passed = len(errors) == 0
    print(f"\nSequence check: {'PASSED' if check_passed else 'FAILED'}")

    # ------------------------------------------------------------------
    # 5. Compute stats
    # ------------------------------------------------------------------
    seq_lens = [len(seq) for seq in user_seqs.values()]
    all_items = set(r[1] for r in all_rows)

    stats = {
        "user_num": len(user_seqs),
        "item_num": len(all_items),
        "interaction_num": len(all_rows),
        "min_user_id": min_uid,
        "max_user_id": max_uid,
        "min_item_id": min_iid,
        "max_item_id": max_iid,
        "avg_seq_len": sum(seq_lens) / len(seq_lens),
        "min_seq_len": min(seq_lens),
        "max_seq_len": max(seq_lens),
        "check_passed": check_passed,
    }

    # ------------------------------------------------------------------
    # 6. Write inter.txt
    # ------------------------------------------------------------------
    if not check_only:
        out_dir = os.path.dirname(os.path.abspath(out_path))
        os.makedirs(out_dir, exist_ok=True)

        with open(out_path, "w") as f:
            for uid, iid, _ in all_rows:
                f.write(f"{uid} {iid}\n")
        print(f"\nWritten inter.txt to: {out_path}")

        # 6b. Write stats JSON
        stats_path = os.path.join(out_dir, "inter_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Written stats to: {stats_path}")

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 50}")
    print(f"Output path:     {out_path if not check_only else '(check_only mode)'}")
    print(f"Users:           {stats['user_num']}")
    print(f"Items:           {stats['item_num']}")
    print(f"Interactions:    {stats['interaction_num']}")
    print(f"Avg seq len:     {stats['avg_seq_len']:.2f}")
    print(f"Check passed:    {stats['check_passed']}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
