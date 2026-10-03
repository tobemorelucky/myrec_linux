# -*- coding: UTF-8 -*-
"""
Generate a manifest of key experiment assets (path / size / mtime / sha1).

Motivation: `data/`, `model/`, `new_model/`, `new_log/` are all gitignored, so
experiment assets have no version-control protection. This tool produces a
checkable snapshot so a later stage can detect accidental overwrites.

Usage:
  python tools/make_asset_manifest.py                        # all datasets
  python tools/make_asset_manifest.py --dataset beauty
  python tools/make_asset_manifest.py --out ASSET_MANIFEST.tsv --no-hash

Output: ASSET_MANIFEST.tsv with columns
  category  dataset  path  size_bytes  mtime  sha1
"""

import argparse
import datetime
import glob
import hashlib
import os

import pickle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS = ["beauty", "ml-1m", "toys"]

# handled/<file> — the precomputed assets every chapter depends on
HANDLED_FILES = [
    "llm_table.pkl",
    "llm_table_pca1536.pkl",
    "itm_emb_pomrec.pkl",
    "llmmi_proto32_sr512.pkl",
    "llmmi_hier_proto32_8_sr512.pkl",
    "semantic_hardneg_top100.pkl",
    "cgscd_basis_crosscov_r32.pkl",
    "cgscd_basis_cca_r32.pkl",
]

# dataset-level files
DATA_FILES = ["train.csv", "dev.csv", "test.csv", "SeqReader.pkl", "item2id.json"]


def sha1_of(path, chunk=1 << 22):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def shape_of_pkl(path):
    """Best-effort shape probe; returns '' on failure."""
    try:
        obj = pickle.load(open(path, "rb"))
    except Exception:
        return ""
    if isinstance(obj, dict):
        parts = []
        for k in sorted(obj.keys()):
            v = obj[k]
            shp = getattr(v, "shape", None)
            parts.append(f"{k}{tuple(shp)}" if shp is not None else f"{k}={v}")
        return ";".join(parts)
    shp = getattr(obj, "shape", None)
    return str(tuple(shp)) if shp is not None else type(obj).__name__


def collect(datasets, with_hash=True, with_shape=True):
    rows = []

    def add(category, dataset, path):
        abspath = os.path.join(ROOT, path)
        if not os.path.exists(abspath):
            return
        st = os.stat(abspath)
        mt = datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        sha = sha1_of(abspath) if with_hash else ""
        shp = shape_of_pkl(abspath) if (with_shape and path.endswith(".pkl")) else ""
        rows.append((category, dataset, path, st.st_size, mt, sha, shp))

    for ds in datasets:
        for f in HANDLED_FILES:
            add("handled", ds, f"data/{ds}/handled/{f}")
        for f in DATA_FILES:
            add("data", ds, f"data/{ds}/{f}")

    # PoMRec checkpoints that back itm_emb_pomrec.pkl (provenance)
    provenance = {
        "beauty": "model/PoMRec/PoMRec__beauty__42__lr=0.002__l2=1e-06.pt",
        "ml-1m": "model/PoMRec/PoMRec__ml-1m__1__lr=0.001__l2=1e-06.pt",
        "toys": "model/PoMRec/toys__1__lr=0.001__l2=1e-06__lamb=3.8__history_max=20.pt",
    }
    for ds in datasets:
        if ds in provenance:
            add("provenance", ds, provenance[ds])

    # experiment summaries + frozen baseline checkpoints
    for pattern, cat in [
        ("new_log/llmmirec_aspcf_phase2/*/summary.tsv", "summary"),
        ("new_log/llmmirec_caisd_phase2/*/summary.tsv", "summary"),
        ("new_log/llmmirec_cgscd/*/summary.tsv", "summary"),
        ("new_log/llmmirec_cgscd/*/*/summary.tsv", "summary"),
    ]:
        for p in sorted(glob.glob(os.path.join(ROOT, pattern))):
            rel = os.path.relpath(p, ROOT)
            ds = rel.split("/")[2] if len(rel.split("/")) > 2 else ""
            add(cat, ds, rel)

    for p in sorted(glob.glob(os.path.join(ROOT, "new_model/llmmirec_*/*/*/*.pt"))):
        rel = os.path.relpath(p, ROOT)
        add("ckpt", rel.split("/")[2] if len(rel.split("/")) > 2 else "", rel)

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="", help="single dataset; default = all")
    ap.add_argument("--out", type=str, default="ASSET_MANIFEST.tsv")
    ap.add_argument("--no-hash", action="store_true")
    ap.add_argument("--no-shape", action="store_true")
    args = ap.parse_args()

    datasets = [args.dataset] if args.dataset else DATASETS
    rows = collect(datasets, with_hash=not args.no_hash, with_shape=not args.no_shape)

    out = os.path.join(ROOT, args.out)
    with open(out, "w") as f:
        f.write("category\tdataset\tpath\tsize_bytes\tmtime\tsha1\tshape\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")

    total = sum(r[3] for r in rows)
    print(f"Wrote {out}")
    print(f"  entries: {len(rows)}   total size: {total / 1e9:.2f} GB")
    for cat in sorted({r[0] for r in rows}):
        sub = [r for r in rows if r[0] == cat]
        print(f"    {cat:12s} {len(sub):4d} entries  {sum(r[3] for r in sub)/1e6:9.1f} MB")


if __name__ == "__main__":
    main()
