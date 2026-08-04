# -*- coding: UTF-8 -*-
"""
dump_embedding_alignment.py
===========================
Load a trained MyModel checkpoint and export embeddings for alignment visualization.

Outputs
-------
e_cf      : collaborative item embeddings from model.i_embeddings  (N, emb_size)
s_raw     : raw LLM semantic embeddings from llm_table BEFORE adapter  (N, d_llm)
s_aligned : LLM embeddings AFTER adapter (Linear→GELU→Linear→LayerNorm)  (N, emb_size)
item_ids  : sampled item IDs (excluding padding 0)
labels    : category ID per sampled item (or 0 if no category)
label_names : human-readable category names

Saved to: analysis_figures/dumps/{dataset}_embedding_alignment_dump.npz
"""

import os
import sys
import csv
import argparse
import logging
import pickle
import numpy as np
import torch

# Ensure project root is on sys.path for imports like "from models.BaseModel import ..."
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from helpers.BaseReader import BaseReader
from helpers.SeqReader import SeqReader
from utils import utils


# =========================
#  Helper: load item → category mapping
# =========================
def load_item_category_map(dataset_dir: str):
    """
    Read item_meta.csv (tab-separated, item_id, i_category, ...).
    Returns dict: item_id (int) → category_id (int).
    Returns None if the file is missing or unreadable.
    """
    meta_path = os.path.join(dataset_dir, "item_meta.csv")
    if not os.path.exists(meta_path):
        logging.warning(f"item_meta.csv not found at {meta_path}, categories unavailable.")
        return None

    cat_map = {}
    try:
        with open(meta_path, "r", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                iid = int(row["item_id"])
                cid = int(row["i_category"])
                cat_map[iid] = cid
    except Exception as e:
        logging.warning(f"Failed to parse {meta_path}: {e}")
        return None

    logging.info(f"Loaded {len(cat_map)} item→category mappings from {meta_path}")
    return cat_map


# =========================
#  Sampling
# =========================
def sample_items(cat_map, n_items, sample_num, top_cates, rng):
    """
    If cat_map is available:
      - pick top_cates most frequent categories
      - stratified sample sample_num items from those categories
      - return (item_ids, labels, label_names)

    If cat_map is None:
      - random sample sample_num non-padding items
      - labels all 0, label_names = ["All"]
    """
    if cat_map is None:
        # Random sampling from 1..n_items-1
        pool = np.arange(1, n_items)
        if sample_num >= len(pool):
            sampled = pool.copy()
        else:
            sampled = rng.choice(pool, size=sample_num, replace=False)
        sampled = sampled.astype(np.int64)
        labels = np.zeros(len(sampled), dtype=np.int64)
        return sampled, labels, ["All"]

    # Count categories (only items present in cat_map)
    from collections import Counter
    cat_counts = Counter()
    for iid, cid in cat_map.items():
        if 1 <= iid < n_items:
            cat_counts[cid] += 1

    if not cat_counts:
        logging.warning("No valid category entries found; falling back to random.")
        return sample_items(None, n_items, sample_num, top_cates, rng)

    # Pick top-cates (excluding category 0 which means "unknown/uncategorized")
    top_list = [c for c, _ in cat_counts.most_common(top_cates + 2) if c != 0][:top_cates]
    logging.info(f"Top-{top_cates} category IDs: {top_list}  (freqs: {[cat_counts[c] for c in top_list]})")

    # Group item IDs by category
    cid_to_items = {c: [] for c in top_list}
    for iid, cid in cat_map.items():
        if cid in cid_to_items and 1 <= iid < n_items:
            cid_to_items[cid].append(iid)

    per_cat = max(1, sample_num // len(top_list))
    sampled_items = []
    sampled_labels = []
    for c in top_list:
        pool = cid_to_items[c]
        take = min(per_cat, len(pool))
        if take > 0:
            picks = rng.choice(pool, size=take, replace=False)
            sampled_items.append(picks)
            sampled_labels.append(np.full(take, c, dtype=np.int64))

    sampled = np.concatenate(sampled_items).astype(np.int64)
    labels = np.concatenate(sampled_labels).astype(np.int64)

    # Trim or pad to sample_num
    if len(sampled) > sample_num:
        idx = rng.choice(len(sampled), size=sample_num, replace=False)
        sampled = sampled[idx]
        labels = labels[idx]

    label_names = [f"Cat_{c}" for c in top_list]
    return sampled, labels, label_names


# =========================
#  Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Dump embeddings for alignment visualization")

    # ---- Paths ----
    parser.add_argument("--model_name", type=str, default="MyModel")
    parser.add_argument("--dataset", type=str, default="beauty")
    parser.add_argument("--path", type=str, default="./data/")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to trained MyModel .pt checkpoint")
    parser.add_argument("--output", type=str, default="",
                        help="Output npz path (default: auto-generated under dumps/)")

    # ---- Model structure (must match checkpoint) ----
    parser.add_argument("--emb_size", type=int, default=64)
    parser.add_argument("--attn_size", type=int, default=8)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--prompt_num", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--lamb", type=float, default=3.0)
    parser.add_argument("--history_max", type=int, default=20)

    # ---- LLM / Alignment ----
    parser.add_argument("--use_llmemb", type=int, default=1)
    parser.add_argument("--llm_fuse", type=int, default=1)
    parser.add_argument("--llm_emb_path", type=str, default="")
    parser.add_argument("--srs_emb_path", type=str, default="")
    parser.add_argument("--gamma_init", type=float, default=0.1)
    parser.add_argument("--gamma_trainable", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--tau", type=float, default=0.2)

    # ---- EMILE / LGD (set to 0 for dump; architecture handles it) ----
    parser.add_argument("--use_emile", type=int, default=1)
    parser.add_argument("--lambda_ipd", type=float, default=0.05)
    parser.add_argument("--ipd_margin", type=float, default=0.2)
    parser.add_argument("--emile_warmup_steps", type=int, default=5000)
    parser.add_argument("--emile_use_fused_itememb", type=int, default=0)

    parser.add_argument("--use_logic_denoise", type=int, default=1)
    parser.add_argument("--logic_denoise_alpha", type=float, default=8.0)
    parser.add_argument("--logic_denoise_b", type=float, default=0.3)
    parser.add_argument("--logic_denoise_topk", type=int, default=5)
    parser.add_argument("--logic_denoise_r", type=float, default=0.15)
    parser.add_argument("--logic_denoise_warmup_steps", type=int, default=20000)

    parser.add_argument("--use_logic_aggr", type=int, default=0)
    parser.add_argument("--lambda_logic_aggr", type=float, default=0.0)

    # ---- Warm-start (we load final ckpt manually) ----
    parser.add_argument("--init_ckpt", type=str, default="")
    parser.add_argument("--init_strict", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=42)

    # ---- Sampling ----
    parser.add_argument("--sample_num", type=int, default=600)
    parser.add_argument("--top_cates", type=int, default=6)

    # ---- Misc ----
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--l2", type=float, default=1e-06)
    parser.add_argument("--num_neg", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--test_all", type=int, default=0)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--load", type=int, default=0)
    parser.add_argument("--train", type=int, default=0)
    parser.add_argument("--regenerate", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=logging.INFO)
    parser.add_argument("--log_file", type=str, default="")
    parser.add_argument("--sep", type=str, default="\t")
    parser.add_argument("--buffer", type=int, default=0)

    parser.add_argument("--rat_alpha_warmup_steps", type=int, default=5000)
    parser.add_argument("--ilr_neg_weight", type=float, default=1.0)
    parser.add_argument("--logic_lambda_max", type=float, default=0.10)
    parser.add_argument("--logic_support_temp", type=float, default=2.0)
    parser.add_argument("--logic_gate_a", type=float, default=8.0)
    parser.add_argument("--logic_gate_b", type=float, default=0.8)

    args = parser.parse_args()

    logging.basicConfig(level=args.verbose, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Checkpoint: {args.ckpt}")

    # ---- Device ----
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if args.gpu != "" and torch.cuda.is_available() else "cpu")
    args.device = device
    logging.info(f"Device: {device}")

    # ---- Resolve model class (same pattern as main.py) ----
    import importlib
    model_module = importlib.import_module(f"models.sequential.{args.model_name}")
    model_class = getattr(model_module, args.model_name)
    reader_module = importlib.import_module(f"helpers.{model_class.reader}")
    reader_class = getattr(reader_module, model_class.reader)

    # ---- Load corpus ----
    corpus_path = os.path.join(args.path, args.dataset, reader_class.__name__ + ".pkl")
    if os.path.exists(corpus_path):
        logging.info(f"Load corpus from {corpus_path}")
        corpus = pickle.load(open(corpus_path, "rb"))
    else:
        logging.info("Corpus not found, building from scratch...")
        corpus = reader_class(args)
        pickle.dump(corpus, open(corpus_path, "wb"))

    n_items = corpus.n_items
    logging.info(f"Item count: {n_items}")

    # ---- Load category map ----
    dataset_dir = os.path.join(args.path, args.dataset)
    cat_map = load_item_category_map(dataset_dir)

    # ---- Build model ----
    # init_ckpt is kept empty so the constructor does NOT auto-load a warm-start checkpoint.
    # We'll load the final checkpoint manually.
    args.init_ckpt = ""
    logging.info("Building model...")
    model = model_class(args, corpus)
    model = model.to(device)
    model.eval()

    # ---- Load checkpoint ----
    logging.info(f"Loading checkpoint from {args.ckpt}")
    model.load_model(args.ckpt, strict=False)
    logging.info("Checkpoint loaded.")

    ie = model.interest_extractor

    # ---- Sample items ----
    rng = np.random.RandomState(args.random_seed)
    item_ids, labels, label_names = sample_items(
        cat_map, n_items, args.sample_num, args.top_cates, rng
    )
    logging.info(f"Sampled {len(item_ids)} items")
    logging.info(f"Label names: {label_names}")

    item_ids_t = torch.from_numpy(item_ids).to(device)

    # ---- Extract embeddings ----
    with torch.no_grad():
        # Collaborative item embeddings
        e_cf = ie.get_cf_emb(item_ids_t).cpu().numpy()

        # Raw LLM embeddings (BEFORE adapter)
        s_raw = ie.llm_table[item_ids_t].cpu().numpy()

        # Aligned LLM embeddings (AFTER adapter)
        s_aligned = ie.adapter(ie.llm_table[item_ids_t]).cpu().numpy()

    # ---- Report shapes ----
    logging.info(f"e_cf      shape: {e_cf.shape}")
    logging.info(f"s_raw     shape: {s_raw.shape}")
    logging.info(f"s_aligned shape: {s_aligned.shape}")

    # ---- Save ----
    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(
            PROJECT_ROOT, "analysis_figures", "dumps",
            f"{args.dataset}_embedding_alignment_dump.npz"
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    np.savez_compressed(
        out_path,
        item_ids=item_ids,
        labels=labels,
        label_names=np.array(label_names),
        e_cf=e_cf,
        s_raw=s_raw,
        s_aligned=s_aligned,
    )
    logging.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
