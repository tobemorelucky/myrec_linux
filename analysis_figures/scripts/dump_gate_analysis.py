# -*- coding: UTF-8 -*-
"""
dump_gate_analysis.py
=====================
Load a trained MyModel checkpoint, run the test set with
return_intermediate=True to capture LGD gate weights, and export
all needed data for plotting.

Exports (per sample):
  user_ids           (N,)
  history_items      (N, L)   — item ids in history, 0 = padding
  target_items       (N,)     — ground-truth item id
  gates              (N, L)   — LGD per-position gate weight
  history_categories (N, L)   — category id per history item (-1 if unavailable)
  target_categories  (N,)     — category id of target item (-1 if unavailable)
  hit@5              (N,)     — bool: target in top-5 predictions

Saved to: analysis_figures/dumps/{dataset}_gate_analysis_dump.npz
"""

import os
import sys
import csv
import argparse
import logging
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import utils

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def load_item_category_map(dataset_dir: str):
    meta_path = os.path.join(dataset_dir, "item_meta.csv")
    if not os.path.exists(meta_path):
        logging.warning(f"item_meta.csv not found at {meta_path}")
        return None
    cat_map = {}
    try:
        with open(meta_path, "r", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                cat_map[int(row["item_id"])] = int(row["i_category"])
    except Exception as e:
        logging.warning(f"Failed to parse {meta_path}: {e}")
        return None
    logging.info(f"Loaded {len(cat_map)} item→category mappings")
    return cat_map


def map_categories(item_ids: np.ndarray, cat_map, default=-1):
    """Map item ids to category ids.  item_ids shape (...,)."""
    if cat_map is None:
        return np.full_like(item_ids, default, dtype=np.int64)
    flat = item_ids.ravel()
    out = np.full(flat.shape, default, dtype=np.int64)
    for i, iid in enumerate(flat):
        out[i] = cat_map.get(int(iid), default)
    return out.reshape(item_ids.shape)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Dump LGD gate analysis data")

    # Paths
    parser.add_argument("--model_name", type=str, default="MyModel")
    parser.add_argument("--dataset", type=str, default="beauty")
    parser.add_argument("--path", type=str, default="./data/")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, default="")

    # Model structure
    parser.add_argument("--emb_size", type=int, default=64)
    parser.add_argument("--attn_size", type=int, default=8)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--prompt_num", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--lamb", type=float, default=3.0)
    parser.add_argument("--history_max", type=int, default=20)

    # LLM
    parser.add_argument("--use_llmemb", type=int, default=1)
    parser.add_argument("--llm_fuse", type=int, default=1)
    parser.add_argument("--llm_emb_path", type=str, default="")
    parser.add_argument("--srs_emb_path", type=str, default="")
    parser.add_argument("--gamma_init", type=float, default=0.1)
    parser.add_argument("--gamma_trainable", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--tau", type=float, default=0.2)

    # EMILE / LGD
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

    # Misc
    parser.add_argument("--init_ckpt", type=str, default="")
    parser.add_argument("--init_strict", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=42)
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
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--pin_memory", type=int, default=1)
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

    # ---- Import model & reader ----
    import importlib
    model_module = importlib.import_module(f"models.sequential.{args.model_name}")
    model_class = getattr(model_module, args.model_name)
    reader_module = importlib.import_module(f"helpers.{model_class.reader}")
    reader_class = getattr(reader_module, model_class.reader)

    # ---- Load corpus ----
    corpus_path = os.path.join(args.path, args.dataset, reader_class.__name__ + ".pkl")
    if os.path.exists(corpus_path):
        corpus = pickle.load(open(corpus_path, "rb"))
    else:
        corpus = reader_class(args)
        pickle.dump(corpus, open(corpus_path, "wb"))
    logging.info(f"Corpus: {corpus.n_users} users, {corpus.n_items} items")

    # ---- Build model ----
    args.init_ckpt = ""
    model = model_class(args, corpus).to(device)
    model.load_model(args.ckpt, strict=False)
    logging.info("Checkpoint loaded")

    # Bypass warmup: set global_step past all warmup thresholds
    max_warmup = max(getattr(args, "logic_denoise_warmup_steps", 20000),
                     getattr(args, "emile_warmup_steps", 5000),
                     getattr(args, "rat_alpha_warmup_steps", 5000))
    model.global_step = max_warmup + 1
    model.eval()

    # ---- Build test dataset ----
    test_dataset = model_class.Dataset(model, corpus, "test")
    test_dataset.prepare()
    n_test = len(test_dataset)
    logging.info(f"Test samples: {n_test}")

    dl = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False,
                    num_workers=args.num_workers,
                    collate_fn=test_dataset.collate_batch,
                    pin_memory=bool(args.pin_memory))

    # ---- Collect data ----
    all_user_ids = []
    all_history = []
    all_target = []
    all_gates = []
    all_lengths = []

    with torch.no_grad():
        for batch in tqdm(dl, desc="Dump gates", ncols=100):
            batch = utils.batch_to_gpu(batch, device)
            out = model(batch, return_intermediate=True)
            gate = out.get("gate", None)
            if gate is None:
                raise RuntimeError(
                    "Gate is None! Check that use_logic_denoise=1 and "
                    "return_intermediate was passed through correctly."
                )

            # Pad to history_max for consistent concatenation
            B, L = batch["history_items"].shape
            if L < args.history_max:
                pad_len = args.history_max - L
                hist_pad = torch.zeros(B, pad_len, dtype=batch["history_items"].dtype, device=device)
                gate_pad = torch.zeros(B, pad_len, dtype=gate.dtype, device=device)
                hist = torch.cat([batch["history_items"], hist_pad], dim=1)
                g = torch.cat([gate, gate_pad], dim=1)
            else:
                hist = batch["history_items"]
                g = gate

            all_user_ids.append(batch["user_id"].cpu().numpy())
            all_history.append(hist.cpu().numpy())
            all_target.append(batch["item_id"][:, 0].cpu().numpy())
            all_gates.append(g.cpu().numpy())
            all_lengths.append(batch["lengths"].cpu().numpy())

    # Concatenate
    user_ids = np.concatenate(all_user_ids, axis=0).astype(np.int64)
    history_items = np.concatenate(all_history, axis=0).astype(np.int64)
    target_items = np.concatenate(all_target, axis=0).astype(np.int64)
    gates = np.concatenate(all_gates, axis=0).astype(np.float32)
    lengths = np.concatenate(all_lengths, axis=0).astype(np.int64)

    # Mask gates at padding positions to 0
    gates = gates * (history_items > 0).astype(np.float32)

    # ---- Categories ----
    dataset_dir = os.path.join(args.path, args.dataset)
    cat_map = load_item_category_map(dataset_dir)

    if cat_map is not None:
        history_categories = map_categories(history_items, cat_map, -1)
        target_categories = map_categories(target_items, cat_map, -1)
        logging.info("Category mapping successful")
    else:
        history_categories = np.full_like(history_items, -1, dtype=np.int64)
        target_categories = np.full_like(target_items, -1, dtype=np.int64)
        logging.info("Category unavailable; using -1 placeholders")

    # ---- Stats ----
    gate_valid = gates[history_items > 0]
    logging.info(f"history_items shape: {history_items.shape}")
    logging.info(f"gates shape: {gates.shape}")
    logging.info(f"gate  mean={float(gate_valid.mean()):.4f}  std={float(gate_valid.std()):.4f}"
                 f"  min={float(gate_valid.min()):.4f}  max={float(gate_valid.max()):.4f}")

    # ---- Save ----
    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(
            PROJECT_ROOT, "analysis_figures", "dumps",
            f"{args.dataset}_gate_analysis_dump.npz"
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    np.savez_compressed(
        out_path,
        user_ids=user_ids,
        history_items=history_items,
        target_items=target_items,
        gates=gates,
        lengths=lengths,
        history_categories=history_categories,
        target_categories=target_categories,
    )
    logging.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
