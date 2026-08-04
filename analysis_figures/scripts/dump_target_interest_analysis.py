# -*- coding: UTF-8 -*-
"""
dump_target_interest_analysis.py
================================
Load a trained MyModel checkpoint and export target-interest consistency
intermediate variables from the test set.

Exports
-------
user_ids            (N,)
target_items        (N,)
target_affinity     (N, K)   cosine similarity: pos_item vs each interest
k_star              (N,)     argmax of target_affinity
interest_weights    (N, K)   softmax over interest distribution
d_pos_H             (N,)     cos_dist(pos, user_vector)
d_pos_star          (N,)     min_k cos_dist(pos, interest_k)
d_neg_H             (N,)     cos_dist(neg, user_vector)  [if neg available]
target_gap          (N,)     d_pos_H - d_pos_star
neg_gap             (N,)     d_neg_H - d_pos_star  [if neg available]

Saved: analysis_figures/dumps/{dataset}_target_interest_analysis_dump.npz
"""

import os
import sys
import argparse
import logging
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import utils


def cos_sim(a, b):
    """Row-wise cosine similarity. a (N,D), b (N,D) -> (N,)"""
    a_n = F.normalize(a, dim=-1, eps=1e-8)
    b_n = F.normalize(b, dim=-1, eps=1e-8)
    return (a_n * b_n).sum(dim=-1)


def cos_dist(a, b):
    """Row-wise cosine distance."""
    return 1.0 - cos_sim(a, b)


def main():
    parser = argparse.ArgumentParser()
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

    # ---- Corpus ----
    corpus_path = os.path.join(args.path, args.dataset, reader_class.__name__ + ".pkl")
    if os.path.exists(corpus_path):
        corpus = pickle.load(open(corpus_path, "rb"))
    else:
        corpus = reader_class(args)
        pickle.dump(corpus, open(corpus_path, "wb"))
    logging.info(f"Corpus: {corpus.n_users} users, {corpus.n_items} items")

    # ---- Model ----
    args.init_ckpt = ""
    model = model_class(args, corpus).to(device)
    model.load_model(args.ckpt, strict=False)
    logging.info("Checkpoint loaded")

    max_warmup = max(getattr(args, "logic_denoise_warmup_steps", 20000),
                     getattr(args, "emile_warmup_steps", 5000),
                     getattr(args, "rat_alpha_warmup_steps", 5000))
    model.global_step = max_warmup + 1
    model.eval()

    # ---- Test dataset ----
    test_dataset = model_class.Dataset(model, corpus, "test")
    test_dataset.prepare()
    n_test = len(test_dataset)
    logging.info(f"Test samples: {n_test}")

    dl = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False,
                    num_workers=args.num_workers,
                    collate_fn=test_dataset.collate_batch,
                    pin_memory=bool(args.pin_memory))

    # ---- Collect ----
    all_affinity = []
    all_k_star = []
    all_weights = []
    all_d_pos_H = []
    all_d_pos_star = []
    all_d_neg_H = []
    all_user_ids = []
    all_targets = []

    has_neg = True

    with torch.no_grad():
        for batch in tqdm(dl, desc="Dump target-interest", ncols=100):
            batch = utils.batch_to_gpu(batch, device)
            out = model(batch, return_intermediate=True)

            iv = out["interest_vectors"]      # (B, K, D)
            w = out["interest_weights"]        # (B, K)
            uv = out["user_vector"]            # (B, D)
            pos = out["pos_item_emb"]          # (B, D)
            neg = out.get("neg_item_emb", None)

            B, K, D = iv.shape

            # target_affinity: cos(pos, interest_k) for each k
            pos_exp = pos[:, None, :]                    # (B, 1, D)
            affinity = cos_sim(pos_exp.expand(-1, K, -1).reshape(B * K, D),
                               iv.reshape(B * K, D)).reshape(B, K)

            k_star = affinity.argmax(dim=1)               # (B,)

            d_pos_H = cos_dist(pos, uv)                  # (B,)
            # d_pos_star: min_k cos_dist(pos, interest_k)
            # cos_dist = 1 - cos_sim → min distance = 1 - max affinity
            d_pos_star = 1.0 - affinity.max(dim=1).values  # (B,)

            all_affinity.append(affinity.cpu().numpy())
            all_k_star.append(k_star.cpu().numpy())
            all_weights.append(w.cpu().numpy())
            all_d_pos_H.append(d_pos_H.cpu().numpy())
            all_d_pos_star.append(d_pos_star.cpu().numpy())
            all_user_ids.append(batch["user_id"].cpu().numpy())
            all_targets.append(batch["item_id"][:, 0].cpu().numpy())

            if neg is not None:
                d_neg_H = cos_dist(neg, uv)
                all_d_neg_H.append(d_neg_H.cpu().numpy())
            else:
                has_neg = False

    # ---- Concatenate ----
    user_ids = np.concatenate(all_user_ids).astype(np.int64)
    target_items = np.concatenate(all_targets).astype(np.int64)
    target_affinity = np.concatenate(all_affinity).astype(np.float32)
    k_star = np.concatenate(all_k_star).astype(np.int64)
    interest_weights = np.concatenate(all_weights).astype(np.float32)
    d_pos_H = np.concatenate(all_d_pos_H).astype(np.float32)
    d_pos_star = np.concatenate(all_d_pos_star).astype(np.float32)
    target_gap = (d_pos_H - d_pos_star).astype(np.float32)

    neg_gap_arr = None
    d_neg_H_arr = None
    if has_neg and len(all_d_neg_H) > 0:
        d_neg_H_arr = np.concatenate(all_d_neg_H).astype(np.float32)
        neg_gap_arr = (d_neg_H_arr - d_pos_star).astype(np.float32)

    # ---- Stats ----
    K = target_affinity.shape[1]
    logging.info(f"interest_vectors: was ({len(user_ids)}, {K}, {model.emb_size})")
    logging.info(f"target_affinity shape: {target_affinity.shape}")
    k_dist = np.bincount(k_star, minlength=K)
    logging.info(f"k* distribution: {dict(enumerate(k_dist))}")
    logging.info(f"target_gap  mean={float(target_gap.mean()):.4f}, std={float(target_gap.std()):.4f}")

    if neg_gap_arr is not None:
        logging.info(f"neg_gap     mean={float(neg_gap_arr.mean()):.4f}, std={float(neg_gap_arr.std()):.4f}")
    else:
        logging.info("neg_gap unavailable (no negative item embeddings in test batches)")

    # ---- Save ----
    out_path = args.output or os.path.join(
        PROJECT_ROOT, "analysis_figures", "dumps",
        f"{args.dataset}_target_interest_analysis_dump.npz"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    save = dict(
        user_ids=user_ids,
        target_items=target_items,
        target_affinity=target_affinity,
        k_star=k_star,
        interest_weights=interest_weights,
        d_pos_H=d_pos_H,
        d_pos_star=d_pos_star,
        target_gap=target_gap,
    )
    if neg_gap_arr is not None:
        save["d_neg_H"] = d_neg_H_arr
        save["neg_gap"] = neg_gap_arr

    np.savez_compressed(out_path, **save)
    logging.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
