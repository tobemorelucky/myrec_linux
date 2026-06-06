# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorch research codebase for multi-interest sequential recommendation. Based on the PoMRec backbone, extended with three modules:
1. **LLM Semantic Alignment & Fusion** — InfoNCE-aligned LLM item embeddings fused via `item_emb = cf_emb + gamma * llm_emb`
2. **IPD (Target-Interest Consistency)** — EMILE-style cosine-distance BPR constraints between aggregated user vector, best-matching interest, positive item, and negative item
3. **LGD (Logic-Guided Denoising)** — Self-conditioned two-pass soft denoising: first pass extracts coarse intent query without denoising, second pass uses similarity-based gates to reweight history items

## Environment

```bash
conda activate hzg_py10
```

Start Claude Code with `bash start_claude.sh`.

## Running Experiments

### Single runs

```bash
# Baseline PoMRec
python main.py --model_name PoMRec --dataset ml-1m --lr 0.001

# PoMRec with LLM semantic enhancement
python main.py --model_name PoMRec --dataset beauty --use_llmemb 1 \
  --llm_emb_path ./data/beauty/llm_emb.pkl --srs_emb_path ./data/beauty/srs_emb.pkl \
  --alpha 0.001 --tau 0.2 --llm_fuse 1

# MyModel: full paper version (LLM + IPD + LGD)
python main.py --model_name MyModel --dataset beauty \
  --lr 0.002 --lamb 3.0 \
  --use_llmemb 1 --llm_fuse 1 \
  --llm_emb_path ./data/beauty/handled/llm_table_pca1536.pkl \
  --srs_emb_path ./data/beauty/handled/itm_emb_pomrec.pkl \
  --gamma_init 0.1 --gamma_trainable 0 \
  --alpha 0.001 --tau 0.2 --rat_alpha_warmup_steps 5000 \
  --init_ckpt <PoMRec_checkpoint.pt> --init_strict 0 \
  --use_emile 1 --lambda_ipd 0.05 --ipd_margin 0.2 --emile_warmup_steps 5000 \
  --use_logic_denoise 1 --logic_denoise_alpha 8.0 --logic_denoise_b 0.3 \
  --logic_denoise_topk 5 --logic_denoise_r 0.15 --logic_denoise_warmup_steps 20000 \
  --use_logic_aggr 0 --lambda_logic_aggr 0.0
```

### Multi-seed scripts (paper version)

All in `bash脚本/`. Each runs seeds `0 1 2 3 41 42 43` with warm-start from PoMRec checkpoint.

| Script | Dataset | GPU | Key hyperparameters |
|--------|---------|-----|---------------------|
| `run_beauty_multiseed_final.sh` | Beauty | 0 | lr=0.002, lamb=3.0, γ=0.1, τ=0.2, λ_ipd=0.05, m=0.2, LGD(α=8,b=0.3,topk=5,r=0.15) |
| `run_ml1m_full3_multiseed.sh` | ML-1M | 1 | lr=0.001, lamb=3.0, γ=0.08, τ=0.3, λ_ipd=0.02, m=0.10, LGD(α=8,b=0.40,topk=5,r=0.08) |
| `run_toys_final_multiseed_best.sh` | Toys | 1 | lr=0.001, lamb=3.8, γ=0.05, τ=0.5, λ_ipd=0.05, m=0.10, LGD(α=10,b=0.3,topk=10,r=0.10) |

Ablation scripts:
- `run_ml1m_ablation_multiseed.sh` — ML-1M: LLM-only and LLM+IPD variants, 5 seeds
- `run_alloff_s42_3datasets.sh` — All modules off on all 3 datasets, seed=42 only

ML-1M and Toys both use GPU 1 — don't run them simultaneously on single-GPU without modifying `CUDA_VISIBLE_DEVICES`.

## Architecture

### Entry point (`main.py`)

Uses `--model_name` to dynamically resolve the model class via `eval('{0}.{0}'.format(model_name))`. Each model class specifies its own `reader`, `runner`, and `extra_log_args`. Pipeline: read data → build corpus → create model → train (BaseRunner) → evaluate.

### Inheritance chain

```
BaseModel (models/BaseModel.py)
  └─ GeneralModel: adds BPR loss, negative sampling, item/user counts
       └─ SequentialModel: adds history sequence handling via SeqReader
            ├─ PoMRec (models/sequential/PoMRec.py): baseline multi-interest model
            └─ MyModel (models/sequential/MyModel.py): paper version (LLM + IPD + LGD)
```

### Data flow

1. `BaseReader` reads `train.csv`/`dev.csv`/`test.csv` from `data/<dataset>/` (tab-separated, `--sep` to override). Minimum columns: `user_id`, `item_id`, `time`. Dev/test may include `neg_items`.
2. `SeqReader` merges all splits, sorts by time, builds per-user history sequences with position indices.
3. Corpus is pickled to `data/<dataset>/SeqReader.pkl`; use `--regenerate 1` to rebuild.
4. `BaseRunner.train()` runs training with early stopping on the first metric@K (default: `NDCG@5`).

### Multi-Interest Extractor

Core module shared by PoMRec and MyModel. Uses dual prompt embeddings (extractor + aggregator prompts) with self-attention over padded history. Outputs K interest vectors with centrality-discreteness weighting (`--lamb`). When LGD is enabled, supports an optional `q_vec` argument for similarity-based history reweighting.

### LLM semantic enhancement (`--use_llmemb 1`)

Loads precomputed LLM item embeddings from `.pkl`, maps via 2-layer MLP adapter, fuses as `cf_emb + gamma * llm_emb`. InfoNCE alignment loss between adapted LLM embeddings and anchor (SRS or CF) embeddings. Requires `--llm_emb_path`; `--srs_emb_path` is recommended but optional.

### MyModel: IPD + LGD (paper version)

1. **IPD** (`--use_emile 1`): Computes cosine distances between positive/negative items, aggregated user vector H, and best-matching interest vector h*. Applies three BPR constraints: d(pos,H) < d(pos,h*), d(pos,H) < d(neg,H), d(pos,h*) < d(neg,H). Uses `--emile_use_fused_itememb` to optionally use fused (CF+LLM) item embeddings for distance computation.

2. **LGD** (`--use_logic_denoise 1`): Self-conditioned two-pass denoising:
   - Pass 1: extract interests **without** denoising → aggregate to intent query q
   - Pass 2: compute similarity(q, each history item CF embedding) → sigmoid gate → soft reweight history → extract final interests
   - No target item used in query construction (avoids label leakage)

3. **logic_aggr** (`--use_logic_aggr`): Parameter is read for compatibility but **not used in forward** — prediction always uses `pred_base`.

### Other model variants in `models/sequential/`

- `MyModel_PASV3-6.5失败版.py` — Failed PAS-v3 update (period-aware semantic-collaborative router), preserved for reference
- `MyModel_PAS.py`, `MyModel_PASv2.py`, `MyModel_PASv3.py` — Earlier PAS attempts
- `MyModel_SADIR.py` and variants — SADIR experiments
- `MyModel_EACD.py` — EACD variant
- `MyModel（论文版）.py` — Backup copy of the paper version

### Key design constraints

- Item ID 0 is always padding. Item embedding tables are `(n_items, emb_size)` with row 0 unused.
- LLM embedding `.pkl` files are 2D numpy arrays. Code auto-pads row 0 if table has N rows instead of N+1.
- Models save to `./model/<ModelName>/` and logs to `./log/<ModelName>/` automatically.
- Log filenames encode key parameters, truncated with MD5 hash suffix if >180 chars.
- Training uses warm-start from PoMRec checkpoint (`--init_ckpt`) with `--init_strict 0` (shape-matched keys only).

## Important files

| File | Role |
|---|---|
| `main.py` | Entry point, arg parsing, training orchestration |
| `models/BaseModel.py` | Base/General/Sequential model classes, BPR loss, Dataset definitions |
| `models/sequential/PoMRec.py` | PoMRec baseline + `MultiInterestExtractor` |
| `models/sequential/MyModel.py` | Paper version: LLM alignment + IPD + LGD |
| `helpers/BaseRunner.py` | Training loop, evaluation (HR@K, NDCG@K), early stopping |
| `helpers/BaseReader.py` | CSV data loading, user/item counts |
| `helpers/SeqReader.py` | History sequence construction with time ordering |
| `utils/utils.py` | Seed init, GPU transfer, metric formatting |
| `utils/layers.py` | MultiHeadAttention, TransformerLayer |
| `tools/export_pomrec_item_emb.py` | Export trained item embeddings from checkpoint |
| `tools/convert_llmemb_jsonl_to_pkl_table.py` | Convert LLM embeddings from JSONL to pkl table |
