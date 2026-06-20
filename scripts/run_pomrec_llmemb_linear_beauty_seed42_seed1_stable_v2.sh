#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU=1
LOG_DIR="logs/pomrec_llmemb_linear_stable_v2"
mkdir -p "${LOG_DIR}"

DATASET="beauty"
SEEDS=(42 1)

LAMB="3.0"
LR="0.0005"
LLM_SCALE="0.1"
GRAD_CLIP="1.0"

for SEED in "${SEEDS[@]}"; do
  LOG_FILE="${LOG_DIR}/${DATASET}_pomrec_llmemb_linear_seed${SEED}_lr${LR}_lamb${LAMB}_scale${LLM_SCALE}_clip${GRAD_CLIP}.log"

  echo "============================================================"
  echo "[START] DATASET=${DATASET} SEED=${SEED}"
  echo "[MODEL] PoMRecLLMEmbLinear"
  echo "[LR] ${LR}"
  echo "[LAMB] ${LAMB}"
  echo "[LLM_SCALE] ${LLM_SCALE}"
  echo "[GRAD_CLIP] ${GRAD_CLIP}"
  echo "[LOG] ${LOG_FILE}"
  echo "============================================================"

  python main.py \
    --model_name PoMRecLLMEmbLinear \
    --dataset "${DATASET}" \
    --path ./data/ \
    --gpu "${GPU}" \
    --random_seed "${SEED}" \
    --emb_size 64 \
    --attn_size 8 \
    --K 3 \
    --prompt_num 4 \
    --n_layers 2 \
    --lamb "${LAMB}" \
    --history_max 20 \
    --lr "${LR}" \
    --l2 1e-6 \
    --batch_size 256 \
    --eval_batch_size 256 \
    --epoch 200 \
    --early_stop 10 \
    --num_neg 1 \
    --dropout 0 \
    --num_workers 5 \
    --grad_clip "${GRAD_CLIP}" \
    --llm_scale "${LLM_SCALE}" \
    > "${LOG_FILE}" 2>&1

  echo "============================================================"
  echo "[DONE] DATASET=${DATASET} SEED=${SEED}"
  echo "============================================================"

  grep -Ei "loss=nan|nan=True|nan|inf|DIAG|Best Iter|Test After Training|HR@5|NDCG@5|HR@10|NDCG@10|HR@20|NDCG@20|HR@50|NDCG@50" "${LOG_FILE}" | tail -100 || true
  echo ""
done

echo "============================================================"
echo "Beauty PoMRecLLMEmbLinear stable v2 experiments finished."
echo "============================================================"

echo ""
echo "Quick summary:"
for f in "${LOG_DIR}/${DATASET}"_pomrec_llmemb_linear_seed*_lr${LR}_lamb${LAMB}_scale${LLM_SCALE}_clip${GRAD_CLIP}.log; do
  echo "============================================================"
  echo "$(basename "$f")"
  grep -Ei "loss=nan|nan=True|nan|inf|DIAG|Best Iter|Test After Training|HR@5|NDCG@5|HR@10|NDCG@10|HR@20|NDCG@20|HR@50|NDCG@50" "$f" | tail -80 || true
done
