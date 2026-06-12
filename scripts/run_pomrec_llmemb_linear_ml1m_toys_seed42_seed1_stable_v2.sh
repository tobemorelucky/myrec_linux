#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU=0
LOG_DIR="logs/pomrec_llmemb_linear_stable_v2"
mkdir -p "${LOG_DIR}"

DATASETS=("ml-1m" "toys")
SEEDS=(42 1)

get_lamb () {
  local dataset="$1"
  case "$dataset" in
    ml-1m) echo "3.0" ;;
    toys)  echo "3.8" ;;
    *)     echo "3.0" ;;
  esac
}

get_lr () {
  local dataset="$1"
  case "$dataset" in
    ml-1m) echo "0.0001" ;;
    toys)  echo "0.0001" ;;
    *)     echo "0.0001" ;;
  esac
}

for DATASET in "${DATASETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    LAMB=$(get_lamb "${DATASET}")
    LR=$(get_lr "${DATASET}")

    LOG_FILE="${LOG_DIR}/${DATASET}_pomrec_llmemb_linear_seed${SEED}_lr${LR}_lamb${LAMB}_scale0.1_clip1.0.log"

    echo "============================================================"
    echo "[START] DATASET=${DATASET} SEED=${SEED}"
    echo "[MODEL] PoMRecLLMEmbLinear"
    echo "[LR] ${LR}"
    echo "[LAMB] ${LAMB}"
    echo "[LLM_SCALE] 0.1"
    echo "[GRAD_CLIP] 1.0"
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
      --grad_clip 1.0 \
      --llm_scale 0.1 \
      > "${LOG_FILE}" 2>&1

    echo "============================================================"
    echo "[DONE] DATASET=${DATASET} SEED=${SEED}"
    echo "============================================================"

    grep -Ei "loss=nan|nan=True|nan|inf|DIAG|Best Iter|Test After Training|HR@5|NDCG@5|HR@10|NDCG@10|HR@20|NDCG@20|HR@50|NDCG@50" "${LOG_FILE}" | tail -100 || true
    echo ""
  done
done

echo "============================================================"
echo "All PoMRecLLMEmbLinear stable v2 experiments finished."
echo "============================================================"

echo ""
echo "Quick summary:"
for f in "${LOG_DIR}"/*.log; do
  echo "============================================================"
  echo "$(basename "$f")"
  grep -Ei "loss=nan|nan=True|nan|inf|DIAG|Best Iter|Test After Training|HR@5|NDCG@5|HR@10|NDCG@10|HR@20|NDCG@20|HR@50|NDCG@50" "$f" | tail -80 || true
done
