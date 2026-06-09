#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="ml-1m"
SEED=42
GPU=0
LOG="logs/pomrec_llmreplace_stage1/${DATASET}_replace_seed${SEED}.log"

CANDIDATES=(
  "./data/${DATASET}/handled/llm_table_pca1536.pkl"
  "./data/${DATASET}/handled/llm_table_pca64.pkl"
  "./data/${DATASET}/handled/default_pca.pkl"
)

LLM_EMB=""
for p in "${CANDIDATES[@]}"; do
  if [[ -f "$p" ]]; then
    LLM_EMB="$p"
    break
  fi
done

if [[ -z "$LLM_EMB" ]]; then
  echo "[ERROR] Cannot find ML-1M LLM embedding."
  echo "Checked:"
  printf '  %s\n' "${CANDIDATES[@]}"
  exit 1
fi

echo "[START] ${DATASET} replace seed=${SEED}"
echo "[LLM_EMB] $LLM_EMB"
echo "[LOG] $LOG"

python main.py \
  --model_name PoMRecLLMEmb \
  --dataset ${DATASET} \
  --path ./data/ \
  --gpu ${GPU} \
  --random_seed ${SEED} \
  --emb_size 64 \
  --attn_size 8 \
  --K 4 \
  --prompt_num 3 \
  --n_layers 2 \
  --lamb 4.0 \
  --history_max 20 \
  --lr 0.002 \
  --l2 1e-6 \
  --batch_size 256 \
  --eval_batch_size 256 \
  --epoch 200 \
  --early_stop 10 \
  --num_neg 1 \
  --dropout 0 \
  --num_workers 5 \
  --llm_fuse_mode replace \
  --llm_emb_path ${LLM_EMB} \
  --freeze_llm_emb 1 \
  --use_llm_align 0 \
  > "$LOG" 2>&1

echo "[DONE] ${DATASET} replace seed=${SEED}"
grep -E "Best Iter|Test After Training|HR@5|NDCG@5|HR@10|NDCG@10|HR@20|NDCG@20" "$LOG" | tail -40 || true
