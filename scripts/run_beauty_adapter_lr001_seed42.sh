#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="beauty"
SEED=42
GPU=0
LR=0.001
LLM_EMB="./data/${DATASET}/handled/llm_table_pca1536.pkl"
LOG_DIR="logs/adapter_scope_ablation_beauty_lr001"
mkdir -p "${LOG_DIR}"

if [[ ! -f "${LLM_EMB}" ]]; then
  echo "[ERROR] LLM embedding not found: ${LLM_EMB}"
  exit 1
fi

COMMON_ARGS="
  --model_name PoMRecLLMEmb
  --dataset ${DATASET}
  --path ./data/
  --gpu ${GPU}
  --random_seed ${SEED}
  --emb_size 64
  --attn_size 8
  --K 4
  --prompt_num 3
  --n_layers 2
  --lamb 4.0
  --history_max 20
  --lr ${LR}
  --l2 1e-6
  --batch_size 256
  --eval_batch_size 256
  --epoch 200
  --early_stop 10
  --num_neg 1
  --dropout 0
  --num_workers 5
  --llm_fuse_mode replace
  --llm_emb_path ${LLM_EMB}
  --freeze_llm_emb 1
  --use_llm_align 0
  --use_tic 0
  --tic_score_mode none
  --use_mvtc 0
  --llm_inject_scope both
"

run_exp () {
  local name="$1"
  local extra_args="$2"
  local log_file="${LOG_DIR}/${name}.log"

  echo "============================================================"
  echo "[START] ${name}"
  echo "[LOG]   ${log_file}"
  echo "============================================================"

  python main.py ${COMMON_ARGS} ${extra_args} > "${log_file}" 2>&1

  echo "============================================================"
  echo "[DONE] ${name}"
  echo "============================================================"

  grep -E "lr|llm_adapter_arch|llm_inject_scope|loss=nan|Best Iter|Test After Training|HR@5|NDCG@5|HR@10|NDCG@10|HR@20|NDCG@20" "${log_file}" | tail -80 || true
  echo ""
}

# 1. LLMEmb-style adapter, lr=0.001
run_exp \
  "beauty_seed42_adapter_llmemb_scope_both_lr001" \
  "
  --llm_adapter_arch llmemb
  "

# 2. w/o LayerNorm adapter, lr=0.001
run_exp \
  "beauty_seed42_adapter_noln_scope_both_lr001" \
  "
  --llm_adapter_arch noln
  "

echo ""
echo "All lr=0.001 adapter ablation experiments finished."
echo "Logs saved to: ${LOG_DIR}"

echo ""
echo "Quick summary:"
for f in ${LOG_DIR}/beauty_seed42_adapter*.log; do
  echo "============================================================"
  echo "$(basename "$f")"
  grep -E "llm_adapter_arch|llm_inject_scope|loss=nan|Best Iter|Test After Training|HR@5|NDCG@5|HR@10|NDCG@10|HR@20|NDCG@20" "$f" | tail -40 || true
done
