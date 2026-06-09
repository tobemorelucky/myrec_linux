#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="beauty"
SEED=42
GPU=0
LLM_EMB="./data/${DATASET}/handled/llm_table_pca1536.pkl"
LOG_DIR="logs/mvtc_lite_beauty"
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
  --lr 0.002
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

  grep -E "tic_score_mode|use_mvtc|mvtc_weight|Best Iter|Test After Training|HR@5|NDCG@5|HR@10|NDCG@10|HR@20|NDCG@20|mvtc_loss" "${log_file}" | tail -80 || true
  echo ""
}

# 1. 原始 replace baseline：必须复现之前 seed42 结果
run_exp \
  "beauty_seed42_replace_base" \
  "
  --tic_score_mode none
  --use_mvtc 0
  "

# 2. 只改候选感知打分，不加 MVTC loss
run_exp \
  "beauty_seed42_replace_candidate_score" \
  "
  --tic_score_mode candidate
  --tic_score_tau 0.2
  --use_mvtc 0
  "

# 3. 候选感知打分 + MVTC CF teacher loss
run_exp \
  "beauty_seed42_replace_candidate_score_mvtc001" \
  "
  --tic_score_mode candidate
  --tic_score_tau 0.2
  --use_mvtc 1
  --mvtc_weight 0.001
  --mvtc_tau_cf 0.2
  "

echo "All MVTC-Lite experiments finished."
echo "Logs saved to: ${LOG_DIR}"

echo ""
echo "Quick summary:"
for f in ${LOG_DIR}/beauty_seed42_replace*.log; do
  echo "============================================================"
  echo "$(basename "$f")"
  grep -E "Best Iter|Test After Training|HR@5|NDCG@5|HR@10|NDCG@10|HR@20|NDCG@20" "$f" | tail -30 || true
done


