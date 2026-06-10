#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="beauty"
SEED=42
GPU=0
LLM_EMB="./data/${DATASET}/handled/llm_table_pca1536.pkl"
LOG_DIR="logs/mvtc_score_modes_beauty"
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
  --use_mvtc 0
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

  grep -E "tic_score_mode|tic_score_lambda|tic_score_eta|Best Iter|Test After Training|HR@5|NDCG@5|HR@10|NDCG@10|HR@20|NDCG@20" "${log_file}" | tail -80 || true
  echo ""
}

# 1. replace baseline: tic_score_mode=none
run_exp \
  "beauty_seed42_replace_score_none" \
  "
  --tic_score_mode none
  "

# 2. pure candidate scoring，对照用
run_exp \
  "beauty_seed42_replace_score_candidate" \
  "
  --tic_score_mode candidate
  --tic_score_tau 0.2
  "

# 3. residual score calibration, lambda=0.05
run_exp \
  "beauty_seed42_replace_score_residual_lam005" \
  "
  --tic_score_mode residual
  --tic_score_tau 0.2
  --tic_score_lambda 0.05
  "

# 4. residual score calibration, lambda=0.1
run_exp \
  "beauty_seed42_replace_score_residual_lam01" \
  "
  --tic_score_mode residual
  --tic_score_tau 0.2
  --tic_score_lambda 0.1
  "

# 5. q-guided mix calibration, eta=0.1
run_exp \
  "beauty_seed42_replace_score_mix_eta01" \
  "
  --tic_score_mode mix
  --tic_score_tau 0.2
  --tic_score_eta 0.1
  "

echo ""
echo "All score-mode experiments finished."
echo "Logs saved to: ${LOG_DIR}"

echo ""
echo "Quick summary:"
for f in ${LOG_DIR}/beauty_seed42_replace_score*.log; do
  echo "============================================================"
  echo "$(basename "$f")"
  grep -E "Best Iter|Test After Training|HR@5|NDCG@5|HR@10|NDCG@10|HR@20|NDCG@20" "$f" | tail -30 || true
done

