#!/usr/bin/env bash
set -e

# =========================================================
# LLMMIRec CAISD Phase 2 (final) — ML-1M
#
# Frozen method: uniform distill + responsibility teacher,
# no relation JS, no confidence, no responsibility_power.
#
# Usage:
#   bash new_bash/run_llmmirec_caisd_phase2_ml1m.sh GPU SEED
#   e.g. bash new_bash/run_llmmirec_caisd_phase2_ml1m.sh 1 42
# =========================================================

GPU=${1:-0}
SEED=${2:-42}
DATASET="ml-1m"

MODEL_NAME="LLMMIRecCAISD"
LR=0.001
BATCH_SIZE=1024
EVAL_BATCH_SIZE=256
LAMBDA_INTEREST_SEMANTIC=0.005

TASID_MODE=${TASID_MODE:-none}
TASID_STUDENT_MODE=${TASID_STUDENT_MODE:-llm_only}
LAMBDA_TASID=${LAMBDA_TASID:-0.01}
TASID_TEMP=${TASID_TEMP:-0.1}

ROOT_LOG_DIR="new_log/llmmirec_caisd_phase2/${DATASET}/seed${SEED}"
ROOT_MODEL_DIR="new_model/llmmirec_caisd_phase2/${DATASET}/seed${SEED}"
SUMMARY_FILE="new_log/llmmirec_caisd_phase2/${DATASET}/summary.tsv"

mkdir -p "${ROOT_LOG_DIR}" "${ROOT_MODEL_DIR}" "$(dirname "${SUMMARY_FILE}")"

if [ ! -f "${SUMMARY_FILE}" ]; then
  echo -e "dataset\tmodel\tseed\tteacher_mode\tlambda_interest_semantic\tlr\tbatch_size\tstart_time\tend_time\ttotal_seconds\tparameter_count\tbest_epoch\tbest_dev\ttest_after_training\tstatus" > "${SUMMARY_FILE}"
fi

LOG_FILE="${ROOT_LOG_DIR}/LLMMIRecCAISD_seed${SEED}.log"
OUT_FILE="${ROOT_LOG_DIR}/LLMMIRecCAISD_seed${SEED}.out"
MODEL_PATH="${ROOT_MODEL_DIR}/LLMMIRecCAISD_seed${SEED}.pt"

TEACHER_PATH="./data/${DATASET}/handled/llmmi_proto32_sr512.pkl"

START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
START_TS=$(date +%s)
echo "[START] ${START_TIME} | CAISD ${DATASET} seed=${SEED} gpu=${GPU}"

python main.py \
  --model_name "${MODEL_NAME}" \
  --dataset "${DATASET}" \
  --path ./data/ \
  --gpu "${GPU}" \
  --random_seed "${SEED}" \
  --emb_size 64 \
  --attn_size 64 \
  --K 4 \
  --history_max 20 \
  --item_encoder aspcf \
  --llm_emb_path "./data/${DATASET}/handled/llm_table_pca1536.pkl" \
  --semantic_rank 512 \
  --semantic_dim 32 \
  --semantic_hidden 128 \
  --complement_dim 32 \
  --tail_hidden 64 \
  --complement_hidden 64 \
  --gate_hidden 64 \
  --aspcf_gate_mode basic \
  --lambda_relation 0.01 \
  --relation_sample_size 128 \
  --relation_teacher_temp 0.1 \
  --relation_student_temp 0.1 \
  --semantic_teacher_path "${TEACHER_PATH}" \
  --semantic_distill_mode uniform \
  --semantic_teacher_mode responsibility \
  --semantic_responsibility_alpha 0.5 \
  --semantic_relation_mode none \
  --lambda_interest_semantic "${LAMBDA_INTEREST_SEMANTIC}" \
  --tasid_mode "${TASID_MODE}" \
  --tasid_student_mode "${TASID_STUDENT_MODE}" \
  --lambda_tasid "${LAMBDA_TASID}" \
  --tasid_temp "${TASID_TEMP}" \
  --dropout 0.1 \
  --lr "${LR}" \
  --l2 1e-6 \
  --batch_size "${BATCH_SIZE}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --num_neg 1 \
  --epoch 200 \
  --early_stop 10 \
  --topk 5,10,20,50 \
  --metric NDCG,HR \
  --num_workers 5 \
  --log_file "${LOG_FILE}" \
  --model_path "${MODEL_PATH}" \
  2>&1 | tee "${OUT_FILE}"

END_TIME=$(date "+%Y-%m-%d %H:%M:%S")
END_TS=$(date +%s)
TOTAL_SECONDS=$((END_TS - START_TS))

BEST_DEV=$(grep "Best Iter(dev)" "${LOG_FILE}" | tail -n 1 | tr '\t' ' ' | sed 's/[[:space:]]\+/ /g')
TEST_AFTER=$(grep "Test After Training" "${LOG_FILE}" | tail -n 1 | tr '\t' ' ' | sed 's/[[:space:]]\+/ /g')
BEST_EPOCH=$(echo "${BEST_DEV}" | grep -oP 'Best Iter\(dev\)=\s*\K[0-9]+' || echo "NA")
PARAM_COUNT=$(grep "#params:" "${LOG_FILE}" | tail -n 1 | grep -oP '[0-9]+' || echo "NA")
STATUS="OK"
if grep -E "(loss=nan|dev=\(HR@5:nan|dev=\(HR@5:0.0000)" "${LOG_FILE}" | grep -v "check" > /dev/null 2>&1; then STATUS="NaN_CRASH"; fi

echo "[DONE] ${START_TIME} -> ${END_TIME} (${TOTAL_SECONDS}s)"
echo "${BEST_DEV}"
echo "${TEST_AFTER}"

echo -e "${DATASET}\t${MODEL_NAME}\t${SEED}\tresponsibility\t${LAMBDA_INTEREST_SEMANTIC}\t${LR}\t${BATCH_SIZE}\t${START_TIME}\t${END_TIME}\t${TOTAL_SECONDS}\t${PARAM_COUNT}\t${BEST_EPOCH}\t${BEST_DEV}\t${TEST_AFTER}\t${STATUS}" >> "${SUMMARY_FILE}"
