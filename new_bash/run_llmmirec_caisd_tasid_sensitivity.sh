#!/usr/bin/env bash
set -e

# =========================================================
# LLMMIRec CAISD — TASID sensitivity sweep (lambda / temperature)
#
# Runs the frozen Phase2 CAISD config with TASID on/off across
# a range of lambda_tasid or tasid_temp values. Each value gets
# its own subdirectory (no overwrites).
#
# Usage:
#   # lambda sensitivity (Beauty)
#   DATASET=beauty SWEEP_TYPE=lambda \
#     VALUES="0 0.001 0.005 0.01 0.05 0.1" \
#     bash new_bash/run_llmmirec_caisd_tasid_sensitivity.sh 0 42
#
#   # temperature sensitivity (ML-1M)
#   DATASET=ml-1m SWEEP_TYPE=temp \
#     VALUES="0.05 0.1 0.2 0.5" \
#     bash new_bash/run_llmmirec_caisd_tasid_sensitivity.sh 1 42
#
# Env overrides:
#   TASID_STUDENT_MODE (default asymmetric)
#   LAMBDA_TASID       (fixed lambda for temp sweep, default 0.01)
#   TASID_TEMP         (fixed temp for lambda sweep, default 0.1)
#   RUN_BASELINE       (1=also run tasid-off baseline, default 1)
# =========================================================

GPU=${1:-0}
SEED=${2:-42}
DATASET=${DATASET:-beauty}
SWEEP_TYPE=${SWEEP_TYPE:-lambda}   # lambda | temp
VALUES=${VALUES:-"0 0.001 0.005 0.01 0.05 0.1"}
TASID_STUDENT_MODE=${TASID_STUDENT_MODE:-asymmetric}
LAMBDA_TASID=${LAMBDA_TASID:-0.01}
TASID_TEMP=${TASID_TEMP:-0.1}
RUN_BASELINE=${RUN_BASELINE:-1}

MODEL_NAME="LLMMIRecCAISD"
if [ "${DATASET}" = "beauty" ]; then
  LR=0.004; LAMBDA_INTEREST_SEMANTIC=0.01
elif [ "${DATASET}" = "ml-1m" ]; then
  LR=0.001; LAMBDA_INTEREST_SEMANTIC=0.005
else
  echo "ERROR: DATASET must be beauty or ml-1m"; exit 1
fi
BATCH_SIZE=1024
EVAL_BATCH_SIZE=256

ROOT_LOG_DIR="new_log/llmmirec_caisd_tasid_sweep/${DATASET}"
ROOT_MODEL_DIR="new_model/llmmirec_caisd_tasid_sweep/${DATASET}"
SUMMARY_FILE="${ROOT_LOG_DIR}/summary_${SWEEP_TYPE}.tsv"
mkdir -p "${ROOT_LOG_DIR}" "${ROOT_MODEL_DIR}"

if [ ! -f "${SUMMARY_FILE}" ]; then
  echo -e "dataset\tmodel\tseed\tsweep_type\tvalue\ttasid_mode\ttasid_student_mode\tlambda_tasid\ttasid_temp\tlr\tbatch_size\tstart_time\tend_time\ttotal_seconds\tparameter_count\tbest_epoch\tbest_dev\ttest_after_training\tstatus" > "${SUMMARY_FILE}"
fi

run_one() {
  local VALUE=$1
  local TASID_MODE=$2
  local LAM=$3
  local TEMP=$4
  local LABEL=$5

  local LOG_DIR="${ROOT_LOG_DIR}/${LABEL}/seed${SEED}"
  local MODEL_DIR="${ROOT_MODEL_DIR}/${LABEL}/seed${SEED}"
  mkdir -p "${LOG_DIR}" "${MODEL_DIR}"

  local LOG_FILE="${LOG_DIR}/LLMMIRecCAISD_seed${SEED}.log"
  local OUT_FILE="${LOG_DIR}/LLMMIRecCAISD_seed${SEED}.out"
  local MODEL_PATH="${MODEL_DIR}/LLMMIRecCAISD_seed${SEED}.pt"

  local START_TIME END_TIME START_TS END_TS TOTAL_SECONDS
  START_TIME=$(date "+%Y-%m-%d %H:%M:%S"); START_TS=$(date +%s)
  echo "[START] ${START_TIME} | ${DATASET} seed=${SEED} ${LABEL} gpu=${GPU}"

  python main.py \
    --model_name "${MODEL_NAME}" \
    --dataset "${DATASET}" \
    --path ./data/ \
    --gpu "${GPU}" \
    --random_seed "${SEED}" \
    --emb_size 64 --attn_size 64 --K 4 --history_max 20 \
    --item_encoder aspcf \
    --llm_emb_path "./data/${DATASET}/handled/llm_table_pca1536.pkl" \
    --semantic_rank 512 --semantic_dim 32 --semantic_hidden 128 \
    --complement_dim 32 --tail_hidden 64 --complement_hidden 64 --gate_hidden 64 \
    --aspcf_gate_mode basic \
    --lambda_relation 0.01 --relation_sample_size 128 \
    --relation_teacher_temp 0.1 --relation_student_temp 0.1 \
    --semantic_teacher_path "./data/${DATASET}/handled/llmmi_proto32_sr512.pkl" \
    --semantic_distill_mode uniform \
    --semantic_teacher_mode responsibility \
    --semantic_relation_mode none \
    --lambda_interest_semantic "${LAMBDA_INTEREST_SEMANTIC}" \
    --tasid_mode "${TASID_MODE}" \
    --tasid_student_mode "${TASID_STUDENT_MODE}" \
    --lambda_tasid "${LAM}" \
    --tasid_temp "${TEMP}" \
    --dropout 0.1 \
    --lr "${LR}" --l2 1e-6 \
    --batch_size "${BATCH_SIZE}" --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --num_neg 1 --epoch 200 --early_stop 10 \
    --topk 5,10,20,50 --metric NDCG,HR \
    --num_workers 5 \
    --log_file "${LOG_FILE}" --model_path "${MODEL_PATH}" \
    2>&1 | tee "${OUT_FILE}"

  END_TIME=$(date "+%Y-%m-%d %H:%M:%S"); END_TS=$(date +%s)
  TOTAL_SECONDS=$((END_TS - START_TS))

  local BEST_DEV TEST_AFTER BEST_EPOCH PARAM_COUNT STATUS
  BEST_DEV=$(grep "Best Iter(dev)" "${LOG_FILE}" | tail -n 1 | tr '\t' ' ' | sed 's/[[:space:]]\+/ /g')
  TEST_AFTER=$(grep "Test After Training" "${LOG_FILE}" | tail -n 1 | tr '\t' ' ' | sed 's/[[:space:]]\+/ /g')
  BEST_EPOCH=$(echo "${BEST_DEV}" | grep -oP 'Best Iter\(dev\)=\s*\K[0-9]+' || echo "NA")
  PARAM_COUNT=$(grep "#params:" "${LOG_FILE}" | tail -n 1 | grep -oP '[0-9]+' || echo "NA")
  STATUS="OK"
  if grep -E "(loss=nan|dev=\(HR@5:nan|dev=\(HR@5:0.0000)" "${LOG_FILE}" | grep -v "check" > /dev/null 2>&1; then STATUS="NaN_CRASH"; fi

  echo "[DONE] ${START_TIME} -> ${END_TIME} (${TOTAL_SECONDS}s)"
  echo "${BEST_DEV}"
  echo "${TEST_AFTER}"
  echo -e "${DATASET}\t${MODEL_NAME}\t${SEED}\t${SWEEP_TYPE}\t${VALUE}\t${TASID_MODE}\t${TASID_STUDENT_MODE}\t${LAM}\t${TEMP}\t${LR}\t${BATCH_SIZE}\t${START_TIME}\t${END_TIME}\t${TOTAL_SECONDS}\t${PARAM_COUNT}\t${BEST_EPOCH}\t${BEST_DEV}\t${TEST_AFTER}\t${STATUS}" >> "${SUMMARY_FILE}"
}

# Baseline: TASID off
if [ "${RUN_BASELINE}" = "1" ]; then
  run_one "none" "none" "0.0" "0.1" "baseline"
fi

# Sweep values
for V in ${VALUES}; do
  if [ "${SWEEP_TYPE}" = "lambda" ]; then
    run_one "${V}" "target" "${V}" "${TASID_TEMP}" "tasid_lam${V}"
  else
    run_one "${V}" "target" "${LAMBDA_TASID}" "${V}" "tasid_temp${V}"
  fi
done

echo "Summary: ${SUMMARY_FILE}"
