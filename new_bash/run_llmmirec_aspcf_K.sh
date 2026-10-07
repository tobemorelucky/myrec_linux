#!/usr/bin/env bash
# =========================================================
# ASPCF with a dataset-specific number of interests (fair-tuning baseline).
#
# WHY: PoMRec's standard config already tunes K per dataset (beauty K=4,
# ml-1m K=2). To compare fairly, ASPCF must be allowed the same K selection.
# This is ordinary hyper-parameter tuning, NOT a method contribution, and
# Adaptive Interest Cardinality / dynamic-K is explicitly NOT a Chapter 4
# innovation.
#
# Everything except --K is copied verbatim from the frozen stable config in
# new_bash/run_llmmirec_aspcf_phase2_{beauty,ml1m}.sh, including the
# per-dataset lr (beauty 0.004, ml-1m 0.001).
#
# K=4 results already exist and are NOT re-run:
#   new_log/llmmirec_aspcf_phase2/<ds>/seed42/   (5 seeds)
#
# Usage:
#   K=2 bash new_bash/run_llmmirec_aspcf_K.sh GPU DATASET SEED
#   e.g. K=2 bash new_bash/run_llmmirec_aspcf_K.sh 0 beauty 42
#
# Writes:
#   new_log/llmmirec_aspcf_K{K}/<dataset>/seed<seed>/LLMMIRecASPCF_seed<seed>.{log,out}
#   new_log/llmmirec_aspcf_K{K}/<dataset>/summary.tsv
#   new_model/llmmirec_aspcf_K{K}/<dataset>/seed<seed>/LLMMIRecASPCF_seed<seed>.pt
# =========================================================
set -euo pipefail

GPU=${1:-0}
DATASET=${2:-beauty}
SEED=${3:-42}
K=${K:-2}

case "${DATASET}" in
  beauty) LR=0.004 ;;
  ml-1m)  LR=0.001 ;;
  toys)   LR=0.001 ;;
  *) echo "ERROR: DATASET must be beauty | ml-1m | toys"; exit 1 ;;
esac

MODEL_NAME="LLMMIRecASPCF"
BATCH_SIZE=1024
EVAL_BATCH_SIZE=256

ROOT_LOG_DIR="new_log/llmmirec_aspcf_K${K}/${DATASET}/seed${SEED}"
ROOT_MODEL_DIR="new_model/llmmirec_aspcf_K${K}/${DATASET}/seed${SEED}"
SUMMARY_FILE="new_log/llmmirec_aspcf_K${K}/${DATASET}/summary.tsv"
mkdir -p "${ROOT_LOG_DIR}" "${ROOT_MODEL_DIR}" "$(dirname "${SUMMARY_FILE}")"

if [ ! -f "${SUMMARY_FILE}" ]; then
  echo -e "dataset\tmodel\tseed\tK\tlr\tbatch_size\tstart_time\tend_time\ttotal_seconds\tparameter_count\tbest_epoch\tbest_dev\ttest_after_training\tstatus" > "${SUMMARY_FILE}"
fi

LOG_FILE="${ROOT_LOG_DIR}/LLMMIRecASPCF_seed${SEED}.log"
OUT_FILE="${ROOT_LOG_DIR}/LLMMIRecASPCF_seed${SEED}.out"
MODEL_PATH="${ROOT_MODEL_DIR}/LLMMIRecASPCF_seed${SEED}.pt"

LLM_PATH="./data/${DATASET}/handled/llm_table_pca1536.pkl"

if ! python -c "import torch" 2>/dev/null; then
  echo "ERROR: current python has no torch. Run: conda activate hzg_py10"; exit 1
fi
[ -f "${LLM_PATH}" ] || { echo "ERROR: missing ${LLM_PATH}"; exit 1; }

START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
START_TS=$(date +%s)
echo "[START] ${START_TIME} | ASPCF ${DATASET} K=${K} lr=${LR} seed=${SEED} gpu=${GPU}"

set +e
python main.py \
  --model_name "${MODEL_NAME}" \
  --dataset "${DATASET}" \
  --path ./data/ \
  --gpu "${GPU}" \
  --random_seed "${SEED}" \
  --emb_size 64 \
  --attn_size 64 \
  --K "${K}" \
  --history_max 20 \
  --item_encoder aspcf \
  --llm_emb_path "${LLM_PATH}" \
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
  > "${OUT_FILE}" 2>&1
PY_RC=$?
set -e

if [ ${PY_RC} -ne 0 ]; then
  echo "ERROR: python exited ${PY_RC}; see ${OUT_FILE}"; tail -n 20 "${OUT_FILE}" || true; exit ${PY_RC}
fi

END_TIME=$(date "+%Y-%m-%d %H:%M:%S")
END_TS=$(date +%s)
TOTAL_SECONDS=$((END_TS - START_TS))

BEST_DEV=$(grep "Best Iter(dev)" "${LOG_FILE}" | tail -n 1 | tr '\t' ' ' | sed 's/[[:space:]]\+/ /g' || true)
TEST_AFTER=$(grep "Test After Training" "${LOG_FILE}" | tail -n 1 | tr '\t' ' ' | sed 's/[[:space:]]\+/ /g' || true)
BEST_EPOCH=$(echo "${BEST_DEV}" | grep -oP 'Best Iter\(dev\)=\s*\K[0-9]+' || echo "NA")
PARAM_COUNT=$(grep "#params:" "${LOG_FILE}" | tail -n 1 | grep -oP '[0-9]+' || echo "NA")

STATUS="OK"
if [ -z "${TEST_AFTER}" ]; then STATUS="NO_RESULT"
elif grep -qE "(loss=nan|dev=\(HR@5:nan|dev=\(HR@5:0\.0000)" "${LOG_FILE}"; then STATUS="NaN_CRASH"; fi

echo "[DONE] ${START_TIME} -> ${END_TIME} (${TOTAL_SECONDS}s)"
echo "${BEST_DEV}"
echo "${TEST_AFTER}"

echo -e "${DATASET}\t${MODEL_NAME}\t${SEED}\t${K}\t${LR}\t${BATCH_SIZE}\t${START_TIME}\t${END_TIME}\t${TOTAL_SECONDS}\t${PARAM_COUNT}\t${BEST_EPOCH}\t${BEST_DEV}\t${TEST_AFTER}\t${STATUS}" >> "${SUMMARY_FILE}"
