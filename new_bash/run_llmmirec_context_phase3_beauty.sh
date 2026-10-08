#!/usr/bin/env bash
# =========================================================
# THESIS Chapter 4 — Phase 3: History Contextualization Axis Test (Beauty, seed 42)
#
# STRUCTURAL CONTROL EXPERIMENT. Not the Chapter 4 final method.
#
# Question: before the multi-interest extractor, does letting history items
# contextualize EACH OTHER beat the current per-item independent encoding?
#
# Configurations (CONFIG env var):
#   ffn_control   per-position MLP, NO cross-position mixing   (capacity control)
#   self_attn     1-layer Pre-LN self-attention over history
#
# ASPCF baseline is NOT re-run (already exists: 0.1592 / 0.1088).
#
# HARD CONSTRAINTS: only the tensor fed into the extractor changes.
# ItemEncoder / QueryMultiInterestExtractor / InterestAggregator are imported
# unmodified. LOSS = BPR + 0.01 * Chapter3 relation loss. No Chapter 4 loss.
#
# Every other flag is copied VERBATIM from run_llmmirec_aspcf_phase2_beauty.sh.
# lr is NOT changed because of the added layers.
#
# Usage: CONFIG=self_attn bash new_bash/run_llmmirec_context_phase3_beauty.sh GPU SEED
# =========================================================
set -euo pipefail

GPU=${1:-0}
SEED=${2:-42}
DATASET=${3:-beauty}
CONFIG=${CONFIG:-}
SMOKE=${SMOKE:-0}

MODEL_NAME="LLMMIRecContextControl"
LR=0.004
BATCH_SIZE=1024
EVAL_BATCH_SIZE=256

case "${CONFIG}" in
  ffn_control) CTX_MODE=ffn_control ;;
  self_attn)   CTX_MODE=self_attn   ;;
  baseline)    CTX_MODE=baseline    ;;
  *) echo "ERROR: CONFIG must be ffn_control | self_attn | baseline"; exit 1 ;;
esac

LABEL="${CONFIG}"
if [ "${SMOKE}" = "1" ]; then
  EPOCH=2; EARLY_STOP=0; NUM_WORKERS=0; LABEL="smoke_${CONFIG}"
else
  EPOCH=200; EARLY_STOP=10; NUM_WORKERS=5
fi

if ! python -c "import torch" 2>/dev/null; then
  echo "ERROR: current python has no torch. Run: conda activate hzg_py10"; exit 1
fi

LLM_PATH="./data/${DATASET}/handled/llm_table_pca1536.pkl"
[ -f "${LLM_PATH}" ] || { echo "ERROR: missing ${LLM_PATH}"; exit 1; }

ROOT_LOG_DIR="new_log/llmmirec_context/${DATASET}/${LABEL}/seed${SEED}"
ROOT_MODEL_DIR="new_model/llmmirec_context/${DATASET}/${LABEL}/seed${SEED}"
SUMMARY_FILE="new_log/llmmirec_context/${DATASET}/summary.tsv"
mkdir -p "${ROOT_LOG_DIR}" "${ROOT_MODEL_DIR}" "$(dirname "${SUMMARY_FILE}")"

if [ ! -f "${SUMMARY_FILE}" ]; then
  echo -e "dataset\tmodel\tseed\tconfig\tcontext_mode\tcontext_heads\tcontext_ffn_hidden\tcontext_dropout\tK\tlr\tbatch_size\tstart_time\tend_time\ttotal_seconds\tparameter_count\tbest_epoch\tbest_dev\ttest_after_training\tstatus" > "${SUMMARY_FILE}"
fi

LOG_FILE="${ROOT_LOG_DIR}/LLMMIRecContextControl_seed${SEED}.log"
OUT_FILE="${ROOT_LOG_DIR}/LLMMIRecContextControl_seed${SEED}.out"
MODEL_PATH="${ROOT_MODEL_DIR}/LLMMIRecContextControl_seed${SEED}.pt"

START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
START_TS=$(date +%s)
echo "[START] ${START_TIME} | CtxCtl ${DATASET} ${LABEL} mode=${CTX_MODE} lr=${LR} seed=${SEED} gpu=${GPU}"

set +e
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
  --context_mode "${CTX_MODE}" \
  --context_dropout 0.1 \
  --context_heads 4 \
  --context_ffn_hidden 128 \
  --context_ffn_control_hidden 256 \
  --dropout 0.1 \
  --lr "${LR}" \
  --l2 1e-6 \
  --batch_size "${BATCH_SIZE}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --num_neg 1 \
  --epoch "${EPOCH}" \
  --early_stop "${EARLY_STOP}" \
  --topk 5,10,20,50 \
  --metric NDCG,HR \
  --num_workers "${NUM_WORKERS}" \
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

echo -e "${DATASET}\t${MODEL_NAME}\t${SEED}\t${LABEL}\t${CTX_MODE}\t4\t128\t0.1\t4\t${LR}\t${BATCH_SIZE}\t${START_TIME}\t${END_TIME}\t${TOTAL_SECONDS}\t${PARAM_COUNT}\t${BEST_EPOCH}\t${BEST_DEV}\t${TEST_AFTER}\t${STATUS}" >> "${SUMMARY_FILE}"
