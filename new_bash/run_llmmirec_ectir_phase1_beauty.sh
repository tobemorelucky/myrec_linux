#!/usr/bin/env bash
# =========================================================
# THESIS PHASE 1 — Chapter 4 Round 1: ECTIR structural test (Beauty, seed 42)
#
# Evidence-Constrained Transport Interest Routing.
#   ECTIR-1    transport routing only                    (Module 1)
#   ECTIR-2    transport + iterative refinement          (Module 1 + 2)
#   ECTIR-Full transport + refinement + evidence aggr.   (Module 1 + 2 + 3)
#
# LOSS = BPR + Chapter 3 existing relation loss (lambda_relation=0.01).
# NO Chapter 4 auxiliary loss.
#
# Training hyperparameters are inherited verbatim from the frozen stable config
# in new_bash/run_llmmirec_aspcf_phase2_beauty.sh (lr=0.004, batch=1024, K=4).
# ASPCF ItemEncoder is used unchanged.
#
# Usage:
#   CONFIG=ectir1 bash new_bash/run_llmmirec_ectir_phase1_beauty.sh GPU SEED
#
# CONFIG (required): ectir1 | ectir2 | full | baseline
#   baseline = sanity run; must reproduce ASPCF exactly.
#
# SMOKE=1 -> 2 epochs, early_stop 0, num_workers 0
# =========================================================
set -euo pipefail

GPU=${1:-0}
SEED=${2:-42}
DATASET=${3:-beauty}
CONFIG=${CONFIG:-}
SMOKE=${SMOKE:-0}

MODEL_NAME="LLMMIRecECTIR"
case "${DATASET}" in
  beauty) LR=0.004 ;;
  ml-1m)  LR=0.001 ;;
  *) echo "ERROR: DATASET must be beauty | ml-1m"; exit 1 ;;
esac

case "${CONFIG}" in
  baseline)      ECTIR_MODE=baseline;         N_REFINE=0 ;;
  ectir1)        ECTIR_MODE=transport;        N_REFINE=0 ;;
  ectir2)        ECTIR_MODE=transport_refine; N_REFINE=1 ;;
  full)          ECTIR_MODE=full;             N_REFINE=1 ;;
  *) echo "ERROR: CONFIG must be baseline | ectir1 | ectir2 | full"; exit 1 ;;
esac

# transport defaults (single sensible set; NOT swept in this round)
EPS=0.1
TAU_C=1.0
N_SINKHORN=5
EVI_HIDDEN=64

BATCH_SIZE=1024
EVAL_BATCH_SIZE=256

if [ "${SMOKE}" = "1" ]; then
  EPOCH=2; EARLY_STOP=0; NUM_WORKERS=0; LABEL="smoke_${CONFIG}"
else
  EPOCH=200; EARLY_STOP=10; NUM_WORKERS=5; LABEL="${CONFIG}"
fi

if ! python -c "import torch" 2>/dev/null; then
  echo "ERROR: current python has no torch. Run: conda activate hzg_py10"; exit 1
fi

LLM_PATH="./data/${DATASET}/handled/llm_table_pca1536.pkl"
[ -f "${LLM_PATH}" ] || { echo "ERROR: missing ${LLM_PATH}"; exit 1; }

ROOT_LOG_DIR="new_log/llmmirec_ectir/${DATASET}/${LABEL}/seed${SEED}"
ROOT_MODEL_DIR="new_model/llmmirec_ectir/${DATASET}/${LABEL}/seed${SEED}"
SUMMARY_FILE="new_log/llmmirec_ectir/${DATASET}/summary.tsv"
mkdir -p "${ROOT_LOG_DIR}" "${ROOT_MODEL_DIR}" "$(dirname "${SUMMARY_FILE}")"

if [ ! -f "${SUMMARY_FILE}" ]; then
  echo -e "dataset\tmodel\tseed\tconfig\tectir_mode\tn_refine\teps\ttau_c\tn_sinkhorn\tK\tlr\tbatch_size\tstart_time\tend_time\ttotal_seconds\tparameter_count\tbest_epoch\tbest_dev\ttest_after_training\tstatus" > "${SUMMARY_FILE}"
fi

LOG_FILE="${ROOT_LOG_DIR}/LLMMIRecECTIR_seed${SEED}.log"
OUT_FILE="${ROOT_LOG_DIR}/LLMMIRecECTIR_seed${SEED}.out"
MODEL_PATH="${ROOT_MODEL_DIR}/LLMMIRecECTIR_seed${SEED}.pt"

START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
START_TS=$(date +%s)
echo "[START] ${START_TIME} | ECTIR ${DATASET} ${LABEL} mode=${ECTIR_MODE} n_refine=${N_REFINE} lr=${LR} seed=${SEED} gpu=${GPU}"

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
  --ectir_mode "${ECTIR_MODE}" \
  --ectir_eps "${EPS}" \
  --ectir_tau_c "${TAU_C}" \
  --ectir_n_sinkhorn "${N_SINKHORN}" \
  --ectir_n_refine "${N_REFINE}" \
  --ectir_evi_hidden "${EVI_HIDDEN}" \
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

echo -e "${DATASET}\t${MODEL_NAME}\t${SEED}\t${LABEL}\t${ECTIR_MODE}\t${N_REFINE}\t${EPS}\t${TAU_C}\t${N_SINKHORN}\t4\t${LR}\t${BATCH_SIZE}\t${START_TIME}\t${END_TIME}\t${TOTAL_SECONDS}\t${PARAM_COUNT}\t${BEST_EPOCH}\t${BEST_DEV}\t${TEST_AFTER}\t${STATUS}" >> "${SUMMARY_FILE}"
