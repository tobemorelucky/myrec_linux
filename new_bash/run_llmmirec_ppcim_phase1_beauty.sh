#!/usr/bin/env bash
# =========================================================
# THESIS PHASE 1 — Chapter 4 Round 1: PPCIM (Beauty, seed 42)
#
# ONE question this round:
#   is candidate-specific latent-interest scoring more effective than
#   candidate-independent single-user-vector scoring?
#
# Configurations:
#   ppcim_history_marginal    mode=marginal,        prior=history   [MAIN]
#   ppcim_uniform_marginal    mode=marginal,        prior=uniform   [does the prior help?]
#   ppcim_posterior_mean      mode=posterior_mean,  prior=history   [marginal vs weighted mean]
#   baseline                  mode=baseline                         [sanity: == ASPCF]
#
# ROUND 1 HARD CONSTRAINTS: only the scoring path changes. ItemEncoder /
# position encoding / QueryMultiInterestExtractor / InterestAggregator /
# relation loss / BPR / K / dropout are all frozen ASPCF.
# Added trainable parameters: 0.
#
# LOSS = BPR + lambda_relation * Chapter3 relation loss. NO Chapter 4 aux loss.
#
# Hyper-parameters inherited verbatim from run_llmmirec_aspcf_phase2_beauty.sh.
#
# Usage: CONFIG=ppcim_history_marginal bash new_bash/run_llmmirec_ppcim_phase1_beauty.sh GPU SEED
# =========================================================
set -euo pipefail

GPU=${1:-0}
SEED=${2:-42}
DATASET=${3:-beauty}
CONFIG=${CONFIG:-}
SMOKE=${SMOKE:-0}

MODEL_NAME="LLMMIRecPPCIM"
case "${DATASET}" in
  beauty) LR=0.004 ;;
  ml-1m)  LR=0.001 ;;
  *) echo "ERROR: DATASET must be beauty | ml-1m"; exit 1 ;;
esac

case "${CONFIG}" in
  baseline)                  PPCIM_MODE=baseline;       PPCIM_PRIOR=history ;;
  ppcim_history_marginal)    PPCIM_MODE=marginal;       PPCIM_PRIOR=history ;;
  ppcim_uniform_marginal)    PPCIM_MODE=marginal;       PPCIM_PRIOR=uniform ;;
  ppcim_posterior_mean)      PPCIM_MODE=posterior_mean; PPCIM_PRIOR=history ;;
  *) echo "ERROR: CONFIG must be baseline | ppcim_history_marginal |"
     echo "       ppcim_uniform_marginal | ppcim_posterior_mean"; exit 1 ;;
esac

PPCIM_TAU=1.0
BATCH_SIZE=1024
EVAL_BATCH_SIZE=256
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

ROOT_LOG_DIR="new_log/llmmirec_ppcim/${DATASET}/${LABEL}/seed${SEED}"
ROOT_MODEL_DIR="new_model/llmmirec_ppcim/${DATASET}/${LABEL}/seed${SEED}"
SUMMARY_FILE="new_log/llmmirec_ppcim/${DATASET}/summary.tsv"
mkdir -p "${ROOT_LOG_DIR}" "${ROOT_MODEL_DIR}" "$(dirname "${SUMMARY_FILE}")"

if [ ! -f "${SUMMARY_FILE}" ]; then
  echo -e "dataset\tmodel\tseed\tconfig\tppcim_mode\tppcim_prior\tppcim_tau\tK\tlr\tbatch_size\tstart_time\tend_time\ttotal_seconds\tparameter_count\tbest_epoch\tbest_dev\ttest_after_training\tstatus" > "${SUMMARY_FILE}"
fi

LOG_FILE="${ROOT_LOG_DIR}/LLMMIRecPPCIM_seed${SEED}.log"
OUT_FILE="${ROOT_LOG_DIR}/LLMMIRecPPCIM_seed${SEED}.out"
MODEL_PATH="${ROOT_MODEL_DIR}/LLMMIRecPPCIM_seed${SEED}.pt"

START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
START_TS=$(date +%s)
echo "[START] ${START_TIME} | PPCIM ${DATASET} ${LABEL} mode=${PPCIM_MODE} prior=${PPCIM_PRIOR} tau=${PPCIM_TAU} lr=${LR} seed=${SEED} gpu=${GPU}"

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
  --ppcim_mode "${PPCIM_MODE}" \
  --ppcim_prior "${PPCIM_PRIOR}" \
  --ppcim_tau "${PPCIM_TAU}" \
  --ppcim_eps 1e-8 \
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

echo -e "${DATASET}\t${MODEL_NAME}\t${SEED}\t${LABEL}\t${PPCIM_MODE}\t${PPCIM_PRIOR}\t${PPCIM_TAU}\t4\t${LR}\t${BATCH_SIZE}\t${START_TIME}\t${END_TIME}\t${TOTAL_SECONDS}\t${PARAM_COUNT}\t${BEST_EPOCH}\t${BEST_DEV}\t${TEST_AFTER}\t${STATUS}" >> "${SUMMARY_FILE}"
