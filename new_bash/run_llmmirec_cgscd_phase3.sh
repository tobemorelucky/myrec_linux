#!/usr/bin/env bash
# =========================================================
# LLMMIRec CGSCD — Chapter 3 Architecture-First round
#
# Representation architecture test: data-driven shared/complementary split
# of the LLM view, guided by cross-view association with the collaborative
# item embedding. NO new auxiliary losses.
#
# Usage:
#   DATASET=beauty METHOD=crosscov USE_RELATION=0 \
#     bash new_bash/run_llmmirec_cgscd_phase3.sh 0 42
#
# Env overrides:
#   DATASET        beauty | ml-1m | toys      (default beauty)
#   METHOD         crosscov | cca             (default crosscov)
#   USE_RELATION   0 = PURE_BPR, 1 = + existing ASPCF relation loss (default 0)
#   RANK           shared subspace rank       (default 32)
#   SMOKE          1 = 2 epochs, early_stop 0 (default 0)
#
# Writes:
#   new_log/llmmirec_cgscd/<dataset>/<label>/seed<seed>/LLMMIRecCGSCD_seed<seed>.{log,out}
#   new_model/llmmirec_cgscd/<dataset>/<label>/seed<seed>/LLMMIRecCGSCD_seed<seed>.pt
#   new_log/llmmirec_cgscd/<dataset>/summary.tsv
# =========================================================
set -euo pipefail

GPU=${1:-0}
SEED=${2:-42}
DATASET=${DATASET:-beauty}
METHOD=${METHOD:-crosscov}
USE_RELATION=${USE_RELATION:-0}
RANK=${RANK:-32}
SMOKE=${SMOKE:-0}

MODEL_NAME="LLMMIRecCGSCD"

case "${DATASET}" in
  beauty) LR=0.004 ;;
  ml-1m)  LR=0.001 ;;
  toys)   LR=0.001 ;;
  *) echo "ERROR: DATASET must be beauty | ml-1m | toys"; exit 1 ;;
esac
BATCH_SIZE=1024
EVAL_BATCH_SIZE=256

if [ "${USE_RELATION}" = "1" ]; then
  LAMBDA_RELATION=0.01
else
  LAMBDA_RELATION=0.0
fi
LABEL="${METHOD}_rel${USE_RELATION}"

if [ "${SMOKE}" = "1" ]; then
  EPOCH=2; EARLY_STOP=0; NUM_WORKERS=0
else
  EPOCH=200; EARLY_STOP=10; NUM_WORKERS=5
fi

# ---- fail fast if the interpreter lacks torch (a past source of phantom "OK" rows) ----
if ! python -c "import torch" 2>/dev/null; then
  echo "ERROR: current python has no torch. Activate the env: conda activate hzg_py10"
  exit 1
fi

LLM_PATH="./data/${DATASET}/handled/llm_table_pca1536.pkl"
CF_PATH="./data/${DATASET}/handled/itm_emb_pomrec.pkl"
BASIS_PATH="./data/${DATASET}/handled/cgscd_basis_${METHOD}_r${RANK}.pkl"

for f in "${LLM_PATH}" "${CF_PATH}" "${BASIS_PATH}"; do
  if [ ! -f "${f}" ]; then
    echo "ERROR: missing asset ${f}"
    echo "  build the basis first:"
    echo "  python tools/build_cgscd_basis.py --dataset ${DATASET} --method ${METHOD} --rank ${RANK}"
    exit 1
  fi
done

ROOT_LOG_DIR="new_log/llmmirec_cgscd/${DATASET}/${LABEL}/seed${SEED}"
ROOT_MODEL_DIR="new_model/llmmirec_cgscd/${DATASET}/${LABEL}/seed${SEED}"
SUMMARY_FILE="new_log/llmmirec_cgscd/${DATASET}/summary.tsv"
mkdir -p "${ROOT_LOG_DIR}" "${ROOT_MODEL_DIR}" "$(dirname "${SUMMARY_FILE}")"

if [ ! -f "${SUMMARY_FILE}" ]; then
  echo -e "dataset\tmodel\tseed\tmethod\tuse_relation\trank\tlr\tbatch_size\tstart_time\tend_time\ttotal_seconds\tparameter_count\tbest_epoch\tbest_dev\ttest_after_training\tstatus" > "${SUMMARY_FILE}"
fi

LOG_FILE="${ROOT_LOG_DIR}/LLMMIRecCGSCD_seed${SEED}.log"
OUT_FILE="${ROOT_LOG_DIR}/LLMMIRecCGSCD_seed${SEED}.out"
MODEL_PATH="${ROOT_MODEL_DIR}/LLMMIRecCGSCD_seed${SEED}.pt"

START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
START_TS=$(date +%s)
echo "[START] ${START_TIME} | CGSCD ${DATASET} ${LABEL} rank=${RANK} lr=${LR} seed=${SEED} gpu=${GPU}"

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
  --item_encoder cgscd \
  --llm_emb_path "${LLM_PATH}" \
  --cgscd_basis_path "${BASIS_PATH}" \
  --cgscd_cf_path "${CF_PATH}" \
  --cgscd_shared_dim 32 \
  --cgscd_shared_hidden 128 \
  --cgscd_compl_dim 32 \
  --cgscd_compl_hidden 64 \
  --cgscd_gate_mode basic \
  --lambda_relation "${LAMBDA_RELATION}" \
  --relation_sample_size 128 \
  --relation_teacher_temp 0.1 \
  --relation_student_temp 0.1 \
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
  echo "ERROR: python exited ${PY_RC}; see ${OUT_FILE}"
  tail -n 20 "${OUT_FILE}" || true
  exit ${PY_RC}
fi

END_TIME=$(date "+%Y-%m-%d %H:%M:%S")
END_TS=$(date +%s)
TOTAL_SECONDS=$((END_TS - START_TS))

BEST_DEV=$(grep "Best Iter(dev)" "${LOG_FILE}" | tail -n 1 | tr '\t' ' ' | sed 's/[[:space:]]\+/ /g' || true)
TEST_AFTER=$(grep "Test After Training" "${LOG_FILE}" | tail -n 1 | tr '\t' ' ' | sed 's/[[:space:]]\+/ /g' || true)
BEST_EPOCH=$(echo "${BEST_DEV}" | grep -oP 'Best Iter\(dev\)=\s*\K[0-9]+' || echo "NA")
PARAM_COUNT=$(grep "#params:" "${LOG_FILE}" | tail -n 1 | grep -oP '[0-9]+' || echo "NA")

STATUS="OK"
if [ -z "${TEST_AFTER}" ]; then
  STATUS="NO_RESULT"
elif grep -qE "(loss=nan|dev=\(HR@5:nan|dev=\(HR@5:0\.0000)" "${LOG_FILE}"; then
  STATUS="NaN_CRASH"
fi

echo "[DONE] ${START_TIME} -> ${END_TIME} (${TOTAL_SECONDS}s)"
echo "${BEST_DEV}"
echo "${TEST_AFTER}"

echo -e "${DATASET}\t${MODEL_NAME}\t${SEED}\t${METHOD}\t${USE_RELATION}\t${RANK}\t${LR}\t${BATCH_SIZE}\t${START_TIME}\t${END_TIME}\t${TOTAL_SECONDS}\t${PARAM_COUNT}\t${BEST_EPOCH}\t${BEST_DEV}\t${TEST_AFTER}\t${STATUS}" >> "${SUMMARY_FILE}"
