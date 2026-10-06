#!/usr/bin/env bash
# =========================================================
# THESIS PHASE 1 — Chapter 3 Round 2: Reliability-Aware Semantic
# Residual Fusion (RASRF), structural test on Beauty seed 42.
#
#   e_final = e_cf + gate(consistency) * e_sem
#
# LOSS = plain BPR. No alignment / relation / contrastive / orthogonal /
# entropy auxiliary loss in this round.
#
# Training hyperparameters are copied from the stable config in
# new_bash/run_llmmirec_aspcf_phase2_beauty.sh (lr=0.004, batch=1024, ...).
# Verified against the `Arguments | Values` table in the actual logs.
#
# Usage:
#   CONFIG=rasrf_scalar bash new_bash/run_llmmirec_rasrf_phase1_beauty.sh GPU SEED
#
# CONFIG (required):
#   rasrf_scalar        item-specific SCALAR gate, no neighbourhood prior
#   rasrf_vector        item-specific VECTOR gate, no neighbourhood prior
#   rasrf_vector_neigh  item-specific VECTOR gate + neighbourhood prior
#   residual_gamma      CONTROL: global learnable gamma (no item-specific gate)
#   rasrf_scalar_neigh  scalar gate + neighbourhood prior
#
# SMOKE=1 -> 2 epochs, early_stop 0, num_workers 0 (for a quick sanity run)
# =========================================================
set -euo pipefail

GPU=${1:-0}
SEED=${2:-42}
DATASET=${3:-beauty}
CONFIG=${CONFIG:-}
SMOKE=${SMOKE:-0}

MODEL_NAME="LLMMIRecRASRF"
case "${DATASET}" in
  beauty) LR=0.004 ;;
  ml-1m)  LR=0.001 ;;
  *) echo "ERROR: DATASET must be beauty | ml-1m"; exit 1 ;;
esac

case "${CONFIG}" in
  rasrf_scalar)
    ENCODER=rasrf; GATE_MODE=scalar; NEIGH_PATH="" ;;
  rasrf_scalar_neigh)
    ENCODER=rasrf; GATE_MODE=scalar
    NEIGH_PATH="./data/${DATASET}/handled/semantic_neighborhood_agreement_k20.pkl" ;;
  rasrf_vector)
    ENCODER=rasrf; GATE_MODE=vector; NEIGH_PATH="" ;;
  rasrf_vector_neigh)
    ENCODER=rasrf; GATE_MODE=vector
    NEIGH_PATH="./data/${DATASET}/handled/semantic_neighborhood_agreement_k20.pkl" ;;
  residual_gamma)
    ENCODER=residual; GATE_MODE=scalar; NEIGH_PATH="" ;;
  *)
    echo "ERROR: CONFIG must be one of: rasrf_scalar | rasrf_scalar_neigh |"
    echo "       rasrf_vector | rasrf_vector_neigh | residual_gamma"
    exit 1 ;;
esac

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
if [ -n "${NEIGH_PATH}" ] && [ ! -f "${NEIGH_PATH}" ]; then
  echo "ERROR: missing neighbourhood prior ${NEIGH_PATH}"
  echo "  build it: python tools/build_semantic_neighborhood_agreement.py --dataset ${DATASET} --k 20"
  exit 1
fi

ROOT_LOG_DIR="new_log/llmmirec_rasrf/${DATASET}/${LABEL}/seed${SEED}"
ROOT_MODEL_DIR="new_model/llmmirec_rasrf/${DATASET}/${LABEL}/seed${SEED}"
SUMMARY_FILE="new_log/llmmirec_rasrf/${DATASET}/summary.tsv"
mkdir -p "${ROOT_LOG_DIR}" "${ROOT_MODEL_DIR}" "$(dirname "${SUMMARY_FILE}")"

if [ ! -f "${SUMMARY_FILE}" ]; then
  echo -e "dataset\tmodel\tseed\tconfig\tencoder\tgate_mode\tneigh\tlr\tbatch_size\tstart_time\tend_time\ttotal_seconds\tparameter_count\tbest_epoch\tbest_dev\ttest_after_training\tstatus" > "${SUMMARY_FILE}"
fi

LOG_FILE="${ROOT_LOG_DIR}/LLMMIRecRASRF_seed${SEED}.log"
OUT_FILE="${ROOT_LOG_DIR}/LLMMIRecRASRF_seed${SEED}.out"
MODEL_PATH="${ROOT_MODEL_DIR}/LLMMIRecRASRF_seed${SEED}.pt"

START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
START_TS=$(date +%s)
echo "[START] ${START_TIME} | RASRF ${DATASET} ${LABEL} enc=${ENCODER} gate=${GATE_MODE} neighbour=$([ -n "${NEIGH_PATH}" ] && echo yes || echo no) seed=${SEED} gpu=${GPU}"

# RASRF-only flags are passed conditionally: the residual control uses the
# same model file but has no gate, so it must not receive --rasrf_* options.
RASRF_ARGS=()
if [ "${ENCODER}" = "rasrf" ]; then
  RASRF_ARGS+=(--rasrf_gate_mode "${GATE_MODE}")
  RASRF_ARGS+=(--rasrf_gate_input agree_diff_inter)
  RASRF_ARGS+=(--rasrf_gate_hidden 64)
  # RASRF_ZERO_INIT=1 makes the semantic correction start at exactly 0
  # (pure-CF at init). Default 0 = repo-standard init.
  RASRF_ARGS+=(--rasrf_zero_init_correction "${RASRF_ZERO_INIT:-0}")
  if [ -n "${NEIGH_PATH}" ]; then
    RASRF_ARGS+=(--rasrf_neigh_path "${NEIGH_PATH}")
  fi
fi

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
  --item_encoder "${ENCODER}" \
  --llm_emb_path "${LLM_PATH}" \
  --adapter_hidden 256 \
  --adapter_activation gelu \
  --adapter_use_ln 0 \
  "${RASRF_ARGS[@]}" \
  --gamma_init 0.1 \
  --gamma_trainable 1 \
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

NEIGH_FLAG=$([ -n "${NEIGH_PATH}" ] && echo "yes" || echo "no")
echo -e "${DATASET}\t${MODEL_NAME}\t${SEED}\t${LABEL}\t${ENCODER}\t${GATE_MODE}\t${NEIGH_FLAG}\t${LR}\t${BATCH_SIZE}\t${START_TIME}\t${END_TIME}\t${TOTAL_SECONDS}\t${PARAM_COUNT}\t${BEST_EPOCH}\t${BEST_DEV}\t${TEST_AFTER}\t${STATUS}" >> "${SUMMARY_FILE}"
