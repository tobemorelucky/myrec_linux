#!/usr/bin/env bash
set -e

# =========================================================
# LLMMIRec Phase 0 — ML-1M seed=42 baseline
#
# Usage:
#   bash new_bash/run_llmmirec_phase0_ml1m_seed42.sh [GPU_ID]
#   SMOKE=1 bash new_bash/run_llmmirec_phase0_ml1m_seed42.sh [GPU_ID]
# =========================================================

GPU=${1:-0}
SEED=42
DATASET="ml-1m"

ROOT_LOG_DIR="new_log/llmmirec_phase0/${DATASET}"
ROOT_MODEL_DIR="new_model/llmmirec_phase0/${DATASET}"

mkdir -p "${ROOT_LOG_DIR}"
mkdir -p "${ROOT_MODEL_DIR}"

SUMMARY_FILE="${ROOT_LOG_DIR}/summary_seed${SEED}.tsv"

# ---- smoke mode ----
if [ "${SMOKE}" = "1" ]; then
  EPOCH=2
  EARLY_STOP=0
  NUM_WORKERS=0
  SMOKE_TAG="[SMOKE]"
else
  EPOCH=200
  EARLY_STOP=10
  NUM_WORKERS=5
  SMOKE_TAG=""
fi

echo -e "dataset\tmodel\titem_encoder\tadapter_activation\tadapter_use_ln\tadapter_hidden\tgamma_init\tgamma_trainable\tseed\tK\tlr\tstart_time\tend_time\ttotal_seconds\tparameter_count\tbest_epoch\tbest_dev\ttest_after_training\tstatus" > "${SUMMARY_FILE}"

echo "========================================================="
echo "LLMMIRec Phase 0 — ${DATASET} seed=${SEED} ${SMOKE_TAG}"
echo "GPU=${GPU}  EPOCH=${EPOCH}  EARLY_STOP=${EARLY_STOP}"
echo "SUMMARY=${SUMMARY_FILE}"
echo "========================================================="


run_llmmirec() {
  local ITEM_ENCODER=$1
  local LLM_EMB_PATH=$2
  local GAMMA_INIT=$3
  local GAMMA_TRAINABLE=$4
  local MODEL_LABEL=$5

  local LOG_DIR="${ROOT_LOG_DIR}/${MODEL_LABEL}"
  local MODEL_DIR="${ROOT_MODEL_DIR}/${MODEL_LABEL}"

  mkdir -p "${LOG_DIR}"
  mkdir -p "${MODEL_DIR}"

  local LOG_FILE="${LOG_DIR}/LLMMIRec_${MODEL_LABEL}_seed${SEED}.log"
  local OUT_FILE="${LOG_DIR}/LLMMIRec_${MODEL_LABEL}_seed${SEED}.out"
  local MODEL_PATH="${MODEL_DIR}/LLMMIRec_${MODEL_LABEL}_seed${SEED}.pt"

  local START_TIME
  local END_TIME
  local START_TS
  local END_TS
  local TOTAL_SECONDS

  START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
  START_TS=$(date +%s)

  echo ""
  echo "========================================================="
  echo "[START] item_encoder=${ITEM_ENCODER}, seed=${SEED} ${SMOKE_TAG}"
  echo "time=${START_TIME}"
  echo "log=${LOG_FILE}"
  echo "model=${MODEL_PATH}"
  echo "========================================================="

  python main.py \
    --model_name LLMMIRec \
    --dataset "${DATASET}" \
    --path ./data/ \
    --gpu "${GPU}" \
    --random_seed "${SEED}" \
    --emb_size 64 \
    --attn_size 64 \
    --K 4 \
    --history_max 20 \
    --item_encoder "${ITEM_ENCODER}" \
    --llm_emb_path "${LLM_EMB_PATH}" \
    --adapter_hidden 256 \
    --adapter_activation gelu \
    --adapter_use_ln 0 \
    --gamma_init "${GAMMA_INIT}" \
    --gamma_trainable "${GAMMA_TRAINABLE}" \
    --dropout 0.1 \
    --lr 0.001 \
    --l2 1e-6 \
    --batch_size 256 \
    --eval_batch_size 256 \
    --num_neg 1 \
    --epoch "${EPOCH}" \
    --early_stop "${EARLY_STOP}" \
    --topk 5,10,20,50 \
    --metric NDCG,HR \
    --num_workers "${NUM_WORKERS}" \
    --log_file "${LOG_FILE}" \
    --model_path "${MODEL_PATH}" \
    2>&1 | tee "${OUT_FILE}"

  END_TIME=$(date "+%Y-%m-%d %H:%M:%S")
  END_TS=$(date +%s)
  TOTAL_SECONDS=$((END_TS - START_TS))

  local BEST_DEV
  local TEST_AFTER
  local BEST_EPOCH
  local PARAM_COUNT
  local STATUS="OK"

  BEST_DEV=$(grep "Best Iter(dev)" "${LOG_FILE}" | tail -n 1 | tr '\t' ' ' | sed 's/[[:space:]]\+/ /g')
  TEST_AFTER=$(grep "Test After Training" "${LOG_FILE}" | tail -n 1 | tr '\t' ' ' | sed 's/[[:space:]]\+/ /g')
  BEST_EPOCH=$(echo "${BEST_DEV}" | grep -oP 'Best Iter\(dev\)=\s*\K[0-9]+' || echo "NA")
  PARAM_COUNT=$(grep "#params:" "${LOG_FILE}" | tail -n 1 | grep -oP '[0-9]+' || echo "NA")

  # Only flag real NaN in loss/dev values (not in diagnostic messages)
  if grep -E "(loss=nan|dev=\(HR@5:nan|dev=\(HR@5:0.0000)" "${LOG_FILE}" | grep -v "check" > /dev/null 2>&1; then
    STATUS="NaN_CRASH"
  fi

  echo "========================================================="
  echo "[DONE] item_encoder=${ITEM_ENCODER}, seed=${SEED} ${SMOKE_TAG}"
  echo "start=${START_TIME}  end=${END_TIME}  total_seconds=${TOTAL_SECONDS}"
  echo "${BEST_DEV}"
  echo "${TEST_AFTER}"
  echo "========================================================="

  echo -e "${DATASET}\tLLMMIRec\t${ITEM_ENCODER}\tgelu\t0\t256\t${GAMMA_INIT}\t${GAMMA_TRAINABLE}\t${SEED}\t4\t0.001\t${START_TIME}\t${END_TIME}\t${TOTAL_SECONDS}\t${PARAM_COUNT}\t${BEST_EPOCH}\t${BEST_DEV}\t${TEST_AFTER}\t${STATUS}" >> "${SUMMARY_FILE}"
}


# =========================================================
# 1. id mode
# =========================================================
run_llmmirec "id" "" "0.1" "0" "id"

# =========================================================
# 2. llm_replace mode
# =========================================================
run_llmmirec "llm_replace" "./data/ml-1m/handled/llm_table_pca1536.pkl" "0.1" "0" "llm_replace"

# =========================================================
# 3. residual mode
# =========================================================
run_llmmirec "residual" "./data/ml-1m/handled/llm_table_pca1536.pkl" "0.1" "0" "residual"


echo ""
echo "========================================================="
echo "All LLMMIRec Phase 0 ML-1M experiments finished."
echo "Summary: ${SUMMARY_FILE}"
echo "========================================================="
