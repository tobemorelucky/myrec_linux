#!/usr/bin/env bash
set -Eeuo pipefail

SEED=42

ROOT_LOG_DIR="new_log/mymodelv2_iuscbr"
ROOT_MODEL_DIR="new_model/mymodelv2_iuscbr"

mkdir -p "${ROOT_LOG_DIR}/beauty"
mkdir -p "${ROOT_LOG_DIR}/ml-1m"
mkdir -p "${ROOT_MODEL_DIR}/beauty"
mkdir -p "${ROOT_MODEL_DIR}/ml-1m"

SUMMARY_FILE="${ROOT_LOG_DIR}/summary_seed${SEED}.tsv"
LOCK_FILE="${SUMMARY_FILE}.lock"

echo -e "dataset\tvariant\tgpu\tseed\tstart_time\tend_time\ttotal_seconds\tbest_dev\ttest_after_training\tstatus" > "${SUMMARY_FILE}"

BEAUTY_LLM="./data/beauty/handled/llm_table_pca1536.pkl"
BEAUTY_SRS="./data/beauty/handled/itm_emb_pomrec.pkl"

ML1M_LLM="./data/ml-1m/handled/llm_table_pca1536.pkl"
ML1M_SRS="./data/ml-1m/handled/itm_emb_pomrec.pkl"

check_file() {
  if [ ! -f "$1" ]; then
    echo "[ERROR] File not found: $1"
    exit 1
  fi
}

check_file "${BEAUTY_LLM}"
check_file "${BEAUTY_SRS}"
check_file "${ML1M_LLM}"
check_file "${ML1M_SRS}"

if ! command -v flock >/dev/null 2>&1; then
  echo "[ERROR] flock not found. Please install util-linux, or remove the flock block manually."
  exit 1
fi

append_summary() {
  local DATASET="$1"
  local VARIANT="$2"
  local RUN_GPU="$3"
  local START_TIME="$4"
  local END_TIME="$5"
  local TOTAL_SECONDS="$6"
  local BEST_DEV="$7"
  local TEST_AFTER="$8"
  local STATUS="$9"

  {
    flock -x 200
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${DATASET}" \
      "${VARIANT}" \
      "${RUN_GPU}" \
      "${SEED}" \
      "${START_TIME}" \
      "${END_TIME}" \
      "${TOTAL_SECONDS}" \
      "${BEST_DEV}" \
      "${TEST_AFTER}" \
      "${STATUS}" >> "${SUMMARY_FILE}"
  } 200>"${LOCK_FILE}"
}

run_one() {
  local RUN_GPU=$1
  local DATASET=$2
  local VARIANT=$3
  local K=$4
  local GAMMA=$5
  local TAU=$6
  local LAMBDA_IPD=$7
  local IPD_MARGIN=$8
  local LR=$9
  local LLM_PATH=${10}
  local SRS_PATH=${11}
  local GLOBAL_B=${12}
  local GLOBAL_R=${13}
  local GLOBAL_SEM=${14}

  local LOG_FILE="${ROOT_LOG_DIR}/${DATASET}/${VARIANT}.log"
  local OUT_FILE="${ROOT_LOG_DIR}/${DATASET}/${VARIANT}.out"
  local MODEL_PATH="${ROOT_MODEL_DIR}/${DATASET}/${VARIANT}.pt"

  local START_TIME
  local END_TIME
  local START_TS
  local END_TS
  local TOTAL_SECONDS
  local BEST_DEV
  local TEST_AFTER
  local RUN_STATUS
  local STATUS_TEXT

  START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
  START_TS=$(date +%s)

  echo ""
  echo "========================================================="
  echo "[START] ${DATASET} | ${VARIANT}"
  echo "GPU=${RUN_GPU}"
  echo "time=${START_TIME}"
  echo "log=${LOG_FILE}"
  echo "out=${OUT_FILE}"
  echo "========================================================="

  set +e
  python main.py \
    --model_name MyModelV2 \
    --dataset "${DATASET}" \
    --path ./data/ \
    --gpu "${RUN_GPU}" \
    --random_seed "${SEED}" \
    --emb_size 64 \
    --attn_size 8 \
    --K "${K}" \
    --prompt_num 3 \
    --n_layers 2 \
    --lamb 3.0 \
    --history_max 20 \
    --use_llmemb 1 \
    --llm_fuse 1 \
    --llm_emb_path "${LLM_PATH}" \
    --srs_emb_path "${SRS_PATH}" \
    --gamma_init "${GAMMA}" \
    --gamma_trainable 0 \
    --alpha 0.001 \
    --tau "${TAU}" \
    --rat_alpha_warmup_steps 5000 \
    --use_sicr 1 \
    --sicr_train_only 1 \
    --sicr_global_gate 1 \
    --sicr_global_alpha 8 \
    --sicr_global_b "${GLOBAL_B}" \
    --sicr_global_r "${GLOBAL_R}" \
    --sicr_entropy_adapt 1 \
    --sicr_entropy_min 0.5 \
    --sicr_entropy_gamma 1.0 \
    --sicr_global_sem_weight "${GLOBAL_SEM}" \
    --sicr_beta 0.0 \
    --sicr_sem_weight 0.0 \
    --sicr_intent_weight 0.0 \
    --sicr_center 1 \
    --sicr_residual 0.2 \
    --sicr_warmup_steps 5000 \
    --sicr_detach 1 \
    --use_emile 1 \
    --lambda_ipd "${LAMBDA_IPD}" \
    --ipd_margin "${IPD_MARGIN}" \
    --emile_warmup_steps 5000 \
    --lr "${LR}" \
    --l2 1e-6 \
    --batch_size 256 \
    --eval_batch_size 256 \
    --num_neg 1 \
    --epoch 200 \
    --early_stop 10 \
    --num_workers 5 \
    --log_file "${LOG_FILE}" \
    --model_path "${MODEL_PATH}" \
    2>&1 | tee "${OUT_FILE}"
  RUN_STATUS=${PIPESTATUS[0]}
  set -e

  END_TIME=$(date "+%Y-%m-%d %H:%M:%S")
  END_TS=$(date +%s)
  TOTAL_SECONDS=$((END_TS - START_TS))

  BEST_DEV=$(grep "Best Iter(dev)" "${LOG_FILE}" | tail -n 1 | tr '\t' ' ' | sed 's/[[:space:]]\+/ /g' || true)
  TEST_AFTER=$(grep "Test After Training" "${LOG_FILE}" | tail -n 1 | tr '\t' ' ' | sed 's/[[:space:]]\+/ /g' || true)

  if [ "${RUN_STATUS}" -eq 0 ]; then
    STATUS_TEXT="OK"
  else
    STATUS_TEXT="FAILED_EXIT_${RUN_STATUS}"
  fi

  echo "========================================================="
  echo "[DONE] ${DATASET} | ${VARIANT}"
  echo "GPU=${RUN_GPU}"
  echo "STATUS=${STATUS_TEXT}"
  echo "${BEST_DEV}"
  echo "${TEST_AFTER}"
  echo "========================================================="

  append_summary \
    "${DATASET}" \
    "${VARIANT}" \
    "${RUN_GPU}" \
    "${START_TIME}" \
    "${END_TIME}" \
    "${TOTAL_SECONDS}" \
    "${BEST_DEV}" \
    "${TEST_AFTER}" \
    "${STATUS_TEXT}"

  return "${RUN_STATUS}"
}

PIDS=()
NAMES=()

launch() {
  run_one "$@" &
  PIDS+=("$!")
  NAMES+=("$2 | $3 | gpu=$1")
}

# =========================================================
# GPU 0:
# Beauty 1 + ML-1M 1
# =========================================================

launch \
  0 \
  "beauty" \
  "MyModelV2_IUSCBR_entropy_sem00_r015_seed42" \
  4 \
  0.1 \
  0.2 \
  0.05 \
  0.2 \
  0.002 \
  "${BEAUTY_LLM}" \
  "${BEAUTY_SRS}" \
  0.3 \
  0.15 \
  0.0

launch \
  0 \
  "ml-1m" \
  "MyModelV2_IUSCBR_entropy_sem00_r008_seed42" \
  2 \
  0.08 \
  0.3 \
  0.02 \
  0.10 \
  0.001 \
  "${ML1M_LLM}" \
  "${ML1M_SRS}" \
  0.4 \
  0.08 \
  0.0

# =========================================================
# GPU 1:
# Beauty 2 + ML-1M 2
# =========================================================

launch \
  1 \
  "beauty" \
  "MyModelV2_IUSCBR_entropy_sem02_r015_seed42" \
  4 \
  0.1 \
  0.2 \
  0.05 \
  0.2 \
  0.002 \
  "${BEAUTY_LLM}" \
  "${BEAUTY_SRS}" \
  0.3 \
  0.15 \
  0.2

launch \
  1 \
  "ml-1m" \
  "MyModelV2_IUSCBR_entropy_sem00_r006_seed42" \
  2 \
  0.08 \
  0.3 \
  0.02 \
  0.10 \
  0.001 \
  "${ML1M_LLM}" \
  "${ML1M_SRS}" \
  0.4 \
  0.06 \
  0.0

FAILED=0

for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "[OK] ${NAMES[$i]}"
  else
    echo "[FAILED] ${NAMES[$i]}"
    FAILED=1
  fi
done

echo ""
echo "========================================================="
echo "All IU-SCBR runs finished."
echo "Summary:"
echo "cat ${SUMMARY_FILE}"
echo "========================================================="

exit "${FAILED}"