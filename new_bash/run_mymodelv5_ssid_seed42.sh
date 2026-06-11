#!/usr/bin/env bash
set -e

GPU=${1:-0}
SEED=42

ROOT_LOG_DIR="new_log/mymodelv5_ssid"
ROOT_MODEL_DIR="new_model/mymodelv5_ssid"

mkdir -p "${ROOT_LOG_DIR}/beauty"
mkdir -p "${ROOT_LOG_DIR}/ml-1m"
mkdir -p "${ROOT_MODEL_DIR}/beauty"
mkdir -p "${ROOT_MODEL_DIR}/ml-1m"

SUMMARY_FILE="${ROOT_LOG_DIR}/summary_seed${SEED}.tsv"
echo -e "dataset\tvariant\tseed\tstart_time\tend_time\ttotal_seconds\tbest_dev\ttest_after_training" > "${SUMMARY_FILE}"

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

run_one() {
  local DATASET=$1
  local VARIANT=$2
  local K=$3
  local GAMMA=$4
  local TAU=$5
  local LAMBDA_IPD=$6
  local IPD_MARGIN=$7
  local LR=$8
  local LLM_PATH=$9
  local SRS_PATH=${10}
  local USE_SSID=${11}
  local LAMBDA_SSID=${12}
  local SSID_TEMP=${13}

  local LOG_FILE="${ROOT_LOG_DIR}/${DATASET}/${VARIANT}.log"
  local OUT_FILE="${ROOT_LOG_DIR}/${DATASET}/${VARIANT}.out"
  local MODEL_PATH="${ROOT_MODEL_DIR}/${DATASET}/${VARIANT}.pt"

  local START_TIME
  local END_TIME
  local START_TS
  local END_TS
  local TOTAL_SECONDS

  START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
  START_TS=$(date +%s)

  echo ""
  echo "========================================================="
  echo "[START] ${DATASET} | ${VARIANT}"
  echo "GPU=${GPU}"
  echo "time=${START_TIME}"
  echo "use_ssid=${USE_SSID}"
  echo "lambda_ssid=${LAMBDA_SSID}"
  echo "ssid_temp=${SSID_TEMP}"
  echo "log=${LOG_FILE}"
  echo "========================================================="

  python main.py \
    --model_name MyModelV5 \
    --dataset "${DATASET}" \
    --path ./data/ \
    --gpu "${GPU}" \
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
    --use_dspc 0 \
    --use_ssid "${USE_SSID}" \
    --lambda_ssid "${LAMBDA_SSID}" \
    --ssid_temp "${SSID_TEMP}" \
    --ssid_warmup_steps 5000 \
    --ssid_detach_sem 1 \
    --ssid_detach_attn 1 \
    --ssid_use_proto_norm 1 \
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

  END_TIME=$(date "+%Y-%m-%d %H:%M:%S")
  END_TS=$(date +%s)
  TOTAL_SECONDS=$((END_TS - START_TS))

  BEST_DEV=$(grep "Best Iter(dev)" "${LOG_FILE}" | tail -n 1 | tr '\t' ' ' | sed 's/[[:space:]]\+/ /g')
  TEST_AFTER=$(grep "Test After Training" "${LOG_FILE}" | tail -n 1 | tr '\t' ' ' | sed 's/[[:space:]]\+/ /g')

  echo "========================================================="
  echo "[DONE] ${DATASET} | ${VARIANT}"
  echo "${BEST_DEV}"
  echo "${TEST_AFTER}"
  echo "========================================================="

  echo -e "${DATASET}\t${VARIANT}\t${SEED}\t${START_TIME}\t${END_TIME}\t${TOTAL_SECONDS}\t${BEST_DEV}\t${TEST_AFTER}" >> "${SUMMARY_FILE}"
}

# =========================================================
# Beauty 1: no SSID
# =========================================================
run_one \
  "beauty" \
  "MyModelV5_noSSID_seed42" \
  4 \
  0.1 \
  0.2 \
  0.05 \
  0.2 \
  0.002 \
  "${BEAUTY_LLM}" \
  "${BEAUTY_SRS}" \
  0 \
  0.0 \
  0.2

# =========================================================
# Beauty 2: SSID lambda=0.001
# =========================================================
run_one \
  "beauty" \
  "MyModelV5_SSID_lam0001_seed42" \
  4 \
  0.1 \
  0.2 \
  0.05 \
  0.2 \
  0.002 \
  "${BEAUTY_LLM}" \
  "${BEAUTY_SRS}" \
  1 \
  0.001 \
  0.2

# =========================================================
# Beauty 3: SSID lambda=0.003
# =========================================================
run_one \
  "beauty" \
  "MyModelV5_SSID_lam0003_seed42" \
  4 \
  0.1 \
  0.2 \
  0.05 \
  0.2 \
  0.002 \
  "${BEAUTY_LLM}" \
  "${BEAUTY_SRS}" \
  1 \
  0.003 \
  0.2

# =========================================================
# ML-1M 1: no SSID
# =========================================================
run_one \
  "ml-1m" \
  "MyModelV5_noSSID_seed42" \
  2 \
  0.08 \
  0.3 \
  0.02 \
  0.10 \
  0.001 \
  "${ML1M_LLM}" \
  "${ML1M_SRS}" \
  0 \
  0.0 \
  0.2

# =========================================================
# ML-1M 2: SSID lambda=0.0005
# =========================================================
run_one \
  "ml-1m" \
  "MyModelV5_SSID_lam00005_seed42" \
  2 \
  0.08 \
  0.3 \
  0.02 \
  0.10 \
  0.001 \
  "${ML1M_LLM}" \
  "${ML1M_SRS}" \
  1 \
  0.0005 \
  0.2

# =========================================================
# ML-1M 3: SSID lambda=0.001
# =========================================================
run_one \
  "ml-1m" \
  "MyModelV5_SSID_lam0001_seed42" \
  2 \
  0.08 \
  0.3 \
  0.02 \
  0.10 \
  0.001 \
  "${ML1M_LLM}" \
  "${ML1M_SRS}" \
  1 \
  0.001 \
  0.2

echo ""
echo "========================================================="
echo "All MyModelV5 SSID runs finished."
echo "Summary:"
echo "cat ${SUMMARY_FILE}"
echo "========================================================="