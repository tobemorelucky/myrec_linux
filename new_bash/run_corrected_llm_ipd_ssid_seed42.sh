#!/usr/bin/env bash
set -e

GPU=${1:-0}
SEED=42

ROOT_LOG_DIR="new_log/corrected_llm_ipd_ssid"
ROOT_MODEL_DIR="new_model/corrected_llm_ipd_ssid"

mkdir -p "${ROOT_LOG_DIR}/beauty"
mkdir -p "${ROOT_LOG_DIR}/ml-1m"
mkdir -p "${ROOT_MODEL_DIR}/beauty"
mkdir -p "${ROOT_MODEL_DIR}/ml-1m"

SUMMARY_FILE="${ROOT_LOG_DIR}/summary_seed${SEED}.tsv"
echo -e "dataset\tmodel\tvariant\tseed\tstart_time\tend_time\ttotal_seconds\tbest_dev\ttest_after_training" > "${SUMMARY_FILE}"

BEAUTY_LLM="./data/beauty/handled/llm_table_pca1536.pkl"
BEAUTY_SRS="./data/beauty/handled/itm_emb_pomrec.pkl"
BEAUTY_INIT="./model/PoMRec/PoMRec__beauty__${SEED}__lr=0.002__l2=1e-06.pt"
BEAUTY_INIT_FALLBACK="./model/PoMRec/PoMRec__beauty__42__lr=0.002__l2=1e-06.pt"

ML1M_LLM="./data/ml-1m/handled/llm_table_pca1536.pkl"
ML1M_SRS="./data/ml-1m/handled/itm_emb_pomrec.pkl"
ML1M_INIT="./model/PoMRec/PoMRec__ml-1m__1__lr=0.001__l2=1e-06.pt"

check_file() {
  if [ ! -f "$1" ]; then
    echo "[WARN] File not found: $1"
  else
    echo "[OK] $1"
  fi
}

check_file "${BEAUTY_LLM}"
check_file "${BEAUTY_SRS}"
check_file "${ML1M_LLM}"
check_file "${ML1M_SRS}"

if [ ! -f "${BEAUTY_INIT}" ]; then
  echo "[WARN] Beauty init ckpt not found for seed=${SEED}, fallback to ${BEAUTY_INIT_FALLBACK}"
  BEAUTY_INIT="${BEAUTY_INIT_FALLBACK}"
fi

run_one() {
  local DATASET=$1
  local MODEL_NAME=$2
  local VARIANT=$3
  local LR=$4
  local GAMMA=$5
  local TAU=$6
  local LLM_PATH=$7
  local SRS_PATH=$8
  local INIT_CKPT=$9
  local USE_EMILE=${10}
  local LAMBDA_IPD=${11}
  local IPD_MARGIN=${12}
  local EMILE_WARMUP=${13}
  local USE_SSID=${14}
  local LAMBDA_SSID=${15}
  local SSID_TEMP=${16}

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
  echo "[START] ${DATASET} | ${MODEL_NAME} | ${VARIANT}"
  echo "GPU=${GPU}"
  echo "init_ckpt=${INIT_CKPT}"
  echo "use_emile=${USE_EMILE}"
  echo "lambda_ipd=${LAMBDA_IPD}"
  echo "emile_warmup=${EMILE_WARMUP}"
  echo "use_ssid=${USE_SSID}"
  echo "lambda_ssid=${LAMBDA_SSID}"
  echo "log=${LOG_FILE}"
  echo "========================================================="

  python main.py \
    --model_name "${MODEL_NAME}" \
    --dataset "${DATASET}" \
    --path ./data/ \
    --gpu "${GPU}" \
    --random_seed "${SEED}" \
    --load 0 \
    --emb_size 64 \
    --attn_size 8 \
    --K 3 \
    --prompt_num 4 \
    --n_layers 1 \
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
    --init_ckpt "${INIT_CKPT}" \
    --init_strict 0 \
    --use_emile "${USE_EMILE}" \
    --lambda_ipd "${LAMBDA_IPD}" \
    --ipd_margin "${IPD_MARGIN}" \
    --emile_use_fused_itememb 0 \
    --emile_warmup_steps "${EMILE_WARMUP}" \
    --use_dspc 0 \
    --use_ssid "${USE_SSID}" \
    --lambda_ssid "${LAMBDA_SSID}" \
    --ssid_temp "${SSID_TEMP}" \
    --ssid_warmup_steps 5000 \
    --ssid_detach_sem 1 \
    --ssid_detach_attn 1 \
    --ssid_use_proto_norm 1 \
    --lr "${LR}" \
    --l2 1e-6 \
    --batch_size 256 \
    --eval_batch_size 256 \
    --num_neg 1 \
    --dropout 0 \
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
  echo "[DONE] ${DATASET} | ${MODEL_NAME} | ${VARIANT}"
  echo "${BEST_DEV}"
  echo "${TEST_AFTER}"
  echo "========================================================="

  echo -e "${DATASET}\t${MODEL_NAME}\t${VARIANT}\t${SEED}\t${START_TIME}\t${END_TIME}\t${TOTAL_SECONDS}\t${BEST_DEV}\t${TEST_AFTER}" >> "${SUMMARY_FILE}"
}

# =========================================================
# Beauty: best config from final script
# lr=0.002, gamma=0.1, tau=0.2, IPD warmup=5000
# =========================================================

run_one \
  "beauty" \
  "MyModelLLM" \
  "beauty_clean_LLM_seed42" \
  0.002 \
  0.1 \
  0.2 \
  "${BEAUTY_LLM}" \
  "${BEAUTY_SRS}" \
  "${BEAUTY_INIT}" \
  0 \
  0.0 \
  0.2 \
  5000 \
  0 \
  0.0 \
  0.2

run_one \
  "beauty" \
  "MyModelLLMIPD" \
  "beauty_clean_LLM_IPD_seed42" \
  0.002 \
  0.1 \
  0.2 \
  "${BEAUTY_LLM}" \
  "${BEAUTY_SRS}" \
  "${BEAUTY_INIT}" \
  1 \
  0.05 \
  0.2 \
  5000 \
  0 \
  0.0 \
  0.2

run_one \
  "beauty" \
  "MyModelV5" \
  "beauty_clean_LLM_IPD_SSID_lam0001_seed42" \
  0.002 \
  0.1 \
  0.2 \
  "${BEAUTY_LLM}" \
  "${BEAUTY_SRS}" \
  "${BEAUTY_INIT}" \
  1 \
  0.05 \
  0.2 \
  5000 \
  1 \
  0.001 \
  0.2

# =========================================================
# ML-1M: best config from final script
# lr=0.001, gamma=0.08, tau=0.3, IPD warmup=20000
# =========================================================

run_one \
  "ml-1m" \
  "MyModelLLM" \
  "ml1m_clean_LLM_seed42" \
  0.001 \
  0.08 \
  0.3 \
  "${ML1M_LLM}" \
  "${ML1M_SRS}" \
  "${ML1M_INIT}" \
  0 \
  0.0 \
  0.10 \
  20000 \
  0 \
  0.0 \
  0.2

run_one \
  "ml-1m" \
  "MyModelLLMIPD" \
  "ml1m_clean_LLM_IPD_seed42" \
  0.001 \
  0.08 \
  0.3 \
  "${ML1M_LLM}" \
  "${ML1M_SRS}" \
  "${ML1M_INIT}" \
  1 \
  0.02 \
  0.10 \
  20000 \
  0 \
  0.0 \
  0.2

run_one \
  "ml-1m" \
  "MyModelV5" \
  "ml1m_clean_LLM_IPD_SSID_lam00005_seed42" \
  0.001 \
  0.08 \
  0.3 \
  "${ML1M_LLM}" \
  "${ML1M_SRS}" \
  "${ML1M_INIT}" \
  1 \
  0.02 \
  0.10 \
  20000 \
  1 \
  0.0005 \
  0.2

echo ""
echo "========================================================="
echo "Corrected LLM/IPD/SSID ablation finished."
echo "Summary:"
echo "cat ${SUMMARY_FILE}"
echo "========================================================="