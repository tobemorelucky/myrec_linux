#!/usr/bin/env bash
set -e

GPU=${1:-0}
SEED=42

ROOT_LOG_DIR="new_log/tirl_latest"
ROOT_MODEL_DIR="new_model/tirl_latest"

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
ML1M_INIT="./model/PoMRec/PoMRec__ml-1m__${SEED}__lr=0.001__l2=1e-06.pt"
ML1M_INIT_FALLBACK="./model/PoMRec/PoMRec__ml-1m__1__lr=0.001__l2=1e-06.pt"

check_required_file() {
  if [ ! -f "$1" ]; then
    echo "[ERROR] Required file not found: $1"
    exit 1
  fi
}

check_required_file "${BEAUTY_LLM}"
check_required_file "${BEAUTY_SRS}"
check_required_file "${ML1M_LLM}"
check_required_file "${ML1M_SRS}"

if [ ! -f "${BEAUTY_INIT}" ]; then
  echo "[WARN] Beauty init ckpt not found: ${BEAUTY_INIT}"
  echo "[WARN] Fallback to: ${BEAUTY_INIT_FALLBACK}"
  BEAUTY_INIT="${BEAUTY_INIT_FALLBACK}"
fi
check_required_file "${BEAUTY_INIT}"

if [ ! -f "${ML1M_INIT}" ]; then
  echo "[WARN] ML-1M init ckpt not found: ${ML1M_INIT}"
  echo "[WARN] Fallback to: ${ML1M_INIT_FALLBACK}"
  ML1M_INIT="${ML1M_INIT_FALLBACK}"
fi
check_required_file "${ML1M_INIT}"

run_one() {
  local DATASET=$1
  local VARIANT=$2
  local LR=$3
  local GAMMA=$4
  local TAU=$5
  local LLM_PATH=$6
  local SRS_PATH=$7
  local INIT_CKPT=$8
  local LAMBDA_TIRL=$9
  local TIRL_WARMUP=${10}
  local LAMBDA_IPD=${11}
  local IPD_MARGIN=${12}
  local EMILE_WARMUP=${13}

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
  echo "[START] ${DATASET} | MyModelTIRL | ${VARIANT}"
  echo "GPU=${GPU}"
  echo "init_ckpt=${INIT_CKPT}"
  echo "lambda_tirl=${LAMBDA_TIRL}"
  echo "tirl_warmup=${TIRL_WARMUP}"
  echo "log=${LOG_FILE}"
  echo "========================================================="

  python main.py \
    --model_name MyModelTIRL \
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
    --use_emile 1 \
    --lambda_ipd "${LAMBDA_IPD}" \
    --ipd_margin "${IPD_MARGIN}" \
    --emile_use_fused_itememb 0 \
    --emile_warmup_steps "${EMILE_WARMUP}" \
    --use_tirl 1 \
    --lambda_tirl "${LAMBDA_TIRL}" \
    --tirl_warmup_steps "${TIRL_WARMUP}" \
    --tirl_mode selected \
    --tirl_detach_route 1 \
    --tirl_neg_reduce mean \
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
  echo "[DONE] ${DATASET} | MyModelTIRL | ${VARIANT}"
  echo "${BEST_DEV}"
  echo "${TEST_AFTER}"
  echo "========================================================="

  echo -e "${DATASET}\tMyModelTIRL\t${VARIANT}\t${SEED}\t${START_TIME}\t${END_TIME}\t${TOTAL_SECONDS}\t${BEST_DEV}\t${TEST_AFTER}" >> "${SUMMARY_FILE}"
}

run_one \
  "beauty" \
  "beauty_TIRL_lam002_seed42" \
  0.002 \
  0.1 \
  0.2 \
  "${BEAUTY_LLM}" \
  "${BEAUTY_SRS}" \
  "${BEAUTY_INIT}" \
  0.02 \
  5000 \
  0.05 \
  0.2 \
  5000

run_one \
  "ml-1m" \
  "ml1m_TIRL_lam001_seed42" \
  0.001 \
  0.08 \
  0.3 \
  "${ML1M_LLM}" \
  "${ML1M_SRS}" \
  "${ML1M_INIT}" \
  0.01 \
  20000 \
  0.02 \
  0.10 \
  20000

echo ""
echo "========================================================="
echo "Latest TIRL runs finished."
echo "Summary:"
echo "cat ${SUMMARY_FILE}"
echo "========================================================="