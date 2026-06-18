#!/usr/bin/env bash
set -e

GPU=${1:-0}

ROOT_LOG_DIR="new_log/clean_llm_ipd_3datasets_seeds01242"
ROOT_MODEL_DIR="new_model/clean_llm_ipd_3datasets_seeds01242"

mkdir -p "${ROOT_LOG_DIR}/beauty"
mkdir -p "${ROOT_LOG_DIR}/ml-1m"
mkdir -p "${ROOT_LOG_DIR}/toys"
mkdir -p "${ROOT_MODEL_DIR}/beauty"
mkdir -p "${ROOT_MODEL_DIR}/ml-1m"
mkdir -p "${ROOT_MODEL_DIR}/toys"

SUMMARY_FILE="${ROOT_LOG_DIR}/summary.tsv"
echo -e "dataset\tmodel\tvariant\tseed\tstart_time\tend_time\ttotal_seconds\tbest_dev\ttest_after_training" > "${SUMMARY_FILE}"

# =========================================================
# Data paths
# =========================================================

BEAUTY_LLM="./data/beauty/handled/llm_table_pca1536.pkl"
BEAUTY_SRS="./data/beauty/handled/itm_emb_pomrec.pkl"

ML1M_LLM="./data/ml-1m/handled/llm_table_pca1536.pkl"
ML1M_SRS="./data/ml-1m/handled/itm_emb_pomrec.pkl"

TOYS_LLM="./data/toys/handled/llm_table_pca1536.pkl"
TOYS_SRS="./data/toys/handled/itm_emb_pomrec.pkl"

# =========================================================
# Checkpoint paths
# =========================================================

BEAUTY_INIT_FALLBACK="./model/PoMRec/PoMRec__beauty__42__lr=0.002__l2=1e-06.pt"

# ML-1M README: fixed seed=1 checkpoint
ML1M_INIT="./model/PoMRec/PoMRec__ml-1m__1__lr=0.001__l2=1e-06.pt"

# Toys README: fixed seed=42 checkpoint
TOYS_INIT="./model/PoMRec/toys__42__lr=0.001__l2=1e-06__lamb=3.8__history_max=20.pt"

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
check_required_file "${TOYS_LLM}"
check_required_file "${TOYS_SRS}"
check_required_file "${ML1M_INIT}"
check_required_file "${TOYS_INIT}"

get_beauty_init() {
  local SEED=$1
  local CKPT="./model/PoMRec/PoMRec__beauty__${SEED}__lr=0.002__l2=1e-06.pt"

  if [ -f "${CKPT}" ]; then
    echo "${CKPT}"
  else
    echo "${BEAUTY_INIT_FALLBACK}"
  fi
}

run_one() {
  local DATASET=$1
  local MODEL_NAME=$2
  local VARIANT=$3
  local SEED=$4
  local LR=$5
  local GAMMA=$6
  local TAU=$7
  local LAMB=$8
  local LLM_PATH=$9
  local SRS_PATH=${10}
  local INIT_CKPT=${11}
  local USE_IPD=${12}
  local LAMBDA_IPD=${13}
  local IPD_MARGIN=${14}
  local EMILE_WARMUP=${15}

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
  echo "seed=${SEED}"
  echo "init_ckpt=${INIT_CKPT}"
  echo "use_ipd=${USE_IPD}"
  echo "lambda_ipd=${LAMBDA_IPD}"
  echo "ipd_margin=${IPD_MARGIN}"
  echo "emile_warmup=${EMILE_WARMUP}"
  echo "log=${LOG_FILE}"
  echo "========================================================="

  if [ "${USE_IPD}" -eq 1 ]; then
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
      --lamb "${LAMB}" \
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
  else
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
      --lamb "${LAMB}" \
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
  fi

  END_TIME=$(date "+%Y-%m-%d %H:%M:%S")
  END_TS=$(date +%s)
  TOTAL_SECONDS=$((END_TS - START_TS))

  BEST_DEV=$(grep "Best Iter(dev)" "${LOG_FILE}" | tail -n 1 | tr '\t' ' ' | sed 's/[[:space:]]\+/ /g' || true)
  TEST_AFTER=$(grep "Test After Training" "${LOG_FILE}" | tail -n 1 | tr '\t' ' ' | sed 's/[[:space:]]\+/ /g' || true)

  echo "========================================================="
  echo "[DONE] ${DATASET} | ${MODEL_NAME} | ${VARIANT}"
  echo "${BEST_DEV}"
  echo "${TEST_AFTER}"
  echo "========================================================="

  echo -e "${DATASET}\t${MODEL_NAME}\t${VARIANT}\t${SEED}\t${START_TIME}\t${END_TIME}\t${TOTAL_SECONDS}\t${BEST_DEV}\t${TEST_AFTER}" >> "${SUMMARY_FILE}"
}

run_pair() {
  local DATASET=$1
  local SEED=$2
  local LR=$3
  local GAMMA=$4
  local TAU=$5
  local LAMB=$6
  local LLM_PATH=$7
  local SRS_PATH=$8
  local INIT_CKPT=$9
  local LAMBDA_IPD=${10}
  local IPD_MARGIN=${11}
  local EMILE_WARMUP=${12}

  run_one \
    "${DATASET}" \
    "MyModelLLM" \
    "${DATASET}_LLM_only_seed${SEED}" \
    "${SEED}" \
    "${LR}" \
    "${GAMMA}" \
    "${TAU}" \
    "${LAMB}" \
    "${LLM_PATH}" \
    "${SRS_PATH}" \
    "${INIT_CKPT}" \
    0 \
    0.0 \
    "${IPD_MARGIN}" \
    "${EMILE_WARMUP}"

  run_one \
    "${DATASET}" \
    "MyModelLLMIPD" \
    "${DATASET}_LLM_IPD_seed${SEED}" \
    "${SEED}" \
    "${LR}" \
    "${GAMMA}" \
    "${TAU}" \
    "${LAMB}" \
    "${LLM_PATH}" \
    "${SRS_PATH}" \
    "${INIT_CKPT}" \
    1 \
    "${LAMBDA_IPD}" \
    "${IPD_MARGIN}" \
    "${EMILE_WARMUP}"
}

echo ""
echo "========================================================="
echo "Start clean LLM / LLM+IPD runs."
echo "GPU=${GPU}"
echo "Summary: ${SUMMARY_FILE}"
echo "========================================================="

# =========================================================
# Beauty seeds 0,1,2
# seed42 already exists in corrected_8groups, skip here.
# Best config:
# lr=0.002, gamma=0.1, tau=0.2, lamb=3.0
# IPD: lambda=0.05, margin=0.2, warmup=5000
# Checkpoint: seed-matched, fallback seed42
# =========================================================

for SEED in 0 1 2; do
  BEAUTY_INIT=$(get_beauty_init "${SEED}")
  echo "[Beauty] seed=${SEED}, init=${BEAUTY_INIT}"

  run_pair \
    "beauty" \
    "${SEED}" \
    0.002 \
    0.1 \
    0.2 \
    3.0 \
    "${BEAUTY_LLM}" \
    "${BEAUTY_SRS}" \
    "${BEAUTY_INIT}" \
    0.05 \
    0.2 \
    5000
done

# =========================================================
# ML-1M seeds 0,1,2
# seed42 already exists in corrected_8groups, skip here.
# Best config:
# lr=0.001, gamma=0.08, tau=0.3, lamb=3.0
# IPD: lambda=0.02, margin=0.10, warmup=20000
# Checkpoint: fixed seed1 checkpoint by README
# =========================================================

for SEED in 0 1 2; do
  run_pair \
    "ml-1m" \
    "${SEED}" \
    0.001 \
    0.08 \
    0.3 \
    3.0 \
    "${ML1M_LLM}" \
    "${ML1M_SRS}" \
    "${ML1M_INIT}" \
    0.02 \
    0.10 \
    20000
done

# =========================================================
# Toys seeds 0,1,2,42
# Best config remembered:
# lr=0.001, gamma=0.05, tau=0.5, lamb=3.8
# IPD: lambda=0.05, margin=0.10, warmup=20000
# Checkpoint: fixed toys seed42 checkpoint by README
# =========================================================

for SEED in 0 1 2 42; do
  run_pair \
    "toys" \
    "${SEED}" \
    0.001 \
    0.05 \
    0.5 \
    3.8 \
    "${TOYS_LLM}" \
    "${TOYS_SRS}" \
    "${TOYS_INIT}" \
    0.05 \
    0.10 \
    20000
done

echo ""
echo "========================================================="
echo "All clean LLM / LLM+IPD runs finished."
echo "Summary:"
echo "cat ${SUMMARY_FILE}"
echo "========================================================="