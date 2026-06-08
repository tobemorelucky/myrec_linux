#!/usr/bin/env bash
set -e

# =========================================================
# Sequential MyModelV2 SICR centered + residual experiments
#
# Usage:
#   bash new_bash/run_mymodelv2_sicr_center_residual_seed42.sh 0
#
# Recommended nohup:
#   nohup bash new_bash/run_mymodelv2_sicr_center_residual_seed42.sh 0 \
#     > new_log/mymodelv2_sicr_center/master_seed42.out 2>&1 &
#
# Args:
#   $1: GPU id, default 0
# =========================================================

GPU=${1:-0}
SEED=42

ROOT_LOG_DIR="new_log/mymodelv2_sicr_center"
ROOT_MODEL_DIR="new_model/mymodelv2_sicr_center"

mkdir -p "${ROOT_LOG_DIR}"
mkdir -p "${ROOT_MODEL_DIR}"
mkdir -p "${ROOT_LOG_DIR}/beauty"
mkdir -p "${ROOT_LOG_DIR}/ml-1m"
mkdir -p "${ROOT_MODEL_DIR}/beauty"
mkdir -p "${ROOT_MODEL_DIR}/ml-1m"

SUMMARY_FILE="${ROOT_LOG_DIR}/summary_seed${SEED}.tsv"

echo -e "dataset\tvariant\tseed\tstart_time\tend_time\ttotal_seconds\tbest_dev\ttest_after_training" > "${SUMMARY_FILE}"

echo "========================================================="
echo "Run MyModelV2 SICR centered + residual experiments"
echo "GPU=${GPU}"
echo "SEED=${SEED}"
echo "SUMMARY=${SUMMARY_FILE}"
echo "========================================================="


run_one() {
  local DATASET=$1
  local VARIANT=$2
  local RESIDUAL=$3

  local EMB_SIZE=$4
  local ATTN_SIZE=$5
  local K=$6
  local PROMPT_NUM=$7
  local N_LAYERS=$8
  local LAMB=$9
  local HISTORY_MAX=${10}

  local LLM_PATH=${11}
  local SRS_PATH=${12}
  local GAMMA_INIT=${13}
  local ALPHA=${14}
  local TAU=${15}

  local SICR_BETA=${16}
  local SICR_SEM_WEIGHT=${17}
  local SICR_INTENT_WEIGHT=${18}

  local LAMBDA_IPD=${19}
  local IPD_MARGIN=${20}

  local LR=${21}
  local L2=${22}
  local BATCH_SIZE=${23}
  local EVAL_BATCH_SIZE=${24}
  local EPOCH=${25}
  local EARLY_STOP=${26}
  local NUM_WORKERS=${27}

  local LOG_DIR="${ROOT_LOG_DIR}/${DATASET}"
  local MODEL_DIR="${ROOT_MODEL_DIR}/${DATASET}"

  mkdir -p "${LOG_DIR}"
  mkdir -p "${MODEL_DIR}"

  local LOG_FILE="${LOG_DIR}/${VARIANT}.log"
  local OUT_FILE="${LOG_DIR}/${VARIANT}.out"
  local MODEL_PATH="${MODEL_DIR}/${VARIANT}.pt"

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
  echo "time=${START_TIME}"
  echo "gpu=${GPU}"
  echo "residual=${RESIDUAL}"
  echo "log=${LOG_FILE}"
  echo "out=${OUT_FILE}"
  echo "model=${MODEL_PATH}"
  echo "========================================================="

  python main.py \
    --model_name MyModelV2 \
    --dataset "${DATASET}" \
    --path ./data/ \
    --gpu "${GPU}" \
    --random_seed "${SEED}" \
    --emb_size "${EMB_SIZE}" \
    --attn_size "${ATTN_SIZE}" \
    --K "${K}" \
    --prompt_num "${PROMPT_NUM}" \
    --n_layers "${N_LAYERS}" \
    --lamb "${LAMB}" \
    --history_max "${HISTORY_MAX}" \
    --use_llmemb 1 \
    --llm_fuse 1 \
    --llm_emb_path "${LLM_PATH}" \
    --srs_emb_path "${SRS_PATH}" \
    --gamma_init "${GAMMA_INIT}" \
    --gamma_trainable 0 \
    --alpha "${ALPHA}" \
    --tau "${TAU}" \
    --rat_alpha_warmup_steps 5000 \
    --use_sicr 1 \
    --sicr_beta "${SICR_BETA}" \
    --sicr_sem_weight "${SICR_SEM_WEIGHT}" \
    --sicr_intent_weight "${SICR_INTENT_WEIGHT}" \
    --sicr_center 1 \
    --sicr_residual "${RESIDUAL}" \
    --sicr_warmup_steps 5000 \
    --sicr_detach 1 \
    --use_emile 1 \
    --lambda_ipd "${LAMBDA_IPD}" \
    --ipd_margin "${IPD_MARGIN}" \
    --emile_warmup_steps 5000 \
    --lr "${LR}" \
    --l2 "${L2}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --num_neg 1 \
    --epoch "${EPOCH}" \
    --early_stop "${EARLY_STOP}" \
    --num_workers "${NUM_WORKERS}" \
    --log_file "${LOG_FILE}" \
    --model_path "${MODEL_PATH}" \
    2>&1 | tee "${OUT_FILE}"

  END_TIME=$(date "+%Y-%m-%d %H:%M:%S")
  END_TS=$(date +%s)
  TOTAL_SECONDS=$((END_TS - START_TS))

  local BEST_DEV
  local TEST_AFTER

  BEST_DEV=$(grep "Best Iter(dev)" "${LOG_FILE}" | tail -n 1 | tr '\t' ' ' | sed 's/[[:space:]]\+/ /g')
  TEST_AFTER=$(grep "Test After Training" "${LOG_FILE}" | tail -n 1 | tr '\t' ' ' | sed 's/[[:space:]]\+/ /g')

  echo "========================================================="
  echo "[DONE] ${DATASET} | ${VARIANT}"
  echo "start=${START_TIME}"
  echo "end=${END_TIME}"
  echo "total_seconds=${TOTAL_SECONDS}"
  echo "${BEST_DEV}"
  echo "${TEST_AFTER}"
  echo "========================================================="

  echo -e "${DATASET}\t${VARIANT}\t${SEED}\t${START_TIME}\t${END_TIME}\t${TOTAL_SECONDS}\t${BEST_DEV}\t${TEST_AFTER}" >> "${SUMMARY_FILE}"
}


# =========================================================
# File paths
# =========================================================

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


# =========================================================
# Beauty configs
# Old MyModel seed42 target:
# HR@5=0.1614, NDCG@5=0.1119
#
# Fixed:
# beta=0.05, sem=0.2, intent=0.3
# residual=0.2/0.3/0.5
# =========================================================

run_one \
  "beauty" \
  "MyModelV2_SICR_center_b005_s02_i03_r02_seed42" \
  0.2 \
  64 8 4 3 2 3.0 20 \
  "${BEAUTY_LLM}" \
  "${BEAUTY_SRS}" \
  0.1 \
  0.001 \
  0.2 \
  0.05 \
  0.2 \
  0.3 \
  0.05 \
  0.2 \
  0.002 \
  1e-6 \
  256 \
  256 \
  200 \
  10 \
  5


run_one \
  "beauty" \
  "MyModelV2_SICR_center_b005_s02_i03_r03_seed42" \
  0.3 \
  64 8 4 3 2 3.0 20 \
  "${BEAUTY_LLM}" \
  "${BEAUTY_SRS}" \
  0.1 \
  0.001 \
  0.2 \
  0.05 \
  0.2 \
  0.3 \
  0.05 \
  0.2 \
  0.002 \
  1e-6 \
  256 \
  256 \
  200 \
  10 \
  5


run_one \
  "beauty" \
  "MyModelV2_SICR_center_b005_s02_i03_r05_seed42" \
  0.5 \
  64 8 4 3 2 3.0 20 \
  "${BEAUTY_LLM}" \
  "${BEAUTY_SRS}" \
  0.1 \
  0.001 \
  0.2 \
  0.05 \
  0.2 \
  0.3 \
  0.05 \
  0.2 \
  0.002 \
  1e-6 \
  256 \
  256 \
  200 \
  10 \
  5


# =========================================================
# ML-1M configs
# Old MyModel mean target:
# HR@5=0.3165, NDCG@5=0.2200
#
# Fixed:
# beta=0.03, sem=0.1, intent=0.3
# residual=0.2/0.3/0.5
# =========================================================

run_one \
  "ml-1m" \
  "MyModelV2_SICR_center_b003_s01_i03_r02_seed42" \
  0.2 \
  64 8 2 3 2 3.0 20 \
  "${ML1M_LLM}" \
  "${ML1M_SRS}" \
  0.08 \
  0.001 \
  0.3 \
  0.03 \
  0.1 \
  0.3 \
  0.02 \
  0.10 \
  0.001 \
  1e-6 \
  256 \
  256 \
  200 \
  10 \
  5


run_one \
  "ml-1m" \
  "MyModelV2_SICR_center_b003_s01_i03_r03_seed42" \
  0.3 \
  64 8 2 3 2 3.0 20 \
  "${ML1M_LLM}" \
  "${ML1M_SRS}" \
  0.08 \
  0.001 \
  0.3 \
  0.03 \
  0.1 \
  0.3 \
  0.02 \
  0.10 \
  0.001 \
  1e-6 \
  256 \
  256 \
  200 \
  10 \
  5


run_one \
  "ml-1m" \
  "MyModelV2_SICR_center_b003_s01_i03_r05_seed42" \
  0.5 \
  64 8 2 3 2 3.0 20 \
  "${ML1M_LLM}" \
  "${ML1M_SRS}" \
  0.08 \
  0.001 \
  0.3 \
  0.03 \
  0.1 \
  0.3 \
  0.02 \
  0.10 \
  0.001 \
  1e-6 \
  256 \
  256 \
  200 \
  10 \
  5


echo ""
echo "========================================================="
echo "All 6 MyModelV2 SICR centered + residual runs finished."
echo "Summary:"
echo "${SUMMARY_FILE}"
echo "cat ${SUMMARY_FILE}"
echo "========================================================="