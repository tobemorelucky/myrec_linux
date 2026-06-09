#!/usr/bin/env bash
set -e

GPU=${1:-0}
SEED=42

ROOT_LOG_DIR="new_log/mymodelv2_ugctic"
ROOT_MODEL_DIR="new_model/mymodelv2_ugctic"

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
  local USE_SICR=${11}
  local SICR_GLOBAL_R=${12}
  local SICR_GLOBAL_B=${13}
  local LAMBDA_SOFT_TIC=${14}

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
  echo "use_sicr=${USE_SICR}"
  echo "lambda_soft_tic=${LAMBDA_SOFT_TIC}"
  echo "soft_tic_gate=band"
  echo "log=${LOG_FILE}"
  echo "========================================================="

  python main.py \
    --model_name MyModelV2 \
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
    --use_sicr "${USE_SICR}" \
    --sicr_train_only 1 \
    --sicr_global_gate 1 \
    --sicr_global_alpha 8 \
    --sicr_global_b "${SICR_GLOBAL_B}" \
    --sicr_global_r "${SICR_GLOBAL_R}" \
    --sicr_entropy_adapt 1 \
    --sicr_entropy_min 0.5 \
    --sicr_entropy_gamma 1.0 \
    --sicr_global_sem_weight 0.0 \
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
    --use_soft_tic 1 \
    --lambda_soft_tic "${LAMBDA_SOFT_TIC}" \
    --soft_tic_temp 0.5 \
    --soft_tic_warmup_steps 5000 \
    --soft_tic_use_fused 0 \
    --soft_tic_gate band \
    --soft_tic_gate_gamma 1.0 \
    --soft_tic_min_weight 0.0 \
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
# Beauty 1: IPD + UGC-TIC band, lambda=0.01, no SICR
# =========================================================
run_one \
  "beauty" \
  "MyModelV2_IPD_UGCTIC_band_lam001_noSICR_seed42" \
  4 \
  0.1 \
  0.2 \
  0.05 \
  0.2 \
  0.002 \
  "${BEAUTY_LLM}" \
  "${BEAUTY_SRS}" \
  0 \
  0.15 \
  0.3 \
  0.01

# =========================================================
# Beauty 2: IPD + UGC-TIC band, lambda=0.005, no SICR
# =========================================================
run_one \
  "beauty" \
  "MyModelV2_IPD_UGCTIC_band_lam0005_noSICR_seed42" \
  4 \
  0.1 \
  0.2 \
  0.05 \
  0.2 \
  0.002 \
  "${BEAUTY_LLM}" \
  "${BEAUTY_SRS}" \
  0 \
  0.15 \
  0.3 \
  0.005

# =========================================================
# ML-1M 1: IU-SCBR(r=0.06) + IPD + UGC-TIC band, lambda=0.005
# =========================================================
run_one \
  "ml-1m" \
  "MyModelV2_IUSCBR_IPD_UGCTIC_band_lam0005_r006_seed42" \
  2 \
  0.08 \
  0.3 \
  0.02 \
  0.10 \
  0.001 \
  "${ML1M_LLM}" \
  "${ML1M_SRS}" \
  1 \
  0.06 \
  0.4 \
  0.005

# =========================================================
# ML-1M 2: IU-SCBR(r=0.06) + IPD + UGC-TIC band, lambda=0.003
# =========================================================
run_one \
  "ml-1m" \
  "MyModelV2_IUSCBR_IPD_UGCTIC_band_lam0003_r006_seed42" \
  2 \
  0.08 \
  0.3 \
  0.02 \
  0.10 \
  0.001 \
  "${ML1M_LLM}" \
  "${ML1M_SRS}" \
  1 \
  0.06 \
  0.4 \
  0.003

echo ""
echo "========================================================="
echo "All UGC-TIC runs finished."
echo "Summary:"
echo "cat ${SUMMARY_FILE}"
echo "========================================================="