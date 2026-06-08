#!/usr/bin/env bash
set -e

GPU=${1:-0}
SEED=42

ROOT_LOG_DIR="new_log/mymodelv2_dual_sicr_min"
ROOT_MODEL_DIR="new_model/mymodelv2_dual_sicr_min"

mkdir -p "${ROOT_LOG_DIR}/beauty"
mkdir -p "${ROOT_LOG_DIR}/ml-1m"
mkdir -p "${ROOT_MODEL_DIR}/beauty"
mkdir -p "${ROOT_MODEL_DIR}/ml-1m"

SUMMARY_FILE="${ROOT_LOG_DIR}/summary_seed${SEED}.tsv"
echo -e "dataset\tvariant\tseed\tstart_time\tend_time\ttotal_seconds\tbest_dev\ttest_after_training" > "${SUMMARY_FILE}"

run_one() {
  local DATASET=$1
  local VARIANT=$2
  local K=$3
  local LAMB=$4
  local GAMMA=$5
  local TAU=$6
  local LAMBDA_IPD=$7
  local IPD_MARGIN=$8
  local LR=$9
  local LLM_PATH=${10}
  local SRS_PATH=${11}
  local GLOBAL_B=${12}
  local GLOBAL_R=${13}
  local SICR_BETA=${14}
  local SICR_SEM=${15}
  local SICR_INTENT=${16}
  local SICR_RESIDUAL=${17}

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
    --use_sicr 1 \
    --sicr_global_gate 1 \
    --sicr_global_alpha 8 \
    --sicr_global_b "${GLOBAL_B}" \
    --sicr_global_r "${GLOBAL_R}" \
    --sicr_beta "${SICR_BETA}" \
    --sicr_sem_weight "${SICR_SEM}" \
    --sicr_intent_weight "${SICR_INTENT}" \
    --sicr_center 1 \
    --sicr_residual "${SICR_RESIDUAL}" \
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


BEAUTY_LLM="./data/beauty/handled/llm_table_pca1536.pkl"
BEAUTY_SRS="./data/beauty/handled/itm_emb_pomrec.pkl"

ML1M_LLM="./data/ml-1m/handled/llm_table_pca1536.pkl"
ML1M_SRS="./data/ml-1m/handled/itm_emb_pomrec.pkl"

# =========================
# Beauty
# =========================

run_one \
  "beauty" \
  "MyModelV2_DualSICR_global_only_b000_r02_seed42" \
  4 \
  3.0 \
  0.1 \
  0.2 \
  0.05 \
  0.2 \
  0.002 \
  "${BEAUTY_LLM}" \
  "${BEAUTY_SRS}" \
  0.3 \
  0.15 \
  0.0 \
  0.2 \
  0.3 \
  0.2

run_one \
  "beauty" \
  "MyModelV2_DualSICR_global_local_b003_r02_seed42" \
  4 \
  3.0 \
  0.1 \
  0.2 \
  0.05 \
  0.2 \
  0.002 \
  "${BEAUTY_LLM}" \
  "${BEAUTY_SRS}" \
  0.3 \
  0.15 \
  0.03 \
  0.2 \
  0.3 \
  0.2

# =========================
# ML-1M
# =========================

run_one \
  "ml-1m" \
  "MyModelV2_DualSICR_global_only_b000_r02_seed42" \
  2 \
  3.0 \
  0.08 \
  0.3 \
  0.02 \
  0.10 \
  0.001 \
  "${ML1M_LLM}" \
  "${ML1M_SRS}" \
  0.4 \
  0.08 \
  0.0 \
  0.0 \
  0.3 \
  0.2

run_one \
  "ml-1m" \
  "MyModelV2_DualSICR_global_local_b002_r02_seed42" \
  2 \
  3.0 \
  0.08 \
  0.3 \
  0.02 \
  0.10 \
  0.001 \
  "${ML1M_LLM}" \
  "${ML1M_SRS}" \
  0.4 \
  0.08 \
  0.02 \
  0.0 \
  0.3 \
  0.2

echo ""
echo "========================================================="
echo "All runs finished."
echo "Summary:"
echo "cat ${SUMMARY_FILE}"
echo "========================================================="