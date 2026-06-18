#!/usr/bin/env bash
set -e

GPU=${1:-0}
SEED=42

ROOT_LOG_DIR="new_log/itirl_4groups_beauty_ml1m_seed42"
ROOT_MODEL_DIR="new_model/itirl_4groups_beauty_ml1m_seed42"

mkdir -p "${ROOT_LOG_DIR}/beauty"
mkdir -p "${ROOT_LOG_DIR}/ml-1m"
mkdir -p "${ROOT_MODEL_DIR}/beauty"
mkdir -p "${ROOT_MODEL_DIR}/ml-1m"

SUMMARY_FILE="${ROOT_LOG_DIR}/summary_seed${SEED}.tsv"
echo -e "dataset\tmodel\tvariant\tseed\tstart_time\tend_time\ttotal_seconds\tbest_dev\ttest_after_training" > "${SUMMARY_FILE}"

BEAUTY_LLM="./data/beauty/handled/llm_table_pca1536.pkl"
BEAUTY_SRS="./data/beauty/handled/itm_emb_pomrec.pkl"
BEAUTY_INIT="./model/PoMRec/PoMRec__beauty__42__lr=0.002__l2=1e-06.pt"

ML1M_LLM="./data/ml-1m/handled/llm_table_pca1536.pkl"
ML1M_SRS="./data/ml-1m/handled/itm_emb_pomrec.pkl"

# 注意：ML-1M 固定使用 README 里的 seed=1 checkpoint
ML1M_INIT="./model/PoMRec/PoMRec__ml-1m__1__lr=0.001__l2=1e-06.pt"

check_required_file() {
  if [ ! -f "$1" ]; then
    echo "[ERROR] Required file not found: $1"
    exit 1
  fi
}

check_required_file "${BEAUTY_LLM}"
check_required_file "${BEAUTY_SRS}"
check_required_file "${BEAUTY_INIT}"
check_required_file "${ML1M_LLM}"
check_required_file "${ML1M_SRS}"
check_required_file "${ML1M_INIT}"

run_one() {
  local DATASET=$1
  local VARIANT=$2
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
  local LAMBDA_ITIRL=${13}
  local ITIRL_WARMUP=${14}
  local MARGIN_TH=${15}

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
  echo "[START] ${DATASET} | MyModelITIRL | ${VARIANT}"
  echo "GPU=${GPU}"
  echo "init_ckpt=${INIT_CKPT}"
  echo "lambda_itirl=${LAMBDA_ITIRL}"
  echo "margin_threshold=${MARGIN_TH}"
  echo "log=${LOG_FILE}"
  echo "========================================================="

  python main.py \
    --model_name MyModelITIRL \
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
    --use_itirl 1 \
    --lambda_itirl "${LAMBDA_ITIRL}" \
    --itirl_warmup_steps "${ITIRL_WARMUP}" \
    --itirl_route_source ipd \
    --itirl_gate_type margin \
    --itirl_conf_threshold 0.45 \
    --itirl_margin_threshold "${MARGIN_TH}" \
    --itirl_gate_mode linear \
    --itirl_gate_temp 0.05 \
    --itirl_score_norm 0 \
    --itirl_neg_reduce mean \
    --itirl_loss_normalize batch \
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

  BEST_DEV=$(grep "Best Iter(dev)" "${LOG_FILE}" | tail -n 1 | tr '\t' ' ' | sed 's/[[:space:]]\+/ /g' || true)
  TEST_AFTER=$(grep "Test After Training" "${LOG_FILE}" | tail -n 1 | tr '\t' ' ' | sed 's/[[:space:]]\+/ /g' || true)

  echo "========================================================="
  echo "[DONE] ${DATASET} | ${VARIANT}"
  echo "${BEST_DEV}"
  echo "${TEST_AFTER}"
  echo "========================================================="

  echo -e "${DATASET}\tMyModelITIRL\t${VARIANT}\t${SEED}\t${START_TIME}\t${END_TIME}\t${TOTAL_SECONDS}\t${BEST_DEV}\t${TEST_AFTER}" >> "${SUMMARY_FILE}"
}

# =========================================================
# Beauty 1: stronger ITIRL
# baseline compare: beauty_LLM_IPD_seed42
# =========================================================
run_one \
  "beauty" \
  "beauty_ITIRL_lam002_margin005_seed42" \
  0.002 \
  0.1 \
  0.2 \
  3.0 \
  "${BEAUTY_LLM}" \
  "${BEAUTY_SRS}" \
  "${BEAUTY_INIT}" \
  0.05 \
  0.2 \
  5000 \
  0.02 \
  5000 \
  0.05

# =========================================================
# Beauty 2: weaker ITIRL
# =========================================================
run_one \
  "beauty" \
  "beauty_ITIRL_lam001_margin005_seed42" \
  0.002 \
  0.1 \
  0.2 \
  3.0 \
  "${BEAUTY_LLM}" \
  "${BEAUTY_SRS}" \
  "${BEAUTY_INIT}" \
  0.05 \
  0.2 \
  5000 \
  0.01 \
  5000 \
  0.05

# =========================================================
# ML-1M 1: moderate ITIRL
# baseline compare: ml1m_LLM_IPD_seed42
# =========================================================
run_one \
  "ml-1m" \
  "ml1m_ITIRL_lam0005_margin010_seed42" \
  0.001 \
  0.08 \
  0.3 \
  3.0 \
  "${ML1M_LLM}" \
  "${ML1M_SRS}" \
  "${ML1M_INIT}" \
  0.02 \
  0.10 \
  20000 \
  0.005 \
  20000 \
  0.10

# =========================================================
# ML-1M 2: weaker ITIRL
# =========================================================
run_one \
  "ml-1m" \
  "ml1m_ITIRL_lam0001_margin010_seed42" \
  0.001 \
  0.08 \
  0.3 \
  3.0 \
  "${ML1M_LLM}" \
  "${ML1M_SRS}" \
  "${ML1M_INIT}" \
  0.02 \
  0.10 \
  20000 \
  0.001 \
  20000 \
  0.10

echo ""
echo "========================================================="
echo "ITIRL 4 groups Beauty + ML-1M seed42 finished."
echo "Summary:"
echo "cat ${SUMMARY_FILE}"
echo "========================================================="