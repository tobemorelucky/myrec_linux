#!/usr/bin/env bash
set -e

GPU=${1:-0}
PART=${2:-0}
SEED=42

ROOT_LOG_DIR="new_log/shncf_4groups_split_seed42"
ROOT_MODEL_DIR="new_model/shncf_4groups_split_seed42"

mkdir -p "${ROOT_LOG_DIR}/beauty"
mkdir -p "${ROOT_LOG_DIR}/ml-1m"
mkdir -p "${ROOT_MODEL_DIR}/beauty"
mkdir -p "${ROOT_MODEL_DIR}/ml-1m"

SUMMARY_FILE="${ROOT_LOG_DIR}/summary_gpu${GPU}_part${PART}_seed${SEED}.tsv"
echo -e "dataset\tmodel\tvariant\tseed\tgpu\tpart\tstart_time\tend_time\ttotal_seconds\tbest_dev\ttest_after_training" > "${SUMMARY_FILE}"

check_required_file() {
  if [ ! -f "$1" ]; then
    echo "[ERROR] Required file not found: $1"
    exit 1
  fi
}

BEAUTY_LLM="./data/beauty/handled/llm_table_pca1536.pkl"
BEAUTY_SRS="./data/beauty/handled/itm_emb_pomrec.pkl"
BEAUTY_INIT="./model/PoMRec/PoMRec__beauty__42__lr=0.002__l2=1e-06.pt"
BEAUTY_SHNC="./data/beauty/handled/semantic_hardneg_top100.pkl"

ML1M_LLM="./data/ml-1m/handled/llm_table_pca1536.pkl"
ML1M_SRS="./data/ml-1m/handled/itm_emb_pomrec.pkl"
ML1M_INIT="./model/PoMRec/PoMRec__ml-1m__1__lr=0.001__l2=1e-06.pt"
ML1M_SHNC="./data/ml-1m/handled/semantic_hardneg_top100.pkl"

for f in \
  "$BEAUTY_LLM" "$BEAUTY_SRS" "$BEAUTY_INIT" "$BEAUTY_SHNC" \
  "$ML1M_LLM" "$ML1M_SRS" "$ML1M_INIT" "$ML1M_SHNC"; do
  check_required_file "$f"
done

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
  local SHNC_PATH=${10}
  local LAMBDA_IPD=${11}
  local IPD_MARGIN=${12}
  local EMILE_WARMUP=${13}
  local LAMBDA_SHNC=${14}
  local SHNC_WARMUP=${15}
  local RANK_START=${16}
  local RANK_END=${17}

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
  echo "[START] ${DATASET} | MyModelSHNC | ${VARIANT}"
  echo "GPU=${GPU}"
  echo "PART=${PART}"
  echo "seed=${SEED}"
  echo "init_ckpt=${INIT_CKPT}"
  echo "shnc_path=${SHNC_PATH}"
  echo "lambda_shnc=${LAMBDA_SHNC}"
  echo "rank_start=${RANK_START}"
  echo "rank_end=${RANK_END}"
  echo "log=${LOG_FILE}"
  echo "========================================================="

  python main.py \
    --model_name MyModelSHNC \
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
    --use_shnc 1 \
    --shnc_path "${SHNC_PATH}" \
    --lambda_shnc "${LAMBDA_SHNC}" \
    --shnc_warmup_steps "${SHNC_WARMUP}" \
    --shnc_num 1 \
    --shnc_neg_reduce mean \
    --shnc_score_norm 0 \
    --shnc_detach_user 0 \
    --shnc_sample_mode random \
    --shnc_filter_history 1 \
    --shnc_rank_start "${RANK_START}" \
    --shnc_rank_end "${RANK_END}" \
    --shnc_resample_attempts 5 \
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

  echo -e "${DATASET}\tMyModelSHNC\t${VARIANT}\t${SEED}\t${GPU}\t${PART}\t${START_TIME}\t${END_TIME}\t${TOTAL_SECONDS}\t${BEST_DEV}\t${TEST_AFTER}" >> "${SUMMARY_FILE}"
}

echo ""
echo "========================================================="
echo "Start SHNC-F split runs."
echo "GPU=${GPU}"
echo "PART=${PART}"
echo "Summary=${SUMMARY_FILE}"
echo "========================================================="

if [ "${PART}" = "0" ]; then
  # =========================================================
  # GPU 0 / PART 0
  # 1) ML-1M lambda=0.005 rank20-100
  # 2) ML-1M lambda=0.002 rank20-100
  # =========================================================

  run_one \
    "ml-1m" \
    "ml1m_SHNCF_lam0005_rank20_100_seed42" \
    0.001 \
    0.08 \
    0.3 \
    3.0 \
    "${ML1M_LLM}" \
    "${ML1M_SRS}" \
    "${ML1M_INIT}" \
    "${ML1M_SHNC}" \
    0.02 \
    0.10 \
    20000 \
    0.005 \
    20000 \
    20 \
    100

  run_one \
    "ml-1m" \
    "ml1m_SHNCF_lam0002_rank20_100_seed42" \
    0.001 \
    0.08 \
    0.3 \
    3.0 \
    "${ML1M_LLM}" \
    "${ML1M_SRS}" \
    "${ML1M_INIT}" \
    "${ML1M_SHNC}" \
    0.02 \
    0.10 \
    20000 \
    0.002 \
    20000 \
    20 \
    100

elif [ "${PART}" = "1" ]; then
  # =========================================================
  # GPU 1 / PART 1
  # 3) ML-1M lambda=0.005 rank10-80
  # 4) Beauty lambda=0.01 rank0-100
  # =========================================================

  run_one \
    "ml-1m" \
    "ml1m_SHNCF_lam0005_rank10_80_seed42" \
    0.001 \
    0.08 \
    0.3 \
    3.0 \
    "${ML1M_LLM}" \
    "${ML1M_SRS}" \
    "${ML1M_INIT}" \
    "${ML1M_SHNC}" \
    0.02 \
    0.10 \
    20000 \
    0.005 \
    20000 \
    10 \
    80

  run_one \
    "beauty" \
    "beauty_SHNCF_lam001_rank0_100_seed42" \
    0.002 \
    0.1 \
    0.2 \
    3.0 \
    "${BEAUTY_LLM}" \
    "${BEAUTY_SRS}" \
    "${BEAUTY_INIT}" \
    "${BEAUTY_SHNC}" \
    0.05 \
    0.2 \
    5000 \
    0.01 \
    5000 \
    0 \
    100

else
  echo "[ERROR] PART must be 0 or 1, got: ${PART}"
  exit 1
fi

echo ""
echo "========================================================="
echo "SHNC-F split run finished."
echo "Summary:"
echo "cat ${SUMMARY_FILE}"
echo "========================================================="
