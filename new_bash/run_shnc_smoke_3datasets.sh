#!/usr/bin/env bash
set -e

GPU=${1:-0}
SEED=42

ROOT_LOG_DIR="new_log/shnc_smoke"
ROOT_MODEL_DIR="new_model/shnc_smoke"

mkdir -p "${ROOT_LOG_DIR}/beauty"
mkdir -p "${ROOT_LOG_DIR}/ml-1m"
mkdir -p "${ROOT_LOG_DIR}/toys"
mkdir -p "${ROOT_MODEL_DIR}/beauty"
mkdir -p "${ROOT_MODEL_DIR}/ml-1m"
mkdir -p "${ROOT_MODEL_DIR}/toys"

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

TOYS_LLM="./data/toys/handled/llm_table_pca1536.pkl"
TOYS_SRS="./data/toys/handled/itm_emb_pomrec.pkl"
TOYS_INIT="./model/PoMRec/toys__42__lr=0.001__l2=1e-06__lamb=3.8__history_max=20.pt"
TOYS_SHNC="./data/toys/handled/semantic_hardneg_top100.pkl"

for f in \
  "$BEAUTY_LLM" "$BEAUTY_SRS" "$BEAUTY_INIT" "$BEAUTY_SHNC" \
  "$ML1M_LLM" "$ML1M_SRS" "$ML1M_INIT" "$ML1M_SHNC" \
  "$TOYS_LLM" "$TOYS_SRS" "$TOYS_INIT" "$TOYS_SHNC"; do
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

  local LOG_FILE="${ROOT_LOG_DIR}/${DATASET}/${VARIANT}.log"
  local MODEL_PATH="${ROOT_MODEL_DIR}/${DATASET}/${VARIANT}.pt"

  echo ""
  echo "========================================================="
  echo "[SMOKE START] ${DATASET} | ${VARIANT}"
  echo "GPU=${GPU}"
  echo "init_ckpt=${INIT_CKPT}"
  echo "shnc_path=${SHNC_PATH}"
  echo "lambda_shnc=${LAMBDA_SHNC}"
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
    --lr "${LR}" \
    --l2 1e-6 \
    --batch_size 256 \
    --eval_batch_size 256 \
    --num_neg 1 \
    --dropout 0 \
    --epoch 1 \
    --early_stop 1 \
    --num_workers 5 \
    --log_file "${LOG_FILE}" \
    --model_path "${MODEL_PATH}"

  echo "========================================================="
  echo "[SMOKE DONE] ${DATASET} | ${VARIANT}"
  grep "Test After Training" "${LOG_FILE}" | tail -n 1 || true
  echo "========================================================="
}

run_one \
  "beauty" \
  "beauty_SHNC_smoke_seed42" \
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
  5000

run_one \
  "ml-1m" \
  "ml1m_SHNC_smoke_seed42" \
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
  20000

run_one \
  "toys" \
  "toys_SHNC_smoke_seed42" \
  0.001 \
  0.05 \
  0.5 \
  3.8 \
  "${TOYS_LLM}" \
  "${TOYS_SRS}" \
  "${TOYS_INIT}" \
  "${TOYS_SHNC}" \
  0.05 \
  0.10 \
  20000 \
  0.01 \
  20000

echo ""
echo "========================================================="
echo "All SHNC smoke tests finished."
echo "========================================================="
