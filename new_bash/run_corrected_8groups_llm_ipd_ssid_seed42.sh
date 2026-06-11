#!/usr/bin/env bash
set -e

GPU=${1:-0}
SEED=42

ROOT_LOG_DIR="new_log/corrected_8groups_llm_ipd_ssid"
ROOT_MODEL_DIR="new_model/corrected_8groups_llm_ipd_ssid"

mkdir -p "${ROOT_LOG_DIR}/beauty"
mkdir -p "${ROOT_LOG_DIR}/ml-1m"
mkdir -p "${ROOT_MODEL_DIR}/beauty"
mkdir -p "${ROOT_MODEL_DIR}/ml-1m"

SUMMARY_FILE="${ROOT_LOG_DIR}/summary_seed${SEED}.tsv"
echo -e "dataset\tmodel\tvariant\tseed\tstart_time\tend_time\ttotal_seconds\tbest_dev\ttest_after_training" > "${SUMMARY_FILE}"

BEAUTY_LLM="./data/beauty/handled/llm_table_pca1536.pkl"
BEAUTY_SRS="./data/beauty/handled/itm_emb_pomrec.pkl"

ML1M_LLM="./data/ml-1m/handled/llm_table_pca1536.pkl"
ML1M_SRS="./data/ml-1m/handled/itm_emb_pomrec.pkl"

BEAUTY_INIT="./model/PoMRec/PoMRec__beauty__${SEED}__lr=0.002__l2=1e-06.pt"
BEAUTY_INIT_FALLBACK="./model/PoMRec/PoMRec__beauty__42__lr=0.002__l2=1e-06.pt"

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
  local MODEL_NAME=$2
  local VARIANT=$3
  local LR=$4
  local GAMMA=$5
  local TAU=$6
  local LLM_PATH=$7
  local SRS_PATH=$8
  local INIT_CKPT=$9
  shift 9
  local EXTRA_ARGS=("$@")

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
  echo "extra_args=${EXTRA_ARGS[*]}"
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
    "${EXTRA_ARGS[@]}" \
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
# Beauty corrected config
# lr=0.002, gamma=0.1, tau=0.2
# IPD: lambda=0.05, margin=0.2, warmup=5000
# =========================================================

run_one \
  "beauty" \
  "MyModelLLM" \
  "beauty_LLM_only_seed42" \
  0.002 \
  0.1 \
  0.2 \
  "${BEAUTY_LLM}" \
  "${BEAUTY_SRS}" \
  "${BEAUTY_INIT}"

run_one \
  "beauty" \
  "MyModelLLMIPD" \
  "beauty_LLM_IPD_seed42" \
  0.002 \
  0.1 \
  0.2 \
  "${BEAUTY_LLM}" \
  "${BEAUTY_SRS}" \
  "${BEAUTY_INIT}" \
  --use_emile 1 \
  --lambda_ipd 0.05 \
  --ipd_margin 0.2 \
  --emile_use_fused_itememb 0 \
  --emile_warmup_steps 5000

run_one \
  "beauty" \
  "MyModelV5" \
  "beauty_LLM_IPD_SSID_lam0001_seed42" \
  0.002 \
  0.1 \
  0.2 \
  "${BEAUTY_LLM}" \
  "${BEAUTY_SRS}" \
  "${BEAUTY_INIT}" \
  --use_dspc 0 \
  --use_ssid 1 \
  --lambda_ssid 0.001 \
  --ssid_temp 0.2 \
  --ssid_warmup_steps 5000 \
  --ssid_detach_sem 1 \
  --ssid_detach_attn 1 \
  --ssid_use_proto_norm 1 \
  --use_emile 1 \
  --lambda_ipd 0.05 \
  --ipd_margin 0.2 \
  --emile_use_fused_itememb 0 \
  --emile_warmup_steps 5000

run_one \
  "beauty" \
  "MyModelV5" \
  "beauty_LLM_IPD_SSID_lam0003_seed42" \
  0.002 \
  0.1 \
  0.2 \
  "${BEAUTY_LLM}" \
  "${BEAUTY_SRS}" \
  "${BEAUTY_INIT}" \
  --use_dspc 0 \
  --use_ssid 1 \
  --lambda_ssid 0.003 \
  --ssid_temp 0.2 \
  --ssid_warmup_steps 5000 \
  --ssid_detach_sem 1 \
  --ssid_detach_attn 1 \
  --ssid_use_proto_norm 1 \
  --use_emile 1 \
  --lambda_ipd 0.05 \
  --ipd_margin 0.2 \
  --emile_use_fused_itememb 0 \
  --emile_warmup_steps 5000

# =========================================================
# ML-1M corrected config
# lr=0.001, gamma=0.08, tau=0.3
# IPD: lambda=0.02, margin=0.10, warmup=20000
# =========================================================

run_one \
  "ml-1m" \
  "MyModelLLM" \
  "ml1m_LLM_only_seed42" \
  0.001 \
  0.08 \
  0.3 \
  "${ML1M_LLM}" \
  "${ML1M_SRS}" \
  "${ML1M_INIT}"

run_one \
  "ml-1m" \
  "MyModelLLMIPD" \
  "ml1m_LLM_IPD_seed42" \
  0.001 \
  0.08 \
  0.3 \
  "${ML1M_LLM}" \
  "${ML1M_SRS}" \
  "${ML1M_INIT}" \
  --use_emile 1 \
  --lambda_ipd 0.02 \
  --ipd_margin 0.10 \
  --emile_use_fused_itememb 0 \
  --emile_warmup_steps 20000

run_one \
  "ml-1m" \
  "MyModelV5" \
  "ml1m_LLM_IPD_SSID_lam00005_seed42" \
  0.001 \
  0.08 \
  0.3 \
  "${ML1M_LLM}" \
  "${ML1M_SRS}" \
  "${ML1M_INIT}" \
  --use_dspc 0 \
  --use_ssid 1 \
  --lambda_ssid 0.0005 \
  --ssid_temp 0.2 \
  --ssid_warmup_steps 5000 \
  --ssid_detach_sem 1 \
  --ssid_detach_attn 1 \
  --ssid_use_proto_norm 1 \
  --use_emile 1 \
  --lambda_ipd 0.02 \
  --ipd_margin 0.10 \
  --emile_use_fused_itememb 0 \
  --emile_warmup_steps 20000

run_one \
  "ml-1m" \
  "MyModelV5" \
  "ml1m_LLM_IPD_SSID_lam0001_seed42" \
  0.001 \
  0.08 \
  0.3 \
  "${ML1M_LLM}" \
  "${ML1M_SRS}" \
  "${ML1M_INIT}" \
  --use_dspc 0 \
  --use_ssid 1 \
  --lambda_ssid 0.001 \
  --ssid_temp 0.2 \
  --ssid_warmup_steps 5000 \
  --ssid_detach_sem 1 \
  --ssid_detach_attn 1 \
  --ssid_use_proto_norm 1 \
  --use_emile 1 \
  --lambda_ipd 0.02 \
  --ipd_margin 0.10 \
  --emile_use_fused_itememb 0 \
  --emile_warmup_steps 20000

echo ""
echo "========================================================="
echo "Corrected 8-group LLM/IPD/SSID ablation finished."
echo "Summary:"
echo "cat ${SUMMARY_FILE}"
echo "========================================================="