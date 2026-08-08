#!/usr/bin/env bash
set -e

# =========================================================
# LLMMIRec Phase 1 — ASPCF Gate ablation (ML-1M seed=42)
#
# Usage:
#   GATE_MODE=basic bash .../run_...sh [GPU_ID]
#   GATE_MODE=conflict bash .../run_...sh [GPU_ID]
#   SMOKE=1 GATE_MODE=basic bash .../run_...sh [GPU_ID]
#
# GPU control: $1 (default 0) passed as --gpu.
# =========================================================

GPU=${1:-0}
SEED=42
DATASET="ml-1m"
GATE_MODE=${GATE_MODE:-basic}
SEMANTIC_RANK=512
LAMBDA_RELATION=0.01

ROOT_LOG_DIR="new_log/llmmirec_phase1_gate/${DATASET}"
ROOT_MODEL_DIR="new_model/llmmirec_phase1_gate/${DATASET}"
mkdir -p "${ROOT_LOG_DIR}" "${ROOT_MODEL_DIR}"

if [ "${SMOKE}" = "1" ]; then
  EPOCH=2; EARLY_STOP=0; NUM_WORKERS=0; SMOKE_TAG="[SMOKE]"
else
  EPOCH=200; EARLY_STOP=10; NUM_WORKERS=5; SMOKE_TAG=""
fi

LABEL="aspcf_${GATE_MODE}_sr${SEMANTIC_RANK}_lr${LAMBDA_RELATION}"
SUMMARY_FILE="${ROOT_LOG_DIR}/summary_seed${SEED}.tsv"

if [ ! -f "${SUMMARY_FILE}" ]; then
  echo -e "dataset\tmodel\titem_encoder\tgate_mode\tsemantic_rank\tlambda_relation\tseed\tK\tlr\tstart_time\tend_time\ttotal_seconds\tparameter_count\tbest_epoch\tbest_dev\ttest_after_training\tstatus" > "${SUMMARY_FILE}"
fi

echo "========================================================="
echo "LLMMIRec Phase 1 — Gate ${GATE_MODE} ${DATASET} ${SMOKE_TAG}"
echo "GPU=${GPU}  lambda_relation=${LAMBDA_RELATION}"
echo "========================================================="

LOG_DIR="${ROOT_LOG_DIR}/${LABEL}"
MODEL_DIR="${ROOT_MODEL_DIR}/${LABEL}"
mkdir -p "${LOG_DIR}" "${MODEL_DIR}"

LOG_FILE="${LOG_DIR}/LLMMIRec_${LABEL}_seed${SEED}.log"
OUT_FILE="${LOG_DIR}/LLMMIRec_${LABEL}_seed${SEED}.out"
MODEL_PATH="${MODEL_DIR}/LLMMIRec_${LABEL}_seed${SEED}.pt"

START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
START_TS=$(date +%s)
echo "[START] ${START_TIME}"

python main.py \
  --model_name LLMMIRec \
  --dataset "${DATASET}" \
  --path ./data/ \
  --gpu "${GPU}" \
  --random_seed "${SEED}" \
  --emb_size 64 \
  --attn_size 64 \
  --K 4 \
  --history_max 20 \
  --item_encoder aspcf \
  --llm_emb_path "./data/${DATASET}/handled/llm_table_pca1536.pkl" \
  --semantic_rank "${SEMANTIC_RANK}" \
  --semantic_dim 32 \
  --semantic_hidden 128 \
  --complement_dim 32 \
  --tail_hidden 64 \
  --complement_hidden 64 \
  --gate_hidden 64 \
  --aspcf_gate_mode "${GATE_MODE}" \
  --lambda_relation "${LAMBDA_RELATION}" \
  --relation_sample_size 128 \
  --relation_teacher_temp 0.1 \
  --relation_student_temp 0.1 \
  --dropout 0.1 \
  --lr 0.002 \
  --l2 1e-6 \
  --batch_size 2048 \
  --eval_batch_size 256 \
  --num_neg 1 \
  --epoch "${EPOCH}" \
  --early_stop "${EARLY_STOP}" \
  --topk 5,10,20,50 \
  --metric NDCG,HR \
  --num_workers "${NUM_WORKERS}" \
  --log_file "${LOG_FILE}" \
  --model_path "${MODEL_PATH}" \
  2>&1 | tee "${OUT_FILE}"

END_TIME=$(date "+%Y-%m-%d %H:%M:%S")
END_TS=$(date +%s)
TOTAL_SECONDS=$((END_TS - START_TS))

BEST_DEV=$(grep "Best Iter(dev)" "${LOG_FILE}" | tail -n 1 | tr '\t' ' ' | sed 's/[[:space:]]\+/ /g')
TEST_AFTER=$(grep "Test After Training" "${LOG_FILE}" | tail -n 1 | tr '\t' ' ' | sed 's/[[:space:]]\+/ /g')
BEST_EPOCH=$(echo "${BEST_DEV}" | grep -oP 'Best Iter\(dev\)=\s*\K[0-9]+' || echo "NA")
PARAM_COUNT=$(grep "#params:" "${LOG_FILE}" | tail -n 1 | grep -oP '[0-9]+' || echo "NA")
STATUS="OK"
if grep -E "(loss=nan|dev=\(HR@5:nan|dev=\(HR@5:0.0000)" "${LOG_FILE}" | grep -v "check" > /dev/null 2>&1; then STATUS="NaN_CRASH"; fi

echo "========================================================="
echo "[DONE] ${START_TIME} -> ${END_TIME} (${TOTAL_SECONDS}s)"
echo "${BEST_DEV}"
echo "${TEST_AFTER}"
echo "========================================================="

echo -e "${DATASET}\tLLMMIRec\taspcf\t${GATE_MODE}\t${SEMANTIC_RANK}\t${LAMBDA_RELATION}\t${SEED}\t4\t0.002\t${START_TIME}\t${END_TIME}\t${TOTAL_SECONDS}\t${PARAM_COUNT}\t${BEST_EPOCH}\t${BEST_DEV}\t${TEST_AFTER}\t${STATUS}" >> "${SUMMARY_FILE}"
