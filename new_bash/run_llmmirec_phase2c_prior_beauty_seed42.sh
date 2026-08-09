#!/usr/bin/env bash
set -e

# =========================================================
# LLMMIRec Phase 2C — Prototype Routing Prior (Beauty seed=42)
#
# Usage:
#   PRIOR_STRENGTH=1.0 bash .../run_...sh [GPU_ID]
#   SMOKE=1 PRIOR_STRENGTH=1.0 bash .../run_...sh [GPU_ID]
#
# GPU control: $1 (default 0) passed as --gpu.
# =========================================================

GPU=${1:-0}
SEED=42
DATASET="beauty"
PRIOR_STRENGTH=${PRIOR_STRENGTH:-1.0}
SEMANTIC_RANK=512
LAMBDA_RELATION=0.01
PROTOTYPE_NUM=32

ROOT_LOG_DIR="new_log/llmmirec_phase2c_prior/${DATASET}"
ROOT_MODEL_DIR="new_model/llmmirec_phase2c_prior/${DATASET}"
mkdir -p "${ROOT_LOG_DIR}" "${ROOT_MODEL_DIR}"

if [ "${SMOKE}" = "1" ]; then
  EPOCH=2; EARLY_STOP=0; NUM_WORKERS=0; SMOKE_TAG="[SMOKE]"
else
  EPOCH=200; EARLY_STOP=10; NUM_WORKERS=5; SMOKE_TAG=""
fi

LABEL="proto_prior${PRIOR_STRENGTH}"
SUMMARY_FILE="${ROOT_LOG_DIR}/summary_seed${SEED}.tsv"

if [ ! -f "${SUMMARY_FILE}" ]; then
  echo -e "dataset\tmodel\titem_encoder\tquery_mode\tcollab_calibration\tprior_strength\tsemantic_rank\tlambda_relation\tprototype_num\tseed\tK\tlr\tstart_time\tend_time\ttotal_seconds\tparameter_count\tbest_epoch\tbest_dev\ttest_after_training\tstatus" > "${SUMMARY_FILE}"
fi

echo "========================================================="
echo "LLMMIRec Phase 2C — prior_strength=${PRIOR_STRENGTH} ${SMOKE_TAG}"
echo "GPU=${GPU}"
echo "========================================================="

LOG_DIR="${ROOT_LOG_DIR}/${LABEL}"
MODEL_DIR="${ROOT_MODEL_DIR}/${LABEL}"
mkdir -p "${LOG_DIR}" "${MODEL_DIR}"

LOG_FILE="${LOG_DIR}/LLMMIRecCHIR_${LABEL}_seed${SEED}.log"
OUT_FILE="${LOG_DIR}/LLMMIRecCHIR_${LABEL}_seed${SEED}.out"
MODEL_PATH="${MODEL_DIR}/LLMMIRecCHIR_${LABEL}_seed${SEED}.pt"

PROTO_PATH="./data/${DATASET}/handled/llmmi_proto${PROTOTYPE_NUM}_sr${SEMANTIC_RANK}.pkl"
if [ ! -f "${PROTO_PATH}" ]; then
  echo "ERROR: Prototype file not found: ${PROTO_PATH}"
  exit 1
fi

START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
START_TS=$(date +%s)
echo "[START] ${START_TIME}"

python main.py \
  --model_name LLMMIRecCHIR \
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
  --aspcf_gate_mode basic \
  --interest_query_mode prototype \
  --prototype_path "${PROTO_PATH}" \
  --collab_calibration prototype \
  --prototype_prior_strength "${PRIOR_STRENGTH}" \
  --lambda_relation "${LAMBDA_RELATION}" \
  --relation_sample_size 128 \
  --relation_teacher_temp 0.1 \
  --relation_student_temp 0.1 \
  --dropout 0.1 \
  --lr 0.008 \
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

echo "========================================================="
echo "[DONE] ${START_TIME} -> ${END_TIME} (${TOTAL_SECONDS}s)"
echo "${BEST_DEV}"
echo "${TEST_AFTER}"
echo "========================================================="

echo -e "${DATASET}\tLLMMIRecCHIR\taspcf\tprototype\tprototype\t${PRIOR_STRENGTH}\t${SEMANTIC_RANK}\t${LAMBDA_RELATION}\t${PROTOTYPE_NUM}\t${SEED}\t4\t0.008\t${START_TIME}\t${END_TIME}\t${TOTAL_SECONDS}\t${PARAM_COUNT}\t${BEST_EPOCH}\t${BEST_DEV}\t${TEST_AFTER}\t${STATUS}" >> "${SUMMARY_FILE}"
