#!/usr/bin/env bash
set -e

# =========================================================
# LLMMIRec HSDIR Phase 1 — ML-1M seed=42
#
# Usage:
#   LAMBDA_HSR=0 bash .../run_...sh [GPU_ID]          # baseline (ASPCF)
#   LAMBDA_HSR=0.01 bash .../run_...sh [GPU_ID]       # HSDIR hierarchical
#   SMOKE=1 LAMBDA_HSR=0.01 bash .../run_...sh [GPU_ID]
#
# GPU control: $1 (default 0) passed as --gpu.
# =========================================================

GPU=${1:-0}
SEED=42
DATASET="ml-1m"
LAMBDA_HSR=${LAMBDA_HSR:-0.01}
HSR_TEACHER_MODE=${HSR_TEACHER_MODE:-hierarchical}
HSR_LOSS_MODE=${HSR_LOSS_MODE:-absolute}
HSR_MARGIN=${HSR_MARGIN:-0.1}
HSR_CONFIDENCE_MODE=${HSR_CONFIDENCE_MODE:-semantic}
HSR_PAIR_MARGIN=${HSR_PAIR_MARGIN:-0.1}
BATCH_SIZE=${BATCH_SIZE:-2048}
LR=${LR:-0.002}
SEMANTIC_RANK=512
LAMBDA_RELATION=0.01

ROOT_LOG_DIR="new_log/llmmirec_hsdir_phase1/${DATASET}"
ROOT_MODEL_DIR="new_model/llmmirec_hsdir_phase1/${DATASET}"
mkdir -p "${ROOT_LOG_DIR}" "${ROOT_MODEL_DIR}"

if [ "${SMOKE}" = "1" ]; then
  EPOCH=2; EARLY_STOP=0; NUM_WORKERS=0; SMOKE_TAG="[SMOKE]"
else
  EPOCH=200; EARLY_STOP=10; NUM_WORKERS=5; SMOKE_TAG=""
fi

LABEL="hsdir_${HSR_TEACHER_MODE}_lh${LAMBDA_HSR}_${HSR_CONFIDENCE_MODE}_${HSR_LOSS_MODE}_pm${HSR_PAIR_MARGIN}_bs${BATCH_SIZE}"
SUMMARY_FILE="${ROOT_LOG_DIR}/summary_seed${SEED}.tsv"

if [ ! -f "${SUMMARY_FILE}" ]; then
  echo -e "dataset\tmodel\tteacher_mode\tlambda_hsr\tsemantic_rank\tlambda_relation\tseed\tK\tlr\tstart_time\tend_time\ttotal_seconds\tparameter_count\tbest_epoch\tbest_dev\ttest_after_training\tstatus" > "${SUMMARY_FILE}"
fi

echo "========================================================="
echo "LLMMIRec HSDIR Phase 1 — ${HSR_TEACHER_MODE} lambda=${LAMBDA_HSR} loss=${HSR_LOSS_MODE} m=${HSR_MARGIN} bs=${BATCH_SIZE} lr=${LR} ${SMOKE_TAG}"
echo "GPU=${GPU}"
echo "========================================================="

LOG_DIR="${ROOT_LOG_DIR}/${LABEL}"
MODEL_DIR="${ROOT_MODEL_DIR}/${LABEL}"
mkdir -p "${LOG_DIR}" "${MODEL_DIR}"

LOG_FILE="${LOG_DIR}/LLMMIRecHSDIR_${LABEL}_seed${SEED}.log"
OUT_FILE="${LOG_DIR}/LLMMIRecHSDIR_${LABEL}_seed${SEED}.out"
MODEL_PATH="${MODEL_DIR}/LLMMIRecHSDIR_${LABEL}_seed${SEED}.pt"

TEACHER_PATH="./data/${DATASET}/handled/llmmi_hier_proto32_8_sr${SEMANTIC_RANK}.pkl"
if [ ! -f "${TEACHER_PATH}" ]; then
  echo "ERROR: Hierarchical teacher not found: ${TEACHER_PATH}"
  echo "  Build: python tools/build_llmmi_hierarchical_teacher.py --dataset ${DATASET}"
  exit 1
fi

START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
START_TS=$(date +%s)
echo "[START] ${START_TIME}"

python main.py \
  --model_name LLMMIRecHSDIR \
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
  --lambda_relation "${LAMBDA_RELATION}" \
  --relation_sample_size 128 \
  --relation_teacher_temp 0.1 \
  --relation_student_temp 0.1 \
  --lambda_hsr "${LAMBDA_HSR}" \
  --hsr_teacher_mode "${HSR_TEACHER_MODE}" \
  --hsr_student_temp 1.0 \
  --hsr_loss_mode "${HSR_LOSS_MODE}" \
  --hsr_margin "${HSR_MARGIN}" \
  --hsr_pair_margin "${HSR_PAIR_MARGIN}" \
  --hsr_confidence_mode "${HSR_CONFIDENCE_MODE}" \
  --teacher_path "${TEACHER_PATH}" \
  --aggregation_mode base \
  --dropout 0.1 \
  --lr "${LR}" \
  --l2 1e-6 \
  --batch_size "${BATCH_SIZE}" \
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

echo "========================================================="
echo "[DONE] ${START_TIME} -> ${END_TIME} (${TOTAL_SECONDS}s)"
echo "${BEST_DEV}"
echo "${TEST_AFTER}"
echo "========================================================="

echo -e "${DATASET}\tLLMMIRecHSDIR\t${HSR_TEACHER_MODE}\t${LAMBDA_HSR}\t${SEMANTIC_RANK}\t${LAMBDA_RELATION}\t${SEED}\t4\t0.002\t${START_TIME}\t${END_TIME}\t${TOTAL_SECONDS}\t${PARAM_COUNT}\t${BEST_EPOCH}\t${BEST_DEV}\t${TEST_AFTER}\tOK" >> "${SUMMARY_FILE}"
