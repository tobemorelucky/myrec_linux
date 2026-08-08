#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# LLMMIRec Phase 0 — single batch_size / lr full training
#
# Usage:
# bash new_bash/run_llmmirec_bs_sweep.sh [GPU] [DATASET] [BATCH_SIZE] [LR]
#
# Defaults:
# GPU        = 1
# DATASET    = ml-1m
# BATCH_SIZE = 256
# LR         = 0.001
#
# Examples:
# bash new_bash/run_llmmirec_bs_sweep.sh
# bash new_bash/run_llmmirec_bs_sweep.sh 1 ml-1m 1024 0.001
# bash new_bash/run_llmmirec_bs_sweep.sh 1 ml-1m 2048 0.0015
# bash new_bash/run_llmmirec_bs_sweep.sh 1 ml-1m 2048 0.002
# bash new_bash/run_llmmirec_bs_sweep.sh 0 beauty 512 0.001
# bash new_bash/run_llmmirec_bs_sweep.sh 1 toys 1024 0.001
# =========================================================

GPU=${1:-1}
DATASET=${2:-ml-1m}
BATCH_SIZE=${3:-256}
LR=${4:-0.001}

SEED=42
NUM_WORKERS=5
EVAL_BATCH_SIZE=256

# ---- validate dataset ----
case "${DATASET}" in
    ml-1m|beauty|toys)
        ;;
    *)
        echo "ERROR: DATASET must be ml-1m, beauty, or toys, got '${DATASET}'"
        exit 1
        ;;
esac

# ---- validate batch size ----
if ! [[ "${BATCH_SIZE}" =~ ^[0-9]+$ ]] || [ "${BATCH_SIZE}" -le 0 ]; then
    echo "ERROR: BATCH_SIZE must be a positive integer, got '${BATCH_SIZE}'"
    exit 1
fi

# ---- validate learning rate ----
if ! [[ "${LR}" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
    echo "ERROR: LR must be a positive numeric value, got '${LR}'"
    exit 1
fi

ROOT_LOG_DIR="new_log/llmmirec_bs_sweep/${DATASET}"
ROOT_MODEL_DIR="new_model/llmmirec_bs_sweep/${DATASET}"

mkdir -p "${ROOT_LOG_DIR}"
mkdir -p "${ROOT_MODEL_DIR}"

SUMMARY_FILE="${ROOT_LOG_DIR}/summary_bs_sweep.tsv"

# 只在 summary 不存在或为空时写表头，避免覆盖历史结果
if [ ! -s "${SUMMARY_FILE}" ]; then
    echo -e "dataset\tbatch_size\tnum_workers\titem_encoder\tseed\tlr\tstart_time\tend_time\ttotal_seconds\tparameter_count\tbest_epoch\tbest_dev\ttest_after_training\tstatus" \
        > "${SUMMARY_FILE}"
fi

# 路径中不用小数点，例如 0.0015 -> 0p0015
LR_TAG="${LR//./p}"

echo "========================================================="
echo "LLMMIRec Full Training"
echo "  GPU        = ${GPU}"
echo "  Dataset    = ${DATASET}"
echo "  Seed       = ${SEED}"
echo "  Batch size = ${BATCH_SIZE}"
echo "  LR         = ${LR}"
echo "  Workers    = ${NUM_WORKERS}"
echo "  Eval batch = ${EVAL_BATCH_SIZE}"
echo "  Summary    = ${SUMMARY_FILE}"
echo "========================================================="

run_one() {
    local BS=$1
    local NW=$2
    local CUR_LR=$3

    local RUN_TAG="id_seed${SEED}_bs${BS}_lr${LR_TAG}"

    local LOG_DIR="${ROOT_LOG_DIR}/${RUN_TAG}"
    local MODEL_DIR="${ROOT_MODEL_DIR}/${RUN_TAG}"

    mkdir -p "${LOG_DIR}"
    mkdir -p "${MODEL_DIR}"

    local LOG_FILE="${LOG_DIR}/LLMMIRec_${RUN_TAG}.log"
    local OUT_FILE="${LOG_DIR}/LLMMIRec_${RUN_TAG}.out"
    local MODEL_PATH="${MODEL_DIR}/LLMMIRec_${RUN_TAG}.pt"

    local START_TIME
    local START_TS
    START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
    START_TS=$(date +%s)

    echo ""
    echo "========================================================="
    echo "[START] bs=${BS}, lr=${CUR_LR}, nw=${NW}, dataset=${DATASET}, seed=${SEED}"
    echo "time=${START_TIME}"
    echo "log=${LOG_FILE}"
    echo "model=${MODEL_PATH}"
    echo "========================================================="

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
        --item_encoder id \
        --llm_emb_path "" \
        --adapter_hidden 256 \
        --adapter_activation gelu \
        --adapter_use_ln 0 \
        --gamma_init 0.1 \
        --gamma_trainable 0 \
        --dropout 0.1 \
        --lr "${CUR_LR}" \
        --l2 1e-6 \
        --batch_size "${BS}" \
        --eval_batch_size "${EVAL_BATCH_SIZE}" \
        --num_neg 1 \
        --epoch 200 \
        --early_stop 10 \
        --topk 5,10,20,50 \
        --metric NDCG,HR \
        --num_workers "${NW}" \
        --log_file "${LOG_FILE}" \
        --model_path "${MODEL_PATH}" \
        2>&1 | tee "${OUT_FILE}"

    local END_TIME
    local END_TS
    END_TIME=$(date "+%Y-%m-%d %H:%M:%S")
    END_TS=$(date +%s)

    local TOTAL_SECONDS=$((END_TS - START_TS))

    local BEST_DEV
    BEST_DEV=$(grep "Best Iter(dev)" "${LOG_FILE}" | tail -n 1 \
        | tr '\t' ' ' | sed 's/[[:space:]]\+/ /g') || BEST_DEV="NA"

    local TEST_AFTER
    TEST_AFTER=$(grep "Test After Training" "${LOG_FILE}" | tail -n 1 \
        | tr '\t' ' ' | sed 's/[[:space:]]\+/ /g') || TEST_AFTER="NA"

    local BEST_EPOCH
    BEST_EPOCH=$(echo "${BEST_DEV}" \
        | grep -oP 'Best Iter\(dev\)=\s*\K[0-9]+') || BEST_EPOCH="NA"

    local PARAM_COUNT
    PARAM_COUNT=$(grep "#params:" "${LOG_FILE}" | tail -n 1 \
        | grep -oP '[0-9]+') || PARAM_COUNT="NA"

    local STATUS="OK"

    if grep -Eiq '(^|[^[:alnum:]_])(nan|[-+]?inf(inity)?)([^[:alnum:]_]|$)' "${LOG_FILE}"; then
        STATUS="NaN_WARN"
    fi

    echo "========================================================="
    echo "[DONE] bs=${BS}, lr=${CUR_LR}, dataset=${DATASET}, seed=${SEED}"
    echo "start=${START_TIME}"
    echo "end=${END_TIME}"
    echo "total_seconds=${TOTAL_SECONDS}"
    echo "${BEST_DEV}"
    echo "${TEST_AFTER}"
    echo "========================================================="

    echo -e "${DATASET}\t${BS}\t${NW}\tid\t${SEED}\t${CUR_LR}\t${START_TIME}\t${END_TIME}\t${TOTAL_SECONDS}\t${PARAM_COUNT}\t${BEST_EPOCH}\t${BEST_DEV}\t${TEST_AFTER}\t${STATUS}" \
        >> "${SUMMARY_FILE}"
}

run_one "${BATCH_SIZE}" "${NUM_WORKERS}" "${LR}"

echo ""
echo "========================================================="
echo "Training finished."
echo "Dataset    : ${DATASET}"
echo "Batch size : ${BATCH_SIZE}"
echo "LR         : ${LR}"
echo "Summary    : ${SUMMARY_FILE}"
echo "========================================================="