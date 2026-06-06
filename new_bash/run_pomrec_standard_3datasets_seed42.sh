#!/usr/bin/env bash
set -e

# =========================================================
# Sequential PoMRec standard runs
#
# Usage:
#   bash new_bash/run_pomrec_standard_3datasets_seed42.sh 0
#
# Background:
#   nohup bash new_bash/run_pomrec_standard_3datasets_seed42.sh 0 \
#     > new_log/pomrec_standard/master_seed42.out 2>&1 &
#
# Args:
#   $1: GPU id, default 0
# =========================================================

GPU=${1:-0}
SEED=42

# 如果你的第三个数据集目录名不是 toys，在这里改
DATASET_BEAUTY="beauty"
DATASET_ML1M="ml-1m"
DATASET_TOYS="toys"

ROOT_LOG_DIR="new_log/pomrec_standard"
ROOT_MODEL_DIR="new_model/pomrec_standard"

mkdir -p "${ROOT_LOG_DIR}"
mkdir -p "${ROOT_MODEL_DIR}"

SUMMARY_FILE="${ROOT_LOG_DIR}/summary_seed${SEED}.tsv"

echo -e "dataset\tseed\tstart_time\tend_time\ttotal_seconds\tepoch_count\tavg_epoch_train_seconds\tbest_dev\ttest_after_training" > "${SUMMARY_FILE}"

echo "========================================================="
echo "Run original PoMRec standard experiments sequentially"
echo "GPU=${GPU}"
echo "SEED=${SEED}"
echo "SUMMARY=${SUMMARY_FILE}"
echo "========================================================="


run_pomrec() {
  local DATASET=$1
  local EMB_SIZE=$2
  local ATTN_SIZE=$3
  local K=$4
  local PROMPT_NUM=$5
  local N_LAYERS=$6
  local LAMB=$7
  local HISTORY_MAX=$8
  local LR=$9
  local L2=${10}
  local BATCH_SIZE=${11}
  local EVAL_BATCH_SIZE=${12}
  local EPOCH=${13}
  local EARLY_STOP=${14}
  local NUM_NEG=${15}
  local NUM_WORKERS=${16}

  local LOG_DIR="${ROOT_LOG_DIR}/${DATASET}"
  local MODEL_DIR="${ROOT_MODEL_DIR}/${DATASET}"

  mkdir -p "${LOG_DIR}"
  mkdir -p "${MODEL_DIR}"

  local LOG_FILE="${LOG_DIR}/PoMRec_seed${SEED}.log"
  local OUT_FILE="${LOG_DIR}/PoMRec_seed${SEED}.out"
  local MODEL_PATH="${MODEL_DIR}/PoMRec_seed${SEED}.pt"

  local START_TIME
  local END_TIME
  local START_TS
  local END_TS
  local TOTAL_SECONDS

  START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
  START_TS=$(date +%s)

  echo ""
  echo "========================================================="
  echo "[START] dataset=${DATASET}, seed=${SEED}"
  echo "time=${START_TIME}"
  echo "log=${LOG_FILE}"
  echo "out=${OUT_FILE}"
  echo "model=${MODEL_PATH}"
  echo "========================================================="

  python main.py \
    --model_name PoMRec \
    --dataset "${DATASET}" \
    --path ./data/ \
    --gpu "${GPU}" \
    --random_seed "${SEED}" \
    --emb_size "${EMB_SIZE}" \
    --attn_size "${ATTN_SIZE}" \
    --K "${K}" \
    --prompt_num "${PROMPT_NUM}" \
    --n_layers "${N_LAYERS}" \
    --lamb "${LAMB}" \
    --history_max "${HISTORY_MAX}" \
    --lr "${LR}" \
    --l2 "${L2}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --num_neg "${NUM_NEG}" \
    --epoch "${EPOCH}" \
    --early_stop "${EARLY_STOP}" \
    --num_workers "${NUM_WORKERS}" \
    --dropout 0 \
    --log_file "${LOG_FILE}" \
    --model_path "${MODEL_PATH}" \
    2>&1 | tee "${OUT_FILE}"

  END_TIME=$(date "+%Y-%m-%d %H:%M:%S")
  END_TS=$(date +%s)
  TOTAL_SECONDS=$((END_TS - START_TS))

  # 统计实际训练了多少轮
  local EPOCH_COUNT
  EPOCH_COUNT=$(grep -E "Epoch[[:space:]]+[0-9]+" "${OUT_FILE}" | wc -l | awk '{print $1}')

  # 从日志中提取每轮训练耗时，即 Epoch 行中 loss 后第一个 [x s]
  local AVG_EPOCH_TRAIN_SECONDS
  AVG_EPOCH_TRAIN_SECONDS=$(
    grep -E "Epoch[[:space:]]+[0-9]+" "${OUT_FILE}" \
      | sed -E 's/.*loss=[0-9.]+[[:space:]]+\[([0-9.]+)[[:space:]]s\].*/\1/' \
      | awk '{sum+=$1; n+=1} END {if (n>0) printf "%.3f", sum/n; else printf "NA"}'
  )

  local BEST_DEV
  local TEST_AFTER
  BEST_DEV=$(grep "Best Iter(dev)" "${LOG_FILE}" | tail -n 1 | tr '\t' ' ' | sed 's/[[:space:]]\+/ /g')
  TEST_AFTER=$(grep "Test After Training" "${LOG_FILE}" | tail -n 1 | tr '\t' ' ' | sed 's/[[:space:]]\+/ /g')

  echo "========================================================="
  echo "[DONE] dataset=${DATASET}, seed=${SEED}"
  echo "start=${START_TIME}"
  echo "end=${END_TIME}"
  echo "total_seconds=${TOTAL_SECONDS}"
  echo "epoch_count=${EPOCH_COUNT}"
  echo "avg_epoch_train_seconds=${AVG_EPOCH_TRAIN_SECONDS}"
  echo "${BEST_DEV}"
  echo "${TEST_AFTER}"
  echo "========================================================="

  echo -e "${DATASET}\t${SEED}\t${START_TIME}\t${END_TIME}\t${TOTAL_SECONDS}\t${EPOCH_COUNT}\t${AVG_EPOCH_TRAIN_SECONDS}\t${BEST_DEV}\t${TEST_AFTER}" >> "${SUMMARY_FILE}"
}


# =========================================================
# 1. Beauty
# 你之前记录的 PoMRec Beauty 最佳/常用参数：
# K=4, attn_size=8, emb_size=64, prompt_num=3,
# n_layers=2, lamb=4.0, history_max=20, lr=0.002
# =========================================================
run_pomrec \
  "${DATASET_BEAUTY}" \
  64 \
  8 \
  4 \
  3 \
  2 \
  4.0 \
  20 \
  0.002 \
  1e-6 \
  256 \
  256 \
  200 \
  10 \
  1 \
  5


# =========================================================
# 2. ml-1m
# 按你之前脚本常用参数：
# K=2, prompt_num=3, lamb=1.0, lr=0.001
# 如果你记录的最佳不同，改这里即可。
# =========================================================
run_pomrec \
  "${DATASET_ML1M}" \
  64 \
  8 \
  2 \
  3 \
  2 \
  1.0 \
  20 \
  0.001 \
  1e-6 \
  256 \
  256 \
  200 \
  10 \
  1 \
  5


# =========================================================
# 3. Toys
# 这里先按 Beauty 风格参数给出：
# K=4, prompt_num=3, lamb=4.0, lr=0.002
# 如果你记录的 Toys 最佳参数不同，改这里。
# =========================================================
run_pomrec \
  "${DATASET_TOYS}" \
  64 \
  8 \
  4 \
  3 \
  2 \
  4.0 \
  20 \
  0.002 \
  1e-6 \
  256 \
  256 \
  200 \
  10 \
  1 \
  5


echo ""
echo "========================================================="
echo "All PoMRec standard experiments finished."
echo "Summary file:"
echo "${SUMMARY_FILE}"
echo "View summary:"
echo "cat ${SUMMARY_FILE}"
echo "========================================================="