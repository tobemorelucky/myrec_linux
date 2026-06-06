#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
MAX_PROCS=1

DATASET="toys"
LOGDIR="./log/${DATASET}"
mkdir -p "${LOGDIR}"

LLM_PKL="./data/${DATASET}/handled/llm_table_pca1536.pkl"
SRS_PKL="./data/${DATASET}/handled/itm_emb_pomrec1.pkl"
INIT_CKPT="./model/PoMRec/toys__42__lr=0.001__l2=1e-06__lamb=3.8__history_max=20.pt"

SEED=1

# 固定已确定参数
K=3
PROMPT_NUM=4
LAMB=3.8
LR=0.001

EMB_SIZE=64
ATTN_SIZE=8
N_LAYERS=2
HISTORY_MAX=20

L2=1e-06
BATCH_SIZE=256
EVAL_BATCH_SIZE=256
EPOCH=200
EARLY_STOP=10
NUM_NEG=1
DROPOUT=0

# ✅ 最稳：避免 eval/predict 的 NVML/fork 问题
NUM_WORKERS=5

WARMUP=5000
ALIGN_ON="pos"

# 根据你最新 A 组结果：alpha=1e-3 最好
BASE_A=1e-3
BASE_G=0.05

running=0

wait_for_slot () {
  while [ "$running" -ge "$MAX_PROCS" ]; do
    # ✅ 关键：wait -n 失败也不要让 set -e 杀掉脚本
    if wait -n; then
      :
    else
      echo "[WARN] A job exited non-zero, continue launching remaining jobs..."
    fi
    running=$((running-1))
  done
}

is_done () {
  local log_file="$1"
  [ -f "$log_file" ] && grep -q "END:" "$log_file"
}

launch_job () {
  local name="$1"
  shift
  local log_file="${LOGDIR}/${name}.log"

  if is_done "$log_file"; then
    echo "Skip finished: ${name}"
    return
  fi

  echo "Launch: ${name} -> ${log_file}"
  nohup python main.py "$@" > "${log_file}" 2>&1 &
  running=$((running+1))
  sleep 0.2
}

# =========================
# B 组：固定 gamma(0.05, trainable=0)，补跑 tau
# =========================
for t in 0.3 0.5 0.7; do
  wait_for_slot
  name="B_tau_t${t}_g${BASE_G}_gt0_a${BASE_A}_s${SEED}"
  launch_job "${name}" \
    --model_name PoMRec --dataset "${DATASET}" \
    --K "${K}" --attn_size "${ATTN_SIZE}" --emb_size "${EMB_SIZE}" \
    --prompt_num "${PROMPT_NUM}" --n_layers "${N_LAYERS}" --lamb "${LAMB}" --history_max "${HISTORY_MAX}" \
    --lr "${LR}" --l2 "${L2}" --batch_size "${BATCH_SIZE}" --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --epoch "${EPOCH}" --early_stop "${EARLY_STOP}" --num_neg "${NUM_NEG}" --dropout "${DROPOUT}" --num_workers "${NUM_WORKERS}" \
    --random_seed "${SEED}" \
    --use_llmemb 1 --llm_fuse 1 \
    --llm_emb_path "${LLM_PKL}" --srs_emb_path "${SRS_PKL}" \
    --gamma_init "${BASE_G}" --gamma_trainable 0 \
    --alpha "${BASE_A}" --tau "${t}" --rat_alpha_warmup_steps "${WARMUP}" --align_on "${ALIGN_ON}" \
    --init_ckpt "${INIT_CKPT}" --init_strict 0
done

# =========================
# C 组：gamma 可训练（trainable=1），只跑 2 个 tau（少量组）
# =========================
for t in 0.5 0.3; do
  wait_for_slot
  name="C_gTrain_t${t}_g${BASE_G}_gt1_a${BASE_A}_s${SEED}"
  launch_job "${name}" \
    --model_name PoMRec --dataset "${DATASET}" \
    --K "${K}" --attn_size "${ATTN_SIZE}" --emb_size "${EMB_SIZE}" \
    --prompt_num "${PROMPT_NUM}" --n_layers "${N_LAYERS}" --lamb "${LAMB}" --history_max "${HISTORY_MAX}" \
    --lr "${LR}" --l2 "${L2}" --batch_size "${BATCH_SIZE}" --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --epoch "${EPOCH}" --early_stop "${EARLY_STOP}" --num_neg "${NUM_NEG}" --dropout "${DROPOUT}" --num_workers "${NUM_WORKERS}" \
    --random_seed "${SEED}" \
    --use_llmemb 1 --llm_fuse 1 \
    --llm_emb_path "${LLM_PKL}" --srs_emb_path "${SRS_PKL}" \
    --gamma_init "${BASE_G}" --gamma_trainable 1 \
    --alpha "${BASE_A}" --tau "${t}" --rat_alpha_warmup_steps "${WARMUP}" --align_on "${ALIGN_ON}" \
    --init_ckpt "${INIT_CKPT}" --init_strict 0
done

wait || true
echo "BC reruns finished. Logs in ${LOGDIR}"
