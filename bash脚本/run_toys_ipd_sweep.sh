#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

# 并发：建议先 1（稳），你要 2 就改成 2
MAX_PROCS=2

DATASET="toys"
LOGDIR="./log/MyModel/${DATASET}"
mkdir -p "${LOGDIR}"

LLM_PKL="./data/${DATASET}/handled/llm_table_pca1536.pkl"
SRS_PKL="./data/${DATASET}/handled/itm_emb_pomrec1.pkl"
INIT_CKPT="./model/PoMRec/toys__42__lr=0.001__l2=1e-06__lamb=3.8__history_max=20.pt"

SEED=1

# ========== 固定：你 toys 的 LLMemb 最佳配置（不要改） ==========
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
NUM_WORKERS=5

WARMUP=5000
ALIGN_ON="pos"

BASE_A=1e-3
BASE_G=0.05
BASE_T=0.3
BASE_GT=1   # 关键：你 best 是 gamma_trainable=1

# ========== 第二模块（IPD）要扫的超参：少量但够用 ==========
LAMBDAS=(0.01 0.02 0.05)   # 建议先这3个
MARGINS=(0.05 0.10)        # 先2个
EMILE_USE_FUSED=0          # 固定（稳）

running=0

wait_for_slot () {
  while [ "$running" -ge "$MAX_PROCS" ]; do
    if wait -n; then
      :
    else
      echo "[WARN] A job exited non-zero, continue..."
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
# IPD sweep（只加第二模块；第三模块不加）
# =========================
for lam in "${LAMBDAS[@]}"; do
  for m in "${MARGINS[@]}"; do
    wait_for_slot
    name="IPD_l${lam}_m${m}_fused${EMILE_USE_FUSED}_gT${BASE_GT}_t${BASE_T}_g${BASE_G}_a${BASE_A}_s${SEED}"

    launch_job "${name}" \
      --model_name PoMRec --dataset "${DATASET}" \
      --K "${K}" --attn_size "${ATTN_SIZE}" --emb_size "${EMB_SIZE}" \
      --prompt_num "${PROMPT_NUM}" --n_layers "${N_LAYERS}" --lamb "${LAMB}" --history_max "${HISTORY_MAX}" \
      --lr "${LR}" --l2 "${L2}" --batch_size "${BATCH_SIZE}" --eval_batch_size "${EVAL_BATCH_SIZE}" \
      --epoch "${EPOCH}" --early_stop "${EARLY_STOP}" --num_neg "${NUM_NEG}" --dropout "${DROPOUT}" --num_workers "${NUM_WORKERS}" \
      --random_seed "${SEED}" \
      --use_llmemb 1 --llm_fuse 1 \
      --llm_emb_path "${LLM_PKL}" --srs_emb_path "${SRS_PKL}" \
      --gamma_init "${BASE_G}" --gamma_trainable "${BASE_GT}" \
      --alpha "${BASE_A}" --tau "${BASE_T}" --rat_alpha_warmup_steps "${WARMUP}" --align_on "${ALIGN_ON}" \
      --init_ckpt "${INIT_CKPT}" --init_strict 0 \
      --use_emile 1 --lambda_ipd "${lam}" --lambda_ilr 0 --ipd_margin "${m}" --emile_use_fused_itememb "${EMILE_USE_FUSED}" \
      --use_logic_denoise 0
  done
done

wait || true
echo "IPD sweep finished. Logs in ${LOGDIR}"