#!/usr/bin/env bash
set -euo pipefail

MAX_PROCS=2

DATASET="ml-1m"
LLM_PKL="./data/ml-1m/handled/llm_table_pca1536.pkl"
SRS_PKL="./data/ml-1m/handled/itm_emb_pomrec.pkl"
LOGDIR="./log/ml-1m"
mkdir -p "${LOGDIR}"

SEED=42
INIT_CKPT="./model/PoMRec/PoMRec__ml-1m__1__lr=0.001__l2=1e-06.pt"

running=0

wait_for_slot () {
  while [ "$running" -ge "$MAX_PROCS" ]; do
    wait -n
    running=$((running-1))
  done
}

launch_job () {
  local name="$1"
  shift
  echo "Launch: ${name}"
  nohup python main.py "$@" > "${LOGDIR}/${name}.log" 2>&1 &
  running=$((running+1))
  sleep 0.2
}

# 固定基础点（你说的默认最强点）
BASE_G=0.05
BASE_A=1e-3
BASE_T=0.5

# ========== Sweep 1: 固定 (g=0.05, tau=0.5)，扫 alpha ==========
for a in 5e-4 1e-3 2e-3; do
  wait_for_slot
  name="S1_alpha_g${BASE_G}_a${a}_tau${BASE_T}_s${SEED}"
  launch_job "${name}" \
    --model_name PoMRec --dataset "${DATASET}" \
    --use_llmemb 1 --llm_fuse 1 \
    --llm_emb_path "${LLM_PKL}" --srs_emb_path "${SRS_PKL}" \
    --gamma_init "${BASE_G}" --gamma_trainable 0 \
    --alpha "${a}" --tau "${BASE_T}" --rat_alpha_warmup_steps 5000 \
    --align_on pos --random_seed "${SEED}" \
    --init_ckpt "${INIT_CKPT}" --init_strict 0
done

# ========== Sweep 2: 固定 (g=0.05, a=1e-3)，扫 tau ==========
for t in 0.3 0.5 0.7; do
  wait_for_slot
  name="S2_tau_g${BASE_G}_a${BASE_A}_tau${t}_s${SEED}"
  launch_job "${name}" \
    --model_name PoMRec --dataset "${DATASET}" \
    --use_llmemb 1 --llm_fuse 1 \
    --llm_emb_path "${LLM_PKL}" --srs_emb_path "${SRS_PKL}" \
    --gamma_init "${BASE_G}" --gamma_trainable 0 \
    --alpha "${BASE_A}" --tau "${t}" --rat_alpha_warmup_steps 5000 \
    --align_on pos --random_seed "${SEED}" \
    --init_ckpt "${INIT_CKPT}" --init_strict 0
done

# ========== Sweep 3: 固定 (a=1e-3, tau=0.5)，扫 gamma ==========
for g in 0.03 0.05 0.07; do
  wait_for_slot
  name="S3_gamma_g${g}_a${BASE_A}_tau${BASE_T}_s${SEED}"
  launch_job "${name}" \
    --model_name PoMRec --dataset "${DATASET}" \
    --use_llmemb 1 --llm_fuse 1 \
    --llm_emb_path "${LLM_PKL}" --srs_emb_path "${SRS_PKL}" \
    --gamma_init "${g}" --gamma_trainable 0 \
    --alpha "${BASE_A}" --tau "${BASE_T}" --rat_alpha_warmup_steps 5000 \
    --align_on pos --random_seed "${SEED}" \
    --init_ckpt "${INIT_CKPT}" --init_strict 0
done

# 等最后一批完成
wait
echo "All 9 single-axis runs finished. Logs in ${LOGDIR}"