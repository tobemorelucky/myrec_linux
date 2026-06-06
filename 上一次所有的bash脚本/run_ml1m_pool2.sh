#!/usr/bin/env bash
set -euo pipefail

MAX_PROCS=2

DATASET="ml-1m"
LLM_PKL="./data/ml-1m/handled/llm_table_pca1536.pkl"
SRS_PKL="./data/ml-1m/handled/itm_emb_pomrec.pkl"
LOGDIR="./log/ml-1m"
mkdir -p "${LOGDIR}"

SEED=42
GAMMAS=(0.02 0.05 0.1 0.2)

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

  # 可选：避免瞬间拉起太猛
  sleep 0.2
}

# ========== 1) align-only ==========
for a in 1e-4 3e-4 1e-3; do
  for t in 0.2 0.5; do
    for w in 0 5000; do
      wait_for_slot
      name="A_alignOnly_a${a}_tau${t}_w${w}_s${SEED}"

      launch_job "${name}" \
        --model_name PoMRec --dataset "${DATASET}" \
        --use_llmemb 1 --llm_fuse 0 \
        --llm_emb_path "${LLM_PKL}" --srs_emb_path "${SRS_PKL}" \
        --alpha "${a}" --tau "${t}" --rat_alpha_warmup_steps "${w}" \
        --align_on pos --random_seed "${SEED}" \
        --init_ckpt "${INIT_CKPT}" --init_strict 0
    done
  done
done

# ========== 2) align+fuse (gamma fixed) ==========
for g in "${GAMMAS[@]}"; do
  for a in 1e-4 3e-4 1e-3; do
    for t in 0.2 0.5; do
      wait_for_slot
      name="B_fuseFix_g${g}_a${a}_tau${t}_s${SEED}"

      launch_job "${name}" \
        --model_name PoMRec --dataset "${DATASET}" \
        --use_llmemb 1 --llm_fuse 1 \
        --llm_emb_path "${LLM_PKL}" --srs_emb_path "${SRS_PKL}" \
        --gamma_init "${g}" --gamma_trainable 0 \
        --alpha "${a}" --tau "${t}" --rat_alpha_warmup_steps 5000 \
        --align_on pos --random_seed "${SEED}" \
        --init_ckpt "${INIT_CKPT}" --init_strict 0
    done
  done
done

# 等最后一批完成
wait
echo "All jobs finished. Logs in ${LOGDIR}"
