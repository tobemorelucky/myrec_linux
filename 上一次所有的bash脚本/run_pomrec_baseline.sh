#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

SEED=42
MAX_PROCS=2

running=0
wait_for_slot () {
  while [ "$running" -ge "$MAX_PROCS" ]; do
    wait -n
    running=$((running-1))
  done
}

MODELDIR="./model/MyModel"
mkdir -p "${MODELDIR}"

launch_one () {
  local dataset="$1"
  local lr="$2"
  local tag="$3"

  local logdir="./log/MyModel/${dataset}"
  mkdir -p "${logdir}"

  local out="${logdir}/nohup_${tag}.out"
  local log_file="${logdir}/${tag}.txt"
  local model_path="${MODELDIR}/${tag}.pt"

  echo "Launch ${tag} -> ${out}"

  # 纯 PoMRec：关闭所有你新增/不该启用的模块
  nohup python main.py \
    --model_name MyModel --dataset "${dataset}" \
    --lr "${lr}" --l2 1e-06 \
    --batch_size 256 --eval_batch_size 256 \
    --epoch 200 --early_stop 10 \
    --num_neg 1 --dropout 0 --num_workers 5 \
    --random_seed "${SEED}" --load 0 \
    --history_max 20 \
    --K 3 --prompt_num 4 --lamb 3.0 --emb_size 64 --attn_size 8 --n_layers 1 \
    \
    --use_llmemb 0 --llm_fuse 0 \
    --gamma_trainable 0 \
    --use_emile 0 --lambda_ipd 0 --lambda_ilr 0 \
    --use_logic_aggr 0 --lambda_logic_aggr 0 \
    --use_logic_denoise 0 \
    --use_e2i_logic 0 --lambda_logic 0 \
    \
    --log_file "${log_file}" \
    --model_path "${model_path}" \
    > "${out}" 2>&1 &

  running=$((running+1))
  sleep 0.2
}

#######################################
# ml-1m baseline（按你常用 lr=0.001）
#######################################
wait_for_slot
launch_one "ml-1m" "0.001" "pomrec_baseline_ml1m_s${SEED}"

#######################################
# beauty baseline（按你常用 lr=0.002）
#######################################
wait_for_slot
launch_one "beauty" "0.002" "pomrec_baseline_beauty_s${SEED}"

wait
echo "All PoMRec baseline jobs finished."