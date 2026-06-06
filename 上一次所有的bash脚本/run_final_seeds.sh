#!/bin/bash
set -euo pipefail

MAX_PROCS=2
dataset=beauty
seeds=(0 1 2 3 40 41 42 43 44)

mkdir -p log/beauty

running=0

launch_one () {
  local seed="$1"
  local log_file="log/beauty/final_g0.1_a0.001_tau0.2_seed${seed}.log"

  echo "Launch seed=${seed} -> ${log_file}"

  nohup python main.py \
    --model_name PoMRec --dataset ${dataset} \
    --K 4 --attn_size 8 --emb_size 64 --prompt_num 3 --n_layers 2 --lamb 4.0 --history_max 20 \
    --lr 0.002 --l2 1e-06 --batch_size 256 --eval_batch_size 256 \
    --epoch 200 --early_stop 10 --num_neg 1 --dropout 0 --num_workers 5 \
    --random_seed ${seed} \
    --use_llmemb 1 \
    --llm_emb_path data/beauty/handled/llm_table_pca1536.pkl \
    --srs_emb_path data/beauty/handled/itm_emb_pomrec.pkl \
    --alpha 0.001 --tau 0.2 --rat_alpha_warmup_steps 5000 --align_on pos \
    --llm_fuse 1 --gamma_init 0.1 --gamma_trainable 0 \
    --init_ckpt ./model/PoMRec/PoMRec__beauty__42__lr=0.002__l2=1e-06.pt --init_strict 0 \
    > "${log_file}" 2>&1 &
}

for seed in "${seeds[@]}"; do
  while [ "$running" -ge "$MAX_PROCS" ]; do
    wait -n
    running=$((running-1))
  done

  launch_one "$seed"
  running=$((running+1))

  # 可选：小延迟，避免瞬间拉起太猛
  sleep 0.2
done

wait
echo "All seed runs finished."
