#!/bin/bash
set -euo pipefail

MAX_PROCS=1
dataset=beauty
base_ckpt=./model/PoMRec/PoMRec__beauty__42__lr=0.002__l2=1e-06.pt

mkdir -p log/beauty

# 固定物理 GPU0
export CUDA_VISIBLE_DEVICES=0

running=0

launch_one () {
  local seed="$1"
  local gamma="$2"
  local alpha="$3"
  local log_file="log/beauty/llmemb_ws_seed${seed}_g${gamma}_a${alpha}.log"

  echo "[GPU0] launch seed=${seed} gamma=${gamma} alpha=${alpha} -> ${log_file}"

  python main.py \
    --model_name PoMRec --dataset ${dataset} \
    --K 4 --attn_size 8 --emb_size 64 --prompt_num 3 --n_layers 2 --lamb 4.0 \
    --history_max 20 --lr 0.002 --l2 1e-06 --batch_size 256 --eval_batch_size 256 \
    --epoch 200 --early_stop 10 --num_neg 1 --dropout 0 --num_workers 5 \
    --random_seed ${seed} \
    --use_llmemb 1 \
    --llm_emb_path data/beauty/handled/llm_table_pca1536.pkl \
    --srs_emb_path data/beauty/handled/itm_emb_pomrec.pkl \
    --align_on pos \
    --alpha ${alpha} --tau 0.2 \
    --llm_fuse 1 --gamma_init ${gamma} --gamma_trainable 0 \
    --rat_alpha_warmup_steps 5000 \
    --init_ckpt ${base_ckpt} --init_strict 0 \
    > "${log_file}" 2>&1 &
}

for seed in 42 43 44 ; do
  for gamma in 0.01 0.02 0.05 0.1; do
    for alpha in 0.0005 0.001 0.002; do

      # 达到并发上限就等任意一个结束
      while [ "$running" -ge "$MAX_PROCS" ]; do
        wait -n
        running=$((running-1))
      done

      launch_one "$seed" "$gamma" "$alpha"
      running=$((running+1))

      # 小延迟，避免瞬间起太快（可选）
      sleep 0.2
    done
  done
done

# 等最后一批结束
wait
echo "[GPU0] All done."
