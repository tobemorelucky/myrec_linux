#!/bin/bash
set -u

MAX_PROCS=5
dataset=beauty
mkdir -p log/beauty

running=0

for seed in 42 43 44 ; do
  for gamma in 0.01 0.02 0.05 0.1; do
    for alpha in 0.0005 0.001 0.002; do

      # 如果正在跑的任务数达到上限，就等任意一个结束
      while [ "$running" -ge "$MAX_PROCS" ]; do
        wait -n
        running=$((running-1))
      done

      log_file="log/beauty/llmemb_bestbase_seed${seed}_g${gamma}_a${alpha}.log"
      echo "启动实验：seed=${seed}, gamma=${gamma}, alpha=${alpha} → ${log_file}"

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
        --llm_fuse 1 --gamma_init ${gamma} --gamma_trainable 1 \
        --rat_alpha_warmup_steps 5000 \
        > "${log_file}" 2>&1 &

      running=$((running+1))
      sleep 0.2
    done
  done
done

# 等最后一批跑完
wait
echo "全部完成"
