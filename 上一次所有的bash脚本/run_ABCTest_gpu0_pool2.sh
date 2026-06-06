#!/bin/bash
set -euo pipefail

# ========== 配置 ==========
export CUDA_VISIBLE_DEVICES=0
MAX_PROCS=2

dataset=beauty
seed=42
alpha=0.001
init_ckpt=./model/PoMRec/PoMRec__beauty__42__lr=0.002__l2=1e-06.pt

mkdir -p log/beauty

running=0

launch_one () {
  local group="$1"       # A/B/C
  local llm_fuse="$2"    # 0/1
  local gamma_init="$3"  # e.g. 0.05 or "NA"
  local gamma_trainable="$4" # 0/1 or "NA"
  local tau="$5"         # 0.2/0.5

  # 日志命名（对齐你的分组）
  local log_file=""
  if [ "$group" = "A" ]; then
    log_file="log/beauty/A_alignOnly_fuse0_tau${tau}_seed${seed}.log"
  elif [ "$group" = "B" ]; then
    log_file="log/beauty/B_fuse1_g${gamma_init}_tau${tau}_train0_seed${seed}.log"
  else
    log_file="log/beauty/C_fuse1_g${gamma_init}_tau${tau}_train1_seed${seed}.log"
  fi

  echo "[GPU0] launch ${group}: fuse=${llm_fuse} gamma_init=${gamma_init} trainable=${gamma_trainable} tau=${tau} -> ${log_file}"

  # 拼命令：A组不传 gamma 参数；B/C 组传 gamma 参数
  if [ "$group" = "A" ]; then
    nohup python main.py \
      --model_name PoMRec --dataset ${dataset} \
      --K 4 --attn_size 8 --emb_size 64 --prompt_num 3 --n_layers 2 --lamb 4.0 --history_max 20 \
      --lr 0.002 --l2 1e-06 --batch_size 256 --eval_batch_size 256 \
      --epoch 200 --early_stop 10 --num_neg 1 --dropout 0 --num_workers 5 --random_seed ${seed} \
      --use_llmemb 1 \
      --llm_emb_path data/beauty/handled/llm_table_pca1536.pkl \
      --srs_emb_path data/beauty/handled/itm_emb_pomrec.pkl \
      --alpha ${alpha} --rat_alpha_warmup_steps 5000 --align_on pos \
      --llm_fuse 0 \
      --tau ${tau} \
      --init_ckpt ${init_ckpt} --init_strict 0 \
      > "${log_file}" 2>&1 &
  else
    nohup python main.py \
      --model_name PoMRec --dataset ${dataset} \
      --K 4 --attn_size 8 --emb_size 64 --prompt_num 3 --n_layers 2 --lamb 4.0 --history_max 20 \
      --lr 0.002 --l2 1e-06 --batch_size 256 --eval_batch_size 256 \
      --epoch 200 --early_stop 10 --num_neg 1 --dropout 0 --num_workers 5 --random_seed ${seed} \
      --use_llmemb 1 \
      --llm_emb_path data/beauty/handled/llm_table_pca1536.pkl \
      --srs_emb_path data/beauty/handled/itm_emb_pomrec.pkl \
      --alpha ${alpha} --rat_alpha_warmup_steps 5000 --align_on pos \
      --llm_fuse 1 --gamma_init ${gamma_init} --gamma_trainable ${gamma_trainable} \
      --tau ${tau} \
      --init_ckpt ${init_ckpt} --init_strict 0 \
      > "${log_file}" 2>&1 &
  fi
}

# ========== 8个实验配置（按你的A/B/C组） ==========
# A组：align-only（对照）
# A1: fuse0 tau0.2
# A2: fuse0 tau0.5
configs=(
  "A 0 NA NA 0.2"
  "A 0 NA NA 0.5"

  # B组：fuse + gamma固定（trainable0）
  "B 1 0.01 0 0.2"
  "B 1 0.05 0 0.2"
  "B 1 0.1  0 0.2"
  "B 1 0.05 0 0.5"

  # C组：fuse + gamma可训练（trainable1）
  "C 1 0.01 1 0.2"
  "C 1 0.01 1 0.5"
)

# ========== 并发池执行（最多2个同时跑） ==========
for cfg in "${configs[@]}"; do
  while [ "$running" -ge "$MAX_PROCS" ]; do
    wait -n
    running=$((running-1))
  done

  # shellcheck disable=SC2086
  launch_one $cfg
  running=$((running+1))

  # 可选：小延迟避免瞬间拉起太猛
  sleep 0.2
done

# 等最后一批结束
wait
echo "[GPU0] All 8 experiments finished."
