#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1

mkdir -p ./log/MyModel/temp

mkdir -p ./log/MyModel/beauty/ablation_off_s42 ./model/MyModel/beauty/ablation_off_s42
mkdir -p ./log/MyModel/ml-1m/ablation_off_s42 ./model/MyModel/ml-1m/ablation_off_s42
mkdir -p ./log/MyModel/toys/ablation_off_s42 ./model/MyModel/toys/ablation_off_s42

########################################
# 1) beauty
########################################
nohup python main.py \
  --model_name MyModel --dataset beauty \
  --lr 0.002 --l2 1e-06 \
  --batch_size 256 --eval_batch_size 256 \
  --epoch 200 --early_stop 10 \
  --num_neg 1 --dropout 0 --num_workers 5 \
  --random_seed 42 --load 0 \
  --history_max 20 \
  --K 3 --prompt_num 4 --lamb 3.0 --emb_size 64 --attn_size 8 --n_layers 1 \
  --use_llmemb 0 --llm_fuse 0 \
  --gamma_init 0.10 --gamma_trainable 0 \
  --alpha 0.001 --tau 0.2 --rat_alpha_warmup_steps 5000 \
  --use_emile 0 \
  --use_logic_denoise 0 \
  --use_logic_aggr 0 --lambda_logic_aggr 0.0 \
  --lambda_ilr 0 \
  --log_file ./log/MyModel/beauty/ablation_off_s42/beauty_s42_all_off.txt \
  --model_path ./model/MyModel/beauty/ablation_off_s42/beauty_s42_all_off.pt \
  > ./log/MyModel/temp/nohup_beauty_s42_all_off.out 2>&1 &

echo "Launched beauty_s42_all_off"

########################################
# 2) ml-1m
########################################
nohup python main.py \
  --model_name MyModel --dataset ml-1m \
  --lr 0.001 --l2 1e-06 \
  --batch_size 256 --eval_batch_size 256 \
  --epoch 200 --early_stop 10 \
  --num_neg 1 --dropout 0 --num_workers 5 \
  --random_seed 42 --load 0 \
  --history_max 20 \
  --K 3 --prompt_num 4 --lamb 3.0 --emb_size 64 --attn_size 8 --n_layers 1 \
  --use_llmemb 0 --llm_fuse 0 \
  --gamma_init 0.08 --gamma_trainable 0 \
  --alpha 0.001 --tau 0.3 --rat_alpha_warmup_steps 5000 \
  --use_emile 0 \
  --use_logic_denoise 0 \
  --use_logic_aggr 0 --lambda_logic_aggr 0.0 \
  --lambda_ilr 0 \
  --log_file ./log/MyModel/ml-1m/ablation_off_s42/ml1m_s42_all_off.txt \
  --model_path ./model/MyModel/ml-1m/ablation_off_s42/ml1m_s42_all_off.pt \
  > ./log/MyModel/temp/nohup_ml1m_s42_all_off.out 2>&1 &

echo "Launched ml1m_s42_all_off"

########################################
# 3) toys
########################################
nohup python main.py \
  --model_name MyModel --dataset toys \
  --lr 0.001 --l2 1e-06 \
  --batch_size 256 --eval_batch_size 256 \
  --epoch 200 --early_stop 10 \
  --num_neg 1 --dropout 0 --num_workers 5 \
  --random_seed 42 --load 0 \
  --history_max 20 \
  --K 3 --prompt_num 4 --lamb 3.8 --emb_size 64 --attn_size 8 --n_layers 1 \
  --use_llmemb 0 --llm_fuse 0 \
  --gamma_init 0.05 --gamma_trainable 0 \
  --alpha 0.001 --tau 0.5 --rat_alpha_warmup_steps 5000 \
  --use_emile 0 \
  --use_logic_denoise 0 \
  --use_logic_aggr 0 --lambda_logic_aggr 0.0 \
  --lambda_ilr 0 \
  --log_file ./log/MyModel/toys/ablation_off_s42/toys_s42_all_off.txt \
  --model_path ./model/MyModel/toys/ablation_off_s42/toys_s42_all_off.pt \
  > ./log/MyModel/temp/nohup_toys_s42_all_off.out 2>&1 &

echo "Launched toys_s42_all_off"

wait
echo "Done. All 3 datasets finished."