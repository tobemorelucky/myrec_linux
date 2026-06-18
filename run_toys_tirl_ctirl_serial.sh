#!/usr/bin/env bash
set -e

cd /home/yyx/hzgcode/pom2.0

mkdir -p new_log/toys_tirl_probe new_model/toys_tirl_probe
mkdir -p new_log/toys_ctirl_probe new_model/toys_ctirl_probe

echo "========== START TIRL: $(date '+%F %T') =========="

python main.py \
  --model_name MyModelTIRL \
  --dataset toys \
  --path ./data/ \
  --gpu 0 \
  --random_seed 42 \
  --load 0 \
  --emb_size 64 \
  --attn_size 8 \
  --K 3 \
  --prompt_num 4 \
  --n_layers 1 \
  --lamb 3.8 \
  --history_max 20 \
  --use_llmemb 1 \
  --llm_fuse 1 \
  --llm_emb_path ./data/toys/handled/llm_table_pca1536.pkl \
  --srs_emb_path ./data/toys/handled/itm_emb_pomrec.pkl \
  --gamma_init 0.05 \
  --gamma_trainable 0 \
  --alpha 0.001 \
  --tau 0.5 \
  --rat_alpha_warmup_steps 5000 \
  --init_ckpt ./model/PoMRec/toys__42__lr=0.001__l2=1e-06__lamb=3.8__history_max=20.pt \
  --init_strict 0 \
  --use_emile 1 \
  --lambda_ipd 0.05 \
  --ipd_margin 0.10 \
  --emile_use_fused_itememb 0 \
  --emile_warmup_steps 20000 \
  --use_tirl 1 \
  --lambda_tirl 0.02 \
  --tirl_warmup_steps 20000 \
  --tirl_mode selected \
  --tirl_detach_route 1 \
  --tirl_neg_reduce mean \
  --lr 0.001 \
  --l2 1e-6 \
  --batch_size 256 \
  --eval_batch_size 256 \
  --num_neg 1 \
  --dropout 0 \
  --epoch 200 \
  --early_stop 10 \
  --num_workers 5 \
  --log_file new_log/toys_tirl_probe/toys_TIRL_lam002_seed42.log \
  --model_path new_model/toys_tirl_probe/toys_TIRL_lam002_seed42.pt \
  > new_log/toys_tirl_probe/toys_TIRL_lam002_seed42.out 2>&1

echo "========== END TIRL: $(date '+%F %T') =========="
echo "========== START CTIRL: $(date '+%F %T') =========="

python main.py \
  --model_name MyModelCTIRL \
  --dataset toys \
  --path ./data/ \
  --gpu 0 \
  --random_seed 42 \
  --load 0 \
  --emb_size 64 \
  --attn_size 8 \
  --K 3 \
  --prompt_num 4 \
  --n_layers 1 \
  --lamb 3.8 \
  --history_max 20 \
  --use_llmemb 1 \
  --llm_fuse 1 \
  --llm_emb_path ./data/toys/handled/llm_table_pca1536.pkl \
  --srs_emb_path ./data/toys/handled/itm_emb_pomrec.pkl \
  --gamma_init 0.05 \
  --gamma_trainable 0 \
  --alpha 0.001 \
  --tau 0.5 \
  --rat_alpha_warmup_steps 5000 \
  --init_ckpt ./model/PoMRec/toys__42__lr=0.001__l2=1e-06__lamb=3.8__history_max=20.pt \
  --init_strict 0 \
  --use_emile 1 \
  --lambda_ipd 0.05 \
  --ipd_margin 0.10 \
  --emile_use_fused_itememb 0 \
  --emile_warmup_steps 20000 \
  --use_ctirl 1 \
  --lambda_ctirl 0.001 \
  --ctirl_warmup_steps 20000 \
  --ctirl_route_temp 0.2 \
  --ctirl_conf_threshold 0.55 \
  --ctirl_gate_mode linear \
  --ctirl_gate_temp 0.05 \
  --ctirl_score_norm 1 \
  --ctirl_neg_reduce mean \
  --ctirl_loss_normalize batch \
  --lr 0.001 \
  --l2 1e-6 \
  --batch_size 256 \
  --eval_batch_size 256 \
  --num_neg 1 \
  --dropout 0 \
  --epoch 200 \
  --early_stop 10 \
  --num_workers 5 \
  --log_file new_log/toys_ctirl_probe/toys_CTIRL_lam0001_th055_seed42.log \
  --model_path new_model/toys_ctirl_probe/toys_CTIRL_lam0001_th055_seed42.pt \
  > new_log/toys_ctirl_probe/toys_CTIRL_lam0001_th055_seed42.out 2>&1

echo "========== END CTIRL: $(date '+%F %T') =========="

echo "========== FINAL RESULTS =========="
grep "Test After Training" new_log/toys_tirl_probe/toys_TIRL_lam002_seed42.log || true
grep "Test After Training" new_log/toys_ctirl_probe/toys_CTIRL_lam0001_th055_seed42.log || true
