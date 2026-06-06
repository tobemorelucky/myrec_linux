#!/usr/bin/env bash
set -e

# 用法：
# bash new_bash/run_sierec_mvtc_seed42.sh 0
# 其中 0 表示使用 GPU 0

GPU=${1:-0}
SEED=42

mkdir -p new_log/beauty
mkdir -p new_log/ml-1m
mkdir -p new_model/beauty
mkdir -p new_model/ml-1m

# =========================
# Beauty paths
# =========================
BEAUTY_LLM="./data/beauty/handled/llm_table_pca1536.pkl"
BEAUTY_SRS="./data/beauty/handled/itm_emb_pomrec.pkl"

# =========================
# ml-1m paths
# 按你的真实文件名修改
# =========================
ML1M_LLM="./data/ml-1m/handled/llm_table_pca1536.pkl"
ML1M_SRS="./data/ml-1m/handled/itm_emb_pomrec.pkl"

check_file() {
  if [ ! -f "$1" ]; then
    echo "[ERROR] File not found: $1"
    exit 1
  fi
}

check_file "$BEAUTY_LLM"
check_file "$BEAUTY_SRS"
check_file "$ML1M_LLM"
check_file "$ML1M_SRS"

echo "Start SIERec MVTC CF-only experiments, seed=${SEED}, gpu=${GPU}"

# =========================================================
# Beauty: SIERec + LLM residual + MVTC CF-only
# =========================================================
nohup python main.py \
  --model_name SIERec \
  --dataset beauty \
  --path ./data/ \
  --gpu ${GPU} \
  --random_seed ${SEED} \
  --emb_size 64 \
  --attn_size 8 \
  --K 4 \
  --prompt_num 3 \
  --n_layers 2 \
  --lamb 4.0 \
  --history_max 20 \
  --use_llmemb 1 \
  --llm_emb_path ${BEAUTY_LLM} \
  --srs_emb_path ${BEAUTY_SRS} \
  --alpha 0.001 \
  --tau 0.2 \
  --rat_alpha_warmup_steps 5000 \
  --llm_fuse 1 \
  --gamma_init 0.1 \
  --gamma_trainable 0 \
  --use_mvtc 1 \
  --lambda_mvtc 0.03 \
  --mvtc_temp 0.5 \
  --mvtc_warmup_steps 5000 \
  --lr 0.002 \
  --l2 1e-6 \
  --batch_size 256 \
  --eval_batch_size 256 \
  --num_neg 1 \
  --epoch 200 \
  --early_stop 10 \
  --num_workers 5 \
  --log_file new_log/beauty/SIERec_MVTC_seed42.log \
  --model_path new_model/beauty/SIERec_MVTC_seed42.pt \
  > new_log/beauty/SIERec_MVTC_seed42.out 2>&1 &

PID_BEAUTY=$!
echo "[Beauty] PID=${PID_BEAUTY}, log=new_log/beauty/SIERec_MVTC_seed42.out"

# =========================================================
# ml-1m: SIERec + LLM residual + MVTC CF-only
# =========================================================
nohup python main.py \
  --model_name SIERec \
  --dataset ml-1m \
  --path ./data/ \
  --gpu ${GPU} \
  --random_seed ${SEED} \
  --emb_size 64 \
  --attn_size 8 \
  --K 2 \
  --prompt_num 3 \
  --n_layers 2 \
  --lamb 1.0 \
  --history_max 20 \
  --use_llmemb 1 \
  --llm_emb_path ${ML1M_LLM} \
  --srs_emb_path ${ML1M_SRS} \
  --alpha 0.001 \
  --tau 0.2 \
  --rat_alpha_warmup_steps 5000 \
  --llm_fuse 1 \
  --gamma_init 0.1 \
  --gamma_trainable 0 \
  --use_mvtc 1 \
  --lambda_mvtc 0.03 \
  --mvtc_temp 0.5 \
  --mvtc_warmup_steps 5000 \
  --lr 0.001 \
  --l2 1e-6 \
  --batch_size 256 \
  --eval_batch_size 256 \
  --num_neg 1 \
  --epoch 200 \
  --early_stop 10 \
  --num_workers 5 \
  --log_file new_log/ml-1m/SIERec_MVTC_seed42.log \
  --model_path new_model/ml-1m/SIERec_MVTC_seed42.pt \
  > new_log/ml-1m/SIERec_MVTC_seed42.out 2>&1 &

PID_ML1M=$!
echo "[ml-1m] PID=${PID_ML1M}, log=new_log/ml-1m/SIERec_MVTC_seed42.out"

echo "All jobs submitted."
echo "Check logs:"
echo "tail -f new_log/beauty/SIERec_MVTC_seed42.out"
echo "tail -f new_log/ml-1m/SIERec_MVTC_seed42.out"