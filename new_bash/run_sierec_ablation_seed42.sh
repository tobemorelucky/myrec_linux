#!/usr/bin/env bash
set -e

# =========================================================
# Usage:
#   bash new_bash/run_sierec_ablation_seed42.sh 0 all
#   bash new_bash/run_sierec_ablation_seed42.sh 0 no_mvtc
#   bash new_bash/run_sierec_ablation_seed42.sh 0 mvtc
#
# Args:
#   $1: GPU id, default 0
#   $2: mode, default all
#       all      -> run no_mvtc + mvtc for beauty and ml-1m
#       no_mvtc  -> run only SIERec + LLM residual
#       mvtc     -> run only SIERec + LLM residual + MVTC
# =========================================================

GPU=${1:-1}
MODE=${2:-all}
SEED=42

mkdir -p new_bash
mkdir -p new_log/beauty
mkdir -p new_log/ml-1m
mkdir -p new_model/beauty
mkdir -p new_model/ml-1m

# =========================================================
# Embedding paths
# 按你的真实文件名检查，如果不一致就在这里改
# =========================================================

BEAUTY_LLM="./data/beauty/handled/llm_table_pca1536.pkl"
BEAUTY_SRS="./data/beauty/handled/itm_emb_pomrec.pkl"

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

echo "========================================================="
echo "Run SIERec ablation experiments"
echo "GPU=${GPU}"
echo "SEED=${SEED}"
echo "MODE=${MODE}"
echo "========================================================="

run_beauty_no_mvtc() {
  echo "[Submit] beauty | SIERec + LLM residual | seed=${SEED}"

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
    --use_mvtc 0 \
    --lr 0.002 \
    --l2 1e-6 \
    --batch_size 256 \
    --eval_batch_size 256 \
    --num_neg 1 \
    --epoch 200 \
    --early_stop 10 \
    --num_workers 5 \
    --log_file new_log/beauty/SIERec_LLM_seed42.log \
    --model_path new_model/beauty/SIERec_LLM_seed42.pt \
    > new_log/beauty/SIERec_LLM_seed42.out 2>&1 &

  echo "[Beauty no_mvtc] PID=$!"
  echo "tail -f new_log/beauty/SIERec_LLM_seed42.out"
}

run_beauty_mvtc() {
  echo "[Submit] beauty | SIERec + LLM residual + MVTC CF-only | seed=${SEED}"

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
    --log_file new_log/beauty/SIERec_LLM_MVTC_lam003_temp05_seed42.log \
    --model_path new_model/beauty/SIERec_LLM_MVTC_lam003_temp05_seed42.pt \
    > new_log/beauty/SIERec_LLM_MVTC_lam003_temp05_seed42.out 2>&1 &

  echo "[Beauty mvtc] PID=$!"
  echo "tail -f new_log/beauty/SIERec_LLM_MVTC_lam003_temp05_seed42.out"
}

run_ml1m_no_mvtc() {
  echo "[Submit] ml-1m | SIERec + LLM residual | seed=${SEED}"

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
    --use_mvtc 0 \
    --lr 0.001 \
    --l2 1e-6 \
    --batch_size 256 \
    --eval_batch_size 256 \
    --num_neg 1 \
    --epoch 200 \
    --early_stop 10 \
    --num_workers 5 \
    --log_file new_log/ml-1m/SIERec_LLM_seed42.log \
    --model_path new_model/ml-1m/SIERec_LLM_seed42.pt \
    > new_log/ml-1m/SIERec_LLM_seed42.out 2>&1 &

  echo "[ml-1m no_mvtc] PID=$!"
  echo "tail -f new_log/ml-1m/SIERec_LLM_seed42.out"
}

run_ml1m_mvtc() {
  echo "[Submit] ml-1m | SIERec + LLM residual + MVTC CF-only | seed=${SEED}"

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
    --log_file new_log/ml-1m/SIERec_LLM_MVTC_lam003_temp05_seed42.log \
    --model_path new_model/ml-1m/SIERec_LLM_MVTC_lam003_temp05_seed42.pt \
    > new_log/ml-1m/SIERec_LLM_MVTC_lam003_temp05_seed42.out 2>&1 &

  echo "[ml-1m mvtc] PID=$!"
  echo "tail -f new_log/ml-1m/SIERec_LLM_MVTC_lam003_temp05_seed42.out"
}

case "${MODE}" in
  all)
    run_beauty_no_mvtc
    run_beauty_mvtc
    run_ml1m_no_mvtc
    run_ml1m_mvtc
    ;;
  no_mvtc)
    run_beauty_no_mvtc
    run_ml1m_no_mvtc
    ;;
  mvtc)
    run_beauty_mvtc
    run_ml1m_mvtc
    ;;
  beauty)
    run_beauty_no_mvtc
    run_beauty_mvtc
    ;;
  ml1m)
    run_ml1m_no_mvtc
    run_ml1m_mvtc
    ;;
  *)
    echo "[ERROR] Unknown MODE: ${MODE}"
    echo "Allowed modes: all | no_mvtc | mvtc | beauty | ml1m"
    exit 1
    ;;
esac

echo "========================================================="
echo "All selected jobs submitted."
echo "Use ps -ef | grep main.py to check running jobs."
echo "========================================================="