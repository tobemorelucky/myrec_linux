#!/usr/bin/env bash
set -uo pipefail

# 只让程序看到第 1 张物理卡，也就是 GPU 0
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# 如果你不是已经激活了环境，可以取消下面两行注释
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate hzg_py10

# 0 = 某个实验失败就停止；1 = 某个失败也继续跑后面的
CONTINUE_ON_ERROR=0

mkdir -p new_log/beauty_sicr new_model/beauty_sicr
mkdir -p new_log/ml1m_sicr new_model/ml1m_sicr
mkdir -p new_log/run_seq

MASTER_LOG="new_log/run_seq/run_sicr_seq_$(date +'%Y%m%d_%H%M%S').log"

run_exp() {
  local exp_name="$1"
  local out_file="$2"
  shift 2

  echo "==================================================" | tee -a "$MASTER_LOG"
  echo "[START] $(date '+%F %T')  $exp_name" | tee -a "$MASTER_LOG"
  echo "[OUT]   $out_file" | tee -a "$MASTER_LOG"
  echo "==================================================" | tee -a "$MASTER_LOG"

  nvidia-smi | tee -a "$MASTER_LOG" || true

  python main.py "$@" > "$out_file" 2>&1
  local exit_code=$?

  if [ "$exit_code" -eq 0 ]; then
    echo "[DONE]  $(date '+%F %T')  $exp_name" | tee -a "$MASTER_LOG"
  else
    echo "[FAIL]  $(date '+%F %T')  $exp_name, exit_code=$exit_code" | tee -a "$MASTER_LOG"
    echo "[CHECK] tail -n 80 $out_file" | tee -a "$MASTER_LOG"

    if [ "$CONTINUE_ON_ERROR" -ne 1 ]; then
      exit "$exit_code"
    fi
  fi
}

common_beauty_args=(
  --model_name MyModelV2
  --dataset beauty
  --path ./data/
  --gpu 0
  --random_seed 42
  --emb_size 64
  --attn_size 8
  --K 4
  --prompt_num 3
  --n_layers 2
  --lamb 3.0
  --history_max 20
  --use_llmemb 1
  --llm_fuse 1
  --llm_emb_path ./data/beauty/handled/llm_table_pca1536.pkl
  --srs_emb_path ./data/beauty/handled/itm_emb_pomrec.pkl
  --gamma_init 0.1
  --gamma_trainable 0
  --alpha 0.001
  --tau 0.2
  --rat_alpha_warmup_steps 5000
  --use_sicr 1
  --sicr_warmup_steps 5000
  --sicr_detach 1
  --use_emile 1
  --lambda_ipd 0.05
  --ipd_margin 0.2
  --emile_warmup_steps 5000
  --lr 0.002
  --l2 1e-6
  --batch_size 256
  --eval_batch_size 256
  --num_neg 1
  --epoch 200
  --early_stop 10
  --num_workers 5
)

common_ml1m_args=(
  --model_name MyModelV2
  --dataset ml-1m
  --path ./data/
  --gpu 0
  --random_seed 42
  --emb_size 64
  --attn_size 8
  --K 2
  --prompt_num 3
  --n_layers 2
  --lamb 3.0
  --history_max 20
  --use_llmemb 1
  --llm_fuse 1
  --llm_emb_path ./data/ml-1m/handled/llm_table_pca1536.pkl
  --srs_emb_path ./data/ml-1m/handled/itm_emb_pomrec.pkl
  --gamma_init 0.08
  --gamma_trainable 0
  --alpha 0.001
  --tau 0.3
  --rat_alpha_warmup_steps 5000
  --use_sicr 1
  --sicr_warmup_steps 5000
  --sicr_detach 1
  --use_emile 1
  --lambda_ipd 0.02
  --ipd_margin 0.10
  --emile_warmup_steps 5000
  --lr 0.001
  --l2 1e-6
  --batch_size 256
  --eval_batch_size 256
  --num_neg 1
  --epoch 200
  --early_stop 10
  --num_workers 5
)

run_exp "beauty_b003_s02_i03_seed42" \
  "new_log/beauty_sicr/MyModelV2_SICR_b003_s02_i03_seed42.out" \
  "${common_beauty_args[@]}" \
  --sicr_beta 0.03 \
  --sicr_sem_weight 0.2 \
  --sicr_intent_weight 0.3 \
  --log_file new_log/beauty_sicr/MyModelV2_SICR_b003_s02_i03_seed42.log \
  --model_path new_model/beauty_sicr/MyModelV2_SICR_b003_s02_i03_seed42.pt

run_exp "beauty_b005_s02_i03_seed42" \
  "new_log/beauty_sicr/MyModelV2_SICR_b005_s02_i03_seed42.out" \
  "${common_beauty_args[@]}" \
  --sicr_beta 0.05 \
  --sicr_sem_weight 0.2 \
  --sicr_intent_weight 0.3 \
  --log_file new_log/beauty_sicr/MyModelV2_SICR_b005_s02_i03_seed42.log \
  --model_path new_model/beauty_sicr/MyModelV2_SICR_b005_s02_i03_seed42.pt

run_exp "beauty_b008_s03_i04_seed42" \
  "new_log/beauty_sicr/MyModelV2_SICR_b008_s03_i04_seed42.out" \
  "${common_beauty_args[@]}" \
  --sicr_beta 0.08 \
  --sicr_sem_weight 0.3 \
  --sicr_intent_weight 0.4 \
  --log_file new_log/beauty_sicr/MyModelV2_SICR_b008_s03_i04_seed42.log \
  --model_path new_model/beauty_sicr/MyModelV2_SICR_b008_s03_i04_seed42.pt

run_exp "ml1m_b002_s00_i03_seed42" \
  "new_log/ml1m_sicr/MyModelV2_SICR_b002_s00_i03_seed42.out" \
  "${common_ml1m_args[@]}" \
  --sicr_beta 0.02 \
  --sicr_sem_weight 0.0 \
  --sicr_intent_weight 0.3 \
  --log_file new_log/ml1m_sicr/MyModelV2_SICR_b002_s00_i03_seed42.log \
  --model_path new_model/ml1m_sicr/MyModelV2_SICR_b002_s00_i03_seed42.pt

run_exp "ml1m_b003_s01_i03_seed42" \
  "new_log/ml1m_sicr/MyModelV2_SICR_b003_s01_i03_seed42.out" \
  "${common_ml1m_args[@]}" \
  --sicr_beta 0.03 \
  --sicr_sem_weight 0.1 \
  --sicr_intent_weight 0.3 \
  --log_file new_log/ml1m_sicr/MyModelV2_SICR_b003_s01_i03_seed42.log \
  --model_path new_model/ml1m_sicr/MyModelV2_SICR_b003_s01_i03_seed42.pt

run_exp "ml1m_b005_s01_i05_seed42" \
  "new_log/ml1m_sicr/MyModelV2_SICR_b005_s01_i05_seed42.out" \
  "${common_ml1m_args[@]}" \
  --sicr_beta 0.05 \
  --sicr_sem_weight 0.1 \
  --sicr_intent_weight 0.5 \
  --log_file new_log/ml1m_sicr/MyModelV2_SICR_b005_s01_i05_seed42.log \
  --model_path new_model/ml1m_sicr/MyModelV2_SICR_b005_s01_i05_seed42.pt

echo "==================================================" | tee -a "$MASTER_LOG"
echo "[ALL DONE] $(date '+%F %T')" | tee -a "$MASTER_LOG"
echo "==================================================" | tee -a "$MASTER_LOG"