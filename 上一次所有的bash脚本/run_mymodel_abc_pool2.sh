#!/usr/bin/env bash
set -euo pipefail

MAX_PROCS=2

export CUDA_VISIBLE_DEVICES=0
DATASET="ml-1m"
LOGDIR="./log/MyModel/${DATASET}"
MODELDIR="./model/MyModel"
mkdir -p "${LOGDIR}" "${MODELDIR}"

SEED=42

COMMON=(
  --model_name MyModel --dataset ${DATASET}
  --lr 0.001 --l2 1e-06
  --batch_size 256 --eval_batch_size 256
  --epoch 200 --early_stop 10
  --num_neg 1 --dropout 0 --num_workers 5
  --random_seed ${SEED} --load 0

  # PoMRec backbone（按你旧 best 日志钉死）
  --K 3 --prompt_num 4 --lamb 3 --emb_size 64 --attn_size 8 --n_layers 1
  --history_max 20

  # LLMemb（按旧 best 钉死）
  --use_llmemb 1 --llm_fuse 1
  --llm_emb_path ./data/ml-1m/handled/llm_table_pca1536.pkl
  --srs_emb_path ./data/ml-1m/handled/itm_emb_pomrec.pkl
  --gamma_init 0.05 --gamma_trainable 0
  --alpha 0.001 --tau 0.5 --rat_alpha_warmup_steps 5000
  --align_on pos --align_sample_k 0

  # warm-start（你旧命令就是从这个起步）
  --init_ckpt ./model/PoMRec/PoMRec__ml-1m__1__lr=0.001__l2=1e-06.pt --init_strict 0
)

running=0

wait_for_slot () {
  while [ "$running" -ge "$MAX_PROCS" ]; do
    wait -n
    running=$((running-1))
  done
}

launch_one () {
  local tag="$1"   # A/B/C
  shift
  wait_for_slot
  echo "Launch ${tag} (seed=${SEED})"
  nohup python main.py "${COMMON[@]}" "$@" &
  running=$((running+1))
  sleep 0.2
}

# A: LLMemb only
launch_one "A" \
  --use_emile 0 \
  --use_logic_denoise 0 \
  --log_file  "${LOGDIR}/A_llm_s${SEED}_g005_a1e-3_tau05.txt" \
  --model_path "${MODELDIR}/A_ml1m_s${SEED}_g005_a1e-3_tau05.pt" \
  > "${LOGDIR}/nohup_A_s${SEED}_g005_a1e-3_tau05.out" 2>&1

# B: + IPD
launch_one "B" \
  --use_emile 1 --lambda_ipd 0.05 --lambda_ilr 0 --ipd_margin 0.2 --emile_use_fused_itememb 0 \
  --use_logic_denoise 0 \
  --log_file  "${LOGDIR}/B_ipd005_s${SEED}.txt" \
  --model_path "${MODELDIR}/B_ml1m_s${SEED}_ipd005_m02.pt" \
  > "${LOGDIR}/nohup_B_s${SEED}_ipd005_m02.out" 2>&1

# C: + IPD + LGD
launch_one "C" \
  --use_emile 1 --lambda_ipd 0.05 --lambda_ilr 0 --ipd_margin 0.2 --emile_use_fused_itememb 0 \
  --use_logic_denoise 1 --logic_denoise_alpha 1.0 --logic_denoise_topk 0 --logic_denoise_use_fused 0 \
  --log_file  "${LOGDIR}/C_ipd005_lgd_a1_top0_s${SEED}.txt" \
  --model_path "${MODELDIR}/C_ml1m_s${SEED}_ipd005_lgd_a1_top0.pt" \
  > "${LOGDIR}/nohup_C_s${SEED}_ipd005_lgd_a1_top0.out" 2>&1

# 等全部结束
wait
echo "All A/B/C finished on ${DATASET} (seed=${SEED}). Logs in ${LOGDIR}"
