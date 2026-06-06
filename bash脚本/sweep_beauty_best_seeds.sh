#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

MAX_PROCS=1
running=0

wait_for_slot () {
  while [ "$running" -ge "$MAX_PROCS" ]; do
    wait -n
    running=$((running-1))
  done
}

DATASET="beauty"
LOGDIR="./log/MyModel/${DATASET}"
MODELDIR="./model/MyModel"
mkdir -p "${LOGDIR}" "${MODELDIR}"

SEEDS=( 1 2 3 42 )

# 你现有的 seed=42 warm-start ckpt（找不到同seed就回退用它）
CKPT_FALLBACK="./model/PoMRec/PoMRec__beauty__42__lr=0.002__l2=1e-06.pt"

launch_one () {
  local seed="$1"
  wait_for_slot

  local init_ckpt="./model/PoMRec/PoMRec__beauty__${seed}__lr=0.002__l2=1e-06.pt"
  if [ ! -f "${init_ckpt}" ]; then
    echo "[WARN] init_ckpt not found for seed=${seed}, fallback to ${CKPT_FALLBACK}"
    init_ckpt="${CKPT_FALLBACK}"
  fi

  local tag="beauty_best_llm_ipd_denoise_r015_s${seed}"
  local out="${LOGDIR}/nohup_${tag}test.out"

  echo "Launch ${tag} -> ${out}"

nohup python main.py \
  --model_name MyModel --dataset beauty \
  --lr 0.002 --l2 1e-06 \
  --batch_size 256 --eval_batch_size 256 \
  --epoch 200 --early_stop 10 \
  --num_neg 1 --dropout 0 --num_workers 5 \
  --random_seed ${seed} --load 0 \
  --history_max 20 \
  --K 3 --prompt_num 4 --lamb 3.0 --emb_size 64 --attn_size 8 --n_layers 1 \
  --use_llmemb 1 --llm_fuse 1 \
  --llm_emb_path ./data/beauty/handled/llm_table_pca1536.pkl \
  --srs_emb_path ./data/beauty/handled/itm_emb_pomrec.pkl \
  --gamma_init 0.1 --gamma_trainable 0 \
  --alpha 0.001 --tau 0.2 --rat_alpha_warmup_steps 5000 \
  --init_ckpt "${init_ckpt}" --init_strict 0 \
  --use_emile 1 --lambda_ipd 0.05 --ipd_margin 0.2 --emile_use_fused_itememb 0 \
  --emile_warmup_steps 5000 \
  --use_logic_denoise 1 \
  --logic_denoise_alpha 0.8 --logic_denoise_r 0.15 \
  --logic_denoise_topk 0 --logic_denoise_warmup_steps 20000 \
  --log_file  "${LOGDIR}/${tag}.txt" \
  --model_path "${MODELDIR}/${tag}.pt" \
  > "${out}" 2>&1 &

  running=$((running+1))
  sleep 0.2
}

for s in "${SEEDS[@]}"; do
  launch_one "${s}"
done

wait
echo "All Beauty seed-sweep finished. Logs in ${LOGDIR}"