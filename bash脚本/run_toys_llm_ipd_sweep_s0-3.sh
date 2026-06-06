#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1

MAX_PROCS=2
running=0
wait_for_slot () {
  while [ "$running" -ge "$MAX_PROCS" ]; do
    wait -n
    running=$((running-1))
  done
}

DATASET="toys"
LOGDIR="./log/MyModel/${DATASET}"
MODELDIR="./model/MyModel/${DATASET}"
mkdir -p "${LOGDIR}" "${MODELDIR}"

SEEDS=(0 1 2 3)

# fallback ckpt（当对应 seed 的 ckpt 不存在时使用）
CKPT_FALLBACK="./model/PoMRec/toys__42__lr=0.001__l2=1e-06__lamb=3.8__history_max=20.pt"

launch_one () {
  local seed="$1"
  wait_for_slot

  local tag="toys_llm_ipd_retest_s${seed}"
  local out="${LOGDIR}/nohup_${tag}.out"

  local init_ckpt="./model/PoMRec/toys__${seed}__lr=0.001__l2=1e-06__lamb=3.8__history_max=20.pt"
  if [ ! -f "${init_ckpt}" ]; then
    echo "[WARN] init_ckpt not found for seed=${seed}, fallback to ${CKPT_FALLBACK}"
    init_ckpt="${CKPT_FALLBACK}"
  fi

  echo "[`date '+%F %T'`] Start ${tag}  init_ckpt=${init_ckpt}"

  nohup python main.py \
    --model_name MyModel --dataset ${DATASET} \
    --lr 0.001 --l2 1e-06 \
    --batch_size 256 --eval_batch_size 256 \
    --epoch 200 --early_stop 10 \
    --num_neg 1 --dropout 0 --num_workers 5 \
    --random_seed ${seed} --load 0 \
    --history_max 20 \
    --K 3 --prompt_num 4 --lamb 3.8 --emb_size 64 --attn_size 8 --n_layers 1 \
    --use_llmemb 1 --llm_fuse 1 \
    --llm_emb_path ./data/toys/handled/llm_table_pca1536.pkl \
    --srs_emb_path ./data/toys/handled/itm_emb_pomrec.pkl \
    --gamma_init 0.05 --gamma_trainable 0 \
    --alpha 0.001 --tau 0.5 --rat_alpha_warmup_steps 5000 \
    --init_ckpt "${init_ckpt}" --init_strict 0 \
    --use_emile 1 --lambda_ipd 0.05 --ipd_margin 0.10 --emile_use_fused_itememb 0 \
    --emile_warmup_steps 20000 \
    --lambda_ilr 0 \
    --use_logic_denoise 0 \
    --use_logic_aggr 0 \
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
echo "Toys LLM+IPD sweep (seeds: ${SEEDS[*]}) finished. Logs in ${LOGDIR}"