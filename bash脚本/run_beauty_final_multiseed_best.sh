#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

MAX_PROCS=1
running=0
wait_for_slot () { while [ "$running" -ge "$MAX_PROCS" ]; do wait -n; running=$((running-1)); done; }

DATASET="beauty"
LOGDIR="./log/MyModel/${DATASET}/final_multiseed_best"
MODELDIR="./model/MyModel/${DATASET}/final_multiseed_best"
mkdir -p "${LOGDIR}" "${MODELDIR}"

SEEDS=(0 1 2 3 41 42 43)

# fallback warm-start ckpt（如果某个 seed 的 ckpt 不存在，就用 42 的）
CKPT_FALLBACK="./model/PoMRec/PoMRec__beauty__42__lr=0.002__l2=1e-06.pt"

launch_one () {
  local seed="$1"
  wait_for_slot

  # 尝试使用同 seed 的 warm-start；没有就回退到 42
  local init_ckpt="./model/PoMRec/PoMRec__beauty__${seed}__lr=0.002__l2=1e-06.pt"
  if [ ! -f "${init_ckpt}" ]; then
    echo "[WARN] init_ckpt not found for seed=${seed}, fallback to ${CKPT_FALLBACK}"
    init_ckpt="${CKPT_FALLBACK}"
  fi

  local tag="beauty_final_s${seed}_tau02_g010_a0001_ipd005_m020_lgdA08_r015"
  local out="${LOGDIR}/nohup_${tag}.out"

  echo "Launch ${tag} -> ${out}"

  nohup python main.py \
    --model_name MyModel --dataset ${DATASET} \
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
    --lambda_ilr 0 \
    --use_logic_aggr 0 --lambda_logic_aggr 0.0 \
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
echo "Beauty multiseed finished. Logs in ${LOGDIR}"