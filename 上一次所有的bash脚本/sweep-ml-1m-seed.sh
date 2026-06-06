#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

MAX_PROCS=2
running=0

wait_for_slot () {
  while [ "$running" -ge "$MAX_PROCS" ]; do
    wait -n
    running=$((running-1))
  done
}

DATASET="ml-1m"
LOGDIR="./log/MyModel/${DATASET}"
MODELDIR="./model/MyModel"
mkdir -p "${LOGDIR}" "${MODELDIR}"

SEEDS=(42)

# ml-1m 的 warm-start ckpt（你的脚本里是固定这一份）
INIT_CKPT="./model/PoMRec/PoMRec__ml-1m__1__lr=0.001__l2=1e-06.pt"

launch_one () {
  local seed="$1"
  wait_for_slot

  local tag="ml1m_A4_aggr_soft_s${seed}"
  local out="${LOGDIR}/nohup_${tag}test.out"

  echo "Launch ${tag} -> ${out}"

nohup python main.py \
  --model_name MyModel --dataset ml-1m \
  --lr 0.001 --l2 1e-06 \
  --batch_size 256 --eval_batch_size 256 \
  --epoch 200 --early_stop 10 \
  --num_neg 1 --dropout 0 --num_workers 5 \
  --random_seed ${seed} --load 0 \
  --history_max 20 \
  --K 3 --prompt_num 4 --lamb 3 --emb_size 64 --attn_size 8 --n_layers 1 \
  --use_llmemb 1 --llm_fuse 1 \
  --llm_emb_path ./data/ml-1m/handled/llm_table_pca1536.pkl \
  --srs_emb_path ./data/ml-1m/handled/itm_emb_pomrec.pkl \
  --gamma_init 0.05 --gamma_trainable 0 \
  --alpha 0.001 --tau 0.5 --rat_alpha_warmup_steps 5000 \
  --init_ckpt "${INIT_CKPT}" --init_strict 0 \
  --use_emile 1 --lambda_ipd 0.05 --ipd_margin 0.10 --emile_use_fused_itememb 0 \
  --emile_warmup_steps 5000 \
  --use_logic_aggr 1 \
  --lambda_logic_aggr 0.2 --logic_lambda_max 0.10 \
  --logic_support_temp 2.5 --logic_gate_a 10 --logic_gate_b 0.80 \
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
echo "All ML-1M seed-sweep finished. Logs in ${LOGDIR}"