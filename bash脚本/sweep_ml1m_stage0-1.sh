#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

MAX_PROCS=2
running=0
wait_for_slot () { while [ "$running" -ge "$MAX_PROCS" ]; do wait -n; running=$((running-1)); done; }

DATASET="ml-1m"
LOGDIR="./log/MyModel/${DATASET}/stage0_pomrec"
MODELDIR="./model/MyModel/${DATASET}/stage0_pomrec"
mkdir -p "${LOGDIR}" "${MODELDIR}"

SEED=42

# 只扫 backbone 三个最敏感超参：lr / K / lamb
LRS=(0.0005 0.001 0.002)
KS=(3)
LAMBS=(2.5 3.0 3.5)

for lr in "${LRS[@]}"; do
  for K in "${KS[@]}"; do
    for lamb in "${LAMBS[@]}"; do
      wait_for_slot
      lr_s="${lr//./}"
      lamb_s="${lamb//./}"
      tag="ml1m_pomrec_lr${lr_s}_K${K}_l${lamb_s}_s${SEED}"
      out="${LOGDIR}/nohup_${tag}.out"

      nohup python main.py \
        --model_name MyModel --dataset ${DATASET} \
        --lr "${lr}" --l2 1e-06 \
        --batch_size 256 --eval_batch_size 256 \
        --epoch 200 --early_stop 10 \
        --num_neg 1 --dropout 0 --num_workers 5 \
        --random_seed ${SEED} --load 0 \
        --history_max 20 \
        --K "${K}" --prompt_num 4 --lamb "${lamb}" --emb_size 64 --attn_size 8 --n_layers 1 \
        --use_llmemb 0 \
        --use_emile 0 \
        --use_logic_denoise 0 --use_logic_aggr 0 \
        --log_file  "${LOGDIR}/${tag}.txt" \
        --model_path "${MODELDIR}/${tag}.pt" \
        > "${out}" 2>&1 &

      running=$((running+1))
      sleep 0.2
    done
  done
done

wait
echo "Stage0 done. Logs in ${LOGDIR}"