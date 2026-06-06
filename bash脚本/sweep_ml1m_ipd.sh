#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

MAX_PROCS=2
running=0

DATASET="ml-1m"
LOGDIR="./log/MyModel/${DATASET}"
MODELDIR="./model/MyModel"
mkdir -p "${LOGDIR}" "${MODELDIR}"

SEED=42

# 固定：LLMemb-old-best（不要改）
BASE=(
  --model_name MyModel --dataset ${DATASET}
  --lr 0.001 --l2 1e-06
  --batch_size 256 --eval_batch_size 256
  --epoch 200 --early_stop 10
  --num_neg 1 --dropout 0 --num_workers 5
  --random_seed ${SEED} --load 0
  --history_max 20

  --K 3 --prompt_num 4 --lamb 3 --emb_size 64 --attn_size 8 --n_layers 1

  --use_llmemb 1 --llm_fuse 1
  --llm_emb_path ./data/ml-1m/handled/llm_table_pca1536.pkl
  --srs_emb_path ./data/ml-1m/handled/itm_emb_pomrec.pkl
  --gamma_init 0.05 --gamma_trainable 0
  --alpha 0.001 --tau 0.5 --rat_alpha_warmup_steps 5000
  --align_on pos --align_sample_k 0
  --init_ckpt ./model/PoMRec/PoMRec__ml-1m__1__lr=0.001__l2=1e-06.pt --init_strict 0
)

LAMBDAS=(0.01 0.02 0.05)
MARGINS=(0.05 0.10)

wait_for_slot () {
  while [ "$running" -ge "$MAX_PROCS" ]; do
    wait -n
    running=$((running-1))
  done
}

for lam in "${LAMBDAS[@]}"; do
  for m in "${MARGINS[@]}"; do
    wait_for_slot

    tag="ipd_l${lam}_m${m}"
    out="${LOGDIR}/nohup_${tag}_s${SEED}.out"

    echo "Launch ${tag} -> ${out}"

    nohup python main.py \
      "${BASE[@]}" \
      --use_emile 1 --lambda_ipd "${lam}" --lambda_ilr 0 --ipd_margin "${m}" --emile_use_fused_itememb 0 \
      --use_logic_denoise 0 \
      --log_file  "${LOGDIR}/${tag}_s${SEED}.txt" \
      --model_path "${MODELDIR}/${tag}_s${SEED}.pt" \
      > "${out}" 2>&1 &

    running=$((running+1))
    sleep 0.2
  done
done

wait
echo "All IPD scan jobs finished. Logs in ${LOGDIR}"
