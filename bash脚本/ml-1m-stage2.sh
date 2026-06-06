#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

MAX_PROCS=1
running=0
wait_for_slot () { while [ "$running" -ge "$MAX_PROCS" ]; do wait -n; running=$((running-1)); done; }

DATASET="ml-1m"
LOGDIR="./log/MyModel/${DATASET}/stage2_ipd"
MODELDIR="./model/MyModel/${DATASET}/stage2_ipd"
mkdir -p "${LOGDIR}" "${MODELDIR}"

SEED=42

# 固定 backbone（按你当前常用）
K=3
LAMB=3.5

# 固定 LLM（先用你当前最常用的一组；等 Stage1 出最优后替换）
GAMMA=0.05
TAU=0.5
ALPHA=0.0005

INIT_CKPT="./model/PoMRec/PoMRec__ml-1m__1__lr=0.001__l2=1e-06.pt"

LAMBDA_IPD=(0.06 0.08 0.10)
MARGIN=(0.10)

for lam in "${LAMBDA_IPD[@]}"; do
  for m in "${MARGIN[@]}"; do
    wait_for_slot

    lam_s="${lam//./}"
    m_s="${m//./}"
    tag="ml1m_ipd_l${lam_s}_m${m_s}_s${SEED}"
    out="${LOGDIR}/nohup_${tag}.out"

    nohup python main.py \
      --model_name MyModel --dataset ${DATASET} \
      --lr 0.001 --l2 1e-06 \
      --batch_size 256 --eval_batch_size 256 \
      --epoch 200 --early_stop 10 \
      --num_neg 1 --dropout 0 --num_workers 5 \
      --random_seed ${SEED} --load 0 \
      --history_max 20 \
      --K "${K}" --prompt_num 4 --lamb "${LAMB}" --emb_size 64 --attn_size 8 --n_layers 1 \
      --use_llmemb 1 --llm_fuse 1 \
      --llm_emb_path ./data/ml-1m/handled/llm_table_pca1536.pkl \
      --srs_emb_path ./data/ml-1m/handled/itm_emb_pomrec.pkl \
      --gamma_init "${GAMMA}" --gamma_trainable 0 \
      --alpha "${ALPHA}" --tau "${TAU}" --rat_alpha_warmup_steps 5000 \
      --init_ckpt "${INIT_CKPT}" --init_strict 0 \
      --use_emile 1 --lambda_ipd "${lam}" --lambda_ilr 0 --ipd_margin "${m}" --emile_use_fused_itememb 0 \
      --use_logic_denoise 0 --use_logic_aggr 0 \
      --log_file  "${LOGDIR}/${tag}.txt" \
      --model_path "${MODELDIR}/${tag}.pt" \
      > "${out}" 2>&1 &

    running=$((running+1))
    sleep 0.2
  done
done

wait
echo "Stage2 done. Logs in ${LOGDIR}"