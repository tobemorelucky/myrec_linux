#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

MAX_PROCS=2
running=0
wait_for_slot () { while [ "$running" -ge "$MAX_PROCS" ]; do wait -n; running=$((running-1)); done; }

DATASET="toys"
LOGDIR="./log/MyModel/${DATASET}"
MODELDIR="./model/MyModel"
mkdir -p "${LOGDIR}" "${MODELDIR}"

SEED=42

LLM_PKL="./data/${DATASET}/handled/llm_table_pca1536.pkl"
SRS_PKL="./data/${DATASET}/handled/itm_emb_pomrec.pkl"

INIT_CKPT="./model/PoMRec/toys__42__lr=0.001__l2=1e-06__lamb=3.8__history_max=20.pt"
[ -f "${INIT_CKPT}" ] || { echo "[FATAL] ckpt not found: ${INIT_CKPT}"; exit 1; }

# 固定你刚确定的最优
TAU=0.5
GAMMA=0.05
RAT_WARMUP=5000

# 下一轮：只扫 alpha（4组）
ALPHAS=(0.0002 0.0005 0.001)

for alpha in "${ALPHAS[@]}"; do
  wait_for_slot
  a_s="${alpha//./}"
  tag="toys_llm_warm_tau${TAU}_g${GAMMA}_a${a_s}_s${SEED}"
  out="${LOGDIR}/nohup_${tag}.out"

  nohup python main.py \
    --model_name MyModel --dataset ${DATASET} \
    --lr 0.001 --l2 1e-06 \
    --batch_size 256 --eval_batch_size 256 \
    --epoch 200 --early_stop 10 \
    --num_neg 1 --dropout 0 --num_workers 5 \
    --random_seed ${SEED} --load 0 \
    --history_max 20 \
    --K 3 --prompt_num 4 --lamb 3.8 --emb_size 64 --attn_size 8 --n_layers 1 \
    --use_llmemb 1 --llm_fuse 1 \
    --llm_emb_path "${LLM_PKL}" --srs_emb_path "${SRS_PKL}" \
    --gamma_init ${GAMMA} --gamma_trainable 0 \
    --alpha ${alpha} --tau ${TAU} --rat_alpha_warmup_steps ${RAT_WARMUP} \
    --init_ckpt "${INIT_CKPT}" --init_strict 0 \
    --use_emile 0 --use_logic_denoise 0 --use_logic_aggr 0 \
    --log_file  "${LOGDIR}/${tag}.txt" \
    --model_path "${MODELDIR}/${tag}.pt" \
    > "${out}" 2>&1 &

  running=$((running+1))
  sleep 0.2
done

wait
echo "Alpha sweep done. Logs in ${LOGDIR}"