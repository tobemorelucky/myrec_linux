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

DATASET="toys"   # 如果实际是 Toys_and_Games_5，就改这里
LOGDIR="./log/MyModel/${DATASET}"
MODELDIR="./model/MyModel"
mkdir -p "${LOGDIR}" "${MODELDIR}"

# 先用一个 seed，4 组快速定方向
SEEDS=(42)

LLM_PKL="./data/${DATASET}/handled/llm_table_pca1536.pkl"
SRS_PKL="./data/${DATASET}/handled/itm_emb_pomrec.pkl"

# warm-start ckpt（你指定的）
INIT_CKPT="./model/PoMRec/toys__42__lr=0.001__l2=1e-06__lamb=3.8__history_max=20.pt"
if [ ! -f "${INIT_CKPT}" ]; then
  echo "[FATAL] INIT_CKPT not found: ${INIT_CKPT}"
  exit 1
fi

# 固定 PoMRec 核心（按你给的）
K=3
PROMPT_NUM=4
LAMB=3.8
LR=0.001
L2=1e-06

# 固定训练配置
BATCH=256
EVAL_BATCH=256
EPOCH=200
EARLY_STOP=10
NUM_NEG=1
DROPOUT=0
NUM_WORKERS=5
HISTORY_MAX=20

EMB_SIZE=64
ATTN_SIZE=8
N_LAYERS=1

# 固定对齐 loss 设置（先不扫）
ALPHA=0.001
TAUS=(0.2 0.5)
GAMMAS=(0.05 0.1)
RAT_WARMUP=5000

launch_one () {
  local seed="$1"
  local tau="$2"
  local gamma="$3"

  wait_for_slot

  local tag="toys_llm4_warm_tau${tau}_g${gamma}_s${seed}"
  local out="${LOGDIR}/nohup_${tag}.out"

  echo "Launch ${tag} -> ${out}"

  nohup python main.py \
    --model_name MyModel --dataset ${DATASET} \
    --lr ${LR} --l2 ${L2} \
    --batch_size ${BATCH} --eval_batch_size ${EVAL_BATCH} \
    --epoch ${EPOCH} --early_stop ${EARLY_STOP} \
    --num_neg ${NUM_NEG} --dropout ${DROPOUT} --num_workers ${NUM_WORKERS} \
    --random_seed ${seed} --load 0 \
    --history_max ${HISTORY_MAX} \
    --K ${K} --prompt_num ${PROMPT_NUM} --lamb ${LAMB} \
    --emb_size ${EMB_SIZE} --attn_size ${ATTN_SIZE} --n_layers ${N_LAYERS} \
    --use_llmemb 1 --llm_fuse 1 \
    --llm_emb_path "${LLM_PKL}" \
    --srs_emb_path "${SRS_PKL}" \
    --gamma_init ${gamma} --gamma_trainable 0 \
    --alpha ${ALPHA} --tau ${tau} --rat_alpha_warmup_steps ${RAT_WARMUP} \
    --init_ckpt "${INIT_CKPT}" --init_strict 0 \
    --use_emile 0 \
    --use_logic_aggr 0 --lambda_logic_aggr 0.0 \
    --use_logic_denoise 0 \
    --log_file  "${LOGDIR}/${tag}.txt" \
    --model_path "${MODELDIR}/${tag}.pt" \
    > "${out}" 2>&1 &

  running=$((running+1))
  sleep 0.2
}

for seed in "${SEEDS[@]}"; do
  for tau in "${TAUS[@]}"; do
    for gamma in "${GAMMAS[@]}"; do
      launch_one "${seed}" "${tau}" "${gamma}"
    done
  done
done

wait
echo "All toys 4-shot LLM warm-start sweep finished. Logs in ${LOGDIR}"