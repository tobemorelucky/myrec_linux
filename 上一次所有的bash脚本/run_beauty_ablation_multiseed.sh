#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

########################################
# 并行控制
########################################
MAX_PROCS=2
running=0
wait_for_slot () {
  while [ "$running" -ge "$MAX_PROCS" ]; do
    wait -n
    running=$((running-1))
  done
}

########################################
# 数据与日志目录
########################################
DATASET="beauty"
SUBDIR="ablation_multiseed"
LOGDIR="./log/MyModel/${DATASET}/${SUBDIR}"
MODELDIR="./model/MyModel/${DATASET}/${SUBDIR}"
mkdir -p "${LOGDIR}" "${MODELDIR}"

########################################
# Seeds
########################################
SEEDS=(0 1 2 3 41 42 43)

########################################
# init_ckpt：按 seed 找；缺失则 fallback
########################################
CKPT_FALLBACK="./model/PoMRec/PoMRec__beauty__42__lr=0.002__l2=1e-06.pt"

########################################
# 固定最佳参数（beauty）
########################################
LR=0.002
L2=1e-06
BATCH_SIZE=256
EVAL_BATCH_SIZE=256
EPOCH=200
EARLY_STOP=10
NUM_NEG=1
DROPOUT=0
NUM_WORKERS=5
HISTORY_MAX=20

K=3
PROMPT_NUM=4
LAMB=3.0
EMB_SIZE=64
ATTN_SIZE=8
N_LAYERS=1

LLM_EMB_PATH="./data/beauty/handled/llm_table_pca1536.pkl"
SRS_EMB_PATH="./data/beauty/handled/itm_emb_pomrec.pkl"

GAMMA=0.1
ALIGN_ALPHA=0.001
TAU=0.2
RAT_WARMUP=5000

IPD_LAMBDA=0.05
IPD_MARGIN=0.2
EMILE_WARMUP=5000

launch_llm_only () {
  local seed="$1"
  wait_for_slot

  local init_ckpt="./model/PoMRec/PoMRec__beauty__${seed}__lr=0.002__l2=1e-06.pt"
  if [ ! -f "${init_ckpt}" ]; then
    echo "[WARN] init_ckpt not found for seed=${seed}, fallback to ${CKPT_FALLBACK}"
    init_ckpt="${CKPT_FALLBACK}"
  fi

  local tag="beauty_s${seed}_llm_only"
  local out="${LOGDIR}/nohup_${tag}.out"

  echo "Launch ${tag} -> ${out}"

  nohup python main.py \
    --model_name MyModel --dataset ${DATASET} \
    --lr ${LR} --l2 ${L2} \
    --batch_size ${BATCH_SIZE} --eval_batch_size ${EVAL_BATCH_SIZE} \
    --epoch ${EPOCH} --early_stop ${EARLY_STOP} \
    --num_neg ${NUM_NEG} --dropout ${DROPOUT} --num_workers ${NUM_WORKERS} \
    --random_seed ${seed} --load 0 \
    --history_max ${HISTORY_MAX} \
    --K ${K} --prompt_num ${PROMPT_NUM} --lamb ${LAMB} --emb_size ${EMB_SIZE} --attn_size ${ATTN_SIZE} --n_layers ${N_LAYERS} \
    --use_llmemb 1 --llm_fuse 1 \
    --llm_emb_path ${LLM_EMB_PATH} \
    --srs_emb_path ${SRS_EMB_PATH} \
    --gamma_init ${GAMMA} --gamma_trainable 0 \
    --alpha ${ALIGN_ALPHA} --tau ${TAU} --rat_alpha_warmup_steps ${RAT_WARMUP} \
    --init_ckpt "${init_ckpt}" --init_strict 0 \
    --use_emile 0 \
    --use_logic_denoise 0 \
    --use_logic_aggr 0 --lambda_logic_aggr 0.0 \
    --lambda_ilr 0 \
    --log_file  "${LOGDIR}/${tag}.txt" \
    --model_path "${MODELDIR}/${tag}.pt" \
    > "${out}" 2>&1 &

  running=$((running+1))
  sleep 0.2
}

launch_llm_ipd () {
  local seed="$1"
  wait_for_slot

  local init_ckpt="./model/PoMRec/PoMRec__beauty__${seed}__lr=0.002__l2=1e-06.pt"
  if [ ! -f "${init_ckpt}" ]; then
    echo "[WARN] init_ckpt not found for seed=${seed}, fallback to ${CKPT_FALLBACK}"
    init_ckpt="${CKPT_FALLBACK}"
  fi

  local tag="beauty_s${seed}_llm_ipd"
  local out="${LOGDIR}/nohup_${tag}.out"

  echo "Launch ${tag} -> ${out}"

  nohup python main.py \
    --model_name MyModel --dataset ${DATASET} \
    --lr ${LR} --l2 ${L2} \
    --batch_size ${BATCH_SIZE} --eval_batch_size ${EVAL_BATCH_SIZE} \
    --epoch ${EPOCH} --early_stop ${EARLY_STOP} \
    --num_neg ${NUM_NEG} --dropout ${DROPOUT} --num_workers ${NUM_WORKERS} \
    --random_seed ${seed} --load 0 \
    --history_max ${HISTORY_MAX} \
    --K ${K} --prompt_num ${PROMPT_NUM} --lamb ${LAMB} --emb_size ${EMB_SIZE} --attn_size ${ATTN_SIZE} --n_layers ${N_LAYERS} \
    --use_llmemb 1 --llm_fuse 1 \
    --llm_emb_path ${LLM_EMB_PATH} \
    --srs_emb_path ${SRS_EMB_PATH} \
    --gamma_init ${GAMMA} --gamma_trainable 0 \
    --alpha ${ALIGN_ALPHA} --tau ${TAU} --rat_alpha_warmup_steps ${RAT_WARMUP} \
    --init_ckpt "${init_ckpt}" --init_strict 0 \
    --use_emile 1 --lambda_ipd ${IPD_LAMBDA} --ipd_margin ${IPD_MARGIN} --emile_use_fused_itememb 0 \
    --emile_warmup_steps ${EMILE_WARMUP} \
    --use_logic_denoise 0 \
    --use_logic_aggr 0 --lambda_logic_aggr 0.0 \
    --lambda_ilr 0 \
    --log_file  "${LOGDIR}/${tag}.txt" \
    --model_path "${MODELDIR}/${tag}.pt" \
    > "${out}" 2>&1 &

  running=$((running+1))
  sleep 0.2
}

for s in "${SEEDS[@]}"; do
  launch_llm_only "${s}"
  launch_llm_ipd "${s}"
done

wait
echo "Beauty ablation multiseed finished. Logs in ${LOGDIR}"