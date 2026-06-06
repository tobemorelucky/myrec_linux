#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1

MAX_PROCS=2
wait_for_slot () { while [ "$(jobs -rp | wc -l)" -ge "$MAX_PROCS" ]; do sleep 3; done; }

DATASET="ml-1m"
SEEDS=(0 1 2 3 41)

SUBDIR="ablation_multiseed"
LOGDIR="./log/MyModel/${DATASET}/${SUBDIR}"
MODELDIR="./model/MyModel/${DATASET}/${SUBDIR}"
mkdir -p "${LOGDIR}" "${MODELDIR}"

# ===== fixed best (ml-1m) =====
LR=0.001
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

LLM_EMB_PATH="./data/ml-1m/handled/llm_table_pca1536.pkl"
SRS_EMB_PATH="./data/ml-1m/handled/itm_emb_pomrec.pkl"

GAMMA=0.08
ALIGN_ALPHA=0.001
TAU=0.3
RAT_WARMUP=5000

INIT_CKPT="./model/PoMRec/PoMRec__ml-1m__1__lr=0.001__l2=1e-06.pt"

IPD_LAMBDA=0.02
IPD_MARGIN=0.10
EMILE_WARMUP=20000

for SEED in "${SEEDS[@]}"; do
  # =========================
  # 1) LLM only
  # =========================
  wait_for_slot
  tag="ml1m_s${SEED}_llm_only"
  out="${LOGDIR}/nohup_${tag}.out"

  nohup python main.py \
    --model_name MyModel --dataset ${DATASET} \
    --lr ${LR} --l2 ${L2} \
    --batch_size ${BATCH_SIZE} --eval_batch_size ${EVAL_BATCH_SIZE} \
    --epoch ${EPOCH} --early_stop ${EARLY_STOP} \
    --num_neg ${NUM_NEG} --dropout ${DROPOUT} --num_workers ${NUM_WORKERS} \
    --random_seed ${SEED} --load 0 \
    --history_max ${HISTORY_MAX} \
    --K ${K} --prompt_num ${PROMPT_NUM} --lamb ${LAMB} --emb_size ${EMB_SIZE} --attn_size ${ATTN_SIZE} --n_layers ${N_LAYERS} \
    --use_llmemb 1 --llm_fuse 1 \
    --llm_emb_path ${LLM_EMB_PATH} \
    --srs_emb_path ${SRS_EMB_PATH} \
    --gamma_init ${GAMMA} --gamma_trainable 0 \
    --alpha ${ALIGN_ALPHA} --tau ${TAU} --rat_alpha_warmup_steps ${RAT_WARMUP} \
    --init_ckpt "${INIT_CKPT}" --init_strict 0 \
    --use_emile 0 \
    --use_logic_denoise 0 \
    --use_logic_aggr 0 --lambda_logic_aggr 0.0 \
    --lambda_ilr 0 \
    --log_file  "${LOGDIR}/${tag}.txt" \
    --model_path "${MODELDIR}/${tag}.pt" \
    > "${out}" 2>&1 &

  echo "Launched ${tag} -> ${out}"
  sleep 0.2

  # =========================
  # 2) LLM + IPD
  # =========================
  wait_for_slot
  tag="ml1m_s${SEED}_llm_ipd"
  out="${LOGDIR}/nohup_${tag}.out"

  nohup python main.py \
    --model_name MyModel --dataset ${DATASET} \
    --lr ${LR} --l2 ${L2} \
    --batch_size ${BATCH_SIZE} --eval_batch_size ${EVAL_BATCH_SIZE} \
    --epoch ${EPOCH} --early_stop ${EARLY_STOP} \
    --num_neg ${NUM_NEG} --dropout ${DROPOUT} --num_workers ${NUM_WORKERS} \
    --random_seed ${SEED} --load 0 \
    --history_max ${HISTORY_MAX} \
    --K ${K} --prompt_num ${PROMPT_NUM} --lamb ${LAMB} --emb_size ${EMB_SIZE} --attn_size ${ATTN_SIZE} --n_layers ${N_LAYERS} \
    --use_llmemb 1 --llm_fuse 1 \
    --llm_emb_path ${LLM_EMB_PATH} \
    --srs_emb_path ${SRS_EMB_PATH} \
    --gamma_init ${GAMMA} --gamma_trainable 0 \
    --alpha ${ALIGN_ALPHA} --tau ${TAU} --rat_alpha_warmup_steps ${RAT_WARMUP} \
    --init_ckpt "${INIT_CKPT}" --init_strict 0 \
    --use_emile 1 --lambda_ipd ${IPD_LAMBDA} --ipd_margin ${IPD_MARGIN} --emile_use_fused_itememb 0 \
    --emile_warmup_steps ${EMILE_WARMUP} \
    --use_logic_denoise 0 \
    --use_logic_aggr 0 --lambda_logic_aggr 0.0 \
    --lambda_ilr 0 \
    --log_file  "${LOGDIR}/${tag}.txt" \
    --model_path "${MODELDIR}/${tag}.pt" \
    > "${out}" 2>&1 &

  echo "Launched ${tag} -> ${out}"
  sleep 0.2
done

wait
echo "Done. Logs in ${LOGDIR}"