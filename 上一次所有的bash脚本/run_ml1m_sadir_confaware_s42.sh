#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

########################################
# 并行控制
########################################
MAX_PROCS=1
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
DATASET="ml-1m"
SUBDIR="sadir_confaware_s42"
LOGDIR="./log/MyModel/${DATASET}/${SUBDIR}"
MODELDIR="./model/MyModel/${DATASET}/${SUBDIR}"
mkdir -p "${LOGDIR}" "${MODELDIR}"

########################################
# Seeds
########################################
SEEDS=(42)

########################################
# init_ckpt
########################################
INIT_CKPT="./model/PoMRec/PoMRec__ml-1m__1__lr=0.001__l2=1e-06.pt"

########################################
# 固定基础参数（沿用你原 ml-1m）
########################################
LR=0.001
L2=1e-06
BATCH=256
EVAL_BATCH=256
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

########################################
# 第一模块
########################################
USE_LLMEMB=1
LLM_FUSE=1
LLM_EMB_PATH="./data/ml-1m/handled/llm_table_pca1536.pkl"
SRS_EMB_PATH="./data/ml-1m/handled/itm_emb_pomrec.pkl"
GAMMA_INIT=0.08
GAMMA_TRAINABLE=0

ALIGN_ALPHA=0.001
TAU=0.3
RAT_WARMUP=5000

########################################
# 第三模块
########################################
USE_EMILE=1
LAMBDA_IPD=0.02
IPD_MARGIN=0.10
EMILE_USE_FUSED_ITEMEMB=0
EMILE_WARMUP=20000

USE_LOGIC_AGGR=0
LAMBDA_LOGIC_AGGR=0.0

########################################
# 旧 LGD fallback 关闭，但 warmup 仍给 SADIR 用
########################################
USE_LOGIC_DENOISE=0
LOGIC_DENOISE_ALPHA=8
LOGIC_DENOISE_B=0.2
LOGIC_DENOISE_TOPK=0
LOGIC_DENOISE_R=0.08
LOGIC_DENOISE_WARMUP=50000

########################################
# conf-aware SADIR 默认参数
########################################
USE_SADIR=1
SADIR_HIDDEN=128
SADIR_ASSIGN_TAU=0.5
SADIR_TRANSITION_LAMBDA_E=0.35
SADIR_TRANSITION_LAMBDA_A=0.85
SADIR_RESIDUAL_E=0.25
SADIR_RESIDUAL_A=0.20
SADIR_PRIOR_LAMBDA=0.30
SADIR_REFINER_HEADS=4
SADIR_REFINER_DROPOUT=0.10
LAMBDA_GATE_CONS=0.001

# 新增：confidence-aware
SADIR_CONF_MIX=0.5
SADIR_CONF_FLOOR=0.15

launch_one () {
  local seed="$1"
  local tag="$2"
  shift 2
  local extra_args=("$@")

  wait_for_slot

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
    --use_llmemb ${USE_LLMEMB} --llm_fuse ${LLM_FUSE} \
    --llm_emb_path ${LLM_EMB_PATH} \
    --srs_emb_path ${SRS_EMB_PATH} \
    --gamma_init ${GAMMA_INIT} --gamma_trainable ${GAMMA_TRAINABLE} \
    --alpha ${ALIGN_ALPHA} --tau ${TAU} --rat_alpha_warmup_steps ${RAT_WARMUP} \
    --init_ckpt "${INIT_CKPT}" --init_strict 0 \
    --use_emile ${USE_EMILE} --lambda_ipd ${LAMBDA_IPD} --ipd_margin ${IPD_MARGIN} \
    --emile_use_fused_itememb ${EMILE_USE_FUSED_ITEMEMB} \
    --emile_warmup_steps ${EMILE_WARMUP} \
    --use_logic_aggr ${USE_LOGIC_AGGR} --lambda_logic_aggr ${LAMBDA_LOGIC_AGGR} \
    --use_logic_denoise ${USE_LOGIC_DENOISE} \
    --logic_denoise_alpha ${LOGIC_DENOISE_ALPHA} \
    --logic_denoise_b ${LOGIC_DENOISE_B} \
    --logic_denoise_topk ${LOGIC_DENOISE_TOPK} \
    --logic_denoise_r ${LOGIC_DENOISE_R} \
    --logic_denoise_warmup_steps ${LOGIC_DENOISE_WARMUP} \
    --use_sadir ${USE_SADIR} \
    --sadir_hidden ${SADIR_HIDDEN} \
    --sadir_assign_tau ${SADIR_ASSIGN_TAU} \
    --sadir_transition_lambda_e ${SADIR_TRANSITION_LAMBDA_E} \
    --sadir_transition_lambda_a ${SADIR_TRANSITION_LAMBDA_A} \
    --sadir_residual_e ${SADIR_RESIDUAL_E} \
    --sadir_residual_a ${SADIR_RESIDUAL_A} \
    --sadir_prior_lambda ${SADIR_PRIOR_LAMBDA} \
    --sadir_refiner_heads ${SADIR_REFINER_HEADS} \
    --sadir_refiner_dropout ${SADIR_REFINER_DROPOUT} \
    --lambda_gate_cons ${LAMBDA_GATE_CONS} \
    --sadir_conf_mix ${SADIR_CONF_MIX} \
    --sadir_conf_floor ${SADIR_CONF_FLOOR} \
    "${extra_args[@]}" \
    --log_file "${LOGDIR}/${tag}.txt" \
    --model_path "${MODELDIR}/${tag}.pt" \
    > "${out}" 2>&1 &

  running=$((running+1))
  sleep 0.2
}

for s in "${SEEDS[@]}"; do
  launch_one "${s}" "ml1m_confaware_mix05_floor010_s${s}" \
    --sadir_conf_mix 0.5 --sadir_conf_floor 0.10

  launch_one "${s}" "ml1m_confaware_mix05_floor015_s${s}" \
    --sadir_conf_mix 0.5 --sadir_conf_floor 0.15

  launch_one "${s}" "ml1m_confaware_mix07_floor010_s${s}" \
    --sadir_conf_mix 0.7 --sadir_conf_floor 0.10

  launch_one "${s}" "ml1m_confaware_mix07_floor015_s${s}" \
    --sadir_conf_mix 0.7 --sadir_conf_floor 0.15
done

wait
echo "ML-1M conf-aware SADIR finished. Logs in ${LOGDIR}"