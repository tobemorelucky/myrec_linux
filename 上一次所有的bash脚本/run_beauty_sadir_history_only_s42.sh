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
DATASET="beauty"
SUBDIR="sadir_history_only_sweep_s42"
LOGDIR="./log/MyModel/${DATASET}/${SUBDIR}"
MODELDIR="./model/MyModel/${DATASET}/${SUBDIR}"
mkdir -p "${LOGDIR}" "${MODELDIR}"

########################################
# Seeds
########################################
SEEDS=(42)

########################################
# init_ckpt：按 seed 找；缺失则 fallback
########################################
CKPT_FALLBACK="./model/PoMRec/PoMRec__beauty__42__lr=0.002__l2=1e-06.pt"

########################################
# 固定基础训练参数（沿用你原 beauty）
########################################
LR=0.002
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
# 第一模块（保持你原配置）
########################################
USE_LLMEMB=1
LLM_FUSE=1
LLM_EMB_PATH="./data/beauty/handled/llm_table_pca1536.pkl"
SRS_EMB_PATH="./data/beauty/handled/itm_emb_pomrec.pkl"
GAMMA_INIT=0.1
GAMMA_TRAINABLE=0

ALIGN_ALPHA=0.001
TAU=0.2
RAT_WARMUP=5000

########################################
# 第三模块（保持你原配置）
########################################
USE_EMILE=1
LAMBDA_IPD=0.05
IPD_MARGIN=0.2
EMILE_USE_FUSED_ITEMEMB=0
EMILE_WARMUP=5000

USE_LOGIC_AGGR=0
LAMBDA_LOGIC_AGGR=0.0

########################################
# 旧 LGD fallback 关闭，但 warmup 仍给 SADIR 用
########################################
USE_LOGIC_DENOISE=0
LOGIC_DENOISE_ALPHA=8.0
LOGIC_DENOISE_B=0.3
LOGIC_DENOISE_TOPK=0
LOGIC_DENOISE_R=0.15
LOGIC_DENOISE_WARMUP=20000

########################################
# SADIR history-only 默认参数
########################################
USE_SADIR=1
SADIR_HIDDEN=128
SADIR_ASSIGN_TAU=0.5
SADIR_TRANSITION_LAMBDA_E=0.35
SADIR_TRANSITION_LAMBDA_A=0.85
SADIR_RESIDUAL_E=0.25
SADIR_RESIDUAL_A=0.20
SADIR_PRIOR_LAMBDA=0.50
SADIR_REFINER_HEADS=4
SADIR_REFINER_DROPOUT=0.10
LAMBDA_GATE_CONS=0.001

launch_one () {
  local seed="$1"
  local tag="$2"
  shift 2
  local extra_args=("$@")

  wait_for_slot

  local init_ckpt="./model/PoMRec/PoMRec__beauty__${seed}__lr=0.002__l2=1e-06.pt"
  if [ ! -f "${init_ckpt}" ]; then
    echo "[WARN] init_ckpt not found for seed=${seed}, fallback to ${CKPT_FALLBACK}"
    init_ckpt="${CKPT_FALLBACK}"
  fi

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
    --init_ckpt "${init_ckpt}" --init_strict 0 \
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
    "${extra_args[@]}" \
    --log_file "${LOGDIR}/${tag}.txt" \
    --model_path "${MODELDIR}/${tag}.pt" \
    > "${out}" 2>&1 &

  running=$((running+1))
  sleep 0.2
}

########################################
# seed=42
########################################
for s in "${SEEDS[@]}"; do

  ##############################
  # 1) gate consistency
  ##############################
  launch_one "${s}" "beauty_sadir_ho_gate0000_s${s}" --lambda_gate_cons 0.0
  launch_one "${s}" "beauty_sadir_ho_gate0005_s${s}" --lambda_gate_cons 0.0005
  launch_one "${s}" "beauty_sadir_ho_gate0010_s${s}" --lambda_gate_cons 0.001
  launch_one "${s}" "beauty_sadir_ho_gate0020_s${s}" --lambda_gate_cons 0.002

  ##############################
  # 2) prior strength
  ##############################
  launch_one "${s}" "beauty_sadir_ho_prior030_s${s}" --sadir_prior_lambda 0.30
  launch_one "${s}" "beauty_sadir_ho_prior050_s${s}" --sadir_prior_lambda 0.50
  launch_one "${s}" "beauty_sadir_ho_prior070_s${s}" --sadir_prior_lambda 0.70
  launch_one "${s}" "beauty_sadir_ho_prior090_s${s}" --sadir_prior_lambda 0.90

  ##############################
  # 3) transition routing
  ##############################
  launch_one "${s}" "beauty_sadir_ho_trans_e025_a070_s${s}" \
    --sadir_transition_lambda_e 0.25 --sadir_transition_lambda_a 0.70
  launch_one "${s}" "beauty_sadir_ho_trans_e035_a085_s${s}" \
    --sadir_transition_lambda_e 0.35 --sadir_transition_lambda_a 0.85
  launch_one "${s}" "beauty_sadir_ho_trans_e045_a100_s${s}" \
    --sadir_transition_lambda_e 0.45 --sadir_transition_lambda_a 1.00

  ##############################
  # 4) residual strength
  ##############################
  launch_one "${s}" "beauty_sadir_ho_res_e015_a012_s${s}" \
    --sadir_residual_e 0.15 --sadir_residual_a 0.12
  launch_one "${s}" "beauty_sadir_ho_res_e025_a020_s${s}" \
    --sadir_residual_e 0.25 --sadir_residual_a 0.20
  launch_one "${s}" "beauty_sadir_ho_res_e035_a028_s${s}" \
    --sadir_residual_e 0.35 --sadir_residual_a 0.28

done

wait
echo "Beauty history-only SADIR sweep finished. Logs in ${LOGDIR}"