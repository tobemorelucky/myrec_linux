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
SEED=42

LOGDIR="./log/MyModel/${DATASET}/sadir_s42"
MODELDIR="./model/MyModel/${DATASET}/sadir_s42"
mkdir -p "${LOGDIR}" "${MODELDIR}"

INIT_CKPT="./model/PoMRec/PoMRec__ml-1m__1__lr=0.001__l2=1e-06.pt"

# ===== 固定原有较优参数 =====
TAU=0.3
GAMMA=0.08
ALIGN_ALPHA=0.001
RAT_WARMUP=5000

LIPD=0.02
IPD_MARGIN=0.10
EMILE_WARMUP=20000

# ===== 兼容旧接口，用作 warmup / fallback =====
USE_LOGIC_DENOISE=1
LOGIC_DENOISE_ALPHA=8
LOGIC_DENOISE_B=0.2
LOGIC_DENOISE_TOPK=0
LOGIC_DENOISE_R=0.08
LOGIC_DENOISE_WARMUP=50000

# ===== SADIR 默认参数（尽量温和）=====
USE_SADIR=1
SADIR_HIDDEN=128
SADIR_ASSIGN_TAU=0.5
SADIR_TRANSITION_LAMBDA_E=0.25
SADIR_TRANSITION_LAMBDA_A=0.75
SADIR_RESIDUAL_E=0.15
SADIR_RESIDUAL_A=0.12
SADIR_PRIOR_LAMBDA=0.30
SADIR_REFINER_HEADS=4
SADIR_REFINER_DROPOUT=0.10

LAMBDA_BRIDGE=0.005
BRIDGE_TEMP=0.20
LAMBDA_GATE_CONS=0.0005

launch_one () {
  local tag="$1"
  shift
  local extra_args=("$@")

  wait_for_slot

  local out="${LOGDIR}/nohup_${tag}.out"
  echo "Launch ${tag} -> ${out}"

  nohup python main.py \
    --model_name MyModel --dataset ${DATASET} \
    --lr 0.001 --l2 1e-06 \
    --batch_size 256 --eval_batch_size 256 \
    --epoch 200 --early_stop 10 \
    --num_neg 1 --dropout 0 --num_workers 5 \
    --random_seed ${SEED} --load 0 \
    --history_max 20 \
    --K 3 --prompt_num 4 --lamb 3.0 --emb_size 64 --attn_size 8 --n_layers 1 \
    --use_llmemb 1 --llm_fuse 1 \
    --llm_emb_path ./data/ml-1m/handled/llm_table_pca1536.pkl \
    --srs_emb_path ./data/ml-1m/handled/itm_emb_pomrec.pkl \
    --gamma_init ${GAMMA} --gamma_trainable 0 \
    --alpha ${ALIGN_ALPHA} --tau ${TAU} --rat_alpha_warmup_steps ${RAT_WARMUP} \
    --init_ckpt "${INIT_CKPT}" --init_strict 0 \
    --use_emile 1 --lambda_ipd ${LIPD} --ipd_margin ${IPD_MARGIN} --emile_use_fused_itememb 0 \
    --emile_warmup_steps ${EMILE_WARMUP} \
    --use_logic_aggr 0 --lambda_logic_aggr 0.0 \
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
    --lambda_bridge ${LAMBDA_BRIDGE} \
    --bridge_temp ${BRIDGE_TEMP} \
    --lambda_gate_cons ${LAMBDA_GATE_CONS} \
    "${extra_args[@]}" \
    --log_file "${LOGDIR}/${tag}.txt" \
    --model_path "${MODELDIR}/${tag}.pt" \
    > "${out}" 2>&1 &

  running=$((running+1))
  sleep 0.2
}

########################################
# 1) 扫 lambda_bridge
########################################
launch_one "ml1m_sadir_s42_bridge000" --lambda_bridge 0.0
launch_one "ml1m_sadir_s42_bridge002" --lambda_bridge 0.002
launch_one "ml1m_sadir_s42_bridge005" --lambda_bridge 0.005

########################################
# 2) 扫 lambda_gate_cons
########################################
launch_one "ml1m_sadir_s42_gate0000" --lambda_gate_cons 0.0
launch_one "ml1m_sadir_s42_gate0005" --lambda_gate_cons 0.0005
launch_one "ml1m_sadir_s42_gate0010" --lambda_gate_cons 0.001

########################################
# 3) 扫 sadir_prior_lambda
########################################
launch_one "ml1m_sadir_s42_prior020" --sadir_prior_lambda 0.20
launch_one "ml1m_sadir_s42_prior030" --sadir_prior_lambda 0.30
launch_one "ml1m_sadir_s42_prior040" --sadir_prior_lambda 0.40

wait
echo "ML-1M SADIR sweep finished. Logs in ${LOGDIR}"