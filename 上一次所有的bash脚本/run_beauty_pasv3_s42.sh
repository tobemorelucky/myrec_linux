#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1

MAX_PROCS=1
running=0
wait_for_slot () {
  while [ "$running" -ge "$MAX_PROCS" ]; do
    wait -n
    running=$((running-1))
  done
}

DATASET="beauty"
SEED=42

SUBDIR="pasv3_lgd_s42_small_sweep"
LOGDIR="./log/MyModel/${DATASET}/${SUBDIR}"
MODELDIR="./model/MyModel/${DATASET}/${SUBDIR}"
mkdir -p "${LOGDIR}" "${MODELDIR}"

CKPT_FALLBACK="./model/PoMRec/PoMRec__beauty__42__lr=0.002__l2=1e-06.pt"

########################################
# 基础训练参数
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
# 第一模块：LLM 语义注入
########################################
GAMMA=0.1
ALIGN_ALPHA=0.001
TAU=0.2
RAT_WARMUP=5000

########################################
# 第三模块：TIC/IPD
########################################
LIPD=0.05
IPD_MARGIN=0.2
EMILE_WARMUP=5000

########################################
# 原 LGD：保留作为稳定底座
########################################
USE_LOGIC_DENOISE=1
LGD_ALPHA=8.0
LGD_B=0.3
LGD_TOPK=5
LGD_R=0.15
LGD_WARMUP=20000

########################################
# PAS-v3 默认参数
########################################
USE_PAS=1
PAS_HIDDEN=128
PAS_TEMP=0.5
PAS_HARD=0
PAS_USE_GUMBEL=0

PAS_E_QUOTA_LONG=3
PAS_E_QUOTA_MID=4
PAS_E_QUOTA_RECENT=5

PAS_A_QUOTA_LONG=4
PAS_A_QUOTA_MID=5
PAS_A_QUOTA_RECENT=6

PAS_RATE_E=0.75
PAS_RATE_A=0.85

LAMBDA_SAMPLER_RATE=0.01
LAMBDA_SAMPLER_PERIOD=0.01

PAS_BIAS_CLIP=2.0

launch_one () {
  local tag="$1"
  shift
  local extra_args=("$@")

  wait_for_slot

  local init_ckpt="./model/PoMRec/PoMRec__beauty__${SEED}__lr=0.002__l2=1e-06.pt"
  if [ ! -f "${init_ckpt}" ]; then
    echo "[WARN] init_ckpt not found for seed=${SEED}, fallback to ${CKPT_FALLBACK}"
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
    --random_seed ${SEED} --load 0 \
    --history_max ${HISTORY_MAX} \
    --K ${K} --prompt_num ${PROMPT_NUM} --lamb ${LAMB} \
    --emb_size ${EMB_SIZE} --attn_size ${ATTN_SIZE} --n_layers ${N_LAYERS} \
    --use_llmemb 1 --llm_fuse 1 \
    --llm_emb_path ./data/beauty/handled/llm_table_pca1536.pkl \
    --srs_emb_path ./data/beauty/handled/itm_emb_pomrec.pkl \
    --gamma_init ${GAMMA} --gamma_trainable 0 \
    --alpha ${ALIGN_ALPHA} --tau ${TAU} --rat_alpha_warmup_steps ${RAT_WARMUP} \
    --init_ckpt "${init_ckpt}" --init_strict 0 \
    --use_emile 1 --lambda_ipd ${LIPD} --ipd_margin ${IPD_MARGIN} --emile_use_fused_itememb 0 \
    --emile_warmup_steps ${EMILE_WARMUP} \
    --use_logic_aggr 0 --lambda_logic_aggr 0.0 \
    --use_logic_denoise ${USE_LOGIC_DENOISE} \
    --logic_denoise_alpha ${LGD_ALPHA} \
    --logic_denoise_b ${LGD_B} \
    --logic_denoise_topk ${LGD_TOPK} \
    --logic_denoise_r ${LGD_R} \
    --logic_denoise_warmup_steps ${LGD_WARMUP} \
    --use_pas ${USE_PAS} \
    --pas_hidden ${PAS_HIDDEN} \
    --pas_temp ${PAS_TEMP} \
    --pas_hard ${PAS_HARD} \
    --pas_use_gumbel ${PAS_USE_GUMBEL} \
    --pas_e_quota_long ${PAS_E_QUOTA_LONG} \
    --pas_e_quota_mid ${PAS_E_QUOTA_MID} \
    --pas_e_quota_recent ${PAS_E_QUOTA_RECENT} \
    --pas_a_quota_long ${PAS_A_QUOTA_LONG} \
    --pas_a_quota_mid ${PAS_A_QUOTA_MID} \
    --pas_a_quota_recent ${PAS_A_QUOTA_RECENT} \
    --pas_rate_e ${PAS_RATE_E} \
    --pas_rate_a ${PAS_RATE_A} \
    --lambda_sampler_rate ${LAMBDA_SAMPLER_RATE} \
    --lambda_sampler_period ${LAMBDA_SAMPLER_PERIOD} \
    --pas_bias_clip ${PAS_BIAS_CLIP} \
    "${extra_args[@]}" \
    --log_file "${LOGDIR}/${tag}.txt" \
    --model_path "${MODELDIR}/${tag}.pt" \
    > "${out}" 2>&1 &

  running=$((running+1))
  sleep 0.2
}

########################################
# 小范围测试
########################################

# 0) LGD-only 对照：关闭 PAS
launch_one "beauty_lgd_only_s42" \
  --use_pas 0 \
  --lambda_sampler_rate 0.0 \
  --lambda_sampler_period 0.0 \
  --lambda_sampler_sc_e 0.0 \
  --lambda_sampler_sc_a 0.0

# 1) 弱 bias，无 SC：Beauty 当前最稳方向
launch_one "beauty_pasv3_bE003_bA002_scE0000_s42" \
  --pas_bias_e 0.03 \
  --pas_bias_a 0.02 \
  --lambda_sampler_sc_e 0.0 \
  --lambda_sampler_sc_a 0.0

# 2) 推荐 bias，无 SC：复现 PAS-v2 当前最优逻辑
launch_one "beauty_pasv3_bE005_bA003_scE0000_s42" \
  --pas_bias_e 0.05 \
  --pas_bias_a 0.03 \
  --lambda_sampler_sc_e 0.0 \
  --lambda_sampler_sc_a 0.0

# 3) 推荐 bias，弱 extractor-side SC
launch_one "beauty_pasv3_bE005_bA003_scE0005_s42" \
  --pas_bias_e 0.05 \
  --pas_bias_a 0.03 \
  --lambda_sampler_sc_e 0.0005 \
  --lambda_sampler_sc_a 0.0

# 4) 弱 bias，弱 extractor-side SC
launch_one "beauty_pasv3_bE003_bA002_scE0005_s42" \
  --pas_bias_e 0.03 \
  --pas_bias_a 0.02 \
  --lambda_sampler_sc_e 0.0005 \
  --lambda_sampler_sc_a 0.0

# 5) 稍强 bias，无 SC
launch_one "beauty_pasv3_bE008_bA005_scE0000_s42" \
  --pas_bias_e 0.08 \
  --pas_bias_a 0.05 \
  --lambda_sampler_sc_e 0.0 \
  --lambda_sampler_sc_a 0.0

wait
echo "Beauty PAS-v3 small sweep finished. Logs in ${LOGDIR}"