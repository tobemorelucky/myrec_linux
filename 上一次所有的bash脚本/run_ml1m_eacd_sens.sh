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

LOGDIR="./log/MyModel/${DATASET}/eacd_sens_s42"
MODELDIR="./model/MyModel/${DATASET}/eacd_sens_s42"
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

# ===== EACD 旧接口参数（保留）=====
LGD_ALPHA=8
LGD_B=0.2
LGD_R=0.08
LGD_WARMUP=50000

# ===== EACD 新参数默认值 =====
DENOISE_TAU=0.2
DENOISE_PRIOR_BETA=1.0
DENOISE_EXTRACTOR_LAMBDA=1.0
LAMBDA_EA=0.02

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
    --use_logic_denoise 1 \
    --logic_denoise_alpha ${LGD_ALPHA} --logic_denoise_b ${LGD_B} \
    --logic_denoise_r ${LGD_R} \
    --logic_denoise_warmup_steps ${LGD_WARMUP} \
    --denoise_tau ${DENOISE_TAU} \
    --denoise_prior_beta ${DENOISE_PRIOR_BETA} \
    --denoise_extractor_lambda ${DENOISE_EXTRACTOR_LAMBDA} \
    --lambda_ea ${LAMBDA_EA} \
    "${extra_args[@]}" \
    --log_file  "${LOGDIR}/${tag}.txt" \
    --model_path "${MODELDIR}/${tag}.pt" \
    > "${out}" 2>&1 &

  running=$((running+1))
  sleep 0.2
}

########################################
# 1) sweep lambda_ea
########################################
for v in 0.005 0.02 0.05; do
  tag="ml1m_eacd_s42_ea${v}"
  launch_one "${tag}" \
    --lambda_ea "${v}"
done

########################################
# 2) sweep denoise_prior_beta
########################################
for v in 0.5 1.0 1.5; do
  tag="ml1m_eacd_s42_beta${v}"
  launch_one "${tag}" \
    --denoise_prior_beta "${v}"
done

########################################
# 3) sweep denoise_extractor_lambda
########################################
for v in 0.5 1.0 1.5; do
  tag="ml1m_eacd_s42_elam${v}"
  launch_one "${tag}" \
    --denoise_extractor_lambda "${v}"
done

wait
echo "ML-1M EACD sensitivity finished. Logs in ${LOGDIR}"