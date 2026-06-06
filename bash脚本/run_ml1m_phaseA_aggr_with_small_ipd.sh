#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

MAX_PROCS=2
running=0
wait_for_slot () { while [ "$running" -ge "$MAX_PROCS" ]; do wait -n; running=$((running-1)); done; }

DATASET="ml-1m"
LOGDIR="./log/MyModel/${DATASET}/resweep_v4_A_aggr_ipd001"
MODELDIR="./model/MyModel/${DATASET}/resweep_v4_A_aggr_ipd001"
mkdir -p "${LOGDIR}" "${MODELDIR}"

SEED=42
INIT_CKPT="./model/PoMRec/PoMRec__ml-1m__1__lr=0.001__l2=1e-06.pt"

# 固定你现在标定好的 LLM
TAU=0.3
GAMMA=0.08
ALPHA=0.001
RAT_WARMUP=5000

# IPD 固定开一点
LIPD=0.01
IPD_MARGIN=0.10
EMILE_WARMUP=20000

# 扫 aggr
AGGR_LAMS=(0.05 0.10 0.20)

for laggr in "${AGGR_LAMS[@]}"; do
  wait_for_slot
  laggr_s="${laggr//./}"
  tag="ml1m_A_aggr_laggr${laggr_s}_lipd001_g008_tau03_s${SEED}"
  out="${LOGDIR}/nohup_${tag}.out"

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
    --alpha ${ALPHA} --tau ${TAU} --rat_alpha_warmup_steps ${RAT_WARMUP} \
    --init_ckpt "${INIT_CKPT}" --init_strict 0 \
    --use_emile 1 --lambda_ipd ${LIPD} --ipd_margin ${IPD_MARGIN} --emile_use_fused_itememb 0 \
    --emile_warmup_steps ${EMILE_WARMUP} \
    --lambda_ilr 0 \
    --use_logic_aggr 1 --lambda_logic_aggr ${laggr} \
    --logic_lambda_max 0.10 --logic_support_temp 2.5 --logic_gate_a 10 --logic_gate_b 0.80 \
    --use_logic_denoise 0 --logic_denoise_b 0.0 --logic_denoise_topk 0 \
    --log_file  "${LOGDIR}/${tag}.txt" \
    --model_path "${MODELDIR}/${tag}.pt" \
    > "${out}" 2>&1 &

  running=$((running+1))
  sleep 0.2
done

wait
echo "Phase A done. Logs in ${LOGDIR}"