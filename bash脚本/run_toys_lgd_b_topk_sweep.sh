#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1

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

# 固定 toys 最优（LLM + IPD）
TAU=0.5
GAMMA=0.05
ALPHA=0.001
RAT_WARMUP=5000

LIPD=0.05
IPD_MARGIN=0.10
EMILE_WARMUP=20000

# 固定 LGD（扫 b/topk）
LGD_ALPHA=10
LGD_R=0.10
LGD_WARMUP=50000
B_LIST=(0.1 0.2 0.3)
TOPK_LIST=(5 10)

for b in "${B_LIST[@]}"; do
  for topk in "${TOPK_LIST[@]}"; do
    wait_for_slot
    b_s="${b//./}"
    tag="toys_llm_ipd_lgdB_tau${TAU}_g${GAMMA}_lipd005_m010_a${LGD_ALPHA}_b${b_s}_top${topk}_r010_s${SEED}"
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
      --alpha ${ALPHA} --tau ${TAU} --rat_alpha_warmup_steps ${RAT_WARMUP} \
      --init_ckpt "${INIT_CKPT}" --init_strict 0 \
      --use_emile 1 --lambda_ipd ${LIPD} --ipd_margin ${IPD_MARGIN} \
      --emile_use_fused_itememb 0 --emile_warmup_steps ${EMILE_WARMUP} \
      --lambda_ilr 0 \
      --use_logic_denoise 1 \
      --logic_denoise_alpha ${LGD_ALPHA} \
      --logic_denoise_b ${b} \
      --logic_denoise_topk ${topk} \
      --logic_denoise_r ${LGD_R} \
      --logic_denoise_warmup_steps ${LGD_WARMUP} \
      --use_logic_aggr 0 --lambda_logic_aggr 0.0 \
      --log_file  "${LOGDIR}/${tag}.txt" \
      --model_path "${MODELDIR}/${tag}.pt" \
      > "${out}" 2>&1 &

    running=$((running+1))
    sleep 0.2
  done
done

wait
echo "Toys LGD (b,topk) sweep done. Logs in ${LOGDIR}"