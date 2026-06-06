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

DATASET="ml-1m"
LOGDIR="./log/MyModel/${DATASET}/final_multiseed_full3_s42cfg"
MODELDIR="./model/MyModel/${DATASET}/final_multiseed_full3_s42cfg"
mkdir -p "${LOGDIR}" "${MODELDIR}"

SEEDS=(0 1 2 3 41 42 43)

# 与你刚才一致的固定配置
INIT_CKPT="./model/PoMRec/PoMRec__ml-1m__1__lr=0.001__l2=1e-06.pt"

TAU=0.3
GAMMA=0.08
ALIGN_ALPHA=0.001
RAT_WARMUP=5000

LIPD=0.02
IPD_MARGIN=0.10
EMILE_WARMUP=20000

# 完整三模块：LGD 开启
LGD_ALPHA=8.0
LGD_B=0.40
LGD_TOPK=5
LGD_R=0.08
LGD_WARMUP=50000

launch_one () {
  local seed="$1"
  wait_for_slot

  local tag="ml1m_full3_s${seed}_g008_tau03_ipd002_lgdA8_b040_top5_r008"
  local out="${LOGDIR}/nohup_${tag}.out"

  echo "Launch ${tag} -> ${out}"

  nohup python main.py \
    --model_name MyModel --dataset ${DATASET} \
    --lr 0.001 --l2 1e-06 \
    --batch_size 256 --eval_batch_size 256 \
    --epoch 200 --early_stop 10 \
    --num_neg 1 --dropout 0 --num_workers 5 \
    --random_seed ${seed} --load 0 \
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
    --lambda_ilr 0 \
    --use_logic_aggr 0 --lambda_logic_aggr 0.0 \
    --use_logic_denoise 1 \
    --logic_denoise_alpha ${LGD_ALPHA} \
    --logic_denoise_b ${LGD_B} \
    --logic_denoise_topk ${LGD_TOPK} \
    --logic_denoise_r ${LGD_R} \
    --logic_denoise_warmup_steps ${LGD_WARMUP} \
    --log_file  "${LOGDIR}/${tag}.txt" \
    --model_path "${MODELDIR}/${tag}.pt" \
    > "${out}" 2>&1 &

  running=$((running+1))
  sleep 0.2
}

for s in "${SEEDS[@]}"; do
  launch_one "${s}"
done

wait
echo "ML-1M full 3-module multiseed finished. Logs in ${LOGDIR}"