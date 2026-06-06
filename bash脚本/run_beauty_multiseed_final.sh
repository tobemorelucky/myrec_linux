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
# 数据与日志目录（新建子文件夹）
########################################
DATASET="beauty"
SUBDIR="final_multiseed_lgdB_b03"
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
# LGD best (你已验证 b=0.3 最优)
########################################
LGD_ALPHA=8.0
LGD_B=0.3
LGD_TOPK=5
LGD_R=0.15
LGD_WARMUP=20000

launch_one () {
  local seed="$1"
  wait_for_slot

  local init_ckpt="./model/PoMRec/PoMRec__beauty__${seed}__lr=0.002__l2=1e-06.pt"
  if [ ! -f "${init_ckpt}" ]; then
    echo "[WARN] init_ckpt not found for seed=${seed}, fallback to ${CKPT_FALLBACK}"
    init_ckpt="${CKPT_FALLBACK}"
  fi

  local tag="beauty_final_llm_ipd_lgdB_a${LGD_ALPHA}_b03_top${LGD_TOPK}_r015_s${seed}"
  local out="${LOGDIR}/nohup_${tag}.out"

  echo "Launch ${tag} -> ${out}"

  nohup python main.py \
    --model_name MyModel --dataset ${DATASET} \
    --lr 0.002 --l2 1e-06 \
    --batch_size 256 --eval_batch_size 256 \
    --epoch 200 --early_stop 10 \
    --num_neg 1 --dropout 0 --num_workers 5 \
    --random_seed ${seed} --load 0 \
    --history_max 20 \
    --K 3 --prompt_num 4 --lamb 3.0 --emb_size 64 --attn_size 8 --n_layers 1 \
    --use_llmemb 1 --llm_fuse 1 \
    --llm_emb_path ./data/beauty/handled/llm_table_pca1536.pkl \
    --srs_emb_path ./data/beauty/handled/itm_emb_pomrec.pkl \
    --gamma_init 0.1 --gamma_trainable 0 \
    --alpha 0.001 --tau 0.2 --rat_alpha_warmup_steps 5000 \
    --init_ckpt "${init_ckpt}" --init_strict 0 \
    --use_emile 1 --lambda_ipd 0.05 --ipd_margin 0.2 --emile_use_fused_itememb 0 \
    --emile_warmup_steps 5000 \
    --use_logic_denoise 1 \
    --logic_denoise_alpha ${LGD_ALPHA} \
    --logic_denoise_b ${LGD_B} \
    --logic_denoise_topk ${LGD_TOPK} \
    --logic_denoise_r ${LGD_R} \
    --logic_denoise_warmup_steps ${LGD_WARMUP} \
    --use_logic_aggr 0 --lambda_logic_aggr 0.0 \
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
echo "Beauty final multiseed finished. Logs in ${LOGDIR}"