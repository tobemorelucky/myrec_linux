#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

MAX_PROCS=2
wait_for_slot () { while [ "$(jobs -rp | wc -l)" -ge "$MAX_PROCS" ]; do sleep 3; done; }

DATASET="beauty"
SEEDS=(0 1 2 3)

SUBDIR="sens_alpha_multiseed"
LOGDIR="./log/MyModel/${DATASET}/${SUBDIR}"
MODELDIR="./model/MyModel/${DATASET}/${SUBDIR}"
mkdir -p "${LOGDIR}" "${MODELDIR}"

# best fixed
TAU=0.2
GAMMA=0.10
RAT_WARMUP=5000

LIPD=0.05
IPD_MARGIN=0.2
EMILE_WARMUP=5000

LGD_ALPHA=8.0
LGD_B=0.3
LGD_TOPK=5
LGD_R=0.15
LGD_WARMUP=20000

INIT_CKPT="./model/PoMRec/PoMRec__beauty__42__lr=0.002__l2=1e-06.pt"

ALPHAS=(0.0006 0.0008  0.0012 0.0014)

for seed in "${SEEDS[@]}"; do
  for a in "${ALPHAS[@]}"; do
    wait_for_slot
    tag="beauty_sens_alpha_a${a}_s${seed}"
    out="${LOGDIR}/nohup_${tag}.out"

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
      --gamma_init ${GAMMA} --gamma_trainable 0 \
      --alpha ${a} --tau ${TAU} --rat_alpha_warmup_steps ${RAT_WARMUP} \
      --init_ckpt "${INIT_CKPT}" --init_strict 0 \
      --use_emile 1 --lambda_ipd ${LIPD} --ipd_margin ${IPD_MARGIN} --emile_use_fused_itememb 0 \
      --emile_warmup_steps ${EMILE_WARMUP} \
      --use_logic_aggr 0 --lambda_logic_aggr 0.0 \
      --use_logic_denoise 1 \
      --logic_denoise_alpha ${LGD_ALPHA} --logic_denoise_b ${LGD_B} \
      --logic_denoise_topk ${LGD_TOPK} --logic_denoise_r ${LGD_R} \
      --logic_denoise_warmup_steps ${LGD_WARMUP} \
      --log_file  "${LOGDIR}/${tag}.txt" \
      --model_path "${MODELDIR}/${tag}.pt" \
      > "${out}" 2>&1 &

    echo "Launched ${tag} -> ${out}"
    sleep 0.2
  done
done

wait
echo "Done. Logs in ${LOGDIR}"