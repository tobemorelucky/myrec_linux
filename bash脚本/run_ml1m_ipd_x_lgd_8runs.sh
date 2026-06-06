#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

MAX_PROCS=2
running=0
wait_for_slot () { while [ "$running" -ge "$MAX_PROCS" ]; do wait -n; running=$((running-1)); done; }

DATASET="ml-1m"
LOGDIR="./log/MyModel/${DATASET}/resweep_v5_ipdXlgd"
MODELDIR="./model/MyModel/${DATASET}/resweep_v5_ipdXlgd"
mkdir -p "${LOGDIR}" "${MODELDIR}"

SEED=42
INIT_CKPT="./model/PoMRec/PoMRec__ml-1m__1__lr=0.001__l2=1e-06.pt"

# 固定你已标定好的 LLM 空间
TAU=0.3
GAMMA=0.08
ALIGN_ALPHA=0.001
RAT_WARMUP=5000

# IPD 扫描（第二模块）
LIPDS=(0 0.005 0.01 0.02)
IPD_MARGIN=0.10
EMILE_WARMUP=20000   # 更温和，避免 ml-1m test 掉

# LGD 固定（第三模块：你修改过的 b/topk gate）
LGD_ALPHA=8
LGD_B=0.2
LGD_TOPK=5
LGD_R=0.08
LGD_WARMUP=50000

for use_lgd in 0 1; do
  for lipd in "${LIPDS[@]}"; do
    wait_for_slot

    lipd_s="${lipd//./}"
    if [ "${lipd_s}" = "" ]; then lipd_s="0"; fi

    if [ "${use_lgd}" -eq 1 ]; then
      tag="ml1m_ipd${lipd_s}_LGD_on_g008_tau03_s${SEED}"
    else
      tag="ml1m_ipd${lipd_s}_LGD_off_g008_tau03_s${SEED}"
    fi
    out="${LOGDIR}/nohup_${tag}.out"

    # use_emile：lipd=0 时直接关掉更干净
    if [ "${lipd}" = "0" ]; then
      USE_EMILE=0
    else
      USE_EMILE=1
    fi

    # LGD 参数：lgd=0 时给默认但不启用
    if [ "${use_lgd}" -eq 1 ]; then
      USE_LGD=1
      denoise_alpha=${LGD_ALPHA}
      denoise_b=${LGD_B}
      denoise_topk=${LGD_TOPK}
      denoise_r=${LGD_R}
      denoise_warm=${LGD_WARMUP}
    else
      USE_LGD=0
      denoise_alpha=1.0
      denoise_b=0.0
      denoise_topk=0
      denoise_r=0.10
      denoise_warm=5000
    fi

    echo "Launch ${tag} (use_emile=${USE_EMILE}, lipd=${lipd}, use_lgd=${USE_LGD}) -> ${out}"

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
      --use_emile ${USE_EMILE} \
      --lambda_ipd ${lipd} --ipd_margin ${IPD_MARGIN} --emile_use_fused_itememb 0 \
      --emile_warmup_steps ${EMILE_WARMUP} \
      --lambda_ilr 0 \
      --use_logic_aggr 0 --lambda_logic_aggr 0.0 \
      --use_logic_denoise ${USE_LGD} \
      --logic_denoise_alpha ${denoise_alpha} --logic_denoise_b ${denoise_b} \
      --logic_denoise_topk ${denoise_topk} --logic_denoise_r ${denoise_r} \
      --logic_denoise_warmup_steps ${denoise_warm} \
      --log_file  "${LOGDIR}/${tag}.txt" \
      --model_path "${MODELDIR}/${tag}.pt" \
      > "${out}" 2>&1 &

    running=$((running+1))
    sleep 0.2
  done
done

wait
echo "Done. Logs in ${LOGDIR}"