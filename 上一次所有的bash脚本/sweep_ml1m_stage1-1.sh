#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1

MAX_PROCS=2
running=0
wait_for_slot () { while [ "$running" -ge "$MAX_PROCS" ]; do wait -n; running=$((running-1)); done; }

DATASET="ml-1m"
LOGDIR="./log/MyModel/${DATASET}/stage1_llm"
MODELDIR="./model/MyModel/${DATASET}/stage1_llm"
mkdir -p "${LOGDIR}" "${MODELDIR}"

SEED=42

# 你先把 Stage0 最优的 backbone 固定在这里（下面给的是你当前常用配置）
K=3
LAMB=3.5

# warm-start（路线A）：用 PoMRec ckpt 作为初始化锚点
INIT_CKPT="./model/PoMRec/PoMRec__ml-1m__1__lr=0.001__l2=1e-06.pt"

GAMMA=(0.05)
TAU=(0.2 0.5)
ALPHA=(0.0005 0.001)

for g in "${GAMMA[@]}"; do
  for tau in "${TAU[@]}"; do
    for a in "${ALPHA[@]}"; do
      wait_for_slot

      g_s="${g//./}"
      tau_s="${tau//./}"
      a_s="${a//./}"
      tag="ml1m_llm_g${g_s}_tau${tau_s}_a${a_s}_s${SEED}"
      out="${LOGDIR}/nohup_${tag}.out"

      nohup python main.py \
        --model_name MyModel --dataset ${DATASET} \
        --lr 0.0005 --l2 1e-06 \
        --batch_size 256 --eval_batch_size 256 \
        --epoch 200 --early_stop 10 \
        --num_neg 1 --dropout 0 --num_workers 5 \
        --random_seed ${SEED} --load 0 \
        --history_max 20 \
        --K "${K}" --prompt_num 4 --lamb "${LAMB}" --emb_size 64 --attn_size 8 --n_layers 1 \
        --use_llmemb 1 --llm_fuse 1 \
        --llm_emb_path ./data/ml-1m/handled/llm_table_pca1536.pkl \
        --srs_emb_path ./data/ml-1m/handled/itm_emb_pomrec.pkl \
        --gamma_init "${g}" --gamma_trainable 0 \
        --alpha "${a}" --tau "${tau}" --rat_alpha_warmup_steps 5000 \
        --init_ckpt "${INIT_CKPT}" --init_strict 0 \
        --use_emile 0 \
        --use_logic_denoise 0 --use_logic_aggr 0 \
        --log_file  "${LOGDIR}/${tag}.txt" \
        --model_path "${MODELDIR}/${tag}.pt" \
        > "${out}" 2>&1 &

      running=$((running+1))
      sleep 0.2
    done
  done
done

wait
echo "Stage1 done. Logs in ${LOGDIR}"