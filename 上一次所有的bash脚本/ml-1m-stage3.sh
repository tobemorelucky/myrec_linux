#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

MAX_PROCS=2
running=0
wait_for_slot () { while [ "$running" -ge "$MAX_PROCS" ]; do wait -n; running=$((running-1)); done; }

DATASET="ml-1m"
LOGDIR="./log/MyModel/${DATASET}/stage3_logic"
MODELDIR="./model/MyModel/${DATASET}/stage3_logic"
mkdir -p "${LOGDIR}" "${MODELDIR}"

SEED=42
# 固定 backbone + LLM + IPD（等 Stage2 最优出来后替换下面三行）
K=3; LAMB=3.5
GAMMA=0.05; TAU=0.5; ALPHA=0.0005
LAMBDA_IPD=0.07; MARGIN=0.10

INIT_CKPT="./model/PoMRec/PoMRec__ml-1m__1__lr=0.001__l2=1e-06.pt"

# 只扫 3 个最关键维度：lambda_max / temp / gate_b（24组）
LAMBDA_MAX=(0.12 0.14)
TEMP=(2.0 2.5 3.0 )
GATE_B=(0.80 0.85 )

for lm in "${LAMBDA_MAX[@]}"; do
  for tp in "${TEMP[@]}"; do
    for gb in "${GATE_B[@]}"; do
      wait_for_slot

      lm_s="${lm//./}"
      tp_s="${tp//./}"
      gb_s="${gb//./}"
      tag="ml1m_logic_lm${lm_s}_t${tp_s}_b${gb_s}_s${SEED}"
      out="${LOGDIR}/nohup_${tag}.out"

      nohup python main.py \
        --model_name MyModel --dataset ${DATASET} \
        --lr 0.001 --l2 1e-06 \
        --batch_size 256 --eval_batch_size 256 \
        --epoch 200 --early_stop 10 \
        --num_neg 1 --dropout 0 --num_workers 5 \
        --random_seed ${SEED} --load 0 \
        --history_max 20 \
        --K "${K}" --prompt_num 4 --lamb "${LAMB}" --emb_size 64 --attn_size 8 --n_layers 1 \
        --use_llmemb 1 --llm_fuse 1 \
        --llm_emb_path ./data/ml-1m/handled/llm_table_pca1536.pkl \
        --srs_emb_path ./data/ml-1m/handled/itm_emb_pomrec.pkl \
        --gamma_init "${GAMMA}" --gamma_trainable 0 \
        --alpha "${ALPHA}" --tau "${TAU}" --rat_alpha_warmup_steps 5000 \
        --init_ckpt "${INIT_CKPT}" --init_strict 0 \
        --use_emile 1 --lambda_ipd "${LAMBDA_IPD}" --lambda_ilr 0 --ipd_margin "${MARGIN}" --emile_use_fused_itememb 0 \
        --use_logic_denoise 0 \
        --use_logic_aggr 1 \
        --lambda_logic_aggr 0.3 \
        --logic_lambda_max "${lm}" \
        --logic_support_temp "${tp}" \
        --logic_gate_a 10 \
        --logic_gate_b "${gb}" \
        --log_file  "${LOGDIR}/${tag}.txt" \
        --model_path "${MODELDIR}/${tag}.pt" \
        > "${out}" 2>&1 &

      running=$((running+1))
      sleep 0.2
    done
  done
done

wait
echo "Stage3 (aggr-only) done. Logs in ${LOGDIR}"