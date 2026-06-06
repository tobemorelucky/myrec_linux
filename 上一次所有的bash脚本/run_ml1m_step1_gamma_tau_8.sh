#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

MAX_PROCS=2
running=0
wait_for_slot () { while [ "$running" -ge "$MAX_PROCS" ]; do wait -n; running=$((running-1)); done; }

DATASET="ml-1m"
LOGDIR="./log/MyModel/${DATASET}/resweep_v3_step1_llm"
MODELDIR="./model/MyModel/${DATASET}/resweep_v3_step1_llm"
mkdir -p "${LOGDIR}" "${MODELDIR}"

SEED=42
INIT_CKPT="./model/PoMRec/PoMRec__ml-1m__1__lr=0.001__l2=1e-06.pt"

# 8组：gamma x tau
GAMMAS=(0.03 0.05 0.08 0.10)
TAUS=(0.3 0.5)

for tau in "${TAUS[@]}"; do
  for g in "${GAMMAS[@]}"; do
    wait_for_slot

    g_s="${g//./}"
    t_s="${tau//./}"
    tag="ml1m_step1_g${g_s}_tau${t_s}_s${SEED}"
    out="${LOGDIR}/nohup_${tag}.out"

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
      --gamma_init ${g} --gamma_trainable 0 \
      --alpha 0.001 --tau ${tau} --rat_alpha_warmup_steps 5000 \
      --init_ckpt "${INIT_CKPT}" --init_strict 0 \
      --use_emile 0 \
      --use_logic_aggr 0 --lambda_logic_aggr 0.0 \
      --use_logic_denoise 0 --logic_denoise_alpha 1.0 --logic_denoise_b 0.0 \
      --logic_denoise_r 0.10 --logic_denoise_topk 0 --logic_denoise_warmup_steps 5000 \
      --log_file  "${LOGDIR}/${tag}.txt" \
      --model_path "${MODELDIR}/${tag}.pt" \
      > "${out}" 2>&1 &

    running=$((running+1))
    sleep 0.2
  done
done

wait
echo "ML-1M Step1 (gamma x tau) done. Logs in ${LOGDIR}"