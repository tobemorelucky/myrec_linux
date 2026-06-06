#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

MAX_PROCS=2
running=0

DATASET="ml-1m"
LOGDIR="./log/MyModel/${DATASET}"
MODELDIR="./model/MyModel"
mkdir -p "${LOGDIR}" "${MODELDIR}"

SEED=42

# 固定参数：包含之前的最佳参数 + 新确定的 IPD 参数
BASE=(
  --model_name MyModel --dataset ${DATASET}
  --lr 0.001 --l2 1e-06
  --batch_size 256 --eval_batch_size 256
  --epoch 200 --early_stop 10
  --num_neg 1 --dropout 0 --num_workers 5
  --random_seed ${SEED} --load 0
  --history_max 20

  --K 3 --prompt_num 4 --lamb 3 --emb_size 64 --attn_size 8 --n_layers 1

  --use_llmemb 1 --llm_fuse 1
  --llm_emb_path ./data/ml-1m/handled/llm_table_pca1536.pkl
  --srs_emb_path ./data/ml-1m/handled/itm_emb_pomrec.pkl
  --gamma_init 0.05 --gamma_trainable 0
  --alpha 0.001 --tau 0.5 --rat_alpha_warmup_steps 5000
  --align_on pos --align_sample_k 0
  --init_ckpt ./model/PoMRec/PoMRec__ml-1m__1__lr=0.001__l2=1e-06.pt --init_strict 0

  # 固定：已确定的最佳 IPD 参数
  --use_emile 1 --lambda_ipd 0.05 --lambda_ilr 0 --ipd_margin 0.10 --emile_use_fused_itememb 0
)

wait_for_slot () {
  while [ "$running" -ge "$MAX_PROCS" ]; do
    wait -n
    running=$((running-1))
  done
}

# 扫描 Logic Denoise 参数 (TopK: 0, 2; Alpha: 0.5, 1.0)
for topk in 0 2; do
  for alpha in 0.5 1.0; do
    wait_for_slot

    # 生成文件名友好的后缀 (将 0.5 转为 05)
    alpha_str="${alpha//./}"
    tag="lgd_top${topk}_a${alpha_str}"
    out="${LOGDIR}/nohup_C_s${SEED}_ipd005_${tag}.out"

    echo "Launch ${tag} -> ${out}"

    nohup python main.py \
      "${BASE[@]}" \
      --use_logic_denoise 1 \
      --logic_denoise_topk "${topk}" \
      --logic_denoise_alpha "${alpha}" \
      --logic_denoise_use_fused 0 \
      --log_file  "${LOGDIR}/C_ipd005_${tag}_s${SEED}.txt" \
      --model_path "${MODELDIR}/C_ml1m_s${SEED}_ipd005_${tag}.pt" \
      > "${out}" 2>&1 &

    running=$((running+1))
    sleep 0.2
  done
done

wait
echo "All Logic Denoise scan jobs finished. Logs in ${LOGDIR}"