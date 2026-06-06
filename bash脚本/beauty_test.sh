#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

########################################
# 最大并行数
########################################
MAX_PROCS=2
running=0

wait_for_slot () {
  while [ "$running" -ge "$MAX_PROCS" ]; do
    wait -n
    running=$((running-1))
  done
}

########################################
# 数据与目录
########################################
DATASET="beauty"
LOGDIR="./log/MyModel/${DATASET}"
MODELDIR="./model/MyModel"
mkdir -p "${LOGDIR}" "${MODELDIR}"

SEED=42

########################################
# 固定：你的 Beauty 最佳主干 + LLMemb 参数（你给的 COMMON）
########################################
COMMON=(
  --model_name MyModel --dataset ${DATASET}
  --lr 0.002 --l2 1e-06
  --batch_size 256 --eval_batch_size 256
  --epoch 200 --early_stop 10
  --num_neg 1 --dropout 0 --num_workers 5
  --random_seed ${SEED} --load 0

  --K 4 --prompt_num 3 --lamb 4.0 --emb_size 64 --attn_size 8 --n_layers 2
  --history_max 20

  --use_llmemb 1
  --llm_emb_path ./data/beauty/handled/llm_table_pca1536.pkl
  --srs_emb_path ./data/beauty/handled/itm_emb_pomrec.pkl
  --alpha 0.001 --tau 0.2 --rat_alpha_warmup_steps 5000 --align_on pos
  --llm_fuse 1 --gamma_init 0.1 --gamma_trainable 0

  --init_ckpt ./model/PoMRec/PoMRec__beauty__42__lr=0.002__l2=1e-06.pt --init_strict 0
)

########################################
# 固定：你说的 IPD 最佳（按你之前脚本习惯：ILR=0）
# 这里默认就是 lambda_ipd=0.05，你的日志名也写 ipd005
########################################
IPD=(
  --use_emile 1
  --lambda_ipd 0.05
  --ipd_margin 0.10
  --emile_use_fused_itememb 0
)

########################################
# 扫描：Entropy-aware Logic（Beauty 版）
########################################
# 4 组就够（2x2）：
# - Beauty entropy 通常更高，所以阈值可以更低（0.55/0.60）
# - logic_beta 可以更大（0.10/0.20）
THRES=(0.55 0.60)
LBETA=(0.10 0.20)

# 建议先用 topk=2（贴合你之前 “lgd_top2” 的习惯）
TOPK=2

for th in "${THRES[@]}"; do
  for lb in "${LBETA[@]}"; do
    wait_for_slot

    ths="${th//./}"
    lbs="${lb//./}"
    tag="ipd005_entT${ths}_lb${lbs}_top${TOPK}"

    out="${LOGDIR}/nohup_${tag}_s${SEED}.out"
    echo "Launch ${tag} -> ${out}"

    nohup python main.py \
      "${COMMON[@]}" \
      "${IPD[@]}" \
      --use_logic_denoise 1 \
      --entropy_thres "${th}" --entropy_beta 15 \
      --logic_beta "${lb}" --logic_topk "${TOPK}" \
      --logic_warmup_steps 20000 --logic_skip_g 0.10 \
      --log_file  "${LOGDIR}/${tag}_s${SEED}.txt" \
      --model_path "${MODELDIR}/${tag}_s${SEED}.pt" \
      > "${out}" 2>&1 &

    running=$((running+1))
    sleep 0.2
  done
done

wait
echo "All Beauty entropy-aware logic scan jobs finished. Logs in ${LOGDIR}"