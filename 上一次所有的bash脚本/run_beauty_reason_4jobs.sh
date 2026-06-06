#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

MAX_PROCS=2
running=0

DATASET="beauty"
LOGDIR="./log/MyModel/${DATASET}"
MODELDIR="./model/MyModel"
mkdir -p "${LOGDIR}" "${MODELDIR}"
SEED=42

# ===== 固定：Beauty 最佳主干超参（保持不变）=====
COMMON=(
  --model_name MyModel --dataset ${DATASET}
  --lr 0.002 --l2 1e-06
  --batch_size 256 --eval_batch_size 256
  --epoch 200 --early_stop 10
  --num_neg 1 --dropout 0 --num_workers 5
  --random_seed ${SEED} --load 0
  --history_max 20

  # PoMRec backbone（保持你最佳）
  --K 4 --prompt_num 3 --lamb 4.0 --emb_size 64 --attn_size 8 --n_layers 2

  # LLMemb
  --use_llmemb 1 --llm_fuse 1
  --llm_emb_path ./data/beauty/handled/llm_table_pca1536.pkl
  --srs_emb_path ./data/beauty/handled/itm_emb_pomrec.pkl
  --alpha 0.001 --tau 0.2 --rat_alpha_warmup_steps 5000 --align_on pos
  --gamma_init 0.1 --gamma_trainable 0

  # warm start
  --init_ckpt ./model/PoMRec/PoMRec__beauty__42__lr=0.002__l2=1e-06.pt --init_strict 0

  # ✅ 新代码里下线：不要再传 use_logic_denoise / use_logic_aggr
)

wait_for_slot () {
  while [ "$running" -ge "$MAX_PROCS" ]; do
    # 等任意一个后台任务结束
    if wait -n; then
      :
    else
      echo "[WARN] A job exited non-zero, continue..."
    fi
    running=$((running-1))
  done
}

launch_job () {
  local name="$1"
  shift
  local out="${LOGDIR}/nohup_${name}_s${SEED}.out"

  wait_for_slot
  echo "Launch ${name} -> ${out}"

  nohup python main.py \
    "${COMMON[@]}" \
    "$@" \
    > "${out}" 2>&1 &

  running=$((running+1))
  sleep 0.2
}

# ========== A: LLMemb only ==========
launch_job "A_llm" \
  --use_emile 0 \
  --use_logic_reason 0 \
  --log_file  "${LOGDIR}/A_llm_s${SEED}.txt" \
  --model_path "${MODELDIR}/A_llm_${DATASET}_s${SEED}.pt"

# ========== B: + EMILE-IPD ==========
launch_job "B_ipd005" \
  --use_emile 1 --lambda_ipd 0.05 --lambda_ilr 0 --ipd_margin 0.2 --emile_use_fused_itememb 0 \
  --use_logic_reason 0 \
  --log_file  "${LOGDIR}/B_ipd005_s${SEED}.txt" \
  --model_path "${MODELDIR}/B_ipd005_${DATASET}_s${SEED}.pt"

# ========== C1: + IPD + HER (soft reasoning: no topk) ==========
launch_job "C1_ipd005_her_soft_lam010" \
  --use_emile 1 --lambda_ipd 0.05 --lambda_ilr 0 --ipd_margin 0.2 --emile_use_fused_itememb 0 \
  --use_logic_reason 1 \
  --lambda_reason 0.10 \
  --reason_temp 0.2 \
  --reason_topk 0 \
  --reason_use_fused 1 \
  --log_file  "${LOGDIR}/C1_ipd005_her_soft_lam010_s${SEED}.txt" \
  --model_path "${MODELDIR}/C1_ipd005_her_soft_lam010_${DATASET}_s${SEED}.pt"

# ========== C2: + IPD + HER (retrieval reasoning: topk=10) ==========
launch_job "C2_ipd005_her_top10_lam015" \
  --use_emile 1 --lambda_ipd 0.05 --lambda_ilr 0 --ipd_margin 0.2 --emile_use_fused_itememb 0 \
  --use_logic_reason 1 \
  --lambda_reason 0.15 \
  --reason_temp 0.1 \
  --reason_topk 10 \
  --reason_use_fused 1 \
  --log_file  "${LOGDIR}/C2_ipd005_her_top10_lam015_s${SEED}.txt" \
  --model_path "${MODELDIR}/C2_ipd005_her_top10_lam015_${DATASET}_s${SEED}.pt"

# 等全部结束
wait || true
echo "All 4 jobs finished on ${DATASET} (seed=${SEED}). Logs in ${LOGDIR}"
echo "A: LLM only | B: +IPD | C1: +HER soft | C2: +HER retrieval(topk=10)"