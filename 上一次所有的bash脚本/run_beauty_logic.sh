#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

MAX_PROCS=2
running=0

wait_for_slot () {
  while [ "$running" -ge "$MAX_PROCS" ]; do
    wait -n
    running=$((running-1))
  done
}

DATASET="beauty"
LOGDIR="./log/MyModel/${DATASET}"
MODELDIR="./model/MyModel"
mkdir -p "${LOGDIR}" "${MODELDIR}"

SEED=42

# ====== 固定：LLM + IPD 最佳（沿用你原 Beauty 脚本；删掉已无效参数 align_on / entropy_* / e2i_logic 等） ======
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
  --alpha 0.001 --tau 0.2 --rat_alpha_warmup_steps 5000
  --llm_fuse 1 --gamma_init 0.1 --gamma_trainable 0

  --init_ckpt ./model/PoMRec/PoMRec__beauty__42__lr=0.002__l2=1e-06.pt --init_strict 0

  --use_emile 1 --lambda_ipd 0.05 --lambda_ilr 0 --ipd_margin 0.10 --emile_use_fused_itememb 0
)

# ====== 只扫第三模块：7 组（4 aggr-only + 2 denoise-only + 1 combo） ======
# Beauty 上第三模块有效的经验：
# - aggr 可以比 ml-1m 稍微“放开一点”（lambda_max 0.10~0.20）
# - support_temp 不要太大（1.5~2.5）
# - denoise 的 r 可以略大（0.15~0.25），但仍不建议 0.30 起步
declare -a EXPS=(
  # ---- A: aggr-only ----
  "A1_aggr_safe     --use_logic_aggr 1 --lambda_logic_aggr 0.3 --logic_lambda_max 0.10 --logic_support_temp 2.0 --logic_gate_a 8  --logic_gate_b 0.75 --use_logic_denoise 0"
  "A2_aggr_strong   --use_logic_aggr 1 --lambda_logic_aggr 0.3 --logic_lambda_max 0.20 --logic_support_temp 1.8 --logic_gate_a 8  --logic_gate_b 0.70 --use_logic_denoise 0"
  "A3_aggr_soft     --use_logic_aggr 1 --lambda_logic_aggr 0.2 --logic_lambda_max 0.10 --logic_support_temp 2.5 --logic_gate_a 8  --logic_gate_b 0.80 --use_logic_denoise 0"
  "A4_aggr_mid      --use_logic_aggr 1 --lambda_logic_aggr 0.2 --logic_lambda_max 0.15 --logic_support_temp 2.0 --logic_gate_a 8  --logic_gate_b 0.75 --use_logic_denoise 0"

  # ---- D: denoise-only ----
  "D1_denoise_r015  --use_logic_denoise 1 --logic_denoise_alpha 1.0 --logic_denoise_r 0.15 --logic_denoise_topk 0 --logic_denoise_warmup_steps 5000  --use_logic_aggr 0"
  "D2_denoise_r020k5 --use_logic_denoise 1 --logic_denoise_alpha 1.2 --logic_denoise_r 0.20 --logic_denoise_topk 5 --logic_denoise_warmup_steps 5000  --use_logic_aggr 0"

  # ---- C: combo（只给 1 个最稳的组合验证叠加是否加分） ----
  "C1_combo         --use_logic_aggr 1 --lambda_logic_aggr 0.3 --logic_lambda_max 0.15 --logic_support_temp 2.0 --logic_gate_a 8 --logic_gate_b 0.75 \
                    --use_logic_denoise 1 --logic_denoise_alpha 1.0 --logic_denoise_r 0.15 --logic_denoise_topk 0 --logic_denoise_warmup_steps 5000"
)

for exp in "${EXPS[@]}"; do
  wait_for_slot

  name="${exp%% *}"
  args="${exp#* }"

  tag="beauty_${name}_s${SEED}"
  out="${LOGDIR}/nohup_${tag}.out"

  echo "Launch ${tag} -> ${out}"

  # shellcheck disable=SC2086
  nohup python main.py \
    "${COMMON[@]}" \
    ${args} \
    --log_file  "${LOGDIR}/${tag}.txt" \
    --model_path "${MODELDIR}/${tag}.pt" \
    > "${out}" 2>&1 &

  running=$((running+1))
  sleep 0.2
done

wait
echo "All Beauty logic sweep jobs finished. Logs in ${LOGDIR}"