#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

MAX_PROCS=2
running=0

wait_for_slot () {
  while [ "$running" -ge "$MAX_PROCS" ]; do
    wait -n
    running=$((running-1))
  done
}

DATASET="ml-1m"
LOGDIR="./log/MyModel/${DATASET}"
MODELDIR="./model/MyModel"
mkdir -p "${LOGDIR}" "${MODELDIR}"

SEED=42

# ====== 固定：LLM + IPD 最佳（按你原脚本；删掉已无效参数 align_on / entropy_* / gate_floor / logic_beta 等） ======
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

  --init_ckpt ./model/PoMRec/PoMRec__ml-1m__1__lr=0.001__l2=1e-06.pt --init_strict 0

  --use_emile 1 --lambda_ipd 0.05 --lambda_ilr 0 --ipd_margin 0.10 --emile_use_fused_itememb 0
)

# ====== 只扫第三模块：6 组（4 aggr-only + 2 denoise-only） ======
# 设计思路：
# - aggr-only：通过 logic_lambda_max + support_temp 把“过度介入”压住
# - denoise-only：r 从 0.10 起步（ml-1m 通常 0.30 会明显伤）
#
# 你跑完 6 组后：
# - 如果 aggr-only 有提升，再把最好的 aggr 与 D1 组合跑 1~2 组即可
# - 如果 denoise-only 有提升，再考虑加 topk 或略增 alpha

declare -a EXPS=(
  # ---- A: aggr-only（最优先，ml-1m 主要问题在 aggr 过强） ----
  "A1_aggr_strict   --use_logic_aggr 1 --lambda_logic_aggr 0.3 --logic_lambda_max 0.05 --logic_support_temp 3.0 --logic_gate_a 10 --logic_gate_b 0.75 --use_logic_denoise 0"
  "A2_aggr_medium   --use_logic_aggr 1 --lambda_logic_aggr 0.3 --logic_lambda_max 0.10 --logic_support_temp 2.0 --logic_gate_a 10 --logic_gate_b 0.80 --use_logic_denoise 0"
  "A3_aggr_ultra    --use_logic_aggr 1 --lambda_logic_aggr 0.2 --logic_lambda_max 0.05 --logic_support_temp 3.0 --logic_gate_a 10 --logic_gate_b 0.75 --use_logic_denoise 0"
  "A4_aggr_soft     --use_logic_aggr 1 --lambda_logic_aggr 0.2 --logic_lambda_max 0.10 --logic_support_temp 2.5 --logic_gate_a 10 --logic_gate_b 0.80 --use_logic_denoise 0"

  # ---- D: denoise-only（保守扫 2 组） ----
  "D1_denoise_r010  --use_logic_denoise 1 --logic_denoise_alpha 0.8 --logic_denoise_r 0.10 --logic_denoise_topk 0 --logic_denoise_warmup_steps 20000 --use_logic_aggr 0"
  "D2_denoise_r015k5 --use_logic_denoise 1 --logic_denoise_alpha 0.8 --logic_denoise_r 0.15 --logic_denoise_topk 5 --logic_denoise_warmup_steps 20000 --use_logic_aggr 0"
)

for exp in "${EXPS[@]}"; do
  wait_for_slot

  name="${exp%% *}"
  args="${exp#* }"

  tag="ml1m_${name}_s${SEED}"
  out="${LOGDIR}/nohup_${tag}.out"

  echo "Launch ${tag} -> ${out}"

  # shellcheck disable=SC2086
  nohup python main.py \
    "${BASE[@]}" \
    ${args} \
    --log_file  "${LOGDIR}/${tag}.txt" \
    --model_path "${MODELDIR}/${tag}.pt" \
    > "${out}" 2>&1 &

  running=$((running+1))
  sleep 0.2
done

wait
echo "All ML-1M logic sweep jobs finished. Logs in ${LOGDIR}"