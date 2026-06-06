#!/usr/bin/env bash
set -euo pipefail

# ========== 基本配置 ==========
export CUDA_VISIBLE_DEVICES=0
MAX_PROCS=2

# 你要改的：新数据集名/路径（与你 main.py --dataset 的一致）
DATASET="toys"               # 例如 "toys" 或 "/home/xx/data/toys"
LOGDIR="./log/${DATASET}"
mkdir -p "${LOGDIR}"

# 你要改的：两类embedding与初始化ckpt
LLM_PKL="./data/${DATASET}/handled/llm_table_pca1536.pkl"
SRS_PKL="./data/${DATASET}/handled/itm_emb_pomrec.pkl"
INIT_CKPT="./model/PoMRec/toys__1__lr=0.001__l2=1e-06__lamb=3.8__history_max=20.pt"

SEED=1

# ========== 你已确定的固定超参 ==========
K=3
PROMPT_NUM=4
LAMB=3.8
LR=0.001

# 其余结构参数：按你项目常用设置给默认（需要的话你再改）
EMB_SIZE=64
ATTN_SIZE=8
N_LAYERS=2
HISTORY_MAX=20

L2=1e-06
BATCH_SIZE=256
EVAL_BATCH_SIZE=256
EPOCH=200
EARLY_STOP=10
NUM_NEG=1
DROPOUT=0
NUM_WORKERS=2

# RAT相关（你之前一直在用）
WARMUP=5000
ALIGN_ON="pos"

running=0
wait_for_slot () {
  while [ "$running" -ge "$MAX_PROCS" ]; do
    wait -n
    running=$((running-1))
  done
}

launch_job () {
  local name="$1"
  shift
  local log_file="${LOGDIR}/${name}.log"

  echo "Launch: ${name} -> ${log_file}"
  nohup python main.py "$@" > "${log_file}" 2>&1 &
  running=$((running+1))
  sleep 0.2
}

# ========== 固定“基础点” ==========
# 你可以把这里当做中心点：先围绕它微调
BASE_G=0.05
BASE_A=1e-3
BASE_T=0.5

# ========== 8组（不多）实验配置 ==========
# 组1：alpha（3个）
# 组2：tau（3个）
# 组3：gamma trainable（2个，固定 g=0.05）
configs=(
  "S1_alpha_a5e-4    ${BASE_G} 0 ${BASE_T} 5e-4"
  "S1_alpha_a1e-3    ${BASE_G} 0 ${BASE_T} 1e-3"
  "S1_alpha_a2e-3    ${BASE_G} 0 ${BASE_T} 2e-3"

  "S2_tau_t0.3       ${BASE_G} 0 0.3     ${BASE_A}"
  "S2_tau_t0.5       ${BASE_G} 0 0.5     ${BASE_A}"
  "S2_tau_t0.7       ${BASE_G} 0 0.7     ${BASE_A}"

  "S3_gTrain_t0.5    ${BASE_G} 1 ${BASE_T} ${BASE_A}"
  "S3_gTrain_t0.3    ${BASE_G} 1 0.3     ${BASE_A}"
)

# ========== 并发池执行 ==========
for cfg in "${configs[@]}"; do
  wait_for_slot
  # cfg fields:
  # name gamma_init gamma_trainable tau alpha
  read -r NAME G_INIT G_TR T AU <<< "${cfg}"

  launch_job "${NAME}_g${G_INIT}_gt${G_TR}_tau${T}_a${AU}_s${SEED}" \
    --model_name PoMRec --dataset "${DATASET}" \
    --K "${K}" --attn_size "${ATTN_SIZE}" --emb_size "${EMB_SIZE}" \
    --prompt_num "${PROMPT_NUM}" --n_layers "${N_LAYERS}" --lamb "${LAMB}" --history_max "${HISTORY_MAX}" \
    --lr "${LR}" --l2 "${L2}" --batch_size "${BATCH_SIZE}" --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --epoch "${EPOCH}" --early_stop "${EARLY_STOP}" --num_neg "${NUM_NEG}" --dropout "${DROPOUT}" --num_workers "${NUM_WORKERS}" \
    --random_seed "${SEED}" \
    --use_llmemb 1 --llm_fuse 1 \
    --llm_emb_path "${LLM_PKL}" --srs_emb_path "${SRS_PKL}" \
    --gamma_init "${G_INIT}" --gamma_trainable "${G_TR}" \
    --alpha "${AU}" --tau "${T}" --rat_alpha_warmup_steps "${WARMUP}" --align_on "${ALIGN_ON}" \
    --init_ckpt "${INIT_CKPT}" --init_strict 0
done

wait
echo "All runs finished. Logs in ${LOGDIR}"
