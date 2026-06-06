#!/usr/bin/env bash
set -euo pipefail

# 保持原有的并发进程数配置
MAX_PROCS=2

# 数据集和文件路径与原脚本一致
DATASET="ml-1m"
LLM_PKL="./data/ml-1m/handled/llm_table_pca1536.pkl"
SRS_PKL="./data/ml-1m/handled/itm_emb_pomrec.pkl"
LOGDIR="./log/ml-1m"
mkdir -p "${LOGDIR}"

# 最佳超参数（对应S2_tau_g0.05_a1e-3_tau0.5_s42.log的参数）
BEST_G=0.05
BEST_A=1e-3
BEST_T=0.5

# 初始化检查点路径不变
INIT_CKPT="./model/PoMRec/PoMRec__ml-1m__1__lr=0.001__l2=1e-06.pt"

running=0

# 保持原有的并发等待函数不变
wait_for_slot () {
  while [ "$running" -ge "$MAX_PROCS" ]; do
    wait -n
    running=$((running-1))
  done
}

# 保持原有的任务启动函数不变
launch_job () {
  local name="$1"
  shift
  echo "Launch: ${name}"
  nohup python main.py "$@" > "${LOGDIR}/${name}.log" 2>&1 &
  running=$((running+1))
  sleep 0.2
}

# ========== 核心修改：遍历指定的随机种子 ==========
# 你指定的种子列表：0、1、2、3、43、42
SEEDS=(0 1 2 3 43 42)

# 遍历每个种子，使用最佳超参数运行
for SEED in "${SEEDS[@]}"; do
  wait_for_slot
  # 日志命名包含最佳超参数标识和种子，便于区分
  name="seed${SEED}_g${BEST_G}_a${BEST_A}_tau${BEST_T}"
  launch_job "${name}" \
    --model_name PoMRec --dataset "${DATASET}" \
    --use_llmemb 1 --llm_fuse 1 \
    --llm_emb_path "${LLM_PKL}" --srs_emb_path "${SRS_PKL}" \
    --gamma_init "${BEST_G}" --gamma_trainable 0 \
    --alpha "${BEST_A}" --tau "${BEST_T}" --rat_alpha_warmup_steps 5000 \
    --align_on pos --random_seed "${SEED}" \
    --init_ckpt "${INIT_CKPT}" --init_strict 0
done

# 等待最后一批任务完成
wait
echo "All seed runs for best parameters finished. Logs in ${LOGDIR}"