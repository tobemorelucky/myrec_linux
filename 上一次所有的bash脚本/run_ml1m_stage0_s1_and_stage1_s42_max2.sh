#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

########################################
# 并发上限：任何时刻最多 2 个训练进程
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
# 公共路径
########################################
DATASET="ml-1m"
BASE_LOGDIR="./log/MyModel/${DATASET}"
BASE_MODELDIR="./model/MyModel/${DATASET}"
mkdir -p "${BASE_LOGDIR}" "${BASE_MODELDIR}"

########################################
# 你 Stage0 目前认为最好的一组（按你 seed42 的 test-best）
# 如果你要换，就改这三行
########################################
S0_LR=0.0005
S0_K=3
S0_LAMB=3.5

########################################
# Stage1: 你目前 best LLM（一键先跑 seed=42）
########################################
GAMMA=0.07
TAU=0.2
ALPHA=0.0005
INIT_CKPT="./model/PoMRec/PoMRec__ml-1m__1__lr=0.001__l2=1e-06.pt"

########################################
# 启动函数
########################################
launch_job () {
  local name="$1"
  shift
  wait_for_slot
  echo "[`date '+%F %T'`] Launch ${name}"
  nohup python main.py "$@" > "${BASE_LOGDIR}/nohup_${name}.out" 2>&1 &
  running=$((running+1))
  sleep 0.2
}

########################################
# Job A: Stage0 seed=1
########################################
SEED0=1
TAG0="stage0_s${SEED0}_lr${S0_LR//./}_K${S0_K}_l${S0_LAMB//./}"
LOG0="${BASE_LOGDIR}/${TAG0}.txt"
PT0="${BASE_MODELDIR}/${TAG0}.pt"

launch_job "${TAG0}" \
  --model_name MyModel --dataset ${DATASET} \
  --lr "${S0_LR}" --l2 1e-06 \
  --batch_size 256 --eval_batch_size 256 \
  --epoch 200 --early_stop 10 \
  --num_neg 1 --dropout 0 --num_workers 5 \
  --random_seed ${SEED0} --load 0 \
  --history_max 20 \
  --K "${S0_K}" --prompt_num 4 --lamb "${S0_LAMB}" --emb_size 64 --attn_size 8 --n_layers 1 \
  --use_llmemb 0 --use_emile 0 \
  --use_logic_denoise 0 --use_logic_aggr 0 \
  --log_file "${LOG0}" \
  --model_path "${PT0}"

########################################
# Job B: Stage1 先跑 seed=42（你要求“第二个命令依旧先测试42的效果”）
# backbone 与 Stage0 保持一致，方便公平对比
########################################
SEED1=42
TAG1="stage1_llm_s${SEED1}_lr${S0_LR//./}_K${S0_K}_l${S0_LAMB//./}_g${GAMMA//./}_tau${TAU//./}_a${ALPHA//./}"
LOG1="${BASE_LOGDIR}/${TAG1}.txt"
PT1="${BASE_MODELDIR}/${TAG1}.pt"

launch_job "${TAG1}" \
  --model_name MyModel --dataset ${DATASET} \
  --lr "${S0_LR}" --l2 1e-06 \
  --batch_size 256 --eval_batch_size 256 \
  --epoch 200 --early_stop 10 \
  --num_neg 1 --dropout 0 --num_workers 5 \
  --random_seed ${SEED1} --load 0 \
  --history_max 20 \
  --K "${S0_K}" --prompt_num 4 --lamb "${S0_LAMB}" --emb_size 64 --attn_size 8 --n_layers 1 \
  --use_llmemb 1 --llm_fuse 1 \
  --llm_emb_path ./data/ml-1m/handled/llm_table_pca1536.pkl \
  --srs_emb_path ./data/ml-1m/handled/itm_emb_pomrec.pkl \
  --gamma_init "${GAMMA}" --gamma_trainable 0 \
  --alpha "${ALPHA}" --tau "${TAU}" --rat_alpha_warmup_steps 5000 \
  --init_ckpt "${INIT_CKPT}" --init_strict 0 \
  --use_emile 0 \
  --use_logic_denoise 0 --use_logic_aggr 0 \
  --log_file "${LOG1}" \
  --model_path "${PT1}"

########################################
# 等待所有任务结束
########################################
wait
echo "[`date '+%F %T'`] ALL DONE."
echo "Stage0 seed=1 out: ${BASE_LOGDIR}/nohup_${TAG0}.out"
echo "Stage1 seed=42 out: ${BASE_LOGDIR}/nohup_${TAG1}.out"