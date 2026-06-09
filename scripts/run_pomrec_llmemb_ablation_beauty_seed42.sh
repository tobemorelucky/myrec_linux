#!/usr/bin/env bash
set -euo pipefail

# =========================
# PoMRecLLMEmb Ablation Runner
# Dataset: beauty
# Seed: 42
# Run one experiment at a time
# =========================

cd "$(dirname "$0")/.."

# 如果你需要自动激活环境，取消下面三行注释并改成你的环境名
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate hzg_py10

DATASET="beauty"
GPU=0
SEED=42

LLM_EMB_PATH="./data/${DATASET}/handled/llm_table_pca1536.pkl"
SRS_EMB_PATH="./data/${DATASET}/handled/itm_emb_pomrec.pkl"

LOG_DIR="./logs/pomrec_llmemb_ablation_${DATASET}_seed${SEED}"
mkdir -p "${LOG_DIR}"

COMMON_ARGS="
  --model_name PoMRecLLMEmb
  --dataset ${DATASET}
  --path ./data/
  --gpu ${GPU}
  --random_seed ${SEED}
  --emb_size 64
  --attn_size 8
  --K 4
  --prompt_num 3
  --n_layers 2
  --lamb 4.0
  --history_max 20
  --lr 0.002
  --l2 1e-6
  --batch_size 256
  --eval_batch_size 256
  --epoch 200
  --early_stop 10
  --num_neg 1
  --dropout 0
  --num_workers 5
"

run_exp () {
  local name="$1"
  local extra_args="$2"
  local log_file="${LOG_DIR}/${name}.log"

  echo "============================================================"
  echo "[START] ${name}"
  echo "[LOG]   ${log_file}"
  echo "============================================================"

  python main.py ${COMMON_ARGS} ${extra_args} > "${log_file}" 2>&1

  echo "============================================================"
  echo "[DONE] ${name}"
  echo "[RESULT] ${name}"
  echo "============================================================"

  grep -E "Best Iter|Test After Training|Recall@|NDCG@|HR@" "${log_file}" | tail -40 || true
  echo ""
}

echo "Repository: $(pwd)"
echo "Dataset: ${DATASET}"
echo "Seed: ${SEED}"
echo "LLM embedding: ${LLM_EMB_PATH}"
echo "SRS anchor: ${SRS_EMB_PATH}"
echo ""

if [[ ! -f "${LLM_EMB_PATH}" ]]; then
  echo "[ERROR] LLM embedding not found: ${LLM_EMB_PATH}"
  exit 1
fi

# 0. none：验证复制版 PoMRec 是否正常
run_exp \
  "beauty_seed42_none_pomrec_copy" \
  "
  --llm_fuse_mode none
  "

# 1. replace：LLMEmb-style，adapter(e_llm) 替换 e_cf
run_exp \
  "beauty_seed42_llm_replace" \
  "
  --llm_fuse_mode replace
  --llm_emb_path ${LLM_EMB_PATH}
  --freeze_llm_emb 1
  --use_llm_align 0
  "

# 2. residual：你的语义残差融合消融
run_exp \
  "beauty_seed42_llm_residual_gamma01" \
  "
  --llm_fuse_mode residual
  --llm_emb_path ${LLM_EMB_PATH}
  --freeze_llm_emb 1
  --gamma_init 0.1
  --gamma_trainable 0
  --use_llm_align 0
  "

# 3. replace + align：完整 LLMEmb-style 适配
if [[ -f "${SRS_EMB_PATH}" ]]; then
  run_exp \
    "beauty_seed42_llm_replace_align001" \
    "
    --llm_fuse_mode replace
    --llm_emb_path ${LLM_EMB_PATH}
    --freeze_llm_emb 1
    --srs_emb_path ${SRS_EMB_PATH}
    --use_llm_align 1
    --align_weight 0.001
    --align_tau 0.2
    "
else
  echo "============================================================"
  echo "[SKIP] replace + align"
  echo "Reason: SRS anchor embedding not found:"
  echo "        ${SRS_EMB_PATH}"
  echo ""
  echo "如果你要跑 align，请先准备："
  echo "        ./data/${DATASET}/handled/itm_emb_pomrec.pkl"
  echo "============================================================"
fi

echo ""
echo "All selected experiments finished."
echo "Logs saved to: ${LOG_DIR}"
echo ""

echo "Quick summary:"
for f in "${LOG_DIR}"/*.log; do
  echo "------------------------------------------------------------"
  echo "$(basename "$f")"
  grep -E "Best Iter|Test After Training|Recall@5|Recall@10|Recall@20|NDCG@5|NDCG@10|NDCG@20|HR@5|HR@10|HR@20" "$f" | tail -30 || true
done