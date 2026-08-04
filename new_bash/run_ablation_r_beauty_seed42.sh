#!/usr/bin/env bash
set -e

GPU=${1:-0}
SEED=42

ROOT_LOG_DIR="new_log/ablation_r/beauty"
ROOT_MODEL_DIR="new_model/ablation_r/beauty"
mkdir -p "${ROOT_LOG_DIR}" "${ROOT_MODEL_DIR}"

LLM_PATH="./data/beauty/handled/llm_table_pca1536.pkl"
SRS_PATH="./data/beauty/handled/itm_emb_pomrec.pkl"
INIT_CKPT="./model/PoMRec/PoMRec__beauty__${SEED}__lr=0.002__l2=1e-06.pt"
INIT_FALLBACK="./model/PoMRec/PoMRec__beauty__42__lr=0.002__l2=1e-06.pt"
if [ ! -f "${INIT_CKPT}" ]; then
  echo "[WARN] init_ckpt not found for seed=${SEED}, fallback to ${INIT_FALLBACK}"
  INIT_CKPT="${INIT_FALLBACK}"
fi

SUMMARY_FILE="${ROOT_LOG_DIR}/summary_seed${SEED}.tsv"
echo -e "dataset\tmodel\tvariant\tseed\tr\tstart_time\tend_time\ttotal_seconds\tbest_dev\ttest_after_training" > "${SUMMARY_FILE}"

# Beauty best: lr=0.002, gamma=0.1, tau=0.2, lamb=3.0
# IPD: lambda=0.05, margin=0.2, warmup=5000
# LGD: alpha=8.0, b=0.3, topk=5, warmup=20000
R_VALUES=(0 0.05 0.10 0.15 0.20)

run_one() {
  local R=$1
  local VARIANT="beauty_r$(echo $R | sed 's/\.//g')_seed${SEED}"
  local LOG_FILE="${ROOT_LOG_DIR}/${VARIANT}.log"
  local OUT_FILE="${ROOT_LOG_DIR}/${VARIANT}.out"
  local MODEL_PATH="${ROOT_MODEL_DIR}/${VARIANT}.pt"

  echo ""
  echo "========================================================="
  echo "[START] ${VARIANT}  r=${R}"
  echo "GPU=${GPU}  seed=${SEED}"
  echo "log=${LOG_FILE}"
  echo "========================================================="

  local START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
  local START_TS=$(date +%s)

  python main.py \
    --model_name MyModel --dataset beauty --path ./data/ --gpu "${GPU}" \
    --random_seed "${SEED}" --load 0 \
    --emb_size 64 --attn_size 8 --K 3 --prompt_num 4 --n_layers 1 --lamb 3.0 --history_max 20 \
    --use_llmemb 1 --llm_fuse 1 --llm_emb_path "${LLM_PATH}" --srs_emb_path "${SRS_PATH}" \
    --gamma_init 0.1 --gamma_trainable 0 --alpha 0.001 --tau 0.2 --rat_alpha_warmup_steps 5000 \
    --init_ckpt "${INIT_CKPT}" --init_strict 0 \
    --use_emile 1 --lambda_ipd 0.05 --ipd_margin 0.2 --emile_use_fused_itememb 0 --emile_warmup_steps 5000 \
    --use_logic_denoise 1 --logic_denoise_alpha 8.0 --logic_denoise_b 0.3 --logic_denoise_topk 5 \
    --logic_denoise_r "${R}" --logic_denoise_warmup_steps 20000 \
    --use_logic_aggr 0 --lambda_logic_aggr 0.0 \
    --lr 0.002 --l2 1e-6 --batch_size 256 --eval_batch_size 256 --num_neg 1 \
    --dropout 0 --epoch 200 --early_stop 10 --num_workers 5 \
    --log_file "${LOG_FILE}" --model_path "${MODEL_PATH}" \
    > "${OUT_FILE}" 2>&1

  local END_TIME=$(date "+%Y-%m-%d %H:%M:%S")
  local END_TS=$(date +%s)
  local TOTAL_SECONDS=$((END_TS - START_TS))
  local BEST_DEV=$(grep "Best Iter(dev)" "${LOG_FILE}" | tail -n 1 | tr '\t' ' ' | sed 's/[[:space:]]\+/ /g' || true)
  local TEST_AFTER=$(grep "Test After Training" "${LOG_FILE}" | tail -n 1 | tr '\t' ' ' | sed 's/[[:space:]]\+/ /g' || true)

  echo ""
  echo "[DONE] ${VARIANT}"
  echo "${BEST_DEV}"
  echo "${TEST_AFTER}"

  echo -e "beauty\tMyModel\t${VARIANT}\t${SEED}\t${R}\t${START_TIME}\t${END_TIME}\t${TOTAL_SECONDS}\t${BEST_DEV}\t${TEST_AFTER}" >> "${SUMMARY_FILE}"
}

for R in "${R_VALUES[@]}"; do
  run_one "$R"
done

echo ""
echo "========================================================="
echo "Beauty r-ablation finished. Summary:"
echo "cat ${SUMMARY_FILE}"
echo "========================================================="
