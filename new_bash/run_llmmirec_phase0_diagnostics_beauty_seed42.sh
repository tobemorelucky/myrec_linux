#!/usr/bin/env bash
set -e

# =========================================================
# LLMMIRec Phase 0 — Beauty seed=42 interest diagnostics
#
# Analyzes trained checkpoints for all three encoder modes.
#
# Usage:
#   bash new_bash/run_llmmirec_phase0_diagnostics_beauty_seed42.sh
#
# Prerequisites: Phase 0.1 Beauty experiments must be completed.
# =========================================================

GPU=0
DATASET="beauty"

ROOT_MODEL_DIR="new_model/llmmirec_phase0/${DATASET}"
DIAG_DIR="new_log/llmmirec_phase0/${DATASET}/diagnostics"
SUMMARY_FILE="new_log/llmmirec_phase0/${DATASET}/summary_seed42.tsv"

mkdir -p "${DIAG_DIR}"

echo "========================================================="
echo "LLMMIRec Phase 0 — Beauty seed=42 interest diagnostics"
echo "DEST=${DIAG_DIR}"
echo "========================================================="

run_diag() {
  local MODE=$1
  local CKPT="${ROOT_MODEL_DIR}/${MODE}/LLMMIRec_${MODE}_seed42.pt"
  local OUT_DIR="${DIAG_DIR}/${MODE}"

  if [ ! -f "${CKPT}" ]; then
    echo "ERROR: Checkpoint not found: ${CKPT}"
    echo "  Run Phase 0.1 Beauty experiments first."
    exit 1
  fi

  mkdir -p "${OUT_DIR}"

  echo ""
  echo "=== Analyzing item_encoder=${MODE} ==="
  echo "checkpoint: ${CKPT}"
  echo "output:     ${OUT_DIR}"

  CUDA_VISIBLE_DEVICES=${GPU} python tools/analyze_llmmirec_interests.py \
    --checkpoint "${CKPT}" \
    --dataset "${DATASET}" \
    --max_batches 50 \
    --output_dir "${OUT_DIR}" \
    --device cuda

  echo "  -> ${OUT_DIR}/stats.json"
  echo "  -> ${OUT_DIR}/per_query.tsv"
}

# 1. id
run_diag "id"

# 2. llm_replace
run_diag "llm_replace"

# 3. residual
run_diag "residual"

echo ""
echo "========================================================="
echo "All diagnostics complete."
echo "Results: ${DIAG_DIR}/"
echo "========================================================="
