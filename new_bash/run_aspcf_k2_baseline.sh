#!/usr/bin/env bash
# =========================================================
# Fair-tuning baseline: ASPCF with K=2 (dataset-specific interest count).
#
# PoMRec's standard config tunes K per dataset (beauty K=4, ml-1m K=2).
# ASPCF is given the same courtesy. This is ordinary hyper-parameter
# selection for a fair baseline, NOT a method contribution.
#
# Purpose:
#   1. quantify how much of ML-1M's ~2% gap vs PoMRec is merely K
#   2. establish a fair dataset-specific ASPCF baseline for Chapter 4
#
# Only K=2 is run; K=4 (5 seeds) already exists and is not re-run.
# No further K values are scanned.
#
# Usage:
#   bash new_bash/run_aspcf_k2_baseline.sh beauty        # default: beauty only
#   bash new_bash/run_aspcf_k2_baseline.sh ml-1m
#   RUN_BOTH=1 bash new_bash/run_aspcf_k2_baseline.sh
# =========================================================
set -euo pipefail

GPU=${1:-0}
SEED=${2:-42}
RUN_BOTH=${RUN_BOTH:-0}
ONLY=${3:-beauty}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_DIR="new_log/ch4_phase0_k2_baseline"
mkdir -p "${OUT_DIR}"

if ! python -c "import torch" 2>/dev/null; then
  echo "ERROR: current python has no torch. Run: conda activate hzg_py10"; exit 1
fi

echo "=== ASPCF K=2 fair-tuning baseline (seed=${SEED}, gpu=${GPU}) ==="
echo "    start: $(date '+%Y-%m-%d %H:%M:%S')"

if [ "${RUN_BOTH}" = "1" ]; then
  nohup bash -c "cd '${ROOT_DIR}' && K=2 bash new_bash/run_llmmirec_aspcf_K.sh ${GPU} beauty ${SEED} && K=2 bash new_bash/run_llmmirec_aspcf_K.sh ${GPU} ml-1m ${SEED}" \
    > "${OUT_DIR}/k2_both.nohup.out" 2>&1 &
  echo "  pid=$!  beauty -> ml-1m (sequential)"
else
  nohup bash -c "cd '${ROOT_DIR}' && K=2 bash new_bash/run_llmmirec_aspcf_K.sh ${GPU} ${ONLY} ${SEED}" \
    > "${OUT_DIR}/k2_${ONLY}.nohup.out" 2>&1 &
  echo "  pid=$!  ${ONLY} only"
fi

echo
echo "Results when done:"
echo "  new_log/llmmirec_aspcf_K2/<dataset>/seed${SEED}/LLMMIRecASPCF_seed${SEED}.log"
echo "  new_log/llmmirec_aspcf_K2/<dataset>/summary.tsv"
