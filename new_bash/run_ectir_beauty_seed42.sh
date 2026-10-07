#!/usr/bin/env bash
# =========================================================
# THESIS PHASE 1 — Chapter 4 Round 1 launch (Beauty, seed 42)
#
# Three ECTIR configurations, queued on two GPUs:
#   GPU0 : ectir1 (transport only)  ->  ectir2 (transport + refinement)
#   GPU1 : full    (transport + refinement + evidence aggregation)
#
# ASPCF baseline is NOT re-run (already available: 0.1592 / 0.1088 seed 42,
# 5-seed mean 0.1582 / 0.1075).
#
# No ML-1M, no multi-seed, no Toys, no hyper-parameter sweep.
#
# Usage: bash new_bash/run_ectir_beauty_seed42.sh [GPU0] [GPU1] [SEED]
# =========================================================
set -euo pipefail

GPU0=${1:-0}
GPU1=${2:-1}
SEED=${3:-42}
DATASET=beauty

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_DIR="new_log/ectir_beauty_seed${SEED}"
mkdir -p "${OUT_DIR}"

DRIVER="${ROOT_DIR}/new_bash/run_llmmirec_ectir_phase1_beauty.sh"

if ! python -c "import torch" 2>/dev/null; then
  echo "ERROR: current python has no torch. Run: conda activate hzg_py10"; exit 1
fi

echo "=== THESIS PHASE 1: Chapter 4 Round 1 — ECTIR (${DATASET}, seed=${SEED}) ==="
echo "    start: $(date '+%Y-%m-%d %H:%M:%S')"

nohup bash -c "cd '${ROOT_DIR}' && CONFIG=ectir1 bash '${DRIVER}' ${GPU0} ${SEED} ${DATASET} && CONFIG=ectir2 bash '${DRIVER}' ${GPU0} ${SEED} ${DATASET}" \
  > "${OUT_DIR}/gpu${GPU0}.nohup.out" 2>&1 &
PID0=$!

nohup bash -c "cd '${ROOT_DIR}' && CONFIG=full bash '${DRIVER}' ${GPU1} ${SEED} ${DATASET}" \
  > "${OUT_DIR}/gpu${GPU1}.nohup.out" 2>&1 &
PID1=$!

echo "  GPU${GPU0} (pid=${PID0}): ectir1 -> ectir2"
echo "  GPU${GPU1} (pid=${PID1}): full"
echo
echo "Results when done:"
echo "  cat new_log/llmmirec_ectir/${DATASET}/summary.tsv"
