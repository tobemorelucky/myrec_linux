#!/usr/bin/env bash
# =========================================================
# THESIS PHASE 1 — Chapter 4 Phase 0 diagnostic (NO TRAINING)
#
# Loads existing checkpoints (PoMRec / LLMMIRec-ID / ASPCF / HSDIR) and
# analyses the test split on six axes:
#   1. HR/NDCG by target popularity bucket
#   2. HR/NDCG by history-length bucket
#   3. interest pairwise cosine / effective rank
#   4. attention entropy
#   5. interest weight entropy / active interests
#   6. positive-vs-negative score margin
#
# Beauty and ML-1M run on separate GPUs. Pure inference; a few minutes each.
#
# Usage: bash new_bash/run_ch4_phase0_diagnosis.sh [GPU_BEAUTY] [GPU_ML1M]
# =========================================================
set -euo pipefail

GPU_BEAUTY=${1:-0}
GPU_ML1M=${2:-1}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_DIR="new_log/ch4_phase0_diagnosis"
mkdir -p "${OUT_DIR}" diagnostics_ch4_phase0

if ! python -c "import torch" 2>/dev/null; then
  echo "ERROR: current python has no torch. Run: conda activate hzg_py10"; exit 1
fi
for ds in beauty ml-1m; do
  [ -f "data/${ds}/SeqReader.pkl" ] || { echo "ERROR: missing SeqReader for ${ds}"; exit 1; }
done

echo "=== Chapter 4 Phase 0 diagnostic (no training) ==="
echo "    start: $(date '+%Y-%m-%d %H:%M:%S')"

nohup python tools/diagnose_ch4_phase0.py --dataset beauty --device "cuda:${GPU_BEAUTY}" \
  > "${OUT_DIR}/beauty.nohup.out" 2>&1 &
PID_B=$!

nohup python tools/diagnose_ch4_phase0.py --dataset ml-1m --device "cuda:${GPU_ML1M}" \
  > "${OUT_DIR}/ml-1m.nohup.out" 2>&1 &
PID_M=$!

echo "  beauty (pid=${PID_B}, GPU${GPU_BEAUTY}) -> ${OUT_DIR}/beauty.nohup.out"
echo "  ml-1m  (pid=${PID_M}, GPU${GPU_ML1M}) -> ${OUT_DIR}/ml-1m.nohup.out"
echo
echo "Results when done:"
echo "  diagnostics_ch4_phase0/{beauty,ml-1m}.json"
echo "  grep -v 'Prepare test' ${OUT_DIR}/beauty.nohup.out"
