#!/usr/bin/env bash
# =========================================================
# THESIS PHASE 1 — Chapter 3 Round 2 launch (Beauty, seed 42)
#
# Four structural configurations, two per GPU, run in parallel:
#
#   GPU0 : rasrf_scalar        then  rasrf_vector
#   GPU1 : rasrf_vector_neigh  then  residual_gamma   (global-gamma CONTROL)
#
# The residual_gamma control is what isolates whether ITEM-SPECIFIC gating is
# what helps, versus a single global learnable scale.
#
# LOSS = plain BPR for all four. No auxiliary losses in this round.
# All training hyperparameters are copied from the stable config in
# new_bash/run_llmmirec_aspcf_phase2_beauty.sh (lr=0.004, batch=1024, ...).
#
# Usage:
#   bash new_bash/run_rasrf_beauty_seed42.sh [GPU0] [GPU1] [SEED]
#
# Logs: new_log/llmmirec_rasrf/beauty/<config>/seed42/  (per run)
#       new_log/rasrf_beauty_seed42/gpu{0,1}.nohup.out (driver output)
# =========================================================
set -euo pipefail

GPU0=${1:-0}
GPU1=${2:-1}
SEED=${3:-42}
DATASET=beauty

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DRIVER_DIR="new_log/rasrf_beauty_seed42"
mkdir -p "${DRIVER_DIR}"

if ! python -c "import torch" 2>/dev/null; then
  echo "ERROR: current python has no torch. Run: conda activate hzg_py10"; exit 1
fi
for f in "data/${DATASET}/handled/llm_table_pca1536.pkl" \
         "data/${DATASET}/handled/semantic_neighborhood_agreement_k20.pkl"; do
  [ -f "$f" ] || { echo "ERROR: missing ${f}"; exit 1; }
done

DRIVER="${ROOT_DIR}/new_bash/run_llmmirec_rasrf_phase1_beauty.sh"

run_chain () {   # gpu config1 config2
  local gpu=$1 c1=$2 c2=$3
  CONFIG="${c1}" bash "${DRIVER}" "${gpu}" "${SEED}" "${DATASET}"
  CONFIG="${c2}" bash "${DRIVER}" "${gpu}" "${SEED}" "${DATASET}"
}

echo "=== THESIS PHASE 1 Round 2: RASRF structural test (${DATASET}, seed=${SEED}) ==="
echo "    start: $(date '+%Y-%m-%d %H:%M:%S')"

nohup bash -c "cd '${ROOT_DIR}' && CONFIG=rasrf_scalar bash '${DRIVER}' ${GPU0} ${SEED} ${DATASET} && CONFIG=rasrf_vector bash '${DRIVER}' ${GPU0} ${SEED} ${DATASET}" \
  > "${DRIVER_DIR}/gpu${GPU0}.nohup.out" 2>&1 &
PID0=$!

nohup bash -c "cd '${ROOT_DIR}' && CONFIG=rasrf_vector_neigh bash '${DRIVER}' ${GPU1} ${SEED} ${DATASET} && CONFIG=residual_gamma bash '${DRIVER}' ${GPU1} ${SEED} ${DATASET}" \
  > "${DRIVER_DIR}/gpu${GPU1}.nohup.out" 2>&1 &
PID1=$!

echo "  GPU${GPU0} (pid=${PID0}): rasrf_scalar -> rasrf_vector"
echo "  GPU${GPU1} (pid=${PID1}): rasrf_vector_neigh -> residual_gamma"
echo
echo "Expected wall clock ~35-45 min total (2 x ~18 min per GPU)."
echo "Check later with:"
echo "  cat new_log/llmmirec_rasrf/${DATASET}/summary.tsv"
