#!/usr/bin/env bash
# =========================================================
# THESIS PHASE 1 Round 1 — ML-1M architecture-only attribution
#
# Complements the Beauty results (THESIS_CH3_ROUND1.md §3.2) so the
# "new encoder vs old relation loss" attribution is complete on ML-1M too.
#
# Runs exactly two configurations, one per GPU, both with lambda_relation=0
# (PURE_BPR) so that ONLY the item-representation architecture differs from
# the ASPCF PURE_BPR control (already measured: beauty 0.1534/0.1034,
# ml-1m 0.3078/0.2123).
#
#   1. CGSCD-crosscov  PURE_BPR
#   2. CGSCD-cca       PURE_BPR
#
# All training hyperparameters are byte-identical to the stable configs in
# run_llmmirec_aspcf_phase2_{beauty,ml1m}.sh — verified against the
# `Arguments | Values` table in the actual training logs (see
# THESIS_CH3_ROUND1.md §8). Only lr differs by dataset, as it always has:
# beauty 0.004, ml-1m 0.001.
#
# Usage:
#   bash new_bash/run_cgscd_ml1m_archonly.sh [GPU_CROSSCOV] [GPU_CCA] [SEED]
#   e.g. bash new_bash/run_cgscd_ml1m_archonly.sh 0 1 42
#
# Logs: new_log/cgscd_ml1m_archonly/<label>_seed<seed>.nohup.out
# Each training run still writes its own log/model under
# new_log/llmmirec_cgscd/ml-1m/<label>/seed<seed>/ via run_llmmirec_cgscd_phase3.sh
# =========================================================
set -euo pipefail

GPU_CROSSCOV=${1:-0}
GPU_CCA=${2:-1}
SEED=${3:-42}
DATASET=ml-1m

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

NOHUP_DIR="new_log/cgscd_ml1m_archonly"
mkdir -p "${NOHUP_DIR}"

# --- fail fast on the two classic problems (missing env / missing assets) ---
if ! python -c "import torch" 2>/dev/null; then
  echo "ERROR: current python has no torch. Run: conda activate hzg_py10"
  exit 1
fi
for m in crosscov cca; do
  f="data/${DATASET}/handled/cgscd_basis_${m}_r32.pkl"
  [ -f "$f" ] || { echo "ERROR: missing ${f}"; exit 1; }
done
[ -f "data/${DATASET}/handled/itm_emb_pomrec.pkl" ] || { echo "ERROR: missing CF table"; exit 1; }
[ -f "data/${DATASET}/handled/llm_table_pca1536.pkl" ] || { echo "ERROR: missing LLM table"; exit 1; }

launch () {
  local method=$1 gpu=$2
  local label="${method}_rel0"
  local out="${NOHUP_DIR}/${label}_seed${SEED}.nohup.out"
  echo "[launch] ${DATASET} ${label} seed=${SEED} gpu=${gpu} -> ${out}"
  DATASET="${DATASET}" METHOD="${method}" USE_RELATION=0 \
    nohup bash new_bash/run_llmmirec_cgscd_phase3.sh "${gpu}" "${SEED}" \
    > "${out}" 2>&1 &
  echo "         pid=$!  gpu=${gpu}"
}

echo "=== THESIS PHASE 1 Round 1: ML-1M architecture-only (PURE_BPR) ==="
echo "    start: $(date '+%Y-%m-%d %H:%M:%S')"
launch crosscov "${GPU_CROSSCOV}"
launch cca      "${GPU_CCA}"
echo
echo "Both launched. Expected wall clock ~80-110 min each (ml-1m, epoch<=200, early_stop=10)."
echo "Check results later with:"
echo "  grep 'Test After Training' new_log/llmmirec_cgscd/ml-1m/{crosscov_rel0,cca_rel0}/seed${SEED}/LLMMIRecCGSCD_seed${SEED}.log"
