#!/usr/bin/env bash
set -euo pipefail

GPU=1
SEEDS=(0 1 2 41 42)

echo "============================================================"
echo "[ML-1M Phase2] START $(date '+%F %T')"
echo "GPU=${GPU}"
echo "Seeds=${SEEDS[*]}"
echo "============================================================"

for SEED in "${SEEDS[@]}"; do

    echo
    echo "============================================================"
    echo "[ML-1M][seed=${SEED}] ASPCF START $(date '+%F %T')"
    echo "============================================================"

    mkdir -p "new_log/llmmirec_aspcf_phase2/ml-1m/seed${SEED}"

    bash new_bash/run_llmmirec_aspcf_phase2_ml1m.sh \
        "${GPU}" "${SEED}" \
        > "new_log/llmmirec_aspcf_phase2/ml-1m/seed${SEED}/queue.out" 2>&1

    echo "[ML-1M][seed=${SEED}] ASPCF DONE $(date '+%F %T')"


    echo
    echo "============================================================"
    echo "[ML-1M][seed=${SEED}] CAISD START $(date '+%F %T')"
    echo "============================================================"

    mkdir -p "new_log/llmmirec_caisd_phase2/ml-1m/seed${SEED}"

    bash new_bash/run_llmmirec_caisd_phase2_ml1m.sh \
        "${GPU}" "${SEED}" \
        > "new_log/llmmirec_caisd_phase2/ml-1m/seed${SEED}/queue.out" 2>&1

    echo "[ML-1M][seed=${SEED}] CAISD DONE $(date '+%F %T')"

done

echo
echo "============================================================"
echo "[ML-1M Phase2] ALL DONE $(date '+%F %T')"
echo "============================================================"
