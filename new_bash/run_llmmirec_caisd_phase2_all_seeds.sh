#!/usr/bin/env bash

set -u

SEEDS=(0 1 2 41 42)

BEAUTY_GPU=0
ML1M_GPU=1

ROOT_QUEUE_LOG="new_log/llmmirec_phase2_queue"
mkdir -p "${ROOT_QUEUE_LOG}/beauty"
mkdir -p "${ROOT_QUEUE_LOG}/ml-1m"


run_beauty() {
    echo "============================================================"
    echo "[Beauty worker] GPU=${BEAUTY_GPU}"
    echo "[Beauty worker] seeds=${SEEDS[*]}"
    echo "============================================================"

    for SEED in "${SEEDS[@]}"; do
        echo
        echo "============================================================"
        echo "[Beauty] seed=${SEED} ASPCF START $(date '+%F %T')"
        echo "============================================================"

        mkdir -p "new_log/llmmirec_aspcf_phase2/beauty/seed${SEED}"

        if bash new_bash/run_llmmirec_aspcf_phase2_beauty.sh \
            "${BEAUTY_GPU}" "${SEED}" \
            > "new_log/llmmirec_aspcf_phase2/beauty/seed${SEED}/nohup.out" 2>&1
        then
            echo "[Beauty] seed=${SEED} ASPCF DONE $(date '+%F %T')"
        else
            echo "[ERROR] Beauty seed=${SEED} ASPCF failed"
            return 1
        fi


        echo
        echo "============================================================"
        echo "[Beauty] seed=${SEED} CAISD START $(date '+%F %T')"
        echo "============================================================"

        mkdir -p "new_log/llmmirec_caisd_phase2/beauty/seed${SEED}"

        if bash new_bash/run_llmmirec_caisd_phase2_beauty.sh \
            "${BEAUTY_GPU}" "${SEED}" \
            > "new_log/llmmirec_caisd_phase2/beauty/seed${SEED}/nohup.out" 2>&1
        then
            echo "[Beauty] seed=${SEED} CAISD DONE $(date '+%F %T')"
        else
            echo "[ERROR] Beauty seed=${SEED} CAISD failed"
            return 1
        fi
    done

    echo
    echo "============================================================"
    echo "[Beauty worker] ALL DONE $(date '+%F %T')"
    echo "============================================================"
}


run_ml1m() {
    echo "============================================================"
    echo "[ML-1M worker] GPU=${ML1M_GPU}"
    echo "[ML-1M worker] seeds=${SEEDS[*]}"
    echo "============================================================"

    for SEED in "${SEEDS[@]}"; do
        echo
        echo "============================================================"
        echo "[ML-1M] seed=${SEED} ASPCF START $(date '+%F %T')"
        echo "============================================================"

        mkdir -p "new_log/llmmirec_aspcf_phase2/ml-1m/seed${SEED}"

        if bash new_bash/run_llmmirec_aspcf_phase2_ml1m.sh \
            "${ML1M_GPU}" "${SEED}" \
            > "new_log/llmmirec_aspcf_phase2/ml-1m/seed${SEED}/nohup.out" 2>&1
        then
            echo "[ML-1M] seed=${SEED} ASPCF DONE $(date '+%F %T')"
        else
            echo "[ERROR] ML-1M seed=${SEED} ASPCF failed"
            return 1
        fi


        echo
        echo "============================================================"
        echo "[ML-1M] seed=${SEED} CAISD START $(date '+%F %T')"
        echo "============================================================"

        mkdir -p "new_log/llmmirec_caisd_phase2/ml-1m/seed${SEED}"

        if bash new_bash/run_llmmirec_caisd_phase2_ml1m.sh \
            "${ML1M_GPU}" "${SEED}" \
            > "new_log/llmmirec_caisd_phase2/ml-1m/seed${SEED}/nohup.out" 2>&1
        then
            echo "[ML-1M] seed=${SEED} CAISD DONE $(date '+%F %T')"
        else
            echo "[ERROR] ML-1M seed=${SEED} CAISD failed"
            return 1
        fi
    done

    echo
    echo "============================================================"
    echo "[ML-1M worker] ALL DONE $(date '+%F %T')"
    echo "============================================================"
}


run_beauty \
    > "${ROOT_QUEUE_LOG}/beauty/queue.log" 2>&1 &
BEAUTY_PID=$!

run_ml1m \
    > "${ROOT_QUEUE_LOG}/ml-1m/queue.log" 2>&1 &
ML1M_PID=$!

echo "Beauty worker PID=${BEAUTY_PID}"
echo "ML-1M worker PID=${ML1M_PID}"

wait "${BEAUTY_PID}"
BEAUTY_STATUS=$?

wait "${ML1M_PID}"
ML1M_STATUS=$?

echo "============================================================"
echo "Phase2 queue finished: $(date '+%F %T')"
echo "Beauty status=${BEAUTY_STATUS}"
echo "ML-1M status=${ML1M_STATUS}"
echo "============================================================"

if [ "${BEAUTY_STATUS}" -ne 0 ] || [ "${ML1M_STATUS}" -ne 0 ]; then
    exit 1
fi
