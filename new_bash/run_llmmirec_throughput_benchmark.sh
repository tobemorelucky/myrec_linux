#!/usr/bin/env bash
# =========================================================
# LLMMIRec training throughput benchmark — ML-1M, id mode
#
# Usage (foreground only):
#   bash new_bash/run_llmmirec_throughput_benchmark.sh [GPU] [MODE] [DATASET]
#
# Defaults:
#   GPU     = 1          (physical GPU ID; 0 = first card, 1 = second)
#   MODE    = quick      (quick | full | targeted)
#   DATASET = ml-1m      (ml-1m | beauty | toys)
#
# Examples:
#   bash new_bash/run_llmmirec_throughput_benchmark.sh
#     → quick,   GPU 1, ml-1m
#
#   bash new_bash/run_llmmirec_throughput_benchmark.sh 0 full
#     → full,    GPU 0, ml-1m
#
#   bash new_bash/run_llmmirec_throughput_benchmark.sh 1 targeted beauty
#     → targeted, GPU 1, beauty
#
# CUDA mapping:
#   --gpu <N>  →  CUDA_VISIBLE_DEVICES=N  →  PyTorch sees cuda:0
#   Physical GPU N becomes logical GPU 0 inside the python process.
#
# NO background processes: no nohup, &, tmux, or screen.
# =========================================================

set -e

PHYSICAL_GPU=${1:-1}
MODE=${2:-quick}
DATASET=${3:-ml-1m}

# ---- validate mode ----
if [ "${MODE}" != "quick" ] && [ "${MODE}" != "full" ] && [ "${MODE}" != "targeted" ]; then
    echo "ERROR: MODE must be 'quick', 'full', or 'targeted', got '${MODE}'"
    exit 1
fi

# ---- validate dataset ----
if [ "${DATASET}" != "ml-1m" ] && [ "${DATASET}" != "beauty" ] && [ "${DATASET}" != "toys" ]; then
    echo "ERROR: DATASET must be 'ml-1m', 'beauty', or 'toys', got '${DATASET}'"
    exit 1
fi

echo "========================================================="
echo "LLMMIRec Throughput Benchmark"
echo "  Physical GPU  : ${PHYSICAL_GPU}"
echo "  Mode          : ${MODE}"
echo "  Dataset       : ${DATASET}"
echo "========================================================="

conda run -n hzg_py10 python tools/benchmark_llmmirec_throughput.py \
  --gpu "${PHYSICAL_GPU}" \
  --mode "${MODE}" \
  --dataset "${DATASET}"

echo ""
echo "Done."
