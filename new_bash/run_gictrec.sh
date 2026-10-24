#!/usr/bin/env bash
set -euo pipefail
# Explicit-only entry: CONFIG C1_CHECKPOINT FULL_TRAIN_BANK G0|G1|G2|G3 NEW_OUTPUT_DIR [GPU]
test "$#" -ge 5
ROOT=/home/yyx/hzgcode/pom2.0
cd "$ROOT"
source /home/yyx/anaconda3/etc/profile.d/conda.sh
conda activate hzg_py10
export PYTHONDONTWRITEBYTECODE=1
python -B tools/run_gictrec.py --config "$1" --init-c1 "$2" --bank "$3" --variant "$4" --output-dir "$5" --gpu "${6:-0}"
