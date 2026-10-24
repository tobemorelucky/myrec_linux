#!/usr/bin/env bash
set -euo pipefail
# CONFIG FULL_CROSSFIT_GRAPH B0|G1|G2|G3 NEW_OUTPUT_DIR [GPU]
test "$#" -ge 4
cd /home/yyx/hzgcode/pom2.0
source /home/yyx/anaconda3/etc/profile.d/conda.sh
conda activate hzg_py10
export PYTHONDONTWRITEBYTECODE=1
python -u -B tools/run_sgitrec.py --config "$1" --graph "$2" --variant "$3" --output-dir "$4" --gpu "${5:-0}"
