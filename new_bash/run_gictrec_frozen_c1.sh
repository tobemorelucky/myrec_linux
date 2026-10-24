#!/usr/bin/env bash
set -euo pipefail
test "$#" -eq 1
cd /home/yyx/hzgcode/pom2.0
source /home/yyx/anaconda3/etc/profile.d/conda.sh
conda activate hzg_py10
export PYTHONDONTWRITEBYTECODE=1
python -u -B tools/run_gictrec_frozen_c1.py --variant "$1"
