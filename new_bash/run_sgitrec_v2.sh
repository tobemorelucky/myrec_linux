#!/usr/bin/env bash
set -euo pipefail
cd /home/yyx/hzgcode/pom2.0
source /home/yyx/anaconda3/etc/profile.d/conda.sh
conda activate hzg_py10
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec python -u -B tools/run_sgitrec_v2.py "$@"
