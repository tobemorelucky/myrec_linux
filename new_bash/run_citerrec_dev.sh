#!/usr/bin/env bash
set -euo pipefail
cd /home/yyx/hzgcode/pom2.0
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec /home/yyx/anaconda3/envs/hzg_py10/bin/python -u -B tools/run_citerrec_dev.py "$@"
