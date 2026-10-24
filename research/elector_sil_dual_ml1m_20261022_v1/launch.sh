#!/usr/bin/env bash
set -euo pipefail
cd /home/yyx/hzgcode/pom2.0
source /home/yyx/anaconda3/etc/profile.d/conda.sh
conda activate hzg_py10
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec /home/yyx/anaconda3/envs/hzg_py10/bin/python -u -B /home/yyx/hzgcode/pom2.0/research/elector_sil_dual_ml1m_20261022_v1/scheduler.py
