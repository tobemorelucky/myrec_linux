#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
SIL_ROOT=/home/yyx/hzgcode/pom2.0
SIL_RUN_ROOT="$SIL_ROOT/research/elector_sil_dual_beauty_20261022_v1"
SIL_GPU=1
cd "$SIL_ROOT"
source /home/yyx/anaconda3/etc/profile.d/conda.sh
conda activate hzg_py10
export CUDA_VISIBLE_DEVICES="$SIL_GPU"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_CACHE_PATH="$SIL_RUN_ROOT/cuda_cache"
export TORCHINDUCTOR_CACHE_DIR="$SIL_RUN_ROOT/torch_cache"
SIL_PYTHON=/home/yyx/anaconda3/envs/hzg_py10/bin/python
exec 9>"$SIL_RUN_ROOT/gpu1_queue.lock"
flock -n 9 || exit 75
for SIL_ARM in B0 C1 M1; do
  # Never share the selected GPU with an unrelated compute process.
  while true; do
    SIL_OTHER_PIDS="$(nvidia-smi --id="$SIL_GPU" --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')"
    SIL_FREE="$(nvidia-smi --id="$SIL_GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
    if [[ -z "$SIL_OTHER_PIDS" && "$SIL_FREE" -ge 10000 ]]; then break; fi
    echo "$(date -Is) waiting for GPU $SIL_GPU before $SIL_ARM"
    sleep 30
  done
  echo "$(date -Is) starting $SIL_ARM on GPU $SIL_GPU"
  nohup "$SIL_PYTHON" -u -B "$SIL_RUN_ROOT/train_arm.py" --arm "$SIL_ARM" \
    >"$SIL_RUN_ROOT/arms/$SIL_ARM/stdout.log" 2>&1 </dev/null &
  SIL_CHILD=$!
  printf '%s\n' "$SIL_CHILD" >"$SIL_RUN_ROOT/arms/$SIL_ARM/train.pid"
  if wait "$SIL_CHILD"; then SIL_CODE=0; else SIL_CODE=$?; fi
  printf '%s\n' "$SIL_CODE" >"$SIL_RUN_ROOT/arms/$SIL_ARM/exit_code.txt"
  echo "$(date -Is) $SIL_ARM exited $SIL_CODE"
  if [[ "$SIL_CODE" -ne 0 ]]; then exit "$SIL_CODE"; fi
done
echo "$(date -Is) queue finished; no Test or cross-arm result summary"
