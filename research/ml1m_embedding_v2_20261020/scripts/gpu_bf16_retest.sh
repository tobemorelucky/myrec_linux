#!/usr/bin/env bash
# Prepared only. First confirm rental instance is kept/reopened and user approves.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[[ "$ROOT" == /root/autodl-tmp/pomrec/research/ml1m_embedding_v2_20261020 ]]
[[ "$V2_GPU_RETEST_APPROVED" == YES ]]
[[ "$V2_RENTAL_CONFIRMED" == YES ]]
[[ -n "$GPU_INDEX" ]]
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$ROOT/runtime_cache/hf" XDG_CACHE_HOME="$ROOT/runtime_cache/xdg"
PYTHON=/root/miniconda3/envs/py10/bin/python
TAG="$(date +%Y%m%d_%H%M%S)"
JOB="$ROOT/ops_runs/bf16_numeric_$TAG"
mkdir "$JOB"
OUT="$ROOT/runs/admission_c_bf16_numeric_$TAG"
# One config, no retries/fallback. 30min limit then 30s kill grace.
set +e
timeout --signal=INT --kill-after=30 1800 "$PYTHON" -B adapter/run_scft.py \
 --config configs/scft_rich_full1024_bf16_numeric.json --admission-steps 3 \
 --admission-output-dir "$OUT" --execute > "$JOB/stage.log" 2>&1
RC=$?
set -e
printf '{"exit_code":%d,"output_dir":"%s","log":"%s"}\n' "$RC" "$OUT" "$JOB/stage.log" > "$JOB/exit.json"
echo "$JOB/exit.json"
# Never run B/D/E, create a D approval pointer, restart or overwrite an output.
exit "$RC"
