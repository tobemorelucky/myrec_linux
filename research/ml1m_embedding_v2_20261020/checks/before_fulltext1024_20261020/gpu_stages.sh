#!/usr/bin/env bash
# Commands only. No GPU execution was authorized when written.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$ROOT/runtime_cache/hf" XDG_CACHE_HOME="$ROOT/runtime_cache/xdg"
export TRITON_CACHE_DIR="$ROOT/runtime_cache/triton" TORCHINDUCTOR_CACHE_DIR="$ROOT/runtime_cache/torch"
PYTHON="${PYTHON:-/root/miniconda3/envs/py10/bin/python}"
stage="${1:-check}"
if [[ "$stage" != "check" && "$stage" != "F" ]];then
  : "${GPU_INDEX:?Use only explicitly approved rental GPU}"
  export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
fi
case "$stage" in
 check) "$PYTHON" -B adapter/preflight.py --config configs/admission_a.json --gpu ;;
 A)
  "$PYTHON" -B adapter/run_encode.py --config configs/admission_a.json \
    --sample-ids 1,2,81,1084,1085,2652,2653,3706 --execute ;;
 B)
  test -f runs/admission_a_old_ml_lora/SAMPLE_COMPLETE.json
  "$PYTHON" -B adapter/stage_gate.py --stage B
  "$PYTHON" -B adapter/run_encode.py --config configs/encode_v1_rebuilt.json --execute ;;
 C)
  test -f configs/scft_rich_frozen.json
  "$PYTHON" -B adapter/run_scft.py --config configs/scft_rich_frozen.json --admission-steps 3 --execute ;;
 D)
  : "${V2_SCFT_APPROVED:?Stage D requires separate approval after C report}"
  test "$V2_SCFT_APPROVED" = YES
  : "${V2_WALL_SECONDS_APPROVED:?Confirm paid runtime budget after stage C}"
  [[ "$V2_WALL_SECONDS_APPROVED" =~ ^[0-9]+$ ]]
  timeout --signal=INT --kill-after=60 "$V2_WALL_SECONDS_APPROVED" "$PYTHON" -B adapter/run_scft.py --config configs/scft_rich_frozen.json --execute ;;
 E)
  test -f runs/scft_rich_seed42/SCFT_COMPLETE.json
  test -f runs/scft_rich_seed42/checkpoint-1000/adapter_model.bin
  "$PYTHON" -B adapter/stage_gate.py --stage E
  "$PYTHON" -B adapter/run_encode.py --config configs/encode_base_proposed.json --execute
  "$PYTHON" -B adapter/run_encode.py --config configs/encode_rich_frozen.json --execute
  "$PYTHON" -B adapter/quality_pca.py --base-dir runs/encode_base --rich-dir runs/encode_rich \
    --mapping /root/autodl-tmp/pomrec/data/ml-1m/item2id.json \
    --train /root/autodl-tmp/pomrec/data/ml-1m/train.csv --out runs/v2_common_pca1536 --execute
  "$PYTHON" -B adapter/quality_pca.py --base-dir runs/v1_rebuilt_raw \
    --mapping /root/autodl-tmp/pomrec/data/ml-1m/item2id.json \
    --train /root/autodl-tmp/pomrec/data/ml-1m/train.csv --out runs/v1_rebuilt_pca1536 --execute ;;
 F) "$PYTHON" -B adapter/stage_gate.py --stage F ;;
 *) echo "Use check or individually approved A/B/C/D/E/F; no automatic chain";exit 2 ;;
esac
