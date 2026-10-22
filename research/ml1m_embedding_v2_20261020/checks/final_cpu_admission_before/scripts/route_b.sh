#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$ROOT/runtime_cache/hf" XDG_CACHE_HOME="$ROOT/runtime_cache/xdg"
export TRITON_CACHE_DIR="$ROOT/runtime_cache/triton" TORCHINDUCTOR_CACHE_DIR="$ROOT/runtime_cache/torch"
PYTHON="${PYTHON:-/root/miniconda3/envs/py10/bin/python}"
case "${1:-check}" in
 check) "$PYTHON" -B adapter/preflight.py --config configs/scft_proposed.json --gpu --framework ;;
 train)
   : "${GPU_INDEX:?Set explicitly approved rental GPU index}";export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
   "$PYTHON" -B adapter/preflight.py --config configs/scft_proposed.json --gpu --framework --strict
   "$PYTHON" -B adapter/run_scft.py --config configs/scft_proposed.json --execute ;;
 encode-base|encode-rich)
   : "${GPU_INDEX:?Set explicitly approved rental GPU index}";export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
   if [[ "$1" == "encode-base" ]];then CONFIG=configs/encode_base_proposed.json;else CONFIG=configs/encode_rich_proposed.json;fi
   "$PYTHON" -B adapter/preflight.py --config "$CONFIG" --gpu --strict
   "$PYTHON" -B adapter/run_encode.py --config "$CONFIG" --execute ;;
 pca)
   "$PYTHON" -B adapter/quality_pca.py --base-dir runs/encode_base --rich-dir runs/encode_rich \
     --mapping /root/autodl-tmp/pomrec/data/ml-1m/item2id.json \
     --train /root/autodl-tmp/pomrec/data/ml-1m/train.csv --out runs/v2_common_pca1536 --execute ;;
 *) echo "Use check, train, encode-base, encode-rich or pca; no production without user approval";exit 2 ;;
esac
