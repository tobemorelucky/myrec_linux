#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$ROOT/runtime_cache/hf" XDG_CACHE_HOME="$ROOT/runtime_cache/xdg"
export TRITON_CACHE_DIR="$ROOT/runtime_cache/triton" TORCHINDUCTOR_CACHE_DIR="$ROOT/runtime_cache/torch"
PYTHON="${PYTHON:-/root/miniconda3/envs/py10/bin/python}"
case "${1:-check}" in
 check) "$PYTHON" -B adapter/preflight.py --config configs/encode_v1_rebuilt.json --gpu ;;
 encode)
   : "${GPU_INDEX:?Set explicitly approved rental GPU index}"
   export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
   "$PYTHON" -B adapter/preflight.py --config configs/encode_v1_rebuilt.json --gpu --strict
   "$PYTHON" -B adapter/run_encode.py --config configs/encode_v1_rebuilt.json --execute ;;
 pca)
   "$PYTHON" -B adapter/quality_pca.py --base-dir runs/v1_rebuilt_raw \
     --mapping /root/autodl-tmp/pomrec/data/ml-1m/item2id.json \
     --train /root/autodl-tmp/pomrec/data/ml-1m/train.csv --out runs/v1_rebuilt_pca1536 --execute ;;
 *) echo "Use check, encode or pca (production stages require new user authorization)";exit 2 ;;
esac
