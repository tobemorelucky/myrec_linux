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
profile="${V2_PROFILE:-full1024}"
if [[ "$profile" = full1024 ]]; then
  AC=configs/admission_a_full1024.json; BC=configs/encode_v1_rebuilt_full1024.json
  SC=configs/scft_rich_full1024_frozen.json
  EC=configs/encode_base_full1024.json; RC=configs/encode_rich_full1024.json
  ER=runs/encode_base_full1024; RR=runs/encode_rich_full1024
  AR=runs/admission_a_old_ml_lora_full1024; SR=runs/scft_rich_full1024_seed42
  BR=runs/v1_rebuilt_raw_full1024; PR=runs/v2_common_pca1536_full1024; VPR=runs/v1_rebuilt_pca1536_full1024
elif [[ "$profile" = legacy256 ]]; then
  AC=configs/admission_a.json; BC=configs/encode_v1_rebuilt.json; SC=configs/scft_rich_frozen.json
  EC=configs/encode_base_proposed.json; RC=configs/encode_rich_frozen.json
  ER=runs/encode_base; RR=runs/encode_rich
  AR=runs/admission_a_old_ml_lora; SR=runs/scft_rich_seed42; BR=runs/v1_rebuilt_raw
  PR=runs/v2_common_pca1536; VPR=runs/v1_rebuilt_pca1536
else echo "Unknown profile"; exit 2; fi
if [[ "$stage" != "check" && "$stage" != "F" ]];then
  : "${GPU_INDEX:?Use only explicitly approved rental GPU}"
  export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
fi
case "$stage" in
 check) "$PYTHON" -B adapter/preflight.py --config "$AC" --gpu ;;
 A)
  "$PYTHON" -B adapter/run_encode.py --config "$AC" \
    --sample-ids 1,2,81,1084,1085,2652,2653,3706 --execute ;;
 B)
  test -f "$AR/SAMPLE_COMPLETE.json"
  "$PYTHON" -B adapter/stage_gate.py --stage B --profile "$profile"
  "$PYTHON" -B adapter/run_encode.py --config "$BC" --execute ;;
 C)
  test -f "$SC"
  "$PYTHON" -B adapter/run_scft.py --config "$SC" --admission-steps 3 --execute ;;
 D)
  : "${V2_SCFT_APPROVED:?Stage D requires separate approval after C report}"
  test "$V2_SCFT_APPROVED" = YES
  : "${V2_WALL_SECONDS_APPROVED:?Confirm paid runtime budget after stage C}"
  [[ "$V2_WALL_SECONDS_APPROVED" =~ ^[0-9]+$ ]]
  timeout --signal=INT --kill-after=60 "$V2_WALL_SECONDS_APPROVED" "$PYTHON" -B adapter/run_scft.py --config "$SC" --execute ;;
 E)
  test -f "$SR/SCFT_COMPLETE.json"
  test -f "$SR/checkpoint-1000/adapter_model.bin"
  "$PYTHON" -B adapter/stage_gate.py --stage E --profile "$profile"
  "$PYTHON" -B adapter/run_encode.py --config "$EC" --execute
  "$PYTHON" -B adapter/run_encode.py --config "$RC" --execute
  "$PYTHON" -B adapter/quality_pca.py --base-dir "$ER" --rich-dir "$RR" \
    --mapping /root/autodl-tmp/pomrec/data/ml-1m/item2id.json \
    --train /root/autodl-tmp/pomrec/data/ml-1m/train.csv --out "$PR" --execute
  "$PYTHON" -B adapter/quality_pca.py --base-dir "$BR" \
    --mapping /root/autodl-tmp/pomrec/data/ml-1m/item2id.json \
    --train /root/autodl-tmp/pomrec/data/ml-1m/train.csv --out "$VPR" --execute ;;
 F) "$PYTHON" -B adapter/stage_gate.py --stage F --profile "$profile" ;;
 *) echo "Use check or individually approved A/B/C/D/E/F; no automatic chain";exit 2 ;;
esac
