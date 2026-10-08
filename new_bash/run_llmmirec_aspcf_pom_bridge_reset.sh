#!/usr/bin/env bash
# Preregistered Chapter 4 backbone reset. Default is a read-only command preview.
set -euo pipefail
if (( $# < 4 || $# > 5 )); then
  echo "Usage: bash $0 beauty|ml-1m clean|pom_centrality|pom_full GPU RUN_ID [--execute]" >&2
  exit 2
fi
DATASET=$1
MODE=$2
GPU=$3
RUN_ID=$4
ACTION=${5:---preview}
[[ "$DATASET" == beauty || "$DATASET" == ml-1m ]] || exit 2
[[ "$MODE" == clean || "$MODE" == pom_centrality || "$MODE" == pom_full ]] || exit 2
[[ "$GPU" =~ ^[0-9]+$ && "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
[[ "$ACTION" == --preview || "$ACTION" == --execute ]] || exit 2
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
[[ "$ROOT" == /home/yyx/hzgcode/pom2.0 ]] || { echo "Unexpected project root" >&2; exit 2; }
cd "$ROOT"
LR=0.004
[[ "$DATASET" != ml-1m ]] || LR=0.001
DISP=0
[[ "$MODE" != pom_full ]] || DISP=1
LOG_DIR="$ROOT/new_log/ch4_backbone_reset/$DATASET/${MODE}_seed42_$RUN_ID"
MODEL_DIR="$ROOT/new_model/ch4_backbone_reset/$DATASET/${MODE}_seed42_$RUN_ID"
for dir in "$LOG_DIR" "$MODEL_DIR"; do
  resolved=$(realpath -m -- "$dir")
  [[ "$resolved" == "$ROOT/"* ]] || { echo "Output escapes project" >&2; exit 2; }
  [[ ! -e "$dir" && ! -L "$dir" ]] || { echo "Refusing existing output: $dir" >&2; exit 2; }
done
CMD=(python -B main.py --model_name LLMMIRecASPCFPoMBridge
  --dataset "$DATASET" --path ./data/ --gpu "$GPU" --random_seed 42
  --bridge_mode "$MODE" --bridge_attn_size 8 --bridge_prompt_num 3
  --bridge_n_layers 2 --bridge_lambda_disp "$DISP"
  --emb_size 64 --attn_size 64 --K 4 --history_max 20
  --item_encoder aspcf --llm_emb_path "./data/$DATASET/handled/llm_table_pca1536.pkl"
  --semantic_rank 512 --semantic_dim 32 --semantic_hidden 128
  --complement_dim 32 --tail_hidden 64 --complement_hidden 64 --gate_hidden 64
  --aspcf_gate_mode basic --lambda_relation 0.01 --relation_sample_size 128
  --relation_teacher_temp 0.1 --relation_student_temp 0.1
  --dropout 0.1 --lr "$LR" --l2 1e-6 --optimizer Adam
  --batch_size 1024 --eval_batch_size 256 --num_neg 1 --epoch 200
  --early_stop 10 --topk 5,10,20,50 --metric NDCG,HR --test_epoch -1
  --num_workers 5 --regenerate 0 --load 0 --train 1
  --log_file "$LOG_DIR/train.log" --model_path "$MODEL_DIR/best.pt")
printf 'PREREGISTERED_COMMAND '
printf '%q ' "${CMD[@]}"
printf '\n'
[[ "$ACTION" == --execute ]] || exit 0
# No cache generation or replacement when the frozen asset is missing.
[[ -f "data/$DATASET/SeqReader.pkl" && -f "data/$DATASET/handled/llm_table_pca1536.pkl" ]] ||
  { echo "Frozen corpus/LLM asset missing; not regenerating" >&2; exit 2; }
source /home/yyx/anaconda3/etc/profile.d/conda.sh
conda activate hzg_py10
export PYTHONDONTWRITEBYTECODE=1
mkdir -p -- "$(dirname -- "$LOG_DIR")" "$(dirname -- "$MODEL_DIR")"
# Atomic reservation; existing runs are never overwritten.
mkdir -- "$LOG_DIR"
mkdir -- "$MODEL_DIR"
printf '%q ' "${CMD[@]}" > "$LOG_DIR/command.txt"
printf '\n' >> "$LOG_DIR/command.txt"
date -Is > "$LOG_DIR/started.txt"
"${CMD[@]}" 2>&1 | tee "$LOG_DIR/console.out"
date -Is > "$LOG_DIR/finished.txt"
python -B - "$LOG_DIR" "$DATASET" "$MODE" "$LR" "$DISP" <<'PY'
import json, re, sys
from pathlib import Path
out = Path(sys.argv[1])
text = (out / 'train.log').read_text()
best = [line for line in text.splitlines() if 'Best Iter(dev)' in line][-1]
test = [line for line in text.splitlines() if 'Test After Training' in line][-1]
metrics = {f'{m}@{k}': float(v) for m,k,v in re.findall(r'(HR|NDCG)@(5|10|20|50):([0-9.eE+-]+)', test)}
required = [f'{m}@{k}' for k in (5,10,20) for m in ('HR','NDCG')]
if any(key not in metrics for key in required):
    raise RuntimeError('Missing one of six required ranking metrics')
counts = re.findall(r'#params:\s*(\d+)', text)
summary = dict(dataset=sys.argv[2], mode=sys.argv[3], seed=42,
               lr=float(sys.argv[4]), lambda_disp=float(sys.argv[5]),
               bridge_attn_size=8, prompt_num=3, n_layers=2, K=4,
               parameter_count=int(counts[-1]), best_dev=best,
               test_after_training=test, ranking_metrics=metrics)
with (out / 'summary.json').open('x') as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
PY
