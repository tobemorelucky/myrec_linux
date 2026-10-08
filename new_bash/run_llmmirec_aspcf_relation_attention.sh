#!/usr/bin/env bash
# Chapter 4 relation-attention prototype. Preview by default; no seed/eta sweep.
set -euo pipefail
if (( $# < 3 || $# > 4 )); then
  echo "Usage: bash $0 beauty|ml-1m GPU RUN_ID [--execute]" >&2
  exit 2
fi
DATASET=$1
GPU=$2
RUN_ID=$3
ACTION=${4:---preview}
[[ "$DATASET" == beauty || "$DATASET" == ml-1m ]] || exit 2
[[ "$GPU" =~ ^[0-9]+$ && "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
[[ "$ACTION" == --preview || "$ACTION" == --execute ]] || exit 2
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
[[ "$ROOT" == /home/yyx/hzgcode/pom2.0 ]] || exit 2
cd "$ROOT"
LR=0.004
[[ "$DATASET" != ml-1m ]] || LR=0.001
LOG_DIR="$ROOT/new_log/ch4_relation_attention/$DATASET/seed42_$RUN_ID"
MODEL_DIR="$ROOT/new_model/ch4_relation_attention/$DATASET/seed42_$RUN_ID"
for dir in "$LOG_DIR" "$MODEL_DIR"; do
  resolved=$(realpath -m -- "$dir")
  [[ "$resolved" == "$ROOT/"* ]] || { echo "Output escapes project" >&2; exit 2; }
  [[ ! -e "$dir" && ! -L "$dir" ]] || { echo "Existing output: $dir" >&2; exit 2; }
done
CMD=(python -B main.py --model_name LLMMIRecASPCFRelationAttention
  --dataset "$DATASET" --path ./data/ --gpu "$GPU" --random_seed 42
  --relation_eta_init 0 --relation_eta_max 0.1
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
[[ -f "data/$DATASET/SeqReader.pkl" && -f "data/$DATASET/handled/llm_table_pca1536.pkl" ]] ||
  { echo "Frozen assets missing; refusing regeneration" >&2; exit 2; }
source /home/yyx/anaconda3/etc/profile.d/conda.sh
conda activate hzg_py10
export PYTHONDONTWRITEBYTECODE=1
mkdir -p -- "$(dirname -- "$LOG_DIR")" "$(dirname -- "$MODEL_DIR")"
mkdir -- "$LOG_DIR"
mkdir -- "$MODEL_DIR"
printf '%q ' "${CMD[@]}" > "$LOG_DIR/command.txt"
printf '\n' >> "$LOG_DIR/command.txt"
date -Is > "$LOG_DIR/started.txt"
"${CMD[@]}" 2>&1 | tee "$LOG_DIR/console.out"
python -B - "$LOG_DIR" "$MODEL_DIR/best.pt" "$DATASET" "$LR" <<'PY'
import json, math, re, sys
from pathlib import Path
import torch
out = Path(sys.argv[1])
text = (out / 'train.log').read_text()
if ' END: ' not in text:
    raise RuntimeError('Missing normal END')
best = [x for x in text.splitlines() if 'Best Iter(dev)' in x][-1]
test = [x for x in text.splitlines() if 'Test After Training' in x][-1]
def metrics(line):
    return {f'{m}@{k}':float(v) for m,k,v in re.findall(r'(HR|NDCG)@(5|10|20|50):([0-9.eE+-]+)',line)}
dev, ranking = metrics(best), metrics(test)
required = [f'{m}@{k}' for k in (5,10,20) for m in ('HR','NDCG')]
if not all(k in ranking and math.isfinite(ranking[k]) for k in required):
    raise RuntimeError('Incomplete or nonfinite test metrics')
if not all(k in dev and math.isfinite(dev[k]) for k in ('HR@5','NDCG@5')):
    raise RuntimeError('Incomplete or nonfinite best dev')
state = torch.load(sys.argv[2],map_location='cpu',weights_only=True)
eta_raw = float(state['extractor.eta_raw'])
summary = dict(dataset=sys.argv[3],model='LLMMIRecASPCFRelationAttention',seed=42,
               lr=float(sys.argv[4]),eta_init=0,eta_max=.1,eta_raw_best=eta_raw,
               eta_best=.1*math.tanh(eta_raw),K=4,attn_size=64,history_max=20,
               parameter_count=int(re.findall(r'#params:\s*(\d+)',text)[-1]),
               best_epoch=int(re.search(r'Best Iter\(dev\)=\s*(\d+)',best).group(1)),
               best_dev=best,best_dev_metrics=dev,test_after_training=test,
               ranking_metrics=ranking)
with (out/'summary.json').open('x') as f:
    json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2))
PY
date -Is > "$LOG_DIR/finished.txt"
