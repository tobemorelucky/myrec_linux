#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/yyx/anaconda3/etc/profile.d/conda.sh
conda activate hzg_py10
export PYTHONDONTWRITEBYTECODE=1
DATASET=${1:?beauty or ml-1m}
GPU=${2:?GPU}
SEED=${3:-42}
case "$DATASET" in
 beauty) LR=0.004 ;;
 ml-1m) LR=0.001 ;;
 *) echo "Unsupported dataset" >&2; exit 2 ;;
esac
ASSET="./data/${DATASET}/handled/llmmi_proto32_sr512.pkl"
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}
LOG_DIR="new_log/aspcf_cicl/${DATASET}/seed${SEED}_${RUN_ID}"
MODEL_DIR="new_model/aspcf_cicl/${DATASET}/seed${SEED}_${RUN_ID}"
python -B - "$DATASET" "$ASSET" "$LOG_DIR" "$MODEL_DIR" <<'PY'
from pathlib import Path
import sys
root=Path.cwd().resolve()
for value in [sys.argv[2],f"data/{sys.argv[1]}/handled/llm_table_pca1536.pkl",f"data/{sys.argv[1]}/SeqReader.pkl"]:
 p=Path(value).resolve()
 if not p.is_relative_to(root) or not p.is_file():raise SystemExit("Missing/unsafe existing asset: "+str(p))
for value in sys.argv[3:]:
 p=Path(value).resolve()
 if not p.is_relative_to(root) or p.exists():raise SystemExit("Unsafe/existing output: "+str(p))
PY
mkdir -p "$LOG_DIR" "$MODEL_DIR"
date -u +%FT%TZ > "$LOG_DIR/start_time.txt"
START=$SECONDS
python -B main.py --model_name LLMMIRecASPCFCICL --dataset "$DATASET" --path ./data/ \
 --gpu "$GPU" --random_seed "$SEED" --emb_size 64 --attn_size 64 --K 4 --history_max 20 \
 --item_encoder aspcf --llm_emb_path "./data/${DATASET}/handled/llm_table_pca1536.pkl" \
 --semantic_rank 512 --semantic_dim 32 --semantic_hidden 128 --complement_dim 32 \
 --tail_hidden 64 --complement_hidden 64 --gate_hidden 64 --aspcf_gate_mode basic \
 --lambda_relation 0.01 --relation_sample_size 128 --relation_teacher_temp 0.1 \
 --relation_student_temp 0.1 --dropout 0.1 --lr "$LR" --l2 1e-6 --optimizer Adam \
 --batch_size 1024 --eval_batch_size 256 --num_neg 1 --epoch 200 --early_stop 10 \
 --metric NDCG,HR --topk 5,10,20,50 --num_workers 5 --test_epoch -1 --dev_only 1 --regenerate 0 \
 --cicl_proto_path "$ASSET" --lambda_contribution 0.01 \
 --log_file "$LOG_DIR/train.log" --model_path "$MODEL_DIR/best.pt" 2>&1 | tee "$LOG_DIR/train.out"
ELAPSED=$((SECONDS-START))
python -B - "$LOG_DIR" "$ELAPSED" "$MODEL_DIR/best.pt" <<'PY'
from pathlib import Path
import json,re,sys
root=Path(sys.argv[1])
text=(root/"train.log").read_text()
best=re.findall(r"Best Iter\(dev\)=.*",text)
if not best or " END: " not in text or "Test After Training:" in text:
 raise SystemExit("Missing normal completion/best-dev or unexpected test")
summary=dict(best_dev=best[-1],elapsed_seconds=int(sys.argv[2]),checkpoint=sys.argv[3],
 protocol="dev-only, NDCG@5 best, no test",status="complete")
with (root/"summary.json").open("x") as f:json.dump(summary,f,indent=2)
print(json.dumps(summary))
PY
