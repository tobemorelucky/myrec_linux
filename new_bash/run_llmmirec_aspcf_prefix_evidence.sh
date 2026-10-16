#!/usr/bin/env bash
# Prepared only: one explicitly authorized run per invocation, no queue.
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/yyx/anaconda3/etc/profile.d/conda.sh
conda activate hzg_py10
export PYTHONDONTWRITEBYTECODE=1
DATASET=$1
GPU=$2
VARIANT=$3
SEED=${4:-42}
case "$DATASET" in beauty) LR=.004 ;; ml-1m) LR=.001 ;; *) exit 2 ;; esac
case "$VARIANT" in A1|A2) ;; *) exit 2 ;; esac
[[ "$SEED" == 42 ]] || { echo "Round 1 is seed42 only"; exit 2; }
MODEL=LLMMIRecASPCFPrefixEvidence
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}
LOG_DIR="new_log/aspcf_prefix_evidence/$DATASET/${VARIANT}_k4_h20_seed42_$RUN_ID"
MODEL_DIR="new_model/aspcf_prefix_evidence/$DATASET/${VARIANT}_k4_h20_seed42_$RUN_ID"
python -B - "$DATASET" "$LOG_DIR" "$MODEL_DIR" <<'PY'
from pathlib import Path
import sys
root=Path.cwd().resolve();ds=sys.argv[1]
for asset in ['train.csv','dev.csv','SeqReader.pkl','handled/llm_table_pca1536.pkl']:
 p=(root/'data'/ds/asset).resolve()
 assert p.is_relative_to(root) and p.is_file(),str(p)
for value in sys.argv[2:]:
 p=Path(value).resolve();assert p.is_relative_to(root) and not p.exists(),str(p)
PY
mkdir -p "$LOG_DIR" "$MODEL_DIR"
CMD=(python -u -B main.py --model_name "$MODEL" --dataset "$DATASET" --path ./data/
 --gpu "$GPU" --random_seed "$SEED" --emb_size 64 --attn_size 64 --K 4 --history_max 20
 --item_encoder aspcf --llm_emb_path "./data/$DATASET/handled/llm_table_pca1536.pkl"
 --semantic_rank 512 --semantic_dim 32 --semantic_hidden 128 --complement_dim 32
 --tail_hidden 64 --complement_hidden 64 --gate_hidden 64 --aspcf_gate_mode basic
 --lambda_relation .01 --relation_sample_size 128 --relation_teacher_temp .1 --relation_student_temp .1
 --dropout .1 --lr "$LR" --l2 1e-6 --optimizer Adam --grad_clip 0
 --batch_size 1024 --eval_batch_size 256 --num_neg 1 --epoch 200 --early_stop 10
 --metric NDCG,HR --topk 5,10,20 --num_workers 5 --test_epoch -1 --regenerate 0 --test_all 0
 --prefix_variant "$VARIANT" --model_path "$MODEL_DIR/best.pt"
 --train 1 --load 0 --dev_only 1 --log_file "$LOG_DIR/train.log")
printf '%q ' "${CMD[@]}" > "$LOG_DIR/command.txt"
printf '\n' >> "$LOG_DIR/command.txt"
START=$SECONDS
"${CMD[@]}" 2>&1 | tee "$LOG_DIR/train.out"
python -B - "$LOG_DIR" "$MODEL_DIR/best.pt" "$((SECONDS-START))" <<'PY'
from pathlib import Path
import sys,re,json,hashlib
log=Path(sys.argv[1]);text=(log/'train.log').read_text()
assert ' END: ' in text and 'Test After Training:' not in text
best=re.findall(r'Best Iter\(dev\)=.*',text);saved=re.findall(r'Epoch\s+(\d+).* \*$',text,re.M)
assert best and saved
cp=Path(sys.argv[2])
with (log/'summary.json').open('x') as f:
 json.dump(dict(status='complete',best_dev_log=best[-1],checkpoint_epoch=int(saved[-1]),
  end_epoch=int(re.findall(r'Epoch\s+(\d+)',text)[-1]),training_seconds=int(sys.argv[3]),
  checkpoint=str(cp),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),
  test_evaluated=False,selection='original BaseRunner Dev NDCG@5; tie save unchanged'),f,indent=2)
PY
# No automatic Test in this first Dev-only development stage.
