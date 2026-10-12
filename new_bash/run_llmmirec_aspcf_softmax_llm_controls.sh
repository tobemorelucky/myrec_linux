#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/yyx/anaconda3/etc/profile.d/conda.sh
conda activate hzg_py10
export PYTHONDONTWRITEBYTECODE=1
DATASET=${1:?dataset}
GPU=${2:?GPU}
SEED=${3:-42}
ENCODER=${4:?id or aspcf}
case "$ENCODER" in id|aspcf) ;; *) exit 2 ;; esac
CONDITION="softmax_k4_${ENCODER}"
MODEL=LLMMIRecASPCFSampledSoftmax
K=4
case "$DATASET" in beauty) LR=.004 ;; ml-1m) LR=.001 ;; *) exit 2 ;; esac
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}
LOG_DIR="new_log/aspcf_softmax_llm_controls/${DATASET}/${CONDITION}_seed${SEED}_${RUN_ID}"
MODEL_DIR="new_model/aspcf_softmax_llm_controls/${DATASET}/${CONDITION}_seed${SEED}_${RUN_ID}"
python -B - "$DATASET" "$LOG_DIR" "$MODEL_DIR" <<'PY'
from pathlib import Path
import sys
root=Path.cwd().resolve()
for x in [f'data/{sys.argv[1]}/SeqReader.pkl',f'data/{sys.argv[1]}/handled/llm_table_pca1536.pkl']:
 p=Path(x).resolve()
 if not p.is_relative_to(root) or not p.is_file():raise SystemExit('Missing/unsafe asset '+x)
for x in sys.argv[2:]:
 p=Path(x).resolve()
 if not p.is_relative_to(root) or p.exists():raise SystemExit('Existing/unsafe output '+x)
PY
mkdir -p "$LOG_DIR" "$MODEL_DIR"
date -u +%FT%TZ > "$LOG_DIR/start_time.txt"
COMMON=(--model_name "$MODEL" --dataset "$DATASET" --path ./data/
 --gpu "$GPU" --random_seed "$SEED" --emb_size 64 --attn_size 64 --K "$K" --history_max 20
 --item_encoder "$ENCODER" --llm_emb_path "./data/${DATASET}/handled/llm_table_pca1536.pkl"
 --semantic_rank 512 --semantic_dim 32 --semantic_hidden 128 --complement_dim 32
 --tail_hidden 64 --complement_hidden 64 --gate_hidden 64 --aspcf_gate_mode basic
 --lambda_relation .01 --relation_sample_size 128 --relation_teacher_temp .1 --relation_student_temp .1
 --dropout .1 --lr "$LR" --l2 1e-6 --optimizer Adam
 --batch_size 1024 --eval_batch_size 256 --num_neg 1 --epoch 200 --early_stop 10
 --metric NDCG,HR --topk 5,10,20,50 --num_workers 5 --test_epoch -1 --regenerate 0
 --sampled_softmax_temperature 1 --test_all 0
 --model_path "$MODEL_DIR/best.pt")
START=$SECONDS
python -B main.py "${COMMON[@]}" --train 1 --load 0 --dev_only 1 --log_file "$LOG_DIR/train.log" 2>&1 | tee "$LOG_DIR/train.out"
TRAIN_ELAPSED=$((SECONDS-START))
python -B - "$LOG_DIR" "$MODEL_DIR/best.pt" "$TRAIN_ELAPSED" "${COMMON[@]}" <<'PY'
from pathlib import Path
import sys,json,re,hashlib
d=Path(sys.argv[1]);t=(d/'train.log').read_text()
assert ' END: ' in t and 'Test After Training:' not in t
best=re.findall(r'Best Iter\(dev\)=.*',t);assert best
cp=Path(sys.argv[2])
with (d/'dev_selection.json').open('x') as f:
 json.dump(dict(best_dev=best[-1],training_seconds=int(sys.argv[3]),checkpoint=str(cp),
                checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),
                common_command=sys.argv[4:],test_phase='exploratory_test',selection='Dev NDCG@5 only'),f,indent=2)
PY
DEV_START=$SECONDS
python -B main.py "${COMMON[@]}" --train 0 --load 1 --dev_only 1 --log_file "$LOG_DIR/dev_full.log" 2>&1 | tee "$LOG_DIR/dev_full.out"
DEV_ELAPSED=$((SECONDS-DEV_START))
TEST_START=$SECONDS
python -B main.py "${COMMON[@]}" --train 0 --load 1 --dev_only 0 --log_file "$LOG_DIR/exploratory_test.log" 2>&1 | tee "$LOG_DIR/exploratory_test.out"
TEST_ELAPSED=$((SECONDS-TEST_START))
python -B - "$LOG_DIR" "$TEST_ELAPSED" "$DEV_ELAPSED" <<'PY'
from pathlib import Path
import sys,json,re,hashlib
d=Path(sys.argv[1]);s=json.loads((d/'dev_selection.json').read_text())
t=(d/'exploratory_test.log').read_text()
assert ' END: ' in t and 'Test After Training:' in t
assert hashlib.sha256(Path(s['checkpoint']).read_bytes()).hexdigest()==s['checkpoint_sha256']
dev=(d/'dev_full.log').read_text();assert 'Dev Evaluation:' in dev and 'Test After Training:' not in dev
s.update(status='complete',test_seconds=int(sys.argv[2]),dev_evaluation_seconds=int(sys.argv[3]),
         dev_full=[x for x in dev.splitlines() if 'Dev Evaluation:' in x][-1],
         test=[x for x in t.splitlines() if 'Test After Training:' in x][-1],
         checkpoint_unchanged=True)
with (d/'summary.json').open('x') as f:json.dump(s,f,indent=2)
print(json.dumps(s))
PY
