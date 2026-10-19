#!/usr/bin/env bash
# Independent SRKI protocol; dry-run by default. No Test path.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DATASET=${1:?dataset required}
GPU=${2:-1}
SEED=${3:-42}
SOURCE=${4:-llm}
ACTION=${5:---dry-run}
K=${K:-2}
case "$DATASET" in beauty) LR=.004;; ml-1m) LR=.001;; *) echo "Only beauty/ml-1m"; exit 2;; esac
case "$SOURCE" in llm|aspcf|permuted_llm) ;; *) exit 2;; esac
[[ "$K" =~ ^[1-9][0-9]*$ ]] || { echo "K must be positive"; exit 2; }
[[ "$ACTION" == "--dry-run" || "$ACTION" == "--execute" ]] || exit 2
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
case "$RUN_ID" in *[!A-Za-z0-9_.-]*|'') echo "Unsafe RUN_ID"; exit 2;; esac
LOGDIR="new_log/aspcf_srki/$DATASET/${SOURCE}_k${K}_h20_seed${SEED}_${RUN_ID}"
MODELDIR="new_model/aspcf_srki/$DATASET/${SOURCE}_k${K}_h20_seed${SEED}_${RUN_ID}"
CMD=(python -B main.py --model_name LLMMIRecASPCFSRKI --dataset "$DATASET"
 --path ./data/ --gpu "$GPU" --random_seed "$SEED"
 --emb_size 64 --attn_size 64 --K "$K" --history_max 20 --item_encoder aspcf
 --llm_emb_path "./data/$DATASET/handled/llm_table_pca1536.pkl"
 --semantic_rank 512 --semantic_dim 32 --semantic_hidden 128 --complement_dim 32
 --tail_hidden 64 --complement_hidden 64 --gate_hidden 64 --aspcf_gate_mode basic
 --srki_kernel_source "$SOURCE" --srki_permutation_seed 20261016
 --lambda_relation .01 --relation_sample_size 128 --relation_teacher_temp .1 --relation_student_temp .1
 --dropout .1 --lr "$LR" --l2 1e-6 --optimizer Adam --batch_size 1024 --eval_batch_size 256
 --num_neg 1 --epoch 200 --early_stop 10 --topk 5,10,20 --metric NDCG,HR --num_workers 5
 --dev_only 1 --test_epoch -1 --regenerate 0 --test_all 0 --model_path "$MODELDIR/best.pt")
printf '%q ' "${CMD[@]}" --train 1 --load 0 --log_file "$LOGDIR/train.log"; printf '\n'
if [[ "$ACTION" == "--dry-run" ]]; then exit 0; fi
# Explicit execution is required. Never reuse a preexisting result directory.
[[ ! -e "$LOGDIR" && ! -e "$MODELDIR" ]] || { echo "Output collision"; exit 2; }
[[ -f "data/$DATASET/SeqReader.pkl" && -f "data/$DATASET/handled/llm_table_pca1536.pkl" ]] || exit 2
source /home/yyx/anaconda3/etc/profile.d/conda.sh
conda activate hzg_py10
export PYTHONDONTWRITEBYTECODE=1
for p in "$LOGDIR" "$MODELDIR"; do
  [[ "$(realpath -m "$p")" == "$ROOT/"* ]] || { echo "Output outside project"; exit 2; }
done
mkdir -p "$LOGDIR" "$MODELDIR"
python -B - "$LOGDIR" "$K" "$SOURCE" "$SEED" "${CMD[@]}" <<'PY'
import hashlib,json,sys,time
from pathlib import Path
out,K,source,seed,*cmd=sys.argv[1:]
payload=dict(model_version="srki-prototype-v1",K=int(K),kernel_source=source,seed=int(seed),
             command=cmd,dev_only=True,test_allowed=False,started_unix=time.time())
for p in ["models/sequential/LLMMIRecASPCFSRKI.py","models/sequential/LLMMIRecASPCF.py","models/sequential/llmmi_components.py","models/BaseModel.py","helpers/BaseRunner.py","main.py","tools/srki_round1_checks.py"]:
    payload.setdefault("source_sha256",{})[p]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
dataset=cmd[cmd.index("--dataset")+1]
for rel in ["train.csv","dev.csv","SeqReader.pkl","item2id.json","handled/llm_table_pca1536.pkl"]:
    p=Path("data",dataset,rel)
    assert p.is_file()
    digest=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):digest.update(block)
    payload.setdefault("asset_sha256",{})[str(p)]=dict(sha256=digest.hexdigest(),bytes=p.stat().st_size)
payload["preflight"]="new_log/aspcf_srki/preflight_real_round1_20261017.json"
Path(out,"config.json").write_text(json.dumps(payload,indent=2))
PY
python -B tools/srki_round1_checks.py main -- "${CMD[@]:3}" --train 1 --load 0 --log_file "$LOGDIR/train.log" > "$LOGDIR/train.out" 2>&1
BEFORE=$(sha256sum "$MODELDIR/best.pt" | cut -d' ' -f1)
"${CMD[@]}" --train 0 --load 1 --log_file "$LOGDIR/dev.log" > "$LOGDIR/dev.out" 2>&1
AFTER=$(sha256sum "$MODELDIR/best.pt" | cut -d' ' -f1)
[[ "$BEFORE" == "$AFTER" ]] || exit 3
python -B tools/srki_round1_checks.py best --gpu "$GPU" --config "$LOGDIR/config.json" --output "$LOGDIR/best_train_probe.json" > "$LOGDIR/best_train_probe.out" 2>&1
python -B - "$LOGDIR" "$BEFORE" <<'PY'
import json,re,sys,time
from pathlib import Path
out=Path(sys.argv[1]);config=json.loads((out/"config.json").read_text())
train=(out/"train.log").read_text();dev=(out/"dev.log").read_text()
assert "Test After Training" not in train+dev and " test=(" not in train+dev
best=re.findall(r"Best Iter\(dev\)=\s*(\d+)",train)[-1]
end=re.findall(r"Epoch\s+(\d+)\s+loss=",train)[-1]
line=[x for x in dev.splitlines() if "Dev Evaluation:" in x][-1]
metrics={k:float(v) for k,v in re.findall(r"((?:HR|NDCG)@\d+):([0-9.]+)",line)}
summary=dict(status="complete",config=config,best_reported_epoch=int(best),end_epoch=int(end),
             dev=metrics,checkpoint_sha256=sys.argv[2],checkpoint_unchanged=True,
             total_seconds=time.time()-config["started_unix"],test_executed=False)
curves=[]
for match in re.finditer(r"Epoch\s+(\d+)\s+loss=([0-9.eE+\-]+) \[([0-9.]+) s\]\s+dev=\((.*?)\) \[([0-9.]+) s\]( \*)?",train):
    ep,loss,tr,met,dv,saved=match.groups()
    curves.append(dict(epoch=int(ep),loss=float(loss),train_seconds=float(tr),dev_seconds=float(dv),
                       dev={k:float(v) for k,v in re.findall(r"((?:HR|NDCG)@\d+):([0-9.]+)",met)},
                       checkpoint_saved=bool(saved)))
assert len(curves)==summary["end_epoch"]
summary["checkpoint_saved_epoch"]=[x["epoch"] for x in curves if x["checkpoint_saved"]][-1]
summary["parameters"]=int(re.findall(r"#params: (\d+)",train)[-1])
summary["mean_epoch_seconds"]=sum(x["train_seconds"]+x["dev_seconds"] for x in curves)/len(curves)
summary["projected_200epoch_hours"]=200*summary["mean_epoch_seconds"]/3600
summary["training_resources"]=json.loads((out/"train.log.resources.json").read_text())
summary["best_train_probe"]=str(out/"best_train_probe.json")
(out/"dev_curve.json").write_text(json.dumps(curves,indent=2))
(out/"summary.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary))
PY
