#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/yyx/anaconda3/etc/profile.d/conda.sh
conda activate hzg_py10
export PYTHONDONTWRITEBYTECODE=1
DATASET=${1:?dataset}
GPU=${2:?GPU}
MODE=${3:?C or I}
SEED=${4:-0}
[[ "$SEED" == 0 ]] || { echo "Locked seed0 round"; exit 2; }
case "$MODE" in C|I) ;; *) exit 2 ;; esac
case "$DATASET" in beauty) LR=.004 ;; ml-1m) LR=.001 ;; *) exit 2 ;; esac
MODEL=LLMMIRecASPCFInterestMetric
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}
LOG_DIR="new_log/aspcf_interest_metric/${DATASET}/${MODE}_k4_h20_seed${SEED}_${RUN_ID}"
MODEL_DIR="new_model/aspcf_interest_metric/${DATASET}/${MODE}_k4_h20_seed${SEED}_${RUN_ID}"
python -B - "$DATASET" "$LOG_DIR" "$MODEL_DIR" "$MODE" "$SEED" <<'PY'
from pathlib import Path
import sys,re,json
root=Path.cwd().resolve();ds=sys.argv[1]
for x in [f'data/{ds}/{y}' for y in ['train.csv','dev.csv','test.csv','SeqReader.pkl','handled/llm_table_pca1536.pkl']]:
 p=Path(x).resolve()
 if not p.is_relative_to(root) or not p.is_file():raise SystemExit('Missing/unsafe asset '+x)
for x in sys.argv[2:4]:
 p=Path(x).resolve()
 if not p.is_relative_to(root) or p.exists():raise SystemExit('Existing/unsafe output '+x)
old=Path(f'new_log/llmmirec_aspcf_phase2/{ds}/seed42/LLMMIRecASPCF_seed42.log').read_text()
oldargs=dict(re.findall(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s+\|\s*(.*?)\s*$',old,re.M))
same=dict(K=4,attn_size=64,batch_size=1024,eval_batch_size=256,history_max=20,emb_size=64,num_neg=1,
 dropout=.1,lambda_relation=.01,lr=.004 if ds=='beauty' else .001,l2=1e-6,epoch=200,early_stop=10,
 grad_clip=0,align_warmup=5,align_ramp=15,semantic_rank=512,semantic_dim=32,semantic_hidden=128,
 complement_dim=32,complement_hidden=64,tail_hidden=64,gate_hidden=64,relation_sample_size=128,
 relation_student_temp=.1,relation_teacher_temp=.1,test_all=0)
for k,v in same.items():assert float(oldargs[k])==v,(k,oldargs.get(k),v)
assert oldargs['optimizer']=='Adam' and oldargs['aspcf_gate_mode']=='basic' and oldargs['item_encoder']=='aspcf'
print(json.dumps(dict(historical_BPR_source=f'new_log/llmmirec_aspcf_phase2/{ds}/seed42/LLMMIRecASPCF_seed42.log',
 matched_configuration=same,optimizer='Adam',new_mode=sys.argv[4],added_parameters=256,
 intentional_changes=['model and C/I diagonal metric','unique output','train dev-only; end best checkpoint Dev/Test'],
 not_present=['sampled_softmax_temperature','train_membership','in-batch scoring']),indent=2))
# Verify C/I share cold initialization on the actual corpus, with a=0.
import argparse,pickle,torch
from models.sequential.LLMMIRecASPCFInterestMetric import LLMMIRecASPCFInterestMetric as M
torch.set_num_threads(2)
corpus=pickle.load(open(Path('data')/ds/'SeqReader.pkl','rb'))
initial=None
for mode in ['C','I']:
 args=M.parse_model_args(argparse.ArgumentParser()).parse_args([])
 args.device=torch.device('cpu');args.model_path='unused';args.buffer=0
 args.llm_emb_path=f'data/{ds}/handled/llm_table_pca1536.pkl';args.metric_mode=mode
 torch.manual_seed(int(sys.argv[5]))
 model=M(args,corpus)
 assert int(torch.count_nonzero(model.a))==0 and model.a.numel()==256
 if initial is None:initial={k:v.clone() for k,v in model.state_dict().items()}
 else:assert all(torch.equal(v,initial[k]) for k,v in model.state_dict().items())
print(json.dumps(dict(initial_shared_parameters_equal=True,new_a_all_zero=True,initialization_seed=int(sys.argv[5]),parameters=model.count_variables(),added=256)))

PY
mkdir -p "$LOG_DIR" "$MODEL_DIR"
date -u +%FT%TZ > "$LOG_DIR/start_time.txt"
COMMON=(--model_name "$MODEL" --dataset "$DATASET" --path ./data/
 --gpu "$GPU" --random_seed "$SEED" --emb_size 64 --attn_size 64 --K 4 --history_max 20
 --item_encoder aspcf --llm_emb_path "./data/${DATASET}/handled/llm_table_pca1536.pkl"
 --semantic_rank 512 --semantic_dim 32 --semantic_hidden 128 --complement_dim 32
 --tail_hidden 64 --complement_hidden 64 --gate_hidden 64 --aspcf_gate_mode basic
 --lambda_relation .01 --relation_sample_size 128 --relation_teacher_temp .1 --relation_student_temp .1
 --dropout .1 --lr "$LR" --l2 1e-6 --optimizer Adam --grad_clip 0 --align_warmup 5 --align_ramp 15
 --batch_size 1024 --eval_batch_size 256 --num_neg 1 --epoch 200 --early_stop 10
 --metric NDCG,HR --topk 5,10,20,50 --num_workers 5 --test_epoch -1 --regenerate 0 --test_all 0
 --metric_mode "$MODE" --model_path "$MODEL_DIR/best.pt")
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
saved=re.findall(r'Epoch\s+(\d+).* \*$',t,re.M);assert saved
with (d/'dev_selection.json').open('x') as f:
 json.dump(dict(best_dev=best[-1],checkpoint_epoch=int(saved[-1]),training_seconds=int(sys.argv[3]),checkpoint=str(cp),
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
import time,datetime
mechanism_start=time.monotonic()
# Lightweight best-checkpoint mechanism check; CPU, Train/Dev only.
import argparse,pickle,ast,itertools,numpy as np,torch
from models.sequential.LLMMIRecASPCFInterestMetric import LLMMIRecASPCFInterestMetric as M
from models.BaseModel import GeneralModel
torch.set_num_threads(2)
args,_=M.parse_model_args(argparse.ArgumentParser(allow_abbrev=False)).parse_known_args(s['common_command'])
args.dataset=s['common_command'][s['common_command'].index('--dataset')+1]
args.device=torch.device('cpu');args.model_path=s['checkpoint'];args.buffer=0
corpus=pickle.load(open(Path('data')/args.dataset/'SeqReader.pkl','rb'))
model=M(args,corpus)
state=torch.load(s['checkpoint'],map_location='cpu',weights_only=True)
assert all(bool(torch.isfinite(v).all()) for v in state.values() if torch.is_tensor(v))
model.load_state_dict(state);model.eval()
rng=np.random.default_rng(20261014)
def batch(phase):
 df=corpus.data_df[phase]
 indices=rng.choice(np.flatnonzero(df.position.to_numpy()>0),32,replace=False)
 histories=[];lengths=[];candidates=[]
 for j in indices:
  row=df.iloc[int(j)];user=int(row.user_id);pos=int(row.position)
  hist=[int(x[0]) for x in corpus.user_his[user][max(0,pos-20):pos]]
  histories.append(hist+[0]*(20-len(hist)));lengths.append(len(hist))
  if phase=='train':
   neg=0
   while neg==0 or neg in corpus.train_clicked_set[user]:
    neg=int(rng.integers(1,corpus.n_items))
   ns=[neg]
  else:
   ns=row.neg_items
   if isinstance(ns,str):ns=ast.literal_eval(ns)
  candidates.append([int(row.item_id)]+[int(x) for x in ns])
 return dict(history_items=torch.tensor(histories),lengths=torch.tensor(lengths),
             item_id=torch.tensor(candidates))
with torch.no_grad():
 out=model(batch('dev'),True)
 V,w,u,e=out['interest_vectors'],out['interest_weights'],out['base_user_vector'],out['candidate_vectors']
 uc=u+M.correction(V,w,u,model.a,'C')
 ui=u+M.correction(V,w,u,model.a,'I')
 sc=(uc[:,None]*e).sum(-1);si=(ui[:,None]*e).sum(-1)
 base=(u[:,None]*e).sum(-1)
 correction=(out['metric_correction'][:,None]*e).sum(-1)
 topc=np.argsort(-sc.numpy(),axis=1)[:,:5];topi=np.argsort(-si.numpy(),axis=1)[:,:5]
 changed=np.any(np.sort(topc,axis=1)!=np.sort(topi,axis=1),axis=1)
 norms=[float((model.a[i]-model.a[j]).double().norm()) for i,j in itertools.combinations(range(4),2)]
 mechanism=dict(mode=args.metric_mode,parameters=model.count_variables(),added_parameters=model.a.numel(),
  a_norm=float(model.a.double().norm()),a_max_abs=float(model.a.abs().max()),
  one_plus_a_min=float((1+model.a).min()),one_plus_a_max=float((1+model.a).max()),
  one_plus_a_negative_fraction=float(((1+model.a)<0).double().mean()),
  base_to_transformed_top5_slot_set_changed_fraction=float(np.any(np.sort(np.argsort(-base.numpy(),axis=1)[:,:5],axis=1)!=np.sort(np.argsort(-out['prediction'].numpy(),axis=1)[:,:5],axis=1),axis=1).mean()),
  score_correction_definition='Frobenius norm of correction scores / Frobenius norm of base scores; 32 fixed Dev users x 1001 original candidate slots',
  a_head_norms=model.a.double().norm(dim=1).tolist(),a_pair_difference_norms=norms,
  score_correction_ratio=float(correction.double().norm()/base.double().norm().clamp_min(1e-12)),
  same_weights_C_I_user_difference_ratio=float((ui-uc).double().norm()/u.double().norm().clamp_min(1e-12)),
  same_weights_C_I_top5_slot_set_changed_fraction=float(changed.mean()),
  same_weights_C_I_max_score_difference=float((si-sc).abs().max()),
  finite_dev_scores=bool(torch.isfinite(out['prediction']).all()),
  probe='32 fixed-rule Dev users, full1001 slots; C/I counterfactual on SAME trained weights, not retrained causal comparison')
rankout=model(batch('train'),True)
ranking=GeneralModel.loss(model,rankout)
gradient=torch.autograd.grad(ranking,model.a)[0]
mechanism.update(train_probe_ranking_loss=float(ranking.detach()),
                 train_probe_a_gradient_norm=float(gradient.double().norm()),
                 train_probe_gradient_finite=bool(torch.isfinite(gradient).all()))
s['mechanism']=mechanism
assert hashlib.sha256(Path(s['checkpoint']).read_bytes()).hexdigest()==s['checkpoint_sha256']

s['mechanism_seconds']=time.monotonic()-mechanism_start
s['training_anomaly_lines']=re.findall(r'^.*(?:loss=(?:nan|inf)|Traceback|RuntimeError).*$',(d/'train.log').read_text(),re.M|re.I)
s['end_epoch']=int(re.findall(r'Epoch\s+(\d+)',(d/'train.log').read_text())[-1])
epoch_times=re.findall(r'Epoch\s+\d+\s+loss=[^ ]+ \[([\d.]+) s\].*?\[([\d.]+) s\]',(d/'train.log').read_text())
s['mean_train_dev_epoch_seconds']=sum(float(a)+float(b) for a,b in epoch_times)/len(epoch_times)
s['end_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
s['wall_seconds_since_start']=int((datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat((d/'start_time.txt').read_text().strip().replace('Z','+00:00'))).total_seconds())
with (d/'summary.json').open('x') as f:json.dump(s,f,indent=2)
print(json.dumps(s))
PY
