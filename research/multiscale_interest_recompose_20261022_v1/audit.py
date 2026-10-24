"""Math + gradients + 16 Train steps and Dev only. No full-training or Test entry."""
import os
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
os.environ.setdefault("OMP_NUM_THREADS","4");os.environ.setdefault("MKL_NUM_THREADS","4")
import sys,json,time,pickle,hashlib,signal,copy,csv,subprocess,importlib.util,math
from pathlib import Path
ROOT=Path("/home/yyx/hzgcode/pom2.0").resolve();OUT=Path(__file__).resolve().parent
assert OUT.is_relative_to(ROOT);sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OUT))
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from structure import LocalRecomposer,RecomposeModel
REUSE=ROOT/"research/frozen_readout_softmax_20261022_v1/probe.py"
assert hashlib.sha256(REUSE.read_bytes()).hexdigest()=="fae7c3155477448075e58d94212ad637bc420d26449b3137bbcc953336e4c81d"
sp=importlib.util.spec_from_file_location("readonly_feature_helpers",REUSE)
reuse=importlib.util.module_from_spec(sp);sp.loader.exec_module(reuse);old=reuse.old
EXTRA=["models/sequential/LLMMIRecASPCFTIA.py","models/sequential/LLMMIRecASPCFSAIT.py","models/sequential/sait_components.py",
 "models/sequential/LLMMIRecASPCFPrefixEvidence.py","models/sequential/LLMMIRecContextControl.py","models/sequential/LLMMIRecHSDIR.py"]
CONFIG=dict(seed=42,batch_size=1024,steps=16,lr=.0001,optimizer="Adam",weight_decay=1e-6,
 modes=["baseline","local","interest"],dilations=[1,3],kernel=3,loss="unchanged filtered Softmax tau1 + original relation .01; NO auxiliary loss",
 init="same fixed existing checkpoint for this implementation screen only; full experiment must use paired fresh original ASPCF initialization",
 Dev="full original 1001 duplicate candidate slots before/after, raw and fixed pop lambda=.30; no selection",
 K="configurable; existing checkpoints K4; synthetic K1/2/4/7",
 no_Test=True,no_full_training=True,global_audit_seconds=170)
def sha(p):return old.sha(p)
def write(p,x):
 p=Path(p);assert p.resolve().is_relative_to(OUT)
 with p.open("x") as f:json.dump(x,f,indent=2,allow_nan=False)
def csvwrite(p,rows):return reuse.csvwrite(p,rows)
def snap():return torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()
def restore(s):torch.set_rng_state(s[0]);torch.cuda.set_rng_state(s[1])
def math_tests():
 torch.set_num_threads(4);torch.manual_seed(42)
 records=[]
 for mode in ("local","interest"):
  m=LocalRecomposer(64,mode).double()
  assert sum(p.numel() for p in m.parameters())==16960
  x=torch.randn(4,20,64,dtype=torch.float64);lengths=torch.tensor([0,1,2,20])
  valid=torch.arange(20)[None]<lengths[:,None]
  for K in (1,2,4,7):
   v=torch.randn(4,K,64,dtype=torch.float64);a=torch.randn(4,K,20,dtype=torch.float64).softmax(-1)*valid[:,None]
   a=a/a.sum(-1,keepdim=True).clamp_min(1e-12)
   delta,read=m(x,v,a,lengths)
   assert delta.shape==v.shape and torch.count_nonzero(delta)==0
  with torch.no_grad():m.out.weight.normal_(0,.05)
  v=torch.randn(4,4,64,dtype=torch.float64);a=torch.randn(4,4,20,dtype=torch.float64).softmax(-1)*valid[:,None]
  a=a/a.sum(-1,keepdim=True).clamp_min(1e-12)
  delta,read=m(x,v,a,lengths)
  altered=x.clone();altered[~valid]=torch.randn_like(altered[~valid])*100
  dd,_=m(altered,v,a,lengths)
  assert torch.equal(delta,dd) and torch.count_nonzero(delta[0])==0
  # Future perturbation and extension invariance tested on the causal BANK,
  # not on the final global-interest-conditioned readout.
  full=torch.randn(1,20,64,dtype=torch.float64,requires_grad=True)
  changed=full.detach().clone();changed[:,8:]+=100
  bank,_=m.bank(full,torch.tensor([20]));other,_=m.bank(changed,torch.tensor([20]))
  assert torch.equal(bank[:,:,:8],other[:,:,:8])
  prefix,_=m.bank(full[:,:8],torch.tensor([8]))
  assert torch.allclose(bank[:,:,:8],prefix,atol=1e-12,rtol=1e-12)
  g=torch.autograd.grad(bank[:,:,7].sum(),full)[0]
  assert torch.count_nonzero(g[:,8:])==0
  # Head order equivariance and explicit conditioning check.
  perm=torch.tensor([2,0,3,1]);pd,_=m(x,v[:,perm],a[:,perm],lengths)
  assert torch.allclose(pd,delta[:,perm],atol=1e-12,rtol=1e-12)
  vv=v.clone();vv[:,0]+=torch.randn_like(vv[:,0])*3
  shifted,_=m(x,vv,a,lengths)
  effect=float((shifted-delta).abs().max())
  if mode=="interest":assert effect>1e-8
  else:assert effect==0
  records.append(dict(mode=mode,parameters=16960,zero_output_identity=True,padding_invariant=True,
   causal_suffix_and_prefix_extension=True,future_bank_gradient_exact_zero=True,head_permutation_equivariant=True,
   conditioning_effect_max_abs=effect))
 return records
def prepare():
 checks=math_tests()
 write(OUT/"PREREG.json",dict(config=CONFIG,math_tests=checks,head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
  initial_status=subprocess.check_output(["git","status","--short","--branch"],cwd=ROOT,text=True),
  source_sha256={p:sha(ROOT/p) for p in old.SOURCES+EXTRA},module_sha256=sha(OUT/"structure.py"),runner_sha256=sha(__file__)))
 print("MATH_PASS "+json.dumps(checks),flush=True)
def negative(m,u,g):
 ii=torch.randint(1,m.item_num,(len(u),),generator=g,device=m.device)
 while True:
  bad=(((m.train_membership[u,ii//64]>>(ii%64))&1)!=0)
  if not bool(bad.any()):break
  ii[bad]=torch.randint(1,m.item_num,(int(bad.sum()),),generator=g,device=m.device)
 return ii
def feed(hist,lengths,user,target,neg,device):
 return dict(history_items=torch.as_tensor(hist,device=device),lengths=torch.as_tensor(lengths,device=device),
  user_id=torch.as_tensor(user,device=device),item_id=torch.as_tensor(np.column_stack([target,neg]),device=device))
def statehash(m):return reuse.statehash(m)
def init_model(a,c,state,mode):
 m=old.Model(a,c) if mode=="baseline" else RecomposeModel(a,c,mode)
 missing,unexpected=m.load_state_dict(state,strict=False)
 expected=[] if mode=="baseline" else ["local_block."+k for k in m.local_block.state_dict()]
 assert set(missing)==set(expected) and not unexpected
 return m.to(a.device)
def equivalence(models,f):
 before=snap();reference=models["baseline"];reference.train()
 o=reference(f,True);loss=reference.loss(o);gg=torch.autograd.grad(loss,reference.extractor.query)[0]
 post=snap()
 for name in ("local","interest"):
  m=models[name];m.train();restore(before);oo=m(f,True);ll=m.loss(oo);gq=torch.autograd.grad(ll,m.extractor.query)[0]
  assert torch.equal(o["prediction"],oo["prediction"]) and torch.equal(loss,ll) and torch.equal(gg,gq)
 restore(post)
 # Candidate changes must never change the interest representation in eval.
 result={}
 for name in ("local","interest"):
  m=models[name];m.eval()
  with torch.no_grad():
   a=m(f,True);alt={**f,"item_id":f["item_id"].flip(1)};b=m(alt,True)
   assert torch.equal(a["interest_vectors"],b["interest_vectors"])
   result[name]=dict(zero_initial_score_loss_query_gradient_exact=True,candidate_invariant_interests=True)
 return result
@torch.no_grad()
def encode_users(m,table,hist,lengths):
 users=[]
 for off in range(0,len(hist),1024):
  h=torch.as_tensor(hist[off:off+1024],device=m.device);l=torch.as_tensor(lengths[off:off+1024],device=m.device)
  raw=table[h];valid=h>0
  ar=torch.arange(h.shape[1],device=m.device);pos=(l[:,None]-ar[None])*valid
  v,a=m.extractor(raw+m.position_emb(pos),l)
  if hasattr(m,"local_block"):v=v+m.local_block(raw,v,a,l)[0]
  w=m.aggregator(raw,l);users.append((v*w[:,:,None]).sum(1))
 return torch.cat(users)
@torch.no_grad()
def evaluate(m,c,hist,lengths,candidates,pop,strength):
 m.eval();table=torch.cat([m.item_encoder(torch.arange(i,min(i+2048,c.n_items),device=m.device)) for i in range(0,c.n_items,2048)])
 u=encode_users(m,table,hist,lengths);rr=[];rp=[]
 for off in range(0,len(hist),256):
  sl=slice(off,off+256);ii=torch.as_tensor(candidates[sl],device=m.device)
  raw=(u[sl,None]*table[ii]).sum(-1).cpu().numpy().astype(np.float64)
  p=pop[ii].cpu().numpy().astype(np.float64)
  rr.append(reuse.ranks(raw));rp.append(reuse.ranks(raw+strength*(p-p.mean(1,keepdims=True))))
 return dict(raw=reuse.metrics(np.concatenate(rr)),pop=reuse.metrics(np.concatenate(rp)))
def run_dataset(ds,pre):
 started=time.perf_counter();old.utils.init_seed(42);spec=old.SPEC[ds]
 prior=ROOT/"research/popularity_interest_audit_20261022_v1"/ds/"DEV_LOCK.json";lock=json.loads(prior.read_text())
 paths=dict(checkpoint=spec["checkpoint"],embedding=spec["embedding"],train=ROOT/"data"/ds/"train.csv",
  dev=ROOT/"data"/ds/"dev.csv",cache=ROOT/"data"/ds/"SeqReader.pkl",prior_lock=prior)
 inputs={k:sha(p) for k,p in paths.items()}
 assert inputs["checkpoint"]==lock["checkpoint_sha256"] and inputs["embedding"]==lock["embedding_sha256"] and lock["lambda_normalized"]==.30
 assert all(sha(ROOT/n)==v for n,v in pre["source_sha256"].items())
 with paths["cache"].open("rb") as f:c=pickle.load(f)
 c.data_df={p:c.data_df[p] for p in ("train","dev")};tr=c.data_df["train"];dv=c.data_df["dev"];counts=tr.groupby("user_id").size()
 assert np.all(tr.position.to_numpy()<tr.user_id.map(counts).to_numpy()) and np.all(dv.position.to_numpy()==dv.user_id.map(counts).to_numpy())
 for u in c.user_his:c.user_his[u]=c.user_his[u][:int(counts.get(u,0))]
 ix=np.random.default_rng(42).choice(np.flatnonzero(tr.position.to_numpy()>0),16*1024,replace=False);tf=tr.iloc[ix]
 th,tl=reuse.construct_hist(c,tf);dh,dl=reuse.construct_hist(c,dv)
 candidates=np.concatenate([dv.item_id.to_numpy(np.int64)[:,None],np.stack(dv.neg_items.to_numpy())],1);assert candidates.shape==(len(dv),1001)
 a=old.arguments(ds,c);a.lr=.0001
 state=torch.load(spec["checkpoint"],map_location="cpu",weights_only=True)
 models={name:init_model(a,c,state,name) for name in CONFIG["modes"]}
 added={name:sum(p.numel() for p in m.parameters())-sum(t.numel() for t in state.values()) for name,m in models.items()}
 # State dict contains exactly all original parameters; no persistent LLM table.
 assert added=={"baseline":0,"local":16960,"interest":16960},added
 assert statehash(models["local"].local_block)==statehash(models["interest"].local_block)
 gen=torch.Generator(device=a.device).manual_seed(20261022);user=tf.user_id.to_numpy(np.int64);target=tf.item_id.to_numpy(np.int64)
 ng=negative(models["baseline"],torch.as_tensor(user[:32],device=a.device),gen).cpu().numpy()
 probe=feed(th[:32],tl[:32],user[:32],target[:32],ng,a.device)
 checks=equivalence(models,probe)
 rawcount=np.bincount(pd.read_csv(paths["train"],sep="\t",usecols=["item_id"]).item_id.to_numpy(np.int64),minlength=c.n_items)
 pop=torch.as_tensor(np.log1p(rawcount),dtype=torch.float32,device=a.device);strength=lock["raw_lambda"]
 initial=evaluate(models["baseline"],c,dh,dl,candidates,pop,strength)
 assert all(abs(initial["pop"][k]-lock["dev_metrics"][k])<1e-10 for k in reuse.METRICS)
 opts={n:torch.optim.Adam(m.parameters(),lr=.0001,weight_decay=1e-6) for n,m in models.items()}
 grads={n:[] for n in models};logs=[];timing={n:0. for n in models};fitstart=time.perf_counter()
 for step in range(16):
  sl=slice(step*1024,(step+1)*1024);uu=torch.as_tensor(user[sl],device=a.device)
  nn=negative(models["baseline"],uu,gen).cpu().numpy();f=feed(th[sl],tl[sl],user[sl],target[sl],nn,a.device)
  before=snap();after=None;losses={}
  for name,m in models.items():
   m.train();restore(before);opts[name].zero_grad(set_to_none=True)
   torch.cuda.synchronize();ts=time.perf_counter()
   out=m(f,True);loss=m.loss(out);assert torch.isfinite(loss);loss.backward()
   assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
   qg=float(m.extractor.query.grad.norm());assert qg>0
   grad=dict(query=qg)
   if name!="baseline":
    for key,param in dict(out=m.local_block.out.weight,conv=m.local_block.convs[0].weight,point=m.local_block.point.weight,Wq=m.local_block.Wq.weight,Wk=m.local_block.Wk.weight).items():
     assert param.grad is not None and torch.isfinite(param.grad).all();grad[key]=float(param.grad.norm())
    if step==0:assert grad["out"]>0
    if step==1:assert all(grad[k]>0 for k in ("out","conv","point","Wq","Wk"))
    grad["delta_RMS"]=float(out["local_delta"].detach().square().mean().sqrt())
   grads[name].append(grad);opts[name].step();losses[name]=float(loss.detach())
   torch.cuda.synchronize();timing[name]+=time.perf_counter()-ts
   current=snap()
   if after is None:after=current
   else:assert all(torch.equal(t1,t2) for t1,t2 in zip(current,after))
  restore(after);logs.append(dict(step=step+1,loss=losses))
 fitseconds=time.perf_counter()-fitstart
 final={n:evaluate(m,c,dh,dl,candidates,pop,strength) for n,m in models.items()}
 changes={n:float((m.extractor.query.detach().cpu()-state["extractor.query"]).norm()) for n,m in models.items()}
 assert all(v>0 for v in changes.values())
 assert all(sha(p)==inputs[k] for k,p in paths.items()) and all(sha(ROOT/n)==v for n,v in pre["source_sha256"].items())
 result=dict(dataset=ds,head=pre["head"],config=CONFIG,added_parameters=added,base_parameters=sum(t.numel() for t in state.values()),
  input_sha256=inputs,source_sha256=pre["source_sha256"],module_sha256=pre["module_sha256"],runner_sha256=pre["runner_sha256"],
  initial_Dev=initial,final_Dev=final,real_checks=checks,step_gradients=grads,short_step_losses=logs,
  per_arm_16step_seconds=timing,shared_fit_seconds=fitseconds,total_seconds=time.perf_counter()-started,
  peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20,query_parameter_change_L2=changes,
  Train_subset_sha256=reuse.datahash(ix),Dev_candidate_sha256=reuse.datahash(candidates),Test_evaluations=0,training_checkpoints_written=0)
 write(OUT/(ds+".json"),result)
 rows=[]
 for stage,schemes in (("initial",{"existing":initial}),("after16",final)):
  for name,scores in schemes.items():
   for mode,mm in scores.items():rows.append(dict(stage=stage,scheme=name,scoring=mode,**mm))
 csvwrite(OUT/(ds+"_dev_metrics.csv"),rows)
 print("DATASET_COMPLETE "+json.dumps({k:result[k] for k in ("dataset","initial_Dev","final_Dev","per_arm_16step_seconds","total_seconds","peak_GPU_MiB")}),flush=True)
 del models,opts,c;torch.cuda.empty_cache()
 return result
def runall():
 signal.alarm(170);start=time.perf_counter();torch.set_num_threads(4)
 pre=json.loads((OUT/"PREREG.json").read_text())
 assert sha(__file__)==pre["runner_sha256"] and sha(OUT/"structure.py")==pre["module_sha256"]
 res=[run_dataset(ds,pre) for ds in ("beauty","ml-1m")]
 write(OUT/"COMPLETE.json",dict(body_seconds=time.perf_counter()-start,datasets=[d["dataset"] for d in res],
  no_Test=True,no_full_training=True,module_sha256=pre["module_sha256"],runner_sha256=pre["runner_sha256"]))
 print("ALL_COMPLETE",flush=True)
if __name__=="__main__":
 import argparse
 p=argparse.ArgumentParser();p.add_argument("--prepare",action="store_true");args=p.parse_args()
 if args.prepare:prepare()
 else:runall()

