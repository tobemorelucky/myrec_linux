"""Frozen Softmax readout screen: Train-only fitting, fixed prior, Dev-only selection."""
import os
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("MKL_NUM_THREADS","4")
import sys,json,time,pickle,hashlib,signal,argparse,csv,copy,importlib.util,subprocess
from pathlib import Path
ROOT=Path("/home/yyx/hzgcode/pom2.0").resolve()
OUT=Path(__file__).resolve().parent
assert OUT.is_relative_to(ROOT)
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
PREV=ROOT/"research/transition_dev_probe_20261022_v1/probe.py"
assert hashlib.sha256(PREV.read_bytes()).hexdigest()=="a08e68fe8270bc3c39760ce672ace772cf0cdfb7d215ff96b17161b9d0a3c58e"
sp=importlib.util.spec_from_file_location("fixed_assets_probe",PREV)
old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
from models.sequential.LLMMIRecASPCFSampledSoftmax import candidate_mask,masked_softmax_loss
METRICS=("HR@5","NDCG@5","HR@10","NDCG@10","HR@20","NDCG@20")
NAMES=("global","conditional","user_mlp")
CONFIG=dict(seed=42,Train_subset_max=131072,epochs=4,Dev_epochs=[0,1,2,4],batch_size=1024,
 lr=dict(global_weights=.01,conditional=.001,user_mlp=.001),optimizer="Adam",weight_decay=1e-6,
 hidden=32,frequency_lambda=.30,loss="original filtered in-batch Softmax tau1 with the SAME locked frequency score; relation constant under freeze omitted",
 selection="best Dev NDCG@5 per arm, earliest on tie; epoch0 included",
 continuation="conditional must improve BOTH domains by >=.001 N5 and beat matched MLP; conditional paired CI inspected",
 no_Test=True,no_full_model_training=True,no_hyperparameter_sweep=True,run_seconds_limit=170)
def sha(p):return old.sha(p)
def write(p,x):
 p=Path(p);assert p.resolve().is_relative_to(OUT)
 with p.open("x") as f:json.dump(x,f,indent=2,allow_nan=False)
def csvwrite(p,rows):
 with Path(p).open("x",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def statehash(m):
 h=hashlib.sha256()
 for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def ranks(x):return np.argwhere((-x).argsort(1)==0)[:,1]+1
def metrics(r):
 return {key:float(((r<=int(key.split("@")[1]))/(np.log2(r+1) if key.startswith("NDCG") else 1)).mean()) for key in METRICS}
def datahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def prepare():
 assert not (OUT/"PREREG.json").exists()
 write(OUT/"PREREG.json",dict(config=CONFIG,head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
  status_before=subprocess.check_output(["git","status","--short","--branch"],cwd=ROOT,text=True),
  script_sha256=sha(__file__),prior="research/popularity_interest_audit_20261022_v1/*/DEV_LOCK.json",
  architecture=dict(global_weights="softmax learned K constants, initialized log mean Train weights",
   conditional="softmax(original frozen logits + Linear128to32-GELU-Linear32to1 shared over heads); inputs [LayerNorm(Vk), frozen history mean+last context]; final layer zero",
   user_mlp="u + Linear64to32(no bias)-GELU-Linear32to64; input LayerNorm(u); final layer zero; both 4160 parameters")))
 print("PREPARED",flush=True)
class Global(nn.Module):
 def __init__(self,w):
  super().__init__();self.logits=nn.Parameter(w.mean(0).log())
 def forward(self,d):
  w=self.logits.softmax(0).expand(len(d["u"]),-1)
  return (w[:,:,None]*d["v"]).sum(1),w
class Conditional(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(128,32),nn.GELU(),nn.Linear(32,1,bias=False))
  nn.init.zeros_(self.net[-1].weight)
 def forward(self,d):
  v=d["v"];h=d["ctx"][:,None].expand(-1,v.shape[1],-1)
  x=torch.cat([F.layer_norm(v,(64,)),h],-1)
  delta=self.net(x).squeeze(-1)
  w=F.softmax(d["logits"]+delta,-1)
  return (w[:,:,None]*v).sum(1),w
class UserMLP(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(64,32,bias=False),nn.GELU(),nn.Linear(32,64))
  nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,d):return d["u"]+self.net(F.layer_norm(d["u"],(64,))),d["w"]
def construct_hist(c,frame):
 n=len(frame);hist=np.zeros((n,20),np.int64);lengths=np.zeros(n,np.int64)
 for ix,(u,p) in enumerate(zip(frame.user_id.to_numpy(),frame.position.to_numpy())):
  seq=c.user_his[int(u)][max(0,int(p)-20):int(p)];lengths[ix]=len(seq)
  hist[ix,:len(seq)]=[v[0] for v in seq]
 assert (lengths>0).all()
 return hist,lengths
@torch.no_grad()
def features(m,table,hist,lengths):
 result={key:[] for key in ("v","w","u","ctx","logits")}
 for off in range(0,len(hist),2048):
  h=torch.as_tensor(hist[off:off+2048],device=m.device);l=torch.as_tensor(lengths[off:off+2048],device=m.device)
  raw=table[h];valid=h>0;ar=torch.arange(h.shape[1],device=m.device)
  pos=(l[:,None]-ar[None])*valid
  v,_=m.extractor(raw+m.position_emb(pos),l)
  mean=(raw*valid[:,:,None]).sum(1)/valid.sum(1,keepdim=True)
  last=raw[torch.arange(len(h),device=m.device),l-1]
  ctx=m.aggregator.ln(mean+last);logits=m.aggregator.mlp(ctx);w=logits.softmax(-1)
  u=(w[:,:,None]*v).sum(1)
  for key,x in dict(v=v,w=w,u=u,ctx=ctx,logits=logits).items():result[key].append(x)
 return {key:torch.cat(v) for key,v in result.items()}
def sliced(d,ix):return {k:v[ix] for k,v in d.items()}
def boot(a,b):
 d=((a<=5)/np.log2(a+1))-((b<=5)/np.log2(b+1))
 g=np.random.default_rng(20261022);vals=[]
 for unused in range(1000):vals.append(float(d[g.integers(0,len(d),len(d))].mean()))
 return dict(delta=float(d.mean()),CI95=np.quantile(vals,[.025,.975]).tolist(),
  note="paired users, conditional on Dev-selected epochs; exploratory, no multiple-selection correction")
def run(ds):
 signal.alarm(170);start=time.perf_counter();torch.set_num_threads(4);old.utils.init_seed(42)
 assert sha(__file__)==json.loads((OUT/"PREREG.json").read_text())["script_sha256"]
 out=OUT/ds;out.mkdir(exist_ok=False)
 prior_root=ROOT/"research/popularity_interest_audit_20261022_v1"
 lock=json.loads((prior_root/ds/"DEV_LOCK.json").read_text());assert lock["lambda_normalized"]==.30
 s=old.SPEC[ds]
 paths=dict(checkpoint=s["checkpoint"],embedding=s["embedding"],train=ROOT/"data"/ds/"train.csv",
  dev=ROOT/"data"/ds/"dev.csv",cache=ROOT/"data"/ds/"SeqReader.pkl",prior_lock=prior_root/ds/"DEV_LOCK.json")
 inputs={k:sha(p) for k,p in paths.items()}
 assert inputs["checkpoint"]==lock["checkpoint_sha256"] and inputs["embedding"]==lock["embedding_sha256"]
 sources={name:sha(ROOT/name) for name in old.SOURCES}
 assert sources==lock["source_sha256"]
 with paths["cache"].open("rb") as f:c=pickle.load(f)
 c.data_df={phase:c.data_df[phase] for phase in ("train","dev")}
 counts=c.data_df["train"].groupby("user_id").size()
 tr=c.data_df["train"];dv=c.data_df["dev"]
 assert np.all(tr.position.to_numpy()<tr.user_id.map(counts).to_numpy())
 assert np.all(dv.position.to_numpy()==dv.user_id.map(counts).to_numpy())
 # Hard remove future sequence suffixes, then use the original target-prefix boundary.
 for u in c.user_his:c.user_his[u]=c.user_his[u][:int(counts.get(u,0))]
 a=old.arguments(ds,c);m=old.Model(a,c).to(a.device)
 m.load_state_dict(torch.load(s["checkpoint"],map_location=a.device,weights_only=True),strict=True)
 m.requires_grad_(False);m.eval();base_hash=statehash(m)
 with torch.no_grad():table=torch.cat([m.item_encoder(torch.arange(i,min(i+2048,c.n_items),device=a.device)) for i in range(0,c.n_items,2048)])
 raw=pd.read_csv(paths["train"],sep="\t",usecols=["item_id"]).item_id.to_numpy(np.int64)
 pop=torch.as_tensor(np.log1p(np.bincount(raw,minlength=c.n_items)),device=a.device,dtype=torch.float32)
 assert lock["raw_lambda"]==.30*lock["scale_factor"]
 strength=lock["raw_lambda"]
 eligible=np.flatnonzero(tr.position.to_numpy()>0);gen=np.random.default_rng(42)
 selected=np.sort(gen.choice(eligible,size=min(CONFIG["Train_subset_max"],len(eligible)),replace=False))
 tf=tr.iloc[selected];dh,dl=construct_hist(c,dv);th,tl=construct_hist(c,tf)
 train=features(m,table,th,tl);dev=features(m,table,dh,dl)
 candidates=np.stack(dv.neg_items.to_numpy());assert candidates.shape==(len(dv),1000)
 candidates=np.concatenate([dv.item_id.to_numpy(np.int64)[:,None],candidates],1)
 targets=torch.as_tensor(tf.item_id.to_numpy(np.int64),device=a.device)
 users=torch.as_tensor(tf.user_id.to_numpy(np.int64),device=a.device)
 models={"global":Global(train["w"]).to(a.device),"conditional":Conditional().to(a.device),"user_mlp":UserMLP().to(a.device)}
 params={k:sum(p.numel() for p in v.parameters()) for k,v in models.items()}
 assert params["conditional"]==params["user_mlp"]==4160
 with torch.no_grad():
  check=sliced(train,slice(0,32))
  for name in ("conditional","user_mlp"):assert torch.equal(models[name](check)[0],check["u"])
  # Verify cached inference against the unchanged true forward; no optimizer or train mode.
  ids=torch.as_tensor(candidates[:32],device=a.device)
  feed=dict(history_items=torch.as_tensor(dh[:32],device=a.device),lengths=torch.as_tensor(dl[:32],device=a.device),item_id=ids)
  original=m(feed,True);cacheuser=dev["u"][:32]
  assert torch.allclose(original["interest_vectors"],dev["v"][:32],rtol=1e-5,atol=1e-5)
  assert torch.allclose(original["user_vector"],cacheuser,rtol=1e-5,atol=1e-5)
  cachepred=(cacheuser[:,None]*table[ids]).sum(-1)
  assert torch.allclose(original["prediction"],cachepred,rtol=1e-4,atol=2e-5)
  unit=dict(initial_conditional_and_MLP_exact_user_identity=True,cache_vs_forward_max_abs=float((original["prediction"]-cachepred).abs().max()),
   backbone_frozen=True,history_only_features=True,Train_and_Dev_prefix_boundary_checked=True,Test_suffix_removed=True)
 print(json.dumps(dict(stage="cached",dataset=ds,Train_rows=len(selected),Dev_rows=len(dv),parameters=params,seconds=time.perf_counter()-start)),flush=True)
 baseline=None
 @torch.no_grad()
 def evaluate(names,matched=False):
  result={name:[] for name in names};weightstats={name:[] for name in names}
  for off in range(0,len(dv),256):
   sl=slice(off,off+256);d=sliced(dev,sl);ids=torch.as_tensor(candidates[sl],device=a.device)
   e=table[ids];base=(d["u"][:,None]*e).sum(-1).cpu().numpy().astype(np.float64)
   p=pop[ids].cpu().numpy().astype(np.float64);p=strength*(p-p.mean(1,keepdims=True))
   for name in names:
    if name=="baseline":u=d["u"];w=d["w"]
    else:u,w=models[name](d)
    score=(u[:,None]*e).sum(-1).cpu().numpy().astype(np.float64)
    if matched and name!="baseline":
     bc=base-base.mean(1,keepdims=True);sc=score-score.mean(1,keepdims=True)
     score=sc*np.sqrt((bc*bc).mean(1,keepdims=True))/np.maximum(np.sqrt((sc*sc).mean(1,keepdims=True)),1e-12)
    result[name].append(ranks(score+p));weightstats[name].append(w.cpu().numpy())
  return {n:np.concatenate(v) for n,v in result.items()},{n:np.concatenate(v).mean(0).tolist() for n,v in weightstats.items()}
 initial,wmeans=evaluate(["baseline"]+list(NAMES));baseline=initial["baseline"]
 bm=metrics(baseline);assert all(abs(bm[k]-lock["dev_metrics"][k])<1e-10 for k in METRICS),dict(actual=bm,expected=lock["dev_metrics"])
 for name in ("conditional","user_mlp"):assert np.array_equal(initial[name],baseline)
 best={name:dict(epoch=0,metrics=metrics(initial[name]),ranks=initial[name].copy(),state=copy.deepcopy(models[name].state_dict())) for name in NAMES}
 curves=[dict(epoch=0,scheme=name,**metrics(initial[name]),train_loss=None) for name in initial]
 opts={n:torch.optim.Adam(h.parameters(),lr=.01 if n=="global" else .001,weight_decay=1e-6) for n,h in models.items()}
 rng=np.random.default_rng(20261022);gradchecks={n:[] for n in NAMES};orders=[];losses_epoch=[]
 for epoch in range(1,5):
  epstart=time.perf_counter();order=rng.permutation(len(selected));orders.append(datahash(order));losssum={n:0. for n in NAMES}
  batches=0
  for off in range(0,len(order),1024):
   ix=torch.as_tensor(order[off:off+1024],device=a.device);d=sliced(train,ix);ii=targets[ix]
   allowed,unused,unused2=candidate_mask(m.train_membership,users[ix],ii);posvec=table[ii]
   p=strength*pop[ii][None,:]
   for name,head in models.items():
    opts[name].zero_grad(set_to_none=True);u,w=head(d)
    score=u@posvec.T+p
    loss,valid=masked_softmax_loss(score,allowed,1.)
    assert torch.isfinite(loss);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    if epoch==1 and batches<2:
     gradchecks[name].append(dict(step=batches+1,gradient_L2=float(torch.sqrt(sum((p.grad*p.grad).sum() for p in head.parameters()))),
      first_layer_L2=None if name=="global" else float(head.net[0].weight.grad.norm()),
      last_layer_L2=None if name=="global" else float(head.net[-1].weight.grad.norm())))
    opts[name].step();losssum[name]+=float(loss.detach())*len(ii)
   batches+=1
  assert all(p.grad is None for p in m.parameters()) and not any(v.training for v in m.modules())
  epochloss={n:v/len(selected) for n,v in losssum.items()};losses_epoch.append(epochloss)
  if epoch in CONFIG["Dev_epochs"]:
   rr,ww=evaluate(list(NAMES))
   for name in NAMES:
    mm=metrics(rr[name]);curves.append(dict(epoch=epoch,scheme=name,**mm,train_loss=epochloss[name]))
    if mm["NDCG@5"]>best[name]["metrics"]["NDCG@5"]:
     best[name]=dict(epoch=epoch,metrics=mm,ranks=rr[name].copy(),state=copy.deepcopy(models[name].state_dict()))
   print(json.dumps(dict(stage="epoch",dataset=ds,epoch=epoch,train_loss=epochloss,Dev_N5={n:metrics(rr[n])["NDCG@5"] for n in NAMES},seconds=time.perf_counter()-epstart)),flush=True)
  else:print(json.dumps(dict(stage="epoch",dataset=ds,epoch=epoch,train_loss=epochloss,seconds=time.perf_counter()-epstart)),flush=True)
 for name in ("conditional","user_mlp"):assert gradchecks[name][0]["last_layer_L2"]>0 and gradchecks[name][1]["first_layer_L2"]>0
 for name in NAMES:
  models[name].load_state_dict(best[name]["state"])
  with (out/("best_"+name+".pt")).open("xb") as f:torch.save(dict(readout=best[name]["state"],epoch=best[name]["epoch"],base_sha=inputs["checkpoint"],config=CONFIG),f)
 rr,ww=evaluate(list(NAMES),matched=True)
 final=[dict(scheme="baseline",selected_epoch=None,**bm,**{"delta_"+k:0. for k in METRICS})]
 for name in NAMES:
  mm=best[name]["metrics"]
  final.append(dict(scheme=name,selected_epoch=best[name]["epoch"],**mm,**{"delta_"+k:mm[k]-bm[k] for k in METRICS}))
 rms=[dict(scheme=n,**metrics(rr[n]),delta_NDCG5=metrics(rr[n])["NDCG@5"]-bm["NDCG@5"]) for n in NAMES]
 pairs=dict(conditional_vs_base=boot(best["conditional"]["ranks"],baseline),
  conditional_vs_global=boot(best["conditional"]["ranks"],best["global"]["ranks"]),
  conditional_vs_MLP=boot(best["conditional"]["ranks"],best["user_mlp"]["ranks"]))
 assert statehash(m)==base_hash
 assert all(sha(p)==inputs[k] for k,p in paths.items()) and all(sha(ROOT/n)==h for n,h in sources.items())
 result=dict(dataset=ds,config=CONFIG,Train_eligible_rows=len(eligible),Train_selected_rows=len(selected),
  Train_subset_indices_sha256=datahash(selected),Train_permutation_sha256=orders,Dev_candidates_sha256=datahash(candidates),
  input_sha256=inputs,source_sha256=sources,script_sha256=sha(__file__),parameters=params,baseline=bm,
  selected_metrics=final,scale_matched_diagnostic=rms,paired_Dev_N5=pairs,weight_means_selected=ww,
  unit_checks=unit,gradient_checks=gradchecks,epoch_train_losses=losses_epoch,curves=curves,
  model_state_before_and_after_sha256=base_hash,seconds=time.perf_counter()-start,peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20,
  Test_evaluations=0,backbone_updates=0,readout_steps_per_arm=4*int(np.ceil(len(selected)/1024)))
 write(out/"result.json",result);csvwrite(out/"dev_metrics.csv",final);csvwrite(out/"curves.csv",curves);csvwrite(out/"scale_matched.csv",rms)
 with (out/"dev_user_ranks.csv").open("x",newline="") as f:
  wr=csv.writer(f);wr.writerow(["user_id","baseline"]+list(NAMES));wr.writerows(zip(dv.user_id,baseline,*[best[n]["ranks"] for n in NAMES]))
 print("COMPLETE "+json.dumps({k:result[k] for k in ("dataset","selected_metrics","scale_matched_diagnostic","paired_Dev_N5","seconds","peak_GPU_MiB")}),flush=True)
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--prepare",action="store_true");parser.add_argument("--dataset",choices=["beauty","ml-1m"])
 args=parser.parse_args()
 if args.prepare:prepare()
 else:assert args.dataset;run(args.dataset)

