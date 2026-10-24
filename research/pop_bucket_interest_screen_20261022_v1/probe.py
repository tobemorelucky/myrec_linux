"""Train-uniform negative, frozen popularity-bucket score screen. No Test entry."""
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
REUSE=ROOT/"research/frozen_readout_softmax_20261022_v1/probe.py"
assert hashlib.sha256(REUSE.read_bytes()).hexdigest()=="fae7c3155477448075e58d94212ad637bc420d26449b3137bbcc953336e4c81d"
sp=importlib.util.spec_from_file_location("frozen_features_only",REUSE)
reuse=importlib.util.module_from_spec(sp);sp.loader.exec_module(reuse)
old=reuse.old
METRICS=reuse.METRICS
CONFIG=dict(seed=42,Train_subset_max=131072,epochs=4,Dev_epochs=[0,1,2,4],batch_size=1024,
 negative_count=64,negative_sampling="uniform replacement on catalog 1..N-1 conditional on excluding user's Train clicked; regenerate each epoch; duplicates retained",
 optimizer="Adam",lr=.01,weight_decay=1e-6,temperature=1.,loss="CE over positive+64 shared uniform negatives, same fixed prior in both fitted schemes",
 frequency_lambda=.30,buckets="Train count=0 plus five positive-count catalog quantiles; tied frequencies never split",
 C1="B0+a[bucket]*raw_ASPCF_score+b[bucket]",M1="B0+sum_k(w_k*theta[k,bucket]*(m_k-m_mean))",
 selection="per-arm best Dev NDCG@5 at 0/1/2/4, earliest tie; epoch0 included",
 go="M1 must exceed BOTH B0 and C1 on BOTH domains; tiny gains only exploratory; no automatic full train",
 no_Test=True,no_full_model_training=True,no_sweep=True,run_seconds_limit=170)
EXTRA_SOURCES=["models/sequential/LLMMIRecPPCIM.py","models/sequential/LLMMIRecASPCFInterestMetric.py","models/sequential/LLMMIRecASPCFCIGR.py"]
def sha(p):return old.sha(p)
def write(p,x):
 p=Path(p);assert p.resolve().is_relative_to(OUT)
 with p.open("x") as f:json.dump(x,f,indent=2,allow_nan=False)
def statehash(x):return reuse.statehash(x)
def csvwrite(p,rows):return reuse.csvwrite(p,rows)
def dhash(x):return reuse.datahash(x)
def metrics(x):return reuse.metrics(x)
def ranks(x):return reuse.ranks(x)
def phi_from_match(match,w):
 # Algebraically w*(m-sum(w*m)); reference subtraction ensures exact zero
 # whenever all interest predictions coincide, including K=1.
 r=match-match[:,:1,:]
 return w[:,:,None]*(r-(w[:,:,None]*r).sum(1,keepdim=True))
def score_features(d,e):
 raw=(d["u"][:,None,:]*e).sum(-1)
 match=torch.einsum("bkd,bcd->bkc",d["v"],e)
 return raw,phi_from_match(match,d["w"])
class C1(nn.Module):
 def __init__(self,G):
  super().__init__();self.affine=nn.Parameter(torch.zeros(2,G))
 def correction(self,raw,phi,b):return self.affine[0][b]*raw+self.affine[1][b]
class M1(nn.Module):
 def __init__(self,K,G):
  super().__init__();self.theta=nn.Parameter(torch.zeros(K,G))
 def correction(self,raw,phi,b):return (phi*self.theta[:,b].permute(1,0,2)).sum(1)
def units():
 torch.manual_seed(42);records=[]
 for K in (1,2,4,7):
  B,C,G=3,9,6
  match=torch.randn(B,K,C,dtype=torch.float64);w=torch.randn(B,K,dtype=torch.float64).softmax(-1)
  raw=(w[:,:,None]*match).sum(1);b=torch.randint(G,(B,C));phi=phi_from_match(match,w)
  heads=[C1(G).double(),M1(K,G).double()];base=raw+torch.randn_like(raw)
  for h in heads:assert torch.equal(base+h.correction(raw,phi,b),base)
  shared=match[:,:1,:].expand(-1,K,-1)
  assert torch.count_nonzero(phi_from_match(shared,w))==0
  if K==1:assert torch.count_nonzero(phi)==0
  h=heads[1]
  with torch.no_grad():h.theta.copy_(torch.randn_like(h.theta))
  actual=h.correction(raw,phi,b)
  direct=(w[:,:,None]*h.theta[:,b].permute(1,0,2)*(match-raw[:,None,:])).sum(1)
  assert torch.allclose(actual,direct,atol=1e-12,rtol=1e-12)
  assert torch.allclose(phi.sum(1),torch.zeros_like(raw),atol=1e-12,rtol=0)
  records.append(dict(K=K,zero_init_exact=True,equal_interest_predictions_exact_zero=True,K1_exact_zero=K==1,
   direct_formula_float64_max_error=float((actual-direct).abs().max())))
 return records
def prepare():
 rec=units()
 write(OUT/"PREREG.json",dict(config=CONFIG,unit_tests=rec,head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
  initial_git_status=subprocess.check_output(["git","status","--short","--branch"],cwd=ROOT,text=True),script_sha256=sha(__file__),
  old_score_sources_sha256={n:sha(ROOT/n) for n in EXTRA_SOURCES},reuse_feature_script_sha256=sha(REUSE),
  overlap="M1 is InterestMetric I-C scalar covariance conditioned on Train popularity bucket; within a bucket it is a single corrected user vector. Not a new fundamental scoring operator."))
 print("PREPARED "+json.dumps(rec),flush=True)
def sample_negatives(m,users,count,generator):
 ids=torch.randint(1,m.item_num,(len(users),count),generator=generator,device=m.device)
 retries=0
 while True:
  bad=(((m.train_membership[users[:,None],ids//64]>>(ids%64))&1)!=0)
  if not bool(bad.any()):break
  ids[bad]=torch.randint(1,m.item_num,(int(bad.sum()),),generator=generator,device=m.device);retries+=1
  assert retries<100
 return ids,retries
def run(ds):
 signal.alarm(170);start=time.perf_counter();torch.set_num_threads(4);old.utils.init_seed(42)
 pre=json.loads((OUT/"PREREG.json").read_text());assert pre["script_sha256"]==sha(__file__)
 out=OUT/ds;out.mkdir(exist_ok=False)
 prior_root=ROOT/"research/popularity_interest_audit_20261022_v1"
 lock=json.loads((prior_root/ds/"DEV_LOCK.json").read_text());assert lock["lambda_normalized"]==.30
 s=old.SPEC[ds]
 paths=dict(checkpoint=s["checkpoint"],embedding=s["embedding"],train=ROOT/"data"/ds/"train.csv",
  dev=ROOT/"data"/ds/"dev.csv",cache=ROOT/"data"/ds/"SeqReader.pkl",prior_lock=prior_root/ds/"DEV_LOCK.json")
 inputs={k:sha(p) for k,p in paths.items()}
 assert inputs["checkpoint"]==lock["checkpoint_sha256"] and inputs["embedding"]==lock["embedding_sha256"]
 sources={n:sha(ROOT/n) for n in old.SOURCES};assert sources==lock["source_sha256"]
 extra={n:sha(ROOT/n) for n in EXTRA_SOURCES};assert extra==pre["old_score_sources_sha256"]
 with paths["cache"].open("rb") as f:c=pickle.load(f)
 c.data_df={p:c.data_df[p] for p in ("train","dev")};tr=c.data_df["train"];dv=c.data_df["dev"]
 counts=tr.groupby("user_id").size()
 assert np.all(tr.position.to_numpy()<tr.user_id.map(counts).to_numpy())
 assert np.all(dv.position.to_numpy()==dv.user_id.map(counts).to_numpy())
 for u in c.user_his:c.user_his[u]=c.user_his[u][:int(counts.get(u,0))]
 a=old.arguments(ds,c);m=old.Model(a,c).to(a.device)
 m.load_state_dict(torch.load(s["checkpoint"],map_location=a.device,weights_only=True),strict=True)
 m.requires_grad_(False);m.eval();basehash=statehash(m)
 with torch.no_grad():table=torch.cat([m.item_encoder(torch.arange(i,min(i+2048,c.n_items),device=a.device)) for i in range(0,c.n_items,2048)])
 rawcount=np.bincount(pd.read_csv(paths["train"],sep="\t",usecols=["item_id"]).item_id.to_numpy(np.int64),minlength=c.n_items)
 thresholds=np.quantile(rawcount[rawcount>0],[.2,.4,.6,.8])
 bins=np.where(rawcount==0,0,np.searchsorted(thresholds,rawcount,side="left")+1)
 G=6;bt=torch.as_tensor(bins,device=a.device)
 pop=torch.as_tensor(np.log1p(rawcount),dtype=torch.float32,device=a.device)
 strength=lock["raw_lambda"];assert strength==.30*lock["scale_factor"]
 eligible=np.flatnonzero(tr.position.to_numpy()>0)
 chosen=np.sort(np.random.default_rng(42).choice(eligible,min(CONFIG["Train_subset_max"],len(eligible)),replace=False))
 tf=tr.iloc[chosen];th,tl=reuse.construct_hist(c,tf);dh,dl=reuse.construct_hist(c,dv)
 train=reuse.features(m,table,th,tl);dev=reuse.features(m,table,dh,dl)
 users=torch.as_tensor(tf.user_id.to_numpy(np.int64),device=a.device)
 targets=torch.as_tensor(tf.item_id.to_numpy(np.int64),device=a.device)
 candidates=np.concatenate([dv.item_id.to_numpy(np.int64)[:,None],np.stack(dv.neg_items.to_numpy())],1)
 assert candidates.shape==(len(dv),1001)
 models=dict(C1=C1(G).to(a.device),M1=M1(m.K,G).to(a.device))
 params={n:sum(p.numel() for p in h.parameters()) for n,h in models.items()}
 assert params=={"C1":12,"M1":m.K*6}
 print(json.dumps(dict(stage="cached",dataset=ds,K=m.K,Train_rows=len(chosen),Dev_rows=len(dv),parameters=params,seconds=time.perf_counter()-start)),flush=True)
 @torch.no_grad()
 def evaluate(names):
  rr={n:[] for n in names}
  for off in range(0,len(dv),256):
   sl=slice(off,off+256);d=reuse.sliced(dev,sl);ii=torch.as_tensor(candidates[sl],device=a.device)
   raw,phi=score_features(d,table[ii]);b=bt[ii]
   raw_np=raw.cpu().numpy().astype(np.float64);p=pop[ii].cpu().numpy().astype(np.float64)
   base=raw_np+strength*(p-p.mean(1,keepdims=True))
   for n in names:
    pred=base if n=="B0" else base+models[n].correction(raw,phi,b).cpu().numpy().astype(np.float64)
    rr[n].append(ranks(pred))
  return {n:np.concatenate(v) for n,v in rr.items()}
 initial=evaluate(["B0","C1","M1"]);baseline=initial["B0"];bm=metrics(baseline)
 assert all(abs(bm[k]-lock["dev_metrics"][k])<1e-10 for k in METRICS)
 assert all(np.array_equal(initial[n],baseline) for n in models)
 best={n:dict(epoch=0,state=copy.deepcopy(h.state_dict()),ranks=baseline.copy(),metrics=bm.copy()) for n,h in models.items()}
 curves=[dict(epoch=0,scheme=n,**metrics(r),train_loss=None) for n,r in initial.items()]
 opts={n:torch.optim.Adam(h.parameters(),lr=.01,weight_decay=1e-6) for n,h in models.items()}
 permgen=np.random.default_rng(20261022);neggen=torch.Generator(device=a.device).manual_seed(20261022)
 gradcheck={n:[] for n in models};fit_seconds=0.;train_epochs=[];orderhashes=[]
 neg_hash=hashlib.sha256()
 for epoch in range(1,5):
  torch.cuda.synchronize();epstart=time.perf_counter();order=permgen.permutation(len(chosen));orderhashes.append(dhash(order))
  losses={n:0. for n in models};retries=0;steps=0;bucketmass=np.zeros(G,np.int64)
  for off in range(0,len(order),1024):
   ix=torch.as_tensor(order[off:off+1024],device=a.device)
   with torch.no_grad():
    negative,retry=sample_negatives(m,users[ix],64,neggen);retries+=retry
    ii=torch.cat([targets[ix,None],negative],1)
    # Both schemes consume exactly these same candidates, labels and frozen score features.
    neg_hash.update(ii.cpu().numpy().tobytes())
    raw,phi=score_features(reuse.sliced(train,ix),table[ii]);b=bt[ii]
    p=pop[ii];base=raw+strength*(p-p.mean(1,keepdim=True))
    labels=torch.zeros(len(ix),dtype=torch.long,device=a.device)
   for n,h in models.items():
    opts[n].zero_grad(set_to_none=True)
    score=base+h.correction(raw,phi,b);loss=F.cross_entropy(score,labels)
    assert torch.isfinite(loss);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in h.parameters())
    if epoch==1 and steps<2:
     gradcheck[n].append(float(torch.sqrt(sum((p.grad*p.grad).sum() for p in h.parameters()))))
    opts[n].step();losses[n]+=float(loss.detach())*len(ix)
   steps+=1
  torch.cuda.synchronize();elapsed=time.perf_counter()-epstart;fit_seconds+=elapsed
  assert all(p.grad is None for p in m.parameters()) and not any(v.training for v in m.modules())
  lossmean={n:v/len(chosen) for n,v in losses.items()}
  train_epochs.append(dict(epoch=epoch,loss=lossmean,steps=steps,shared_fit_seconds=elapsed,negative_rejection_rounds=retries))
  record=dict(stage="epoch",dataset=ds,epoch=epoch,train_loss=lossmean,fit_seconds=elapsed)
  if epoch in CONFIG["Dev_epochs"]:
   ev=evaluate(list(models))
   for n,r in ev.items():
    mm=metrics(r);curves.append(dict(epoch=epoch,scheme=n,**mm,train_loss=lossmean[n]))
    if mm["NDCG@5"]>best[n]["metrics"]["NDCG@5"]:best[n]=dict(epoch=epoch,state=copy.deepcopy(models[n].state_dict()),ranks=r.copy(),metrics=mm)
   record["Dev_N5"]={n:metrics(r)["NDCG@5"] for n,r in ev.items()}
  print(json.dumps(record),flush=True)
 assert all(g[0]>0 and g[1]>0 for g in gradcheck.values())
 trained_best={}
 for n in models:
  trained_best[n]=max((r for r in curves if r["scheme"]==n and r["epoch"]>0),key=lambda r:(r["NDCG@5"],-r["epoch"]))
 final=[dict(scheme="B0",selected_epoch=0,**bm,**{"delta_"+k:0. for k in METRICS})]
 for n,h in models.items():
  lastnorm=float(torch.sqrt(sum((p*p).sum() for p in h.parameters())))
  h.load_state_dict(best[n]["state"])
  with (out/("best_"+n+".pt")).open("xb") as f:torch.save(dict(score_parameters=best[n]["state"],epoch=best[n]["epoch"],config=CONFIG,base_sha256=inputs["checkpoint"]),f)
  final.append(dict(scheme=n,selected_epoch=best[n]["epoch"],**best[n]["metrics"],**{"delta_"+k:best[n]["metrics"][k]-bm[k] for k in METRICS}))
  best[n]["last_epoch_parameter_norm"]=lastnorm
 pairs=dict(M1_vs_B0=reuse.boot(best["M1"]["ranks"],baseline),M1_vs_C1=reuse.boot(best["M1"]["ranks"],best["C1"]["ranks"]))
 assert statehash(m)==basehash and all(sha(p)==inputs[k] for k,p in paths.items())
 assert all(sha(ROOT/n)==h for n,h in {**sources,**extra}.items()) and sha(REUSE)==pre["reuse_feature_script_sha256"]
 result=dict(dataset=ds,head=pre["head"],config=CONFIG,K=m.K,G=G,bucket_thresholds=thresholds.tolist(),
  Train_eligible=len(eligible),Train_selected=len(chosen),Train_subset_sha256=dhash(chosen),Train_order_sha256=orderhashes,
  shared_Train_candidate_stream_sha256=neg_hash.hexdigest(),Dev_candidates_sha256=dhash(candidates),
  input_sha256=inputs,source_sha256=sources,old_score_source_sha256=extra,script_sha256=sha(__file__),
  parameters=params,M1_common_head_shift_is_null=True,M1_effective_contrast_parameters=(m.K-1)*G,
  selected_metrics=final,best_trained_metrics=trained_best,curves=curves,paired_Dev_N5=pairs,gradient_checks=gradcheck,
  last_epoch_parameter_norm={n:best[n]["last_epoch_parameter_norm"] for n in models},train_epochs=train_epochs,
  shared_fit_seconds=fit_seconds,total_seconds=time.perf_counter()-start,peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20,
  backbone_state_sha256_before_after=basehash,backbone_updates=0,Test_evaluations=0,unit_tests=pre["unit_tests"],
  selected_state={n:{k:v.detach().cpu().tolist() for k,v in h.state_dict().items()} for n,h in models.items()})
 write(out/"result.json",result);csvwrite(out/"dev_metrics.csv",final);csvwrite(out/"curves.csv",curves)
 trained_rows=[dict(scheme="B0",epoch=0,**bm)]+[dict(scheme=n,epoch=trained_best[n]["epoch"],**{k:trained_best[n][k] for k in METRICS}) for n in models]
 csvwrite(out/"best_trained_metrics.csv",trained_rows)
 with (out/"dev_user_ranks.csv").open("x",newline="") as f:
  wr=csv.writer(f);wr.writerow(["user_id","B0","C1","M1"]);wr.writerows(zip(dv.user_id,baseline,best["C1"]["ranks"],best["M1"]["ranks"]))
 print("COMPLETE "+json.dumps({k:result[k] for k in ("dataset","selected_metrics","best_trained_metrics","paired_Dev_N5","shared_fit_seconds","total_seconds","peak_GPU_MiB")}),flush=True)
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--prepare",action="store_true");ap.add_argument("--dataset",choices=["beauty","ml-1m"]);args=ap.parse_args()
 if args.prepare:prepare()
 else:assert args.dataset;run(args.dataset)

