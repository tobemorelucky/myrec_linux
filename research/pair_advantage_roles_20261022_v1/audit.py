"""One <=170s audit containing both datasets: math, real gradients, 16 matched steps and Dev."""
import os
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("MKL_NUM_THREADS","4")
import sys,json,time,pickle,hashlib,signal,copy,math,csv,subprocess,importlib.util
from pathlib import Path
ROOT=Path("/home/yyx/hzgcode/pom2.0").resolve();OUT=Path(__file__).resolve().parent
assert OUT.is_relative_to(ROOT);sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OUT))
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from role_objective import PairRoleModel,pair_role_loss
REUSE=ROOT/"research/frozen_readout_softmax_20261022_v1/probe.py"
assert hashlib.sha256(REUSE.read_bytes()).hexdigest()=="fae7c3155477448075e58d94212ad637bc420d26449b3137bbcc953336e4c81d"
sp=importlib.util.spec_from_file_location("readonly_helpers",REUSE)
reuse=importlib.util.module_from_spec(sp);sp.loader.exec_module(reuse);old=reuse.old
METRICS=reuse.METRICS
CONFIG=dict(seed=42,batch_size=1024,steps=16,role_weight=.1,role_alpha=.5,role_temperature=1.,
 learning_rate=.0001,optimizer="Adam",weight_decay=1e-6,init="same existing best-Dev checkpoint for all three arms; diagnostic warm-start, NOT a full-training result",
 arms=["SS_continue","uniform_pair","pair_roles"],Dev="full original 1001 duplicate slots; before and after once, both raw and fixed pop .30",
 training="original filtered in-batch Softmax + original relation .01, same paired dropout/relation RNG, pair aux same legal negatives",
 Test_allowed=False,full_training_allowed=False,global_seconds_limit=170)
EXTRA=["models/sequential/LLMMIRecASPCFCIGR.py","models/sequential/LLMMIRecCAISD.py","models/sequential/LLMMIRecASPCFMultiPositive.py"]
def sha(p):return old.sha(p)
def write(p,x):
 p=Path(p);assert p.resolve().is_relative_to(OUT)
 with p.open("x") as f:json.dump(x,f,indent=2,allow_nan=False)
def statehash(m):return reuse.statehash(m)
def snapshot():return torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()
def restore(st):torch.set_rng_state(st[0]);torch.cuda.set_rng_state(st[1])
def math_tests():
 results={}
 for K in (1,2,4,7):
  B,D=5,8;v=torch.randn(B,K,D,dtype=torch.float64,requires_grad=True);e=torch.randn(B,D,dtype=torch.float64,requires_grad=True)
  allowed=torch.ones(B,B,dtype=torch.bool)
  loss,detail=pair_role_loss(v,e,allowed,.5,1.)
  ell=detail["per_head_loss"].requires_grad_()
  # Verify the analytic derivative d objective/d per-head loss equals r.
  f=.5*ell.mean(1)-.5*(torch.logsumexp(-ell,1)-math.log(K))
  grad=torch.autograd.grad(f.sum(),ell)[0]
  assert torch.allclose(grad,detail["responsibility"],atol=1e-12,rtol=1e-12)
  assert torch.allclose(grad.sum(1),torch.ones(B,B,dtype=torch.float64))
  assert grad.min()>=.5/K-1e-12
  if K==1:
   control,_=pair_role_loss(v,e,allowed,0.,1.)
   assert torch.allclose(loss,control,atol=1e-12,rtol=1e-12)
  none=torch.eye(B,dtype=torch.bool);z,_=pair_role_loss(v,e,none,.5,1.)
  assert float(z)==0 and all(torch.count_nonzero(t)==0 for t in torch.autograd.grad(z,(v,e)))
  assert torch.autograd.gradcheck(lambda x,y:pair_role_loss(x,y,allowed,.5,1.)[0],(v,e),eps=1e-6,atol=1e-5)
  results[str(K)]=dict(finite=True,gradcheck=True,analytic_responsibility=True,all_invalid_zero=True)
 # Exact shared-interest symmetry is NOT broken automatically.
 shared=torch.randn(5,1,8,dtype=torch.float64).expand(-1,4,-1).clone().requires_grad_()
 e=torch.randn(5,8,dtype=torch.float64);loss,detail=pair_role_loss(shared,e,torch.ones(5,5,dtype=torch.bool),.5,1.)
 assert torch.allclose(detail["responsibility"],torch.full_like(detail["responsibility"],.25),atol=1e-12)
 results["symmetric_interests"]="uniform responsibilities; cannot guarantee symmetry breaking"
 # Given a negative, the head with lower pair loss gets greater responsibility.
 v=torch.tensor([[[2.,0.],[0.,2.]],[[1.,0.],[0.,1.]]],dtype=torch.float64)
 e=torch.tensor([[1.,0.],[0.,1.]],dtype=torch.float64)
 loss,d=pair_role_loss(v,e,torch.ones(2,2,dtype=torch.bool),.5,1.)
 assert d["responsibility"][0,0,1]>d["responsibility"][0,1,1]
 return results
def prepare():
 tests=math_tests()
 write(OUT/"PREREG.json",dict(config=CONFIG,math_tests=tests,head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
  initial_status=subprocess.check_output(["git","status","--short","--branch"],cwd=ROOT,text=True),
  module_sha256=sha(OUT/"role_objective.py"),runner_sha256=sha(__file__),extra_source_sha256={p:sha(ROOT/p) for p in EXTRA}))
 print("MATH_PASS "+json.dumps(tests),flush=True)
def tofeed(hist,lengths,user,target,negative,device):
 return dict(history_items=torch.as_tensor(hist,device=device),lengths=torch.as_tensor(lengths,device=device),
  user_id=torch.as_tensor(user,device=device),item_id=torch.as_tensor(np.column_stack([target,negative]),device=device))
def negative_one(m,user,gen):
 ids=torch.randint(1,m.item_num,(len(user),),generator=gen,device=m.device)
 while True:
  bad=(((m.train_membership[user,ids//64]>>(ids%64))&1)!=0)
  if not bool(bad.any()):break
  ids[bad]=torch.randint(1,m.item_num,(int(bad.sum()),),generator=gen,device=m.device)
 return ids
def role_gradient_stats(model,feed):
 st=snapshot();mode=model.training;model.eval()
 out=model(feed,True)
 from models.sequential.LLMMIRecASPCFSampledSoftmax import candidate_mask
 mask,_,_=candidate_mask(model.train_membership,feed["user_id"],feed["item_id"][:,0])
 v=out["interest_vectors"];e=out["candidate_vectors"][:,0]
 adv,d=pair_role_loss(v,e,mask,.5,1.);uni,_=pair_role_loss(v,e,mask,0.,1.)
 ga=torch.autograd.grad(adv,v,retain_graph=True)[0];gu=torch.autograd.grad(uni,v,retain_graph=True)[0]
 gq=torch.autograd.grad(adv,model.extractor.query)[0]
 legal=d["legal"];r=d["responsibility"].permute(0,2,1)[legal]
 if len(r)==0:raise ValueError("No negatives")
 # Row-wise cosine between head gradients, averaged only nonzero head pairs.
 vv=ga.permute(1,0,2).reshape(model.K,-1)
 norm=F.normalize(vv,dim=1);cm=norm@norm.T
 off=~torch.eye(model.K,dtype=torch.bool,device=v.device)
 result=dict(aux=float(adv.detach()),uniform_aux=float(uni.detach()),query_aux_gradient_L2=float(gq.norm()),
  r_abs_difference_from_uniform=float((r-1/model.K).abs().mean()),r_max_mean=float(r.max(1).values.mean()),
  r_entropy_mean=float(-(r*r.clamp_min(1e-12).log()).sum(1).mean()),
  r_floor=float(r.min()),mean_responsibility_by_head=r.mean(0).tolist(),
  V_aux_gradient_relative_change_vs_uniform=float((ga-gu).norm()/gu.norm().clamp_min(1e-30)),
  head_V_aux_gradient_cosine_mean=float(cm[off].mean()) if model.K>1 else None)
 model.train(mode);restore(st);return result
def equivalence_check(model,feed):
 role=model.role_weight;model.role_weight=0;model.train()
 st=snapshot();o=model(feed,True);l=model.loss(o)
 gv=torch.autograd.grad(l,model.extractor.query)[0]
 # Match the original SS class directly, including RNG and full parameter initialization.
 reference=old.Model(model._audit_args,model._audit_corpus).to(model.device)
 reference.load_state_dict(model.state_dict(),strict=True);reference.train();restore(st)
 oo=reference(feed,True);ll=reference.loss(oo);gg=torch.autograd.grad(ll,reference.extractor.query)[0]
 assert torch.equal(o["prediction"],oo["prediction"]) and torch.equal(l,ll) and torch.equal(gv,gg)
 model.role_weight=role;restore(st);del reference
 return dict(weight_zero_prediction_loss_query_gradient_exact=True)
@torch.no_grad()
def evaluate(m,c,frame,dh,dl,candidates,pop,strength):
 m.eval();table=torch.cat([m.item_encoder(torch.arange(i,min(i+2048,c.n_items),device=m.device)) for i in range(0,c.n_items,2048)])
 feat=reuse.features(m,table,dh,dl);rr=[];rc=[]
 for off in range(0,len(frame),256):
  sl=slice(off,off+256);ii=torch.as_tensor(candidates[sl],device=m.device)
  raw=(feat["u"][sl,None]*table[ii]).sum(-1).cpu().numpy().astype(np.float64)
  p=pop[ii].cpu().numpy().astype(np.float64)
  rr.append(reuse.ranks(raw));rc.append(reuse.ranks(raw+strength*(p-p.mean(1,keepdims=True))))
 return dict(raw=reuse.metrics(np.concatenate(rr)),pop=reuse.metrics(np.concatenate(rc)))
def dataset_run(ds,pre):
 begun=time.perf_counter();old.utils.init_seed(42)
 spec=old.SPEC[ds];prior=ROOT/"research/popularity_interest_audit_20261022_v1"/ds/"DEV_LOCK.json"
 lock=json.loads(prior.read_text());assert lock["lambda_normalized"]==.30
 paths=dict(checkpoint=spec["checkpoint"],embedding=spec["embedding"],train=ROOT/"data"/ds/"train.csv",
  dev=ROOT/"data"/ds/"dev.csv",cache=ROOT/"data"/ds/"SeqReader.pkl",prior_lock=prior)
 inputs={k:sha(p) for k,p in paths.items()};assert inputs["checkpoint"]==lock["checkpoint_sha256"] and inputs["embedding"]==lock["embedding_sha256"]
 sources={p:sha(ROOT/p) for p in old.SOURCES};assert sources==lock["source_sha256"]
 with paths["cache"].open("rb") as f:c=pickle.load(f)
 c.data_df={p:c.data_df[p] for p in ("train","dev")};tr=c.data_df["train"];dv=c.data_df["dev"];counts=tr.groupby("user_id").size()
 assert np.all(tr.position.to_numpy()<tr.user_id.map(counts).to_numpy()) and np.all(dv.position.to_numpy()==dv.user_id.map(counts).to_numpy())
 for u in c.user_his:c.user_his[u]=c.user_his[u][:int(counts.get(u,0))]
 eligible=np.flatnonzero(tr.position.to_numpy()>0);rng=np.random.default_rng(42);idx=rng.choice(eligible,16*1024,replace=False)
 tf=tr.iloc[idx];th,tl=reuse.construct_hist(c,tf);dh,dl=reuse.construct_hist(c,dv)
 cand=np.concatenate([dv.item_id.to_numpy(np.int64)[:,None],np.stack(dv.neg_items.to_numpy())],1);assert cand.shape==(len(dv),1001)
 a=old.arguments(ds,c);a.lr=.0001
 models={}
 for name,weight,alpha in (("SS_continue",0.,0.),("uniform_pair",.1,0.),("pair_roles",.1,.5)):
  models[name]=PairRoleModel(a,c,weight,alpha).to(a.device)
  models[name]._audit_args=a;models[name]._audit_corpus=c
 fresh=models["pair_roles"];device=a.device
 gen=torch.Generator(device=device).manual_seed(20261022)
 user=tf.user_id.to_numpy(np.int64);target=tf.item_id.to_numpy(np.int64)
 uu=torch.as_tensor(user[:64],device=device);nn=negative_one(fresh,uu,gen).cpu().numpy()
 probe=tofeed(th[:64],tl[:64],user[:64],target[:64],nn,device)
 cold=role_gradient_stats(fresh,probe)
 for m in models.values():m.load_state_dict(torch.load(spec["checkpoint"],map_location=device,weights_only=True),strict=True)
 samehash=statehash(models["SS_continue"]);assert all(statehash(m)==samehash for m in models.values())
 units=equivalence_check(models["pair_roles"],probe)
 warm=role_gradient_stats(models["pair_roles"],probe)
 assert math.isfinite(warm["query_aux_gradient_L2"]) and warm["query_aux_gradient_L2"]>0
 assert warm["V_aux_gradient_relative_change_vs_uniform"]>1e-6
 rawcount=np.bincount(pd.read_csv(paths["train"],sep="\t",usecols=["item_id"]).item_id.to_numpy(np.int64),minlength=c.n_items)
 pop=torch.as_tensor(np.log1p(rawcount),device=device,dtype=torch.float32);strength=lock["raw_lambda"]
 initial=evaluate(models["SS_continue"],c,dv,dh,dl,cand,pop,strength)
 assert all(abs(initial["pop"][k]-lock["dev_metrics"][k])<1e-10 for k in METRICS)
 opts={n:torch.optim.Adam(m.parameters(),lr=.0001,weight_decay=1e-6) for n,m in models.items()}
 step_log=[];norms={n:[] for n in models};fitstart=time.perf_counter();neggen=torch.Generator(device=device).manual_seed(20261022)
 for step in range(16):
  sl=slice(step*1024,(step+1)*1024);us=torch.as_tensor(user[sl],device=device)
  neg=negative_one(models["SS_continue"],us,neggen).cpu().numpy()
  feed=tofeed(th[sl],tl[sl],user[sl],target[sl],neg,device)
  before=snapshot();after=None;losses={}
  for name,m in models.items():
   m.train();restore(before);opts[name].zero_grad(set_to_none=True);o=m(feed,True);loss=m.loss(o)
   assert torch.isfinite(loss);loss.backward()
   assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
   qg=float(m.extractor.query.grad.norm());wg=float(m.extractor.Wv.weight.grad.norm())
   norms[name].append(dict(query=qg,Wv=wg))
   assert qg>0 and wg>0
   opts[name].step();losses[name]=float(loss.detach())
   current=snapshot()
   if after is None:after=current
   else:assert all(torch.equal(x,y) for x,y in zip(after,current))
  restore(after);step_log.append(dict(step=step+1,loss=losses))
 torch.cuda.synchronize();fitseconds=time.perf_counter()-fitstart
 final={n:evaluate(m,c,dv,dh,dl,cand,pop,strength) for n,m in models.items()}
 after_grad=role_gradient_stats(models["pair_roles"],probe)
 changed={n:float((m.extractor.query.detach()-torch.load(spec["checkpoint"],map_location=device,weights_only=True)["extractor.query"]).norm()) for n,m in models.items()}
 assert all(x>0 for x in changed.values())
 assert all(sha(p)==inputs[k] for k,p in paths.items()) and all(sha(ROOT/p)==h for p,h in {**sources,**pre["extra_source_sha256"]}.items())
 result=dict(dataset=ds,head=pre["head"],config=CONFIG,parameters=sum(p.numel() for p in models["pair_roles"].parameters()),
  extra_parameters=0,inputs_sha256=inputs,source_sha256=sources,initial_Dev=initial,final_Dev=final,
  cold_gradient_diagnostic=cold,warm_gradient_diagnostic=warm,after_short_gradient_diagnostic=after_grad,
  equivalence=units,short_Train_steps=step_log,step_gradient_norms=norms,query_parameter_change_L2=changed,
  common_init_state_sha256=samehash,Train_subset_sha256=reuse.datahash(idx),Dev_candidates_sha256=reuse.datahash(cand),
  fit_seconds_all_three_arms=fitseconds,total_seconds=time.perf_counter()-begun,
  peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20,Test_evaluations=0,checkpoints_written=0)
 write(OUT/(ds+".json"),result)
 rows=[]
 for stage,schemes in (("initial",{"existing":initial}),("after16",final)):
  for name,scores in schemes.items():
   for scoremode,mm in scores.items():rows.append(dict(stage=stage,scheme=name,scoring=scoremode,**mm))
 csvwrite(OUT/(ds+"_dev_metrics.csv"),rows)
 print("DATASET_COMPLETE "+json.dumps({k:result[k] for k in ("dataset","initial_Dev","final_Dev","cold_gradient_diagnostic","warm_gradient_diagnostic","fit_seconds_all_three_arms","total_seconds","peak_GPU_MiB")}),flush=True)
 del models,opts,c;torch.cuda.empty_cache()
 return result
def runall():
 signal.alarm(170);started=time.perf_counter();torch.set_num_threads(4)
 pre=json.loads((OUT/"PREREG.json").read_text())
 assert sha(__file__)==pre["runner_sha256"] and sha(OUT/"role_objective.py")==pre["module_sha256"]
 results=[dataset_run(ds,pre) for ds in ("beauty","ml-1m")]
 write(OUT/"COMPLETE.json",dict(total_seconds=time.perf_counter()-started,config=CONFIG,datasets=[r["dataset"] for r in results],
  Test_evaluations=0,full_training_started=False,module_sha256=sha(OUT/"role_objective.py"),runner_sha256=sha(__file__)))
 print("ALL_COMPLETE",flush=True)
if __name__=="__main__":
 import argparse
 p=argparse.ArgumentParser();p.add_argument("--prepare",action="store_true");args=p.parse_args()
 if args.prepare:prepare()
 else:runall()

