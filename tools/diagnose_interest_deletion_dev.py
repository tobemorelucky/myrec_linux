"""One bounded frozen Dev deletion pass per checkpoint; no Train/Test or selector."""
import os,sys,json,time,signal,hashlib,pickle,argparse,csv,itertools,subprocess
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]="1"
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from helpers.BaseRunner import BaseRunner
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF as Base
from models.sequential.LLMMIRecASPCFRefineBPR import LLMMIRecASPCFRefineBPR as User
from tools.diagnose_aspcf_frozen_dev_scoring import SPEC
from utils import utils
SOURCES=["models/sequential/LLMMIRecASPCF.py","models/sequential/llmmi_components.py",
 "models/sequential/LLMMIRecASPCFRefineBPR.py","models/BaseModel.py","helpers/BaseRunner.py",
 "tools/diagnose_interest_deletion_dev.py"]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4194304),b""):h.update(b)
 return h.hexdigest()
def stat(x):
 a=np.asarray(x,dtype=float).reshape(-1)
 return dict(mean=float(a.mean()),median=float(np.median(a)),p90=float(np.quantile(a,.9)),max=float(a.max()))
def corr(x,y):
 x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
 if np.std(x)<1e-15 or np.std(y)<1e-15:return None
 return float(np.corrcoef(x,y)[0,1])
def bootstrap(delta):
 # Exact grouped implementation of paired empirical-user bootstrap means.
 values,counts=np.unique(np.asarray(delta,dtype=float),return_counts=True)
 g=np.random.default_rng(20261019)
 means=g.multinomial(len(delta),counts/counts.sum(),size=2000)@values/len(delta)
 return np.quantile(means,[.025,.975]).tolist()
def rank(pred):
 order=(-pred).argsort(axis=1);r=np.argwhere(order==0)[:,1]+1
 return order,r,(r<=5)/np.log2(r+1)
def configure(ds,K,kind):
 if kind=="A":
  r=ROOT/"new_log/refine_bpr_round1/20261018_dualview_bpr_k4_seed42_round1"/ds/"B1"
  j=json.loads((r/"config.json").read_text());s=json.loads((r/"summary.json").read_text())
  args=argparse.Namespace(**j["effective_args"]);path=Path(args.model_path)
  expected=s["checkpoint_sha256"];reference=s["dev"];cls=User
 else:
  stem,expected,_=SPEC[(ds,K)]
  path=ROOT/"new_model"/stem/ds/"seed42/LLMMIRecASPCF_seed42.pt"
  old={}
  for line in (ROOT/"new_log"/stem/ds/"seed42/LLMMIRecASPCF_seed42.log").read_text().splitlines():
   cols=line.split("|")
   if len(cols)>1:old[cols[0].strip()]=cols[1].strip()
  p=BaseRunner.parse_runner_args(Base.parse_model_args(argparse.ArgumentParser()))
  cli=[x for a in p._actions if a.dest in old and a.dest!="help" and "..." not in old[a.dest]
       for x in [a.option_strings[0],old[a.dest]]]
  args=p.parse_args(cli);cls=Base
  refpath=ROOT/"new_log/aspcf_frozen_dev_scoring/20261018_s0_s1_s2_seed42"/(ds+"_K"+str(K)+".json")
  reference=json.loads(refpath.read_text())["metrics"]["S0"]
 assert path.is_file() and sha(path)==expected
 args.device=torch.device("cuda");args.buffer=0
 args.llm_emb_path=str(ROOT/"data"/ds/"handled/llm_table_pca1536.pkl")
 assert args.K==K and args.history_max==20 and args.num_neg==1 and args.lambda_relation==.01 and args.test_all==0
 return cls,args,path,expected,reference
def run(ds,K,kind,outdir):
 start=time.perf_counter();cls,args,path,expected,ref=configure(ds,K,kind)
 with (ROOT/"data"/ds/"SeqReader.pkl").open("rb") as f:corpus=pickle.load(f)
 corpus.data_df={"dev":corpus.data_df["dev"]}
 assert corpus.data_df["dev"].user_id.is_unique
 model=cls(args,corpus).to(args.device)
 model.load_state_dict(torch.load(path,map_location=args.device,weights_only=True),strict=True);model.eval()
 data=cls.Dataset(model,corpus,"dev");assert all(len(v)==1000 for v in data.data["neg_items"])
 loader=DataLoader(data,batch_size=args.eval_batch_size,shuffle=False,num_workers=0,collate_fn=data.collate_batch)
 preds=[[] for _ in range(K+1)];U=[];L=[];valids=[[] for _ in range(K)]
 features=[dict(att=[],score_div=[],weight=[],score_change=[],slot_top_change=[],item_top_change=[]) for _ in range(K)]
 branches=[];maxerr=0.;inputs={k:hashlib.sha256() for k in ["users","history","candidates"]}
 torch.cuda.reset_peak_memory_stats()
 with torch.inference_mode():
  for batch in loader:
   for name,keys in [("users",["user_id"]),("history",["history_items","lengths"]),("candidates",["item_id"])]:
    for k in keys:
     x=batch[k].numpy();inputs[name].update(k.encode());inputs[name].update(str(x.shape).encode());inputs[name].update(x.tobytes())
   users=batch["user_id"].numpy();lengths=batch["lengths"].numpy();candidates=batch["item_id"].numpy()
   U.append(users);L.append(lengths);b=utils.batch_to_gpu(batch,args.device);o=model(b,True)
   v,w,e,A=o["interest_vectors"],o["interest_weights"],o["candidate_vectors"],o["attention_maps"]
   u=(v*w[:,:,None]).sum(1)
   assert torch.equal(u,o["base_user_vector"] if kind=="A" else o["user_vector"])
   p0=o["prediction"].cpu().numpy();preds[0].append(p0);ord0=(-p0).argsort(axis=1)[:,:5]
   item0=np.take_along_axis(candidates,ord0,1)
   z=torch.bmm(v,e.transpose(1,2));zc=z-z.mean(-1,keepdim=True)
   centeredmean=zc.mean(1,keepdim=True)
   # Fixed-attention exact decomposition of SAME original Wv path.
   h=b["history_items"];valid=(h>0).unsqueeze(-1)
   es=torch.zeros_like(o["history_vectors"]);ec=torch.zeros_like(es)
   es[...,:32]=torch.sqrt(o["history_alpha_sem"][...,None]+1e-8)*o["history_semantic"]
   ec[...,32:]=torch.sqrt(o["history_alpha_comp"][...,None]+1e-8)*o["history_complement"]
   es=es*valid;ec=ec*valid
   pos=(b["lengths"][:,None]-torch.arange(h.shape[1],device=h.device)[None])*(h>0)
   vp=torch.bmm(A,F.linear(model.position_emb(pos),model.extractor.Wv.weight,model.extractor.Wv.bias))
   vs=torch.bmm(A,F.linear(es,model.extractor.Wv.weight,None))
   vc=torch.bmm(A,F.linear(ec,model.extractor.Wv.weight,None))
   maxerr=max(maxerr,float((vs+vc+vp-v).abs().max()))
   assert torch.allclose(vs+vc+vp,v,atol=1e-4,rtol=1e-4)
   branch=[]
   for x in [vs,vc,vp]:
    zs=torch.bmm(x,e.transpose(1,2));zs=zs-zs.mean(-1,keepdim=True)
    branch.append(zs.square().mean(-1).sqrt().cpu().numpy())
   branches.append(np.stack(branch,-1))
   for k in range(K):
    keep=w.clone();keep[:,k]=0;den=keep.sum(-1)
    ok=(den>1e-8)&torch.isfinite(den)&(b["lengths"]>0)
    un=(v*(keep/den.clamp_min(1e-8)[:,None])[:,:,None]).sum(1)
    if kind=="A":un=un+model.refiner(un)
    prediction=(un[:,None]*e).sum(-1)
    ok=ok&torch.isfinite(prediction).all(-1)
    prediction=torch.where(ok[:,None],prediction,o["prediction"])
    p=prediction.cpu().numpy();preds[k+1].append(p);valids[k].append(ok.cpu().numpy())
    at=torch.stack([(A[:,k]-A[:,j]).abs().sum(-1) for j in range(K) if j!=k],1).mean(1)
    div=(zc[:,k]-centeredmean[:,0]).square().mean(-1).sqrt()
    delta=p-p0;delta-=delta.mean(1,keepdims=True)
    top=(-p).argsort(axis=1)[:,:5];items=np.take_along_axis(candidates,top,1)
    f=features[k]
    f["att"].append(at.cpu().numpy());f["score_div"].append(div.cpu().numpy());f["weight"].append(w[:,k].cpu().numpy())
    f["score_change"].append(np.sqrt((delta**2).mean(1)))
    f["slot_top_change"].append(np.any(np.sort(top,axis=1)!=np.sort(ord0,axis=1),axis=1))
    f["item_top_change"].append(np.array([set(x)!=set(y) for x,y in zip(item0,items)]))
 preds=[np.concatenate(p) for p in preds];U=np.concatenate(U);L=np.concatenate(L);br=np.concatenate(branches)
 fullmetrics={k:float(v) for k,v in BaseRunner.evaluate_method(preds[0],[5,10,20],["HR","NDCG"]).items()}
 assert all(abs(fullmetrics[k]-ref[k])<1e-12 for k in fullmetrics),("Baseline reproduction failed",ds,K,kind,fullmetrics,ref)
 _,r0,n0=rank(preds[0]);records=[];rows=dict(user_id=U,history_length=L,full_rank=r0,full_NDCG5=n0)
 for k in range(K):
  p=preds[k+1];ok=np.concatenate(valids[k]);_,r,n=rank(p);delta=n-n0
  f={key:np.concatenate(v) for key,v in features[k].items()}
  metrics={key:float(val) for key,val in BaseRunner.evaluate_method(p,[5,10,20],["HR","NDCG"]).items()}
  rec=dict(deleted_head=k,metrics=metrics,NDCG5_delta=float(delta.mean()),bootstrap95=bootstrap(delta),
      skipped=int((~ok).sum()),fallback="Skipped rows use original score; exclude from valid-only summary",
      valid_only_NDCG5_delta=float(delta[ok].mean()) if ok.any() else None,
      win_loss_tie=[int((delta>0).sum()),int((delta<0).sum()),int((delta==0).sum())],
      positive_rank_change_users=int((r!=r0).sum()),rank_improved_users=int((r<r0).sum()),rank_worse_users=int((r>r0).sum()),
      positive_rank_delta_mean=float((r-r0).mean()),positive_abs_rank_delta=stat(abs(r-r0)),
      top5_slot_set_change_users=int(f["slot_top_change"].sum()),top5_item_set_change_users=int(f["item_top_change"].sum()),
      centered_score_change=stat(f["score_change"]),attention_difference=stat(f["att"]),head_score_difference=stat(f["score_div"]),
      correlations={key:{"with_abs_NDCG5_delta":corr(f[key],abs(delta)),
                        "with_abs_rank_delta":corr(f[key],abs(r-r0)),
                        "with_centered_score_change":corr(f[key],f["score_change"])}
                    for key in ["att","score_div","weight"]},history_groups=[])
  for lo,hi in [(1,5),(6,10),(11,20)]:
   m=(L>=lo)&(L<=hi)
   if m.any():
    d=delta[m];rec["history_groups"].append(dict(history=str(lo)+"-"+str(hi),n=int(m.sum()),
        full_NDCG5=float(n0[m].mean()),deleted_NDCG5=float(n[m].mean()),delta=float(d.mean()),bootstrap95=bootstrap(d),
        win_loss_tie=[int((d>0).sum()),int((d<0).sum()),int((d==0).sum())],
        rank_delta_mean=float((r[m]-r0[m]).mean()),top5_slot_change_users=int(f["slot_top_change"][m].sum())))
  records.append(rec)
  for key,val in [("rank",r),("NDCG5",n),("delta",delta),("valid",ok),("attention_difference",f["att"]),
                  ("head_score_difference",f["score_div"]),("score_change",f["score_change"]),
                  ("top5_slot_change",f["slot_top_change"]),("top5_item_change",f["item_top_change"])]:
   rows["without_"+str(k)+"_"+key]=val
 branch_summary=[]
 for k in range(K):
  s,c,p=br[:,k,0],br[:,k,1],br[:,k,2]
  branch_summary.append(dict(head=k,semantic_centered_score_RMS=stat(s),complement_centered_score_RMS=stat(c),
     position_bias_centered_score_RMS=stat(p),
     semantic_RMS_share_of_sem_plus_comp=stat(s/np.maximum(s+c,1e-30))))
 result=dict(dataset=ds,K=K,model=kind,phase="dev",users=len(U),candidate_slots=1001,
   full_metrics=fullmetrics,baseline_exactly_reproduced=True,checkpoint=str(path),checkpoint_sha256=expected,
   deletions=records,branch_decomposition=dict(max_absolute_reconstruction_error=maxerr,heads=branch_summary,
     scope="Fixed full-model attention and gates; Wv contributions, not pure causal effects or semantic interest labels. RMS shares are not variance explained."),
   input_sha256={k:v.hexdigest() for k,v in inputs.items()},checkpoint_unchanged=sha(path)==expected,
   seconds=time.perf_counter()-start,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
   train_executed=False,test_executed=False,bootstrap="2000 paired user resamples; seed20261019; marginal CIs, no multiple-testing correction",
   skip_rule="remaining weight mass <=1e-8 or no history/nonfinite result; no label-dependent deletion")
 assert result["checkpoint_unchanged"]
 label=ds+"_"+kind+"_K"+str(K)
 with (outdir/(label+"_users.csv")).open("x",newline="") as f:
  wr=csv.writer(f);wr.writerow(rows.keys());wr.writerows(zip(*rows.values()))
 with (outdir/(label+".json")).open("x") as f:json.dump(result,f,indent=2,allow_nan=False)
 print("PASS "+json.dumps(dict(model=label,full=fullmetrics,deletions=[{k:d[k] for k in ["deleted_head","NDCG5_delta","bootstrap95","win_loss_tie","skipped"]} for d in records],seconds=result["seconds"])),flush=True)
 del model,corpus,data,loader,preds;torch.cuda.empty_cache()
 return result
def main():
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError("570-second total diagnosis limit")))
 p=argparse.ArgumentParser();p.add_argument("--output",required=True);a=p.parse_args()
 out=(ROOT/a.output).resolve();assert out.is_relative_to(ROOT) and not out.exists()
 gpu=subprocess.check_output(["nvidia-smi","--id=1","--query-gpu=uuid,memory.free","--format=csv,noheader,nounits"],text=True).strip().split(",")
 apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"],text=True)
 assert gpu[0].strip() not in apps and float(gpu[1])>7000,"GPU1 occupied"
 out.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
 sources={f:sha(ROOT/f) for f in SOURCES};start=time.perf_counter();signal.alarm(570)
 results=[run(ds,K,kind,out) for ds in ["beauty","ml-1m"] for K,kind in [(2,"base"),(4,"base"),(4,"A")]]
 for ds in ["beauty","ml-1m"]:
  selected=[x for x in results if x["dataset"]==ds]
  assert all(x["input_sha256"]==selected[0]["input_sha256"] for x in selected)
 assert all(sha(ROOT/f)==h for f,h in sources.items())
 with (out/"complete.json").open("x") as f:json.dump(dict(results=results,seconds=time.perf_counter()-start,
   source_sha256=sources,source_unchanged=True,train_executed=False,test_executed=False),f,indent=2,allow_nan=False)
 print("COMPLETE "+str(out),flush=True);signal.alarm(0)
if __name__=="__main__":main()

