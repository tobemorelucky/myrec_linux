"""Frequency calibration, one locked Test pass, and frozen Dev interest diagnostics."""
import os
os.environ.setdefault("PYTHONDONTWRITEBYTECODE","1")
import sys,json,time,pickle,hashlib,argparse,signal,csv,importlib.util,itertools
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
ROOT=Path("/home/yyx/hzgcode/pom2.0").resolve()
OUT=Path(__file__).resolve().parent
PREV=ROOT/"research/transition_dev_probe_20261022_v1"
assert hashlib.sha256((PREV/"probe.py").read_bytes()).hexdigest()=="a08e68fe8270bc3c39760ce672ace772cf0cdfb7d215ff96b17161b9d0a3c58e"
sp=importlib.util.spec_from_file_location("prior_fixed_probe",PREV/"probe.py")
prev=importlib.util.module_from_spec(sp);sp.loader.exec_module(prev)
BaseRunner=prev.BaseRunner;Model=prev.Model;utils=prev.utils
from models.sequential.LLMMIRecASPCFSampledSoftmax import candidate_mask,build_train_membership
METRICS=("HR@5","NDCG@5","HR@10","NDCG@10","HR@20","NDCG@20")
GRID=(0.,.03,.1,.2,.3)
def write(path,x):
 with Path(path).open("x") as f:json.dump(x,f,indent=2,allow_nan=False)
def sha(path):return prev.sha(path)
def stat(x):return prev.stats(x)
def metric(x):return {k:float(v) for k,v in BaseRunner.evaluate_method(x,[5,10,20],["HR","NDCG"]).items()}
def rank(x):return np.argwhere((-x).argsort(axis=1)==0)[:,1]+1
def n5(r):return (r<=5)/np.log2(r+1)
def csvwrite(path,rows):
 with Path(path).open("x",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load(ds,phase,model=True):
 prior=json.loads((PREV/ds/"result.json").read_text())
 sources={name:sha(ROOT/name) for name in prev.SOURCES}
 assert sources==prior["source_sha256"]
 spec=prev.SPEC[ds]
 inputs={k:sha(ROOT/"data"/ds/(k+".csv")) for k in ["train","dev","test"]}
 inputs["checkpoint"]=sha(spec["checkpoint"]);inputs["embedding"]=sha(spec["embedding"])
 inputs["cache"]=sha(ROOT/"data"/ds/"SeqReader.pkl")
 assert all(inputs[k]==prior["input_sha256"][k] for k in prior["input_sha256"])
 with (ROOT/"data"/ds/"SeqReader.pkl").open("rb") as f:c=pickle.load(f)
 if not model:return prior,c,None,None,inputs,sources
 c.data_df={phase:c.data_df[phase]}
 args=prev.arguments(ds,c);m=Model(args,c).to(args.device)
 m.load_state_dict(torch.load(spec["checkpoint"],map_location=args.device,weights_only=True),strict=True)
 m.eval();d=m.Dataset(m,c,phase)
 assert len(d)>0 and all(len(v)==1000 for v in d.data["neg_items"])
 loader=DataLoader(d,batch_size=256,shuffle=False,num_workers=0,collate_fn=d.collate_batch)
 return prior,c,m,loader,inputs,sources
def check_inputs(ds,inputs,sources):
 s=prev.SPEC[ds]
 paths={k:ROOT/"data"/ds/(k+".csv") for k in ["train","dev","test"]}
 paths.update(checkpoint=s["checkpoint"],embedding=s["embedding"],cache=ROOT/"data"/ds/"SeqReader.pkl")
 assert all(sha(p)==inputs[k] for k,p in paths.items())
 assert all(sha(ROOT/name)==h for name,h in sources.items())
def bucket_ids(counts,thresholds):
 a=np.searchsorted(thresholds,counts,side="left")+1
 return np.where(counts==0,0,a)
def divergence(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float);a=a/a.sum();b=b/b.sum();m=(a+b)/2
 take=a>0;take2=b>0
 return dict(TV=float(abs(a-b).sum()/2),JSD_nats=float((a[take]*np.log(a[take]/m[take])).sum()/2+(b[take2]*np.log(b[take2]/m[take2])).sum()/2))
def audit_distribution(ds):
 started=time.perf_counter();prior,c,_,_,inputs,sources=load(ds,"dev",False);n=c.n_items
 train_raw=pd.read_csv(ROOT/"data"/ds/"train.csv",sep="\t",usecols=["user_id","item_id"])
 raw=np.bincount(train_raw.item_id.to_numpy(np.int64),minlength=n)
 df=c.data_df["train"];df=df.loc[np.asarray(df["position"])>0]
 ids=df.item_id.to_numpy(np.int64);users=df.user_id.to_numpy(np.int64)
 target=np.bincount(ids,minlength=n);user_rows=np.bincount(users,minlength=c.n_users)
 removed=np.zeros(n,np.float64)
 for u,clicked in c.train_clicked_set.items():
  ar=np.asarray(list(clicked),np.int64)
  removed[ar]+=user_rows[u]
 eligible=len(ids)-removed
 assert (eligible>=0).all()
 # Exact marginal under uniform random partitioning into batches; common co-batch factor cancels.
 expected=target.astype(np.float64)*eligible;expected/=expected.sum()
 membership=build_train_membership(c).to("cuda:0")
 g=np.random.default_rng(20261022);perm=g.permutation(len(ids));emp=torch.zeros(n,device="cuda:0",dtype=torch.float64)
 allowed_slots=0;slots=0
 for off in range(0,min(len(ids),32*1024),1024):
  ix=perm[off:off+1024];ii=torch.as_tensor(ids[ix],device="cuda:0")
  uu=torch.as_tensor(users[ix],device="cuda:0")
  allowed,_,_=candidate_mask(membership,uu,ii)
  weights=allowed.sum(0)-1
  emp.scatter_add_(0,ii,weights.to(torch.float64))
  allowed_slots+=int(weights.sum());slots+=len(ix)*(len(ix)-1)
 emp=emp.cpu().numpy();del membership
 neg={};neg_meta={}
 for phase in ["dev","test"]:
  values=c.data_df[phase]["neg_items"].to_numpy()
  freq=np.zeros(n,np.int64);duplicate=0;hist_hits=0;total=0
  for off in range(0,len(values),1024):
   ar=np.stack(values[off:off+1024]).astype(np.int64)
   assert ar.shape[1]==1000 and ar.min()>=0 and ar.max()<n
   freq+=np.bincount(ar.ravel(),minlength=n)
   sort=np.sort(ar,axis=1);duplicate+=int((sort[:,1:]==sort[:,:-1]).sum())
   total+=ar.size
  neg[phase]=freq;neg_meta[phase]=dict(rows=len(values),slots=total,padding_slots=int(freq[0]),duplicate_slots_beyond_first=duplicate)
 thresholds=np.quantile(raw[raw>0],[.2,.4,.6,.8]).tolist()
 bins=bucket_ids(raw,thresholds);uniform=np.ones(n);uniform[0]=0
 distributions=dict(raw_train=raw,actual_positive_targets=target,expected_filtered_inbatch=expected,
  simulated_32_batches_filtered=emp,dev_negative_slots=neg["dev"],test_negative_slots=neg["test"],catalog_uniform=uniform)
 comparisons={name:divergence(expected,a) for name,a in distributions.items() if name!="expected_filtered_inbatch"}
 bucket_mass={name:[float(a[bins==b].sum()/a.sum()) for b in range(6)] for name,a in distributions.items()}
 # Shape comparison only. Eval frequencies never enter the predictive correction.
 shape={}
 lograw=np.log1p(raw[1:])
 for phase in ["dev","test"]:
  lr=np.log((expected[1:]+1e-12)/(neg[phase][1:]/neg[phase].sum()+1e-12))
  corr=float(np.corrcoef(lograw,lr)[0,1])
  shape[phase]=dict(correlation_log_raw_frequency_vs_log_Qtrain_over_Qeval=corr,linear_variance_R2=corr*corr,
   note="distribution shape only; NOT fraction of recommendation gain causally explained")
 summary=dict(dataset=ds,raw_train_events=int(raw.sum()),actual_positive_target_rows=len(ids),
  removed_position_zero_rows=int(raw.sum()-len(ids)),positive_frequency_vs_raw=divergence(raw,target),
  negative_slot_metadata=neg_meta,expected_formula="target_count[j]*(N - sum_u(target_rows[u]*1[j in TrainClicked[u]])); normalized",
  expected_distribution_note="exact expected item-slot marginal for random shuffled batches; query-conditional masks remain non-IID",
  simulation=dict(seed=20261022,batches=min(32,int(np.ceil(len(ids)/1024))),allowed_negative_slots=allowed_slots,
   removed_fraction=1-allowed_slots/slots,note="current exact mask on fixed simulated batches, not archived historical epoch trace"),
  distribution_vs_expected_filtered=comparisons,bucket_thresholds=thresholds,bucket_mass=bucket_mass,
  frequency_correction_shape=shape,inputs_sha256=inputs,source_sha256=sources,seconds=time.perf_counter()-started,
  no_Train_or_Test_model_evaluation=True)
 out=OUT/ds;out.mkdir(exist_ok=True)
 np.savez_compressed(out/"frequency_distribution.npz",raw=raw,target=target,expected=expected,simulated=emp,
                     dev_negative=neg["dev"],test_negative=neg["test"],bucket=bins,thresholds=thresholds)
 rows=[dict(item_id=i,raw_train=int(raw[i]),actual_positive_targets=int(target[i]),
   expected_filtered_probability=float(expected[i]),dev_negative_slots=int(neg["dev"][i]),
   test_negative_slots=int(neg["test"][i]),bucket=int(bins[i])) for i in range(n)]
 csvwrite(out/"frequency_distribution.csv",rows);check_inputs(ds,inputs,sources)
 write(out/"distribution.json",summary);print(json.dumps(summary),flush=True)

def aggregate_rows(all_ranks,labels,bins):
 rows=[]
 base=all_ranks["lambda0"]
 for key,rr in all_ranks.items():
  got={f"HR@{k}":float((rr<=k).mean()) for k in [5,10,20]}
  got.update({f"NDCG@{k}":float(((rr<=k)/np.log2(rr+1)).mean()) for k in [5,10,20]})
  out=dict(scheme=key,**got)
  for k in METRICS:
   b=float((base<=int(k.split("@")[1])).mean()) if k.startswith("HR") else float(((base<=int(k.split("@")[1]))/np.log2(base+1)).mean())
   out["delta_"+k]=out[k]-b
  out["relative_NDCG5_pct"]=100*(n5(rr).mean()/n5(base).mean()-1)
  rows.append(out)
 groups=[]
 for b in range(6):
  take=bins==b
  if not take.any():continue
  for key,rr in all_ranks.items():
   groups.append(dict(bucket=b,users=int(take.sum()),scheme=key,
    **{f"HR@{k}":float((rr[take]<=k).mean()) for k in [5,10,20]},
    **{f"NDCG@{k}":float(((rr[take]<=k)/np.log2(rr[take]+1)).mean()) for k in [5,10,20]},
    delta_NDCG5=float((n5(rr)-n5(base))[take].mean()),
    signed_share_of_total_NDCG5_gain=None if abs((n5(rr)-n5(base)).sum())<1e-12 else float((n5(rr)-n5(base))[take].sum()/(n5(rr)-n5(base)).sum())))
 return rows,groups
def evaluate_frequency(ds,phase):
 started=time.perf_counter();out=OUT/ds
 if phase=="test":
  assert all((OUT/name/"DEV_LOCK.json").is_file() for name in ["beauty","ml-1m"])
  lock=json.loads((out/"DEV_LOCK.json").read_text())
  assert lock["script_sha256"]==sha(__file__)
  write(out/"TEST_STARTED.json",dict(started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
   lock_sha256=sha(out/"DEV_LOCK.json"),test_pass_budget=1,no_coefficient_change_after_lock=True))
 prior,c,m,loader,inputs,sources=load(ds,phase)
 freq=np.load(out/"frequency_distribution.npz");raw=freq["raw"];bins=freq["bucket"]
 scale=float(prior["scale_factors"]["P3_item_popularity"])
 coeffs=GRID if phase=="dev" else (0.,float(lock["lambda_normalized"]))
 coeffs=tuple(dict.fromkeys(coeffs))
 keys=[f"lambda{a:g}" for a in coeffs]+["pure_frequency"]
 rr={k:[] for k in keys};labels=[];us=[];predicted={k:np.zeros(6,np.int64) for k in keys}
 scorebias={b:dict(users=0,positive_score_sum=0.,negative_score_sum=0.,raw_log_frequency_sum=0.) for b in range(6)}
 poptied=0;total=0;scorecorr=[];inputhash=hashlib.sha256()
 pop=torch.as_tensor(np.log1p(raw),dtype=torch.float32,device=m.device)
 torch.cuda.reset_peak_memory_stats()
 with torch.inference_mode():
  for batch in loader:
   ids=batch["item_id"].numpy();targets=ids[:,0];bb=bins[targets];labels.append(bb);us.append(batch["user_id"].numpy())
   for name in ["user_id","history_items","lengths","item_id"]:
    ar=batch[name].numpy();inputhash.update(name.encode());inputhash.update(str(ar.shape).encode());inputhash.update(ar.tobytes())
   feed=utils.batch_to_gpu(batch,m.device);base=m(feed)["prediction"].cpu().numpy().astype(np.float64)
   pr=pop[feed["item_id"]].cpu().numpy().astype(np.float64);center=pr-pr.mean(1,keepdims=True)
   bc=base-base.mean(1,keepdims=True)
   denom=np.sqrt(np.square(bc).sum(1)*np.square(center).sum(1))
   scorecorr.extend((np.sum(bc*center,1)/np.maximum(denom,1e-20)).tolist())
   poptied+=int((pr[:,1:]==pr[:,:1]).any(1).sum());total+=len(base)
   for b in range(6):
    take=bb==b
    scorebias[b]["users"]+=int(take.sum());scorebias[b]["positive_score_sum"]+=float(base[take,0].sum())
    scorebias[b]["negative_score_sum"]+=float(base[take,1:].mean(1).sum())
    scorebias[b]["raw_log_frequency_sum"]+=float(pr[take,0].sum())
   for key in keys:
    score=pr if key=="pure_frequency" else base+float(key[6:])*scale*center
    rank1=rank(score);rr[key].append(rank1)
    # Validate exact original evaluator on every batch; per-user ranks use same NumPy argsort.
    original=metric(score)
    assert all(abs(original[f"HR@{k}"]-(rank1<=k).mean())<1e-12 for k in [5,10,20])
    assert all(abs(original[f"NDCG@{k}"]-((rank1<=k)/np.log2(rank1+1)).mean())<1e-12 for k in [5,10,20])
    top=(-score).argsort(1)[:,:5]
    predicted[key]+=np.bincount(bins[np.take_along_axis(ids,top,1)].ravel(),minlength=6)
 ranks_all={k:np.concatenate(v) for k,v in rr.items()};groups_label=np.concatenate(labels)
 rows,groups=aggregate_rows(ranks_all,labels,groups_label)
 baseline=next(x for x in rows if x["scheme"]=="lambda0")
 expected=prior["metrics"][0]["NDCG@5"] if phase=="dev" else (0.1294 if ds=="beauty" else .2496165979817967)
 assert abs(baseline["NDCG@5"]-expected)<(.000051 if ds=="beauty" and phase=="test" else 1e-10)
 for v in scorebias.values():
  n=max(v["users"],1)
  for name in list(v):
   if name.endswith("_sum"):v[name[:-4]+"_mean"]=v.pop(name)/n
 result=dict(dataset=ds,phase=phase,rows=total,candidates=1001,scale_factor_from_prior_Train_RMS=scale,
   metrics=rows,popularity_groups=groups,predicted_top5_bucket_counts={k:v.tolist() for k,v in predicted.items()},
   score_bias_by_target_bucket=scorebias,per_user_score_logpop_correlation=stat(scorecorr),
   pure_pop_positive_tied_with_negative_user_fraction=poptied/total,
   input_sha256=inputs,source_sha256=sources,script_sha256=sha(__file__),dev_or_test_inputs_sha256=inputhash.hexdigest(),
   model_parameters=m.count_variables(),additional_trainable_parameters=0,model_training_updates=0,
   seconds=time.perf_counter()-started,peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20)
 check_inputs(ds,inputs,sources)
 write(out/(phase+"_result.json"),result);csvwrite(out/(phase+"_metrics.csv"),rows)
 csvwrite(out/(phase+"_popularity_buckets.csv"),groups)
 with (out/(phase+"_user_ranks.csv")).open("x",newline="") as f:
  w=csv.writer(f);w.writerow(["user_id","popularity_bucket"]+keys)
  w.writerows(zip(np.concatenate(us),groups_label,*[ranks_all[k] for k in keys]))
 if phase=="dev":
  selected=max((r for r in rows if r["scheme"]!="pure_frequency"),key=lambda r:(r["NDCG@5"],-float(r["scheme"][6:])))
  write(out/"DEV_LOCK.json",dict(dataset=ds,lambda_normalized=float(selected["scheme"][6:]),
   raw_lambda=float(selected["scheme"][6:])*scale,scale_factor=scale,dev_metrics=selected,
   grid=GRID,checkpoint_sha256=inputs["checkpoint"],embedding_sha256=inputs["embedding"],
   all_inputs_sha256=inputs,source_sha256=sources,script_sha256=sha(__file__),
   frequency_file_sha256=sha(out/"frequency_distribution.npz"),
   old_scale_source_sha256=sha(PREV/ds/"result.json"),lock_before_any_Test_scoring=True))
 else:
  write(out/"TEST_COMPLETE.json",dict(dataset=ds,one_Test_pass=True,lock_sha256=sha(out/"DEV_LOCK.json"),
   result_sha256=sha(out/"test_result.json"),checkpoint_unchanged=True,metrics=rows))
 print(json.dumps(dict(dataset=ds,phase=phase,metrics=rows,seconds=result["seconds"])),flush=True)
def diagnostics(ds):
 started=time.perf_counter();out=OUT/ds;lock=json.loads((out/"DEV_LOCK.json").read_text())
 prior,c,m,loader,inputs,sources=load(ds,"dev")
 frequency=np.load(out/"frequency_distribution.npz");raw=frequency["raw"];bins=frequency["bucket"]
 pop=torch.as_tensor(np.log1p(raw),dtype=torch.float32,device=m.device)
 alpha=lock["raw_lambda"];ranks_all={};weights=[];cosines=[];effective=[];centered_effective=[];attn_entropy=[]
 overlap={f"{p}_N{N}_{kind}":[] for p in ["raw","corrected"] for N in [5,10,20] for kind in ["slot","item"]}
 target_bins=[];users=[];positive_winner=[];max_identity=0.;peak=0
 torch.cuda.reset_peak_memory_stats()
 with torch.inference_mode():
  for batch in loader:
   ids=batch["item_id"].numpy();target_bins.append(bins[ids[:,0]]);users.append(batch["user_id"].numpy())
   feed=utils.batch_to_gpu(batch,m.device);o=m(feed,True);v=o["interest_vectors"];w=o["interest_weights"];a=o["attention_maps"]
   z=torch.bmm(v,o["candidate_vectors"].transpose(1,2)).cpu().numpy().astype(np.float64)
   ww=w.cpu().numpy();base=o["prediction"].cpu().numpy().astype(np.float64)
   check=(z*ww[:,:,None]).sum(1);max_identity=max(max_identity,float(np.max(abs(base-check))))
   assert np.allclose(check,base,rtol=2e-5,atol=2e-5)
   pp=pop[feed["item_id"]].cpu().numpy().astype(np.float64);prior_score=alpha*(pp-pp.mean(1,keepdims=True))
   weights.append(ww);positive_winner.append(z[:,:,0].argmax(1))
   vn=F.normalize(v,dim=-1);gram=vn@vn.transpose(1,2)
   cosines.append(torch.stack([gram[:,i,j] for i,j in itertools.combinations(range(4),2)],1).cpu().numpy())
   for vv,store in [(v,effective),(v-v.mean(1,keepdim=True),centered_effective)]:
    eig=torch.linalg.eigvalsh(vv@vv.transpose(1,2)).clamp_min(0);prob=eig/eig.sum(1,keepdim=True).clamp_min(1e-20)
    store.append(torch.exp(-(prob*prob.clamp_min(1e-20).log()).sum(1)).cpu().numpy())
   attn_entropy.append((-(a*a.clamp_min(1e-20).log()).sum(-1)).cpu().numpy())
   std=np.sqrt(np.mean(np.square(base-base.mean(1,keepdims=True)),1,keepdims=True))
   variants={"full":base}
   normheads=[]
   for k in range(4):
    den=1-ww[:,k:k+1];assert (den>1e-8).all()
    drop=(check-ww[:,k:k+1]*z[:,k,:])/den
    dstd=np.sqrt(np.mean(np.square(drop-drop.mean(1,keepdims=True)),1,keepdims=True))
    matched=base.mean(1,keepdims=True)+(drop-drop.mean(1,keepdims=True))*std/np.maximum(dstd,1e-12)
    variants[f"drop{k+1}_renorm"]=drop;variants[f"drop{k+1}_RMSmatched"]=matched
    hz=z[:,k,:];hs=np.sqrt(np.mean(np.square(hz-hz.mean(1,keepdims=True)),1,keepdims=True))
    hn=base.mean(1,keepdims=True)+(hz-hz.mean(1,keepdims=True))*std/np.maximum(hs,1e-12)
    variants[f"single{k+1}_RMSmatched"]=hn;normheads.append(hn)
   for key,score in variants.items():
    for suffix,extra in [("raw",0.),("corrected",prior_score)]:
     sk=key+"_"+suffix;pred=score+extra;rr=rank(pred)
     got=metric(pred);assert abs(got["NDCG@5"]-n5(rr).mean())<1e-12
     ranks_all.setdefault(sk,[]).append(rr)
   normheads=np.stack(normheads,1)
   for phase,hscore in [("raw",z),("corrected",normheads+prior_score[:,None,:])]:
    order=(-hscore).argsort(-1)[...,:20]
    for N in [5,10,20]:
     per_slot=[];per_item=[]
     for i,j in itertools.combinations(range(4),2):
      left=order[:,i,:N];right=order[:,j,:N]
      inter=(left[:,:,None]==right[:,None,:]).any(-1).sum(-1)
      per_slot.append(inter/(2*N-inter))
      li=np.sort(np.take_along_axis(ids,left,1),axis=1);ri=np.sort(np.take_along_axis(ids,right,1),axis=1)
      lv=np.c_[np.ones(len(li),bool),li[:,1:]!=li[:,:-1]]
      rv=np.c_[np.ones(len(ri),bool),ri[:,1:]!=ri[:,:-1]]
      ii=((li[:,:,None]==ri[:,None,:])&rv[:,None,:]).any(-1)&lv
      common=ii.sum(1);union=lv.sum(1)+rv.sum(1)-common
      per_item.append(common/np.maximum(union,1))
     overlap[f"{phase}_N{N}_slot"].append(np.stack(per_slot,1))
     overlap[f"{phase}_N{N}_item"].append(np.stack(per_item,1))
 rr={k:np.concatenate(v) for k,v in ranks_all.items()};bb=np.concatenate(target_bins);ww=np.concatenate(weights)
 wnorm=-(ww*np.log(np.maximum(ww,1e-20))).sum(1)
 rows=[];bucketrows=[];unique={}
 for key,r in rr.items():
  suffix=key.split("_")[-1];br=rr["full_"+suffix];d=n5(r)-n5(br)
  rows.append(dict(scheme=key,NDCG5=float(n5(r).mean()),delta_vs_full_same_prior=float(d.mean()),
    paired_SE=float(d.std(ddof=1)/np.sqrt(len(d))),
    **{f"HR@{k}":float((r<=k).mean()) for k in [5,10,20]},
    **{f"NDCG@{k}":float(((r<=k)/np.log2(r+1)).mean()) for k in [5,10,20]}))
  for b in range(6):
   take=bb==b
   if take.any():bucketrows.append(dict(scheme=key,bucket=b,users=int(take.sum()),NDCG5=float(n5(r[take]).mean()),delta_vs_full_same_prior=float(d[take].mean())))
 for suffix in ["raw","corrected"]:
  hit=np.stack([rr[f"single{k+1}_RMSmatched_{suffix}"]<=5 for k in range(4)],1)
  unique[suffix]=dict(only_head_top5_success=[int((hit[:,k]&(hit.sum(1)==1)).sum()) for k in range(4)],
    per_head_Top5_success=hit.sum(0).tolist(),
    by_popularity_bucket={str(b):dict(users=int((bb==b).sum()),single_head_NDCG5=[float(n5(rr[f"single{k+1}_RMSmatched_{suffix}"])[bb==b].mean()) for k in range(4)]) for b in range(6) if (bb==b).any()})
 raw_ref=json.loads((out/"dev_result.json").read_text())["metrics"][0]["NDCG@5"]
 assert abs(float(n5(rr["full_raw"]).mean())-raw_ref)<1e-10
 result=dict(dataset=ds,phase="dev",rows=len(ww),fixed_lambda=lock["lambda_normalized"],
  cosine_head_pairs=stat(np.concatenate(cosines)),effective_rank_energy_entropy=stat(np.concatenate(effective)),
  centered_effective_rank_max3=stat(np.concatenate(centered_effective)),
  attention_entropy=stat(np.concatenate(attn_entropy)),
  aggregation=dict(mean_weights=ww.mean(0).tolist(),weight_entropy=stat(wnorm),max_weight=stat(ww.max(1)),
   fraction_max_gt_half=float((ww.max(1)>.5).mean()),fraction_max_gt_075=float((ww.max(1)>.75).mean()),
   argmax_counts=np.bincount(ww.argmax(1),minlength=4).tolist()),
  head_candidate_TopN_Jaccard={k:stat(np.concatenate(v)) for k,v in overlap.items()},
  deletion_and_single_head_metrics=rows,by_target_popularity=bucketrows,head_rank_responsibility=unique,
  per_item_prior_is_shared_across_heads=True,
  positive_raw_head_argmax_counts=np.bincount(np.concatenate(positive_winner),minlength=4).tolist(),
  winner_change_due_to_common_additive_prior="mathematically zero at fixed item for unnormalized raw head scores; candidate rankings and utility can change",
  controls="drop: remaining w renormalized; RMSmatched: restore baseline per-user centered candidate score RMS before adding same fixed global prior; single heads RMSmatched likewise",
  limitations="frozen interventions, not retrained causal attribution; single-head comparisons are not trained K1 models; no semantic labels",
  input_sha256=inputs,source_sha256=sources,max_weighted_score_identity_error=max_identity,
  seconds=time.perf_counter()-started,peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20,no_Test_head_diagnostics=True)
 check_inputs(ds,inputs,sources)
 write(out/"interest_dev.json",result);csvwrite(out/"interest_metrics.csv",rows);csvwrite(out/"interest_popularity_buckets.csv",bucketrows)
 with (out/"interest_dev_user_ranks.csv").open("x",newline="") as f:
  wr=csv.writer(f);wr.writerow(["user_id","popularity_bucket"]+list(rr));wr.writerows(zip(np.concatenate(users),bb,*rr.values()))
 print(json.dumps(dict(dataset=ds,cosine=result["cosine_head_pairs"],rank=result["effective_rank_energy_entropy"],aggregation=result["aggregation"],head_metrics=rows,seconds=result["seconds"])),flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--dataset",choices=["ml-1m","beauty"],required=True);p.add_argument("--stage",choices=["distribution","dev","test","interests"],required=True)
 x=p.parse_args();torch.set_num_threads(4);utils.init_seed(42)
 signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError("170s bounded stage")))
 signal.alarm(170)
 if x.stage=="distribution":audit_distribution(x.dataset)
 elif x.stage in ["dev","test"]:evaluate_frequency(x.dataset,x.stage)
 else:diagnostics(x.dataset)
 signal.alarm(0)

