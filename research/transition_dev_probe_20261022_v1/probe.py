"""Frozen-checkpoint Dev transition probe. No optimizer, backward, Train/Test Dataset or main.py."""
import os
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import sys,json,time,hashlib,pickle,argparse,signal,csv,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader
ROOT=Path("/home/yyx/hzgcode/pom2.0").resolve()
sys.path.insert(0,str(ROOT))
from helpers.BaseRunner import BaseRunner
from models.sequential.LLMMIRecASPCFSampledSoftmax import LLMMIRecASPCFSampledSoftmax as Model
from utils import utils
LAMBDAS=(.01,.03,.10)
BRANCHES=("P1_semantic","P2_transition","P3_prototype_popularity","P3_item_popularity")
M=32;RANK=512;TAU=.1;BETA=10.;EPS=1e-12
OUT=Path(__file__).resolve().parent
W=ROOT/"research/ml1m_embedding_v2_20261020"
SPEC={
"ml-1m":dict(checkpoint=W/"runs/cross_v3_20261021/p920/aspcf_A1_k4_seed42_20261021_cross/best.pt",
 log=W/"runs/cross_v3_20261021/p920/aspcf_A1_k4_seed42_20261021_cross/train.log",
 embedding=W/"runs/cross_v3_20261021/p920/incoming/A1_oldlora_rich512/rich_pca.pkl",
 checkpoint_sha="65343d00f84bcfb4e8deb4fce47149f55da08cdfdd98c54ec77b0bcd1f26409b",
 embedding_sha="03cf9649844da2c172487a0f552a49a4d331d8262135e3b70e48d6e772ad7905",
 dev_n5=.2742475064258289),
"beauty":dict(checkpoint=ROOT/"new_model/aspcf_sampled_softmax/beauty/k4_tau1_seed42_20261010T112637Z_round1/best.pt",
 log=ROOT/"new_log/aspcf_sampled_softmax/beauty/k4_tau1_seed42_20261010T112637Z_round1/train.log",
 embedding=ROOT/"data/beauty/handled/llm_table_pca1536.pkl",
 checkpoint_sha="0ee64e02c64bcf96ef8f88c220c8ac6683dc4bab69ff814cbca11bb3f9970f1d",
 embedding_sha="bc282725f08c2bb7fbb524739d1711cee44810ab402f18119aa838d9e5ed859f",
 dev_n5=.145674)}
SOURCES=["models/sequential/LLMMIRecASPCF.py","models/sequential/LLMMIRecASPCFSampledSoftmax.py",
 "models/sequential/llmmi_components.py","models/BaseModel.py","helpers/BaseRunner.py","helpers/SeqReader.py"]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(2**20),b""):h.update(block)
 return h.hexdigest()
def inside(path):
 p=Path(path).resolve()
 if not p.is_relative_to(ROOT):raise ValueError("Outside project: "+str(p))
 return p
def write_json(path,x):
 with inside(path).open("x") as f:json.dump(x,f,indent=2,allow_nan=False)
def stats(x):
 x=np.asarray(x,dtype=np.float64).ravel()
 return dict(mean=float(x.mean()),std=float(x.std()),min=float(x.min()),p10=float(np.quantile(x,.1)),
             median=float(np.median(x)),p90=float(np.quantile(x,.9)),max=float(x.max()))
def probability(count):
 mass=count.sum(1); p=(count.sum(0)+1e-8)/(count.sum()+M*1e-8)
 T=(count+BETA*p[None])/(mass[:,None]+BETA)
 assert np.isfinite(T).all() and np.allclose(T.sum(1),1) and np.all(T>0)
 return T,p,mass
def arguments(ds,corpus):
 s=SPEC[ds];old={}
 for line in s["log"].read_text().splitlines():
  cols=line.split("|")
  if len(cols)>=2:old[cols[0].strip()]=cols[1].strip()
 parser=BaseRunner.parse_runner_args(Model.parse_model_args(argparse.ArgumentParser()))
 tokens=[]
 for a in parser._actions:
  if a.dest in old and a.dest!="help" and "..." not in old[a.dest]:
   tokens.extend([a.option_strings[0],old[a.dest]])
 args=parser.parse_args(tokens);args.device=torch.device("cuda:0");args.buffer=0
 args.llm_emb_path=str(s["embedding"]);args.model_path=str(s["checkpoint"])
 args.test_all=0;args.eval_batch_size=256
 assert args.K==4 and args.history_max==20 and args.emb_size==64 and args.num_neg==1
 assert args.lambda_relation==.01 and args.sampled_softmax_temperature==1
 return args
def signal_batch(feed,q,z,T,p,pop):
 ids=feed["item_id"];h=feed["history_items"];lengths=feed["lengths"]
 rows=torch.arange(len(ids),device=ids.device);last=h[rows,lengths-1]
 a=q[last];b=q[ids]
 sem=(z[last,None,:]*z[ids]).sum(-1)
 numerator=torch.einsum("bm,bcm->bc",a@T,b)
 denominator=torch.einsum("m,bcm->bc",p,b)
 trans=torch.log(numerator+EPS)-torch.log(denominator+EPS)
 return dict(P1_semantic=sem,P2_transition=trans,
             P3_prototype_popularity=torch.log(denominator+EPS),
             P3_item_popularity=torch.log(pop[ids]+1))
def metric(pred):
 return {k:float(v) for k,v in BaseRunner.evaluate_method(pred,[5,10,20],["HR","NDCG"]).items()}
def ranks(pred):
 return np.argwhere((-pred).argsort(axis=1)==0)[:,1]+1
def run(ds):
 signal.alarm(170);start=time.perf_counter();s=SPEC[ds];target=inside(OUT/ds);target.mkdir(exist_ok=True)
 torch.set_num_threads(4);utils.init_seed(42);rng=np.random.default_rng(20261022)
 paths={"checkpoint":s["checkpoint"],"embedding":s["embedding"],"train":ROOT/"data"/ds/"train.csv",
        "dev":ROOT/"data"/ds/"dev.csv","cache":ROOT/"data"/ds/"SeqReader.pkl"}
 before={k:sha(inside(p)) for k,p in paths.items()}
 source_hash={name:sha(ROOT/name) for name in SOURCES}
 assert before["checkpoint"]==s["checkpoint_sha"] and before["embedding"]==s["embedding_sha"]
 print(json.dumps(dict(dataset=ds,stage="inputs_verified")),flush=True)
 with paths["embedding"].open("rb") as f:table=np.asarray(pickle.load(f),dtype=np.float32)
 assert table.ndim==2 and table.shape[1]==1536 and np.isfinite(table).all()
 assert np.all(table[0]==0) and np.all(np.linalg.norm(table[1:],axis=1)>0)
 n=len(table);z=table[:,:RANK].copy();z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-8);z[0]=0
 train=pd.read_csv(paths["train"],sep="\t",usecols=["user_id","item_id","time"])
 train=train.sort_values(["user_id","time"],kind="mergesort")
 uid=train.user_id.to_numpy(np.int64);ids=train.item_id.to_numpy(np.int64);ts=train.time.to_numpy()
 assert ids.min()>0 and ids.max()<n
 fit_ids=np.unique(ids);fit_start=time.perf_counter()
 reuse_assets=(target/"relations.npz").is_file()
 if reuse_assets:
  prior_manifest=json.loads((target/"prototype_manifest.json").read_text())
  assert prior_manifest["source_embedding_sha256"]==before["embedding"]
  assert prior_manifest["fit_train_sha256"]==before["train"]
  assert prior_manifest["relations_sha256"]==sha(target/"relations.npz")
  cached=np.load(target/"relations.npz")
 km=MiniBatchKMeans(n_clusters=M,random_state=42,batch_size=1024,n_init=3,max_iter=100)
 if reuse_assets:
  centers=cached["centers"];q=cached["assignments"]
 else:
  km.fit(z[fit_ids]);centers=km.cluster_centers_.astype(np.float32)
  centers/=np.maximum(np.linalg.norm(centers,axis=1,keepdims=True),1e-8)
  logits=z@centers.T/TAU;logits-=logits.max(1,keepdims=True)
  q=np.exp(logits);q/=q.sum(1,keepdims=True);q[0]=0
 assert np.allclose(q[1:].sum(1),1,atol=1e-6)
 same=uid[1:]==uid[:-1];left=ids[:-1][same];right=ids[1:][same];owners=uid[:-1][same]
 folds=owners%3;counts=np.zeros((3,M,M),np.float64)
 if reuse_assets:
  counts=cached["counts_by_fold"]
 else:
  for f in range(3):
   a=left[folds==f];b=right[folds==f]
   for off in range(0,len(a),8192):
    qa=q[a[off:off+8192]].astype(np.float64);qb=q[b[off:off+8192]].astype(np.float64)
    counts[f]+=qa.T@qb
 total=counts.sum(0);T,p,mass=probability(total)
 fold_tables=[probability(total-counts[f]) for f in range(3)]
 pop=np.bincount(ids,minlength=n).astype(np.float64)
 pop_fold=np.stack([np.bincount(ids[uid%3!=f],minlength=n) for f in range(3)]).astype(np.float64)
 ent=-(q[1:]*np.log(np.maximum(q[1:],EPS))).sum(1)
 hard=q[left].argmax(1)
 pair_user=np.unique(owners*M+hard)
 support=np.bincount(pair_user%M,minlength=M)
 qleft=q[left].astype(np.float64);neff=mass**2/np.maximum((qleft*qleft).sum(0),EPS)
 semantic=np.exp((centers@centers.T)/TAU-((centers@centers.T)/TAU).max(1,keepdims=True))
 semantic/=semantic.sum(1,keepdims=True)
 relation=dict(adjacent_pairs=len(left),train_rows=len(ids),train_users=int(np.unique(uid).size),
  timestamp_tied_adjacent_pairs=int((ts[1:][same]==ts[:-1][same]).sum()),
  fit_catalog_items=len(fit_ids),catalog_items=n-1,source_mass=stats(mass),source_effective_events=stats(neff),
  hard_source_user_support=stats(support),item_assignment_entropy=stats(ent),assignment_effective_states=stats(np.exp(ent)),
  assignment_max_probability=stats(q[1:].max(1)),smoothed_transition_min=float(T.min()),
  weighted_transition_KL_to_marginal=float(((T*np.log(T/p[None])).sum(1)*mass/mass.sum()).sum()),
  relative_count_asymmetry=float(np.linalg.norm(total-total.T)/np.maximum(np.linalg.norm(total),EPS)),
  semantic_transition_weighted_KL=float(((T*np.log(T/semantic)).sum(1)*mass/mass.sum()).sum()),
  folds=[dict(fold=f,excluded_users=int(np.unique(uid[uid%3==f]).size),pairs_excluded=int((folds==f).sum())) for f in range(3)])
 # Exactly one last-Train target/user; statistics exclude that user's complete fold.
 starts=np.r_[0,np.flatnonzero(uid[1:]!=uid[:-1])+1];ends=np.r_[starts[1:],len(ids)]
 eligible=np.flatnonzero(ends-starts>=2);chosen=np.sort(rng.choice(eligible,min(1024,len(eligible)),replace=False))
 hist=np.zeros((len(chosen),20),np.int64);lengths=np.zeros(len(chosen),np.int64);users=uid[starts[chosen]]
 candidates=rng.choice(fit_ids,size=(len(chosen),1001),replace=True)
 for j,ix in enumerate(chosen):
  end=ends[ix]-1;seq=ids[max(starts[ix],end-20):end];hist[j,:len(seq)]=seq;lengths[j]=len(seq);candidates[j,0]=ids[end]
 if not reuse_assets:
  np.savez_compressed(target/"relations.npz",centers=centers,assignments=q,counts_by_fold=counts,
                      transition=T,marginal=p,fit_item_ids=fit_ids)
 proto_meta=dict(version="transition_dev_probe_v1",source_embedding=str(paths["embedding"]),
  source_embedding_sha256=before["embedding"],fit_source="distinct item IDs in raw Train only; no frequency weighting",
  fit_train_sha256=before["train"],fit_item_ids_sha256=hashlib.sha256(fit_ids.tobytes()).hexdigest(),
  dimensions=RANK,M=M,assignment_temperature=TAU,unit_normalize_before_fit=True,
  kmeans=dict(seed=42,batch_size=1024,n_init=3,max_iter=100),
  smoothing_beta=BETA,marginal_epsilon_pseudocount=1e-8,probability_epsilon=EPS,
  source_is_not_old_SAIT=True,relations_sha256=sha(target/"relations.npz"))
 if not reuse_assets:write_json(target/"prototype_manifest.json",proto_meta)
 else:assert proto_meta==prior_manifest
 relation_seconds=time.perf_counter()-fit_start
 with paths["cache"].open("rb") as f:corpus=pickle.load(f)
 assert corpus.n_items==n
 corpus.data_df={"dev":corpus.data_df["dev"]}
 args=arguments(ds,corpus);model=Model(args,corpus).to(args.device)
 model.load_state_dict(torch.load(paths["checkpoint"],map_location=args.device,weights_only=True),strict=True)
 model.eval();assert not model.training
 qg=torch.as_tensor(q,device=args.device);zg=torch.as_tensor(z,device=args.device)
 Tg=torch.as_tensor(T,dtype=torch.float32,device=args.device)
 pg=torch.as_tensor(p,dtype=torch.float32,device=args.device);popg=torch.as_tensor(pop,dtype=torch.float32,device=args.device)
 torch.cuda.reset_peak_memory_stats()
 sums={"base":0.,**{k:0. for k in BRANCHES}};calib_signals={k:[] for k in BRANCHES}
 calib_ranks={k:[] for k in BRANCHES};calib_base=[];calibration_start=time.perf_counter()
 with torch.inference_mode():
  for off in range(0,len(chosen),256):
   take=slice(off,off+256);cpu=dict(user_id=torch.from_numpy(users[take]),history_items=torch.from_numpy(hist[take]),
    lengths=torch.from_numpy(lengths[take]),item_id=torch.from_numpy(candidates[take]))
   feed=utils.batch_to_gpu(cpu,args.device);base=model(feed)["prediction"].cpu().numpy().astype(np.float64)
   sums["base"]+=float(np.square(base-base.mean(1,keepdims=True)).sum());calib_base.append(base)
   sg={k:np.empty_like(base) for k in BRANCHES}
   for f in range(3):
    use=feed["user_id"]%3==f
    if not bool(use.any()):continue
    ft,fp,_=fold_tables[f]
    ff={k:v[use] for k,v in feed.items()}
    values=signal_batch(ff,qg,zg,torch.tensor(ft,dtype=torch.float32,device=args.device),
      torch.tensor(fp,dtype=torch.float32,device=args.device),torch.tensor(pop_fold[f],dtype=torch.float32,device=args.device))
    loc=use.cpu().numpy()
    for k in BRANCHES:sg[k][loc]=values[k].cpu().numpy()
   for k in BRANCHES:
    sums[k]+=float(np.square(sg[k]-sg[k].mean(1,keepdims=True)).sum());calib_signals[k].append(sg[k]);calib_ranks[k].append(ranks(sg[k]))
 elements=len(chosen)*1001;sigmas={k:float(np.sqrt(v/elements)) for k,v in sums.items()}
 assert all(v>1e-8 for v in sigmas.values())
 scales={k:sigmas["base"]/sigmas[k] for k in BRANCHES}
 oof={}
 for k in BRANCHES:
  v=np.concatenate(calib_signals[k]);neg=v[:,1:];pos=v[:,0]
  oof[k]=dict(signal=stats(v),positive_mean=float(pos.mean()),negative_mean=float(neg.mean()),
    positive_over_random_AUC=float(((pos[:,None]>neg)+.5*(pos[:,None]==neg)).mean()),
    ndcg5=float(((np.concatenate(calib_ranks[k])<=5)/np.log2(np.concatenate(calib_ranks[k])+1)).mean()))
 tran=np.concatenate(calib_signals["P2_transition"]);proto_pop=np.concatenate(calib_signals["P3_prototype_popularity"])
 item_pop=np.concatenate(calib_signals["P3_item_popularity"])
 relation["OOF_diagnostic"]=oof
 relation["OOF_transition_correlation_proto_pop"]=float(np.corrcoef(tran.ravel(),proto_pop.ravel())[0,1])
 relation["OOF_transition_correlation_item_pop"]=float(np.corrcoef(tran.ravel(),item_pop.ravel())[0,1])
 relation["OOF_definition"]="1024 fixed sampled users at last Train target; all source/target transitions and marginals from user's fold excluded; candidate pool fixed Train catalog, duplicates retained; descriptive diagnostic, not formal Dev metric"
 calibration_seconds=time.perf_counter()-calibration_start
 del calib_signals,calib_base,calib_ranks,tran,proto_pop,item_pop,qleft
 dataset=model.Dataset(model,corpus,"dev")
 assert len(dataset)>0 and all(len(x)==1000 for x in dataset.data["neg_items"])
 assert len(np.unique(dataset.data["user_id"]))==len(dataset)
 loader=DataLoader(dataset,batch_size=256,shuffle=False,num_workers=0,collate_fn=dataset.collate_batch)
 definitions=[("B0",0.)]+[(k,a) for k in BRANCHES for a in LAMBDAS]
 keys=[f"{k}:{a:g}" for k,a in definitions];acc={key:{} for key in keys}
 all_ranks={key:[] for key in keys};all_users=[];inputhash=hashlib.sha256()
 rms_num={k:0. for k in BRANCHES};rms_den=0.;count=0;dev_start=time.perf_counter()
 print(json.dumps(dict(dataset=ds,stage="dev_start",relation_seconds=relation_seconds,scales=scales)),flush=True)
 with torch.inference_mode():
  for batch in loader:
   for name in ["user_id","history_items","lengths","item_id"]:
    ar=batch[name].numpy();inputhash.update(name.encode());inputhash.update(str(ar.shape).encode());inputhash.update(ar.tobytes())
   all_users.append(batch["user_id"].numpy());feed=utils.batch_to_gpu(batch,args.device)
   base=model(feed)["prediction"].cpu().numpy().astype(np.float64)
   vals=signal_batch(feed,qg,zg,Tg,pg,popg);raw={k:v.cpu().numpy().astype(np.float64) for k,v in vals.items()}
   centered={k:v-v.mean(1,keepdims=True) for k,v in raw.items()}
   rms_den+=float(np.square(base-base.mean(1,keepdims=True)).sum())
   for k in BRANCHES:rms_num[k]+=float(np.square(scales[k]*centered[k]).sum())
   for (k,a),key in zip(definitions,keys):
    pred=base if k=="B0" else base+a*scales[k]*centered[k]
    assert np.isfinite(pred).all()
    got=metric(pred)
    for name,val in got.items():acc[key][name]=acc[key].get(name,0.)+len(base)*val
    all_ranks[key].append(ranks(pred))
   count+=len(base)
 rows=[]
 for (k,a),key in zip(definitions,keys):
  rows.append(dict(branch=k,lambda_normalized=a,raw_lambda=0. if k=="B0" else a*scales[k],
   **{name:v/count for name,v in acc[key].items()}))
 baseline=rows[0]
 assert abs(baseline["NDCG@5"]-s["dev_n5"])<.000051,("Baseline mismatch",ds,baseline)
 rank_arrays={key:np.concatenate(v) for key,v in all_ranks.items()}
 base_nd=(rank_arrays[keys[0]]<=5)/np.log2(rank_arrays[keys[0]]+1)
 for row,key in zip(rows,keys):
  for name in tuple(baseline):
   if name.startswith(("HR@","NDCG@")):row["delta_"+name]=row[name]-baseline[name]
  rr=rank_arrays[key];nd=(rr<=5)/np.log2(rr+1);d=nd-base_nd
  row["relative_NDCG@5_pct"]=100*row["delta_NDCG@5"]/baseline["NDCG@5"]
  row["paired_SE_NDCG5_vs_B0"]=float(d.std(ddof=1)/np.sqrt(len(d)))
  row["NDCG5_win_loss_tie"]=[int((d>0).sum()),int((d<0).sum()),int((d==0).sum())]
  row["dev_correction_centered_RMS_ratio"]=0. if row["branch"]=="B0" else row["lambda_normalized"]*np.sqrt(rms_num[row["branch"]]/rms_den)
 best={k:max((r for r in rows if r["branch"]==k),key=lambda r:(r["NDCG@5"],-r["lambda_normalized"])) for k in BRANCHES}
 comparisons={}
 chosen_p2=best["P2_transition"];pk=f'P2_transition:{chosen_p2["lambda_normalized"]:g}'
 for k in ["B0","P1_semantic","P3_prototype_popularity","P3_item_popularity"]:
  control=baseline if k=="B0" else best[k];ck=keys[0] if k=="B0" else f'{k}:{control["lambda_normalized"]:g}'
  a=rank_arrays[pk];b=rank_arrays[ck]
  d=(a<=5)/np.log2(a+1)-(b<=5)/np.log2(b+1)
  comparisons[k]=dict(delta_NDCG5=float(d.mean()),paired_SE=float(d.std(ddof=1)/np.sqrt(len(d))),
   note="descriptive paired difference after Dev coefficient selection, not independent significance")
 result=dict(dataset=ds,phase="dev",rows=count,candidates=1001,duplicates_preserved=True,
  train_model_updates=0,test_executed=False,additional_learned_parameters=0,
  checkpoint=str(paths["checkpoint"]),embedding=str(paths["embedding"]),input_sha256=before,
  source_sha256=source_hash,dev_inputs_sha256=inputhash.hexdigest(),
  config={k:str(v) if k=="device" else v for k,v in vars(args).items()},
  parameters=model.count_variables(),protocol=dict(M=M,semantic_rank=RANK,tau=TAU,beta=BETA,
   lambdas=LAMBDAS,scale_fit="Train user-fold-out centered RMS; frozen before Dev",
   popularity_control="prototype log(p @ q_j); extra direct item Train log(count_j+1)",
   P1="cosine of last history item and candidate in actual frozen first512 semantic space",
   P2="log(q_last @ T @ q_candidate + 1e-12) - log(p @ q_candidate + 1e-12)"),
  prototype_manifest=proto_meta,relation_diagnostics=relation,
  train_scale_RMS=sigmas,scale_factors=scales,metrics=rows,best_by_Dev_NDCG5=best,
  selected_P2_comparisons=comparisons,seconds=dict(relation=relation_seconds,calibration=calibration_seconds,
    dev=time.perf_counter()-dev_start,total=time.perf_counter()-start),
  peak_GPU_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
  scope="Frozen checkpoint scoring feasibility probe, not a final Chapter4 algorithm. OOF diagnostics exclude own-user Train targets. Dev uses full Train statistics.")
 assert all(sha(p)==before[k] for k,p in paths.items())
 assert all(sha(ROOT/name)==h for name,h in source_hash.items())
 result["protected_inputs_unchanged"]=True
 write_json(target/"result.json",result)
 with (target/"metrics.csv").open("x",newline="") as f:
  wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
 with (target/"dev_user_ranks.csv").open("x",newline="") as f:
  wr=csv.writer(f);wr.writerow(["user_id"]+keys);wr.writerows(zip(np.concatenate(all_users),*[rank_arrays[k] for k in keys]))
 print(json.dumps(dict(dataset=ds,stage="complete",baseline=baseline,best=best,comparisons=comparisons,seconds=result["seconds"])),flush=True)
 signal.alarm(0)
if __name__=="__main__":
 signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError("170-second per-dataset audit limit")))
 parser=argparse.ArgumentParser();parser.add_argument("--dataset",choices=SPEC,required=True);opts=parser.parse_args()
 assert OUT.is_relative_to(ROOT) and not (OUT/opts.dataset/"result.json").exists()
 run(opts.dataset)

