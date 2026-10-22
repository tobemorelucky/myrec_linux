"""Bounded Train-only semantic/behavior association; no Dev/Test/cache/model imports."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
os.environ['CUDA_VISIBLE_DEVICES']=''
import pathlib,json,pickle,hashlib,time,signal,csv,datetime
import numpy as np
import pandas as pd
from scipy import sparse
ROOT=pathlib.Path('/home/yyx/hzgcode/pom2.0')
W=ROOT/'research/ml1m_embedding_v2_20261020'
OUT=W/'runs/cross_v3_20261021/p920/behavior_train_only'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def deadline(*args):raise TimeoutError('Train-only CPU analysis exceeded 175 seconds')
signal.signal(signal.SIGALRM,deadline);signal.alarm(175)
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def main():
 started=time.monotonic();os.chdir(ROOT);assert OUT.resolve().is_relative_to(W.resolve())
 OUT.mkdir(exist_ok=False)
 train=ROOT/'data/ml-1m/train.csv';df=pd.read_csv(train,sep='\t',usecols=['user_id','item_id','time'])
 assert not df.isna().any().any()
 n=3707;df=df.sort_values(['user_id','time'],kind='stable')
 item=df.item_id.to_numpy(dtype=np.int64);user=df.user_id.to_numpy(dtype=np.int64)
 assert item.min()>0 and item.max()<n
 users,uid=np.unique(user,return_inverse=True)
 incidence=sparse.coo_matrix((np.ones(len(df),dtype=np.float64),(uid,item)),shape=(len(users),n)).tocsr()
 incidence.data[:]=1;incidence.eliminate_zeros()
 degree=np.asarray(incidence.sum(axis=0)).ravel()
 pop=np.bincount(item,minlength=n).astype(float)
 valid=user[:-1]==user[1:];src=item[:-1][valid];dst=item[1:][valid]
 transition=sparse.coo_matrix((np.ones(len(src)),(src,dst)),shape=(n,n)).tocsr()
 outdeg=np.asarray(transition.sum(axis=1)).ravel();indeg=np.asarray(transition.sum(axis=0)).ravel()
 catalog=np.flatnonzero(pop>0);queries=np.sort(np.random.RandomState(20261021).choice(catalog,512,replace=False))
 co=(incidence[:,queries].T@incidence).tocsr()
 # Fixed controls shared by V1/V2; each semantic target gets one random target in same popularity decile.
 ranked=np.empty(n,dtype=int);ranked[:]=-1
 order=catalog[np.argsort(pop[catalog],kind='stable')]
 ranked[order]=np.minimum(np.arange(len(order))*10//len(order),9)
 bins=[catalog[ranked[catalog]==i] for i in range(10)]
 uniform=np.empty((len(queries),10),dtype=int);r=np.random.RandomState(20261022)
 for i,q in enumerate(queries):uniform[i]=r.choice(catalog[catalog!=q],10,replace=False)
 metric_names=['directed_transition_present','directed_transition_count','transition_out_probability','transition_degree_cosine','transition_lift','co_user_count','co_user_cosine','co_user_lift','target_train_frequency']
 def metrics(target):
  qi=np.repeat(np.arange(len(queries)),10);q=np.repeat(queries,10);t=target.reshape(-1)
  tc=np.asarray(transition[q,t]).reshape(-1);cc=np.asarray(co[qi,t]).reshape(-1)
  return np.column_stack([tc>0,tc,tc/np.maximum(outdeg[q],1),tc/np.sqrt(np.maximum(outdeg[q]*indeg[t],1)),tc*len(src)/np.maximum(outdeg[q]*indeg[t],1),cc,cc/np.sqrt(np.maximum(degree[q]*degree[t],1)),cc*len(users)/np.maximum(degree[q]*degree[t],1),pop[t]]).reshape(len(queries),10,-1)
 inputs={
 'V1':('imported_assets/v1_rebuilt_20261020/base_pca.pkl','5546240cc16beff4666a74a18cb758f41dbf23f21f46a9082778193501c59aad'),
 'V2':('imported_assets/v2_rich512_l2_d400_20261021/rich_pca.pkl','e9e7f7d7fd2cdbf23c79102842826275e785314413f08e7e08ab28c557eee3eb')}
 result=dict(start=datetime.datetime.now().astimezone().isoformat(),stage='Train only',train_path=str(train),train_sha256=sha(train),train_rows=len(df),train_users=len(users),train_catalog=len(catalog),directed_adjacent_transitions=len(src),sample_queries=len(queries),seed_queries=20261021,seed_uniform=20261022,seed_popularity_matched=20261023,candidate_catalog='Train positive catalog only; self excluded',tie_handling='stable original train row order for tied user timestamps',topk=10,controls='one fixed uniform set shared across embeddings, plus five fixed popularity-decile matched target sets per embedding',formulas=dict(transition_degree_cosine='count(i->j)/sqrt(out_transition_count(i)*in_transition_count(j))',transition_lift='count(i->j)*all_transitions/(out_transition_count(i)*in_transition_count(j))',co_user_cosine='shared_users(i,j)/sqrt(user_degree(i)*user_degree(j))',co_user_lift='shared_users(i,j)*n_users/(user_degree(i)*user_degree(j))'),versions={})
 qavg={};csvrows=[]
 uniform_values=metrics(uniform)
 for label,(rel,expected) in inputs.items():
  p=W/rel;assert sha(p)==expected
  with p.open('rb') as f:table=pickle.load(f)
  assert table.shape==(n,1536) and table.dtype==np.float32 and np.isfinite(table).all() and not table[0].any()
  emb=table[catalog].copy();norm=np.linalg.norm(emb,axis=1);assert np.all(norm>0);emb/=norm[:,None]
  lookup=np.full(n,-1);lookup[catalog]=np.arange(len(catalog))
  similarity=emb[lookup[queries]]@emb.T
  similarity[np.arange(len(queries)),lookup[queries]]=-np.inf
  ids=np.argpartition(-similarity,9,axis=1)[:,:10]
  scores=np.take_along_axis(similarity,ids,axis=1);sort=np.argsort(-scores,axis=1,kind='stable')
  ids=np.take_along_axis(ids,sort,axis=1);scores=np.take_along_axis(scores,sort,axis=1);neighbors=catalog[ids]
  sem=metrics(neighbors);matched=[]
  for repeat in range(5):
   r=np.random.RandomState(20261023+repeat);targets=np.empty_like(neighbors)
   for i,q in enumerate(queries):
    for k,t in enumerate(neighbors[i]):
     choices=bins[ranked[t]];choices=choices[choices!=q]
     targets[i,k]=r.choice(choices)
   matched.append(metrics(targets))
  matched=np.mean(matched,axis=0)
  qavg[label]=dict(semantic=sem.mean(1),uniform=uniform_values.mean(1),popularity_matched=matched.mean(1))
  record=dict(path=str(p),sha256=expected,shape=list(table.shape),mean_top10_cosine=float(scores.mean()))
  for control,value in qavg[label].items():record[control]={m:float(v) for m,v in zip(metric_names,value.mean(0))}
  record['semantic_minus_popularity_matched']={m:float(v) for m,v in zip(metric_names,(qavg[label]['semantic']-qavg[label]['popularity_matched']).mean(0))}
  result['versions'][label]=record
  for i,q in enumerate(queries):
   for k,t in enumerate(neighbors[i]):
    csvrows.append(dict(version=label,query=int(q),rank=k+1,neighbor=int(t),cosine=float(scores[i,k]),query_train_frequency=int(pop[q]),**{m:float(v) for m,v in zip(metric_names,sem[i,k])}))
 del table,emb,similarity
 r=np.random.RandomState(20261028);boot=r.randint(0,len(queries),(1000,len(queries)))
 result['V2_minus_V1']={}
 for control in ['semantic','popularity_excess']:
  delta=qavg['V2']['semantic']-qavg['V1']['semantic']
  if control=='popularity_excess':delta-=qavg['V2']['popularity_matched']-qavg['V1']['popularity_matched']
  result['V2_minus_V1'][control]={m:dict(mean=float(delta[:,j].mean()),query_bootstrap_95CI=[float(x) for x in np.quantile(delta[boot,j].mean(1),[.025,.975])]) for j,m in enumerate(metric_names)}
 result['elapsed_seconds']=time.monotonic()-started
 result['limitations']=['Only Train structure association, not independent predictive evidence; no Dev/Test labels read.','512 fixed uniformly sampled Train-catalog queries; 10 semantic cosine neighbors, not an all-item benchmark.','Adjacency preserves timestamp ties in CSV order, and is not proof of causal or next-item generalization.','Degree/popularity adjustment reduces but cannot remove confounding; popularity-decile matching remains approximate.','Query bootstrap intervals are descriptive; shared users/items and repeated-neighbor dependencies are not independent.','No retraining, GPU, cache, corpus, checkpoint, or original-asset writes.']
 save(OUT/'RESULTS.json',result);save(OUT/'QUERY_IDS.json',queries.tolist())
 with (OUT/'SEMANTIC_PAIRS.csv').open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
 fields=['version','control']+metric_names
 with (OUT/'SUMMARY.csv').open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for label,rec in result['versions'].items():
   for control in ['semantic','uniform','popularity_matched','semantic_minus_popularity_matched']:writer.writerow(dict(version=label,control=control,**rec[control]))
 lines=['# Train-only semantic-neighbor / behavior association','',f'CPU elapsed {result["elapsed_seconds"]:.3f}s; Train rows {len(df)}, users {len(users)}, catalog {len(catalog)}; fixed query sample 512; top10. No Dev/Test labels read.','','|Embedding/control|Adjacent transition present|Transition degree cosine|Transition lift|Co-user cosine|Co-user lift|Target frequency|','|---|---:|---:|---:|---:|---:|---:|']
 for label,rec in result['versions'].items():
  for control in ['semantic','uniform','popularity_matched','semantic_minus_popularity_matched']:
   s=rec[control];lines.append('|'+label+'/'+control+'|'+'|'.join(f'{s[k]:.6g}' for k in ['directed_transition_present','transition_degree_cosine','transition_lift','co_user_cosine','co_user_lift','target_train_frequency'])+'|')
 lines+=['','Pair CSV contains every sampled semantic edge. JSON gives exact formulas, input SHA, fixed seeds, and paired query-bootstrap differences.','']+['- '+s for s in result['limitations']]
 (OUT/'RESULTS.md').write_text('\n'.join(lines))
 signal.alarm(0)
 print(json.dumps(result,allow_nan=False))
if __name__=='__main__':main()
