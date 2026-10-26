"""ONE recorded 128-user Dev pass per dataset; no Test. Within-V2 graph-off counterfactual included."""
import sys,json,time,signal,argparse,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch,numpy as np,pandas as pd
from torch.utils.data import DataLoader
from gict_data import args_from_config,train_corpus,sha
from models.sequential.SGITRecV2 import SGITRecV2
from models.sequential.LLMMIRecASPCFAdaptiveSIL import LLMMIRecASPCFAdaptiveSIL as C1
from utils import utils
OUT=ROOT/'research/sgitrec_v3_20261025_v1'
def ranks(x):
 order=(-x).argsort(axis=1);r=np.empty_like(order)
 np.put_along_axis(r,order,np.broadcast_to(np.arange(1,x.shape[1]+1),x.shape),axis=1)
 return r
def n5(r):return (r<=5)/np.log2(r+1)
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['beauty','ml-1m'],required=True);x=p.parse_args()
 signal.alarm(170);torch.set_num_threads(4);tick=time.perf_counter()
 mark=OUT/'analysis'/(x.dataset+'_DEV_STARTED.json')
 with mark.open('x') as f:json.dump(dict(dataset=x.dataset,samples=128,phase='Dev',once=True,CPU_threads=4),f)
 old=json.loads((ROOT/'research/sgitrec_v2_20261025_v1/PLAN.json').read_text())['datasets'][x.dataset]
 a=args_from_config(old['config']);a.device=torch.device('cpu')
 a.sgit_graph_path=str(ROOT/'research/sgitrec_v2_20261025_v1/graphs'/(x.dataset+'_bridge_v2.pt'))
 c=train_corpus(a)
 frame=pd.read_csv(Path(a.path)/x.dataset/'dev.csv',sep='\t')
 selection=np.linspace(0,len(frame)-1,128,dtype=int)
 frame=utils.eval_list_columns(frame.iloc[selection].copy().reset_index(drop=True))
 frame['position']=frame.user_id.map(c.data_df['train'].groupby('user_id').size()).astype(np.int64)
 assert frame.user_id.is_unique and all(len(v)==1000 for v in frame.neg_items)
 c.data_df['dev']=frame
 c1path=ROOT/'research/sgitrec_round1_20261024_v1/runs'/('%s_k%d_s42_B0'%(x.dataset,old['K']))/'best.pt'
 v2path=ROOT/'research/sgitrec_v2_20261025_v1/runs'/('%s_k%d_s42_V2'%(x.dataset,old['K']))/'best.pt'
 base=C1(a,c).eval();base.load_state_dict(torch.load(c1path,map_location='cpu',weights_only=True),strict=True)
 model=SGITRecV2(a,c).eval();model.load_state_dict(torch.load(v2path,map_location='cpu',weights_only=True),strict=True)
 data=model.Dataset(model,c,'dev');data.prepare()
 graph=torch.load(a.sgit_graph_path,map_location='cpu',weights_only=True)
 counts=np.bincount(c.data_df['train'].item_id,minlength=c.n_items);q50,q90=np.quantile(counts[counts>0],[.5,.9])
 chunks={k:[] for k in ['candidate_ids','users','history','lengths','raw_C1','raw_V2','raw_off','prior_C1','prior_V2','prior_off','direct','bridge','support']}
 hsh=hashlib.sha256()
 with torch.inference_mode():
  for feed in DataLoader(data,batch_size=8,shuffle=False,num_workers=0,collate_fn=data.collate_batch):
   aout=base(feed,True);bout=model(feed,True)
   off=(bout['c1_interest_vectors']*bout['interest_weights'][...,None]).sum(1)
   offraw=(off[:,None]*bout['candidate_vectors']).sum(-1)
   arrays=dict(raw_C1=aout['raw_prediction'].numpy(),raw_V2=bout['raw_prediction'].numpy(),raw_off=offraw.numpy(),
    prior_C1=aout['prediction'].numpy(),prior_V2=bout['prediction'].numpy(),prior_off=model.corrected_scores(offraw,feed['item_id']).numpy())
   ids=feed['item_id'].numpy();hist=feed['history_items'].numpy();users=feed['user_id'].numpy();lengths=feed['lengths'].numpy()
   for k in ['user_id','history_items','lengths','item_id']:
    ar=feed[k].numpy();hsh.update(k.encode());hsh.update(str(ar.shape).encode());hsh.update(ar.tobytes())
   direct=np.zeros_like(ids,dtype=bool);bridge=np.zeros_like(ids,dtype=bool);support=np.zeros_like(ids,dtype=np.int32)
   for row,u in enumerate(users):
    f=int(u)%3;h=hist[row,:lengths[row]]
    ti=graph['transition_ids'][f,h].numpy().reshape(-1);ts=graph['transition_support'][f,h].numpy().reshape(-1)
    bi=graph['bridge_ids'][f,h].numpy().reshape(-1);bs=graph['bridge_support'][f,h].numpy().reshape(-1)
    direct[row]=np.isin(ids[row],ti[ti>0]);bridge[row]=np.isin(ids[row],bi[bi>0])
    lookup={}
    for d,s in zip(np.r_[ti,bi],np.r_[ts,bs]):
     if d>0:lookup[int(d)]=max(lookup.get(int(d),0),int(s))
    support[row]=[lookup.get(int(d),0) for d in ids[row]]
   padded_history=np.zeros((len(hist),a.history_max),np.int64);padded_history[:,:hist.shape[1]]=hist
   arrays.update(candidate_ids=ids,users=users,history=padded_history,lengths=lengths,direct=direct,bridge=bridge,support=support)
   for k,v in arrays.items():chunks[k].append(v)
 z={k:np.concatenate(v) for k,v in chunks.items()}
 rs={k:ranks(z[k]) for k in ['raw_C1','raw_V2','raw_off','prior_C1','prior_V2','prior_off']}
 neg=(np.arange(1001)[None]>0)&(z['candidate_ids']!=z['candidate_ids'][:,:1])
 reached=z['direct']|z['bridge'];delta=z['raw_V2']-z['raw_off'];shift=rs['prior_off']-rs['prior_V2']
 def group(mask):
  n=int(mask.sum())
  return dict(slots=n,mean_delta_score=float(delta[mask].mean()) if n else None,
   mean_abs_delta_score=float(np.abs(delta[mask]).mean()) if n else None,
   score_changed_fraction=float((np.abs(delta[mask])>1e-6).mean()) if n else None,
   mean_rank_promotion=float(shift[mask].mean()) if n else None,
   rank_promoted_fraction=float((shift[mask]>0).mean()) if n else None,
   top5_before=int(((rs['prior_off']<=5)&mask).sum()),top5_after=int(((rs['prior_V2']<=5)&mask).sum()))
 popularity={}
 freq=counts[z['candidate_ids']]
 for name,mask in [('tail',freq<=q50),('middle',(freq>q50)&(freq<=q90)),('head',freq>q90)]:
  popularity[name]=group(mask&neg)
 bypath={}
 for name,mask in [('direct',z['direct'][:,0]),('bridge_only',z['bridge'][:,0]&~z['direct'][:,0]),('unreachable',~reached[:,0])]:
  bypath[name]=dict(users=int(mask.sum()))
  for scheme in ['prior_C1','prior_off','prior_V2']:
   bypath[name][scheme]=dict(NDCG5=float(n5(rs[scheme][mask,0]).mean()) if mask.any() else None,
                            HR5=float((rs[scheme][mask,0]<=5).mean()) if mask.any() else None)
 support={}
 for name,mask in [('low_1',z['support']==1),('mid_2to4',(z['support']>=2)&(z['support']<=4)),('high_ge5',z['support']>=5)]:
  support[name]=group(mask&neg&reached)
 path=OUT/'analysis'/(x.dataset+'_DEV_arrays.npz')
 with path.open('xb') as f:np.savez_compressed(f,**z)
 result=dict(dataset=x.dataset,samples=128,selection='evenly spaced Dev row indices independent of labels',sample_indices=selection.tolist(),
  candidates=1001,duplicates_preserved=True,phase='Dev_only',Test=False,full_Dev_rerun=False,
  C1_checkpoint_sha256=sha(c1path),V2_checkpoint_sha256=sha(v2path),graph_sha256=sha(a.sgit_graph_path),input_hash=hsh.hexdigest(),
  arrays=str(path),arrays_sha256=sha(path),by_positive_reachability=bypath,
  unreachable_wrong_candidates=group(neg&~reached),reachable_wrong_candidates=group(neg&reached),
  popularity_thresholds_Train_only=[float(q50),float(q90)],wrong_candidate_popularity=popularity,support_groups=support,
  distinction='C1 vs V2 includes backbone training differences; V2 vs within-model graph-off isolates residual at fixed backbone',
  aggregation_source_fact='V2 interest_weights from history-only aggregator, then sum_k w_k(v4_k+delta_k) before dot candidate; no candidate-conditioned graph path choice',
  seconds=time.perf_counter()-tick)
 with (OUT/'analysis'/(x.dataset+'_DEV_diagnosis.json')).open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
