"""Offline T->S completion of immutable Train-fold graphs; independent path calibration."""
import sys,json,time,signal,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np,torch
from gict_data import args_from_config,train_corpus,sha,project_output
from models.sequential.LLMMIRecASPCFAdaptiveSIL import array_sha

def smooth_table(base):
 F,N,R=base['transition_ids'].shape
 arrays={k:np.zeros((F,N,R),dtype=t) for k,t in [('ids',np.int64),('prob',np.float32),('support',np.int32),('path_count',np.int32),('witness',np.int64)]}
 stats=[]
 for f in range(F):
  t=base['transition_ids'][f].numpy();tp=base['transition_prob'][f].numpy();sup=base['transition_support'][f].numpy()
  s=base['semantic_ids'][f].numpy();sp=base['semantic_prob'][f].numpy()
  for i in range(1,N):
   via=t[i];dest=s[via].reshape(-1)
   weight=(tp[i,:,None]*(sup[i,:,None]/(sup[i,:,None]+1.0))*sp[via]).reshape(-1)
   witness=np.repeat(via,R);supports=np.repeat(sup[i],R)
   allowed=(dest>0)&(witness>0)&(weight>0)
   if not allowed.any():continue
   dest,weight,witness,supports=[a[allowed] for a in [dest,weight,witness,supports]]
   ids,back=np.unique(dest,return_inverse=True);mass=np.bincount(back,weights=weight)
   support=np.zeros(len(ids),np.int32);np.maximum.at(support,back,supports)
   count=np.bincount(back);best=np.full(len(ids),-np.inf);np.maximum.at(best,back,weight)
   witnesses=np.full(len(ids),N,np.int64);pickbest=weight==best[back]
   np.minimum.at(witnesses,back[pickbest],witness[pickbest])
   pick=np.lexsort((ids,-mass))[:R];n=len(pick)
   for key,x in [('ids',ids),('prob',mass/mass.sum()),('support',support),('path_count',count),('witness',witnesses)]:
    arrays[key][f,i,:n]=x[pick]
  valid=arrays['ids'][f]>0
  direct=(arrays['ids'][f,:,:,None]==t[:,None,:]).any(-1)&valid
  st=(arrays['ids'][f,:,:,None]==base['bridge_ids'][f].numpy()[:,None,:]).any(-1)&valid
  stats.append(dict(fold=f,edges=int(valid.sum()),sources=int(valid.any(-1).sum()),extra_vs_direct=int((valid&~direct).sum()),extra_vs_ST=int((valid&~st).sum())))
 return {'smooth_'+k:torch.from_numpy(v) for k,v in arrays.items()},stats

def calibrate(smooth,index):
 values=[];scales=[];stats=[]
 for f in range(len(smooth['smooth_ids'])):
  ids=smooth['smooth_ids'][f].numpy();p=index['target_marginal'][f].double().numpy()
  source=index['source_events'][f].double().numpy()
  prob=smooth['smooth_prob'][f].double().numpy();valid=ids>0
  lift=np.log1p(prob/np.maximum(p[ids],1e-12))*valid
  weights=source[:,None]*valid
  rms=float(np.sqrt((weights*lift**2).sum()/max(weights.sum(),1)))
  if not np.isfinite(rms) or rms<=1e-12:rms=1.
  values.append(torch.tensor(lift/rms,dtype=torch.float32));scales.append(rms)
  stats.append(dict(fold=f,RMS=rms,Train_weighted_RMS=float(np.sqrt((weights*(lift/rms)**2).sum()/max(weights.sum(),1)))))
 smooth['smooth_value']=torch.stack(values);smooth['smooth_rms']=torch.tensor(scales)
 return smooth,stats

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--base-graph',required=True);p.add_argument('--index',required=True);p.add_argument('--output',required=True)
 x=p.parse_args();signal.alarm(170);torch.set_num_threads(4);tick=time.perf_counter()
 out=project_output(x.output);a=args_from_config(x.config);c=train_corpus(a)
 base=torch.load(x.base_graph,map_location='cpu',weights_only=True)
 index=torch.load(x.index,map_location='cpu',weights_only=True)
 meta=index['metadata']
 assert meta['scope']=='full_crossfit_train' and meta['schema']=='sgit_candidate_index_v3'
 assert meta['base_graph_sha256']==sha(x.base_graph)
 assert meta['train_frame_sha256']==array_sha(c.data_df['train'][['user_id','item_id','position']].to_numpy(np.int64))
 assert meta['embedding_sha256']==sha(a.llm_emb_path)
 assert meta['R']==8 and meta['folds']==3
 for rel in ['transition','bridge']:
  for name in ['ids','prob','support']:assert torch.equal(index[rel+'_'+name],base[rel+'_'+name])
 # The semantic targets really belong to each corresponding other-fold Train catalog.
 for f in range(3):
  eligible={int(i) for u,seq in c.user_his.items() if u%3!=f for i,_ in seq}
  used=set(base['semantic_ids'][f].flatten().tolist())-{0}
  assert used<=eligible
 smooth,stats=smooth_table(base);smooth,scales=calibrate(smooth,index)
 # Verify retained strongest witnesses have a real T edge followed by an S edge.
 for f in range(3):
  for source in np.flatnonzero((smooth['smooth_ids'][f].numpy()>0).any(-1)):
   ids=smooth['smooth_ids'][f,source];via=smooth['smooth_witness'][f,source];valid=ids>0
   assert ((base['transition_ids'][f,source,None,:]==via[valid,None]).any(-1)).all()
   assert ((base['semantic_ids'][f,via[valid]]==ids[valid,None]).any(-1)).all()
 payload={k:v for k,v in index.items() if k!='metadata'};payload.update(smooth)
 metadata=dict(meta,schema='sgit_candidate_predictions_v3',V3_index_sha256=sha(x.index),
  smooth_formula='sum_d Train_P(d|i)*support(i,d)/(support(i,d)+1)*semantic_P(j|d); aggregate, normalize full mass, retain Top8',
  smooth_scope='same other-fold Train-eligible semantic targets, no Dev/Test reads',
  support_policy='max constituent distinct-user support, NOT sum of independent users',
  smooth_stats=stats,smooth_calibration=scales,smooth_calibration_formula='same V3 log1p(prob/Train target marginal), independent Train source-weighted RMS',
  calibration_note='transition and bridge values reused bit-for-bit; TS marginal/source float32 tensors from existing V3 index',
  tensor_bytes=sum(t.numel()*t.element_size() for t in payload.values()))
 payload['metadata']=metadata
 with out.open('xb') as f:torch.save(payload,f)
 result=dict(passed=True,dataset=a.dataset,path=str(out),sha256=sha(out),bytes=out.stat().st_size,
  tensor_bytes=metadata['tensor_bytes'],seconds=time.perf_counter()-tick,TS=stats,calibration=scales,
  fold_Train_catalog_verified=True,direction_witnesses_verified=True,ST_direct_unchanged=True,Dev=False,Test=False)
 with Path(str(out)+'.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
