"""Train-only fold-specific target-lift and RMS calibration of immutable directed Top8 tables."""
import sys,json,time,signal,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch,numpy as np
from gict_data import args_from_config,train_corpus,sha,project_output
from models.sequential.LLMMIRecASPCFAdaptiveSIL import array_sha
def build_index(base,histories):
 F,N,R=base['transition_ids'].shape
 out={};stats=[];marginals=[];sources=[];rms=[]
 for f in range(F):
  target=np.zeros(N,np.float64);source=np.zeros(N,np.float64)
  for u,seq in histories.items():
   if u%F==f:continue
   ids=np.array([x[0] for x in seq],np.int64)
   if len(ids)>1:
    source+=np.bincount(ids[:-1],minlength=N);target+=np.bincount(ids[1:],minlength=N)
  p=(target+.5)/(target.sum()+.5*max(N-1,1));p[0]=0
  marginals.append(torch.tensor(p,dtype=torch.float32));sources.append(torch.tensor(source,dtype=torch.float32))
  rr=[];summary={}
  for rel in ['transition','bridge']:
   ids=base[rel+'_ids'][f];prob=base[rel+'_prob'][f].double().numpy()
   valid=ids.numpy()>0
   lift=np.log1p(prob/np.maximum(p[ids.numpy()],1e-12))*valid
   weights=source[:,None]*valid
   scale=float(np.sqrt((weights*lift**2).sum()/max(weights.sum(),1)))
   if not np.isfinite(scale) or scale<=1e-12:scale=1.
   value=torch.tensor(lift/scale,dtype=torch.float32)
   rr.append(scale);out.setdefault(rel+'_value',[]).append(value)
   nz=prob[valid];cal=value.numpy()[valid];pop=p[ids.numpy()[valid]]
   summary[rel]=dict(edges=int(valid.sum()),prob_mean=float(nz.mean()) if len(nz) else 0.,
    probability_median=float(np.median(nz)) if len(nz) else 0.,RMS=scale,
    Train_weighted_normalized_RMS=float(np.sqrt((weights*(lift/scale)**2).sum()/max(weights.sum(),1))),
    value_mean=float(cal.mean()) if len(cal) else 0.,corr_probability_target_pop=float(np.corrcoef(nz,pop)[0,1]) if len(nz)>1 else None,
    corr_value_target_pop=float(np.corrcoef(cal,pop)[0,1]) if len(nz)>1 else None)
  rms.append(rr);stats.append(dict(fold=f,relations=summary,target_events=int(target.sum())))
 for key,value in base.items():
  if key.startswith(('transition_','bridge_')) and key.split('_',1)[1] in ['ids','prob','support']:
   out[key]=value.clone()
 for rel in ['transition','bridge']:out[rel+'_value']=torch.stack(out[rel+'_value'])
 out['target_marginal']=torch.stack(marginals);out['source_events']=torch.stack(sources)
 out['relation_rms']=torch.tensor(rms,dtype=torch.float32)
 return out,stats
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--base-graph',required=True);p.add_argument('--output',required=True)
 x=p.parse_args();signal.alarm(170);torch.set_num_threads(4);tick=time.perf_counter()
 target=project_output(x.output);a=args_from_config(x.config);c=train_corpus(a)
 base=torch.load(x.base_graph,map_location='cpu',weights_only=True);meta=base['metadata']
 assert meta['schema']=='sgit_semantic_bridge_v2' and meta['scope']=='full_crossfit_train'
 assert meta['train_frame_sha256']==array_sha(c.data_df['train'][['user_id','item_id','position']].to_numpy(np.int64))
 assert meta['embedding_sha256']==sha(a.llm_emb_path) and meta['folds']==3 and meta['R']==8
 tensors,stats=build_index(base,c.user_his)
 metadata=dict(meta,schema='sgit_candidate_index_v3',base_graph_sha256=sha(x.base_graph),
  calibration='log1p(prob / smoothed target marginal) / separate relation Train-source-weighted RMS',
  smoothing=.5,calibration_scope='each fold uses ONLY other folds Train source and target events; no own user contribution or held-out input',
  per_fold_calibration=stats,sources={str(ROOT/p):sha(ROOT/p) for p in ['tools/build_sgit_v3_index.py','tools/gict_data.py']},
  tensor_bytes=sum(t.numel()*t.element_size() for t in tensors.values()))
 with target.open('xb') as f:torch.save(dict(metadata=metadata,**tensors),f)
 result=dict(dataset=a.dataset,path=str(target),sha256=sha(target),tensor_bytes=metadata['tensor_bytes'],
  per_fold=stats,seconds=time.perf_counter()-tick,Train_only=True,Dev=False,Test=False)
 with Path(str(target)+'.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
