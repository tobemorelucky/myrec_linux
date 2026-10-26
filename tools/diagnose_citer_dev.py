"""One bounded, fixed-sample Dev diagnosis of legacy A/B/C1; never opens Test."""
import sys,os,json,time,signal,argparse,hashlib,importlib.util,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch,numpy as np,pandas as pd
from torch.utils.data import DataLoader
from gict_data import args_from_config,train_corpus,sha
from citer_dev_protocol import install_guard,load_dev
from models.sequential.SGITRecV3Controlled import SGITRecV3Controlled
from models.sequential.LLMMIRecASPCFAdaptiveSIL import LLMMIRecASPCFAdaptiveSIL as C1
from models.sequential.citer_components import path_statistics
from utils import utils
OUT=ROOT/'research/citerrec_v1_20261026_v1'
def correlation(a,b):
 a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
 return float(np.corrcoef(a,b)[0,1]) if len(a)>1 and a.std()>1e-12 and b.std()>1e-12 else None
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=['beauty','ml-1m']);p.add_argument('--gpu',type=int)
 x=p.parse_args();signal.alarm(170);torch.set_num_threads(4);install_guard();tick=time.perf_counter();lease=None;device='cpu'
 if x.gpu is not None:
  spec=importlib.util.spec_from_file_location('citer_diag_resource',ROOT/'research/elector_sil_objective_full_20261023_v2/scheduler.py')
  res=importlib.util.module_from_spec(spec);spec.loader.exec_module(res);inv=res.observe()
  assert res.idle(inv[x.gpu]);lease=res.acquire(ROOT/('research/gpu_resource_scheduler_20261022_v1/gpu%d.lock'%x.gpu));assert lease is not None
  assert res.idle(res.observe()[x.gpu])
  os.environ['CUDA_VISIBLE_DEVICES']=inv[x.gpu]['uuid'];cache=OUT/'analysis'/('cuda_'+x.dataset);cache.mkdir()
  os.environ['CUDA_CACHE_PATH']=str(cache);torch.cuda.set_device(0);device='cuda:0'
 try:
  old=ROOT/'research/sgitrec_v3_predictions_20261025_v1';plan=json.load(open(old/'PLAN.json'));ds=plan['datasets'][x.dataset]
  a=args_from_config(ds['configs']['42']['path']);a.device=torch.device(device);a.v3_graph_path=ds['graphs']['predictions']['path']
  c=train_corpus(a)
  count=sum(1 for _ in open(Path(a.path)/x.dataset/'dev.csv'))-1
  indices=np.linspace(0,count-1,min(count,1024),dtype=int)
  with (OUT/'analysis'/(x.dataset+'_DEV_STARTED.json')).open('x') as f:json.dump(dict(phase='Dev',samples=len(indices),indices=indices.tolist(),selection='evenly spaced rows independent of labels'),f)
  load_dev(c,a,indices)
  graph=torch.load(a.v3_graph_path,map_location='cpu',weights_only=True)
  gid=graph['transition_ids'].to(device);sup=graph['transition_support'].to(device)
  models={}
  paths={'C1':ROOT/'research/sgitrec_round1_20261024_v1/runs'/f'{x.dataset}_k{a.K}_s42_B0/best.pt'}
  for label,variant in [('A','TargetInterest'),('B','Direct-Fixed')]:paths[label]=old/'runs'/f'{x.dataset}_k{a.K}_s42_{variant}/best.pt'
  models['C1']=C1(a,c).to(device)
  for label,variant in [('A','TargetInterest'),('B','Direct-Fixed')]:
   a.v3_variant=variant;models[label]=SGITRecV3Controlled(a,c).to(device)
  for label,m in models.items():m.load_state_dict(torch.load(paths[label],map_location=device,weights_only=True),strict=True);m.eval()
  dataset=models['A'].Dataset(models['A'],c,'dev');dataset.prepare()
  chunks={k:[] for k in ['users','lengths','candidate_ids','support','path_fraction','entropy','dispersion','delta_A','delta_B','B_base_raw','raw_C1','prior_C1','raw_A','prior_A','raw_B','prior_B']}
  hasher=hashlib.sha256()
  with torch.inference_mode():
   for cpu in DataLoader(dataset,batch_size=64,shuffle=False,num_workers=0,collate_fn=dataset.collate_batch):
    for k in ['user_id','history_items','lengths','item_id']:
     z=cpu[k].numpy();hasher.update(k.encode());hasher.update(str(z.shape).encode());hasher.update(z.tobytes())
    f=utils.batch_to_gpu(cpu,torch.device(device));z={label:m(f,True) for label,m in models.items()}
    valid=(torch.arange(f['history_items'].shape[1],device=device)[None]<f['lengths'][:,None])&(f['history_items']>0)
    fold=f['user_id']%3
    edges=dict(ids=gid[fold[:,None],f['history_items']],support=sup[fold[:,None],f['history_items']])
    support,fraction=path_statistics(valid,edges,f['item_id'])
    alpha=z['A']['candidate_interest_weights'];entropy=-(alpha.clamp_min(1e-8)*alpha.clamp_min(1e-8).log()).sum(1)/math.log(a.K)
    dot=torch.einsum('bkd,bcd->bkc',z['A']['interest_vectors'],z['A']['candidate_vectors'])
    dispersion=dot.std(1,unbiased=False)/(dot.square().mean(1).sqrt()+1e-6)
    values=dict(users=f['user_id'],lengths=f['lengths'],candidate_ids=f['item_id'],support=support,path_fraction=fraction,
     entropy=entropy,dispersion=dispersion,delta_A=z['A']['candidate_graph_delta'],delta_B=z['B']['candidate_graph_delta'],
     B_base_raw=z['B']['raw_prediction']-z['B']['candidate_graph_delta'])
    for label,o in z.items():values['raw_'+label]=o['raw_prediction'];values['prior_'+label]=o['prediction']
    for k,v in values.items():chunks[k].append(v.detach().cpu().numpy())
  arrays={k:np.concatenate(v) for k,v in chunks.items()}
  ranks={label:np.argwhere((-arrays['prior_'+label]).argsort(axis=1)==0)[:,1]+1 for label in models}
  n5={label:(rank<=5)/np.log2(rank+1) for label,rank in ranks.items()}
  def group(mask):
   n=int(mask.sum());out=dict(users=n)
   for label in models:
    out[label]=dict(NDCG5=float(n5[label][mask].mean()) if n else None,HR5=float((ranks[label][mask]<=5).mean()) if n else None)
   if n:
    out.update(A_minus_C1=out['A']['NDCG5']-out['C1']['NDCG5'],B_minus_C1=out['B']['NDCG5']-out['C1']['NDCG5'],
     A_minus_B=out['A']['NDCG5']-out['B']['NDCG5'],mean_entropy=float(arrays['entropy'][mask,0].mean()),
     mean_support=float(arrays['support'][mask,0].mean()),mean_length=float(arrays['lengths'][mask].mean()))
   return out
  length=arrays['lengths'];support=arrays['support'][:,0];confidence=1-arrays['entropy'][:,0]
  counts=np.bincount(c.data_df['train'].item_id,minlength=c.n_items);q50,q90=np.quantile(counts[counts>0],[.5,.9])
  freq=counts[arrays['candidate_ids'][:,0]]
  groups=dict(length={n:group(m) for n,m in [('1to3',length<=3),('4to7',(length>=4)&(length<=7)),('8to19',(length>=8)&(length<20)),('20',length==20)]},
   path={n:group(m) for n,m in [('absent',support==0),('support1',support==1),('support2to4',(support>=2)&(support<=4)),('support5plus',support>=5)]},
   frequency={n:group(m) for n,m in [('tail',freq<=q50),('middle',(freq>q50)&(freq<=q90)),('head',freq>q90)]},
   confidence={n:group(m) for n,m in [('low_lt05',confidence<.05),('mid05to25',(confidence>=.05)&(confidence<.25)),('high25plus',confidence>=.25)]})
  regions={'A_preferred':[],'B_preferred':[]}
  for dimension in ['path','confidence','length']:
   for name,g in groups[dimension].items():
    if g['users']<40:continue
    if g['A_minus_C1']>0 and g['A_minus_B']>1e-4:regions['A_preferred'].append(dict(feature=dimension,region=name,**g))
    if g['B_minus_C1']>0 and g['A_minus_B']< -1e-4:regions['B_preferred'].append(dict(feature=dimension,region=name,**g))
  da=n5['A']-n5['C1'];db=n5['B']-n5['C1']
  reachable=arrays['support']>0
  neg=(np.arange(1001)[None]>0)&(arrays['candidate_ids']!=arrays['candidate_ids'][:,:1])
  def slot(mask):
   n=int(mask.sum())
   return dict(slots=n,mean_entropy=float(arrays['entropy'][mask].mean()) if n else None,mean_dispersion=float(arrays['dispersion'][mask].mean()) if n else None,
     B_delta_vs_B_base_correlation=correlation(arrays['delta_B'][mask],arrays['B_base_raw'][mask]),
     B_delta_vs_independent_C1_correlation=correlation(arrays['delta_B'][mask],arrays['raw_C1'][mask]),
     B_abs_delta_mean=float(np.abs(arrays['delta_B'][mask]).mean()) if n else None)
  path=OUT/'analysis'/(x.dataset+'_DEV_arrays.npz')
  with path.open('xb') as f:np.savez_compressed(f,**arrays)
  result=dict(passed=True,dataset=x.dataset,samples=len(indices),phase='Dev_only',input_hash=hasher.hexdigest(),
   overall=group(np.ones(len(length),bool)),groups=groups,observable_complementarity_regions=regions,
   sample_complementarity=bool(regions['A_preferred'] and regions['B_preferred']),
   outcome_overlap=dict(A_only_improves=int(((da>0)&(db<=0)).sum()),B_only_improves=int(((db>0)&(da<=0)).sum()),both_improve=int(((da>0)&(db>0)).sum()),both_worse=int(((da<0)&(db<0)).sum())),
   candidates=dict(all=slot(np.ones_like(reachable,bool)),reachable=slot(reachable),unreachable=slot(~reachable),wrong_reachable=slot(neg&reachable)),
   target_reachability=float(reachable[:,0].mean()),length20_fraction=float((length==20).mean()),
   Train_frequency_quantiles=[float(q50),float(q90)],checkpoints={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()},
   graph_sha256=sha(a.v3_graph_path),arrays=str(path),arrays_sha256=sha(path),seconds=time.perf_counter()-tick,
   limitations=['Sample exploratory, not population significance.','Independent trained backbones confound A/B versus C1; outcome overlap is not an achievable label-free oracle policy.','Positive-path grouping uses Dev labels offline only; gates never know which candidate is positive.','No ML-vs-Beauty length causality claim; inspect within-dataset length variability.'],
   Test=False)
  with (OUT/'analysis'/(x.dataset+'_DEV_diagnosis.json')).open('x') as f:json.dump(result,f,indent=2)
  print(json.dumps(result,indent=2))
 finally:
  if lease is not None:os.close(lease)
if __name__=='__main__':main()
