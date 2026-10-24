"""Toys-only integration; existing encoders/structure/loss/evaluation are retained."""
import os,sys,json,hashlib,pickle,types,gc,logging,csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path('/home/yyx/hzgcode/pom2.0').resolve();OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OUT))
from objective_model import ObjectiveSIL,ASPCF,base
from helpers.BaseRunner import BaseRunner
from utils import utils
KEYS=['HR@5','NDCG@5','HR@10','NDCG@10','HR@20','NDCG@20']
ORDER=['BPR-B0','SS-B0','SS-C0','SS-C1']
def read(p):return json.loads(Path(p).read_text())
def write(p,x,exclusive=True):
 p=Path(p);assert p.parent.resolve().is_relative_to(OUT)
 with p.open('x' if exclusive else 'w') as f:json.dump(x,f,indent=2,allow_nan=False,default=str)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def array_sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def arguments(arm,device='cpu',check=False):
 assert arm in ORDER
 a=types.SimpleNamespace(**read(OUT/'PLAN.json')['args']);objective,variant=arm.split('-')
 a.objective=objective;a.sil_variant=variant;a.sil_mode='baseline' if variant=='B0' else 'fixed'
 a.sil_use_behavior=1;a.sil_use_local=1;a.sil_use_recent=1;a.sil_alpha_initial=.1
 a.sil_asset_path=str(OUT/('CHECK_TRAIN_ASSET.json' if check else 'TRAIN_ASSET.json'))
 a.device=torch.device(device);a.model_path=str(OUT/'arms'/arm/'best.pt')
 return a
def read_corpus(phase='dev'):
 with (ROOT/'data/toys/SeqReader.pkl').open('rb') as f:c=pickle.load(f)
 tr=c.data_df['train'];counts=tr.groupby('user_id').size()
 # Validate the stored training membership rather than incorporating residual held-outs.
 actual=tr.groupby('user_id').item_id.agg(set).to_dict()
 assert actual=={u:s for u,s in c.train_clicked_set.items() if s},'Train membership/cache mismatch'
 c.train_clicked_set=actual
 data=c.data_df[phase]
 expected=data.user_id.map(counts).fillna(0).to_numpy()
 if phase=='test':expected=expected+data.user_id.map(c.data_df['dev'].groupby('user_id').size()).fillna(0).to_numpy()
 assert np.array_equal(data.position.to_numpy(),expected),'Chronological prefix mismatch'
 assert all(len(v)==1000 for v in data.neg_items),'1001 candidate protocol mismatch'
 c.data_df={'train':tr,phase:data}
 if hasattr(c,'all_df'):c.all_df=None
 if hasattr(c,'residual_clicked_set'):c.residual_clicked_set={}
 if phase!='test':
  for u in c.user_his:c.user_his[u]=c.user_his[u][:int(counts.get(u,0))]
 gc.collect();return c

def common_hash(m):
 h=hashlib.sha256()
 for n,p in m.named_parameters():
  if not n.startswith('sil_block.'):
   h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def model(arm,c,device='cpu',check=False):
 utils.init_seed(42);a=arguments(arm,device,check)
 m=ASPCF(a,c) if arm=='BPR-B0' else ObjectiveSIL(a,c)
 return m,a

def metrics_from_ranks(ranks):
 return {k:float(((ranks<=int(k.split('@')[1]))/(np.log2(ranks+1) if k.startswith('NDCG') else 1)).mean()) for k in KEYS}

def score_views(dataset,rank_path=None):
 m=dataset.model;m.eval();counts=np.asarray(read(OUT/'TRAIN_METADATA.json')['train_item_counts'])
 lock=read(OUT/'POP_PRIOR_LOCK.json') if (OUT/'POP_PRIOR_LOCK.json').exists() else None
 coeff=lock['raw_lambda'] if lock else None
 totals={s:np.zeros(6,np.float64) for s in (['raw','locked_prior'] if lock else ['raw'])}
 total=0;inputhash=hashlib.sha256();fp=None;writer=None
 if rank_path:
  fp=Path(rank_path).open('x',newline='');writer=csv.writer(fp);writer.writerow(['user_id','raw','locked_prior'])
 try:
  loader=DataLoader(dataset,batch_size=256,shuffle=False,num_workers=0,collate_fn=dataset.collate_batch)
  with torch.inference_mode():
   for batch in loader:
    for name in ['user_id','history_items','lengths','item_id']:
     ar=batch[name].numpy();inputhash.update(name.encode());inputhash.update(str(ar.shape).encode());inputhash.update(ar.tobytes())
    feed=utils.batch_to_gpu(batch,m.device);out=m(feed,True)
    raw=out.get('raw_prediction',out['prediction']).cpu().numpy().astype(np.float64)
    assert raw.shape[1]==1001 and np.isfinite(raw).all()
    scores={'raw':raw}
    if lock:
     freq=np.log1p(counts[batch['item_id'].numpy()]).astype(np.float32).astype(np.float64)
     scores['locked_prior']=raw+coeff*(freq-freq.mean(1,keepdims=True))
    ranks={}
    for scheme,s in scores.items():
     rr=np.argwhere((-s).argsort(axis=1)==0)[:,1]+1;ranks[scheme]=rr
     totals[scheme]+=np.array([metrics_from_ranks(rr)[k] for k in KEYS])*len(rr)
    if writer:writer.writerows(zip(batch['user_id'].numpy(),ranks['raw'],ranks['locked_prior']))
    total+=len(raw)
  assert total==len(dataset)
  return dict(metrics={s:dict(zip(KEYS,(v/total).tolist())) for s,v in totals.items()},input_hash=inputhash.hexdigest(),rows=total)
 finally:
  if fp:fp.close()

def verify(fast=True):
 manifest=read(OUT/'MANIFEST.json')
 for p,h in manifest['source_sha256'].items():assert sha(ROOT/p)==h,p
 for p,h in read(OUT/'SOURCE_LOCK.json').items():assert sha(OUT/p)==h,p
 for p,v in manifest['assets'].items():
  path=Path(p);s=path.stat();assert path.resolve().is_relative_to(ROOT)
  assert s.st_size==v['size'] and s.st_mtime_ns==v['mtime_ns'],p
  if not fast:assert sha(path)==v['sha256'],p
 return manifest

def lock_prior(m,c,checkpoint_sha):
 assert not (OUT/'POP_PRIOR_LOCK.json').exists()
 tr=c.data_df['train'].sort_values(['user_id','position'])
 ids=tr.item_id.to_numpy(np.int64);uid=tr.user_id.to_numpy(np.int64)
 offsets=np.r_[0,np.flatnonzero(uid[1:]!=uid[:-1])+1];ends=np.r_[offsets[1:],len(uid)]
 rng=np.random.default_rng(20261022);eligible=np.flatnonzero(ends-offsets>=2)
 chosen=np.sort(rng.choice(eligible,min(1024,len(eligible)),replace=False));users=uid[offsets[chosen]]
 catalog=np.unique(ids);cand=rng.choice(catalog,(len(chosen),1001),replace=True)
 history=np.zeros((len(chosen),20),np.int64);lengths=np.zeros(len(chosen),np.int64)
 for row,u in enumerate(chosen):
  end=ends[u]-1;seq=ids[max(offsets[u],end-20):end];history[row,:len(seq)]=seq;lengths[row]=len(seq);cand[row,0]=ids[end]
 fold_counts=np.stack([np.bincount(ids[uid%3!=f],minlength=c.n_items) for f in range(3)])
 sums=np.zeros(2,np.float64);m.eval()
 with torch.inference_mode():
  for off in range(0,len(chosen),256):
   sl=slice(off,off+256);feed={k:torch.from_numpy(v[sl]).to(m.device) for k,v in dict(user_id=users,history_items=history,lengths=lengths,item_id=cand).items()}
   raw=m(feed)['prediction'].cpu().numpy().astype(np.float64)
   # Match the previous probe's float32 log operation followed by float64 RMS.
   counts=fold_counts[(users[sl]%3)[:,None],np.asarray(cand[sl])]
   freq=torch.log(torch.as_tensor(counts,dtype=torch.float32)+1).numpy().astype(np.float64)
   sums[0]+=np.square(raw-raw.mean(1,keepdims=True)).sum();sums[1]+=np.square(freq-freq.mean(1,keepdims=True)).sum()
 sigmas=np.sqrt(sums/(len(chosen)*1001));assert np.isfinite(sigmas).all() and (sigmas>1e-8).all()
 scale=float(sigmas[0]/sigmas[1]);lock=dict(lambda_normalized=.30,scale_factor=scale,raw_lambda=.30*scale,
  base_centered_RMS=float(sigmas[0]),frequency_centered_RMS=float(sigmas[1]),fit_source='Train only; full user-fold exclusion from counts; user_id%3',reference='BPR-B0 best raw-Dev checkpoint; frozen before first Test',reference_checkpoint_sha256=checkpoint_sha,users=len(chosen),candidates_per_user=1001,Train_csv_sha256=read(OUT/'TRAIN_METADATA.json')['train_csv_sha256'],probe_inputs_sha256=array_sha(cand)+':'+array_sha(history),no_Dev_Test_statistics=True)
 write(OUT/'POP_PRIOR_LOCK.json',lock)
 asset=read(OUT/'TRAIN_METADATA.json');asset.update(lambda_pop=.30,locked_scale=scale,raw_pop_lambda=.30*scale,locked_prior_sha256=sha(OUT/'POP_PRIOR_LOCK.json'))
 write(OUT/'TRAIN_ASSET.json',asset);return lock
