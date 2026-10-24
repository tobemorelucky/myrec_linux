"""Frozen GICTRec v1; explicit eight-arm continuation and original dual-score evaluator."""
import os,sys,json,time,hashlib,csv,fcntl,copy,logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
ROOT=Path('/home/yyx/hzgcode/pom2.0').resolve()
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from gict_data import train_corpus,args_from_config,sha
from models.sequential.GICTRec import GICTRec
from helpers.BaseRunner import BaseRunner
from utils import utils
KEYS=['HR@5','NDCG@5','HR@10','NDCG@10','HR@20','NDCG@20']
def read(p):return json.loads(Path(p).read_text())
def write(p,x,exclusive=True):
 p=Path(p);assert p.parent.resolve().is_relative_to(ROOT)
 with p.open('x' if exclusive else 'w') as f:json.dump(x,f,indent=2,allow_nan=False,default=str)
def verify():
 for p,h in read(OUT/'SOURCE_LOCK.json').items():assert sha(p)==h,p
 for p,h in read(OUT/'PLAN.json')['archives'].items():assert sha(p)==h,p
 for p,h in read(ROOT/'research/THESIS_CH4_FROZEN_20261024.index.json')['sources'].items():assert sha(p)==h,p
def config(ds,variant,device='cpu'):
 spec=read(OUT/'PLAN.json')['datasets'][ds]
 assert sha(spec['checkpoint'])==spec['checkpoint_sha256']
 assert sha(spec['config'])==spec['config_sha256']
 a=args_from_config(spec['config']);a.gict_variant=variant;a.gict_neighbors=8;a.gict_temperature=.1
 a.device=torch.device(device);a.test_epoch=-1
 assert a.random_seed==42 and a.lr>0 and a.epoch==200
 assert a.metric.split(',')[0]=='NDCG' and a.topk.split(',')[0]=='5'
 return a,spec
def make_model(ds,variant,device='cpu'):
 a,spec=config(ds,variant,device);c=train_corpus(a)
 bank=torch.load(spec['bank'],map_location='cpu',weights_only=True)
 assert bank['metadata']['scope']=='full_train'
 assert len(bank['user_ids'])==len(c.user_his) and set(bank['user_ids'].tolist())==set(c.user_his)
 assert bank['metadata']['teacher_checkpoint_sha256']==spec['checkpoint_sha256']
 assert sha(spec['bank'])==read(OUT/'BANKS_CHECKED.json')[ds]['bank_sha256']
 utils.init_seed(a.random_seed)
 m=GICTRec(a,c);m.load_c1(torch.load(spec['checkpoint'],map_location='cpu',weights_only=True))
 m.attach_bank(bank)
 if variant=='G0':m.transfer.output.weight.requires_grad_(False)
 return m.to(a.device),a,c
def heldout(c,a,phase):
 assert phase in ('dev','test')
 path=Path(a.path)/a.dataset/(phase+'.csv')
 frame=utils.eval_list_columns(pd.read_csv(path,sep='\t'))
 counts=c.data_df['train'].groupby('user_id').size()
 assert frame.user_id.is_unique and all(len(n)==1000 for n in frame.neg_items)
 frame['position']=frame.user_id.map(counts)
 assert frame.position.notna().all()
 if phase=='test':
  dev=c.data_df['dev']
  assert dev.user_id.is_unique and set(frame.user_id)<=set(dev.user_id)
  # Only after checkpoint lock: a Test prefix includes the preceding Dev event.
  for row in dev.itertuples(index=False):
   assert c.user_his[int(row.user_id)][-1][1]<=row.time
   c.user_his[int(row.user_id)].append((int(row.item_id),int(row.time)))
  frame['position']=frame.position+1
 frame['position']=frame.position.astype(np.int64)
 assert all(len(c.user_his[int(u)])>=int(pos) for u,pos in zip(frame.user_id,frame.position))
 c.data_df[phase]=frame
def datasets(m,a,c):
 heldout(c,a,'dev')
 data={s:m.Dataset(m,c,s) for s in ('train','dev')}
 for d in data.values():d.prepare()
 return data
def metrics(ranks):
 return {k:float(((ranks<=int(k.split('@')[1]))/(np.log2(ranks+1) if k.startswith('NDCG') else 1)).mean()) for k in KEYS}
def score_views(dataset,rank_path=None):
 m=dataset.model;m.eval();total=0;hasher=hashlib.sha256()
 totals={s:np.zeros(6,np.float64) for s in ('raw','locked_prior')}
 fp=Path(rank_path).open('x',newline='') if rank_path else None
 writer=csv.writer(fp) if fp else None
 if writer:writer.writerow(['user_id','raw','locked_prior'])
 loader=DataLoader(dataset,batch_size=256,shuffle=False,num_workers=0,collate_fn=dataset.collate_batch)
 try:
  with torch.inference_mode():
   for batch in loader:
    users=batch['user_id'].detach().cpu().numpy().copy()
    for name in ('user_id','history_items','lengths','item_id'):
     ar=batch[name].detach().cpu().numpy()
     hasher.update(name.encode());hasher.update(str(ar.shape).encode());hasher.update(ar.tobytes())
    got=m(utils.batch_to_gpu(batch,m.device),True)
    arrays={'raw':got['raw_prediction'].detach().cpu().numpy().astype(np.float64),
            'locked_prior':got['prediction'].detach().cpu().numpy().astype(np.float64)}
    ranks={}
    for s,scores in arrays.items():
     assert scores.shape==(len(users),1001) and np.isfinite(scores).all()
     r=np.argwhere((-scores).argsort(axis=1)==0)[:,1]+1;ranks[s]=r
     totals[s]+=np.array([metrics(r)[k] for k in KEYS])*len(r)
    if writer:writer.writerows(zip(users,ranks['raw'],ranks['locked_prior']))
    total+=len(users)
  assert total==len(dataset)
  return dict(metrics={s:dict(zip(KEYS,(v/total).tolist())) for s,v in totals.items()},input_hash=hasher.hexdigest(),rows=total)
 finally:
  if fp:fp.close()
def protocol_lock(ds,phase,got):
 path=OUT/(ds+'_'+phase+'_PROTOCOL.json')
 with (OUT/(ds+'_'+phase+'.guard')).open('a') as guard:
  fcntl.flock(guard,fcntl.LOCK_EX)
  if not path.exists():write(path,dict(input_hash=got['input_hash'],rows=got['rows'],candidates=1001,duplicate_slots_preserved=True,evaluator='NumPy row-wise (-scores).argsort; six metrics'))
  else:
   old=read(path);assert old['input_hash']==got['input_hash'] and old['rows']==got['rows']
def common_hash(m):
 h=hashlib.sha256()
 for n,p in m.named_parameters():
  if not n.startswith('transfer.'):
   h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
