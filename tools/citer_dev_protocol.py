"""Explicit Train/Dev-only protocol. Test data and Test-output creation are denied."""
import sys,os,json,hashlib,csv
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader
from utils import utils
KEYS=['HR@5','NDCG@5','HR@10','NDCG@10','HR@20','NDCG@20']
def install_guard():
 def audit(event,args):
  if event!='open' or not args:return
  obj=args[0]
  if not isinstance(obj,(str,bytes,os.PathLike)):return
  name=Path(os.fsdecode(obj)).name
  writing=(isinstance(args[1],str) and any(c in args[1] for c in 'wa+')) or (len(args)>2 and isinstance(args[2],int) and bool(args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT)))
  if name in ('test.csv','SeqReader.pkl','test_user_ranks.csv') or (name.startswith('TEST_') and writing):
   raise PermissionError('Dev-only experiment forbids '+name)
 sys.addaudithook(audit)
def load_dev(c,a,indices=None):
 path=Path(a.path)/a.dataset/'dev.csv'
 df=pd.read_csv(path,sep='\t')
 if indices is not None:df=df.iloc[indices].copy().reset_index(drop=True)
 df=utils.eval_list_columns(df)
 df['position']=df.user_id.map(c.data_df['train'].groupby('user_id').size()).astype(np.int64)
 assert df.user_id.is_unique and all(len(v)==1000 for v in df.neg_items)
 c.data_df['dev']=df
 assert set(c.data_df)=={'train','dev'}
def metrics(ranks):
 return {k:float(((ranks<=int(k.split('@')[1]))/(np.log2(ranks+1) if k.startswith('NDCG') else 1)).mean()) for k in KEYS}
def score_views(dataset,rank_path=None):
 assert dataset.phase=='dev'
 if hasattr(dataset,'corpus'):assert set(dataset.corpus.data_df)<= {'train','dev'}
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
     ar=batch[name].detach().cpu().numpy();hasher.update(name.encode());hasher.update(str(ar.shape).encode());hasher.update(ar.tobytes())
    got=m(utils.batch_to_gpu(batch,m.device),True)
    arrays={'raw':got['raw_prediction'].detach().cpu().numpy().astype(np.float64),
     'locked_prior':got['prediction'].detach().cpu().numpy().astype(np.float64)}
    ranks={}
    for scheme,scores in arrays.items():
     assert scores.shape==(len(users),1001) and np.isfinite(scores).all()
     rs=np.argwhere((-scores).argsort(axis=1)==0)[:,1]+1;ranks[scheme]=rs
     totals[scheme]+=np.array([metrics(rs)[k] for k in KEYS])*len(rs)
    if writer:writer.writerows(zip(users,ranks['raw'],ranks['locked_prior']))
    total+=len(users)
  assert total==len(dataset)
  return dict(metrics={s:dict(zip(KEYS,(v/total).tolist())) for s,v in totals.items()},input_hash=hasher.hexdigest(),rows=total)
 finally:
  if fp:fp.close()
