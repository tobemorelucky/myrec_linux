"""Read-only input helpers and original NumPy Dev evaluation."""
import sys,json,pickle,types,hashlib,random,time
from pathlib import Path
import numpy as np,torch
ROOT=Path('/home/yyx/hzgcode/pom2.0').resolve();OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OUT))
from model import SoftGuideSIL,base
from helpers.BaseRunner import BaseRunner
from utils import utils
ARMS=('B0','C0','C1','M2');KEYS=('HR@5','NDCG@5','HR@10','NDCG@10','HR@20','NDCG@20')
def sha(p):return base.file_sha(p)
def write(name,value):
 p=OUT/name;assert p.parent.resolve().is_relative_to(OUT)
 with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def verify():
 manifest=json.loads((OUT/'MANIFEST.json').read_text())
 for p,h in manifest['protected_sha256'].items():assert sha(ROOT/p)==h,p
 return manifest

def corpus(ds):
 with (ROOT/'data'/ds/'SeqReader.pkl').open('rb') as f:c=pickle.load(f)
 c.data_df={phase:c.data_df[phase] for phase in ('train','dev')}
 counts=c.data_df['train'].groupby('user_id').size();dev=c.data_df['dev']
 assert np.all(dev.position.to_numpy()==dev.user_id.map(counts).to_numpy())
 for u in c.user_his:c.user_his[u]=c.user_his[u][:int(counts.get(u,0))]
 return c

def model(ds,c,arm,device='cpu',K=4):
 info=json.loads((OUT/'MANIFEST.json').read_text())['inputs'][ds]
 a=types.SimpleNamespace(**info['config']['args']);a.device=torch.device(device);a.K=K
 a.sil_variant=arm;a.sil_alpha_initial=.1;a.lr=.0001;a.buffer=0
 utils.init_seed(42);m=SoftGuideSIL(a,c)
 if K==4:
  state=torch.load(info['checkpoint'],map_location='cpu',weights_only=True)
  missing,unexpected=m.load_state_dict(state,strict=False)
  expected=[k for k in m.state_dict() if k.startswith('sil_block.')]
  assert set(missing)==set(expected) and not unexpected,(missing,unexpected)
 return m.to(a.device),a

def histories(c,frame):
 hist=np.zeros((len(frame),20),np.int64);lengths=np.zeros(len(frame),np.int64)
 for j,(u,p) in enumerate(zip(frame.user_id.to_numpy(),frame.position.to_numpy())):
  seq=c.user_his[int(u)][max(0,int(p)-20):int(p)];lengths[j]=len(seq);hist[j,:len(seq)]=[v[0] for v in seq]
 return hist,lengths

def feed(h,l,u,pos,neg,device):
 return dict(history_items=torch.as_tensor(h,device=device),lengths=torch.as_tensor(l,device=device),
             user_id=torch.as_tensor(u,device=device),item_id=torch.as_tensor(np.column_stack([pos,neg]),device=device))
def negative(m,u,g):
 item=torch.randint(1,m.item_num,(len(u),),device=m.device,generator=g)
 while True:
  bad=((m.train_membership[u,item//64]>>(item%64))&1).bool()
  if not bad.any():return item
  item[bad]=torch.randint(1,m.item_num,(int(bad.sum()),),device=m.device,generator=g)
def rng():return torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()
def restore(r):torch.set_rng_state(r[0]);torch.cuda.set_rng_state(r[1])

@torch.no_grad()
def users(m,table,h,l):
 result=[]
 for off in range(0,len(h),512):
  hh=torch.as_tensor(h[off:off+512],device=m.device);ll=torch.as_tensor(l[off:off+512],device=m.device)
  raw=table[hh];pos=(ll[:,None]-torch.arange(hh.shape[1],device=m.device)[None])*(hh>0)
  v,A=m.extractor(raw+m.position_emb(pos),ll)
  v,_=m.recompose(hh,raw,ll,v,A)
  result.append((v*m.aggregator(raw,ll)[:,:,None]).sum(1))
 return torch.cat(result)

@torch.no_grad()
def evaluate(m,c,h,l,candidates,validate=False):
 start=time.perf_counter();m.eval()
 table=torch.cat([m.item_encoder(torch.arange(off,min(off+2048,c.n_items),device=m.device)) for off in range(0,c.n_items,2048)])
 u=users(m,table,h,l);rr=[]
 for off in range(0,len(h),256):
  ids=torch.as_tensor(candidates[off:off+256],device=m.device)
  raw=(u[off:off+256,None]*table[ids]).sum(-1)
  corrected=m.corrected_scores(raw,ids).cpu().numpy()
  rank=np.argwhere((-corrected).argsort(1)==0)[:,1]+1;rr.append(rank)
  if validate and off==0:
   n=32;f=dict(history_items=torch.as_tensor(h[:n],device=m.device),lengths=torch.as_tensor(l[:n],device=m.device),item_id=ids[:n])
   actual=m(f,True)
   assert torch.allclose(u[:n],actual['user_vector'],atol=1e-6,rtol=1e-5)
   assert np.allclose(corrected[:n],actual['prediction'].cpu().numpy(),atol=2e-5,rtol=1e-5)
   got=BaseRunner.evaluate_method(corrected,[5,10,20],['HR','NDCG'])
   for key,val in got.items():
    k=int(key.split('@')[1]);expected=((rank<=k)/(np.log2(rank+1) if key.startswith('NDCG') else 1)).mean()
    assert abs(val-expected)<1e-12
 rank=np.concatenate(rr)
 return dict(metrics={key:float(((rank<=int(key.split('@')[1]))/(np.log2(rank+1) if key.startswith('NDCG') else 1)).mean()) for key in KEYS},seconds=time.perf_counter()-start)
