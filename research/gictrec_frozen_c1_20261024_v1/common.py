"""Frozen-C1 optimization; reuse unchanged first-round data and six-metric evaluator."""
import sys,json,os,time,hashlib,csv,copy,logging,gc,fcntl,importlib.util
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path('/home/yyx/hzgcode/pom2.0').resolve();OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('gict_round1_common_frozen',ROOT/'research/gictrec_round1_20261024_v1/common.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
from models.sequential.GICTRecFrozen import GICTRecFrozen
from gict_data import train_corpus,sha
from utils import utils
KEYS=old.KEYS
read=old.read
score_views=old.score_views
heldout=old.heldout
datasets=old.datasets
common_hash=old.common_hash
def write(p,x,exclusive=True):
 p=Path(p);assert p.parent.resolve().is_relative_to(ROOT)
 with p.open('x' if exclusive else 'w') as f:json.dump(x,f,indent=2,allow_nan=False,default=str)
def verify():
 old.verify()
 for p,h in read(OUT/'SOURCE_LOCK.json').items():assert sha(p)==h,p
def model(variant,device='cpu'):
 a,spec=old.config('ml-1m',variant,device);a.branch_lr=.001
 c=train_corpus(a);bank=torch.load(spec['bank'],map_location='cpu',weights_only=True)
 assert bank['metadata']['scope']=='full_train' and set(bank['user_ids'].tolist())==set(c.user_his)
 assert bank['metadata']['teacher_checkpoint_sha256']==spec['checkpoint_sha256']
 assert sha(spec['bank'])==read(ROOT/'research/gictrec_round1_20261024_v1/BANKS_CHECKED.json')['ml-1m']['bank_sha256']
 utils.init_seed(42);m=GICTRecFrozen(a,c);m.load_c1(torch.load(spec['checkpoint'],map_location='cpu',weights_only=True));m.attach_bank(bank)
 return m.to(a.device),a,c,spec
def protocol_lock(phase,got):
 # First-round candidate/history hash is a read-only reference; no old outputs modified.
 reference=ROOT/'research/gictrec_round1_20261024_v1'/('ml-1m_'+phase+'_PROTOCOL.json')
 assert got['input_hash']==read(reference)['input_hash'] and got['rows']==read(reference)['rows']
 path=OUT/(phase+'_PROTOCOL.json')
 with (OUT/(phase+'.guard')).open('a') as guard:
  fcntl.flock(guard,fcntl.LOCK_EX)
  if not path.exists():write(path,dict(input_hash=got['input_hash'],rows=got['rows'],candidates=1001,duplicate_slots_preserved=True))
  else:assert got['input_hash']==read(path)['input_hash']
def transfer_state(m):
 return {n:t.detach().cpu().clone() for n,t in m.transfer.state_dict().items()}
def batch_diagnostics(m,out,grad=None):
 ratio=out['transfer_delta'].detach().norm(dim=-1)/out['c1_interest_vectors'].detach().norm(dim=-1).clamp_min(1e-8)
 return dict(projection_norm=float(m.transfer.output.weight.detach().norm()),gradient_norm=grad,
  residual_ratio_mean=float(ratio.mean()),residual_ratio_max=float(ratio.max()))
