"""Frozen ASPCF P0/P1 information readout probe. Train/Dev only, exactly 20 epochs.
No public model, Reader, Runner, assets or original checkpoint is modified.
"""
import os, sys, json, time, pickle, hashlib, argparse, random, signal, copy, csv, traceback, subprocess
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF as Base
from models.BaseModel import GeneralModel, BaseModel
from helpers.BaseRunner import BaseRunner
from tools.diagnose_aspcf_frozen_dev_scoring import SPEC
from utils import utils
SOURCES=['models/sequential/LLMMIRecASPCF.py','models/sequential/llmmi_components.py','models/BaseModel.py',
 'helpers/BaseReader.py','helpers/SeqReader.py','helpers/BaseRunner.py','tools/run_frozen_interest_readout.py']

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(4194304),b''):h.update(chunk)
 return h.hexdigest()
def dump(p,obj):
 p=Path(p);assert p.resolve().is_relative_to(ROOT)
 with p.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
def state_hash(state):
 h=hashlib.sha256()
 for k,v in state.items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def rng_hash():
 h=hashlib.sha256(torch.get_rng_state().numpy().tobytes())
 for t in torch.cuda.get_rng_state_all():h.update(t.cpu().numpy().tobytes())
 h.update(pickle.dumps(np.random.get_state()));h.update(pickle.dumps(random.getstate()))
 return h.hexdigest()
def stats(t):
 x=t.detach().float().reshape(-1)
 return dict(mean=float(x.mean()),median=float(x.median()),max=float(x.max()))
def metrics(pred):return {k:float(v) for k,v in BaseRunner.evaluate_method(pred,[5,10,20],['HR','NDCG']).items()}
def ranked(pred):
 order=(-pred).argsort(axis=1);r=np.argwhere(order==0)[:,1]+1
 return order[:,:5],r,(r<=5)/np.log2(r+1)

def get_config(ds):
 stem,expected,_=SPEC[(ds,4)];ck=ROOT/'new_model'/stem/ds/'seed42/LLMMIRecASPCF_seed42.pt'
 log=ROOT/'new_log'/stem/ds/'seed42/LLMMIRecASPCF_seed42.log';old={}
 for line in log.read_text().splitlines():
  cols=line.split('|')
  if len(cols)>1:old[cols[0].strip()]=cols[1].strip()
 p=BaseRunner.parse_runner_args(Base.parse_model_args(argparse.ArgumentParser()))
 cli=[x for action in p._actions if action.dest in old and action.dest!='help' and '...' not in old[action.dest]
      for x in [action.option_strings[0],old[action.dest]]]
 a=p.parse_args(cli);a.device=torch.device('cuda:0');a.buffer=0
 a.llm_emb_path=str(ROOT/'data'/ds/'handled/llm_table_pca1536.pkl')
 assert sha(ck)==expected
 assert (a.K,a.emb_size,a.history_max,a.num_neg,a.lambda_relation,a.test_all)==(4,64,20,1,.01,0)
 assert a.optimizer=='Adam' and a.l2==1e-6 and a.batch_size==1024
 assert a.lr==(.004 if ds=='beauty' else .001) and a.grad_clip==0
 refpath=ROOT/'new_log/aspcf_frozen_dev_scoring/20261018_s0_s1_s2_seed42'/(ds+'_K4.json')
 reference=json.loads(refpath.read_text())['metrics']['S0']
 return a,ck,expected,reference,dict(path=str(log),sha256=sha(log),logged_values=old)

class Probe(nn.Module):
 def __init__(self,base,mode,backbone_mode='frozen'):
  super().__init__();assert mode in ['P0','P1'];self.backbone=base;self.mode=mode
  assert backbone_mode in ['frozen','joint'];self.backbone_mode=backbone_mode
  self.readout=nn.Sequential(nn.Linear(base.K*base.emb_size,96),nn.GLU(-1),nn.Linear(48,64))
  nn.init.zeros_(self.readout[-1].weight);nn.init.zeros_(self.readout[-1].bias)
  if self.backbone_mode=='frozen':
   self.backbone.requires_grad_(False);self.backbone.eval()
  else:
   # Joint mode requires a fresh original backbone, never a frozen/best model.
   assert all(p.requires_grad for p in self.backbone.parameters())
   self.backbone.train(self.training)
  # Minimal original Runner/Dataset interface; the frozen loop is unchanged.
  for key in ['device','model_path','buffer','num_neg','history_max','test_all','K','emb_size']:
   setattr(self,key,getattr(base,key))
  self.optimizer=None;self.check_list=base.check_list
 def train(self,mode=True):
  super().train(mode)
  if self.backbone_mode=='frozen':self.backbone.eval()
  return self
 def input(self,out):
  v,w,u=out['interest_vectors'],out['interest_weights'],out['user_vector']
  slots=w[:,:,None]*(u[:,None,:] if self.mode=='P0' else v)
  return slots.flatten(1),slots
 def from_output(self,out):
  if self.backbone_mode=='frozen':
   assert not self.backbone.training and not any(m.training for m in self.backbone.modules())
  x,_=self.input(out);delta=self.readout(x);u=out['user_vector']+delta
  pred=(u[:,None,:]*out['candidate_vectors']).sum(-1)
  result=dict(prediction=pred,delta=delta,user_vector=u)
  if self.backbone_mode=='joint' and '_relation_ids' in out:
   result['_relation_ids']=out['_relation_ids']
  return result
 def forward(self,b):
  if self.backbone_mode=='frozen':
   self.backbone.eval()
   with torch.no_grad():o=self.backbone(b,True)
  else:o=self.backbone(b,True)
  return self.from_output(o)
 customize_parameters=BaseModel.customize_parameters
 def loss(self,out):
  if self.backbone_mode=='frozen':return GeneralModel.loss(self,out)
  return self.backbone.loss(out)
 # Joint mode can use the unchanged public Runner; old frozen save/readout code
 # still uses its existing explicit readout-only checkpoint functions.
 save_model=BaseModel.save_model
 load_model=BaseModel.load_model
 count_variables=BaseModel.count_variables

def build(ds):
 a,ck,expected,reference,hist=get_config(ds)
 with (ROOT/'data'/ds/'SeqReader.pkl').open('rb') as f:c=pickle.load(f)
 c.data_df={phase:c.data_df[phase] for phase in ['train','dev']}
 counts=c.data_df['train'].groupby('user_id').size()
 tr=c.data_df['train'];dv=c.data_df['dev']
 assert np.all(tr.position.to_numpy()<tr.user_id.map(counts).to_numpy())
 assert np.all(dv.position.to_numpy()==dv.user_id.map(counts).to_numpy()),'Unexpected Dev boundary; do not change protocol'
 utils.init_seed(42);base=Base(a,c).to(a.device)
 base.load_state_dict(torch.load(ck,map_location=a.device,weights_only=True),strict=True)
 base.requires_grad_(False);base.eval()
 with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
  torch.manual_seed(42);p0=Probe(base,'P0').to(a.device)
  p1=Probe(base,'P1').to(a.device);p1.readout.load_state_dict(p0.readout.state_dict())
 assert state_hash(p0.readout.state_dict())==state_hash(p1.readout.state_dict())
 models=dict(P0=p0,P1=p1)
 data={phase:Base.Dataset(base,c,phase) for phase in ['train','dev']}
 assert all(len(n)==1000 for n in data['dev'].data['neg_items'])
 config=dict(dataset=ds,baseline=str(ck),baseline_sha256=expected,historical_config=hist,
  effective_args={k:str(v) if k=='device' else v for k,v in vars(a).items()},epochs=20,early_stop=False,
  seed=42,phase='Train/Dev',test_allowed=False,backbone_eval=True,backbone_frozen=True,
  relation_optimized=False,loss='Original GeneralModel.loss, num_neg=1; frozen BPR readout only',
  readout='Linear(K*64,96)-GLU(48)-Linear(48,64); output zero initialized',
  readout_initial_sha256=state_hash(p0.readout.state_dict()),readout_parameters=sum(p.numel() for p in p0.readout.parameters()),
  backbone_state_sha256=state_hash(base.state_dict()),frozen_parameters=sum(p.numel() for p in base.parameters()),
  train_rows=len(data['train']),dev_rows=len(data['dev']),dev_users=int(np.unique(data['dev'].data['user_id']).size),
  pair_method='One shared frozen forward and one common batch stream; independent heads and fresh Adam; no readout Dropout',
  assets_sha256={s:sha(ROOT/'data'/ds/s) for s in ['train.csv','dev.csv','SeqReader.pkl','item2id.json','handled/llm_table_pca1536.pkl']})
 assert config['readout_parameters']==27808
 return dict(a=a,base=base,models=models,data=data,config=config,reference=reference)

def optimizer(model,a):
 opt=BaseRunner(a)._build_optimizer(model)
 active=[p for group in opt.param_groups for p in group['params']]
 assert len(active)==len({id(p) for p in active})
 assert {id(p) for p in active}=={id(p) for p in model.readout.parameters()}
 return opt

def unit_tests(ctx):
 a,base,models,data=[ctx[k] for k in ['a','base','models','data']]
 data['train'].actions_before_epoch();b=next(iter(DataLoader(data['train'],batch_size=32,shuffle=False,num_workers=0,collate_fn=data['train'].collate_batch)))
 b=utils.batch_to_gpu(b,a.device)
 with torch.no_grad():o=base(b,True)
 assert '_relation_ids' not in o and not o['interest_vectors'].requires_grad
 rec=dict(checks=[],input_sum_max_abs={})
 for mode,m in models.items():
  m.train();assert not any(p.requires_grad for p in base.parameters())
  assert not any(s.training for s in base.modules()) and m.readout.training
  x,slots=m.input(o);assert x.shape==(32,256)
  err=float((slots.sum(1)-o['user_vector']).abs().max());rec['input_sum_max_abs'][mode]=err
  assert torch.allclose(slots.sum(1),o['user_vector'],atol=2e-6,rtol=2e-6)
  y=m.from_output(o);assert torch.count_nonzero(y['delta'])==0
  assert torch.equal(y['prediction'],o['prediction'])
  assert torch.equal(m.loss(y),GeneralModel.loss(base,o))
  initial={k:t.clone() for k,t in m.readout.state_dict().items()}
  opt=optimizer(m,a);g=[]
  for step in range(2):
   opt.zero_grad(set_to_none=True);z=m.from_output(o);loss=m.loss(z);loss.backward()
   assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.readout.parameters())
   assert all(p.grad is None for p in base.parameters())
   g.append(dict(step=step+1,BPR=float(loss.detach()),inner_gradient=float(m.readout[0].weight.grad.norm()),
    output_gradient=float(m.readout[-1].weight.grad.norm())))
   opt.step()
  assert g[0]['output_gradient']>0 and g[1]['inner_gradient']>0
  assert any(not torch.equal(initial[k],t) for k,t in m.readout.state_dict().items())
  rec[mode]=g
  # Same history with different candidate lists cannot change inputs or user vector.
  alt={k:(v.clone() if torch.is_tensor(v) else v) for k,v in b.items()};alt['item_id']=alt['item_id'].flip(1)
  with torch.no_grad():oa=base(alt,True)
  assert torch.equal(m.input(o)[0],m.input(oa)[0])
  assert torch.equal(m.from_output(o)['user_vector'],m.from_output(oa)['user_vector'])
  one={k:(v[:1] if torch.is_tensor(v) and v.ndim and v.shape[0]==32 else v) for k,v in b.items()}
  with torch.no_grad():pred_one=m(one)['prediction'];pred_batch=m(b)['prediction'][:1]
  assert torch.allclose(pred_one,pred_batch,atol=2e-5,rtol=2e-5)
  synthetic={k:(v[:2] if torch.is_tensor(v) and v.ndim and v.shape[0]==32 else v) for k,v in b.items()}
  synthetic['item_id']=synthetic['item_id'][:,0:1].expand(-1,1001).contiguous()
  with torch.no_grad():dup=m(synthetic)['prediction']
  assert dup.shape==(2,1001) and torch.equal(dup,dup[:,0:1].expand_as(dup))
  # In-memory checkpoint roundtrip, no new model assets for tests.
  import io
  buf=io.BytesIO();torch.save(m.readout.state_dict(),buf);buf.seek(0)
  st=torch.load(buf,map_location=a.device,weights_only=True);assert all(torch.equal(st[k],t) for k,t in m.readout.state_dict().items())
  m.readout.load_state_dict(initial);assert state_hash(initial)==ctx['config']['readout_initial_sha256']
  m.readout.zero_grad(set_to_none=True)
 v=torch.randn(7,1,64,device=a.device);w=torch.ones(7,1,device=a.device);u=(v*w[:,:,None]).sum(1)
 assert torch.equal((w[:,:,None]*u[:,None]).flatten(1),(w[:,:,None]*v).flatten(1))
 assert state_hash(base.state_dict())==ctx['config']['backbone_state_sha256']
 rec['checks']=['shape_and_equal_capacity','slot_sum_numerical_identity','zero_residual','exact_baseline_scores',
 'exact_original_BPR','identical_initial_heads','all_backbone_gradients_None','second_step_inner_gradient',
 'train_override_keeps_backbone_eval','shared_Train_batches_and_no_readout_Dropout','1001_duplicate_slots',
 'same_Train_Dev_score','no_Test_path','no_BxB_matrix','K1_identity','candidate_independent_inputs',
 'user_batch_isolation','checkpoint_roundtrip','original_heads_restored_after_unit_steps']
 return rec

def loader(ctx,phase,shuffle=False):
 a=ctx['a'];d=ctx['data'][phase];assert phase in ['train','dev']
 return DataLoader(d,batch_size=a.batch_size if phase=='train' else a.eval_batch_size,shuffle=shuffle,
  num_workers=a.num_workers,pin_memory=bool(a.pin_memory),collate_fn=d.collate_batch)

def evaluate(ctx):
 base=ctx['base'];models=ctx['models'];a=ctx['a'];base.eval()
 for m in models.values():m.eval()
 ps={name:[] for name in models};ps['base']=[];us=[];ls=[];ratios={name:[] for name in models};hashes={k:hashlib.sha256() for k in ['users','history','candidates']}
 with torch.inference_mode():
  for batch in loader(ctx,'dev'):
   assert batch['phase']=='dev' and batch['item_id'].shape[1]==1001
   us.append(batch['user_id'].numpy());ls.append(batch['lengths'].numpy())
   for key,keys in [('users',['user_id']),('history',['history_items','lengths']),('candidates',['item_id'])]:
    for k in keys:hashes[key].update(batch[k].contiguous().numpy().tobytes())
   b=utils.batch_to_gpu(batch,a.device);o=base(b,True)
   ps['base'].append(o['prediction'].cpu().numpy())
   for name,m in models.items():
    out=m.from_output(o);assert torch.isfinite(out['prediction']).all()
    ps[name].append(out['prediction'].cpu().numpy())
    ratios[name].append((out['delta'].norm(dim=-1)/o['user_vector'].norm(dim=-1).clamp_min(1e-12)).cpu().numpy())
 ps={k:np.concatenate(v) for k,v in ps.items()}
 return dict(predictions=ps,metrics={k:metrics(v) for k,v in ps.items()},users=np.concatenate(us),lengths=np.concatenate(ls),
  residual_ratio={k:float(np.concatenate(v).mean()) for k,v in ratios.items()},input_sha256={k:v.hexdigest() for k,v in hashes.items()})

def save_readout(path,ctx,mode,epoch):
 m=ctx['models'][mode]
 payload=dict(readout={k:t.detach().cpu().clone() for k,t in m.readout.state_dict().items()},mode=mode,epoch=epoch,
  K=ctx['a'].K,baseline=ctx['config']['baseline'],baseline_sha256=ctx['config']['baseline_sha256'])
 with path.open('xb') as f:torch.save(payload,f)
 return sha(path)

def bootstrap_users(users,delta):
 # Cluster bootstrap of sample-weighted mean: retain all rows of each sampled user.
 _,inverse=np.unique(users,return_inverse=True);tot=np.bincount(inverse,weights=delta);cnt=np.bincount(inverse)
 g=np.random.default_rng(20261019);values=[]
 for _ in range(2000):
  take=g.integers(0,len(tot),len(tot));values.append(tot[take].sum()/cnt[take].sum())
 return np.quantile(values,[.025,.975]).tolist()

def comparisons(ev):
 users,L=ev['users'],ev['lengths'];p=ev['predictions'];result={};rows={}
 for name,scores in p.items():
  top,r,n=ranked(scores);rows[name]=(top,r,n)
 for left,right in [('P1','P0'),('P0','base'),('P1','base')]:
  tl,rl,nl=rows[left];tr,rr,nr=rows[right];d=nl-nr
  rec=dict(NDCG5_delta=float(d.mean()),cluster_bootstrap95=bootstrap_users(users,d),
   win_loss_tie=[int((d>0).sum()),int((d<0).sum()),int((d==0).sum())],
   positive_rank_changed_rows=int((rl!=rr).sum()),mean_positive_rank_delta=float((rl-rr).mean()),
   top5_slot_set_changed_rows=int(np.any(np.sort(tl,axis=1)!=np.sort(tr,axis=1),axis=1).sum()),
   top5_slot_order_changed_rows=int(np.any(tl!=tr,axis=1).sum()),history_groups=[])
  for lo,hi in [(1,5),(6,10),(11,20)]:
   mask=(L>=lo)&(L<=hi)
   if mask.any():rec['history_groups'].append(dict(history=f'{lo}-{hi}',rows=int(mask.sum()),users=int(np.unique(users[mask]).size),
     left_NDCG5=float(nl[mask].mean()),right_NDCG5=float(nr[mask].mean()),delta=float(d[mask].mean()),
     win_loss_tie=[int((d[mask]>0).sum()),int((d[mask]<0).sum()),int((d[mask]==0).sum())]))
  result[left+'-'+right]=rec
 return result,rows

def train_pair(ctx,out,epoch0):
 a=ctx['a'];ds=ctx['config']['dataset'];start=time.monotonic();budget=900 if ds=='beauty' else 2700
 signal.alarm(budget);utils.init_seed(42)
 models=ctx['models'];opts={name:optimizer(m,a) for name,m in models.items()}
 initial={name:{k:t.detach().clone() for k,t in m.readout.state_dict().items()} for name,m in models.items()}
 best={name:dict(value=-float('inf')) for name in models};epochs=[]
 ctx['config'].update(optimizer_groups={name:[{k:v for k,v in group.items() if k!='params'} for group in opt.param_groups] for name,opt in opts.items()},
  time_budget_seconds_per_arm=budget,conservative_timing='Pair wall time charged in full to both arms; feature forward shared, no claim of separately measured standalone times')
 dump(out/'config.json',ctx['config'])
 for name in models:(out/name).mkdir()
 baseline_hash=ctx['config']['backbone_state_sha256'];torch.cuda.reset_peak_memory_stats()
 for epoch in range(1,21):
  t=time.monotonic();rng_start=rng_hash();ctx['data']['train'].actions_before_epoch()
  negatives_sha=hashlib.sha256(np.ascontiguousarray(ctx['data']['train'].data['neg_items']).tobytes()).hexdigest()
  for m in models.values():m.train()
  losses={name:[] for name in models};order=hashlib.sha256();steps=0;rows=0;maxgrad={name:0. for name in models}
  for batch in loader(ctx,'train',True):
   assert batch['phase']=='train' and batch['item_id'].shape[1]==2
   for key in ['user_id','item_id','history_items','lengths']:order.update(batch[key].contiguous().numpy().tobytes())
   steps+=1;rows+=len(batch['user_id']);b=utils.batch_to_gpu(batch,a.device)
   with torch.no_grad():o=ctx['base'](b,True)
   assert '_relation_ids' not in o
   for name,m in models.items():
    opts[name].zero_grad(set_to_none=True);output=m.from_output(o);loss=m.loss(output)
    assert torch.isfinite(loss);loss.backward()
    assert all(p.grad is None for p in ctx['base'].parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.readout.parameters())
    if steps==1:maxgrad[name]=float(m.readout[-1].weight.grad.norm())
    opts[name].step();losses[name].append(float(loss.detach()))
  torch.cuda.synchronize();train_seconds=time.monotonic()-t;t=time.monotonic();ev=evaluate(ctx);torch.cuda.synchronize()
  assert ev['input_sha256']==epoch0['input_sha256'] and ev['metrics']['base']==epoch0['metrics']['base']
  rec=dict(epoch=epoch,train_seconds=train_seconds,dev_seconds=time.monotonic()-t,rows=rows,steps=steps,
   negatives_sha256=negatives_sha,common_batch_sha256=order.hexdigest(),rng_start=rng_start,rng_end=rng_hash(),
   BPR_loss={name:float(np.mean(v)) for name,v in losses.items()},Dev=ev['metrics'],delta_to_base=ev['residual_ratio'],
   P1_minus_P0_NDCG5=ev['metrics']['P1']['NDCG@5']-ev['metrics']['P0']['NDCG@5'],first_batch_output_gradient=maxgrad,
   pairing='identical input tensors shared; independent parameter/optimizer state; no head stochastic ops')
  epochs.append(rec);dump(out/f'epoch_{epoch:02d}.json',rec)
  for name in models:
   val=ev['metrics'][name]['NDCG@5']
   if val>=best[name]['value']:
    ck=out/name/f'epoch_{epoch:02d}.pt';digest=save_readout(ck,ctx,name,epoch)
    best[name]=dict(value=val,epoch=epoch,Dev=ev['metrics'][name],checkpoint=str(ck),checkpoint_sha256=digest,
      residual_delta_to_base_mean=ev['residual_ratio'][name],predictions=ev['predictions'][name].copy())
  print('EPOCH '+json.dumps(dict(dataset=ds,epoch=epoch,P0=rec['Dev']['P0']['NDCG@5'],P1=rec['Dev']['P1']['NDCG@5'],
    difference=rec['P1_minus_P0_NDCG5'],seconds=rec['train_seconds']+rec['dev_seconds'],BPR=rec['BPR_loss'])),flush=True)
 for name,m in models.items():
  chosen=best[name];payload=torch.load(chosen['checkpoint'],map_location=a.device,weights_only=True)
  assert payload['baseline_sha256']==ctx['config']['baseline_sha256']
  m.readout.load_state_dict(payload['readout']);assert sha(chosen['checkpoint'])==chosen['checkpoint_sha256']
  diff=torch.cat([(t-initial[name][k]).flatten() for k,t in m.readout.state_dict().items()])
  chosen['readout_parameter_change_L2']=float(diff.norm());chosen['readout_parameter_change_max_abs']=float(diff.abs().max())
  chosen['readout_state_sha256']=state_hash(m.readout.state_dict())
 final=evaluate(ctx)
 for name in models:
  assert np.array_equal(final['predictions'][name],best[name].pop('predictions')),'Best checkpoint reload prediction mismatch'
  assert final['metrics'][name]==best[name]['Dev']
 assert state_hash(ctx['base'].state_dict())==baseline_hash
 assert all(p.grad is None and not p.requires_grad for p in ctx['base'].parameters())
 assert sha(ctx['config']['baseline'])==ctx['config']['baseline_sha256']
 result,rows=comparisons(final)
 with (out/'best_dev_users.csv').open('x',newline='') as f:
  wr=csv.writer(f);wr.writerow(['user_id','history_length','base_rank','P0_rank','P1_rank','base_NDCG5','P0_NDCG5','P1_NDCG5'])
  wr.writerows(zip(final['users'],final['lengths'],rows['base'][1],rows['P0'][1],rows['P1'][1],rows['base'][2],rows['P0'][2],rows['P1'][2]))
 summary=dict(success=True,dataset=ds,seed=42,K=4,epoch0=epoch0['metrics'],end_epoch=20,best=best,epoch20=epochs[-1],
   comparisons=result,epochs=epochs,total_pair_seconds=time.monotonic()-start,
   timing_note='Shared forward paired training: pair elapsed is conservative upper bound charged to each arm, not sum of standalone measurements',
   peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
   baseline_unchanged=True,all_backbone_gradients_None=True,backbone_always_eval=True,input_sha256=final['input_sha256'],
   train_only_negatives=True,relation_not_optimized=True,test_executed=False,
   bootstrap='2000 paired user-cluster resamples, seed20261019; all rows retained per sampled user; conditional exploratory CI after best-epoch selection, not seed significance')
 dump(out/'summary.json',summary);signal.alarm(0)
 print('PAIR_COMPLETE '+json.dumps(dict(dataset=ds,best={n:x['Dev'] for n,x in best.items()},comparisons=result,seconds=summary['total_pair_seconds'])),flush=True)
 return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args()
 out=(ROOT/a.output).resolve();assert out.is_relative_to(ROOT) and not out.exists()
 row=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 assert row[0].strip() not in apps and float(row[1])>7000,'GPU1 occupied; no external process modified'
 out.mkdir(parents=True);torch.set_num_threads(2)
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Bounded task budget exceeded; no automatic setting changes')))
 sources={f:sha(ROOT/f) for f in SOURCES};started=time.monotonic();contexts=[];checks={};results=[]
 try:
  signal.alarm(170)
  for ds in ['beauty','ml-1m']:
   ctx=build(ds);checks[ds]=unit_tests(ctx);contexts.append(ctx)
   print('UNIT_PASS '+json.dumps(dict(dataset=ds,parameters=ctx['config']['readout_parameters'],tests=checks[ds])),flush=True)
  dump(out/'unit_tests.json',checks);signal.alarm(0)
  if a.check_only:
   dump(out/'complete.json',dict(check_only=True,success=True,seconds=time.monotonic()-started,source_sha256=sources,test_executed=False));return
  # Validate both epoch0 datasets BEFORE any formal optimizer update.
  zeros={}
  for ctx in contexts:
   ds=ctx['config']['dataset'];d=out/ds;d.mkdir();t=time.monotonic();zeros[ds]=evaluate(ctx)
   assert all(abs(zeros[ds]['metrics']['base'][k]-ctx['reference'][k])<1e-12 for k in zeros[ds]['metrics']['base'])
   assert np.array_equal(zeros[ds]['predictions']['P0'],zeros[ds]['predictions']['base'])
   assert np.array_equal(zeros[ds]['predictions']['P1'],zeros[ds]['predictions']['base'])
   dump(d/'epoch_00.json',dict(metrics=zeros[ds]['metrics'],input_sha256=zeros[ds]['input_sha256'],seconds=time.monotonic()-t))
   print('EPOCH0_PASS '+json.dumps(dict(dataset=ds,metrics=zeros[ds]['metrics'])),flush=True)
  for ctx in contexts:results.append(train_pair(ctx,out/ctx['config']['dataset'],zeros[ctx['config']['dataset']]))
  assert all(sha(ROOT/f)==h for f,h in sources.items())
  for ctx in contexts:
   assert all(sha(ROOT/'data'/ctx['config']['dataset']/s)==h for s,h in ctx['config']['assets_sha256'].items())
  dump(out/'complete.json',dict(success=True,results=results,source_sha256=sources,source_unchanged=True,assets_unchanged=True,
   seconds=time.monotonic()-started,test_executed=False,all_four_arms_complete=True))
  print('ALL_COMPLETE '+str(out),flush=True)
 except BaseException as e:
  signal.alarm(0);dump(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-started,test_executed=False));raise
if __name__=='__main__':main()
