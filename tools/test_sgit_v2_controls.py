"""Finite R/B/S isolation, old-Full regression and Train-only verification. No Dev/Test scoring."""
import sys,os,json,time,copy,signal,argparse,importlib.util,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch,numpy as np
from gict_data import args_from_config,train_corpus,sha
from run_sgitrec import core_hash
from test_sgitrec import make_feed
from models.sequential.SGITRec import SGITRec
from models.sequential.SGITRecV2 import SGITRecV2
from models.sequential.SGITRecV2Controlled import SGITRecV2Controlled
from models.sequential.sgit_v2_control_components import FACTORS,ControlledBridgeMemory
from models.sequential.sgit_components import InterestGraphMemory
from models.sequential.LLMMIRecASPCFAdaptiveSIL import LLMMIRecASPCFAdaptiveSIL as C1,segment_ids
from utils import utils
OUT=ROOT/'research/sgitrec_v2_controls_20261025_v1'
OLD=ROOT/'research/sgitrec_v2_20261025_v1'
def edge(B,L,R):
 return dict(ids=torch.ones(B,L,R,dtype=torch.long),prob=torch.rand(B,L,R).softmax(-1),support=torch.randint(1,10,(B,L,R)).float())
def clone_for(full,variant):
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(20261025);m=ControlledBridgeMemory(64,FACTORS[variant])
 m.load_state_dict({k:v for k,v in full.state_dict().items() if k in m.state_dict()},strict=True)
 return m
def math_check():
 reports=[]
 for K in [1,2,3,4]:
  torch.manual_seed(7);B,L,D,R=4,5,64,8
  v=torch.randn(B,K,D,requires_grad=True);h=torch.randn(B,L,D,requires_grad=True)
  A=torch.randn(B,K,L,requires_grad=True).softmax(-1)
  valid=torch.arange(L)[None]<torch.tensor([0,1,3,5])[:,None];A=A.masked_fill(~valid[:,None],0)
  t=edge(B,L,R);b=edge(B,L,R);cv=torch.randn(B,L,R,D,requires_grad=True);bv=torch.randn_like(cv,requires_grad=True)
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261025);full=ControlledBridgeMemory(D,FACTORS['V2-Full'])
  for variant in FACTORS:
   m=clone_for(full,variant);y,z=m(v,A,valid,t,cv,b,bv,h);assert torch.equal(y,v)
   if not FACTORS[variant]['R']:assert torch.equal(z['calibrated_attention'],A)
   # Branch-disabled model must never dereference even a malicious bridge object.
   if not FACTORS[variant]['B']:
    class Forbidden:
     def __getitem__(self,key):raise AssertionError('Bridge data accessed')
    m(v,A,valid,t,cv,Forbidden(),None,h)
    assert not any(n.startswith(('bridge_output.','bridge_match.')) for n,p in m.named_parameters())
   opt=torch.optim.Adam(m.parameters(),lr=.01)
   for _ in range(4):
    opt.zero_grad();y,z=m(v,A,valid,t,cv,b,bv,h)
    (y.square().sum()+y.sum()).backward(retain_graph=True);opt.step()
   for name,p in m.named_parameters():
    assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,(K,variant,name)
   assert torch.equal(y[0],v[0])
   empty={n:torch.zeros_like(x) for n,x in t.items()}
   absent,_=m(v,A,valid,empty,cv,empty,bv,h);assert torch.equal(absent,v)
  with torch.no_grad():
   full.output.weight.normal_(0,.02);full.bridge_output.weight.normal_(0,.02)
  noS=clone_for(full,'V2-NoS')
  _,f=full(v,A,valid,t,cv,b,bv,h);_,s=noS(v,A,valid,t,cv,b,bv,h)
  for name in ['collaborative_memory','bridge_memory','transition_route','bridge_route','source_support']:
   assert torch.equal(f[name],s[name]),name
  assert torch.allclose(s['bridge_beta'],f['bridge_beta']*(1+torch.log1p(f['source_support'])),atol=1e-7,rtol=1e-6)
  direct=clone_for(full,'V2-Direct')
  old=InterestGraphMemory(64,False)
  old.load_state_dict({k:v for k,v in direct.state_dict().items() if k in old.state_dict()},strict=True)
  ref,ri=old(v,A,valid,t,cv,variant='G2')
  got,gi=direct(v,A,valid,t,cv,None,None,h)
  assert torch.equal(ref,got) and torch.equal(ri['transition_route'],gi['transition_route'])
  reports.append(dict(K=K,all_factors_gradients=True,NoS_changes_only_beta_denominator=True,
      matched_direct_equals_G2_memory=True,disabled_bridge_unread=True,zero_and_empty_exact=True))
 return dict(passed=True,math=reports)
def config_path(ds,seed):
 old=json.loads((OLD/'PLAN.json').read_text())['datasets'][ds]
 if seed==42:return old['config']
 return str(ROOT/'research/elector_sil_night12_20261023_v1/runs'/('%s_k%d_s0_SS-C1'%(ds,old['K']))/'config.json')
def setup(ds,seed=42):
 a=args_from_config(config_path(ds,seed));a.device=torch.device('cpu')
 a.sgit_graph_path=str(OLD/'graphs'/(ds+'_bridge_v2.pt'))
 return a,train_corpus(a)
def real_check(ds):
 a,c=setup(ds);feed=make_feed(c,'cpu',16)
 oldpath=ROOT/'research/sgitrec_v1_20261024/graphs'/(ds+'_v2.pt')
 bp=torch.load(a.sgit_graph_path,map_location='cpu',weights_only=True)
 tp=torch.load(oldpath,map_location='cpu',weights_only=True)
 for name,value in tp.items():
  if name.startswith('transition_'):assert torch.equal(value,bp[name]),name
 utils.init_seed(42);original=SGITRecV2(a,c).eval()
 initial=copy.deepcopy(original.state_dict())
 with torch.no_grad():
  original.graph_memory.output.weight.normal_(0,.03)
  original.graph_memory.bridge_output.weight.normal_(0,.03)
 canonical=original.state_dict()
 with torch.no_grad():ref=original(feed,True)
 counts={};hashes={};rngs={}
 for variant in FACTORS:
  a.sgit_control=variant;a.sgit_graph_path=str(oldpath) if not FACTORS[variant]['B'] else str(OLD/'graphs'/(ds+'_bridge_v2.pt'))
  utils.init_seed(42);m=SGITRecV2Controlled(a,c).eval();rngs[variant]=torch.get_rng_state().clone()
  counts[variant]=m.count_variables();hashes[variant]=core_hash(m)
  # All shared graph parameters keep the exact canonical Full initialization.
  for n,p in m.graph_memory.named_parameters():
   assert torch.equal(p,initial['graph_memory.'+n]),(variant,n)
  direct=m.graph_cache.lookup(feed['history_items'],feed['user_id'],'transition')
  fold=feed['user_id']%3
  for n,x in direct.items():
   assert torch.equal(x,tp['transition_'+n][fold[:,None],feed['history_items']]),n
  m.load_state_dict({k:v for k,v in canonical.items() if k in m.state_dict()},strict=True)
  if not FACTORS[variant]['B']:
   assert not any(n.startswith('bridge_') for n,b in m.graph_cache.named_buffers())
   actual=m.graph_cache.lookup
   def guarded(history,users,rel):
    assert rel=='transition';return actual(history,users,rel)
   m.graph_cache.lookup=guarded
  with torch.no_grad():
   out=m(feed,True)
   if variant=='V2-Full':
    assert torch.equal(out['prediction'],ref['prediction'])
    assert torch.equal(out['interest_vectors'],ref['interest_vectors'])
   if not FACTORS[variant]['R']:assert torch.equal(out['calibrated_attention'],out['attention_maps'])
   m.sgit_enabled=False;off=m(feed,True)
   bare=C1(a,c).eval();bare.load_state_dict({n:p for n,p in m.state_dict().items() if not n.startswith('graph_memory.')},strict=True)
   expected=bare(feed,True);assert torch.equal(off['prediction'],expected['prediction'])
   m.sgit_enabled=True
   other=dict(feed);other['item_id']=feed['item_id'].flip(0);o2=m(other,True)
   assert torch.allclose(out['interest_vectors'],o2['interest_vectors'],atol=3e-6,rtol=3e-6)
   bad=dict(feed);bad['history_items']=feed['history_items'].clone()
   bad['history_items'][torch.arange(20)[None]>=feed['lengths'][:,None]]=3
   assert torch.allclose(out['interest_vectors'],m(bad,True)['interest_vectors'],atol=3e-6,rtol=3e-6)
  del m,bare
 assert len(set(hashes.values()))==1
 assert all(torch.equal(next(iter(rngs.values())),x) for x in rngs.values())
 # Full preserves OLD V2 training stochastic path and loss, not just eval.
 a,c2=setup(ds);a.sgit_control='V2-Full'
 utils.init_seed(42);full=SGITRecV2Controlled(a,c2).train()
 utils.init_seed(42);old=SGITRecV2(a,c2).train()
 full.cur_epoch=old.cur_epoch=1
 rng=torch.get_rng_state()
 fo=full(feed,True);fl=full.loss(fo);frng=torch.get_rng_state()
 torch.set_rng_state(rng);oo=old(feed,True);ol=old.loss(oo);orng=torch.get_rng_state()
 assert torch.equal(fo['prediction'],oo['prediction']) and torch.equal(fl,ol) and torch.equal(frng,orng)
 fl.backward();ol.backward()
 for (n,p),(nn,q) in zip(full.named_parameters(),old.named_parameters()):
  assert n==nn
  if q.grad is not None:assert torch.allclose(p.grad,q.grad,atol=2e-7,rtol=2e-5),n
 return dict(passed=True,dataset=ds,parameters=counts,C1_common_init_hash=hashes['V2-Full'],
    factor_shared_initial_rng_exact=True,original_Full_train_eval_loss_rng_exact=True,
    old_direct_graph_all_tensors_equal=True,old_relation_function_same=True,
    no_bridge_file_or_lookup_or_parameters_in_R=True,graph_off_C1_exact=True,padding=True,candidate_independent=True,
    historical_V1_caveat='V1 graph RNG seed20261024 vs V2 seed20261025; V1 candidate reuse vs V2 history/direct-only reuse. Cross-version deltas are not single-factor ablations.')
def seed_checks(ds):
 results={}
 for seed in [0,42]:
  a,c=setup(ds,seed)
  utils.init_seed(seed);base=C1(a,c)
  h=core_hash(base)
  asp=dict((n,p.detach().clone()) for n,p in base.named_parameters() if not n.startswith('sil_block.'))
  original_cfg=json.loads(Path(config_path(ds,seed)).read_text())
  if seed==0:
   import hashlib
   hh=hashlib.sha256()
   for n,p in asp.items():hh.update(n.encode());hh.update(p.numpy().tobytes())
   assert hh.hexdigest()==original_cfg['common_initial_parameter_sha256']
   # Historical C1 adapter and current C1 share the entire initial state.
   path=ROOT/'research/elector_sil_night12_20261023_v1/objective_model.py'
   spec=importlib.util.spec_from_file_location('historical_objective',path)
   hist=importlib.util.module_from_spec(spec);spec.loader.exec_module(hist)
   utils.init_seed(seed);legacy=hist.ObjectiveSIL(a,c)
   lp=dict(legacy.named_parameters())
   for n,p in base.named_parameters():assert n in lp and torch.equal(p,lp[n]),n
   cp=Path(config_path(ds,seed)).parent/'best.pt'
   base.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True),strict=True)
   r=json.loads((cp.parent/'result.json').read_text())
   assert r['Test_accepted'] and r['Test_passes']==1 and sha(cp)==r['checkpoint_sha256']
   expected_hash=json.loads((ROOT/'research/gictrec_round1_20261024_v1'/(ds+'_test_PROTOCOL.json')).read_text())['input_hash']
   assert r['Test_input_hash']==expected_hash
  results[str(seed)]=dict(C1_initial_sha256=h,source_config=config_path(ds,seed),config_sha256=sha(config_path(ds,seed)),historical_C1_verified=True)
  del base,c
 return dict(passed=True,dataset=ds,seeds=results)

def gpu_check(ds,gpu):
 spec=importlib.util.spec_from_file_location('controls_resource',ROOT/'research/elector_sil_objective_full_20261023_v2/scheduler.py')
 resource=importlib.util.module_from_spec(spec);spec.loader.exec_module(resource)
 inv=resource.observe();assert resource.idle(inv[gpu]),'GPU busy; never preempt'
 lease=resource.acquire(ROOT/('research/gpu_resource_scheduler_20261022_v1/gpu%d.lock'%gpu));assert lease is not None
 try:
  assert resource.idle(resource.observe()[gpu])
  os.environ['CUDA_VISIBLE_DEVICES']=inv[gpu]['uuid']
  cache=OUT/'checks'/('cuda_'+ds);cache.mkdir()
  os.environ['CUDA_CACHE_PATH']=str(cache);torch.cuda.set_device(0)
  a,c=setup(ds);a.device=torch.device('cuda:0')
  feed=make_feed(c,a.device,32);big=make_feed(c,a.device,1024);result={}
  for variant in ['V2-R','V2-B','V2-NoS','V2-Full']:
   a.sgit_control=variant
   a.sgit_graph_path=str(ROOT/'research/sgitrec_v1_20261024/graphs'/(ds+'_v2.pt')) if not FACTORS[variant]['B'] else str(OLD/'graphs'/(ds+'_bridge_v2.pt'))
   utils.init_seed(42);m=SGITRecV2Controlled(a,c).to(a.device)
   m.cur_epoch=1;m.train();opt=torch.optim.Adam(m.customize_parameters(),lr=a.lr,weight_decay=a.l2)
   for step in range(6):
    opt.zero_grad();o=m(feed,True);loss=m.loss(o)
    assert torch.isfinite(loss);loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    opt.step()
   for name,p in m.graph_memory.named_parameters():
    assert p.grad is not None and p.grad.abs().sum()>0,(variant,name)
   assert m.extractor.query.grad.abs().sum()>0 and m.sil_block.local_qkv.weight.grad.abs().sum()>0
   # No target/negative features decide graph routing: replay identical training RNG.
   rng=torch.get_rng_state();crng=torch.cuda.get_rng_state()
   o1=m(feed,True)
   torch.set_rng_state(rng);torch.cuda.set_rng_state(crng)
   other=dict(feed);other['item_id']=feed['item_id'].flip(0);o2=m(other,True)
   assert torch.allclose(o1['interest_vectors'],o2['interest_vectors'],atol=3e-6,rtol=3e-6)
   opt.zero_grad();loss=m.loss(m(big,True));loss.backward();opt.step()
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
   for _ in range(2):
    opt.zero_grad();o=m(big,True);loss=m.loss(o);loss.backward();opt.step()
   torch.cuda.synchronize();elapsed=(time.perf_counter()-tick)/2
   result[variant]=dict(parameters=m.count_variables(),step_seconds=elapsed,batch=1024,
     peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,graph_buffer_bytes=m.graph_cache.bytes(),
     real_gradients_nonzero=True,training_candidates_independent=True)
   del m,opt,o,o1,o2;torch.cuda.empty_cache()
  return dict(passed=True,dataset=ds,variants=result)
 finally:os.close(lease)

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['math','real','seed','gpu'],required=True);p.add_argument('--dataset',choices=['ml-1m','beauty'])
 p.add_argument('--gpu',type=int);x=p.parse_args();signal.alarm(170);torch.set_num_threads(4);tick=time.perf_counter()
 got=math_check() if x.mode=='math' else real_check(x.dataset) if x.mode=='real' else gpu_check(x.dataset,x.gpu) if x.mode=='gpu' else seed_checks(x.dataset)
 got.update(seconds=time.perf_counter()-tick,Dev=False,Test=False)
 dest=OUT/'checks'/((x.dataset+'_'+x.mode) if x.dataset else x.mode)
 with dest.with_suffix('.json').open('x') as f:json.dump(got,f,indent=2)
 print(json.dumps(got,indent=2))
if __name__=='__main__':main()
