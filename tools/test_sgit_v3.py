"""Bounded sparse-join reference, C1 equivalence, Train isolation and real resource tests."""
import sys,os,json,time,signal,argparse,copy,importlib.util,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np,torch
from gict_data import args_from_config,train_corpus,sha
from test_sgitrec import make_feed
from sgit_graph import build_graphs
from sgit_v2_graph import build_bridge
from build_sgit_v3_index import build_index
from models.sequential.sgit_v3_components import sparse_candidate_read,CandidateTransitionScorer
from models.sequential.SGITRecV3 import SGITRecV3
from models.sequential.LLMMIRecASPCFAdaptiveSIL import LLMMIRecASPCFAdaptiveSIL as C1,segment_ids
from utils import utils
OUT=ROOT/'research/sgitrec_v3_20261025_v1'
def core_hash(m):
 import hashlib
 h=hashlib.sha256()
 for n,p in m.named_parameters():
  if not n.startswith('v3_scorer.'):h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def math():
 torch.manual_seed(33);proof=[]
 for K in [1,2,3,4]:
  B,L,R,D=3,4,3,64
  A=torch.randn(B,K,L).softmax(-1).requires_grad_()
  valid=torch.arange(L)[None]<torch.tensor([0,1,4])[:,None]
  ids=torch.tensor([[[1,2,3],[2,4,5],[3,5,6],[1,7,9]]]*B)
  values=torch.rand(B,L,R);edges=dict(ids=ids,value=values)
  candidates=torch.tensor([[9,3,3,99,1,5,2],[1,2,1,99,3,5,9],[3,9,3,99,1,5,2]])
  sparse=sparse_candidate_read(A,valid,edges,candidates)
  dense=(A[...,None,None]*values[:,None,:,None,:]*
    ((ids[:,:,None,:]==candidates[:,None,:,None])&valid[:,:,None,None])[:,None]).sum((2,4))
  assert torch.allclose(sparse,dense,atol=2e-7,rtol=1e-6)
  assert torch.equal(sparse[2,:,0],sparse[2,:,2])
  assert torch.count_nonzero(sparse[:,:,3])==0
  gd=torch.autograd.grad(dense.sum(),A,retain_graph=True)[0]
  gs=torch.autograd.grad(sparse.sum(),A,retain_graph=True)[0]
  assert torch.allclose(gd,gs,atol=1e-6)
  v=torch.randn(B,K,D,requires_grad=True);weights=torch.ones(B,K)/K
  table=torch.randn(101,D,requires_grad=True);e=table[candidates]
  m=CandidateTransitionScorer(D,chunk=3)
  zero,_=m(v,A,weights,valid,candidates,e,edges,edges,True)
  assert torch.count_nonzero(zero)==0
  opt=torch.optim.Adam(m.parameters(),lr=.01)
  for step in range(4):
   opt.zero_grad();delta,detail=m(v,A,weights,valid,candidates,e,edges,edges,True)
   loss=(delta*torch.arange(1,8)).sum()+.01*delta.square().sum()
   loss.backward(retain_graph=True);opt.step()
  for name,p in m.named_parameters():
   assert p.grad is not None and torch.isfinite(p.grad).all(),(K,name)
   if K>1 or name=='scale':assert p.grad.abs().sum()>0,(K,name)
  assert torch.count_nonzero(delta[0])==0 and torch.count_nonzero(delta[:,3])==0
  none,_=m(v,A,weights,valid,candidates,e,None,None)
  assert torch.count_nonzero(none)==0
  # Independent direct and bridge controls with known scales.
  with torch.no_grad():m.scale.copy_(torch.tensor([1.,0.]))
  td,_=m(v,A,weights,valid,candidates,e,edges,None)
  both,_=m(v,A,weights,valid,candidates,e,edges,edges)
  assert torch.equal(td,both)
  perm=torch.tensor([6,2,0,5,1,3,4])
  pdelta,_=m(v,A,weights,valid,candidates[:,perm],e[:,perm],edges,None)
  assert torch.allclose(pdelta,td[:,perm],atol=2e-7,rtol=2e-6)
  proof.append(dict(K=K,sparse_equals_dense=True,duplicate_slots=True,no_path_zero=True,zero_init=True,finite_gradients=True))
 # Explicit different candidates select different interests.
 m=CandidateTransitionScorer(64)
 with torch.no_grad():m.query.weight.copy_(torch.eye(64));m.key.weight.copy_(torch.eye(64))
 v=torch.eye(64)[:2][None];e=v.clone();alpha=m.alpha(v,torch.ones(1,2)/2,e)
 assert alpha[0,:,0].argmax()!=alpha[0,:,1].argmax()
 histories={1:[(1,1),(2,2),(3,3)],2:[(4,1),(5,2),(6,3)],3:[(1,1),(7,2),(8,3)],4:[(4,1),(5,2),(1,3)]}
 table=torch.randn(10,7);table[0]=0
 t,meta=build_graphs(histories,table)
 base=dict(metadata=dict(meta,schema='sgit_crossfit_item_graph_v1',scope='full_crossfit_train'),**t)
 bridge=build_bridge(base);idx,stats=build_index(bridge,histories)
 changed=copy.deepcopy(histories);changed[3]=[(9,1),(8,2),(1,3),(4,4)]
 tt,_=build_graphs(changed,table);bb=build_bridge(dict(metadata=base['metadata'],**tt));again,_=build_index(bb,changed)
 for key,value in idx.items():assert torch.equal(value[0],again[key][0]),key
 for rel in ['transition','bridge']:
  assert torch.equal(idx[rel+'_ids'],bridge[rel+'_ids']) and torch.equal(idx[rel+'_prob'],bridge[rel+'_prob'])
 return dict(math=proof,candidate_interest_selection_different=True,fold_future_invariance_including_calibration=True,
             graph_ids_prob_unchanged=True,shared_batch_candidate_and_duplicate_mapping=True)
def config(ds):
 old=json.loads((ROOT/'research/sgitrec_v2_20261025_v1/PLAN.json').read_text())['datasets'][ds]
 a=args_from_config(old['config']);a.v3_graph_path=str(OUT/'graphs'/(ds+'_candidate_v3.pt'));a.v3_candidate_chunk=128
 return a
def real(ds,gpu):
 lease=None;device='cpu'
 if gpu is not None:
  spec=importlib.util.spec_from_file_location('v3_resource',ROOT/'research/elector_sil_objective_full_20261023_v2/scheduler.py')
  r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
  inv=r.observe();assert r.idle(inv[gpu]),'Busy GPU: wait, never preempt'
  lease=r.acquire(ROOT/('research/gpu_resource_scheduler_20261022_v1/gpu%d.lock'%gpu));assert lease is not None
  assert r.idle(r.observe()[gpu])
  os.environ['CUDA_VISIBLE_DEVICES']=inv[gpu]['uuid']
  cache=OUT/'checks'/('cuda_'+ds);cache.mkdir()
  os.environ['CUDA_CACHE_PATH']=str(cache);torch.cuda.set_device(0);device='cuda:0'
 try:
  a=config(ds);a.device=torch.device(device);c=train_corpus(a);feed=make_feed(c,device,64)
  params={}
  for K in [1,2,3,4]:
   a.K=K;utils.init_seed(42);ref=C1(a,c).to(device).eval()
   utils.init_seed(42);m=SGITRecV3(a,c).to(device).eval()
   assert core_hash(m)==core_hash(ref)
   with torch.no_grad():old=ref(feed,True);new=m(feed,True)
   assert torch.equal(old['prediction'],new['prediction']) and new['interest_vectors'].shape==(64,K,64)
   params[str(K)]=m.count_variables();del ref,m
  a.K=2 if ds=='beauty' else 3
  utils.init_seed(42);ref=C1(a,c).to(device)
  utils.init_seed(42);m=SGITRecV3(a,c).to(device);m.cur_epoch=ref.cur_epoch=1
  m.train();ref.train();rng=torch.get_rng_state();crng=torch.cuda.get_rng_state() if gpu is not None else None
  out=m(feed,True);loss=m.loss(out);loss.backward()
  torch.set_rng_state(rng)
  if crng is not None:torch.cuda.set_rng_state(crng)
  old=ref(feed,True);oldloss=ref.loss(old);oldloss.backward()
  assert torch.equal(loss,oldloss) and torch.equal(out['_ss_allowed'],old['_ss_allowed'])
  for name,p in ref.named_parameters():
   q=dict(m.named_parameters())[name]
   if p.grad is not None:assert torch.allclose(p.grad,q.grad,atol=2e-7,rtol=2e-5),name
  assert m.v3_scorer.scale.grad.abs().sum()>0
  opt=torch.optim.Adam(m.customize_parameters(),lr=a.lr,weight_decay=a.l2)
  for _ in range(6):
   opt.zero_grad();out=m(feed,True);loss=m.loss(out);assert torch.isfinite(loss);loss.backward()
   assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
   opt.step()
  for name,p in m.v3_scorer.named_parameters():assert p.grad is not None and p.grad.abs().sum()>0,name
  assert m.extractor.query.grad.abs().sum()>0 and m.sil_block.local_qkv.weight.grad.abs().sum()>0
  m.eval()
  with torch.no_grad():
   out=m(feed,True)
   cand=feed['item_id'].clone()
   # Select genuinely nonpadding Train items absent from BOTH history path sets.
   for row in range(len(cand)):
    blocked=set()
    for rel in ['transition','bridge']:
     blocked.update(m.candidate_graph.lookup(feed['history_items'][row:row+1],feed['user_id'][row:row+1],rel)['ids'].flatten().cpu().tolist())
    catalog=c.data_df['train'].item_id.unique()
    absent=next(int(i) for i in catalog if int(i) not in blocked)
    cand[row]=absent
   miss=dict(feed,item_id=cand);on=m(miss,True)
   m.v3_enabled=False;off=m(miss,True)
   assert torch.equal(on['raw_prediction'],off['raw_prediction']) and torch.equal(on['prediction'],off['prediction'])
   m.v3_enabled=True
   other=dict(feed,item_id=feed['item_id'].flip(0));o2=m(other,True)
   assert torch.equal(out['interest_vectors'],o2['interest_vectors']) and torch.equal(out['interest_weights'],o2['interest_weights'])
   bad=dict(feed);bad['history_items']=feed['history_items'].clone()
   bad['history_items'][torch.arange(20,device=device)[None]>=feed['lengths'][:,None]]=3
   assert torch.equal(m(bad,True)['prediction'],out['prediction'])
   empty=dict(feed,history_items=feed['history_items'][:,:0],lengths=feed['lengths']*0)
   eo=m(empty,True);assert torch.count_nonzero(eo['raw_prediction'])==0 and torch.isfinite(eo['prediction']).all()
   m.direct_enabled=False;m.bridge_enabled=False;off=m(feed,True)
   ref.load_state_dict({n:p for n,p in m.state_dict().items() if not n.startswith('v3_scorer.')},strict=True);ref.eval()
   assert torch.equal(off['prediction'],ref(feed,True)['prediction'])
   m.direct_enabled=True;m.bridge_enabled=True
  x=torch.randn(1,8,64,device=device,requires_grad=True)
  ids,valid=segment_ids(x.new_zeros(1,8),torch.tensor([8],device=device),'fixed',0.,fixed_len=4)
  y=m.sil_block.local_bank(x,ids,valid)
  assert torch.count_nonzero(torch.autograd.grad(y[0,2].square().sum(),x)[0][0,3:])==0
  big=make_feed(c,device,1024);m.train()
  opt.zero_grad();loss=m.loss(m(big,True));loss.backward();opt.step()
  if gpu is not None:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
  tick=time.perf_counter()
  for _ in range(2):
   opt.zero_grad();o=m(big,True);loss=m.loss(o);loss.backward();opt.step()
  if gpu is not None:torch.cuda.synchronize()
  step=(time.perf_counter()-tick)/2
  train_peak=torch.cuda.max_memory_allocated()/2**20 if gpu is not None else None
  # Real Train-derived synthetic candidate protocol cost, NOT Dev/Test metrics.
  batch=128 if gpu is not None else 16;ev=make_feed(c,device,batch)
  catalog=torch.tensor(sorted(c.data_df['train'].item_id.unique()),device=device)
  candidates=catalog[torch.randint(len(catalog),(batch,1001),device=device)]
  candidates[:,5]=candidates[:,0];candidates[:,1000]=candidates[:,0];ev['item_id']=candidates
  m.eval()
  with torch.inference_mode():
   if gpu is not None:torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
   tick=time.perf_counter();pred=m(ev,True)
   if gpu is not None:torch.cuda.synchronize()
   evsec=time.perf_counter()-tick
   assert torch.equal(pred['prediction'][:,0],pred['prediction'][:,5])
   assert torch.equal(pred['prediction'][:,0],pred['prediction'][:,1000]),'Cross-chunk duplicate scores differ'
   assert pred['prediction'].shape==(batch,1001) and torch.isfinite(pred['prediction']).all()
  return dict(dataset=ds,device=device,parameters=params,K=a.K,initial_C1_score_Loss_gradients_exact=True,
   all_graph_off_C1_exact=True,no_path_nonpadding_exact=True,real_score_branch_and_C1_gradients=True,
   scale_values=m.v3_scorer.scale.detach().cpu().tolist(),history_independent_of_candidates=True,
   padding_empty_short=True,causality=True,duplicate_1001_slots_exact=True,filtered_mask_unchanged=True,
   sparse_join='B*K*(C+L*R); no B*N or B*C*L*R intermediates',
   train_batch=1024,train_step_seconds=step,train_peak_GPU_MiB=train_peak,
   evaluation_batch=batch,evaluation_candidates=1001,evaluation_seconds=evsec,
   evaluation_peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20 if gpu is not None else None,
   peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
   graph_device_bytes=m.candidate_graph.bytes(),graph_file_bytes=Path(a.v3_graph_path).stat().st_size,Dev=False,Test=False)
 finally:
  if lease is not None:os.close(lease)
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['math','real'],required=True)
 p.add_argument('--dataset',choices=['ml-1m','beauty']);p.add_argument('--gpu',type=int)
 x=p.parse_args();signal.alarm(170);torch.set_num_threads(4);tick=time.perf_counter()
 got=math() if x.mode=='math' else real(x.dataset,x.gpu)
 got.update(passed=True,seconds=time.perf_counter()-tick)
 label='math' if x.mode=='math' else x.dataset+('_gpu' if x.gpu is not None else '_cpu')
 with (OUT/'checks'/(label+'.json')).open('x') as f:json.dump(got,f,indent=2)
 print(json.dumps(got,indent=2))
if __name__=='__main__':main()
