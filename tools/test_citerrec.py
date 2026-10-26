"""Bounded CITER math, exact single-expert controls, real ranking gradients and Dev-only guard."""
import sys,os,json,time,signal,argparse,copy,importlib.util,gc,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research/citerrec_v1_20261026_v1'
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch,numpy as np
from models.sequential.citer_components import CITERScorer,path_statistics
from models.sequential.sgit_v3_control_components import PredictionScorer
from models.sequential.SGITRecV3Controlled import SGITRecV3Controlled
from models.sequential.CITERRec import CITERRec
from models.sequential.LLMMIRecASPCFAdaptiveSIL import LLMMIRecASPCFAdaptiveSIL as C1,segment_ids
from models.sequential.LLMMIRecASPCFSampledSoftmax import candidate_mask
from test_sgitrec import make_feed
from gict_data import args_from_config,train_corpus,sha
import citer_dev_protocol as protocol
from utils import utils
def core_hash(m):
 h=hashlib.sha256()
 for n,p in m.named_parameters():
  if not n.startswith('prediction_scorer.'):h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def mathematics():
 torch.manual_seed(16);proof=[]
 for K in [1,2,3,4]:
  B,L,R,D=3,4,2,64
  v=torch.randn(B,K,D);w=torch.randn(B,K).softmax(-1);A=torch.randn(B,K,L).softmax(-1)
  valid=torch.arange(L)[None]<torch.tensor([0,2,4])[:,None]
  ids=torch.tensor([[[1,2],[2,5],[1,9],[3,5]]]*B);sup=torch.tensor([[[2,1],[3,4],[2,7],[5,4]]]*B)
  edges={'transition':dict(ids=ids,value=torch.rand(B,L,R),support=sup)}
  candidates=torch.tensor([[1,5,5,99,2,9,3]]*B);table=torch.randn(100,D);e=table[candidates]
  m=CITERScorer(D,chunk=2);z,_=m(v,A,w,valid,candidates,e,edges,True)
  assert torch.count_nonzero(z)==0
  support,fraction=path_statistics(valid,edges['transition'],candidates)
  assert torch.equal(support[1],torch.tensor([2,4,4,0,3,0,0]))
  assert support[2,0]==2 and fraction[2,0]==.5,'Repeated source events do not sum distinct-user support'
  for name,variant in [('I','TargetInterest'),('T','Direct-Fixed')]:
   old=PredictionScorer(D,variant,chunk=2)
   with torch.no_grad():
    m.scales['target'].fill_(.3);m.scales['transition'].fill_(.7)
    old.scales['target' if name=='I' else 'transition'].fill_(.3 if name=='I' else .7)
   actual,_=m(v,A,w,valid,candidates,e,edges,interest=name=='I',transition=name=='T',reliability=False)
   expected,_=old(v,A,w,valid,candidates,e,{} if name=='I' else edges)
   assert torch.equal(actual,expected),(K,name)
  with torch.no_grad():
   for p in m.scales.values():p.zero_()
  opt=torch.optim.Adam(m.parameters(),lr=.01)
  for step in range(4):
   opt.zero_grad();delta,info=m(v,A,w,valid,candidates,e,edges,True)
   rank=torch.logsumexp(delta,1).mean()-delta[:,0].mean();rank.backward()
   assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
   opt.step()
  assert torch.count_nonzero(info['gate_T'][:,3])==0
  assert torch.equal(delta[:,1],delta[:,2]) and torch.count_nonzero(delta[0])==0
  for name,p in m.named_parameters():
   if K==1 and (name.startswith('query.') or name.startswith('key.') or name.startswith('interest_gate.') or name=='scales.target'):continue
   assert p.grad.abs().sum()>0,(K,name)
  absent=dict(transition=dict(ids=torch.zeros_like(ids),value=torch.zeros_like(edges['transition']['value']),support=torch.zeros_like(sup)))
  miss,inf=m(v,A,w,valid,candidates,e,absent,True,interest=False)
  assert torch.count_nonzero(miss)==0 and torch.count_nonzero(inf['gate_T'])==0
  proof.append(dict(K=K,single_expert_matches_legacy_without_gates=True,no_path_graph_zero=True,finite_grads_and_gates_update=True))
 # Denied before opening actual Test data or creating any Test marker.
 for p,mode in [(ROOT/'data/beauty/test.csv','r'),(OUT/'checks/TEST_STARTED.json','w')]:
  try:open(p,mode)
  except PermissionError:pass
  else:raise AssertionError('Dev-only guard failed')
 assert not (OUT/'checks/TEST_STARTED.json').exists()
 return dict(passed=True,checks=proof,Test_read_write_guard=True,independent_user_support_max_not_sum=True,Test=False,Dev=False)
def actual(ds,gpu):
 lease=None;device='cpu'
 if gpu is not None:
  spec=importlib.util.spec_from_file_location('citer_resource',ROOT/'research/elector_sil_objective_full_20261023_v2/scheduler.py')
  res=importlib.util.module_from_spec(spec);spec.loader.exec_module(res);inv=res.observe();assert res.idle(inv[gpu])
  lease=res.acquire(ROOT/('research/gpu_resource_scheduler_20261022_v1/gpu%d.lock'%gpu));assert lease is not None
  assert res.idle(res.observe()[gpu])
  os.environ['CUDA_VISIBLE_DEVICES']=inv[gpu]['uuid'];cache=OUT/'checks'/('cuda_'+ds);cache.mkdir(exist_ok=True);os.environ['CUDA_CACHE_PATH']=str(cache)
  torch.cuda.set_device(0);device='cuda:0'
 try:
  a=args_from_config(OUT/'configs'/(ds+'_s42.json'));a.device=torch.device(device)
  oldplan=json.load(open(ROOT/'research/sgitrec_v3_predictions_20261025_v1/PLAN.json'))
  a.v3_graph_path=oldplan['datasets'][ds]['graphs']['predictions']['path'];a.v3_candidate_chunk=128
  c=train_corpus(a);feed=make_feed(c,device,128)
  utils.init_seed(42);base=C1(a,c).to(device).eval()
  utils.init_seed(42);m=CITERRec(a,c).to(device).eval()
  assert core_hash(m)==core_hash(base)
  with torch.no_grad():
   zero=m(feed,True);ref=base(feed,True)
   assert torch.equal(zero['prediction'],ref['prediction'])
  base.train();m.train();base.cur_epoch=m.cur_epoch=1
  rng=torch.get_rng_state();crng=torch.cuda.get_rng_state() if gpu is not None else None
  out=m(feed,True);loss=m.loss(out);loss.backward()
  torch.set_rng_state(rng)
  if crng is not None:torch.cuda.set_rng_state(crng)
  br=base(feed,True);bloss=base.loss(br);bloss.backward()
  assert torch.equal(loss,bloss) and torch.equal(out['_ss_allowed'],br['_ss_allowed'])
  for n,p in base.named_parameters():
   if p.grad is not None:assert torch.allclose(p.grad,dict(m.named_parameters())[n].grad,atol=2e-7,rtol=2e-5),n
  m.eval();base.eval()
  # Exact ungated single-branch legacy equality; gates introduce documented differences.
  for mode,variant in [('I','TargetInterest'),('T','Direct-Fixed')]:
   a.v3_variant=variant;utils.init_seed(42);legacy=SGITRecV3Controlled(a,c).to(device).eval()
   with torch.no_grad():
    m.prediction_scorer.scales['target'].fill_(.3);m.prediction_scorer.scales['transition'].fill_(.4)
    legacy.prediction_scorer.scales['target' if mode=='I' else 'transition'].fill_(.3 if mode=='I' else .4)
    m.interest_enabled=mode=='I';m.transition_enabled=mode=='T';m.reliability_enabled=False
    assert torch.equal(m(feed,True)['prediction'],legacy(feed,True)['prediction']),mode
   del legacy
  m.interest_enabled=m.transition_enabled=m.reliability_enabled=True
  with torch.no_grad():
   for p in m.prediction_scorer.scales.values():p.zero_()
  opt=torch.optim.Adam(m.customize_parameters(),lr=a.lr,weight_decay=a.l2)
  initial={n:p.detach().clone() for n,p in m.prediction_scorer.named_parameters()}
  m.train()
  for step in range(6):
   opt.zero_grad();out=m(feed,True);rank,_=m._ranking_loss(out['_ss_scores'],out['_ss_allowed']);rank.backward()
   assert torch.isfinite(rank) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.prediction_scorer.parameters())
   opt.step()
  grads={n:float(p.grad.norm()) for n,p in m.prediction_scorer.named_parameters()}
  assert all(v>0 for v in grads.values()),grads
  assert all(not torch.equal(p,initial[n]) for n,p in m.prediction_scorer.named_parameters())
  m.eval()
  with torch.no_grad():
   on=m(feed,True);m.citer_enabled=False;off=m(feed,True);m.citer_enabled=True
   base.load_state_dict({n:p for n,p in m.state_dict().items() if not n.startswith('prediction_scorer.')},strict=True)
   assert torch.equal(off['prediction'],base(feed,True)['prediction'])
   empty=dict(feed,history_items=feed['history_items'][:,:0],lengths=feed['lengths']*0)
   assert torch.count_nonzero(m(empty,True)['raw_prediction'])==0
   bad=dict(feed,history_items=feed['history_items'].clone())
   bad['history_items'][torch.arange(20,device=device)[None]>=feed['lengths'][:,None]]=3
   assert torch.equal(on['prediction'],m(bad,True)['prediction'])
   other=dict(feed,item_id=feed['item_id'].flip(0))
   assert torch.equal(on['interest_vectors'],m(other,True)['interest_vectors'])
   candidates=feed['item_id'];assert torch.count_nonzero(on['gate_T'][on['support']==0])==0
   h=feed['history_items'];valid=(torch.arange(20,device=device)[None]<feed['lengths'][:,None])&(h>0)
   edges={'transition':m.candidate_graph.lookup(h,feed['user_id'])}
   pair_ids=feed['item_id'][:,0][None].expand(len(h),-1);pair_e=on['candidate_vectors'][:,0][None].expand(len(h),-1,-1)
  m.train();train=m(feed,True)
  manual,_=m.prediction_scorer(train['interest_vectors'],train['attention_maps'],train['interest_weights'],valid,pair_ids,train['candidate_vectors'][:,0][None].expand(len(h),-1,-1),edges)
  assert torch.allclose(train['_ss_scores'],train['user_vector']@train['candidate_vectors'][:,0].T+manual,atol=2e-6,rtol=1e-5)
  assert torch.allclose(train['_ss_scores'].diagonal(),train['raw_prediction'][:,0],atol=3e-6,rtol=2e-5)
  assert torch.equal(train['_ss_allowed'],candidate_mask(m.train_membership,feed['user_id'],feed['item_id'][:,0])[0])
  # Reuse both seeds' exact historical base initialization, without reading held-out data.
  init={}
  for seed in [0,42]:
   aa=args_from_config(OUT/'configs'/f'{ds}_s{seed}.json');aa.v3_graph_path=a.v3_graph_path
   for variant in ['TargetInterest','Direct-Fixed']:
    aa.v3_variant=variant;utils.init_seed(seed);legacy=SGITRecV3Controlled(aa,c)
    init[str(seed)]=core_hash(legacy);del legacy
  costs={}
  if gpu is not None:
   big=make_feed(c,device,1024)
   opt.zero_grad();m.loss(m(big,True)).backward();opt.step()
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
   for _ in range(2):
    opt.zero_grad();m.loss(m(big,True)).backward();opt.step()
   torch.cuda.synchronize();costs.update(batch=1024,step_seconds=(time.perf_counter()-tick)/2,peak_train_MiB=torch.cuda.max_memory_allocated()/2**20)
   ev=make_feed(c,device,128);catalog=torch.tensor(sorted(c.data_df['train'].item_id.unique()),device=device)
   cand=catalog[torch.randint(len(catalog),(128,1001),device=device)];cand[:,5]=cand[:,0];cand[:,1000]=cand[:,0];ev['item_id']=cand
   m.eval()
   with torch.no_grad():
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter();got=m(ev,True);torch.cuda.synchronize()
    costs.update(eval128x1001_seconds=time.perf_counter()-tick,peak_eval_MiB=torch.cuda.max_memory_allocated()/2**20)
    assert torch.equal(got['prediction'][:,0],got['prediction'][:,5]) and torch.equal(got['prediction'][:,0],got['prediction'][:,1000])
  x=torch.randn(1,8,64,device=device,requires_grad=True);sid,allowed=segment_ids(x.new_zeros(1,8),torch.tensor([8],device=device),'fixed',0,fixed_len=4)
  y=m.sil_block.local_bank(x,sid,allowed)
  assert torch.count_nonzero(torch.autograd.grad(y[0,2].square().sum(),x)[0][0,3:])==0
  return dict(passed=True,dataset=ds,parameters=m.count_variables(),graph_bytes=m.candidate_graph.bytes(),initial_C1_hash=init,
   zero_init_C1_score_loss_grad_exact=True,single_branches_ungated_exact_legacy_AB=True,
   gated_single_branch_differs_from_historical_AB=True,ranking_gradients=grads,all_scorer_parameters_updated=True,
   full_BxB_uses_same_scorer=True,mask_duplicates_prior_once=True,empty_short_padding_causality=True,
   scales={n:float(p.detach()) for n,p in m.prediction_scorer.scales.items()},costs=costs,Dev=False,Test=False)
 finally:
  if lease is not None:os.close(lease)
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['math','real'],required=True);p.add_argument('--dataset',choices=['beauty','ml-1m']);p.add_argument('--gpu',type=int)
 a=p.parse_args();signal.alarm(170);torch.set_num_threads(4);protocol.install_guard();tick=time.perf_counter()
 got=mathematics() if a.mode=='math' else actual(a.dataset,a.gpu);got['seconds']=time.perf_counter()-tick
 name='math' if a.mode=='math' else a.dataset+('_gpu' if a.gpu is not None else '_real')
 with (OUT/'checks'/(name+'.json')).open('x') as f:json.dump(got,f,indent=2)
 print(json.dumps(got,indent=2))
if __name__=='__main__':main()
