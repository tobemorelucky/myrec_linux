"""Bounded mathematical/real-Train regression for twelve-arm prediction study."""
import sys,os,json,time,signal,argparse,copy,hashlib,importlib.util,gc,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np,torch
from models.sequential.sgit_v3_control_components import PredictionScorer,VARIANTS
from models.sequential.SGITRecV3Controlled import SGITRecV3Controlled
from models.sequential.LLMMIRecASPCFAdaptiveSIL import LLMMIRecASPCFAdaptiveSIL as C1,segment_ids
from models.sequential.LLMMIRecASPCFSampledSoftmax import candidate_mask,masked_softmax_loss
from models.sequential.sgit_v3_components import sparse_candidate_read
from sgit_graph import build_graphs
from sgit_v2_graph import build_bridge
from build_sgit_v3_index import build_index
from build_sgit_v3_control_index import smooth_table,calibrate
from test_sgitrec import make_feed
from gict_data import args_from_config,train_corpus
from utils import utils
OUT=ROOT/'research/sgitrec_v3_predictions_20261025_v1'

def core_hash(m):
 h=hashlib.sha256()
 for n,p in m.named_parameters():
  if not n.startswith('prediction_scorer.'):h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def math_checks():
 torch.manual_seed(59);rows=[]
 for K in [1,2,3,4]:
  B,L,R,D,C=3,4,2,64,7
  v=torch.randn(B,K,D);A=torch.randn(B,K,L).softmax(-1);w=torch.randn(B,K).softmax(-1)
  valid=torch.arange(L)[None]<torch.tensor([0,1,4])[:,None]
  candidates=torch.tensor([[9,3,3,99,1,5,2]]*B)
  ids=torch.tensor([[[1,2],[2,5],[3,5],[1,9]]]*B)
  edges={r:dict(ids=ids,value=torch.rand(B,L,R)) for r in ['transition','bridge','smooth']}
  table=torch.randn(101,D);e=table[candidates]
  for variant in VARIANTS:
   m=PredictionScorer(D,variant,chunk=3)
   used={r:edges[r] for r in m.spec['relations']}
   zero,_=m(v,A,w,valid,candidates,e,used,True);assert torch.count_nonzero(zero)==0
   with torch.no_grad():
    for p in m.scales.values():p.fill_(.37)
   delta,info=m(v,A,w,valid,candidates,e,used,True)
   alpha=info['candidate_interest_weights']
   ref=torch.zeros(B,C)
   for rel in m.spec['relations']:
    ed=edges[rel]
    dense=(A[...,None,None]*ed['value'][:,None,:,None,:]*((ids[:,:,None,:]==candidates[:,None,:,None])&valid[:,:,None,None])[:,None]).sum((2,4))
    ref=ref+.37*(alpha*dense).sum(1)
    assert torch.allclose(dense,sparse_candidate_read(A,valid,ed,candidates),atol=2e-7)
   if m.spec['target']:
    val=((alpha-w[...,None])*torch.einsum('bkd,bcd->bkc',v,e)).sum(1)
    ref=ref+.37*val.masked_fill(~valid.any(-1)[:,None],0)
   assert torch.allclose(delta,ref,atol=1e-6,rtol=1e-5),(K,variant)
   assert torch.equal(delta[:,1],delta[:,2]) and torch.count_nonzero(delta[0])==0
   if not m.spec['target']:assert torch.count_nonzero(delta[:,3])==0
   # Ranking objective alone, not relation or auxiliary loss.
   opt=torch.optim.Adam(m.parameters(),lr=.01)
   for step in range(3):
    opt.zero_grad();d,_=m(v,A,w,valid,candidates,e,used)
    loss=torch.logsumexp(d+torch.randn_like(d)*.01,1).mean()-d[:,0].mean()
    loss.backward();opt.step()
   for name,p in m.named_parameters():
    assert p.grad is not None and torch.isfinite(p.grad).all(),(K,variant,name)
    if K>1 or name.startswith('scales.') and variant!='TargetInterest':assert p.grad.abs().sum()>0,(K,variant,name)
   rows.append(dict(K=K,variant=variant,dense_reference=True,duplicate_slots=True,ranking_gradients=True))
 # Candidate-dependent interest assignment and strict B/C controlled equality.
 v=torch.eye(64)[:2][None];e=v.clone();w=torch.ones(1,2)/2
 m=PredictionScorer(64,'Direct-Target')
 with torch.no_grad():m.query.weight.copy_(torch.eye(64));m.key.weight.copy_(torch.eye(64))
 alpha=m.alpha(v,w,e);assert alpha[0,:,0].argmax()!=alpha[0,:,1].argmax()
 with torch.no_grad():m.query.weight.zero_()
 assert torch.allclose(m.alpha(v,w,e),w[...,None].expand(-1,-1,2))
 histories={1:[(1,1),(2,2),(3,3)],2:[(4,1),(5,2),(6,3)],3:[(1,1),(7,2),(8,3)],4:[(4,1),(5,2),(1,3)]}
 table=torch.randn(10,7);table[0]=0
 def all_graphs(h):
  t,meta=build_graphs(h,table)
  base=build_bridge(dict(metadata=dict(meta,schema='sgit_crossfit_item_graph_v1',scope='full_crossfit_train'),**t))
  index,_=build_index(base,h);sm,_=smooth_table(base);sm,_=calibrate(sm,index)
  return dict(index,**sm)
 first=all_graphs(histories);changed=copy.deepcopy(histories);changed[3]=[(9,1),(8,2),(1,3),(4,4)]
 second=all_graphs(changed)
 for key in first:assert torch.equal(first[key][0],second[key][0]),key
 return dict(passed=True,checks=rows,fold_future_invariance_TS_ST_and_calibration=True,candidate_specific_interests=True,
  B_C_same_when_condition_removed=True,K1_target_exactly_zero=True,Dev=False,Test=False)

def actual(ds,gpu):
 lease=None;device='cpu'
 if gpu is not None:
  spec=importlib.util.spec_from_file_location('prediction_resource',ROOT/'research/elector_sil_objective_full_20261023_v2/scheduler.py')
  res=importlib.util.module_from_spec(spec);spec.loader.exec_module(res)
  inv=res.observe();assert res.idle(inv[gpu]),'GPU busy: never preempt'
  lease=res.acquire(ROOT/('research/gpu_resource_scheduler_20261022_v1/gpu%d.lock'%gpu));assert lease is not None
  assert res.idle(res.observe()[gpu])
  os.environ['CUDA_VISIBLE_DEVICES']=inv[gpu]['uuid']
  cache=OUT/'checks'/('cuda_'+ds);cache.mkdir()
  os.environ['CUDA_CACHE_PATH']=str(cache);torch.cuda.set_device(0);device='cuda:0'
 try:
  a=args_from_config(OUT/'configs'/(ds+'_s42.json'));a.device=torch.device(device);a.v3_graph_path=str(OUT/'graphs'/(ds+'_predictions_v3.pt'))
  c=train_corpus(a);feed=make_feed(c,device,128);rows={};a.v3_candidate_chunk=128
  utils.init_seed(42);ref=C1(a,c).to(device).eval();initial=core_hash(ref)
  with torch.no_grad():baseline=ref(feed,True)
  ref.train();ref.cur_epoch=1
  # Base gradient/reference mask from same initial RNG.
  rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state() if gpu is not None else None
  refout=ref(feed,True);base_rank,_=ref._ranking_loss(refout['_ss_scores'],refout['_ss_allowed'])
  base_total=ref.loss(refout);base_total.backward()
  base_grad={n:p.grad.detach().clone() for n,p in ref.named_parameters() if p.grad is not None}
  for variant in VARIANTS:
   a.v3_variant=variant;utils.init_seed(42);m=SGITRecV3Controlled(a,c).to(device).eval();m.cur_epoch=1
   assert core_hash(m)==initial
   with torch.no_grad():out=m(feed,True)
   assert torch.equal(out['prediction'],baseline['prediction']) and torch.equal(out['raw_prediction'],baseline['raw_prediction'])
   m.train();torch.set_rng_state(rng)
   if gpu is not None:torch.cuda.set_rng_state(cuda_rng)
   out=m(feed,True);loss=m.loss(out);loss.backward()
   assert torch.equal(loss,base_total) and torch.equal(out['_ss_allowed'],refout['_ss_allowed'])
   for n,g in base_grad.items():assert torch.allclose(dict(m.named_parameters())[n].grad,g,atol=2e-7,rtol=2e-5),(variant,n)
   m.zero_grad(set_to_none=True)
   before={n:p.detach().clone() for n,p in m.prediction_scorer.named_parameters()}
   opt=torch.optim.Adam(m.customize_parameters(),lr=a.lr,weight_decay=a.l2)
   grads={}
   for step in range(6):
    opt.zero_grad();out=m(feed,True);ranking,_=m._ranking_loss(out['_ss_scores'],out['_ss_allowed'])
    assert torch.isfinite(ranking)
    ranking.backward()
    for n,p in m.prediction_scorer.named_parameters():
     assert p.grad is not None and torch.isfinite(p.grad).all(),(variant,n)
     grads[n]=float(p.grad.norm())
    opt.step()
   for n,p in m.prediction_scorer.named_parameters():
    assert grads[n]>0 and not torch.equal(p,before[n]),(variant,n,grads[n])
   m.eval()
   with torch.no_grad():
    out=m(feed,True);m.v3_enabled=False;off=m(feed,True);m.v3_enabled=True
    ref.load_state_dict({n:p for n,p in m.state_dict().items() if not n.startswith('prediction_scorer.')},strict=True);ref.eval()
    assert torch.equal(off['prediction'],ref(feed,True)['prediction'])
    other=dict(feed,item_id=feed['item_id'].flip(0));o2=m(other,True)
    assert torch.equal(out['interest_vectors'],o2['interest_vectors']) and torch.equal(out['attention_maps'],o2['attention_maps'])
    bad=dict(feed,history_items=feed['history_items'].clone())
    bad['history_items'][torch.arange(20,device=device)[None]>=feed['lengths'][:,None]]=3
    assert torch.equal(m(bad,True)['prediction'],out['prediction'])
    empty=dict(feed,history_items=feed['history_items'][:,:0],lengths=feed['lengths']*0)
    eo=m(empty,True);assert torch.count_nonzero(eo['raw_prediction'])==0 and torch.isfinite(eo['prediction']).all()
    # All BxB training pairs must equal the same scorer in row-candidate mode.
    users=m(feed,True);history=feed['history_items'];valid=(torch.arange(20,device=device)[None]<feed['lengths'][:,None])&(history>0)
    edges={r:m.candidate_graph.lookup(history,feed['user_id'],r) for r in m.spec['relations']}
    shared,_=m.prediction_scorer(users['interest_vectors'],users['attention_maps'],users['interest_weights'],valid,
      feed['item_id'][:,0],users['candidate_vectors'][:,0],edges)
    expanded=feed['item_id'][:,0][None].expand(len(history),-1)
    emb=users['candidate_vectors'][:,0][None].expand(len(history),-1,-1)
    paired,_=m.prediction_scorer(users['interest_vectors'],users['attention_maps'],users['interest_weights'],valid,expanded,emb,edges)
    assert torch.allclose(shared,paired,atol=2e-6,rtol=1e-5)
    if m.spec['relations']:
     absent=feed['item_id'].clone()
     catalog=c.data_df['train'].item_id.unique()
     for row in range(len(absent)):
      blocked={int(z) for ed in edges.values() for z in ed['ids'][row].flatten().cpu().tolist()}
      absent[row]=next(int(i) for i in catalog if int(i) not in blocked)
     no=m(dict(feed,item_id=absent),True)
     assert torch.count_nonzero(no['candidate_graph_delta'])==0
   # Check nonzero full training scores, filtered duplicate and clicked rules unchanged.
   m.train();train_out=m(feed,True)
   assert torch.allclose(train_out['_ss_scores'].diagonal(),train_out['raw_prediction'][:,0],atol=3e-6,rtol=2e-5)
   masked=candidate_mask(m.train_membership,feed['user_id'],feed['item_id'][:,0])[0]
   assert torch.equal(train_out['_ss_allowed'],masked)
   rows[variant]=dict(parameters=m.count_variables(),ranking_gradient_norms=grads,
    scales={n:float(p.detach()) for n,p in m.prediction_scorer.scales.items()},
    graph_bytes=m.candidate_graph.bytes() if m.candidate_graph is not None else 0,
    initial_C1_score_loss_grad_equal=True,all_BxB_pairs_same_formula=True,initial_C1_hash=initial)
   if gpu is not None:
    big=make_feed(c,device,1024)
    opt.zero_grad();m.loss(m(big,True)).backward();opt.step();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    tick=time.perf_counter()
    for _ in range(2):
     opt.zero_grad();m.loss(m(big,True)).backward();opt.step()
    torch.cuda.synchronize()
    rows[variant].update(train_batch=1024,step_seconds=(time.perf_counter()-tick)/2,train_peak_MiB=torch.cuda.max_memory_allocated()/2**20)
    ev=make_feed(c,device,128);catalog=torch.tensor(sorted(c.data_df['train'].item_id.unique()),device=device)
    candidates=catalog[torch.randint(len(catalog),(128,1001),device=device)];candidates[:,5]=candidates[:,0];candidates[:,1000]=candidates[:,0];ev['item_id']=candidates
    m.eval()
    with torch.no_grad():
     torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter();got=m(ev,True);torch.cuda.synchronize()
     rows[variant].update(eval128x1001_seconds=time.perf_counter()-tick,eval_peak_MiB=torch.cuda.max_memory_allocated()/2**20)
     assert torch.equal(got['prediction'][:,0],got['prediction'][:,5]) and torch.equal(got['prediction'][:,0],got['prediction'][:,1000])
     assert got['prediction'].shape==(128,1001) and torch.isfinite(got['prediction']).all()
     assert torch.equal(got['prediction'],m.corrected_scores(got['raw_prediction'],candidates))
   # Restore base reference after loading trained parameters for off-test.
   utils.init_seed(42);fresh=C1(a,c).to(device).eval();ref.load_state_dict(fresh.state_dict());del fresh,m,opt;gc.collect()
  x=torch.randn(1,8,64,device=device,requires_grad=True)
  ids,valid=segment_ids(x.new_zeros(1,8),torch.tensor([8],device=device),'fixed',0.,fixed_len=4)
  y=ref.sil_block.local_bank(x,ids,valid)
  assert torch.count_nonzero(torch.autograd.grad(y[0,2].square().sum(),x)[0][0,3:])==0
  return dict(passed=True,dataset=ds,device=device,variants=rows,short_empty_padding=True,causality=True,
   train_matrix_not_plain_dot=True,ranking_only_gradients=True,no_target_in_history_or_graph=True,
   duplicate_1001_and_prior_once=True,initial_C1_hash=initial,Dev=False,Test=False,
   RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
 finally:
  if lease is not None:os.close(lease)
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['math','real'],required=True);p.add_argument('--dataset',choices=['beauty','ml-1m']);p.add_argument('--gpu',type=int)
 a=p.parse_args();signal.alarm(170);torch.set_num_threads(4);tick=time.perf_counter()
 result=math_checks() if a.mode=='math' else actual(a.dataset,a.gpu)
 result['seconds']=time.perf_counter()-tick
 name='math' if a.mode=='math' else a.dataset+('_gpu' if a.gpu is not None else '_real')
 with (OUT/'checks'/(name+'.json')).open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
