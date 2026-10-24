"""SGITRec bounded math/leakage/real cost checks; never runs Test or full training."""
import sys,argparse,json,time,signal,copy,os,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import torch,numpy as np
from sgit_graph import build_graphs,transition_graph
from gict_data import args_from_config,train_corpus,sha
from models.sequential.sgit_components import InterestGraphMemory
from models.sequential.SGITRec import SGITRec
from models.sequential.LLMMIRecASPCFAdaptiveSIL import LLMMIRecASPCFAdaptiveSIL as C1,segment_ids
from run_sgitrec import core_hash
from utils import utils
OUT=ROOT/'research/sgitrec_v1_20261024'
def synthetic(device='cpu'):
 torch.manual_seed(9)
 histories={1:[(1,1),(2,2),(3,3)],2:[(5,1),(1,2),(2,3)],3:[(2,1),(1,2)],
            4:[(1,1),(2,2),(4,3)],5:[(1,1),(2,2),(2,3)],6:[(2,1),(1,2)]}
 table=torch.randn(9,6);table[0]=0
 graph,meta=build_graphs(histories,table,R=8,folds=3)
 t=graph['transition_ids'][0,1];j=(t==2).nonzero(as_tuple=True)[0]
 assert len(j)==1 and graph['transition_counts'][0,1,j]==4 and graph['transition_support'][0,1,j]==4
 assert not (graph['transition_ids'][0,2]==1).any(),'Direction was symmetrized'
 changed=copy.deepcopy(histories);changed[3]=[(8,1),(7,2),(6,3),(5,4)];changed[6]=[(8,1)]
 other,_=build_graphs(changed,table,R=8,folds=3)
 for key in graph:assert torch.equal(graph[key][0],other[key][0]),'Excluded users future affected '+key
 for k in [1,2,3,4]:
  B,L,R,D=3,4,2,64
  v=torch.randn(B,k,D,device=device,requires_grad=True)
  aa=torch.randn(B,k,L,device=device,requires_grad=True);A=aa.softmax(-1)
  features=torch.randn(B,L,R,D,device=device,requires_grad=True)
  semfeat=torch.randn_like(features,requires_grad=True)
  edges=dict(ids=torch.ones(B,L,R,dtype=torch.long,device=device),prob=torch.rand(B,L,R,device=device).softmax(-1),support=torch.full((B,L,R),4.,device=device))
  sem=dict(ids=edges['ids'].clone(),prob=edges['prob'].clone())
  valid=torch.ones(B,L,dtype=torch.bool,device=device);valid[2]=False
  m=InterestGraphMemory(D,True).to(device);w=torch.ones(B,k,device=device)/k
  y,_=m(v,A,valid,edges,features,sem,semfeat,weights=w)
  assert torch.equal(y,v)
  y.square().sum().backward(retain_graph=True)
  assert m.output.weight.grad.abs().sum()>0
  m.zero_grad()
  with torch.no_grad():m.output.weight.normal_(0,.02)
  y,detail=m(v,A,valid,edges,features,sem,semfeat,weights=w)
  y.square().sum().backward(retain_graph=True)
  for p in [m.query.weight,m.key.weight,m.value.weight,m.fusion.weight]:
   assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
  assert aa.grad is not None and aa.grad.abs().sum()>0
  assert torch.equal(y[2],v[2])
  empty={name:torch.zeros_like(x) for name,x in edges.items()}
  z,_=m(v,A,valid,empty,features,dict(ids=sem['ids']*0,prob=sem['prob']*0),semfeat,weights=w)
  assert torch.equal(z,v)
  no_sem,_=m(v,A,valid,edges,features,dict(ids=sem['ids']*0,prob=sem['prob']*0),semfeat,weights=w)
  only_c,_=m(v,A,valid,edges,features,variant='G2',weights=w)
  assert torch.equal(no_sem,only_c)
  if k==1:
   g1,_=m(v,A,valid,edges,features,variant='G1',weights=w)
   assert torch.equal(g1,only_c)
  if k>1:assert (detail['transition_route'][:,0]-detail['transition_route'][:,1]).abs().max()>1e-5
 return dict(K=[1,2,3,4],gradients=True,zero_init=True,no_neighbors=True,no_semantic=True,padding=True,
  directed_counts=True,distinct_user_support=True,leave_whole_user_fold_out=True,own_future_mutation_invariant=True,K1_G1_G2_equivalent=True)

def make_feed(c,device,batch=32):
 users=sorted(u for u,h in c.user_his.items() if len(h)>1)
 users=np.asarray(users)[np.linspace(0,len(users)-1,batch,dtype=int)]
 history=np.zeros((batch,20),np.int64);lengths=[];items=[]
 for row,u in enumerate(users):
  seq=c.user_his[int(u)];n=min(len(seq)-1,20)
  if row<5:n=min(row,n)
  history[row,:n]=[x[0] for x in seq[:n]];lengths.append(n);items.append([seq[n][0],1])
 return {key:torch.as_tensor(value,device=device) for key,value in dict(user_id=users,history_items=history,lengths=lengths,item_id=items).items()}

def real(ds,device,batch):
 folder=ROOT/('research/elector_sil_formal_20261023_v1/runs/beauty_k2_s42_SS-C1' if ds=='beauty' else 'research/elector_sil_k3_20261023_v1/runs/ml-1m_k3_s42_SS-C1')
 a=args_from_config(folder/'config.json');a.device=torch.device(device);a.sgit_graph_path=str(OUT/'graphs'/(ds+'_v2.pt'))
 c=train_corpus(a);feed=make_feed(c,device,32)
 utils.init_seed(42);reference=C1(a,c).to(device).eval();initial=core_hash(reference)
 variants={}
 for variant in ['B0','G1','G2','G3']:
  a.sgit_variant=variant;utils.init_seed(42);m=SGITRec(a,c).to(device).eval()
  assert core_hash(m)==initial,'Fresh same-seed C1 initialization differs'
  with torch.no_grad():old=reference(feed,True);new=m(feed,True)
  assert torch.equal(old['prediction'],new['prediction'])
  variants[variant]=m.count_variables()
  if variant!='G3':del m
 m.train();reference.train();m.zero_grad();reference.zero_grad()
 rng=torch.get_rng_state();crng=torch.cuda.get_rng_state(a.device) if str(device).startswith('cuda') else None
 new=m(feed,True);loss=m.loss(new);loss.backward()
 torch.set_rng_state(rng)
 if crng is not None:torch.cuda.set_rng_state(crng,a.device)
 old=reference(feed,True);base_loss=reference.loss(old);base_loss.backward()
 assert torch.equal(loss,base_loss)
 refparams=dict(reference.named_parameters())
 for name,p in m.named_parameters():
  if name in refparams and refparams[name].grad is not None:
   assert p.grad is not None and torch.allclose(p.grad,refparams[name].grad,atol=2e-7,rtol=2e-5),name
 assert m.graph_memory.output.weight.grad.abs().sum()>0
 optimizer=torch.optim.Adam(m.customize_parameters(),lr=a.lr,weight_decay=a.l2)
 for step in range(4):
  m.train();m.cur_epoch=1;optimizer.zero_grad();got=m(feed,True);trainloss=m.loss(got)
  assert torch.isfinite(trainloss);trainloss.backward()
  assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
  optimizer.step()
 assert core_hash(m)!=initial,'C1 did not train jointly'
 for name in ['query','key','value']:
  assert getattr(m.graph_memory,name).weight.grad.abs().sum()>0,name
 assert m.graph_memory.fusion.weight.grad.abs().sum()>0
 assert m.extractor.query.grad is not None and m.extractor.query.grad.abs().sum()>0
 assert m.sil_block.local_qkv.weight.grad is not None and m.sil_block.local_qkv.weight.grad.abs().sum()>0
 m.eval();reference.load_state_dict({n:p for n,p in m.state_dict().items() if not n.startswith('graph_memory.')},strict=True);reference.eval()
 with torch.no_grad():
  m.sgit_enabled=False;off=m(feed,True);ref=reference(feed,True)
  assert torch.equal(off['prediction'],ref['prediction']);m.sgit_enabled=True
  first=m(feed,True);other=dict(feed);other['item_id']=feed['item_id'].flip(0);second=m(other,True)
  assert torch.allclose(first['interest_vectors'],second['interest_vectors'],atol=3e-6,rtol=3e-6)
  padded=dict(feed);padded['history_items']=feed['history_items'].clone()
  mask=torch.arange(20,device=device)[None]>=feed['lengths'][:,None]
  padded['history_items'][mask]=3
  pad=m(padded,True);assert torch.allclose(first['interest_vectors'],pad['interest_vectors'],atol=3e-6,rtol=3e-6)
  empty=dict(feed);empty['history_items']=feed['history_items'][:,:0];empty['lengths']=feed['lengths']*0
  zero=m(empty,True);assert torch.count_nonzero(zero['interest_vectors'])==0 and torch.isfinite(zero['prediction']).all()
 x=torch.randn(1,8,64,device=device,requires_grad=True)
 ids,valid=segment_ids(x.new_zeros(1,8),torch.tensor([8],device=device),'fixed',0.,fixed_len=4)
 local=m.sil_block.local_bank(x,ids,valid);gradient=torch.autograd.grad(local[0,2].square().sum(),x)[0]
 assert torch.count_nonzero(gradient[0,3:])==0
 # All memory relations absent must exactly recover the CURRENT C1 after training.
 saved={name:buf.clone() for name,buf in m.graph_cache.named_buffers()}
 with torch.no_grad():
  for name,buf in m.graph_cache.named_buffers():buf.zero_()
  absent=m(feed,True);assert torch.equal(absent['prediction'],ref['prediction'])
  for name,buf in m.graph_cache.named_buffers():buf.copy_(saved[name])
 # Real batch cost; no checkpoint or Test is read, models discarded afterwards.
 large=make_feed(c,device,batch);cost={}
 for enabled,label in [(False,'C1'),(True,'G3')]:
  m.sgit_enabled=enabled;m.train();optimizer.zero_grad()
  for _ in range(2):
   optimizer.zero_grad();out=m(large,True);value=m.loss(out);value.backward();optimizer.step()
  if str(device).startswith('cuda'):torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(a.device)
  tick=time.perf_counter()
  for _ in range(3):
   optimizer.zero_grad();out=m(large,True);value=m.loss(out);value.backward();optimizer.step()
  if str(device).startswith('cuda'):torch.cuda.synchronize()
  seconds=(time.perf_counter()-tick)/3
  cost[label]=dict(batch=batch,step_seconds=seconds,rows_per_second=batch/seconds,
   peak_MiB=torch.cuda.max_memory_allocated(a.device)/2**20 if str(device).startswith('cuda') else None)
 return dict(dataset=ds,K=a.K,device=str(device),parameters=variants,initial_C1_exact=True,shared_fresh_initialization=True,
  initial_loss_exact=True,initial_C1_gradients_close=True,graph_off_exact=True,no_neighbors_exact=True,short_history=True,
  padding=True,local_causality=True,candidate_independent=True,original_C1_and_graph_gradients=True,
  cache_bytes=m.graph_cache.bytes(),encoding_cost=out.get('graph_encoding_cost'),throughput=cost,
  sampled_steps_only=True,Dev=False,Test=False)

def candidate_cost(ds,device,batch):
 # Train-derived synthetic candidate slots: resource check, never Dev/Test metrics.
 folder=ROOT/('research/elector_sil_formal_20261023_v1/runs/beauty_k2_s42_SS-C1' if ds=='beauty' else 'research/elector_sil_k3_20261023_v1/runs/ml-1m_k3_s42_SS-C1')
 a=args_from_config(folder/'config.json');a.device=torch.device(device);a.sgit_variant='G3'
 a.sgit_graph_path=str(OUT/'graphs'/(ds+'_v2.pt'));c=train_corpus(a)
 utils.init_seed(42);m=SGITRec(a,c).to(device).eval();feed=make_feed(c,device,batch)
 catalog=torch.tensor(sorted(c.data_df['train'].item_id.unique()),device=device)
 torch.manual_seed(42)
 feed['item_id']=catalog[torch.randint(len(catalog),(batch,1001),device=device)]
 with torch.inference_mode():
  if str(device).startswith('cuda'):torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(a.device)
  start=time.perf_counter();got=m(feed,True)
  if str(device).startswith('cuda'):torch.cuda.synchronize()
  elapsed=time.perf_counter()-start
  assert got['prediction'].shape==(batch,1001) and torch.isfinite(got['prediction']).all()
 return dict(dataset=ds,device=str(device),batch=batch,candidates=1001,seconds_per_forward=elapsed,
  peak_MiB=torch.cuda.max_memory_allocated(a.device)/2**20 if str(device).startswith('cuda') else None,
  Train_only_synthetic_candidates=True,Dev=False,Test=False,encoding_cost=got['graph_encoding_cost'])


def full_k_check(ds,device,batch):
 folder=ROOT/('research/elector_sil_formal_20261023_v1/runs/beauty_k2_s42_SS-C1' if ds=='beauty' else 'research/elector_sil_k3_20261023_v1/runs/ml-1m_k3_s42_SS-C1')
 a=args_from_config(folder/'config.json');a.device=torch.device(device);a.sgit_variant='G3'
 a.sgit_graph_path=str(OUT/'graphs'/(ds+'_v2.pt'));c=train_corpus(a);feed=make_feed(c,device,8)
 for k in [1,4]:
  a.K=k;utils.init_seed(42);ref=C1(a,c).to(device).eval()
  utils.init_seed(42);m=SGITRec(a,c).to(device).eval()
  old=ref(feed,True);new=m(feed,True)
  assert new['interest_vectors'].shape==(8,k,64) and torch.equal(old['prediction'],new['prediction'])
  m.train();m.zero_grad();loss=m.loss(m(feed,True));assert torch.isfinite(loss)
  loss.backward();assert m.graph_memory.output.weight.grad.abs().sum()>0
 return dict(dataset=ds,device=str(device),additional_real_K=[1,4],zero_equivalence=True,output_gradient=True,Dev=False,Test=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['synthetic','real','candidates','k'],required=True);p.add_argument('--dataset',choices=['beauty','ml-1m'])
 p.add_argument('--gpu',type=int);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--output',required=True)
 args=p.parse_args();signal.alarm(170);torch.set_num_threads(4);start=time.perf_counter()
 output=Path(args.output).resolve();assert output.is_relative_to(ROOT) and not output.exists()
 if args.mode=='synthetic':result=synthetic()
 else:
  device='cpu';lease=None
  if args.gpu is not None:
   spec=importlib.util.spec_from_file_location('sgit_test_resource',ROOT/'research/elector_sil_objective_full_20261023_v2/scheduler.py')
   r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
   assert r.idle(r.observe()[args.gpu]),'GPU occupied: never preempt'
   lease=r.acquire(ROOT/('research/gpu_resource_scheduler_20261022_v1/gpu%d.lock'%args.gpu));assert lease is not None
   assert r.idle(r.observe()[args.gpu]);torch.cuda.set_device(args.gpu);device='cuda:'+str(args.gpu)
  try:result={'real':real,'candidates':candidate_cost,'k':full_k_check}[args.mode](args.dataset,device,args.batch_size)
  finally:
   if lease is not None:os.close(lease)
 result.update(passed=True,seconds=time.perf_counter()-start)
 with output.open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
