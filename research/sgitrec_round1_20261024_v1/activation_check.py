"""Finite structural activation check; RAM models only, no Dev/Test or saved weights."""
import sys,os,argparse,json,time,signal,copy,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(OUT))
import torch,numpy as np
from preflight import CONFIGS,summary,routing
from gict_data import args_from_config,train_corpus
from models.sequential.SGITRec import SGITRec
from utils import utils
def evaluate(m,feed):
 m.eval();m.sgit_variant='G2'
 with torch.no_grad():
  g2=m(feed,True);m.sgit_variant='G1';g1=m(feed,True)
  ids=m.graph_cache.lookup(feed['history_items'],feed['user_id'],'transition')['ids']
  read=g2['collaborative_memory']
  rel=[]
  for a in range(read.shape[1]):
   for b in range(a+1,read.shape[1]):rel.append((read[:,a]-read[:,b]).norm(dim=-1)/read[:,a].norm(dim=-1).clamp_min(1e-8))
  result=dict(route=routing(g2['transition_route'],ids,m.item_num),memory_pair_relative_difference=summary(torch.stack(rel)),
   G2_G1_max_interest_difference=float((g2['interest_vectors']-g1['interest_vectors']).abs().max()),
   G2_G1_max_raw_score_difference=float((g2['raw_prediction']-g1['raw_prediction']).abs().max()))
 m.sgit_variant='G3'
 return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=list(CONFIGS));p.add_argument('--gpu',required=True,type=int)
 x=p.parse_args();signal.alarm(170);torch.set_num_threads(4);start=time.perf_counter()
 spec=importlib.util.spec_from_file_location('activation_resource',ROOT/'research/elector_sil_objective_full_20261023_v2/scheduler.py')
 r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);inv=r.observe();assert r.idle(inv[x.gpu])
 lease=r.acquire(ROOT/('research/gpu_resource_scheduler_20261022_v1/gpu%d.lock'%x.gpu));assert lease is not None
 try:
  assert r.idle(r.observe()[x.gpu]);os.environ['CUDA_VISIBLE_DEVICES']=inv[x.gpu]['uuid']
  cache=OUT/'checks'/('cuda'+str(x.gpu));cache.mkdir(exist_ok=True);os.environ['CUDA_CACHE_PATH']=str(cache)
  torch.cuda.set_device(0);a=args_from_config(CONFIGS[x.dataset]);a.device=torch.device('cuda:0');a.sgit_variant='G3'
  a.sgit_graph_path=str(ROOT/'research/sgitrec_v1_20261024/graphs'/(x.dataset+'_v2.pt'));c=train_corpus(a)
  rng=np.random.default_rng(42);users=np.array(sorted(u for u,h in c.user_his.items() if len(h)>=3))
  users=users[np.linspace(0,len(users)-1,128,dtype=int)];h=np.zeros((128,20),np.int64);lengths=[];items=[]
  for row,u in enumerate(users):
   seq=c.user_his[int(u)];pos=int(rng.integers(2,len(seq)));pre=seq[max(0,pos-20):pos]
   h[row,:len(pre)]=[v[0] for v in pre];lengths.append(len(pre));items.append([seq[pos][0],1])
  feed={k:torch.as_tensor(v,device=a.device) for k,v in dict(user_id=users,history_items=h,lengths=lengths,item_id=items).items()}
  utils.init_seed(42);m=SGITRec(a,c).to(a.device).eval()
  with torch.no_grad():base=m(feed,True)
  V=base['c1_interest_vectors'][:1].detach().requires_grad_(True);A=base['attention_maps'][:1].detach()
  edges=m.graph_cache.lookup(feed['history_items'][:1],feed['user_id'][:1],'transition')
  with torch.no_grad():fv=m.item_encoder(edges['ids'])
  valid=torch.arange(20,device=a.device)[None]<feed['lengths'][:1,None]
  memory=copy.deepcopy(m.graph_memory);memory.output.weight.data.copy_(torch.eye(64,device=a.device))
  weight=base['interest_weights'][:1].detach();direction=torch.linspace(-1,1,64,device=a.device)
  grads={}
  for mode in ['G1','G2']:
   _,detail=memory(V,A,valid,edges,fv,variant=mode,weights=weight)
   derivative=torch.autograd.grad((detail['graph_delta'][0,0]*direction).sum(),V)[0]
   grads[mode]=dict(own_interest_norm=float(derivative[:,0].norm()),other_interest_norm=float(derivative[:,1:].norm()))
  assert grads['G2']['own_interest_norm']>0 and grads['G2']['other_interest_norm']==0 and grads['G1']['other_interest_norm']>0
  before=evaluate(m,feed)
  # Only this temporary model is optimized. Formal runs always start afresh.
  utils.init_seed(42);optimizer=torch.optim.Adam(m.customize_parameters(),lr=a.lr,weight_decay=a.l2)
  losses=[]
  for step in range(16):
   m.train();m.sgit_variant='G3';m.cur_epoch=1;optimizer.zero_grad();out=m(feed,True);loss=m.loss(out)
   assert torch.isfinite(loss);loss.backward()
   assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
   optimizer.step();losses.append(float(loss.detach()))
  after=evaluate(m,feed)
  assert m.graph_memory.output.weight.norm()>0
  result=dict(dataset=x.dataset,steps=16,Jacobian=grads,structurally_not_G1=True,before=before,after=after,
   graph_projection_norm=float(m.graph_memory.output.weight.norm()),losses=losses,formal_initialization_unchanged=True,
   no_saved_weights=True,Dev=False,Test=False,seconds=time.perf_counter()-start,passed=True)
  target=OUT/'checks'/(x.dataset+'_activation.json')
  with target.open('x') as f:json.dump(result,f,indent=2)
  print(json.dumps(result,indent=2))
 finally:os.close(lease)
if __name__=='__main__':main()
