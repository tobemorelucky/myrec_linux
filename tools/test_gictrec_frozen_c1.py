"""Bounded Train-only interest diagnostics and frozen-mode CPU/GPU checks, no Test."""
import sys,time,signal,json,os,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research/gictrec_frozen_c1_20261024_v1'
sys.path.insert(0,str(OUT))
from common import *
from models.sequential.LLMMIRecASPCFAdaptiveSIL import LLMMIRecASPCFAdaptiveSIL
from models.sequential.gict_components import js_distance
def stats(x):
 x=x.detach().float().flatten().cpu()
 return dict(min=float(x.min()),median=float(x.median()),p90=float(x.quantile(.9)),max=float(x.max()),mean=float(x.mean()),std=float(x.std(unbiased=False)))
def check(m,ref,feed):
 m._ss_stats=None  # Separate diagnostic batches, including CPU -> GPU transition.
 m.train();assert m.training and not m.dropout.training
 with torch.no_grad():
  first=m(feed,True);second=m(feed,True);reference=ref(feed,True)
 assert torch.equal(first['prediction'],reference['prediction'])
 assert torch.equal(first['prediction'],second['prediction'])
 assert not first['c1_interest_vectors'].requires_grad and not first['candidate_vectors'].requires_grad
 before=common_hash(m);optimizer=torch.optim.Adam([m.transfer.output.weight],lr=.001,weight_decay=1e-6)
 initial=m.transfer.output.weight.detach().clone();gradient=None
 for _ in range(8):
  m.train();m.cur_epoch=1
  optimizer.zero_grad();got=m(feed,True);loss=m.loss(got);loss.backward()
  assert torch.isfinite(loss) and got['loss_relation']>=0
  assert all(p.grad is None for n,p in m.named_parameters() if not n.startswith('transfer.'))
  gradient=float(m.transfer.output.weight.grad.norm());assert gradient>0
  optimizer.step()
 assert common_hash(m)==before
 assert not torch.equal(initial,m.transfer.output.weight) and m.transfer.output.weight.norm()>0
 m.eval()
 with torch.no_grad():
  after=m(feed,True);changed=dict(feed);changed['item_id']=feed['item_id'].flip(0);altered=m(changed,True)
 assert torch.equal(after['interest_vectors'],altered['interest_vectors'])
 assert torch.equal(after['gict_neighbor_users'],altered['gict_neighbor_users'])
 valid=after['gict_neighbor_valid'];assert not (after['gict_neighbor_users']==feed['user_id'][:,None])[valid].any()
 result=dict(initial_C1_exact=True,deterministic_frozen_dropout=True,frozen_parameters_unchanged=True,only4096_trainable=True,
  projection_update_norm=float((m.transfer.output.weight-initial).norm()),gradient_norm=gradient,
  residual_ratio=stats(after['transfer_delta'].norm(dim=-1)/after['c1_interest_vectors'].norm(dim=-1).clamp_min(1e-8)),
  no_candidate_query_leak=True,teacher_detached=True,steps=8)
 assert after['transfer_delta'].abs().sum()>0
 return result,after
def main():
 signal.alarm(170);torch.set_num_threads(4);start=time.perf_counter();verify()
 rpath=ROOT/'research/elector_sil_objective_full_20261023_v2/scheduler.py'
 spec=importlib.util.spec_from_file_location('frozen_diagnostic_lease',rpath);resource=importlib.util.module_from_spec(spec);spec.loader.exec_module(resource)
 idle=[g for g,v in resource.observe().items() if resource.idle(v)];assert idle,'No idle GPU, wait rather than preempt'
 gpu=idle[0];lease=resource.acquire(ROOT/('research/gpu_resource_scheduler_20261022_v1/gpu%d.lock'%gpu));assert lease is not None
 try:
  assert resource.idle(resource.observe()[gpu])
  m,a,c,spec=model('G3','cpu');m.cur_epoch=1
  state=torch.load(spec['checkpoint'],map_location='cpu',weights_only=True)
  ref=LLMMIRecASPCFAdaptiveSIL(a,c).eval();ref.load_state_dict(state,strict=True)
  users=sorted(u for u,h in c.user_his.items() if len(h)>1)
  users=np.asarray(users)[np.linspace(0,len(users)-1,128,dtype=int)]
  history=np.zeros((128,20),np.int64);lengths=[];items=[]
  for row,u in enumerate(users):
   seq=c.user_his[int(u)];n=0 if row==0 else min(1+row%20,len(seq)-1)
   history[row,:n]=[x[0] for x in seq[:n]];lengths.append(n);items.append([seq[n][0],1])
  feed={k:torch.as_tensor(v) for k,v in dict(user_id=users,history_items=history,lengths=lengths,item_id=items).items()}
  cpu,_=check(m,ref,feed)
  device=torch.device('cuda:'+str(gpu));torch.cuda.set_device(gpu);m.device=device;ref.device=device
  m=m.to(device);ref=ref.to(device);feed={k:v.to(device) for k,v in feed.items()}
  torch.cuda.reset_peak_memory_stats(gpu);checks={}
  for variant in ['G1','G2','G3']:
   m.gict_variant=variant;m.load_c1(state);utils.init_seed(42)
   checks[variant],after=check(m,ref,feed)
   if variant=='G3':
    p=after['semantic_interest_profiles'][1:];K=p.shape[1];indices=torch.triu_indices(K,K,offset=1,device=device)
    within=js_distance(p[:,indices[0]],p[:,indices[1]])
    teacher=m.semantic_index.interest_profiles[:1024]
    teacher_within=js_distance(teacher[:,indices[0]],teacher[:,indices[1]])
    weights=after['match_weights'][1:].flatten(2);top=weights.topk(2,-1)
    best=top.indices[...,0];neighbor_choice=best//K
    user_choice=after['gict_neighbor_users'][1:].gather(1,neighbor_choice)
    all_same=(user_choice==user_choice[:,:1]).all(1)
    pair_same=(user_choice[:,indices[0]]==user_choice[:,indices[1]])
    rel=after['reliability'][1:]
    detail=dict(sampled_Train_prefixes=128,teacher_sample_users=1024,
     intra_user_interest_JS=stats(within),intra_user_JS_below_1e_minus4=float((within<1e-4).float().mean()),
     teacher_intra_interest_JS=stats(teacher_within),
     match_top1_top2_probability_gap=stats(top.values[...,0]-top.values[...,1]),
     match_top1_probability=stats(top.values[...,0]),
     same_neighbor_all_K_fraction=float(all_same.float().mean()),same_neighbor_interest_pair_fraction=float(pair_same.float().mean()),
     same_exact_teacher_interest_pair_fraction=float((best[:,indices[0]]==best[:,indices[1]]).float().mean()),
     matching_JS=stats(after['match_distance'][1:]),reliability=stats(rel),
     reliability_below_1e_minus3=float((rel<1e-3).float().mean()),reliability_above_099=float((rel>.99).float().mean()))
  # Verify the full Dev protocol and anchor before any formal training; no Test read.
  m.load_c1(state);m.eval();heldout(c,a,'dev');d=m.Dataset(m,c,'dev');d.prepare()
  got=score_views(d);protocol_lock('dev',got)
  expected=read(Path(spec['config']).parent/'best_metadata.json')
  for scheme in ['raw','locked_prior']:
   assert all(abs(got['metrics'][scheme][k]-expected[scheme][k])<1e-10 for k in KEYS)
  result=dict(passed=True,seconds=time.perf_counter()-start,physical_GPU=gpu,cpu=cpu,gpu=checks,distribution_diagnostics=detail,
   full_Dev_anchor_exact=True,Dev=got['metrics'],GPU_peak_MiB=torch.cuda.max_memory_allocated(gpu)/2**20,
   formal_training=False,Test=False,
   interpretation='JS gaps/same-neighbor fractions describe sampled frozen interests; optimizer causality remains unproven.')
  write(OUT/'DIAGNOSTICS.json',result);print(json.dumps(result,indent=2))
 finally:os.close(lease)
if __name__=='__main__':main()
