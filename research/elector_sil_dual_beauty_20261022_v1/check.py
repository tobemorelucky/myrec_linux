"""Under-3-minute dual-memory checks. No full training, no Test."""
import os
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
import sys,json,time,signal,importlib.util,copy
from pathlib import Path
import numpy as np
import torch
ROOT=Path("/home/yyx/hzgcode/pom2.0").resolve();OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OUT))
from train_arm import Model,snapshot,read_corpus,arguments,common_hash,old,reuse,verify_sources,write
from models.sequential.LLMMIRecASPCFAdaptiveSIL import SILRecomposer,segment_ids,file_sha
def state():return torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()
def restore(v):torch.set_rng_state(v[0]);torch.cuda.set_rng_state(v[1])


def short_math():
 torch.manual_seed(42)
 result=[]
 for mode in ("fixed","semantic"):
  block=SILRecomposer().double()
  assert sum(p.numel() for p in block.parameters())==37504
  for length in (0,1,2,3,4,8):
   for multi in (False,True):
    x=torch.randn(2,12,64,dtype=torch.float64,requires_grad=True)
    ll=torch.full((2,),length,dtype=torch.long)
    gaps=torch.full((2,12),2. if multi else 0.,dtype=torch.float64)
    ids,valid=segment_ids(gaps,ll,mode,1.,fixed_len=2 if multi else 4)
    v=torch.randn(2,4,64,dtype=torch.float64)
    with torch.no_grad():block.output.weight.zero_()
    delta,info=block(x,v,ids,valid,ll)
    assert torch.count_nonzero(delta)==0
    expected=max(min(length,3),0)
    assert torch.equal(info["behavior_memory_valid"].sum(1),torch.full((2,),expected))
    for b in range(2):
     ix=info["behavior_positions"][b][info["behavior_memory_valid"][b]]
     assert torch.equal(ix,torch.arange(max(0,length-3),length))
     assert len(ix.unique())==len(ix)
     if expected:assert torch.equal(info["behavior_memories"][b,-expected:],x[b,ix])
    allowed=info["joint_memory_valid"]
    att=info["memory_attention"]
    assert torch.count_nonzero(att.masked_select(~allowed[:,None].expand_as(att)))==0
    assert torch.allclose(att.sum(-1),torch.full((2,4),1. if length else 0.,dtype=torch.float64))
    recent_valid=valid & (torch.arange(12)[None]>=(ll-3).clamp_min(0)[:,None])
    membership=torch.nn.functional.one_hot(ids,12)*valid[:,:,None]
    duplicate=(info["segment_sizes"]==1)&((membership*recent_valid[:,:,None]).sum(1)>0)
    assert not (info["segment_memory_valid"]&duplicate).any()
    with torch.no_grad():block.output.weight.normal_(0,.02)
    active,info=block(x,v,ids,valid,ll)
    altered=x.detach().clone();altered[~valid]=1000.
    alternate,_=block(altered,v,ids,valid,ll)
    assert torch.equal(active,alternate)
    if length<=1:assert torch.count_nonzero(active)==0
    if length>=2:
     behavior_attention=info["memory_attention"][:,:,12:]
     assert (behavior_attention[:,0]-behavior_attention[:,1]).abs().max()>1e-8
     grad=torch.autograd.grad((active*torch.randn_like(active)).sum(),x)[0]
     assert torch.isfinite(grad).all() and grad[:,:length].abs().sum()>0
     assert torch.count_nonzero(grad[:,length:])==0
    result.append(dict(mode=mode,length=length,multi=multi,valid_behavior_slots=expected,pass_checks=True))
 # Reuse the already reviewed mathematical suite for K, local causality,
 # boundary-prefix invariance, head permutations, and all parameter gradients.
 spec=importlib.util.spec_from_file_location("first_sil_math",ROOT/"research/elector_sil_adaptive_20261022_v1/prepare.py")
 previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
 return result,previous.math_tests()


def fresh_real():
 c=read_corpus();models={};hashes={}
 for arm in ("B0","C1","M1"):
  a=arguments(arm,c);old.utils.init_seed(42)
  models[arm]=Model(a,c).to(a.device);hashes[arm]=common_hash(models[arm])
 assert len(set(hashes.values()))==1
 assert models["B0"].count_variables()==943174
 assert models["C1"].count_variables()==models["M1"].count_variables()==980678
 tr=c.data_df["train"];frame=tr[tr.position>=4].iloc[:32]
 hist,lengths=reuse.construct_hist(c,frame)
 users=frame.user_id.to_numpy(np.int64);target=frame.item_id.to_numpy(np.int64)
 neg=np.full(32,1,dtype=np.int64)
 tests=[]
 # Same stochastic forward, exact score/loss/query derivative at zero residual.
 for length in (1,2,3,4):
  h=hist.copy()
  for i,original in enumerate(lengths):h[i,:length]=hist[i,original-length:original]
  h[:,length:]=0
  f=dict(history_items=torch.as_tensor(h,device=a.device),
         lengths=torch.full((32,),length,device=a.device,dtype=torch.long),
         user_id=torch.as_tensor(users,device=a.device),
         item_id=torch.as_tensor(np.column_stack([target,neg]),device=a.device))
  before=state();reference=None;post=None
  for arm,m in models.items():
   m.train();restore(before)
   out=m(f,True);loss=m.loss(out);g=torch.autograd.grad(loss,m.extractor.query)[0]
   value=(out["prediction"],loss,g)
   if reference is None:reference=[x.detach() for x in value];post=state()
   else:
    assert all(torch.equal(x,y) for x,y in zip(reference,value))
    assert all(torch.equal(x,y) for x,y in zip(post,state()))
  restore(post);tests.append(dict(length=length,zero_score_loss_gradient_exact=True))
 # A few updates are only an integration check; no checkpoint is saved.
 a=arguments("B0",c);f["history_items"]=torch.as_tensor(hist,device=a.device);f["lengths"]=torch.as_tensor(lengths,device=a.device)
 opt={arm:torch.optim.Adam(m.customize_parameters(),lr=.004,weight_decay=1e-6) for arm,m in models.items()}
 for step in range(3):
  before=state()
  for arm,m in models.items():
   m.train();restore(before);opt[arm].zero_grad()
   out=m(f,True);loss=m.loss(out);assert torch.isfinite(loss);loss.backward()
   assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
   if arm!="B0":
    assert m.sil_block.output.weight.grad.norm()>0
    if step:
     assert m.sil_block.local_qkv.weight.grad.norm()>0
     assert m.sil_block.read_key.weight.grad.norm()>0
   opt[arm].step()
 # Length1 must remain exactly the model's ASPCF path even with active residual.
 for arm in ("C1","M1"):
  m=models[arm];m.eval()
  one={**f,"history_items":f["history_items"][:,:1],"lengths":torch.ones_like(f["lengths"])}
  with torch.no_grad():
   out=m(one,True)
   assert torch.equal(out["interest_vectors"],out["base_interest_vectors"])
 return list(hashes.values())[0],tests


def main():
 signal.alarm(170);torch.set_num_threads(4);start=time.perf_counter()
 verify_sources()
 short,math=short_math()
 sha,real=fresh_real()
 verify_sources()
 result=dict(seconds=time.perf_counter()-start,short_history_tests=short,causal_and_K_tests=math,
             real_short_history_tests=real,common_initial_parameter_sha256=sha,
             parameters=dict(B0=943174,C1=980678,M1=980678),
             added_parameters=37504,source_sha256=file_sha(OUT/"model_snapshot.py"),
             no_Test=True,no_full_training_started_by_check=True,checks_passed=True)
 write(OUT/"CHECK_PASS.json",result,True)
 print("CHECK_PASS "+json.dumps({k:result[k] for k in ("seconds","common_initial_parameter_sha256","parameters","checks_passed")}),flush=True)


if __name__=="__main__":main()
