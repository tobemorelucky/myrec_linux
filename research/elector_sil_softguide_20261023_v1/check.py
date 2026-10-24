"""Bounded CPU mathematical, equivalence and complete-model gradient checks."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import time,signal,copy,json
import numpy as np,torch
from common import *
from model import MemoryRead,base

def main():
 signal.alarm(170);torch.set_num_threads(2);start=time.perf_counter();verify();torch.manual_seed(73)
 for K in (1,2,4,7):
  lengths=torch.tensor([0,1,2,3,4,8,20]);B=len(lengths);L=20
  x=torch.randn(B,L,64);v=torch.randn(B,K,64)
  ids,valid=base.segment_ids(torch.zeros(B,L),lengths,'fixed',0.,2,6,4)
  z=torch.nn.functional.normalize(torch.randn(B,L,1536),dim=-1).requires_grad_()
  a_logits=torch.randn(B,K,L,requires_grad=True)
  A=base.safe_softmax(a_logits,valid[:,None])
  original=base.SILRecomposer(64,2,3,True,True)
  with torch.no_grad():original.output.weight.normal_(0,.01)
  c1=MemoryRead(copy.deepcopy(original),'C1',K)
  c0=MemoryRead(copy.deepcopy(original),'C0',K)
  m2=MemoryRead(copy.deepcopy(original),'M2',K)
  expected,old_info=original(x,v,ids,valid,lengths)
  got,new_info=c1(x,v,ids,valid,lengths)
  assert torch.equal(got,expected)
  m2.semantic_max=0.;neutral,_=m2(x,v,ids,valid,lengths,z,A)
  assert torch.equal(neutral,expected);m2.semantic_max=1.
  changed,info=m2(x,v,ids,valid,lengths,z,A)
  assert torch.isfinite(changed).all() and torch.count_nonzero(changed[:2])==0
  assert torch.allclose(info['memory_attention'].sum(-1),(lengths>0)[:,None].expand(-1,K).float())
  assert torch.count_nonzero(info['memory_attention'].masked_select(~info['joint_memory_valid'][:,None].expand_as(info['memory_attention'])))==0
  loss=changed[2:].square().sum();loss.backward()
  assert a_logits.grad.norm()>0 and m2.semantic_alpha.grad.norm()>0 and z.grad is None
  for block in (c0,c1,m2):
   xx=x.clone().requires_grad_();local_ids=torch.zeros_like(ids) if block.variant=='C0' else ids
   local=block.local_bank(xx,local_ids,valid)
   grad=torch.autograd.grad(local[-1,2].square().sum(),xx)[0]
   assert torch.count_nonzero(grad[-1,3:])==0
  with torch.no_grad():
   dirty_x=x.clone();dirty_z=z.detach().clone();dirty_x[~valid]=123.;dirty_z[~valid]=-234.
   dirty,dd=m2(dirty_x,v,ids,valid,lengths,dirty_z,A.detach())
   assert torch.equal(dirty,changed.detach())
   identical=torch.ones_like(z)
   unused,ii=m2(x,v,ids,valid,lengths,identical,A.detach())
   assert ii['semantic_bias'].abs().max()<1e-6
  assert sum(p.numel() for p in c0.parameters())==sum(p.numel() for p in c1.parameters())
  assert sum(p.numel() for p in m2.parameters())==sum(p.numel() for p in c1.parameters())+K
 c=corpus('beauty');models={arm:model('beauty',c,arm)[0] for arm in ARMS}
 frame=c.data_df['train'].query('position>=2').sample(16,random_state=42);h,l=histories(c,frame)
 f=feed(h,l,frame.user_id.to_numpy(np.int64),frame.item_id.to_numpy(np.int64),np.ones(16,np.int64),'cpu')
 initial=torch.get_rng_state();reference=None;after=None
 for arm,m in models.items():
  m.train();torch.set_rng_state(initial)
  out=m(f,True);loss=m.loss(out);g=torch.autograd.grad(loss,m.extractor.query)[0]
  now=(out['raw_prediction'].detach(),out['prediction'].detach(),loss.detach(),g)
  if reference is None:reference=now;after=torch.get_rng_state()
  else:assert all(torch.equal(a,b) for a,b in zip(reference,now)) and torch.equal(after,torch.get_rng_state())
  m.zero_grad();torch.set_rng_state(initial);out=m(f,True);m.loss(out).backward()
  assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
  if arm!='B0':assert m.sil_block.output.weight.grad.norm()>0
  m.eval()
  with torch.no_grad():
   one=m(f,True);two=m({**f,'item_id':f['item_id'].flip(1)},True)
   assert torch.equal(one['interest_vectors'],two['interest_vectors'])
 params={k:m.count_variables() for k,m in models.items()}
 assert params==dict(B0=943174,C0=980678,C1=980678,M2=980682)
 for k in (1,2,7):
  m,a=model('beauty',c,'M2',K=k);m.train();out=m(f,True)
  assert out['interest_vectors'].shape==(16,k,64);m.loss(out).backward()
  assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
  m.zero_grad();empty={**f,'history_items':torch.zeros_like(f['history_items']),'lengths':torch.zeros_like(f['lengths'])}
  out=m(empty,True);assert torch.count_nonzero(out['interest_vectors'])==0
  m.loss(out).backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
  del m
 verify();result=dict(passed=True,seconds=time.perf_counter()-start,parameters_Beauty_K4=params,
                     configurable_K=[1,2,4,7],C1_exact_original=True,C0_C1_parameters_match=True,
                     semantic_zero_recovers_C1=True,identical_semantics_neutral=True,padding_and_short_history=True,
                     local_causality=True,target_invariant_interests=True,semantic_profile_gradient=True,frozen_LLM_no_gradient=True,
                     exact_zero_initial_scores_loss_query_gradient_and_RNG=True,full_model_finite_backward=True,no_Test=True)
 write('CHECK_PASS.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
