"""CPU-only bounded math/gradient and protocol equivalence checks; no Test."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import copy,gc,importlib.util,json,signal,time
import numpy as np,torch
from train_task import *
from plan import expected_parameters,select_by_dev
from objective_model import soft
def main():
 signal.alarm(170);torch.set_num_threads(2);start=time.perf_counter();verify()
 result={'passed':False,'no_Test':True,'no_Dev_performance_gate':True,'datasets':{}}
 # Synthetic local causality, masks, empty/short history, all configured K.
 for K in (1,2,4,8):
  lengths=torch.tensor([0,1,2,3,4,8,20]);B=len(lengths);L=20
  torch.manual_seed(901+K);x=torch.randn(B,L,64);v=torch.randn(B,K,64)
  ids,valid=base.segment_ids(torch.zeros(B,L),lengths,'fixed',0.,2,6,4)
  for name in ('C1','A1','A2'):
   block=soft.MemoryRead(base.SILRecomposer(),'C1',K)
   block.use_local=name!='A1';block.use_behavior=name!='A2'
   zero,info=block(x,v,ids,valid,lengths);assert torch.count_nonzero(zero)==0
   with torch.no_grad():block.output.weight.normal_(0,.01)
   got,info=block(x,v,ids,valid,lengths)
   assert torch.isfinite(got).all() and torch.count_nonzero(got[:2])==0
   assert torch.allclose(info['memory_attention'].sum(-1),(lengths>0)[:,None].float().expand(-1,K))
   mask=info['joint_memory_valid'][:,None].expand_as(info['memory_attention'])
   assert torch.count_nonzero(info['memory_attention'].masked_select(~mask))==0
   if name=='A2':
    assert info['joint_memory_valid'].shape==(B,L)
    assert torch.equal(info['segment_memory_valid'],info['segment_sizes']>0)
    assert torch.count_nonzero(info['behavior_memory_valid'])==0
   dirty=x.clone();dirty[~valid]=876.
   assert torch.equal(block(dirty,v,ids,valid,lengths)[0],got)
   xx=x.clone().requires_grad_();local=block.local_bank(xx,ids,valid)
   grad=torch.autograd.grad(local[-1,2].square().sum(),xx)[0]
   assert torch.count_nonzero(grad[-1,3:])==0
   isolated=torch.autograd.grad(block.local_bank(xx,ids,valid)[-1,5].square().sum(),xx)[0]
   if name!='A1':assert torch.count_nonzero(isolated[-1,:4])==0
   if name=='A1':assert torch.equal(local,x.masked_fill(~valid[:,:,None],0))
   vv=v.clone().requires_grad_();block(xx,vv,ids,valid,lengths)[0].square().sum().backward()
   assert torch.isfinite(xx.grad).all() and vv.grad.norm()>0
 assert select_by_dev({2:.2,4:.2,8:.1})==2 and select_by_dev({2:.1,4:.2,8:.3})==8
 # Complete model checks, both datasets; no test Dataset ever constructed.
 for ds in ('beauty','ml-1m'):
  c=read_corpus(ds);K=4 if ds=='beauty' else 2
  frame=c.data_df['train'].query('position>=3').sample(16,random_state=73)
  h=np.zeros((16,20),np.int64);length=[]
  for j,(u,p) in enumerate(zip(frame.user_id,frame.position)):
   seq=c.user_his[int(u)][max(0,int(p)-20):int(p)];length.append(len(seq));h[j,:len(seq)]=[v[0] for v in seq]
  f=dict(history_items=torch.from_numpy(h),lengths=torch.tensor(length),
    user_id=torch.as_tensor(frame.user_id.to_numpy(np.int64)),
    item_id=torch.as_tensor(np.column_stack([frame.item_id.to_numpy(np.int64),np.ones(16,np.int64)])))
  reference=None;initial_rng=None;after=None;core=None;params={}
  # Archive and new C1 are identical even with trained nonzero residuals.
  old_path=ROOT/'research/elector_sil_objective_full_20261023_v2/softguide_snapshot.py'
  spec=importlib.util.spec_from_file_location('old_frozen_softguide',old_path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
  a=arguments(ds,'SS-C1',K,42,'cpu');utils.init_seed(42)
  old_model=old.SoftGuideSIL(a,c);new_model=ObjectiveSIL(a,c)
  checkpoint=next(r for r in json.loads((OUT/'REUSED.json').read_text()) if r['dataset']==ds and r['arm']=='SS-C1')['checkpoint']
  state=torch.load(checkpoint,map_location='cpu',weights_only=True)
  old_model.load_state_dict(state,strict=True);new_model.load_state_dict(state,strict=True)
  old_model.eval();new_model.eval()
  with torch.no_grad():
   left=old_model(f,True);right=new_model(f,True)
   assert torch.equal(left['prediction'],right['prediction']) and torch.equal(left['interest_vectors'],right['interest_vectors'])
  del old_model,new_model,state;gc.collect()
  for arm in ('SS-B0','SS-C0','SS-C1','SS-A1','SS-A2'):
   a=arguments(ds,arm,K,42,'cpu');utils.init_seed(42);m=ObjectiveSIL(a,c)
   params[arm]=m.count_variables();assert params[arm]==expected_parameters(ds,K,arm)
   digest=common_hash(m)
   if core is None:core=digest;initial_rng=torch.get_rng_state()
   else:assert core==digest and torch.equal(initial_rng,torch.get_rng_state())
   expected=json.loads((OUT/'initialization'/('%s_k%d_s42.json'%(ds,K))).read_text())['common_initial_sha256']
   assert digest==expected
   m.train();torch.set_rng_state(initial_rng);out=m(f,True);loss=m.loss(out)
   grad=torch.autograd.grad(loss,m.extractor.query)[0];now=(out['prediction'].detach(),loss.detach(),grad)
   if reference is None:reference=now;after=torch.get_rng_state()
   else:assert all(torch.equal(x,y) for x,y in zip(reference,now)) and torch.equal(after,torch.get_rng_state())
   if arm in ('SS-A1','SS-A2'):
    opt=torch.optim.Adam(m.customize_parameters(),lr=a.lr,weight_decay=a.l2)
    for step in range(3):
     opt.zero_grad();got=m(f,True);value=m.loss(got);assert torch.isfinite(value);value.backward()
     assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
     assert m.sil_block.output.weight.grad.norm()>0
     if arm=='SS-A1':assert m.sil_block.local_qkv.weight.grad is None
     if arm=='SS-A2' and step>0:assert m.sil_block.local_qkv.weight.grad.norm()>0
     opt.step()
    assert m.item_encoder.llm_table.grad is None
   m.eval()
   with torch.no_grad():
    out=m(f,True);other=m({**f,'item_id':f['item_id'].flip(1)},True)
    assert torch.equal(out['interest_vectors'],other['interest_vectors'])
   del m;gc.collect()
  result['datasets'][ds]=dict(parameters=params,common_seed42_sha256=core,
    archived_trained_C1_exact=True,zero_initial_scores_loss_query_gradient_RNG_exact=True,
    ablation_three_updates_finite=True,frozen_LLM=True,target_independent=True)
  del c;gc.collect()
 # Configurable K2/K8 and alternate seed initialization on real Beauty input.
 c=read_corpus('beauty')
 for K in (2,8):
  hashes=[];rngs=[]
  for arm in ('SS-B0','SS-C0','SS-C1'):
   a=arguments('beauty',arm,K,0,'cpu');utils.init_seed(0);m=ObjectiveSIL(a,c)
   assert m.count_variables()==expected_parameters('beauty',K,arm)
   hashes.append(common_hash(m));rngs.append(torch.get_rng_state());del m;gc.collect()
  assert len(set(hashes))==1 and all(torch.equal(rngs[0],r) for r in rngs)
 # Preserve the exact original scoring, history boundary and Dev evaluator implementations.
 import ast
 original_source=(ROOT/'research/elector_sil_objective_full_20261023_v2/train_task.py').read_text()
 def nodes(source):
  return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(source).body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
 old_nodes=nodes(original_source);new_nodes=nodes((OUT/'train_task.py').read_text())
 for name in ('read_corpus','score_views','DevRunner','common_hash'):assert old_nodes[name]==new_nodes[name],name
 from scheduler import initial_tasks,beauty_remaining,resource,RESOURCE
 assert len(initial_tasks('beauty'))==6 and len(initial_tasks('ml-1m'))==17 and len(beauty_remaining(4))==17
 for gpu in (0,1):assert (RESOURCE/('gpu%d.lock'%gpu)).resolve().is_relative_to(ROOT)
 real_observe,real_acquire,real_sleep,real_close=resource.observe,resource.acquire,time.sleep,os.close
 import scheduler
 scheduler.beauty_has_priority=lambda inv:False
 def exercise(sequence,locked=False):
  seq=iter(sequence);closed=[]
  resource.observe=lambda:next(seq);resource.acquire=lambda p:None if locked else 987654
  time.sleep=lambda seconds:None;os.close=lambda fd:closed.append(fd)
  return scheduler.acquire_gpu('beauty'),closed
 base_inv={0:dict(pids=[123],free=14000,util=30),1:dict(pids=[],free=16000,util=0)}
 try:
  assert exercise([base_inv,base_inv])[0]==(1,987654)
  assert exercise([base_inv],True)[0]==(None,None)
  busy={0:base_inv[0],1:dict(pids=[321],free=14000,util=30)}
  assert exercise([base_inv,busy])[0]==(None,None)
  fallback={0:dict(pids=[],free=16000,util=0),1:busy[1]}
  assert exercise([fallback,fallback])[0]==(0,987654)
 finally:resource.observe,resource.acquire,time.sleep,os.close=real_observe,real_acquire,real_sleep,real_close
 verify()
 result.update(passed=True,seconds=time.perf_counter()-start,K_checks=[1,2,4,8],
    ablations_local_causality_padding_gradient=True,A2_singletons_restored=True,
    A1_bypass_only=True,Beauty_K_selection_Dev_only=True)
 write(OUT/'CHECK_PASS.json',result);print(json.dumps(result))
if __name__=='__main__':main()
