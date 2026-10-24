"""Bounded CPU-only check of copied ablations and exact seven-job protocol."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import copy,gc,signal,time,ast
from train_task import *
from plan import manifest,check_initialization,expected_parameters
from objective_model import soft
def main():
 signal.alarm(170);torch.set_num_threads(2);start=time.perf_counter();mfest=verify()
 assert len(mfest['tasks'])==7
 # Existing exact model-source and inherited full ablation check are reused, not regenerated.
 old=ROOT/'research/elector_sil_formal_20261023_v1'
 for name in ('base_snapshot.py','softguide_snapshot.py','objective_model.py'):assert sha(OUT/name)==sha(old/name)
 def nodes(source):
  return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 before=nodes((old/'train_task.py').read_text());after=nodes((OUT/'train_task.py').read_text())
 for name in ('read_corpus','score_views','DevRunner','common_hash'):assert before[name]==after[name]
 lengths=torch.tensor([0,1,2,3,4,8,20]);B=len(lengths);L=20;K=3
 x=torch.randn(B,L,64);v=torch.randn(B,K,64)
 ids,valid=base.segment_ids(torch.zeros(B,L),lengths,'fixed',0.,2,6,4)
 for variant in ('A1','A2'):
  block=soft.MemoryRead(base.SILRecomposer(),'C1',K)
  block.use_local=variant!='A1';block.use_behavior=variant!='A2'
  assert torch.count_nonzero(block(x,v,ids,valid,lengths)[0])==0
  with torch.no_grad():block.output.weight.normal_(0,.01)
  got,info=block(x,v,ids,valid,lengths)
  assert torch.isfinite(got).all() and torch.count_nonzero(got[:2])==0
  dirty=x.clone();dirty[~valid]=987.
  assert torch.equal(block(dirty,v,ids,valid,lengths)[0],got)
  xx=x.clone().requires_grad_();local=block.local_bank(xx,ids,valid)
  grad=torch.autograd.grad(local[-1,2].square().sum(),xx)[0]
  assert torch.count_nonzero(grad[-1,3:])==0
  if variant=='A2':
   assert torch.count_nonzero(info['behavior_memory_valid'])==0
   assert torch.equal(info['segment_memory_valid'],info['segment_sizes']>0)
  mask=info['joint_memory_valid'][:,None].expand_as(info['memory_attention'])
  assert torch.count_nonzero(info['memory_attention'].masked_select(~mask))==0
 results={}
 for ds in ('beauty','ml-1m'):
  c=read_corpus(ds);frame=c.data_df['train'].query('position>=3').sample(16,random_state=42)
  h=np.zeros((16,20),np.int64);length=[]
  for j,(u,p) in enumerate(zip(frame.user_id,frame.position)):
   seq=c.user_his[int(u)][max(0,int(p)-20):int(p)];length.append(len(seq));h[j,:len(seq)]=[v[0] for v in seq]
  f=dict(history_items=torch.from_numpy(h),lengths=torch.tensor(length),
   user_id=torch.as_tensor(frame.user_id.to_numpy(np.int64)),
   item_id=torch.as_tensor(np.column_stack([frame.item_id.to_numpy(np.int64),np.ones(16,np.int64)])))
  for task in [t for t in mfest['tasks'] if t['dataset']==ds]:
   K=task['K'];arm=task['arm'];a=arguments(ds,arm,K,42,'cpu');utils.init_seed(42);m=ObjectiveSIL(a,c)
   check_initialization(ds,K,42,arm,common_hash(m))
   assert m.count_variables()==expected_parameters(ds,K,arm)
   rng=torch.get_rng_state();m.train();torch.set_rng_state(rng);out=m(f,True);loss=m.loss(out)
   assert out['interest_vectors'].shape==(16,K,64) and torch.isfinite(loss)
   loss.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
   if arm!='SS-B0':assert m.sil_block.output.weight.grad.norm()>0
   unused=0
   if arm in ('SS-A1','SS-A2'):
    opt=torch.optim.Adam(m.customize_parameters(),lr=a.lr,weight_decay=a.l2)
    for step in range(2):
     opt.zero_grad();out=m(f,True);value=m.loss(out);assert torch.isfinite(value);value.backward()
     assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
     opt.step()
    if arm=='SS-A1':
     for attr in ('local_input_norm','local_qkv','local_out','local_output_norm'):
      module=getattr(m.sil_block,attr)
      assert all(p.grad is None for p in module.parameters())
      unused+=sum(p.numel() for p in module.parameters())
     assert unused==16640
    else:assert m.sil_block.local_qkv.weight.grad.norm()>0
   m.eval()
   with torch.no_grad():
    u=m(f,True);other=m({**f,'item_id':f['item_id'].flip(1)},True)
    assert torch.equal(u['interest_vectors'],other['interest_vectors'])
   results[task_id(ds,K,42,arm)]=dict(parameters=m.count_variables(),inactive_local_parameters=unused,
        core_initial_sha256=common_hash(m) if arm not in ('SS-A1','SS-A2') else 'initial ledger verified before CPU-only updates')
   del m;gc.collect()
  del c;gc.collect()
 verify();value=dict(passed=True,seconds=time.perf_counter()-start,tasks=results,
   exact_model_sources_reused=True,original_evaluation_AST_exact=True,
   padding_local_causality_finite_gradients=True,A2_singletons_restored=True,
   no_Dev_performance_gate=True,no_Test=True,no_checkpoints_saved=True)
 write(OUT/'CHECK_PASS.json',value);print(json.dumps(value))
if __name__=='__main__':main()
