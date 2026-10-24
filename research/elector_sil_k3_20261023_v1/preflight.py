import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import signal,time,gc,copy
from train_task import *
from plan import write,verify,expected_parameters
def main():
 signal.alarm(170);torch.set_num_threads(2);start=time.perf_counter();verify();result={}
 for ds in ('beauty','ml-1m'):
  c=read_corpus(ds);a=arguments(ds,'SS-C1',3,42,'cpu')
  utils.init_seed(42);m=ObjectiveSIL(a,c);core=common_hash(m);rng=torch.get_rng_state()
  assert m.count_variables()==expected_parameters(ds,3,'SS-C1')
  aa=copy.copy(a);aa.sil_variant='B0';utils.init_seed(42);reference=ObjectiveSIL(aa,c)
  assert common_hash(reference)==core and torch.equal(torch.get_rng_state(),rng)
  frame=c.data_df['train'].query('position>=3').sample(16,random_state=42)
  hist=np.zeros((16,20),np.int64);length=[]
  for j,(u,p) in enumerate(zip(frame.user_id,frame.position)):
   seq=c.user_his[int(u)][max(0,int(p)-20):int(p)]
   length.append(len(seq));hist[j,:len(seq)]=[x[0] for x in seq]
  f=dict(history_items=torch.from_numpy(hist),lengths=torch.tensor(length),
    user_id=torch.as_tensor(frame.user_id.to_numpy(np.int64)),
    item_id=torch.as_tensor(np.column_stack([frame.item_id.to_numpy(np.int64),np.ones(16,np.int64)])))
  reference.train();torch.set_rng_state(rng);left=reference(f,True);ll=reference.loss(left)
  lg=torch.autograd.grad(ll,reference.extractor.query)[0];after=torch.get_rng_state()
  m.train();torch.set_rng_state(rng);right=m(f,True);loss=m.loss(right)
  rg=torch.autograd.grad(loss,m.extractor.query)[0]
  assert right['interest_vectors'].shape==(16,3,64)
  assert torch.equal(left['prediction'],right['prediction']) and torch.equal(ll,loss) and torch.equal(lg,rg)
  assert torch.equal(after,torch.get_rng_state())
  m.zero_grad();out=m(f,True);m.loss(out).backward()
  assert m.sil_block.output.weight.grad.norm()>0
  assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
  write(OUT/'initialization'/(ds+'_k3_s42.json'),dict(common_initial_sha256=core,K=3,seed=42,source='CPU initialization-only C1/B0 equivalence'))
  result[ds]=dict(parameters=m.count_variables(),common_initial_sha256=core,zero_initial_loss_score_gradient_RNG_exact=True,finite_backward=True)
  del m,reference,c;gc.collect()
 verify();r=dict(passed=True,seconds=time.perf_counter()-start,datasets=result,no_Test=True,no_Dev_performance_gate=True)
 write(OUT/'CHECK_PASS.json',r);print(json.dumps(r))
if __name__=='__main__':main()
