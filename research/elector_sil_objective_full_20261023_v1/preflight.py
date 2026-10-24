"""Bounded CPU preflight: reuse/fresh initialization and exact original loss checks."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import signal,time,json,gc
import torch,numpy as np
from train_task import *

def main():
 signal.alarm(170);torch.set_num_threads(2);t=time.perf_counter();manifest=verify();results={}
 for ds in ('beauty','ml-1m'):
  c=read_corpus(ds);frame=c.data_df['train'].query('position>=3').sample(16,random_state=42)
  hist=np.zeros((16,20),np.int64);length=[]
  for j,(u,p) in enumerate(zip(frame.user_id,frame.position)):
   seq=c.user_his[int(u)][max(0,int(p)-20):int(p)];length.append(len(seq));hist[j,:len(seq)]=[v[0] for v in seq]
  f=dict(history_items=torch.from_numpy(hist),lengths=torch.tensor(length),user_id=torch.as_tensor(frame.user_id.to_numpy(np.int64)),item_id=torch.as_tensor(np.column_stack([frame.item_id.to_numpy(np.int64),np.ones(16,np.int64)])))
  params={};new={}
  for arm in manifest['all_six']:
   a=arguments(ds,arm,'cpu');utils.init_seed(42);m=ObjectiveSIL(a,c);new[arm]=m;params[arm]=m.count_variables()
   assert common_hash(m)==manifest['datasets'][ds]['expected_common_initial_sha256']
  assert params['BPR-B0']==params['SS-B0']
  assert params['BPR-C1']==params['SS-C0']==params['SS-C1']==params['SS-B0']+37504
  assert params['SS-M2']==params['SS-C1']+manifest['datasets'][ds]['K']
  # Same original ASPCF parameters, same raw BPR and relation loss and query gradient.
  a=arguments(ds,'BPR-B0','cpu');utils.init_seed(42);reference=ASPCF(a,c)
  assert common_hash(reference)==manifest['datasets'][ds]['expected_common_initial_sha256']
  before=torch.get_rng_state();reference.train();orig=reference(f,True);loss=reference.loss(orig)
  qgrad=torch.autograd.grad(loss,reference.extractor.query)[0];after=torch.get_rng_state()
  for arm in ('BPR-B0','BPR-C1'):
   m=new[arm];m.train();torch.set_rng_state(before);out=m(f,True);got=m.loss(out);grad=torch.autograd.grad(got,m.extractor.query)[0]
   assert torch.equal(out['raw_prediction'],orig['prediction']) and torch.equal(loss,got) and torch.equal(qgrad,grad)
   assert torch.equal(after,torch.get_rng_state())
   assert torch.equal(out['_bpr_scores'],out['raw_prediction'])
  # Reused trained SS-C1 is also exactly equivalent after nonzero residuals.
  old_root=Path(manifest['datasets'][ds]['old_root']);a=arguments(ds,'SS-C1','cpu')
  old=base.LLMMIRecASPCFAdaptiveSIL(a,c)
  state=torch.load(old_root/'arms/C1/best.pt',map_location='cpu',weights_only=True)
  old.load_state_dict(state,strict=True);candidate=new['SS-C1'];candidate.load_state_dict(state,strict=True)
  rng=torch.get_rng_state();old.train();oo=old(f,True);ll=old.loss(oo);gg=torch.autograd.grad(ll,old.extractor.query)[0]
  torch.set_rng_state(rng);candidate.train();nn=candidate(f,True);nloss=candidate.loss(nn);ng=torch.autograd.grad(nloss,candidate.extractor.query)[0]
  assert torch.equal(oo['prediction'],nn['prediction']) and torch.equal(ll,nloss) and torch.equal(gg,ng)
  # Two engineering updates, no Dev performance threshold or checkpoint output.
  m=new['BPR-C1'];opt=torch.optim.Adam(m.customize_parameters(),lr=a.lr,weight_decay=a.l2)
  for step in range(2):
   m.train();opt.zero_grad();out=m(f,True);v=m.loss(out);assert torch.isfinite(v);v.backward()
   assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
   assert m.sil_block.output.weight.grad.norm()>0;opt.step()
  results[ds]=dict(parameters=params,common_initial_sha256=manifest['datasets'][ds]['expected_common_initial_sha256'],
       original_BPR_raw_score_loss_query_gradient_exact=True,original_BPR_RNG_exact=True,
       trained_C1_SS_forward_loss_query_gradient_exact=True,two_BPR_updates_finite=True,
       reused_SS_checkpoint_Test_and_config_verified=True,embedding=manifest['datasets'][ds]['args']['llm_emb_path'])
  del new,m,old,candidate,reference,state,c;gc.collect()
 verify();write(OUT/'CHECK_PASS.json',dict(passed=True,seconds=time.perf_counter()-t,datasets=results,no_Dev_performance_gate=True,no_Test=True,no_complete_training=True))
 print(json.dumps(dict(passed=True,seconds=time.perf_counter()-t,datasets=results)),flush=True)
if __name__=='__main__':main()
