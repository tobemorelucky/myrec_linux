"""16 paired end-to-end Train steps and full Dev; hard 170s bound, no Test."""
import os,sys,json,time,signal,fcntl,subprocess,csv,io,traceback,hashlib
from pathlib import Path
ROOT=Path('/home/yyx/hzgcode/pom2.0');OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OUT))

def claim():
 text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 inv={int(r[0]):dict(uuid=r[1].strip(),free=int(r[2]),util=int(r[3]),pids=[]) for r in csv.reader(io.StringIO(text)) if r}
 text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
 for row in csv.reader(io.StringIO(text)):
  if row:
   assert len(row)==2 and row[0].strip().isdigit()
   for v in inv.values():
    if v['uuid']==row[1].strip():v['pids'].append(int(row[0]))
 for gpu in (0,1):
  v=inv[gpu]
  if v['pids'] or v['free']<10240 or v['util']>10:continue
  fd=os.open(ROOT/'research/gpu_resource_scheduler_20261022_v1'/('gpu%d.lock'%gpu),os.O_RDWR)
  try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:os.close(fd);continue
  time.sleep(1)
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
  if v['uuid'] not in apps:return gpu,fd
  os.close(fd)
 raise RuntimeError('No idle GPU available; no sharing/preemption')

def main():
 signal.alarm(170);start=time.perf_counter();gpu,fd=claim()
 try:
  os.environ['CUDA_VISIBLE_DEVICES']=str(gpu);os.environ['PYTHONDONTWRITEBYTECODE']='1'
  os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
  import numpy as np,torch,gc
  import common as h
  from model import base
  torch.set_num_threads(4);manifest=h.verify()
  assert json.loads((OUT/'CHECK_PASS.json').read_text())['passed']
  lock=json.loads((OUT/'SOURCE_LOCK.json').read_text())
  assert all(h.sha(OUT/name)==sha for name,sha in lock.items())
  all_results={}
  for ds in ('beauty','ml-1m'):
   ds_start=time.perf_counter();h.utils.init_seed(42);c=h.corpus(ds)
   models={arm:h.model(ds,c,arm,device='cuda:0')[0] for arm in h.ARMS}
   parameters={arm:m.count_variables() for arm,m in models.items()}
   assert parameters['C0']==parameters['C1']==parameters['B0']+37504 and parameters['M2']==parameters['C1']+4
   for a in ('C0','M2'):
    for name,value in models['C1'].sil_block.state_dict().items():assert torch.equal(value,models[a].sil_block.state_dict()[name])
   eligible=np.flatnonzero(c.data_df['train'].position.to_numpy()>0)
   ix=np.random.default_rng(42).choice(eligible,size=16*1024,replace=False)
   tr=c.data_df['train'].iloc[ix];dev=c.data_df['dev']
   th,tl=h.histories(c,tr);dh,dl=h.histories(c,dev)
   users=tr.user_id.to_numpy(np.int64);positive=tr.item_id.to_numpy(np.int64)
   candidates=np.concatenate([dev.item_id.to_numpy(np.int64)[:,None],np.stack(dev.neg_items.to_numpy())],1)
   assert candidates.shape==(len(dev),1001)
   initial=h.evaluate(models['B0'],c,dh,dl,candidates,True)
   assert all(abs(initial['metrics'][key]-manifest['inputs'][ds]['initial_Dev'][key])<1e-10 for key in h.KEYS),(ds,initial)
   opts={arm:torch.optim.Adam(m.customize_parameters(),lr=.0001,weight_decay=1e-6) for arm,m in models.items()}
   g=torch.Generator(device='cuda:0').manual_seed(20261023)
   times={arm:[] for arm in models};gradients={arm:[] for arm in models};losses=[]
   torch.cuda.reset_peak_memory_stats()
   for step in range(16):
    sl=slice(step*1024,(step+1)*1024);uu=torch.as_tensor(users[sl],device='cuda:0')
    neg=h.negative(models['B0'],uu,g).cpu().numpy()
    feed=h.feed(th[sl],tl[sl],users[sl],positive[sl],neg,'cuda:0')
    before=h.rng();after=None;row=dict(step=step+1)
    for arm,m in models.items():
     m.train();h.restore(before);opts[arm].zero_grad(set_to_none=True)
     torch.cuda.synchronize();t=time.perf_counter()
     out=m(feed,True);loss=m.loss(out);assert torch.isfinite(loss);loss.backward()
     assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
     record=dict(query=float(m.extractor.query.grad.norm()));assert record['query']>0
     if arm!='B0':
      for label,p in dict(output=m.sil_block.output.weight,local_qkv=m.sil_block.local_qkv.weight,
                          read_query=m.sil_block.read_query.weight,recent_query=m.sil_block.recent_query.weight,
                          read_key=m.sil_block.read_key.weight,read_value=m.sil_block.read_value.weight).items():
       assert p.grad is not None;record[label]=float(p.grad.norm())
      assert record['output']>0
      if step>=1:assert all(record[key]>0 for key in ('local_qkv','read_query','recent_query','read_key','read_value'))
      record['delta_RMS']=float(out['sil_delta'].detach().square().mean().sqrt())
     if arm=='M2':
      record['semantic_alpha_gradient']=float(m.sil_block.semantic_alpha.grad.norm())
      if step>=1:assert record['semantic_alpha_gradient']>0
      record['semantic_bias_RMS']=float(out['semantic_bias'].detach().square().mean().sqrt())
      assert record['semantic_bias_RMS']>0
      assert m.item_encoder.llm_table.grad is None and not m.m2_unit_semantics.requires_grad
     gradients[arm].append(record);opts[arm].step()
     torch.cuda.synchronize();times[arm].append(time.perf_counter()-t);row[arm]=float(loss.detach())
     current=h.rng()
     if after is None:after=current
     else:assert all(torch.equal(a,b) for a,b in zip(current,after))
    h.restore(after);losses.append(row)
   final={arm:h.evaluate(m,c,dh,dl,candidates,True) for arm,m in models.items()}
   # Measure the semantic branch on the same final M2 parameters, not a new fit.
   m=models['M2'];m.eval()
   probe=dict(history_items=torch.as_tensor(dh[:64],device='cuda:0'),lengths=torch.as_tensor(dl[:64],device='cuda:0'),item_id=torch.as_tensor(candidates[:64,:2],device='cuda:0'))
   with torch.no_grad():
    on=m(probe,True);m.sil_block.semantic_max=0.;off=m(probe,True);m.sil_block.semantic_max=1.
    activation=dict(attention_mean_absolute_change=float((on['memory_attention']-off['memory_attention']).abs().mean()),
                    user_mean_absolute_change=float((on['user_vector']-off['user_vector']).abs().mean()),
                    semantic_alpha=m.sil_block.semantic_alpha.sigmoid().cpu().tolist(),
                    interest_profile_pairwise_cosine_mean=float((on['semantic_profile']@on['semantic_profile'].transpose(-1,-2)).mean()),
                    semantic_bias_RMS=float(on['semantic_bias'].square().mean().sqrt()))
   assert activation['attention_mean_absolute_change']>0 and activation['user_mean_absolute_change']>0
   med={arm:float(np.median(t[2:])) for arm,t in times.items()}
   estimate={arm:dict(seconds_per_epoch_compute_and_Dev=float(np.ceil(len(eligible)/1024))*med[arm]+final[arm]['seconds'],loader_IO_excluded=True) for arm in models}
   result=dict(dataset=ds,K=4,steps_per_arm=16,initial_Dev=initial,after16_Dev=final,parameters=parameters,
               fit_seconds={arm:float(sum(t)) for arm,t in times.items()},median_step_seconds=med,
               step_losses=losses,gradients=gradients,M2_activation=activation,seconds=time.perf_counter()-ds_start,
               peak_shared_process_GPU_MiB=torch.cuda.max_memory_allocated()/2**20,estimated_full_cost=estimate,
               Train_subset_sha256=hashlib.sha256(ix.tobytes()).hexdigest(),Dev_candidates_sha256=hashlib.sha256(candidates.tobytes()).hexdigest(),
               checkpoint_sha256=manifest['inputs'][ds]['checkpoint_sha256'],Test_evaluations=0,checkpoints_written=0)
   h.write(ds+'.json',result);all_results[ds]=result
   print(json.dumps(dict(dataset=ds,after16_Dev=final,parameters=parameters,fit_seconds=result['fit_seconds'],activation=activation,seconds=result['seconds'],estimated_full_cost=estimate)),flush=True)
   del out,loss,on,off,m,models,opts,c;gc.collect();torch.cuda.empty_cache()
  h.verify();h.write('COMPLETE.json',dict(passed=True,seconds=time.perf_counter()-start,GPU=gpu,Test_evaluations=0,full_training=False,protected_unchanged=True))
 finally:os.close(fd)
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  with (OUT/'SCREEN_FAILURE.json').open('x') as f:json.dump(dict(error=repr(exc),traceback=traceback.format_exc()),f,indent=2)
  raise
