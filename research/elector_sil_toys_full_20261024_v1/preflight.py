"""Bounded real-Toys integration and resource check; no Test evaluation."""
import signal,time,importlib.util,resource as host_resource
from common import *
def main():
 signal.alarm(175);start=time.perf_counter();torch.set_num_threads(4)
 spec=importlib.util.spec_from_file_location('safe_resource',ROOT/'research/elector_sil_objective_full_20261023_v2/scheduler.py')
 res=importlib.util.module_from_spec(spec);spec.loader.exec_module(res)
 state=read(ROOT/'research/elector_sil_night12_20261023_v1/STATE.json');assert state['state']=='COMPLETED' and not state['running']
 gpu=None;fd=None
 for g,v in res.observe().items():
  if res.idle(v):
   f=res.acquire(ROOT/'research/gpu_resource_scheduler_20261022_v1'/('gpu%d.lock'%g))
   if f is not None:
    if res.idle(res.observe()[g]):gpu,fd=g,f;break
    os.close(f)
 assert gpu is not None,'No safely idle GPU for bounded preflight; no task preemption'
 os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
 try:
  print('PREFLIGHT_GPU_RESERVED',gpu,flush=True)
  c=read_corpus();tr=c.data_df['train'];actual=tr[['user_id','item_id','position']].to_numpy(np.int64)
  counts=np.bincount(tr.item_id.to_numpy(np.int64),minlength=c.n_items)
  meta=dict(schema='adaptive_sil_train_v1',dataset='toys',n_items=int(c.n_items),train_frame_sha256=array_sha(actual),train_csv_sha256=sha(ROOT/'data/toys/train.csv'),embedding_sha256=sha(ROOT/'data/toys/handled/llm_table_pca1536.pkl'),semantic_dimension=1536,history_max=20,threshold=0.,threshold_source='unused fixed segmentation; no semantic threshold fit or search',min_segment_length=2,max_segment_length=6,fixed_segment_length=4,train_item_counts=counts.tolist())
  write(OUT/'TRAIN_METADATA.json',meta)
  write(OUT/'CHECK_TRAIN_ASSET.json',dict(meta,lambda_pop=.30,locked_scale=1.,raw_pop_lambda=.30,stage='preflight only; not valid for formal training'))
  print('TRAIN_ONLY_METADATA_READY',len(tr),c.n_items,flush=True)
  init=None;params={};peaks={};gradient={}
  for arm in ORDER:
   m,a=model(arm,c,'cuda:0',True);h=common_hash(m)
   if init is None:init=h
   assert init==h,'Common initialization mismatch'
   params[arm]=m.count_variables();m=m.to(a.device);torch.cuda.reset_peak_memory_stats()
   d=m.Dataset(m,c,'train');d.prepare()
   d.data['neg_items']=np.ones((len(d),1),np.int64)
   rng=np.random.default_rng(20261024);ix=rng.choice(len(d),size=1024,replace=False)
   for j in ix:
    u=int(d.data['user_id'][j]);neg=int(rng.integers(1,c.n_items))
    while neg in c.train_clicked_set[u]:neg=int(rng.integers(1,c.n_items))
    d.data['neg_items'][j,0]=neg
   feed=utils.batch_to_gpu(d.collate_batch([d[int(j)] for j in ix]),a.device)
   m.eval()
   with torch.no_grad():
    check=m(feed,True)
    if arm!='BPR-B0':assert torch.equal(check['interest_vectors'],check['base_interest_vectors'])
   del check
   if arm=='SS-C1':
    dev=m.Dataset(m,c,'dev');dev.prepare();batch=dev.collate_batch([dev[j] for j in range(256)])
    with torch.no_grad():out=m(utils.batch_to_gpu(batch,a.device),True)
    assert out['prediction'].shape==(256,1001) and torch.isfinite(out['prediction']).all()
    del out,dev,batch
   if getattr(m,'sil_block',None) is not None:
    with torch.no_grad():m.sil_block.output.weight.normal_(0,.02)
   m.train();m.zero_grad(set_to_none=True);out=m(feed);loss=m.loss(out);assert torch.isfinite(loss);loss.backward()
   grads={n:float(p.grad.norm()) for n,p in m.named_parameters() if p.grad is not None}
   assert grads and all(np.isfinite(v) for v in grads.values())
   if arm in ('SS-C0','SS-C1'):assert all(v>0 for n,v in grads.items() if n.startswith('sil_block.'))
   gradient[arm]=dict(loss=float(loss),finite=True,new_active_groups=sum(n.startswith('sil_block.') and v>0 for n,v in grads.items()))
   peaks[arm]=torch.cuda.max_memory_allocated()/2**20
   print('FULL_MODEL_CHECK',arm,'params',params[arm],'GPU_peak_MiB',peaks[arm],flush=True)
   del m,d,feed,out,loss;gc.collect();torch.cuda.empty_cache()
  assert params['SS-C0']==params['SS-C1']==params['SS-B0']+37504 and params['BPR-B0']==params['SS-B0']
  # Preserve original NumPy tie/duplicate-slot ranking, with bounded host RAM.
  rng=np.random.default_rng(42);scores=rng.integers(-2,3,size=(701,1001)).astype(float)
  rr=np.argwhere((-scores).argsort(1)==0)[:,1]+1
  truth=BaseRunner.evaluate_method(scores,[5,10,20],['HR','NDCG']);got=metrics_from_ranks(rr)
  assert all(abs(truth[k]-got[k])<1e-15 for k in KEYS)
  # Generic short-history checks use the unchanged shared C1 operator.
  block=base.SILRecomposer().double();x=torch.randn(8,20,64,dtype=torch.float64);lens=torch.tensor([0,1,2,3,4,5,8,20]);ids,valid=base.segment_ids(torch.zeros(8,20,dtype=torch.float64),lens,'fixed',0.)
  v=torch.randn(8,4,64,dtype=torch.float64);delta,_=block(x,v,ids,valid,lens);assert torch.count_nonzero(delta)==0
  with torch.no_grad():block.output.weight.normal_(0,.02)
  delta,_=block(x,v,ids,valid,lens);altered=x.clone();altered[~valid]=1000;other,_=block(altered,v,ids,valid,lens)
  assert torch.equal(delta,other) and torch.count_nonzero(delta[:2])==0
  xx=torch.randn(1,20,64,dtype=torch.float64,requires_grad=True);ii,vv=base.segment_ids(torch.zeros(1,20,dtype=torch.float64),torch.tensor([20]),'fixed',0.)
  bank=block.local_bank(xx,ii,vv);gg=torch.autograd.grad(bank[:,7].square().sum(),xx)[0];assert torch.count_nonzero(gg[:,8:])==0
  sources=['main.py','models/BaseModel.py','models/sequential/LLMMIRecASPCF.py','models/sequential/LLMMIRecASPCFSampledSoftmax.py','models/sequential/llmmi_components.py','models/sequential/llmmi_utils.py','helpers/BaseReader.py','helpers/SeqReader.py','helpers/BaseRunner.py','utils/utils.py']
  assets={}
  for rel in ['train.csv','dev.csv','test.csv','SeqReader.pkl','item2id.json','handled/llm_table_pca1536.pkl']:
   p=ROOT/'data/toys'/rel;assert p.resolve().is_relative_to(ROOT);s=p.stat();assets[str(p)]=dict(size=s.st_size,mtime_ns=s.st_mtime_ns,sha256=sha(p))
  write(OUT/'MANIFEST.json',dict(source_sha256={p:sha(ROOT/p) for p in sources},assets=assets,order=ORDER,parameters=params))
  write(OUT/'CHECK_PASS.json',dict(passed=True,seconds=time.perf_counter()-start,GPU=gpu,parameters=params,common_initial_sha256=init,GPU_peak_MiB=peaks,peak_host_RSS_MiB=host_resource.getrusage(host_resource.RUSAGE_SELF).ru_maxrss/1024,gradient_checks=gradient,short_history_padding_causality=True,zero_initial_identity=True,NumPy_duplicate_tie_protocol_equivalent=True,no_Test=True,no_full_training=True))
  print('PREFLIGHT_PASS',json.dumps(read(OUT/'CHECK_PASS.json')),flush=True)
 finally:os.close(fd)
if __name__=='__main__':main()
