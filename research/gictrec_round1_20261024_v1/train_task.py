"""Eight authorized complete continuations. Best locked-prior Dev -> one fixed Test."""
import argparse,traceback,gc
from common import *
class Runner(BaseRunner):
 def __init__(self,a,path):super().__init__(a);self.path=path
 def evaluate(self,dataset,topks,metrics_):
  assert dataset.phase=='dev'
  got=score_views(dataset);protocol_lock(self.ds,'dev',got)
  self.last=dict(epoch=int(dataset.model.cur_epoch),raw=got['metrics']['raw'],locked_prior=got['metrics']['locked_prior'],metrics=got['metrics']['locked_prior'],selection='locked-prior Dev NDCG@5',input_hash=got['input_hash'])
  with (self.path/'dev_metrics.jsonl').open('a') as f:f.write(json.dumps(self.last)+'\n')
  return self.last['metrics']
 def fit(self,dataset,epoch=-1):
  value=super().fit(dataset,epoch)
  if not np.isfinite(value):raise FloatingPointError('Nonfinite epoch loss; stop, no retry')
  return value
def run(ds,variant):
 verify();p=OUT/'runs'/(ds+'_'+variant);assert p.is_dir() and not (p/'train.log').exists()
 logging.basicConfig(filename=p/'train.log',filemode='x',level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',force=True)
 logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
 start=time.perf_counter();m,a,c=make_model(ds,variant,'cuda:0')
 a.model_path=str(p/'best.pt');m.model_path=a.model_path
 initial=common_hash(m);data=datasets(m,a,c)
 cfg=dict(dataset=ds,variant=variant,K=a.K,seed=42,args=vars(a),head=read(OUT/'PLAN.json')['head'],
  source_sha256=read(OUT/'SOURCE_LOCK.json'),init_c1=read(OUT/'PLAN.json')['datasets'][ds],common_initial_sha256=initial,
  bank_sha256=sha(read(OUT/'PLAN.json')['datasets'][ds]['bank']),parameters_registered=m.count_variables(),
  trainable_parameters=sum(t.numel() for t in m.parameters() if t.requires_grad),new_active_parameters=0 if variant=='G0' else 4096,
  physical_GPU=os.environ['CUDA_VISIBLE_DEVICES'],selection='locked-prior Dev NDCG@5; epoch0 eligible',lambda_pop=.30,
  locked_scale_asset=a.sil_asset_path,prior_raw_lambda=m.sil_raw_pop_lambda,optimizer='fresh Adam state; original constant lr '+str(a.lr),
  G1='whole-user neighbor enhancement control, not full SAGE reproduction',budget='200 maximum continuation epochs; original BaseRunner stopping; common epoch0 anchor')
 write(p/'config.json',cfg)
 r=Runner(a,p);r.ds=ds
 m.cur_epoch=0
 # Full Dev only, including raw and locked-prior anchor; no optimizer step yet.
 anchor=r.evaluate(data['dev'],[5,10,20],['HR','NDCG'])
 expected=read(Path(cfg['init_c1']['config']).parent/'best_metadata.json')
 for scheme in ('raw','locked_prior'):
  assert all(abs(r.last[scheme][k]-expected[scheme][k])<1e-10 for k in KEYS),'epoch0 differs from frozen C1'
 write(p/'EPOCH0.json',r.last)
 criteria=[anchor['NDCG@5']];best=copy.deepcopy(r.last);best_epoch=0
 torch.save(m.state_dict(),p/'best.pt');write(p/'best_metadata.json',best)
 write(p/'READY.json',dict(pid=os.getpid(),GPU=os.environ['CUDA_VISIBLE_DEVICES'],dataset=ds,variant=variant,K=a.K,Train_rows=len(data['train']),Dev_rows=len(data['dev']),cache_bytes=m.semantic_index.bytes(),common_initial_sha256=initial,parameters=cfg['trainable_parameters']))
 logging.info('TRAIN_START %s %s; full continuation; epoch0 N5 %.9f',ds,variant,criteria[0])
 epoch_times=[];epoch=0
 for epoch in range(1,a.epoch+1):
  tick=time.perf_counter();gc.collect();torch.cuda.empty_cache();m._cur_epoch=epoch
  loss=r.fit(data['train'],epoch);train_seconds=time.perf_counter()-tick
  values=r.evaluate(data['dev'],[5,10,20],['HR','NDCG']);criteria.append(values['NDCG@5'])
  better=values['NDCG@5']>=best['metrics']['NDCG@5']
  if better:
   best=copy.deepcopy(r.last);best_epoch=epoch;torch.save(m.state_dict(),p/'best.pt');write(p/'best_metadata.json',best,False)
  elapsed=time.perf_counter()-tick;epoch_times.append(dict(epoch=epoch,train_seconds=train_seconds,total_seconds=elapsed))
  logging.info('Epoch %d loss=%.6f train=%.2fs total=%.2fs locked-prior Dev=%s %s',epoch,loss,train_seconds,elapsed,values,'*' if better else '')
  if a.early_stop>0 and r.eval_termination(criteria):
   logging.info('Early stop at %d',epoch);break
 m.actions_after_train();train_seconds=time.perf_counter()-start
 m.load_state_dict(torch.load(p/'best.pt',map_location=m.device,weights_only=True),strict=True);checkpoint=sha(p/'best.pt')
 train_result=dict(train_complete=True,Dev_best_selected=True,Test_accepted=False,best=best,best_epoch=best_epoch,epochs=epoch,
  training_seconds=train_seconds,epoch_times=epoch_times,checkpoint_sha256=checkpoint,config=cfg,epoch0=read(p/'EPOCH0.json'))
 write(p/'train_result.json',train_result)
 # Test can only be read after best weights, prior, and configuration are frozen.
 del data;gc.collect();heldout(c,a,'test');td=m.Dataset(m,c,'test');td.prepare()
 write(p/'TEST_STARTED.json',dict(checkpoint_sha256=checkpoint,best_epoch=best_epoch,passes=1,lambda_pop=.30,bank_sha256=cfg['bank_sha256']))
 tick=time.perf_counter();got=score_views(td,p/'test_user_ranks.csv');test_seconds=time.perf_counter()-tick
 assert sha(p/'best.pt')==checkpoint;protocol_lock(ds,'test',got)
 with (p/'test_metrics.csv').open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['scheme',*KEYS]);writer.writeheader()
  for s,v in got['metrics'].items():writer.writerow(dict(scheme=s,**v))
 result=dict(**train_result,Test=got['metrics'],Test_passes=1,Test_seconds=test_seconds,Test_input_hash=got['input_hash'],exploratory=True)
 result['Test_accepted']=True
 verify();write(p/'result.json',result);write(p/'TEST_COMPLETE.json',dict(one_Test_pass=True,result_sha256=sha(p/'result.json'),checkpoint_sha256=checkpoint))
 logging.info('COMPLETE: fixed best-Dev checkpoint accepted once on Test')
if __name__=='__main__':
 q=argparse.ArgumentParser();q.add_argument('--dataset',choices=['beauty','ml-1m'],required=True);q.add_argument('--variant',choices=['G0','G1','G2','G3'],required=True);x=q.parse_args()
 p=OUT/'runs'/(x.dataset+'_'+x.variant)
 try:run(x.dataset,x.variant)
 except BaseException as e:
  write(p/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),no_retry=True));raise
