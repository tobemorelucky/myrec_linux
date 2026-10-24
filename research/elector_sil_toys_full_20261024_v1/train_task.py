"""Only four authorized complete Toys runs; Dev selection followed by one Test."""
import time,logging,argparse,traceback
from common import *
class DevRunner(BaseRunner):
 def __init__(self,a,p,arm):super().__init__(a);self.path=p;self.arm=arm;self.last_dev=None
 def evaluate(self,dataset,topks,metrics):
  assert dataset.phase=='dev';got=score_views(dataset)
  scheme='raw' if self.arm=='BPR-B0' else 'locked_prior'
  self.last_dev=dict(epoch=int(dataset.model.cur_epoch),metrics=got['metrics'][scheme],raw=got['metrics']['raw'],locked_prior=got['metrics'].get('locked_prior'),selection=scheme+' Dev NDCG@5',checkpoint='best.pt')
  with (self.path/'dev_metrics.jsonl').open('a') as f:f.write(json.dumps(self.last_dev)+'\n')
  return self.last_dev['metrics']
 def fit(self,dataset,epoch=-1):
  value=super().fit(dataset,epoch)
  if not np.isfinite(value):raise FloatingPointError('Nonfinite epoch loss; no automatic retry')
  return value

def run(arm):
 p=OUT/'arms'/arm;assert arm in ORDER and p.is_dir() and not (p/'train.log').exists() and not (p/'best.pt').exists()
 logging.basicConfig(filename=p/'train.log',filemode='x',level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',force=True)
 logging.getLogger().addHandler(logging.StreamHandler(sys.stdout));start=time.perf_counter()
 verify();checks=read(OUT/'CHECK_PASS.json');assert checks['passed']
 if arm!='BPR-B0':assert read(OUT/'POP_PRIOR_LOCK.json')['lambda_normalized']==.30
 c=read_corpus();m,a=model(arm,c,'cuda:0');init=common_hash(m)
 assert init==checks['common_initial_sha256']
 assert m.count_variables()==checks['parameters'][arm]
 m=m.to(a.device);data={phase:m.Dataset(m,c,phase) for phase in ('train','dev')}
 for d in data.values():d.prepare()
 cfg=dict(dataset='toys',arm=arm,K=4,seed=42,args=vars(a),head=read(OUT/'PLAN.json')['head'],parameters=m.count_variables(),common_initial_parameter_sha256=init,physical_GPU=os.environ['CUDA_VISIBLE_DEVICES'],initialization='fresh seed42 common ASPCF core; no checkpoint initialization',selection='raw Dev NDCG@5' if arm=='BPR-B0' else 'locked-prior Dev NDCG@5',lambda_pop=.30,assets=read(OUT/'MANIFEST.json')['assets'],source_sha256=read(OUT/'SOURCE_LOCK.json'))
 write(p/'config.json',cfg);runner=DevRunner(a,p,arm);original_save=m.save_model
 def save(path=None):original_save(path);write(p/'best_metadata.json',runner.last_dev,False)
 def load(path=None):m.load_state_dict(torch.load(path or m.model_path,map_location=m.device,weights_only=True),strict=True)
 m.save_model=save;m.load_model=load
 write(p/'READY.json',dict(pid=os.getpid(),GPU=os.environ['CUDA_VISIBLE_DEVICES'],arm=arm,Train_rows=len(data['train']),Dev_rows=len(data['dev']),parameters=m.count_variables(),GPU_allocated_MiB=torch.cuda.memory_allocated()/2**20,common_initial_sha256=init))
 logging.info('TRAIN_START %s; Toys K4 seed42 lr=.001 B1024, original loss, full Dev early stopping',arm)
 runner.train(data);m.actions_after_train();training_seconds=time.perf_counter()-start
 best=read(p/'best_metadata.json');load();checkpoint_sha=sha(p/'best.pt')
 write(p/'train_result.json',dict(train_complete=True,Dev_best_selected=True,Test_accepted=False,best=best,checkpoint_sha256=checkpoint_sha,training_seconds=training_seconds,config=cfg))
 if arm=='BPR-B0':
  lock=lock_prior(m,c,checkpoint_sha);logging.info('Train-only reference prior LOCKED, scale=%s before first Test',lock['scale_factor'])
 else:
  lock=read(OUT/'POP_PRIOR_LOCK.json');assert sha(OUT/'TRAIN_ASSET.json')==read(p/'PRIOR_INPUT.json')['train_asset_sha256']
 # No Test dataset exists until best weights and the shared prior have been fixed.
 del data,c;gc.collect()
 tc=read_corpus('test');td=m.Dataset(m,tc,'test');td.prepare();load()
 assert sha(p/'best.pt')==checkpoint_sha
 write(p/'TEST_STARTED.json',dict(best_epoch=best['epoch'],checkpoint_sha256=checkpoint_sha,passes=1,lambda_pop=.30,prior_sha256=sha(OUT/'POP_PRIOR_LOCK.json')))
 t=time.perf_counter();got=score_views(td,p/'test_user_ranks.csv');Test_seconds=time.perf_counter()-t
 proto=OUT/'TEST_PROTOCOL_LOCK.json'
 if arm=='BPR-B0':write(proto,dict(input_hash=got['input_hash'],rows=got['rows'],candidates=1001,duplicate_slots_preserved=True,evaluator='original NumPy row-wise argsort; batchwise exact ranking and weighted metric accumulation',first_arm=arm))
 else:assert read(proto)['input_hash']==got['input_hash'],'Test candidate/history inputs changed'
 assert sha(p/'best.pt')==checkpoint_sha
 with (p/'test_metrics.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['scheme',*KEYS]);w.writeheader()
  for s,values in got['metrics'].items():w.writerow(dict(scheme=s,**values))
 result=dict(dataset='toys',arm=arm,K=4,seed=42,train_complete=True,Dev_best_selected=True,Test_accepted=True,Test_passes=1,exploratory=True,best=best,Test=got['metrics'],parameters=m.count_variables(),training_seconds=training_seconds,Test_seconds=Test_seconds,seconds=time.perf_counter()-start,checkpoint_sha256=checkpoint_sha,Test_input_hash=got['input_hash'],lambda_pop=.30,prior_scale=lock,config=cfg)
 verify();write(p/'result.json',result);write(p/'TEST_COMPLETE.json',dict(Test_accepted=True,one_Test_pass=True,checkpoint_sha256=checkpoint_sha,result_sha256=sha(p/'result.json')))
 logging.info('TASK_FINISHED %s; best Dev fixed; one Test accepted',arm)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=ORDER,required=True);arm=parser.parse_args().arm
 try:
  if arm!='BPR-B0':write(OUT/'arms'/arm/'PRIOR_INPUT.json',dict(train_asset_sha256=sha(OUT/'TRAIN_ASSET.json'),prior_sha256=sha(OUT/'POP_PRIOR_LOCK.json')))
  run(arm)
 except BaseException as e:
  write(OUT/'arms'/arm/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),no_retry=True));raise
