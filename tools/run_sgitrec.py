"""Explicit fresh-initialization joint training entry. Not run during development."""
import sys,argparse,time,json,logging,gc,os,atexit,importlib.util,hashlib,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import torch,fcntl
from models.sequential.SGITRec import SGITRec
from gict_data import args_from_config,train_corpus,project_output,sha
spec=importlib.util.spec_from_file_location('sgit_historical_protocol',ROOT/'research/gictrec_round1_20261024_v1/common.py')
protocol=importlib.util.module_from_spec(spec);spec.loader.exec_module(protocol)
from helpers.BaseRunner import BaseRunner
from utils import utils
def write(path,value):
 with Path(path).open('x') as f:json.dump(value,f,indent=2,default=str)
def core_hash(m):
 h=hashlib.sha256()
 for n,p in m.named_parameters():
  if not n.startswith('graph_memory.'):h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def inherit_or_acquire_lease(resource,gpu,inherited=None):
 lock=ROOT/('research/gpu_resource_scheduler_20261022_v1/gpu%d.lock'%gpu)
 if inherited is None:
  lease=resource.acquire(lock)
  if lease is None:raise RuntimeError('GPU lease occupied')
  return lease
 actual=os.fstat(inherited);expected=os.stat(lock)
 if (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):raise RuntimeError('Wrong inherited GPU lease')
 fcntl.flock(inherited,fcntl.LOCK_EX|fcntl.LOCK_NB)
 return inherited
def verify_manifest(path):
 p=Path(path).resolve();assert p.is_relative_to(ROOT)
 plan=json.loads(p.read_text())
 for key in ['source_sha256','protected_sha256']:
  for name,digest in plan[key].items():
   target=Path(name).resolve();assert target.is_relative_to(ROOT) and sha(target)==digest,name
 assert plan['authorized_tasks']==6 and plan['no_extra_tasks']
 return plan
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--graph',required=True)
 p.add_argument('--variant',choices=['B0','G1','G2','G3'],required=True);p.add_argument('--output-dir',required=True);p.add_argument('--gpu',type=int,default=0)
 p.add_argument('--manifest');p.add_argument('--gpu-lease-fd',type=int)
 x=p.parse_args();out=project_output(x.output_dir)
 plan=verify_manifest(x.manifest) if x.manifest else None
 # No checkpoint argument: all arms start from fresh same-seed C1.
 gpu_spec=importlib.util.spec_from_file_location('sgit_resource',ROOT/'research/elector_sil_objective_full_20261023_v2/scheduler.py')
 resource=importlib.util.module_from_spec(gpu_spec);gpu_spec.loader.exec_module(resource)
 inventory=resource.observe()
 if x.gpu not in inventory or not resource.idle(inventory[x.gpu]):raise RuntimeError("GPU occupied; wait in a safe queue after authorization")
 lease=inherit_or_acquire_lease(resource,x.gpu,x.gpu_lease_fd)
 atexit.register(os.close,lease)
 if not resource.idle(resource.observe()[x.gpu]):raise RuntimeError("GPU became occupied")
 # Bind the verified physical UUID, even if a calling queue set another
 # CUDA_VISIBLE_DEVICES mapping. All writable runtime caches stay in this run.
 out.mkdir();os.environ['CUDA_VISIBLE_DEVICES']=inventory[x.gpu]['uuid']
 os.environ['CUDA_CACHE_PATH']=str(out/'cuda_cache')
 os.environ['TORCHINDUCTOR_CACHE_DIR']=str(out/'torch_cache')
 (out/'cuda_cache').mkdir();torch.cuda.set_device(0)
 logging.basicConfig(filename=out/'train.log',filemode='x',level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',force=True)
 logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
 logging.info('INITIALIZING %s GPU%d; new C1 seed42, no checkpoint continuation',x.variant,x.gpu)
 a=args_from_config(x.config);a.device=torch.device('cuda:0');a.sgit_variant=x.variant;a.sgit_graph_path=x.graph;a.test_epoch=-1
 c=train_corpus(a);utils.init_seed(a.random_seed);m=SGITRec(a,c).to(a.device)
 if plan:
  ds=plan['datasets'][a.dataset]
  assert x.variant in ds['order'] and a.K==ds['K'] and a.random_seed==42
  assert core_hash(m)==ds['C1_initial_sha256'],'Common initialization mismatch'
  assert sha(x.config)==ds['config_sha256'] and sha(x.graph)==ds['graph_sha256']
 assert all(v.requires_grad for v in m.parameters()),'Unexpected frozen trainable parameters'
 assert a.metric.split(',')[0]=='NDCG' and a.topk.split(',')[0]=='5' and a.test_epoch==-1
 m.model_path=str(out/'best.pt');a.model_path=m.model_path
 protocol.heldout(c,a,'dev');data={s:m.Dataset(m,c,s) for s in ('train','dev')}
 for d in data.values():d.prepare()
 cfg=dict(args=vars(a),head=plan['HEAD'] if plan else None,physical_GPU=x.gpu,GPU_uuid=inventory[x.gpu]['uuid'],fresh_seed_initialization=True,C1_initial_sha256=core_hash(m),
  graph_sha256=sha(x.graph),config_source=x.config,manifest_sha256=sha(x.manifest) if x.manifest else None,parameters=m.count_variables(),selection='locked .30 prior Dev NDCG@5',
  loss='unchanged Filtered Softmax plus ASPCF relation',source_sha256=plan['source_sha256'] if plan else {str(ROOT/z):sha(ROOT/z) for z in ['models/sequential/SGITRec.py','models/sequential/sgit_components.py','tools/run_sgitrec.py','tools/sgit_graph.py','tools/gict_data.py','research/gictrec_round1_20261024_v1/common.py','research/elector_sil_objective_full_20261023_v2/scheduler.py']},
  torch_version=str(torch.__version__),numpy_version=protocol.np.__version__,python_version=sys.version)
 write(out/'config.json',cfg)
 class Runner(BaseRunner):
  def fit(self,dataset,epoch=-1):
   self.encoding_counts=dict(batches=0,unique_neighbors=0,newly_encoded_items=0,maximum_unique_neighbors=0,history_slots_queried=0,neighbor_slots_read=0)
   torch.cuda.reset_peak_memory_stats(m.device);tick=time.perf_counter()
   value=super().fit(dataset,epoch)
   torch.cuda.synchronize(m.device)
   costs=dict(epoch=epoch,training_seconds=time.perf_counter()-tick,peak_allocated_MiB=torch.cuda.max_memory_allocated(m.device)/2**20,**self.encoding_counts)
   costs['mean_unique_neighbors']=costs['unique_neighbors']/max(costs['batches'],1)
   costs['mean_newly_encoded_items']=costs['newly_encoded_items']/max(costs['batches'],1)
   self.last_cost=costs
   with (out/'training_cost.jsonl').open('a') as f:f.write(json.dumps(costs)+'\n')
   return value
  def evaluate(self,dataset,topks,metrics):
   assert dataset.phase=='dev'
   got=protocol.score_views(dataset)
   reference=ROOT/'research/gictrec_round1_20261024_v1'/(a.dataset+'_dev_PROTOCOL.json')
   assert reference.is_file() and got['input_hash']==json.loads(reference.read_text())['input_hash']
   self.latest=dict(epoch=int(m.cur_epoch),training_cost=self.last_cost,**got)
   with (out/'dev_metrics.jsonl').open('a') as f:f.write(json.dumps(self.latest)+'\n')
   return got['metrics']['locked_prior']
 runner=Runner(a)
 def encoding_hook(module,inputs,output):
  if not module.training:return
  cost=getattr(module,'last_graph_encoding_cost',dict(unique_neighbor_items=0,newly_encoded_items=0))
  counts=runner.encoding_counts;counts['batches']+=1
  counts['unique_neighbors']+=cost['unique_neighbor_items'];counts['newly_encoded_items']+=cost['newly_encoded_items']
  counts['history_slots_queried']+=cost.get('history_slots',0);counts['neighbor_slots_read']+=cost.get('neighbor_slots',0)
  counts['maximum_unique_neighbors']=max(counts['maximum_unique_neighbors'],cost['unique_neighbor_items'])
 m.register_forward_hook(encoding_hook)
 original=m.save_model
 def save(path=None):
  original(path)
  with (out/'best_metadata.json').open('w') as f:json.dump(runner.latest,f,indent=2)
 m.save_model=save
 write(out/'READY.json',dict(pid=os.getpid(),physical_GPU=x.gpu,GPU_uuid=inventory[x.gpu]['uuid'],variant=x.variant,dataset=a.dataset,
  parameters=m.count_variables(),C1_initial_sha256=core_hash(m),train_rows=len(data['train']),Dev_rows=len(data['dev'])))
 logging.info('TRAIN_START %s %s K%d seed%d lr=%g epoch_limit=%d early_stop=%d prior=.30',a.dataset,x.variant,a.K,a.random_seed,a.lr,a.epoch,a.early_stop)
 start=time.perf_counter();runner.train(data);m.actions_after_train();training_seconds=time.perf_counter()-start
 if plan:verify_manifest(x.manifest)
 cp=sha(out/'best.pt');write(out/'TRAIN_DEV_COMPLETE.json',dict(checkpoint_sha256=cp,seconds=training_seconds,best=json.loads((out/'best_metadata.json').read_text())))
 del data;gc.collect();protocol.heldout(c,a,'test');td=m.Dataset(m,c,'test');td.prepare()
 write(out/'TEST_STARTED.json',dict(checkpoint_sha256=cp,passes=1))
 test_start=time.perf_counter();got=protocol.score_views(td,out/'test_user_ranks.csv');test_seconds=time.perf_counter()-test_start;assert cp==sha(out/'best.pt')
 reference=ROOT/'research/gictrec_round1_20261024_v1'/(a.dataset+'_test_PROTOCOL.json')
 if reference.exists():assert got['input_hash']==json.loads(reference.read_text())['input_hash']
 result=dict(train_complete=True,Dev_best_selected=True,Test_accepted=True,Test_passes=1,training_seconds=training_seconds,Test_seconds=test_seconds,epochs=runner.last_cost['epoch'],parameters=m.count_variables(),exploratory=True,
  best=json.loads((out/'best_metadata.json').read_text()),Test=got['metrics'],Test_input_hash=got['input_hash'],checkpoint_sha256=cp,config=cfg)
 with (out/'test_metrics.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['scheme',*protocol.KEYS]);w.writeheader()
  for scheme,metrics in got['metrics'].items():w.writerow(dict(scheme=scheme,**metrics))
 if plan:verify_manifest(x.manifest)
 write(out/'result.json',result);write(out/'TEST_COMPLETE.json',dict(one_Test_pass=True,checkpoint_sha256=cp,result_sha256=sha(out/'result.json')))
if __name__=='__main__':main()
