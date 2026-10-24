"""Explicit fresh-initialization joint training entry. Not run during development."""
import sys,argparse,time,json,logging,gc,os,atexit,importlib.util,hashlib,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import torch
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
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--graph',required=True)
 p.add_argument('--variant',choices=['B0','G1','G2','G3'],required=True);p.add_argument('--output-dir',required=True);p.add_argument('--gpu',type=int,default=0)
 x=p.parse_args();out=project_output(x.output_dir)
 # No checkpoint argument: all arms start from fresh same-seed C1.
 gpu_spec=importlib.util.spec_from_file_location('sgit_resource',ROOT/'research/elector_sil_objective_full_20261023_v2/scheduler.py')
 resource=importlib.util.module_from_spec(gpu_spec);gpu_spec.loader.exec_module(resource)
 inventory=resource.observe()
 if x.gpu not in inventory or not resource.idle(inventory[x.gpu]):raise RuntimeError("GPU occupied; wait in a safe queue after authorization")
 lease=resource.acquire(ROOT/('research/gpu_resource_scheduler_20261022_v1/gpu%d.lock'%x.gpu))
 if lease is None:raise RuntimeError("GPU lease occupied")
 atexit.register(os.close,lease)
 if not resource.idle(resource.observe()[x.gpu]):raise RuntimeError("GPU became occupied")
 # Bind the verified physical UUID, even if a calling queue set another
 # CUDA_VISIBLE_DEVICES mapping. All writable runtime caches stay in this run.
 out.mkdir();os.environ['CUDA_VISIBLE_DEVICES']=inventory[x.gpu]['uuid']
 os.environ['CUDA_CACHE_PATH']=str(out/'cuda_cache')
 os.environ['TORCHINDUCTOR_CACHE_DIR']=str(out/'torch_cache')
 (out/'cuda_cache').mkdir();torch.cuda.set_device(0)
 a=args_from_config(x.config);a.device=torch.device('cuda:0');a.sgit_variant=x.variant;a.sgit_graph_path=x.graph;a.test_epoch=-1
 c=train_corpus(a);utils.init_seed(a.random_seed);m=SGITRec(a,c).to(a.device)
 m.model_path=str(out/'best.pt');a.model_path=m.model_path
 logging.basicConfig(filename=out/'train.log',filemode='x',level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
 logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
 protocol.heldout(c,a,'dev');data={s:m.Dataset(m,c,s) for s in ('train','dev')}
 for d in data.values():d.prepare()
 cfg=dict(args=vars(a),physical_GPU=x.gpu,GPU_uuid=inventory[x.gpu]['uuid'],fresh_seed_initialization=True,C1_initial_sha256=core_hash(m),
  graph_sha256=sha(x.graph),config_source=x.config,parameters=m.count_variables(),selection='locked .30 prior Dev NDCG@5',
  loss='unchanged Filtered Softmax plus ASPCF relation',source_sha256={str(ROOT/z):sha(ROOT/z) for z in ['models/sequential/SGITRec.py','models/sequential/sgit_components.py','tools/run_sgitrec.py']})
 write(out/'config.json',cfg)
 class Runner(BaseRunner):
  def evaluate(self,dataset,topks,metrics):
   assert dataset.phase=='dev'
   got=protocol.score_views(dataset);self.latest=dict(epoch=int(m.cur_epoch),**got)
   with (out/'dev_metrics.jsonl').open('a') as f:f.write(json.dumps(self.latest)+'\n')
   return got['metrics']['locked_prior']
 runner=Runner(a);original=m.save_model
 def save(path=None):
  original(path)
  with (out/'best_metadata.json').open('w') as f:json.dump(runner.latest,f,indent=2)
 m.save_model=save
 start=time.perf_counter();runner.train(data);m.actions_after_train()
 cp=sha(out/'best.pt');write(out/'TRAIN_DEV_COMPLETE.json',dict(checkpoint_sha256=cp,seconds=time.perf_counter()-start,best=json.loads((out/'best_metadata.json').read_text())))
 del data;gc.collect();protocol.heldout(c,a,'test');td=m.Dataset(m,c,'test');td.prepare()
 write(out/'TEST_STARTED.json',dict(checkpoint_sha256=cp,passes=1))
 got=protocol.score_views(td,out/'test_user_ranks.csv');assert cp==sha(out/'best.pt')
 reference=ROOT/'research/gictrec_round1_20261024_v1'/(a.dataset+'_test_PROTOCOL.json')
 if reference.exists():assert got['input_hash']==json.loads(reference.read_text())['input_hash']
 result=dict(train_complete=True,Dev_best_selected=True,Test_accepted=True,Test_passes=1,
  best=json.loads((out/'best_metadata.json').read_text()),Test=got['metrics'],Test_input_hash=got['input_hash'],checkpoint_sha256=cp,config=cfg)
 with (out/'test_metrics.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['scheme',*protocol.KEYS]);w.writeheader()
  for scheme,metrics in got['metrics'].items():w.writerow(dict(scheme=scheme,**metrics))
 write(out/'result.json',result);write(out/'TEST_COMPLETE.json',dict(one_Test_pass=True,checkpoint_sha256=cp,result_sha256=sha(out/'result.json')))
if __name__=='__main__':main()
