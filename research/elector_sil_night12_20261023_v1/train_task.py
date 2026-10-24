"""Fresh end-to-end training, existing Dev rule, then one fixed-weight Test."""
import os,sys,json,time,pickle,hashlib,logging,argparse,types,traceback,csv
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
ROOT=Path('/home/yyx/hzgcode/pom2.0').resolve();OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OUT))
from objective_model import ObjectiveSIL,base,ASPCF
from helpers.BaseRunner import BaseRunner
from utils import utils
from plan import folder,task_id,expected_parameters,check_initialization,archived
KEYS=['HR@5','NDCG@5','HR@10','NDCG@10','HR@20','NDCG@20']
def sha(p):return base.file_sha(p)
def write(p,data,exclusive=True):
 p=Path(p);assert p.parent.resolve().is_relative_to(OUT)
 with p.open('x' if exclusive else 'w') as f:json.dump(data,f,indent=2,allow_nan=False,default=str)
def verify():
 m=json.loads((OUT/'MANIFEST.json').read_text())
 for p,h in m['protected_sha256'].items():assert sha(ROOT/p)==h,p
 for p,h in json.loads((OUT/'SOURCE_LOCK.json').read_text()).items():assert sha(OUT/p)==h,p
 return m

def read_corpus(ds,Test=False):
 with (ROOT/'data'/ds/'SeqReader.pkl').open('rb') as f:c=pickle.load(f)
 counts=c.data_df['train'].groupby('user_id').size()
 if Test:
  nd=c.data_df['dev'].groupby('user_id').size();t=c.data_df['test']
  assert np.array_equal(t.position.to_numpy(),(t.user_id.map(counts).fillna(0)+t.user_id.map(nd).fillna(0)).to_numpy())
  c.data_df={p:c.data_df[p] for p in ('train','test')}
 else:
  d=c.data_df['dev'];assert np.array_equal(d.position.to_numpy(),d.user_id.map(counts).to_numpy())
  c.data_df={p:c.data_df[p] for p in ('train','dev')}
  for u in c.user_his:c.user_his[u]=c.user_his[u][:int(counts.get(u,0))]
 phase='test' if Test else 'dev';assert np.stack(c.data_df[phase].neg_items.to_numpy()).shape==(len(c.data_df[phase]),1000)
 return c

def arguments(ds,arm,K,seed,device='cuda:0'):
 manifest=json.loads((OUT/'MANIFEST.json').read_text());a=types.SimpleNamespace(**manifest['datasets'][ds]['args'])
 a.K=K;a.random_seed=seed;a.objective='SS';a.sil_variant='C1' if arm in ('SS-A1','SS-A2') else arm.split('-')[1]
 a.sil_use_local=int(arm!='SS-A1');a.sil_use_behavior=int(arm!='SS-A2');a.sil_use_recent=1;a.sil_alpha_initial=.1
 a.sil_mode='baseline' if arm=='SS-B0' else 'fixed';a.device=torch.device(device)
 a.model_path=str(folder(ds,K,seed,arm)/'best.pt');a.dev_only=1;a.load=0;a.regenerate=0;a.buffer=0;a.test_all=0
 return a

def common_hash(m):
 h=hashlib.sha256()
 for name,p in m.named_parameters():
  if not name.startswith('sil_block.'):
   h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

@torch.inference_mode()
def score_views(dataset):
 m=dataset.model;m.eval();raw_parts=[];prior_parts=[];users=[];inputhash=hashlib.sha256();export_error=0.
 loader=DataLoader(dataset,batch_size=256,shuffle=False,num_workers=0,collate_fn=dataset.collate_batch,pin_memory=False)
 for batch in loader:
  for name in ['user_id','history_items','lengths','item_id']:
   ar=batch[name].numpy();inputhash.update(name.encode());inputhash.update(str(ar.shape).encode());inputhash.update(ar.tobytes())
  feed=utils.batch_to_gpu(batch,m.device);out=m(feed,True)
  raw=out['raw_prediction'].cpu().numpy().astype(np.float64)
  prior=out['prediction'].cpu().numpy()
  if dataset.phase=='test':
   frequency=m.sil_log_frequency[feed['item_id']].cpu().numpy().astype(np.float64)
   corrected=raw+m.sil_raw_pop_lambda*(frequency-frequency.mean(1,keepdims=True))
   export_error=max(export_error,float(np.abs(prior-corrected).max()));assert export_error<1e-10
   prior=corrected
  assert raw.shape[1]==1001 and np.isfinite(raw).all() and np.isfinite(prior).all()
  raw_parts.append(raw);prior_parts.append(prior);users.append(batch['user_id'].cpu().numpy())
 raw=np.concatenate(raw_parts);prior=np.concatenate(prior_parts)
 metrics={scheme:{k:float(v) for k,v in BaseRunner.evaluate_method(scores,[5,10,20],['HR','NDCG']).items()} for scheme,scores in [('raw',raw),('locked_prior',prior)]}
 ranks={scheme:np.argwhere((-scores).argsort(1)==0)[:,1]+1 for scheme,scores in [('raw',raw),('locked_prior',prior)]} if dataset.phase=='test' else None
 return dict(metrics=metrics,input_hash=inputhash.hexdigest(),ranks=ranks,users=np.concatenate(users),export_error=export_error)

class DevRunner(BaseRunner):
 def __init__(self,args,folder):super().__init__(args);self.folder=folder;self.last_dev=None
 def evaluate(self,dataset,topks,metrics):
  assert dataset.phase=='dev';got=score_views(dataset)
  self.last_dev=dict(epoch=int(dataset.model.cur_epoch),metrics=got['metrics']['locked_prior'],raw=got['metrics']['raw'],locked_prior=got['metrics']['locked_prior'],selection='locked_prior NDCG@5 unchanged from reusable SS runs')
  with (self.folder/'dev_metrics.jsonl').open('a') as f:f.write(json.dumps(self.last_dev)+'\n')
  return got['metrics']['locked_prior']
 def fit(self,dataset,epoch=-1):
  assert dataset.phase=='train';value=super().fit(dataset,epoch)
  if not np.isfinite(value):raise FloatingPointError('Nonfinite full epoch loss; halt without retry')
  return value

def run(ds,arm,K,seed):
 task_id(ds,K,seed,arm)
 assert archived(ds,K,seed,arm) is None,'Trusted matching result must be reused, never retrained'
 folder=globals()['folder'](ds,K,seed,arm)
 assert folder.is_dir() and not (folder/'train.log').exists() and not (folder/'best.pt').exists()
 logging.basicConfig(filename=folder/'train.log',filemode='x',level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',force=True)
 logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
 start=time.perf_counter();manifest=verify();assert json.loads((OUT/'CHECK_PASS.json').read_text())['passed']
 c=read_corpus(ds);a=arguments(ds,arm,K,seed);utils.init_seed(seed);m=ObjectiveSIL(a,c)
 init_hash=common_hash(m);check_initialization(ds,K,seed,arm,init_hash)
 expected=expected_parameters(ds,K,arm)
 assert m.count_variables()==expected
 m=m.to(a.device);data={phase:ObjectiveSIL.Dataset(m,c,phase) for phase in ('train','dev')}
 for d in data.values():d.prepare()
 config=dict(dataset=ds,arm=arm,K=K,seed=seed,asset_sha256=manifest['assets'][ds],args=vars(a),head=manifest['head'],source_sha256=sha(OUT/'objective_model.py'),softguide_sha256=sha(OUT/'softguide_snapshot.py'),common_initial_parameter_sha256=init_hash,parameters=expected,physical_GPU=os.environ['CUDA_VISIBLE_DEVICES'],initialization='fresh seed%d, no input checkpoint'%seed,selection='locked-prior Dev NDCG@5 common to all six',primary_comparison='raw Test',prior_scale='same immutable per-dataset reference Train-RMS as archived SS, lambda .30')
 write(folder/'config.json',config)
 runner=DevRunner(a,folder);original_save=m.save_model
 def tracked_save(path=None):
  original_save(path);write(folder/'best_metadata.json',dict(**runner.last_dev,checkpoint='best.pt'),False)
 def fixed_load(path=None):
  p=path or m.model_path;m.load_state_dict(torch.load(p,map_location=m.device,weights_only=True),strict=True)
  logging.info('Load fixed best model from %s',p)
 m.save_model=tracked_save;m.load_model=fixed_load
 write(folder/'READY.json',dict(pid=os.getpid(),GPU=os.environ['CUDA_VISIBLE_DEVICES'],model=arm,dataset=ds,K=K,seed=seed,Train_rows=len(data['train']),Dev_rows=len(data['dev']),parameters=expected,common_initial_sha256=init_hash))
 write(folder/'RUNNING.json',dict(pid=os.getpid(),train_complete=False,Dev_best_selected=False,Test_accepted=False))
 logging.info('TRAIN_START %s %s K=%s seed=%s; original %s loss; fresh initialization',ds,arm,a.K,seed,a.objective)
 runner.train(data);m.actions_after_train();training_seconds=time.perf_counter()-start
 best=json.loads((folder/'best_metadata.json').read_text());checkpoint_sha=sha(folder/'best.pt')
 write(folder/'train_result.json',dict(train_complete=True,Dev_best_selected=True,Test_accepted=False,best=best,checkpoint_sha256=checkpoint_sha,training_seconds=training_seconds,config=config))
 # The only Test Dataset is constructed after best-Dev is fixed and reloaded.
 tc=read_corpus(ds,True);td=ObjectiveSIL.Dataset(m,tc,'test');td.prepare();fixed_load()
 assert sha(folder/'best.pt')==checkpoint_sha
 write(folder/'TEST_STARTED.json',dict(checkpoint_sha256=checkpoint_sha,best_epoch=best['epoch'],passes=1,lambda_pop=.30,exploratory=True))
 t=time.perf_counter();got=score_views(td);Test_seconds=time.perf_counter()-t
 prior_test=json.loads((ROOT/'research/elector_sil_test_20261023_v1/SUMMARY.json').read_text())['history'][ds]
 assert got['input_hash']==prior_test['input_hash'],'Test 1001-slot/history inputs differ from historical evaluator'
 assert sha(folder/'best.pt')==checkpoint_sha
 with (folder/'test_user_ranks.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['user_id','raw','locked_prior']);w.writerows(zip(got['users'],got['ranks']['raw'],got['ranks']['locked_prior']))
 with (folder/'test_metrics.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['scheme',*KEYS]);w.writeheader()
  for scheme,values in got['metrics'].items():w.writerow(dict(scheme=scheme,**values))
 result=dict(dataset=ds,arm=arm,K=K,seed=seed,train_complete=True,Dev_best_selected=True,Test_accepted=True,Test_passes=1,exploratory=True,best=best,Test=got['metrics'],parameters=expected,training_seconds=training_seconds,Test_seconds=Test_seconds,seconds=time.perf_counter()-start,checkpoint_sha256=checkpoint_sha,config=config,Test_input_hash=got['input_hash'],lambda_pop=.30,raw_prior_lambda=m.sil_raw_pop_lambda)
 verify();write(folder/'result.json',result);write(folder/'TEST_COMPLETE.json',dict(Test_accepted=True,one_Test_pass=True,checkpoint_sha256=checkpoint_sha,result_sha256=sha(folder/'result.json')))
 logging.info('TASK_FINISHED %s %s; best fixed by Dev; one Test complete',ds,arm)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['beauty','ml-1m'],required=True);p.add_argument('--arm',choices=['SS-B0','SS-C0','SS-C1','SS-A1','SS-A2'],required=True);p.add_argument('--K',type=int,choices=[2,3,4,8],required=True);p.add_argument('--seed',type=int,choices=[0,42],required=True);arg=p.parse_args()
 try:run(arg.dataset,arg.arm,arg.K,arg.seed)
 except BaseException as exc:
  write(folder(arg.dataset,arg.K,arg.seed,arg.arm)/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),no_retry=True));raise
