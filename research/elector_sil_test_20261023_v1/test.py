"""One locked best-Dev Test pass per ELECTOR-SIL checkpoint, two score views."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import sys,json,time,hashlib,pickle,importlib.util,types,csv,fcntl,subprocess,io,traceback,gc
from pathlib import Path
ROOT=Path('/home/yyx/hzgcode/pom2.0').resolve();OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
KEYS=['HR@5','NDCG@5','HR@10','NDCG@10','HR@20','NDCG@20']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
 return h.hexdigest()
def write(p,value):
 p=Path(p);assert p.parent.resolve().is_relative_to(OUT)
 with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def state(name,**values):
 record=dict(state=name,pid=os.getpid(),time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**values)
 p=OUT/'.STATE.tmp'
 with p.open('w') as f:json.dump(record,f,indent=2)
 os.replace(p,OUT/'STATE.json');print(json.dumps(record),flush=True)
def lease(path):
 fd=os.open(path,os.O_RDWR|os.O_CREAT,0o600)
 try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:os.close(fd);return None
 return fd
def inventory():
 s=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
 d={int(r[0]):dict(uuid=r[1].strip(),free=int(r[2]),util=int(r[3]),pids=[]) for r in csv.reader(io.StringIO(s)) if r}
 s=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True,timeout=15)
 for r in csv.reader(io.StringIO(s)):
  if r:
   assert len(r)==2 and r[0].strip().isdigit()
   for info in d.values():
    if info['uuid']==r[1].strip():info['pids'].append(int(r[0]))
 return d
def idle(info):return not info['pids'] and info['free']>=10240 and info['util']<=10
def choose_gpu():
 while True:
  inv=inventory()
  for gpu in (0,1):
   if gpu not in inv or not idle(inv[gpu]):continue
   fd=lease(ROOT/'research/gpu_resource_scheduler_20261022_v1'/('gpu%d.lock'%gpu))
   if fd is None:continue
   time.sleep(1);again=inventory()
   if gpu in again and idle(again[gpu]):return gpu,fd,again[gpu]['uuid']
   os.close(fd)
  state('WAITING_GPU',inventory=inv,poll_seconds=60);time.sleep(60)
def module(path,tag):
 spec=importlib.util.spec_from_file_location(tag,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def evaluate(e,c,manifest):
 import numpy as np,torch
 from torch.utils.data import DataLoader
 from helpers.BaseRunner import BaseRunner
 from utils import utils
 p=OUT/'arms'/e['id'];start=time.perf_counter()
 assert not (p/'TEST_STARTED.json').exists() and not (p/'result.json').exists()
 assert sha(e['checkpoint'])==e['checkpoint_sha256']
 cfg=e['config'];src=Path(e['run_root'])/'model_snapshot.py'
 assert sha(src)==cfg['source_sha256']
 Model=module(src,'locked_test_'+e['id'].replace('-','_')).LLMMIRecASPCFAdaptiveSIL
 a=types.SimpleNamespace(**cfg['args']);a.device=torch.device('cuda:0');a.buffer=0
 a.model_path=e['checkpoint'];a.test_all=0
 utils.init_seed(42);m=Model(a,c)
 assert m.count_variables()==e['parameters']
 loaded=torch.load(e['checkpoint'],map_location='cpu',weights_only=True)
 m.load_state_dict(loaded,strict=True);del loaded
 asset=json.loads(Path(a.sil_asset_path).read_text())
 lock=json.loads(Path(e['historical_lock']).read_text())
 assert asset['lambda_pop']==lock['lambda_normalized']==.30
 assert m.sil_raw_pop_lambda==e['raw_prior_lambda']==lock['raw_lambda']
 counts=np.bincount(c.data_df['train'].item_id.to_numpy(np.int64),minlength=c.n_items)
 assert np.array_equal(m.sil_log_frequency.numpy(),np.log1p(counts).astype(np.float32))
 assert sha(a.llm_emb_path)==lock['embedding_sha256']
 m=m.to(a.device);m.eval();m.requires_grad_(False)
 data=Model.Dataset(m,c,'test')
 assert len(data)==manifest['history'][e['dataset']]['rows']
 assert all(len(x)==1000 for x in data.data['neg_items']) and len(np.unique(data.data['user_id']))==len(data)
 loader=DataLoader(data,batch_size=256,shuffle=False,num_workers=0,collate_fn=data.collate_batch,pin_memory=False)
 write(p/'TEST_STARTED.json',dict(id=e['id'],checkpoint_sha256=e['checkpoint_sha256'],best_epoch=e['best_dev']['epoch'],test_pass_budget=1,scores=['raw','locked_prior'],lambda_pop=.30))
 state('RUNNING_TEST',experiment=e['id'],best_epoch=e['best_dev']['epoch'])
 ranks={'raw':[],'locked_prior':[]};users=[];inputhash=hashlib.sha256();count=0;dup_gt=0;max_export_error=0.
 torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
 with torch.inference_mode():
  for batch in loader:
   for key in ['user_id','history_items','lengths','item_id']:
    ar=batch[key].numpy();inputhash.update(key.encode());inputhash.update(str(ar.shape).encode());inputhash.update(ar.tobytes())
   ids=batch['item_id'].numpy();assert ids.shape[1]==1001
   dup_gt+=int((ids[:,1:]==ids[:,:1]).any(1).sum());users.append(batch['user_id'].numpy())
   feed=utils.batch_to_gpu(batch,a.device);out=m(feed,return_intermediate=True)
   raw=out['raw_prediction'].cpu().numpy().astype(np.float64)
   pop=m.sil_log_frequency[feed['item_id']].cpu().numpy().astype(np.float64)
   corrected=raw+e['raw_prior_lambda']*(pop-pop.mean(1,keepdims=True))
   exported=out['prediction'].cpu().numpy()
   error=float(np.abs(exported-corrected).max());max_export_error=max(max_export_error,error)
   assert error<1e-10
   for scheme,scores in [('raw',raw),('locked_prior',corrected)]:
    assert np.isfinite(scores).all()
    rr=np.argwhere((-scores).argsort(axis=1)==0)[:,1]+1
    original=BaseRunner.evaluate_method(scores,[5,10,20],['HR','NDCG'])
    for key in KEYS:
     k=int(key.split('@')[1]);expected=((rr<=k)/(np.log2(rr+1) if key.startswith('NDCG') else 1)).mean()
     assert abs(float(original[key])-expected)<1e-12
    ranks[scheme].append(rr)
   if count==0:
    rr_export=np.argwhere((-exported).argsort(axis=1)==0)[:,1]+1
    assert np.array_equal(rr_export,ranks['locked_prior'][-1])
   count+=len(raw)
   del out,feed
 torch.cuda.synchronize()
 assert inputhash.hexdigest()==manifest['history'][e['dataset']]['input_hash'],'Test feeds differ from historical 1001-slot protocol'
 arrays={key:np.concatenate(value) for key,value in ranks.items()};metrics={}
 for scheme,rr in arrays.items():
  metrics[scheme]={key:float(((rr<=int(key.split('@')[1]))/(np.log2(rr+1) if key.startswith('NDCG') else 1)).mean()) for key in KEYS}
 assert sha(e['checkpoint'])==e['checkpoint_sha256']
 result=dict(id=e['id'],dataset=e['dataset'],K=e['K'],arm=e['arm'],rows=count,candidates=1001,duplicates_preserved=True,
             ground_truth_duplicate_users=dup_gt,Test_accepted=True,Test_passes=1,train_complete=True,Dev_best_selected=True,
             exploratory=True,metrics=metrics,best_dev=e['best_dev'],parameters=e['parameters'],training_seconds=e['training_seconds'],
             seconds=time.perf_counter()-start,peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20,checkpoint=e['checkpoint'],checkpoint_sha256=e['checkpoint_sha256'],
             full_config=cfg,lambda_pop=.30,raw_prior_lambda=e['raw_prior_lambda'],scale=e['locked_scale'],
             input_hash=inputhash.hexdigest(),max_exported_prior_score_difference=max_export_error,checkpoint_unchanged=True,model_training_updates=0)
 write(p/'result.json',result)
 with (p/'user_ranks.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['user_id','raw','locked_prior']);w.writerows(zip(np.concatenate(users),arrays['raw'],arrays['locked_prior']))
 write(p/'TEST_COMPLETE.json',dict(Test_accepted=True,one_Test_pass=True,result_sha256=sha(p/'result.json'),checkpoint_unchanged=True))
 print(json.dumps(dict(id=e['id'],metrics=metrics,seconds=result['seconds'])),flush=True)
 del loader,data,m,cfg;gc.collect();torch.cuda.empty_cache()
 return result

def main():
 job=lease(OUT/'job.lock');assert job is not None,'Test job already running'
 gpu_fd=None;start=time.perf_counter()
 try:
  assert not (OUT/'TEST_STARTED.json').exists(),'Refusing duplicate Test run'
  manifest=json.loads((OUT/'MANIFEST.json').read_text())
  assert sha(OUT/'test.py')==json.loads((OUT/'SCRIPT_LOCK.json').read_text())['script_sha256']
  for name,h in manifest['protected_sha256'].items():assert sha(ROOT/name)==h,name
  gpu,gpu_fd,uuid=choose_gpu();os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
  os.environ['CUDA_CACHE_PATH']=str(OUT/'cuda_cache');os.environ['TORCHINDUCTOR_CACHE_DIR']=str(OUT/'torch_cache')
  import numpy as np,torch
  torch.set_num_threads(4)
  assert torch.cuda.is_available() and torch.cuda.device_count()==1
  write(OUT/'TEST_STARTED.json',dict(pid=os.getpid(),GPU=gpu,GPU_uuid=uuid,manifest_sha256=sha(OUT/'MANIFEST.json'),script_sha256=sha(OUT/'test.py'),exploratory=True))
  corpora={};results=[]
  for e in manifest['experiments']:
   ds=e['dataset']
   if ds not in corpora:
    with (ROOT/'data'/ds/'SeqReader.pkl').open('rb') as f:c=pickle.load(f)
    tr=c.data_df['train'];dv=c.data_df['dev'];te=c.data_df['test']
    n=tr.groupby('user_id').size();nd=dv.groupby('user_id').size()
    expected=te.user_id.map(n).fillna(0)+te.user_id.map(nd).fillna(0)
    assert np.array_equal(te.position.to_numpy(),expected.to_numpy())
    assert (te.position>0).all() and np.stack(te.neg_items.to_numpy()).shape==(len(te),1000)
    corpora[ds]=c
   results.append(evaluate(e,corpora[ds],manifest))
  differences=[];historical=[]
  for ds,k in [('beauty',4),('ml-1m',2),('ml-1m',4),('ml-1m',8)]:
   group={x['arm']:x for x in results if x['dataset']==ds and x['K']==k}
   for scheme in ['raw','locked_prior']:
    for treatment,control in [('C1','B0'),('M1','B0'),('M1','C1')]:
     aa=group[treatment]['metrics'][scheme];bb=group[control]['metrics'][scheme]
     differences.append(dict(dataset=ds,K=k,scheme=scheme,comparison=treatment+'-'+control,
          absolute={key:aa[key]-bb[key] for key in KEYS},relative_pct={key:100*(aa[key]/bb[key]-1) for key in KEYS}))
  for row in results:
   ds=row['dataset']
   for scheme,old_scheme in [('raw','lambda0'),('locked_prior','lambda0.3')]:
    old=next(x for x in manifest['history'][ds]['metrics'] if x['scheme']==old_scheme)
    historical.append(dict(id=row['id'],scheme=scheme,historical_K=4,different_checkpoint=True,different_K=row['K']!=4,
            absolute={key:row['metrics'][scheme][key]-old[key] for key in KEYS},
            relative_pct={key:100*(row['metrics'][scheme][key]/old[key]-1) for key in KEYS}))
  for name,h in manifest['protected_sha256'].items():assert sha(ROOT/name)==h,name
  summary=dict(results=results,differences=differences,historical_comparisons=historical,history=manifest['history'],
               protected_files_unchanged=True,train_updates=0,exploratory=True,seconds=time.perf_counter()-start,GPU=gpu)
  write(OUT/'SUMMARY.json',summary)
  write(OUT/'TEST_COMPLETE.json',dict(Test_accepted=True,checkpoints=12,scores_per_checkpoint=2,summary_sha256=sha(OUT/'SUMMARY.json'),protected_files_unchanged=True,seconds=summary['seconds']))
  state('COMPLETED',checkpoints=12,seconds=summary['seconds'],GPU=gpu)
 except BaseException as exc:
  write(OUT/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),no_automatic_retry=True))
  state('FAILED',error=repr(exc));raise
 finally:
  if gpu_fd is not None:os.close(gpu_fd)
  os.close(job)
if __name__=='__main__':main()
