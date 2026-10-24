"""Nonpreemptive dataset queue; each GPU hosts at most one project task."""
import os,json,time,fcntl,subprocess,csv,io,hashlib,traceback,argparse
from pathlib import Path
ROOT=Path('/home/yyx/hzgcode/pom2.0').resolve();OUT=Path(__file__).resolve().parent
RESOURCE=ROOT/'research/gpu_resource_scheduler_20261022_v1'
PYTHON='/home/yyx/anaconda3/envs/hzg_py10/bin/python'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(2**20),b''):h.update(v)
 return h.hexdigest()
def write(p,x):
 p=Path(p);assert p.parent.resolve().is_relative_to(ROOT)
 with p.open('x') as f:json.dump(x,f,indent=2)
def status(ds,name,**kw):
 x=dict(state=name,pid=os.getpid(),dataset=ds,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
 p=OUT/ds/'.STATE.tmp'
 with p.open('w') as f:json.dump(x,f,indent=2)
 os.replace(p,OUT/ds/'STATE.json');print(json.dumps(x),flush=True)
def acquire(p):
 fd=os.open(p,os.O_RDWR|os.O_CREAT,0o600)
 try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:os.close(fd);return None
 return fd
def observe():
 text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
 inv={int(r[0]):dict(uuid=r[1].strip(),free=int(r[2]),util=int(r[3]),pids=[]) for r in csv.reader(io.StringIO(text)) if r}
 text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True,timeout=15)
 for row in csv.reader(io.StringIO(text)):
  if row:
   if len(row)!=2 or not row[0].strip().isdigit():raise RuntimeError('Unrecognized GPU process listing')
   for info in inv.values():
    if info['uuid']==row[1].strip():info['pids'].append(int(row[0]))
 return inv
def idle(v):return not v['pids'] and v['free']>=10240 and v['util']<=10
def choose(ds,preferences):
 while True:
  inv=observe()
  for gpu in preferences:
   if gpu not in inv or not idle(inv[gpu]):continue
   fd=acquire(RESOURCE/('gpu%d.lock'%gpu))
   if fd is None:continue
   time.sleep(1);fresh=observe()
   if gpu in fresh and idle(fresh[gpu]):return gpu,fd,fresh[gpu]['uuid']
   os.close(fd)
  status(ds,'WAITING_GPU',preferences=preferences,poll_seconds=60,inventory=inv);time.sleep(60)
def verify():
 manifest=json.loads((OUT/'MANIFEST.json').read_text())
 for p,h in manifest['protected_sha256'].items():assert sha(ROOT/p)==h,p
 for p,h in json.loads((OUT/'SOURCE_LOCK.json').read_text()).items():assert sha(OUT/p)==h,p
 assert json.loads((OUT/'CHECK_PASS.json').read_text())['passed']
 return manifest

def main(ds):
 job=None;gpu_fd=None;child=None;gpu=None
 try:
  name='elector_objective_full_20261023_v1_'+ds
  job=acquire(RESOURCE/(name+'.lock'))
  if job is None:raise RuntimeError('Duplicate dataset queue lock')
  if (RESOURCE/(name+'.json')).exists():raise RuntimeError('Experiment already registered; no duplicate submission')
  manifest=verify()
  write(RESOURCE/(name+'.json'),dict(pid=os.getpid(),run_root=str(OUT/ds)))
  write(OUT/ds/'SCHEDULER_STARTED.json',dict(pid=os.getpid(),preferences=manifest['datasets'][ds]['GPU_preference'],fail_fast=True))
  status(ds,'STARTED',reused=['SS-B0','SS-C1'],order=manifest['train_order'])
  for arm in manifest['train_order']:
   verify();folder=OUT/ds/'arms'/arm
   if any(folder.iterdir()):raise RuntimeError('Existing outputs: refuse overwrite/retry '+arm)
   gpu,gpu_fd,uuid=choose(ds,manifest['datasets'][ds]['GPU_preference'])
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_CACHE_PATH=str(OUT/ds/'cuda_cache'),TORCHINDUCTOR_CACHE_DIR=str(OUT/ds/'torch_cache'))
   log=(folder/'stdout.log').open('x')
   child=subprocess.Popen(['nohup',PYTHON,'-u','-B',str(OUT/'train_task.py'),'--dataset',ds,'--arm',arm],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,close_fds=True,pass_fds=(job,gpu_fd),start_new_session=True)
   log.close();write(folder/'PID.json',dict(pid=child.pid))
   status(ds,'RUNNING',GPU=gpu,GPU_uuid=uuid,current_arm=arm,train_pid=child.pid,log=str(folder/'train.log'))
   code=child.wait();write(folder/'exit_status.json',dict(code=code,pid=child.pid))
   if code!=0:raise RuntimeError('Task failed '+arm+' exit='+str(code))
   result=json.loads((folder/'result.json').read_text())
   if not (result['train_complete'] and result['Dev_best_selected'] and result['Test_accepted'] and result['Test_passes']==1):raise RuntimeError('Task missing full Train/Dev/Test acceptance')
   child=None;os.close(gpu_fd);gpu_fd=None
  status(ds,'COMPLETED',new_tasks=4,reused_tasks=2)
 except BaseException as e:
  write(OUT/ds/'SCHEDULER_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),current_train_pid=child.pid if child else None,no_retry=True,following_tasks_halted=True))
  status(ds,'FAILED',error=repr(e),GPU=gpu);raise
 finally:
  if gpu_fd is not None:os.close(gpu_fd)
  if job is not None:os.close(job)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['beauty','ml-1m'],required=True);main(p.parse_args().dataset)
