"""Two independent fail-fast dataset queues sharing nonpreemptive GPU leases."""
import argparse,fcntl,importlib.util,json,os,subprocess,sys,time,traceback
from pathlib import Path
from plan import *
RESOURCE=ROOT/'research/gpu_resource_scheduler_20261022_v1'
PYTHON='/home/yyx/anaconda3/envs/hzg_py10/bin/python'
spec=importlib.util.spec_from_file_location('existing_resource_functions',ROOT/'research/elector_sil_objective_full_20261023_v2/scheduler.py')
resource=importlib.util.module_from_spec(spec);spec.loader.exec_module(resource)
def alive(pid):
 try:
  text=Path('/proc/%d/stat'%int(pid)).read_text()
  return text[text.rfind(')')+2:].split()[0] not in ('Z','X')
 except FileNotFoundError:return False
def beauty_has_priority(inv):
 path=OUT/'queues/beauty/STATE.json'
 if not path.exists():return True
 s=read(path)
 if s.get('phase')!='K_search' or s['state'] in ('FAILED','COMPLETED'):return False
 return not (s.get('train_pid') and alive(s['train_pid'])) and sum(resource.idle(v) for v in inv.values())<2
def acquire_gpu(ds):
 dispatch=resource.acquire(RESOURCE/'elector_formal_20261023_dispatch.lock')
 if dispatch is None:return None,None
 try:
  inv=resource.observe()
  if ds=='ml-1m' and beauty_has_priority(inv):return None,None
  for gpu in (0,1):
   if gpu not in inv or not resource.idle(inv[gpu]):continue
   fd=resource.acquire(RESOURCE/('gpu%d.lock'%gpu))
   if fd is None:continue
   time.sleep(1);fresh=resource.observe()
   if gpu in fresh and resource.idle(fresh[gpu]):return gpu,fd
   os.close(fd)
  return None,None
 finally:os.close(dispatch)
def initial_tasks(ds):
 if ds=='beauty':
  return [(K,42,arm,'K_search') for K in (2,8) for arm in MAIN]
 return [(2,seed,arm,'six_seed') for seed in (0,1,2,3,41) for arm in MAIN]+[(2,42,arm,'ablation') for arm in ('SS-A1','SS-A2')]
def beauty_remaining(K):
 return [(K,seed,arm,'six_seed') for seed in (0,1,2,3,41) for arm in MAIN]+[(K,42,arm,'ablation') for arm in ('SS-A1','SS-A2')]
def main(ds):
 q=OUT/'queues'/ds
 queue_fd=resource.acquire(RESOURCE/('elector_formal_20261023_'+ds+'.lock'))
 assert queue_fd is not None,'Duplicate dataset scheduler'
 child=None;gpu_fd=None;task_fd=None;phase='K_search' if ds=='beauty' else 'six_seed'
 tasks=initial_tasks(ds);extended=ds!='beauty';index=0;current=None;last=None
 def status(name,**extra):
  nonlocal last
  value=dict(state=name,pid=os.getpid(),dataset=ds,phase=phase,current_task=current,
    train_pid=child.pid if child else None,completed_new_tasks=index,pending_known_tasks=len(tasks)-index,
    time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra)
  signature=json.dumps({k:v for k,v in value.items() if k!='time_utc'},sort_keys=True)
  if signature!=last:
   tmp=q/'STATE.tmp'
   with tmp.open('w') as f:json.dump(value,f,indent=2)
   os.replace(tmp,q/'STATE.json');print(json.dumps(value),flush=True);last=signature
 try:
  verify();assert read(OUT/'CHECK_PASS.json')['passed']
  write(q/'SCHEDULER_STARTED.json',dict(pid=os.getpid(),dataset=ds,first_tasks=tasks,fail_fast=True,
        no_preemption=True,GPUs=[0,1],Beauty_K_only_Dev=True))
  status('STARTED')
  while True:
   if index==len(tasks):
    if ds=='beauty' and not extended:
     K=select_beauty_K();tasks.extend(beauty_remaining(K));extended=True
    else:break
   K,seed,arm,phase=tasks[index];current=task_id(ds,K,seed,arm)
   assert archived(ds,K,seed,arm) is None,'Unexpected duplicate of a trusted archived task'
   while True:
    gpu,gpu_fd=acquire_gpu(ds)
    if gpu is not None:break
    status('WAITING_GPU',poll_seconds=15);time.sleep(15)
   verify()
   path=folder(ds,K,seed,arm)
   path.mkdir(exist_ok=False)
   task_fd=resource.acquire(RESOURCE/('elector_formal_20261023_'+current+'.lock'))
   assert task_fd is not None,'Duplicate task lock'
   write(path/'CLAIM.json',dict(scheduler_pid=os.getpid(),GPU=gpu,phase=phase))
   env=os.environ.copy()
   env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
     CUDA_CACHE_PATH=str(q/'cuda_cache'),TORCHINDUCTOR_CACHE_DIR=str(q/'torch_cache'))
   with (path/'stdout.log').open('x') as log:
    child=subprocess.Popen(['nohup',PYTHON,'-u','-B',str(OUT/'train_task.py'),'--dataset',ds,
      '--arm',arm,'--K',str(K),'--seed',str(seed)],cwd=ROOT,stdin=subprocess.DEVNULL,
      stdout=log,stderr=subprocess.STDOUT,env=env,close_fds=True,pass_fds=(queue_fd,gpu_fd,task_fd),start_new_session=True)
   write(path/'PID.json',dict(pid=child.pid))
   status('RUNNING',GPU=gpu,log=str(path/'train.log'))
   code=child.wait()
   write(path/'exit_status.json',dict(code=code,pid=child.pid))
   assert code==0,'Task failed '+current+' exit='+str(code)
   acceptance(path)
   child=None;os.close(gpu_fd);gpu_fd=None;os.close(task_fd);task_fd=None
   index+=1
  status('COMPLETED',ready_for_user_requested_summary=True,no_automatic_cross_model_summary=True)
 except BaseException as exc:
  write(q/'SCHEDULER_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),
      current_task=current,active_train_pid=child.pid if child else None,following_tasks_halted=True,no_retry=True))
  status('FAILED',error=repr(exc));raise
 finally:
  if gpu_fd is not None:os.close(gpu_fd)
  if task_fd is not None:os.close(task_fd)
  os.close(queue_fd)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['beauty','ml-1m'],required=True);main(p.parse_args().dataset)
