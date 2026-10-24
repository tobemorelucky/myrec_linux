"""Shared12-task nonpreemptive pool; Beauty completion never strands a free GPU."""
import importlib.util,os,subprocess,time,traceback
from plan import *
RESOURCE=ROOT/'research/gpu_resource_scheduler_20261022_v1'
PYTHON='/home/yyx/anaconda3/envs/hzg_py10/bin/python'
spec=importlib.util.spec_from_file_location('safe_gpu_resource_functions',ROOT/'research/elector_sil_objective_full_20261023_v2/scheduler.py')
resource=importlib.util.module_from_spec(spec);spec.loader.exec_module(resource)
def pick(pending,running,failed):
 active={entry['task']['dataset'] for entry in running.values()}
 candidates=[(i,t) for i,t in enumerate(pending) if t['dataset'] not in failed]
 return min(candidates,key=lambda x:(x[1]['priority'],x[1]['dataset'] in active,x[0]))[0] if candidates else None
def choose():
 inv=resource.observe()
 for gpu in (0,1):
  if gpu not in inv or not resource.idle(inv[gpu]):continue
  fd=resource.acquire(RESOURCE/('gpu%d.lock'%gpu))
  if fd is None:continue
  time.sleep(1);fresh=resource.observe()
  if gpu in fresh and resource.idle(fresh[gpu]):return gpu,fd
  os.close(fd)
 return None,None
def main():
 job=resource.acquire(RESOURCE/'elector_night12_20261023.lock')
 assert job is not None,'Duplicate nightly pool'
 running={};pending=[];finished=[];failed={};blocked=[];last=None
 def state(label):
  nonlocal last
  value=dict(state=label,pid=os.getpid(),pending=pending,completed=finished,failed_datasets=failed,halted_tasks=blocked,
   running={name:dict(pid=entry['child'].pid,GPU=entry['gpu'],task=entry['task']) for name,entry in running.items()},
   time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
  signature={k:v for k,v in value.items() if k!='time_utc'}
  if signature!=last:
   with (OUT/'STATE.tmp').open('w') as f:json.dump(value,f,indent=2)
   os.replace(OUT/'STATE.tmp',OUT/'STATE.json');print(json.dumps(value),flush=True);last=signature
 try:
  m=verify();assert read(OUT/'CHECK_PASS.json')['passed'];pending=list(m['tasks'])
  assert len(pending)==12 and {t['dataset'] for t in pending}=={'beauty','ml-1m'}
  write(OUT/'SCHEDULER_STARTED.json',dict(pid=os.getpid(),tasks=pending,no_preemption=True,max_tasks=12,poll_seconds=30))
  while True:
   for name,entry in list(running.items()):
    code=entry['child'].poll()
    if code is None:continue
    p=OUT/'runs'/name;write(p/'exit_status.json',dict(code=code,pid=entry['child'].pid))
    try:
     assert code==0,'exit='+str(code);acceptance(p);finished.append(name)
    except BaseException as exc:
     ds=entry['task']['dataset'];failed[ds]=repr(exc)
     write(OUT/(ds+'_QUEUE_FAILURE.json'),dict(task=name,error=repr(exc),no_retry=True,unstarted_dataset_tasks_halted=True))
    os.close(entry['gpu_fd']);os.close(entry['task_fd']);del running[name]
   for task in list(pending):
    if task['dataset'] in failed:pending.remove(task);blocked.append(task)
   while len(running)<2:
    index=pick(pending,running,failed)
    if index is None:break
    gpu,gpu_fd=choose()
    if gpu is None:break
    verify();t=pending[index];name=task_id(t['dataset'],t['K'],t['seed'],t['arm'])
    p=folder(t['dataset'],t['K'],t['seed'],t['arm']);p.mkdir(exist_ok=False)
    task_fd=resource.acquire(RESOURCE/('elector_night12_20261023_'+name+'.lock'))
    assert task_fd is not None,'Duplicate task lock'
    write(p/'CLAIM.json',dict(scheduler_pid=os.getpid(),GPU=gpu,task=t))
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',
     OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_CACHE_PATH=str(OUT/('cuda_cache_gpu%d'%gpu)),
     TORCHINDUCTOR_CACHE_DIR=str(OUT/('torch_cache_gpu%d'%gpu)))
    with (p/'stdout.log').open('x') as log:
     child=subprocess.Popen(['nohup',PYTHON,'-u','-B',str(OUT/'train_task.py'),'--dataset',t['dataset'],
       '--arm',t['arm'],'--K',str(t['K']),'--seed',str(t['seed'])],cwd=ROOT,stdin=subprocess.DEVNULL,
       stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,close_fds=True,pass_fds=(job,gpu_fd,task_fd))
    write(p/'PID.json',dict(pid=child.pid));running[name]=dict(child=child,gpu=gpu,gpu_fd=gpu_fd,task_fd=task_fd,task=t)
    pending.pop(index)
   if not pending and not running:
    state('COMPLETED' if not failed else 'FAILED_PARTIAL')
    if failed:raise RuntimeError(repr(failed))
    break
   state('RUNNING' if running else 'WAITING_GPU');time.sleep(30)
 except BaseException as exc:
  write(OUT/'SCHEDULER_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),no_retry=True,
   active_training_preserved=[e['child'].pid for e in running.values()],remaining_tasks_halted=pending))
  raise
 finally:
  for e in running.values():os.close(e['gpu_fd']);os.close(e['task_fd'])
  os.close(job)
if __name__=='__main__':main()
