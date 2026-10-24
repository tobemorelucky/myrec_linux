"""At most two new K3 tasks; retain existing child, never dispatch cancelled jobs."""
import importlib.util,os,subprocess,time,traceback
from plan import *
RESOURCE=ROOT/'research/gpu_resource_scheduler_20261022_v1'
PYTHON='/home/yyx/anaconda3/envs/hzg_py10/bin/python'
p=ROOT/'research/elector_sil_objective_full_20261023_v2/scheduler.py'
spec=importlib.util.spec_from_file_location('gpu_resource_functions',p)
resource=importlib.util.module_from_spec(spec);spec.loader.exec_module(resource)
def proc_live(pid,start):
 try:
  t=Path('/proc/%d/stat'%pid).read_text();f=t[t.rfind(')')+2:].split()
  return f[19]==start and f[0] not in ('Z','X')
 except FileNotFoundError:return False
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
 queue_fd=resource.acquire(RESOURCE/'elector_k3_20261023_v1.lock')
 assert queue_fd is not None,'Duplicate K3 queue'
 verify();assert read(OUT/'CHECK_PASS.json')['passed']
 cancelled=read(OUT/'CANCELLATION_COMPLETE.json')
 for old in cancelled['parents']:
  assert not proc_live(old['process']['pid'],old['process']['start']),'Old dispatcher must be stopped'
 adopted=cancelled['adopted'];adopt_status={};tasks=read(OUT/'MANIFEST.json')['tasks']
 assert len(tasks)==2 and all(t['K']==3 and t['seed']==42 and t['arm']=='SS-C1' for t in tasks)
 pending=list(tasks);running={};finished=[];errors=[];last=None
 def state(label):
  nonlocal last
  value=dict(state=label,pid=os.getpid(),pending=pending,
   running={k:dict(pid=v['child'].pid,GPU=v['gpu']) for k,v in running.items()},
   completed=finished,retained_training=adopted,retained_status=adopt_status,errors=errors,
   time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
  signature={k:v for k,v in value.items() if k!='time_utc'}
  if signature!=last:
   tmp=OUT/'STATE.tmp'
   with tmp.open('w') as f:json.dump(value,f,indent=2)
   os.replace(tmp,OUT/'STATE.json');print(json.dumps(value),flush=True);last=signature
 try:
  write(OUT/'SCHEDULER_STARTED.json',dict(pid=os.getpid(),new_tasks=tasks,old_queues_cancelled=True))
  while True:
   for record in adopted:
    pid=record['pid']
    if str(pid) in adopt_status or proc_live(pid,record['start']):continue
    path=ROOT/'research/elector_sil_formal_20261023_v1/runs'/record['task']
    try:
     evidence=acceptance(path);adopt_status[str(pid)]='COMPLETED'
     write(OUT/('RETAINED_%d_COMPLETE.json'%pid),dict(pid=pid,actual_exit_code=None,acceptance_verified=True,result_path=str(path/'result.json'),**evidence))
    except BaseException as exc:
     adopt_status[str(pid)]='FAILED';errors.append('Retained training '+repr(exc))
   for name,entry in list(running.items()):
    code=entry['child'].poll()
    if code is None:continue
    path=OUT/'runs'/name;write(path/'exit_status.json',dict(code=code,pid=entry['child'].pid))
    try:
     assert code==0,'exit='+str(code);acceptance(path);finished.append(name)
    except BaseException as exc:errors.append(name+': '+repr(exc))
    os.close(entry['gpu_fd']);os.close(entry['task_fd']);del running[name]
   if errors:
    state('PENDING_TASKS_HALTED')
    if not running and len(adopt_status)==len(adopted):raise RuntimeError('; '.join(errors))
   elif pending:
    gpu,gpu_fd=choose()
    if gpu is not None:
     verify();t=pending[0];name=task_id(t['dataset'],3,42,'SS-C1')
     path=folder(t['dataset'],3,42,'SS-C1');path.mkdir(exist_ok=False)
     task_fd=resource.acquire(RESOURCE/('elector_k3_20261023_'+t['dataset']+'.lock'))
     assert task_fd is not None,'Duplicate dataset K3 task'
     env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',
       OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_CACHE_PATH=str(OUT/'cuda_cache'),
       TORCHINDUCTOR_CACHE_DIR=str(OUT/'torch_cache'))
     with (path/'stdout.log').open('x') as log:
      child=subprocess.Popen(['nohup',PYTHON,'-u','-B',str(OUT/'train_task.py'),'--dataset',t['dataset'],
       '--arm','SS-C1','--K','3','--seed','42'],cwd=ROOT,stdin=subprocess.DEVNULL,
       stdout=log,stderr=subprocess.STDOUT,env=env,close_fds=True,
       pass_fds=(queue_fd,gpu_fd,task_fd),start_new_session=True)
     write(path/'PID.json',dict(pid=child.pid));running[name]=dict(child=child,gpu=gpu,gpu_fd=gpu_fd,task_fd=task_fd)
     pending.pop(0)
    state('RUNNING' if running else 'WAITING_GPU')
   elif not running and len(adopt_status)==len(adopted):
    state('COMPLETED');break
   else:state('RUNNING' if running else 'RETAINED_TRAINING_ONLY')
   time.sleep(15)
 except BaseException as exc:
  write(OUT/'SCHEDULER_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),no_retry=True,
    pending_halted=True,active_training_preserved=[v['child'].pid for v in running.values()]))
  raise
 finally:
  for entry in running.values():os.close(entry['gpu_fd']);os.close(entry['task_fd'])
  os.close(queue_fd)
if __name__=='__main__':main()
