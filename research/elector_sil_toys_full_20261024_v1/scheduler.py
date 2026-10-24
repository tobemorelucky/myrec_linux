"""Exactly four Toys tasks, serial, nonpreemptive, shared GPU locks."""
import time,subprocess,importlib.util,traceback
from common import *
RES=ROOT/'research/gpu_resource_scheduler_20261022_v1';PYTHON='/home/yyx/anaconda3/envs/hzg_py10/bin/python'
spec=importlib.util.spec_from_file_location('existing_resource_functions',ROOT/'research/elector_sil_objective_full_20261023_v2/scheduler.py');resource=importlib.util.module_from_spec(spec);spec.loader.exec_module(resource)
def state(label,**kw):
 value=dict(state=label,pid=os.getpid(),order=ORDER,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
 write(OUT/'STATE.tmp',value,False);os.replace(OUT/'STATE.tmp',OUT/'STATE.json');print(json.dumps(value),flush=True)
def choose(completed):
 while True:
  priority=read(ROOT/'research/elector_sil_night12_20261023_v1/STATE.json')
  if priority.get('running') or priority.get('pending'):
   state('WAITING_EXISTING_12',completed=completed,poll_seconds=60);time.sleep(60);continue
  inv=resource.observe()
  for gpu in (0,1):
   if gpu not in inv or not resource.idle(inv[gpu]):continue
   fd=resource.acquire(RES/('gpu%d.lock'%gpu))
   if fd is None:continue
   time.sleep(1)
   if resource.idle(resource.observe()[gpu]):return gpu,fd
   os.close(fd)
  state('WAITING_GPU',completed=completed,inventory=inv,poll_seconds=60);time.sleep(60)
def main():
 job=resource.acquire(RES/'elector_toys_full_20261024_v1.lock');assert job is not None,'Duplicate Toys queue lock'
 gpu_fd=None;task_fd=None;child=None;done=[]
 try:
  assert not (OUT/'SCHEDULER_STARTED.json').exists(),'Already launched; no automatic resume/retry'
  verify();assert read(OUT/'CHECK_PASS.json')['passed']
  write(OUT/'SCHEDULER_STARTED.json',dict(pid=os.getpid(),order=ORDER,max_tasks=4,serial=True,poll_seconds=60))
  for arm in ORDER:
   p=OUT/'arms'/arm;assert not any(p.iterdir()),'Existing output; refuse duplicate task'
   verify();gpu,gpu_fd=choose(done);task_fd=resource.acquire(RES/('elector_toys_full_20261024_v1_'+arm+'.lock'));assert task_fd is not None
   write(p/'CLAIM.json',dict(queue_pid=os.getpid(),GPU=gpu,arm=arm))
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',CUDA_CACHE_PATH=str(OUT/('cuda_cache_gpu%d'%gpu)),TORCHINDUCTOR_CACHE_DIR=str(OUT/('torch_cache_gpu%d'%gpu)))
   with (p/'stdout.log').open('x') as log:
    child=subprocess.Popen(['nohup',PYTHON,'-u','-B',str(OUT/'train_task.py'),'--arm',arm],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,close_fds=True,pass_fds=(job,gpu_fd,task_fd))
   write(p/'PID.json',dict(pid=child.pid));state('RUNNING',current_arm=arm,train_pid=child.pid,GPU=gpu,completed=done,remaining=ORDER[len(done)+1:],log=str(p/'stdout.log'))
   code=child.wait();write(p/'exit_status.json',dict(code=code,pid=child.pid));assert code==0,arm+' exit='+str(code)
   r=read(p/'result.json');d=read(p/'TEST_COMPLETE.json')
   assert r['train_complete'] and r['Dev_best_selected'] and r['Test_accepted'] and r['Test_passes']==1
   assert d['one_Test_pass'] and d['result_sha256']==sha(p/'result.json') and d['checkpoint_sha256']==r['checkpoint_sha256']==sha(p/'best.pt')
   done.append(arm);child=None;os.close(gpu_fd);gpu_fd=None;os.close(task_fd);task_fd=None
  state('COMPLETED',completed=done,remaining=[])
 except BaseException as e:
  write(OUT/'QUEUE_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),completed=done,following_tasks_halted=True,no_retry=True,current_child_pid=child.pid if child else None))
  state('FAILED',completed=done,error=repr(e));raise
 finally:
  if gpu_fd is not None:os.close(gpu_fd)
  if task_fd is not None:os.close(task_fd)
  os.close(job)
if __name__=='__main__':main()
