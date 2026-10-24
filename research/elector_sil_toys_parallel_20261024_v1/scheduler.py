"""Adopt existing BPR without restart, dispatch only three remaining SS jobs."""
import time,subprocess,importlib.util,traceback
from common import *
OLD=ROOT/'research/elector_sil_toys_full_20261024_v1';RES=ROOT/'research/gpu_resource_scheduler_20261022_v1';PYTHON='/home/yyx/anaconda3/envs/hzg_py10/bin/python'
spec=importlib.util.spec_from_file_location('resource_helpers',ROOT/'research/elector_sil_objective_full_20261023_v2/scheduler.py');resource=importlib.util.module_from_spec(spec);spec.loader.exec_module(resource)
def alive(pid):
 try:
  s=Path('/proc/%d/stat'%pid).read_text().split(') ',1)[1].split()[0]
  return s!='Z'
 except FileNotFoundError:return False
def accepted(p):
 r=read(p/'result.json');d=read(p/'TEST_COMPLETE.json')
 assert r['train_complete'] and r['Dev_best_selected'] and r['Test_accepted'] and r['Test_passes']==1
 assert d['one_Test_pass'] and d['result_sha256']==sha(p/'result.json') and d['checkpoint_sha256']==r['checkpoint_sha256']==sha(p/'best.pt')
 return r
def choose():
 priority=read(ROOT/'research/elector_sil_night12_20261023_v1/STATE.json')
 if priority.get('running') or priority.get('pending'):return None,None
 inv=resource.observe()
 for gpu in (1,0):
  if gpu not in inv or not resource.idle(inv[gpu]):continue
  fd=resource.acquire(RES/('gpu%d.lock'%gpu))
  if fd is None:continue
  time.sleep(1)
  if resource.idle(resource.observe()[gpu]):return gpu,fd
  os.close(fd)
 return None,None
def main():
 job=resource.acquire(RES/'elector_toys_parallel_20261024_v1.lock');assert job is not None,'Duplicate parallel queue'
 adopt=read(OUT/'HANDOFF.json');pid=adopt['preserved_BPR_pid'];adopted_pending=True
 active={};pending=list(ORDER);done=[];halted=False;last=None
 def state(label):
  nonlocal last
  value=dict(state=label,pid=os.getpid(),adopted_BPR=dict(pid=pid,GPU=0,completed=not adopted_pending,original_root=str(OLD)),running={a:dict(pid=e['child'].pid,GPU=e['GPU']) for a,e in active.items()},pending=pending,completed=done,halted=halted,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
  signature={k:v for k,v in value.items() if k!='time_utc'}
  if signature!=last:
   write(OUT/'STATE.tmp',value,False);os.replace(OUT/'STATE.tmp',OUT/'STATE.json');print(json.dumps(value),flush=True);last=signature
 try:
  verify();assert read(OUT/'CALIBRATION_PASS.json')['passed']
  write(OUT/'SCHEDULER_STARTED.json',dict(pid=os.getpid(),adopted_BPR_pid=pid,new_tasks=3,total_full_runs=4,max_GPUs=2,poll_seconds=30))
  while True:
   if adopted_pending:
    ap=OLD/'arms/BPR-B0'
    if not alive(pid):
     try:
      accepted(ap);done.append('BPR-B0');adopted_pending=False
      write(OUT/'ADOPTED_BPR_COMPLETE.json',dict(pid=pid,result=str(ap/'result.json'),preserved_no_restart=True))
     except BaseException as e:
      halted=True;adopted_pending=False;write(OUT/'ADOPTED_BPR_FAILURE.json',dict(error=repr(e),no_retry=True,following_tasks_halted=True))
   for arm,e in list(active.items()):
    code=e['child'].poll()
    if code is None:continue
    p=OUT/'arms'/arm;write(p/'exit_status.json',dict(code=code,pid=e['child'].pid))
    try:assert code==0,'exit='+str(code);accepted(p);done.append(arm)
    except BaseException as exc:halted=True;write(OUT/(arm+'_QUEUE_FAILURE.json'),dict(error=repr(exc),no_retry=True,following_tasks_halted=True))
    os.close(e['gpu_fd']);os.close(e['task_fd']);del active[arm]
   while pending and not halted and len(active)+(1 if adopted_pending else 0)<2:
    gpu,gfd=choose()
    if gpu is None:break
    arm=pending[0];p=OUT/'arms'/arm;assert not any(p.iterdir()),'Duplicate arm output'
    tfd=resource.acquire(RES/('elector_toys_full_20261024_v1_'+arm+'.lock'));assert tfd is not None
    verify();write(p/'CLAIM.json',dict(queue_pid=os.getpid(),GPU=gpu,arm=arm))
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',CUDA_CACHE_PATH=str(OUT/('cuda_cache_gpu%d'%gpu)),TORCHINDUCTOR_CACHE_DIR=str(OUT/('torch_cache_gpu%d'%gpu)))
    with (p/'stdout.log').open('x') as log:
     child=subprocess.Popen(['nohup',PYTHON,'-u','-B',str(OUT/'train_task.py'),'--arm',arm],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,pass_fds=(job,gfd,tfd))
    write(p/'PID.json',dict(pid=child.pid));active[arm]=dict(child=child,GPU=gpu,gpu_fd=gfd,task_fd=tfd);pending.pop(0)
   if not active and not adopted_pending and (not pending or halted):
    state('FAILED' if halted else 'COMPLETED');break
   state('FAILED_DRAINING_ACTIVE' if halted else 'RUNNING' if active or adopted_pending else 'WAITING_GPU');time.sleep(30)
 except BaseException as e:
  write(OUT/'QUEUE_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),no_retry=True,active_tasks_preserved=[x['child'].pid for x in active.values()]))
  raise
 finally:
  for e in active.values():os.close(e['gpu_fd']);os.close(e['task_fd'])
  os.close(job)
if __name__=='__main__':main()
