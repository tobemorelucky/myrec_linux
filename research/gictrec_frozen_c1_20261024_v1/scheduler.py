"""Only G1/G2/G3; shared GPU locks, no preemption, fail stops pending jobs."""
import os,json,time,subprocess,importlib.util,hashlib,traceback
from pathlib import Path
ROOT=Path('/home/yyx/hzgcode/pom2.0').resolve();OUT=Path(__file__).resolve().parent
RESOURCE=ROOT/'research/gpu_resource_scheduler_20261022_v1'
spec=importlib.util.spec_from_file_location('frozen_gpu_resource',ROOT/'research/elector_sil_objective_full_20261023_v2/scheduler.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x,exclusive=True):
 p=Path(p);assert p.parent.resolve().is_relative_to(ROOT)
 with p.open('x' if exclusive else 'w') as f:json.dump(x,f,indent=2)
def verify():
 assert read(OUT/'DIAGNOSTICS.json')['passed']
 for p,h in read(OUT/'SOURCE_LOCK.json').items():assert sha(p)==h,p
def main():
 job=r.acquire(RESOURCE/'gictrec_frozen_c1_20261024_v1.lock');assert job is not None
 assert not (OUT/'SCHEDULER_STARTED.json').exists(),'Do not resubmit'
 verify();pending=['G1','G2','G3'];active={};done=[];failed=False;last=None
 write(OUT/'SCHEDULER_STARTED.json',dict(pid=os.getpid(),tasks=pending,full_training_runs=3,poll_seconds=30))
 try:
  while True:
   for variant,e in list(active.items()):
    code=e['child'].poll()
    if code is None:continue
    p=OUT/'runs'/variant;write(p/'exit_status.json',dict(pid=e['child'].pid,code=code))
    try:
     assert code==0,'exit='+str(code)
     result=read(p/'result.json');end=read(p/'TEST_COMPLETE.json')
     assert result['train_complete'] and result['Dev_best_selected'] and result['Test_accepted'] and result['Test_passes']==1
     assert result['frozen_C1_unchanged'] and result['optimizer_steps']>0
     assert end['result_sha256']==sha(p/'result.json') and result['checkpoint_sha256']==end['checkpoint_sha256']==sha(p/'best.pt')
     done.append(variant)
    except BaseException as error:
     failed=True;write(OUT/(variant+'_QUEUE_FAILURE.json'),dict(error=repr(error),pending_paused=pending,no_retry=True))
    os.close(e['gfd']);os.close(e['tfd']);del active[variant]
   if not failed:
    for gpu,info in r.observe().items():
     if not pending or not r.idle(info):continue
     gfd=r.acquire(RESOURCE/('gpu%d.lock'%gpu))
     if gfd is None:continue
     time.sleep(1)
     if not r.idle(r.observe()[gpu]):os.close(gfd);continue
     variant=pending[0];tfd=r.acquire(RESOURCE/('gictrec_frozen_c1_20261024_v1_'+variant+'.lock'))
     if tfd is None:os.close(gfd);continue
     verify();p=OUT/'runs'/variant;assert not any(p.iterdir()),'Existing output: no retry'
     write(p/'CLAIM.json',dict(queue_pid=os.getpid(),GPU=gpu,variant=variant))
     env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
       CUDA_CACHE_PATH=str(OUT/('cuda_cache_gpu%d'%gpu)),TORCHINDUCTOR_CACHE_DIR=str(OUT/('torch_cache_gpu%d'%gpu)))
     with (p/'stdout.log').open('x') as log:
      child=subprocess.Popen(['nohup','/home/yyx/anaconda3/envs/hzg_py10/bin/python','-u','-B',str(ROOT/'tools/run_gictrec_frozen_c1.py'),'--variant',variant],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,pass_fds=(job,gfd,tfd))
     write(p/'PID.json',dict(pid=child.pid,GPU=gpu));active[variant]=dict(child=child,gpu=gpu,gfd=gfd,tfd=tfd);pending.pop(0)
   state=dict(pid=os.getpid(),running={v:dict(pid=e['child'].pid,GPU=e['gpu']) for v,e in active.items()},pending=pending,completed=done,failed=failed)
   if state!=last:
    write(OUT/'STATE.tmp',state,False);os.replace(OUT/'STATE.tmp',OUT/'STATE.json');print(json.dumps(state),flush=True);last=state
   if not active and (not pending or failed):write(OUT/'QUEUE_COMPLETE.json',state);break
   time.sleep(30)
 except BaseException as error:
  write(OUT/'SCHEDULER_FAILURE.json',dict(error=repr(error),traceback=traceback.format_exc(),active_preserved=[e['child'].pid for e in active.values()]));raise
 finally:
  for e in active.values():os.close(e['gfd']);os.close(e['tfd'])
  os.close(job)
if __name__=='__main__':main()
