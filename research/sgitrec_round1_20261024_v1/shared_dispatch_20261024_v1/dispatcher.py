"""Adopt the running B0; dispatch ONLY remaining original G1/G2/G3 to any idle GPU."""
import os,sys,json,time,subprocess,traceback,hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent;BASE=OUT.parent
PYTHON='/home/yyx/anaconda3/envs/hzg_py10/bin/python'
RESOURCE=ROOT/'research/gpu_resource_scheduler_20261022_v1'
spec=importlib.util.spec_from_file_location('original_sgit_queue',BASE/'scheduler.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original);resource=original.resource
def read(p):return json.loads(Path(p).read_text())
def sha(p):return original.sha(p)
def write(p,value):
 p=Path(p);assert p.parent.resolve().is_relative_to(ROOT)
 with p.open('x') as f:json.dump(value,f,indent=2)
def alive(pid,tick=None):
 try:
  row=Path('/proc/'+str(pid)+'/stat').read_text().split()
  return row[2]!='Z' and (tick is None or row[21]==str(tick))
 except FileNotFoundError:return False
def accepted(run):
 x=read(run/'result.json');mark=read(run/'TEST_COMPLETE.json')
 assert x['train_complete'] and x['Dev_best_selected'] and x['Test_accepted'] and x['Test_passes']==1
 assert mark['one_Test_pass'] and mark['result_sha256']==sha(run/'result.json')
 assert mark['checkpoint_sha256']==x['checkpoint_sha256']==sha(run/'best.pt')
 return dict(best_epoch=x['best']['epoch'],checkpoint_sha256=x['checkpoint_sha256'])
def verify():
 plan=original.verify();handoff=read(OUT/'HANDOFF.json')
 assert sha(BASE/'PLAN.json')==handoff['original_PLAN_sha256']
 assert sha(__file__)==handoff['dispatcher_sha256']
 assert handoff['authorized_original_total']==6 and handoff['pending']==['G3','G2','G1']
 assert not alive(handoff['old_queue_pid'],handoff['old_queue_start_tick']),'Old dispatcher still alive'
 return plan,handoff
def status(running,pending,completed,halted,failures):
 value=dict(pid=os.getpid(),state='HALTED_WAITING_ACTIVE' if halted and running else ('FAILED' if halted else ('RUNNING' if running else 'WAITING_GPU')),
  utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
  running={name:{k:v for k,v in job.items() if k not in ['process','gpu_fd','task_fd']} for name,job in running.items()},
  pending=pending,completed=completed,halted=halted,failures=failures,poll_seconds=20,
  scheduling='longest estimated pending task first; either physical GPU, one task per GPU')
 if not running and not pending and not halted:value['state']='COMPLETED'
 temp=OUT/'STATE.tmp'
 with temp.open('w') as f:json.dump(value,f,indent=2)
 os.replace(temp,OUT/'STATE.json');print(json.dumps(value),flush=True)
def main():
 pool=None;running={};completed=['beauty_B0','beauty_G3'];pending=[];halted=False;failures=[]
 try:
  pool=resource.acquire(RESOURCE/'sgitrec_round1_shared_20261024_v1.lock')
  if pool is None:raise RuntimeError('Duplicate shared dispatcher')
  assert not (OUT/'STARTED.json').exists(),'Dispatcher already registered; no duplicate/retry'
  plan,h=verify();pending=h['pending'].copy()
  for variant in pending:
   assert not (BASE/'runs'/('ml-1m_k3_s42_'+variant)).exists(),variant
   assert not (BASE/'queues/ml-1m'/(variant+'_CLAIM.json')).exists(),variant
  for variant in ['B0','G3']:accepted(BASE/'runs'/('beauty_k2_s42_'+variant))
  assert alive(h['adopted_pid'],h['adopted_start_tick']),'Adopted training exited before handoff'
  running['ml-1m_B0']=dict(pid=h['adopted_pid'],start_tick=h['adopted_start_tick'],GPU=0,owned=False,
   run=str(BASE/'runs/ml-1m_k3_s42_B0'),log=str(BASE/'runs/ml-1m_k3_s42_B0/train.log'))
  write(OUT/'STARTED.json',dict(pid=os.getpid(),adopted=h['adopted_pid'],pending=pending,no_new_experiments=True))
  while True:
   for name,job in list(running.items()):
    code=job['process'].poll() if job['owned'] else (None if alive(job['pid'],job.get('start_tick')) else 'adopted_exited')
    if code is None:continue
    try:
     if job['owned'] and code!=0:raise RuntimeError('Training exit='+str(code))
     result=accepted(Path(job['run']))
     completed.append(name);write(OUT/'tasks'/(name+'_ACCEPTED.json'),dict(pid=job['pid'],**result))
    except BaseException as e:
     halted=True;failures.append(dict(task=name,pid=job['pid'],error=repr(e)))
     write(OUT/'tasks'/(name+'_FAILURE.json'),dict(error=repr(e),traceback=traceback.format_exc(),no_retry=True))
    finally:
     if job['owned']:
      os.close(job['gpu_fd']);os.close(job['task_fd'])
     del running[name]
   if not halted:
    inventory=resource.observe()
    for gpu in [1,0]:
     if not pending:break
     if gpu in [v['GPU'] for v in running.values()]:continue
     if gpu not in inventory or not resource.idle(inventory[gpu]):continue
     lease=resource.acquire(RESOURCE/('gpu%d.lock'%gpu))
     if lease is None:continue
     if not resource.idle(resource.observe()[gpu]):os.close(lease);continue
     variant=pending[0];name='ml-1m_'+variant;run=BASE/'runs'/('ml-1m_k3_s42_'+variant)
     task=resource.acquire(RESOURCE/('sgitrec_round1_20261024_v1_ml-1m_k3_s42_'+variant+'.lock'))
     if task is None:os.close(lease);raise RuntimeError('Duplicate task lease '+name)
     try:
      verify();assert not run.exists()
      write(BASE/'queues/ml-1m'/(variant+'_CLAIM.json'),dict(task=name,dispatcher=os.getpid(),run=str(run),handoff=str(OUT)))
      ds=plan['datasets']['ml-1m']
      command=['nohup',PYTHON,'-u','-B',str(ROOT/'tools/run_sgitrec.py'),'--config',ds['config'],'--graph',ds['graph'],
       '--variant',variant,'--output-dir',str(run),'--gpu',str(gpu),'--gpu-lease-fd',str(lease),'--manifest',str(BASE/'PLAN.json')]
      env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
      stdout=OUT/'tasks'/(name+'.stdout.log')
      with stdout.open('x') as log:
       child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
        start_new_session=True,close_fds=True,pass_fds=(pool,task,lease))
      job=dict(pid=child.pid,process=child,GPU=gpu,owned=True,run=str(run),log=str(run/'train.log'),gpu_fd=lease,task_fd=task)
      running[name]=job;pending.pop(0)
      pid_record=dict(pid=child.pid,GPU=gpu,GPU_uuid=inventory[gpu]['uuid'],command=command,stdout=str(stdout),dispatcher=os.getpid())
      write(OUT/'tasks'/(name+'_PID.json'),pid_record)
      write(BASE/'queues/ml-1m'/(variant+'_PID.json'),pid_record)
     except BaseException:
      if not any(v.get('gpu_fd')==lease for v in running.values()):os.close(lease);os.close(task)
      raise
   status(running,pending,completed,halted,failures)
   if not running and (not pending or halted):break
   time.sleep(20)
  write(OUT/('FAILED.json' if halted else 'QUEUE_COMPLETE.json'),dict(completed=completed,pending=pending,failures=failures,no_more_tasks=True))
 except BaseException as e:
  write(OUT/'DISPATCH_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),
   active={n:v['pid'] for n,v in running.items()},pending=pending,no_kill=True,no_retry=True))
  raise
 finally:
  if pool is not None:os.close(pool)
if __name__=='__main__':main()
