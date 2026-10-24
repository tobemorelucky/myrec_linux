"""Nonpreemptive eight-arm shared pool; existing Toys has priority after BPR exit."""
import os,json,time,subprocess,importlib.util,traceback,hashlib
from pathlib import Path
ROOT=Path('/home/yyx/hzgcode/pom2.0').resolve();OUT=Path(__file__).resolve().parent
RESOURCE=ROOT/'research/gpu_resource_scheduler_20261022_v1'
PYTHON='/home/yyx/anaconda3/envs/hzg_py10/bin/python'
spec=importlib.util.spec_from_file_location('resource_helpers',ROOT/'research/elector_sil_objective_full_20261023_v2/scheduler.py')
resource=importlib.util.module_from_spec(spec);spec.loader.exec_module(resource)
def read(p):return json.loads(Path(p).read_text())
def write(p,x,exclusive=True):
 p=Path(p);assert p.parent.resolve().is_relative_to(ROOT)
 with p.open('x' if exclusive else 'w') as f:json.dump(x,f,indent=2,default=str)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(2**20),b''):h.update(v)
 return h.hexdigest()
def alive(pid):
 try:return Path('/proc/%d/stat'%pid).read_text().split(') ',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False
def accepted(p):
 r=read(p/'result.json');done=read(p/'TEST_COMPLETE.json')
 assert r['train_complete'] and r['Dev_best_selected'] and r['Test_accepted'] and r['Test_passes']==1
 assert done['one_Test_pass'] and done['result_sha256']==sha(p/'result.json')
 cp=p/'best.pt'
 if not cp.exists():cp=Path(read(OUT/'PLAN.json')['Toys']['old_root'])/'arms/BPR-B0/best.pt'
 assert done['checkpoint_sha256']==r['checkpoint_sha256']==sha(cp)
 return r
def verify():
 assert read(OUT/'PREFLIGHT.json')['passed']
 for p,h in read(OUT/'SOURCE_LOCK.json').items():assert sha(p)==h,p
 for p,h in read(OUT/'PLAN.json')['archives'].items():assert sha(p)==h,p
def main():
 job=resource.acquire(RESOURCE/'gictrec_round1_20261024_v1.lock')
 assert job is not None,'Duplicate round1 dispatcher'
 assert not (OUT/'SCHEDULER_STARTED.json').exists(),'Do not resubmit a registered round'
 verify();plan=read(OUT/'PLAN.json');toy=plan['Toys'];old=Path(toy['old_root']);fix=Path(toy['repair_root'])
 # Exactly eight new Chapter5 runs. Toys entries only recover previously authorized work.
 pending=[dict(kind='gict',dataset=ds,variant=v,key=ds+'_'+v) for v in plan['variants'] for ds in ['ml-1m','beauty']]
 assert len(pending)==8
 toys_pending=list(toy['only_remaining']);toy_phase='WAITING_ORIGINAL_BPR';toy_halted=False
 toy_job=None;active={};done=[];halted=set();last=None
 write(OUT/'SCHEDULER_STARTED.json',dict(pid=os.getpid(),new_GICT_runs=8,Toys_original_BPR_pid=toy['BPR_pid'],poll_seconds=30,fail_stop='per-dataset queue; no retry/parameter change'))
 def state():
  nonlocal last
  x=dict(pid=os.getpid(),running={key:dict(pid=e['child'].pid,GPU=e['gpu'],kind=e['task']['kind'],log=str(e['path']/'stdout.log')) for key,e in active.items()},
   pending_GICT=[t['key'] for t in pending if t['dataset'] not in halted],paused_GICT_datasets=sorted(halted),completed=done,
   Toys=dict(phase=toy_phase,halted=toy_halted,pending=toys_pending,BPR_alive=alive(toy['BPR_pid'])))
  if x!=last:
   write(OUT/'STATE.tmp',x,False);os.replace(OUT/'STATE.tmp',OUT/'STATE.json');print(json.dumps(x),flush=True);last=x
 try:
  while True:
   if toy_phase=='WAITING_ORIGINAL_BPR' and not alive(toy['BPR_pid']) and not alive(toy['old_dispatcher_pid']):
    toy_job=resource.acquire(RESOURCE/'elector_toys_evalfix_20261024_v1.lock')
    if toy_job is not None:
     try:
      if (old/'arms/BPR-B0/TEST_COMPLETE.json').exists():
       accepted(old/'arms/BPR-B0');toy_phase='SS_READY';done.append('Toys_BPR_original_accepted')
      else:
       tr=read(old/'arms/BPR-B0/train_result.json');failure=read(old/'arms/BPR-B0/FAILURE.json')
       assert tr['train_complete'] and tr['Dev_best_selected']
       assert 'numpy' in failure['traceback'].lower() and 'score_views' in failure['traceback']
       assert not (OUT/'toys_bpr_acceptance/TEST_STARTED.json').exists()
       toy_phase='REPAIR_READY'
     except BaseException as e:
      toy_halted=True;toy_phase='BLOCKED';write(OUT/'TOYS_BLOCKED.json',dict(error=repr(e),no_retry=True))
   for key,e in list(active.items()):
    code=e['child'].poll()
    if code is None:continue
    write(e['path']/'exit_status.json',dict(pid=e['child'].pid,code=code))
    try:
     assert code==0,'exit='+str(code)
     result=accepted(e['path'])
     if e['task']['kind']=='toys_repair':toy_phase='SS_READY'
     elif e['task']['kind']=='toys_ss':
      if (OUT/'toys_bpr_acceptance/TEST_COMPLETE.json').exists():
       expected=read(OUT/'toys_bpr_acceptance/TEST_PROTOCOL.json')['input_hash']
      else:expected=read(old/'TEST_PROTOCOL_LOCK.json')['input_hash']
      assert result['Test_input_hash']==expected,'Toys candidate/history mismatch'
     done.append(key)
    except BaseException as error:
     write(OUT/(key+'_QUEUE_FAILURE.json'),dict(error=repr(error),no_retry=True,following_tasks_paused=True))
     if e['task']['kind']=='gict':halted.add(e['task']['dataset'])
     else:toy_halted=True;toy_phase='FAILED'
    os.close(e['gfd']);os.close(e['tfd']);del active[key]
   # Protect completed artifacts and skip accepted Toys tasks rather than repeat.
   if toy_phase=='SS_READY' and not toy_halted:
    for arm in list(toys_pending):
     for base in [fix,ROOT/'research/elector_sil_toys_parallel_20261024_v1']:
      p=base/'arms'/arm
      if (p/'TEST_COMPLETE.json').exists():
       accepted(p);toys_pending.remove(arm);done.append('Toys_'+arm+'_reused');break
   for gpu,info in resource.observe().items():
    if not resource.idle(info):continue
    gfd=resource.acquire(RESOURCE/('gpu%d.lock'%gpu))
    if gfd is None:continue
    time.sleep(1)
    if not resource.idle(resource.observe()[gpu]):os.close(gfd);continue
    task=None
    toy_active=any(e['task']['kind'].startswith('toys') for e in active.values())
    if not toy_halted and not toy_active and toy_phase=='REPAIR_READY':
     task=dict(kind='toys_repair',key='Toys_BPR_acceptance')
    elif not toy_halted and not toy_active and toy_phase=='SS_READY' and toys_pending:
     task=dict(kind='toys_ss',key='Toys_'+toys_pending[0],arm=toys_pending[0])
    else:task=next((t for t in pending if t['dataset'] not in halted),None)
    if task is None:os.close(gfd);continue
    key=task['key'];lockname=('elector_toys_full_20261024_v1_'+task['arm']) if task['kind']=='toys_ss' else 'gict_round1_'+key
    tfd=resource.acquire(RESOURCE/(lockname+'.lock'))
    if tfd is None:os.close(gfd);continue
    verify()
    if task['kind']=='gict':
     p=OUT/'runs'/key;command=[PYTHON,'-u','-B',str(OUT/'train_task.py'),'--dataset',task['dataset'],'--variant',task['variant']]
     assert not any(p.iterdir()),'Existing GICT task output: no duplicate/retry'
     pending.remove(task)
    elif task['kind']=='toys_repair':
     p=OUT/'toys_bpr_acceptance';command=[PYTHON,'-u','-B',str(OUT/'toys_accept.py')];toy_phase='REPAIR_RUNNING'
     assert not any(p.iterdir())
    else:
     p=fix/'arms'/task['arm'];command=[PYTHON,'-u','-B',str(fix/'train_task.py'),'--arm',task['arm']]
     assert not any(p.iterdir()),'Existing repair SS output: refuse automatic retry'
     toys_pending.remove(task['arm'])
    write(p/'CLAIM.json',dict(queue_pid=os.getpid(),GPU=gpu,task=key))
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
     CUDA_CACHE_PATH=str(OUT/('cuda_cache_gpu%d'%gpu)),TORCHINDUCTOR_CACHE_DIR=str(OUT/('torch_cache_gpu%d'%gpu)))
    with (p/'stdout.log').open('x') as log:
     fds=(job,gfd,tfd)+((toy_job,) if toy_job is not None else ())
     child=subprocess.Popen(['nohup',*command],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,close_fds=True,pass_fds=fds,start_new_session=True)
    write(p/'PID.json',dict(pid=child.pid,GPU=gpu))
    active[key]=dict(child=child,gpu=gpu,gfd=gfd,tfd=tfd,path=p,task=task)
   state()
   gict_finished=not any(t['dataset'] not in halted for t in pending)
   toy_finished=toy_halted or (toy_phase=='SS_READY' and not toys_pending)
   if not active and gict_finished and toy_finished:
    write(OUT/'QUEUE_COMPLETE.json',dict(completed=done,paused_datasets=sorted(halted),Toys=toy_phase));break
   time.sleep(30)
 except BaseException as e:
  write(OUT/'SCHEDULER_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),active_processes_preserved=[e['child'].pid for e in active.values()],no_retry=True));raise
 finally:
  for e in active.values():os.close(e['gfd']);os.close(e['tfd'])
  if toy_job is not None:os.close(toy_job)
  os.close(job)
if __name__=='__main__':main()
