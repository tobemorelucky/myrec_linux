"""Exactly six authorized SGITRec tasks: two nonpreemptive dataset queues."""
import os,sys,time,json,fcntl,subprocess,traceback,argparse,importlib.util,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
PYTHON='/home/yyx/anaconda3/envs/hzg_py10/bin/python'
RESOURCE=ROOT/'research/gpu_resource_scheduler_20261022_v1'
spec=importlib.util.spec_from_file_location('sgit_queue_resources',ROOT/'research/elector_sil_objective_full_20261023_v2/scheduler.py')
resource=importlib.util.module_from_spec(spec);spec.loader.exec_module(resource)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(2**20),b''):h.update(v)
 return h.hexdigest()
def write(p,v):
 p=Path(p);assert p.parent.resolve().is_relative_to(ROOT)
 with p.open('x') as f:json.dump(v,f,indent=2)
def status(ds,stage,**kw):
 folder=OUT/'queues'/ds
 value=dict(dataset=ds,state=stage,queue_pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
 temp=folder/'STATE.tmp'
 with temp.open('w') as f:json.dump(value,f,indent=2)
 os.replace(temp,folder/'STATE.json');print(json.dumps(value),flush=True)
def verify():
 plan=read(OUT/'PLAN.json')
 for key in ['source_sha256','protected_sha256']:
  for p,digest in plan[key].items():
   q=Path(p).resolve();assert q.is_relative_to(ROOT) and sha(q)==digest,p
 assert plan['authorized_tasks']==6 and sum(len(s['order']) for s in plan['datasets'].values())==6
 for ds,s in plan['datasets'].items():
  assert sha(s['config'])==s['config_sha256'] and sha(s['graph'])==s['graph_sha256']
  assert read(OUT/'checks'/(ds+'_routing_v2.json'))['passed']
  assert read(OUT/'checks'/(ds+'_activation.json'))['structurally_not_G1']
 assert read(OUT/'checks/lease_check.json')['passed']
 return plan
def choose(ds,preferences,completed,current,pending):
 while True:
  inv=resource.observe()
  for gpu in preferences:
   if gpu not in inv or not resource.idle(inv[gpu]):continue
   lease=resource.acquire(RESOURCE/('gpu%d.lock'%gpu))
   if lease is None:continue
   fresh=resource.observe()
   if gpu in fresh and resource.idle(fresh[gpu]):return gpu,lease,fresh[gpu]['uuid']
   os.close(lease)
  status(ds,'WAITING_GPU',completed=completed,current=current,pending=pending,preferences=preferences,inventory=inv,poll_seconds=60)
  time.sleep(60)
def main(ds):
 queue=task=gpu_lease=None;child=None;completed=[];current=None;pending=[]
 folder=OUT/'queues'/ds
 try:
  queue=resource.acquire(RESOURCE/('sgitrec_round1_20261024_v1_'+ds+'.lock'))
  if queue is None:raise RuntimeError('Duplicate dataset queue')
  if (folder/'STARTED.json').exists():raise RuntimeError('Queue already registered; no retry or duplicate submission')
  plan=verify();config=plan['datasets'][ds];pending=config['order'].copy()
  write(folder/'STARTED.json',dict(pid=os.getpid(),order=pending,manifest_sha256=sha(OUT/'PLAN.json')))
  for position,variant in enumerate(config['order']):
   verify();current=variant;pending=config['order'][position+1:]
   name=ds+'_k%d_s42_'%config['K']+variant;run=OUT/'runs'/name
   assert not run.exists(),'Existing run: refuse overwrite/retrain'
   task=resource.acquire(RESOURCE/('sgitrec_round1_20261024_v1_'+name+'.lock'))
   if task is None:raise RuntimeError('Duplicate task '+name)
   write(folder/(variant+'_CLAIM.json'),dict(task=name,queue_pid=os.getpid(),run=str(run)))
   gpu,gpu_lease,uuid=choose(ds,config['GPU_preference'],completed,current,pending)
   env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
   stdout=folder/(variant+'.stdout.log')
   with stdout.open('x') as log:
    command=['nohup',PYTHON,'-u','-B',str(ROOT/'tools/run_sgitrec.py'),'--config',config['config'],'--graph',config['graph'],
     '--variant',variant,'--output-dir',str(run),'--gpu',str(gpu),'--gpu-lease-fd',str(gpu_lease),'--manifest',str(OUT/'PLAN.json')]
    child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
     close_fds=True,pass_fds=(queue,task,gpu_lease),start_new_session=True)
   write(folder/(variant+'_PID.json'),dict(pid=child.pid,GPU=gpu,GPU_uuid=uuid,command=command,stdout=str(stdout)))
   status(ds,'RUNNING',GPU=gpu,GPU_uuid=uuid,train_pid=child.pid,current=variant,completed=completed,pending=pending,run=str(run),log=str(run/'train.log'))
   code=child.wait();write(folder/(variant+'_EXIT.json'),dict(code=code,pid=child.pid))
   if code!=0:raise RuntimeError('Task failed '+name+' exit='+str(code)+'; remaining tasks halted, no retries')
   result=read(run/'result.json');accepted=read(run/'TEST_COMPLETE.json')
   assert result['train_complete'] and result['Dev_best_selected'] and result['Test_accepted'] and result['Test_passes']==1
   assert accepted['one_Test_pass'] and accepted['result_sha256']==sha(run/'result.json')
   assert accepted['checkpoint_sha256']==result['checkpoint_sha256']==sha(run/'best.pt')
   completed.append(variant);child=None;os.close(gpu_lease);gpu_lease=None;os.close(task);task=None
  status(ds,'COMPLETED',completed=completed,pending=[],new_tasks=len(completed))
  write(folder/'QUEUE_COMPLETE.json',dict(completed=completed,no_more_tasks=True))
 except BaseException as error:
  write(folder/'FAILURE.json',dict(error=repr(error),traceback=traceback.format_exc(),current=current,
   child_pid=child.pid if child else None,completed=completed,pending=pending,no_retry=True,no_kill=True))
  status(ds,'FAILED',completed=completed,pending=pending,current=current,error=repr(error))
  raise
 finally:
  for fd in [gpu_lease,task,queue]:
   if fd is not None:os.close(fd)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['ml-1m','beauty'],required=True);main(p.parse_args().dataset)
