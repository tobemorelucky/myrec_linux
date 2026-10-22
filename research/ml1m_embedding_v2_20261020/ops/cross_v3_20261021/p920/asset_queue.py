"""Explicitly started, project-local asset queue. Never posts chat messages; never preempts."""
import pathlib,os,sys,json,time,subprocess,argparse,datetime
ROOT=pathlib.Path('/home/yyx/hzgcode/pom2.0');W=ROOT/'research/ml1m_embedding_v2_20261020';RUNS=W/'runs/cross_v3_20261021/p920'
SELF=W/'ops/cross_v3_20261021/p920/run_cross_aspcf.py';PY='/home/yyx/anaconda3/envs/hzg_py10/bin/python'
INCOMING={'A1':'A1_oldlora_rich512','B0':'B0_v2lora_base'}  # V3 explicitly excluded pending independent numerical gate
def now():return datetime.datetime.now().astimezone().isoformat()
def idle(gpu):
 rows=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=uuid,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
 uuid,util,mem=[x.strip() for x in rows]
 procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 return int(util)<=5 and int(mem)<=512 and not any(x.split(',')[0].strip()==uuid for x in procs.splitlines())
def main():
 p=argparse.ArgumentParser();p.add_argument('--queue-name',required=True);p.add_argument('--wait-hours',type=float,default=24);a=p.parse_args()
 assert 0<a.wait_hours<=48 and __import__('re').fullmatch('[A-Za-z0-9_-]+',a.queue_name)
 out=RUNS/('queue_'+a.queue_name);assert out.resolve()==out and not out.exists();out.mkdir()
 with (out/'STARTED.json').open('x') as f:json.dump(dict(time=now(),pid=os.getpid(),wait_hours=a.wait_hours,variants=INCOMING),f,indent=2)
 def event(x):
  with (out/'EVENTS.jsonl').open('a') as f:f.write(json.dumps(dict(time=now(),**x))+'\n')
 states={v:'waiting' for v in INCOMING};children={};launched={};end=time.monotonic()+a.wait_hours*3600
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
 while any(v=='waiting' for v in states.values()) or children:
  for v,child in list(children.items()):
   rc=child.poll()
   if rc is not None:states[v]='complete' if rc==0 else 'failed';event(dict(variant=v,exit_code=rc,state=states[v]));del children[v]
  for v in INCOMING:
   if states[v]!='waiting':continue
   if time.monotonic()>=end:states[v]='expired_without_start';event(dict(variant=v,state=states[v]));continue
   ready=RUNS/'incoming'/INCOMING[v]/'READY.json'
   if not ready.exists():continue
   try:
    assert ready.resolve()==ready
    r=json.loads(ready.read_text());assert r['variant']==v
    gpus=[0] if v=='A1' else [1] if v=='B0' else [0,1]
    reserved={g for vv,g in launched.items() if vv in children}
    gpu=next((g for g in gpus if g not in reserved and idle(g)),None)
    if gpu is None:continue
    time.sleep(3)
    if not idle(gpu):continue
    run_name=r['run_name']
    prep=[PY,'-u','-B',str(SELF),'--mode','prepare','--variant',v,'--gpu',str(gpu),'--run-name',run_name,'--embedding-sha256',r['embedding_sha256'],'--delivery-manifest',str(ready)]
    with (out/(v+'_prepare.out')).open('xb') as f:rc=subprocess.run(prep,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,timeout=180).returncode
    if rc:raise RuntimeError('Asset QA/config prepare failed')
    run=RUNS/run_name
    with (out/(v+'_supervise.out')).open('xb') as f:
     child=subprocess.Popen([PY,'-u','-B',str(SELF),'--run-dir',str(run),'--mode','supervise'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    children[v]=child;launched[v]=gpu;states[v]='running';event(dict(variant=v,state='running',gpu=gpu,pid=child.pid,run_dir=str(run)))
   except BaseException as e:states[v]='failed';event(dict(variant=v,state='failed',error=repr(e)))
  if children or any(x=='waiting' for x in states.values()):time.sleep(30)
 with (out/'COMPLETE.json').open('x') as f:json.dump(dict(time=now(),states=states),f,indent=2)
if __name__=='__main__':main()
