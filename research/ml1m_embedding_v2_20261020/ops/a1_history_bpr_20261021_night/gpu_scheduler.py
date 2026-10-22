"""Two serial A1 queues; GPU0 reserved for existing V3 until its receiver exits."""
import pathlib,json,subprocess,os,time,fcntl,datetime,csv,traceback,argparse
R=pathlib.Path('/home/yyx/hzgcode/pom2.0');W=R/'research/ml1m_embedding_v2_20261020'
O=W/'ops/a1_history_bpr_20261021_night';U=W/'runs/a1_history_bpr_20261021_night';V=W/'runs/cross_v3_20261021/p920'
PY='/home/yyx/anaconda3/envs/hzg_py10/bin/python'
def now():return datetime.datetime.now().astimezone().isoformat()
def put(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));os.replace(t,p)
def idle(i):
 x=subprocess.check_output(['nvidia-smi','--id='+str(i),'--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',');u,m,z=[s.strip() for s in x]
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 return int(m)<512 and int(z)<=5 and not any(s.split(',')[0].strip()==u for s in apps.splitlines())
def v3_done():
 if (V/'V3_RECEIVER_EXIT.json').exists():return True
 if (V/'V3_RECEIVER_FAILED.json').exists():return True
 return False
def summary(gpu):
 fields=['experiment','model','mechanism','Loss','K','lr','batch','embedding_sha256','teacher_sha256','best_epoch','end_epoch','training_seconds','parameter_count','status','log_path']+[phase+'_'+m for phase in ['Dev','Test'] for m in ['HR@5','NDCG@5','HR@10','NDCG@10','HR@20','NDCG@20']]
 records=[]
 for p in sorted(U.glob('E*/job.json')):
  j=json.loads(p.read_text());c=j['config'];d=p.parent
  s=dict(experiment=d.name,model=c['model_name'],mechanism=j['mechanism'],Loss=j['loss'],K=c['K'],lr=c['lr'],batch=c['batch_size'],embedding_sha256=j['embedding_sha256'],teacher_sha256=j.get('teacher_sha256',''),status='queued',log_path=str(d/'train.log'))
  if (d/'COMPLETE.json').exists():
   z=json.loads((d/'COMPLETE.json').read_text());s.update({k:z[k] for k in ['best_epoch','end_epoch','training_seconds','parameter_count']});s['status']='Train_Dev_Test_complete'
   for phase in ['Dev','Test']:
    s.update({phase+'_'+k:v for k,v in z[phase].items()})
  elif (d/'FAILED.json').exists():s['status']='failed_no_retry'
  elif (d/'TRAIN_PID.json').exists():s['status']='running_or_in_acceptance'
  records.append(s)
 p=U/('RESULTS_gpu'+str(gpu)+'.csv');t=p.with_suffix('.tmp')
 with t.open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
 os.replace(t,p)
 lines=['# A1 BPR exploratory queue results','', 'V3 has priority; no Test-based tuning; all Test uses best-Dev once.','', '|experiment|status|Dev N5|Test N5|best/end|','|---|---|---:|---:|---|']
 for s in records:lines.append('|'+s['experiment']+'|'+s['status']+'|'+str(s.get('Dev_NDCG@5',''))+'|'+str(s.get('Test_NDCG@5',''))+'|'+str(s.get('best_epoch',''))+'/'+str(s.get('end_epoch',''))+'|')
 t=U/('RESULTS_gpu'+str(gpu)+'.md.tmp');t.write_text('\n'.join(lines));os.replace(t,U/('RESULTS_gpu'+str(gpu)+'.md'))
def main():
 a=argparse.ArgumentParser();a.add_argument('--gpu',type=int,choices=[0,1],required=True);gpu=a.parse_args().gpu
 q=U/('queue_gpu'+str(gpu));q.mkdir(exist_ok=False)
 put(q/'PID.json',dict(pid=os.getpid(),time=now(),gpu=gpu))
 ids=['E5','E6'] if gpu==0 else ['E1','E3','E4','E2']
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
 end=time.monotonic()+18*3600
 for label in ids:
  job=U/label/'job.json';state='waiting_V3_GPU0_reservation' if gpu==0 and not v3_done() else 'waiting_GPU'
  put(q/'STATUS.json',dict(state=state,next=label,time=now()))
  while True:
   if time.monotonic()>end:put(q/'HOLD.json',dict(reason='queue waiting exceeded18h',time=now()));return
   if gpu==0 and not v3_done():time.sleep(30);continue
   lock=(U/('gpu'+str(gpu)+'.lock')).open('a+')
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:lock.close();time.sleep(30);continue
   if not idle(gpu):lock.close();time.sleep(30);continue
   time.sleep(2)
   if not idle(gpu):lock.close();time.sleep(30);continue
   break
  try:
   out=job.parent
   if (out/'COMPLETE.json').exists() or (out/'FAILED.json').exists():raise RuntimeError('existing terminal, refuse rerun')
   with (out/'supervisor.out').open('xb') as f:
    child=subprocess.Popen([PY,'-u','-B',str(O/'entry.py'),'--job',str(job),'--mode','supervise'],cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    put(q/'STATUS.json',dict(state='running',experiment=label,supervisor_pid=child.pid,gpu=gpu,time=now()))
    rc=child.wait()
   with (q/'EVENTS.jsonl').open('a') as f:f.write(json.dumps(dict(experiment=label,exit_code=rc,time=now(),no_retry=True))+'\n')
  except BaseException as e:
   with (q/'EVENTS.jsonl').open('a') as f:f.write(json.dumps(dict(experiment=label,error=repr(e),time=now(),no_retry=True))+'\n')
  finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close();summary(gpu)
 put(q/'COMPLETE.json',dict(time=now(),experiments=ids));summary(gpu)
if __name__=='__main__':main()
