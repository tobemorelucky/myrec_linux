"""V3-only receiver: require genuine short/D800 manifests; no unconditional asset queue."""
import pathlib,json,hashlib,subprocess,time,datetime,os,traceback
R=pathlib.Path('/home/yyx/hzgcode/pom2.0');W=R/'research/ml1m_embedding_v2_20261020';O=W/'runs/cross_v3_20261021/p920';A=O/'incoming/V3';PY='/home/yyx/anaconda3/envs/hzg_py10/bin/python';ENTRY=W/'ops/cross_v3_20261021/p920/run_cross_aspcf.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2)
def idle(i):
 x=subprocess.check_output(['nvidia-smi','--id='+str(i),'--query-gpu=uuid,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',');u,util,m=[s.strip() for s in x]
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 return int(util)<=5 and int(m)<=512 and not any(s.split(',')[0].strip()==u for s in apps.splitlines())
def main():
 ready=json.loads((A/'READY.json').read_text());m=ready['producer_manifest'];short=json.loads((A/'SHORT_ACCEPTED.json').read_text());prod=json.loads((A/'V3_PRODUCTION_COMPLETE.json').read_text());parent=json.loads((A/'PARENT_RELEASED.json').read_text())
 assert ready['variant']=='V3' and ready['lora_family']=='v3' and m['passed'] and m['variant']=='V3'
 assert short['passed'] and short['actual_updates']==12 and short['microbatches']==48 and short['restore']['adapter_tau_optimizer_scheduler_step_rng']
 assert len(short['actual_parameter_changes'])==448 and all(s['delta_norm']>0 for s in short['actual_parameter_changes'])
 assert prod['optimizer_steps']==prod['completed_effective_updates']==800 and parent['approved'] and parent['max_formal_lora_runs']==1
 assert sha(A/'SHORT_ACCEPTED.json')==m['short_proof_sha256']==parent['short_proof_sha256']
 assert sha(A/'V3_PRODUCTION_COMPLETE.json')==m['production_manifest_sha256'] and sha(A/'PARENT_RELEASED.json')==m['parent_release_sha256']
 assert short['config_sha256']==prod['config_sha256']==m['training_config_sha256']
 assert prod['files']['checkpoint-800/adapter_model.bin']==m['adapter_sha256'] and m['student_temperature']==m['teacher_temperature']==.2
 assert sha(A/'rich_pca.pkl')==ready['embedding_sha256']==m['pkl_sha256']
 assert not (O/'V3_RECEIVER_STARTED.json').exists()
 save(O/'V3_RECEIVER_STARTED.json',dict(pid=os.getpid(),time=datetime.datetime.now().astimezone().isoformat(),numerical_short_passed=True,D800_passed=True,one_recommendation_run=True))
 deadline=time.monotonic()+4*3600;gpu=None
 while gpu is None:
  gpu=next((i for i in [0] if idle(i)),None)  # GPU0 reserved for V3; A1 queues cannot claim it until V3 terminal
  if gpu is None:
   if time.monotonic()>deadline:raise TimeoutError('No idle GPU; no preemption or alternate experiment')
   time.sleep(30)
  else:
   time.sleep(3)
   if not idle(gpu):gpu=None
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
 run=O/ready['run_name'];assert not run.exists()
 cmd=[PY,'-u','-B',str(ENTRY),'--mode','prepare','--variant','V3','--gpu',str(gpu),'--run-name',ready['run_name'],'--embedding-sha256',ready['embedding_sha256'],'--delivery-manifest',str(A/'READY.json')]
 with (O/'V3_prepare.out').open('x') as f:rc=subprocess.run(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180).returncode
 assert rc==0,'Asset/config QA failed; no training'
 with (O/'V3_supervise.out').open('x') as f:
  child=subprocess.Popen([PY,'-u','-B',str(ENTRY),'--mode','supervise','--run-dir',str(run)],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  save(O/'V3_RECOMMENDATION_PID.json',dict(pid=child.pid,gpu=gpu,run_dir=str(run),time=datetime.datetime.now().astimezone().isoformat()))
  rc=child.wait()
 save(O/'V3_RECEIVER_EXIT.json',dict(exit_code=rc,no_retry=True));assert rc==0,'Recommendation failed; no retry'
 with (O/'FINAL_SUMMARY.out').open('x') as f:subprocess.run([PY,str(W/'ops/cross_v3_20261021/p920/summarize.py')],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=60,check=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  p=O/'V3_RECEIVER_FAILED.json'
  if not p.exists():save(p,dict(error=repr(e),traceback=traceback.format_exc(),no_retry=True))
  raise
