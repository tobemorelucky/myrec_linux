"""Fixed GPU0 one-shot launch; lock lifecycle ends on every exit; original receiver reused."""
import pathlib,os,sys,json,datetime,importlib.util,fcntl,traceback
R=pathlib.Path('/home/yyx/hzgcode/pom2.0');W=R/'research/ml1m_embedding_v2_20261020';O=W/'ops/recovery_v3_hsdir_20261022';C=W/'runs/cross_v3_20261021/p920';U=W/'runs/a1_history_bpr_20261021_night'
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2)
def now():return datetime.datetime.now().astimezone().isoformat()
def main():
 spec=importlib.util.spec_from_file_location('original_V3_receiver',W/'ops/cross_v3_20261021/p920/receive_V3.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 assert not (C/'V3_RECEIVER_STARTED.json').exists(),'Never restart existing V3 run'
 lock=(U/'gpu0.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  assert m.idle(0),'GPU0 busy, refuse preemption'
  save(O/'V3_GPU0_LEASE.json',dict(pid=os.getpid(),state='preparing',time=now(),gpu=0,release='POSIX lock closes on any exit; no persistent logical reservation'))
  m.main()
 except BaseException as e:
  p=C/'V3_RECEIVER_FAILED.json'
  if not p.exists():save(p,dict(error=repr(e),traceback=traceback.format_exc(),time=now(),no_retry=True))
  raise
 finally:
  fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
  save(O/'V3_GPU0_RELEASED.json',dict(time=now(),released=True,no_persistent_reservation=True))
  import subprocess
  subprocess.run([sys.executable,'-B',str(O/'consolidate.py')],cwd=R,timeout=60)
if __name__=='__main__':main()
