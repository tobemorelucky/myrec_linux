"""Persistent nonpreemptive ML-1M queue with dataset and physical-GPU flock leases."""
import os,sys,json,time,fcntl,subprocess,csv,io,traceback,hashlib
from pathlib import Path
ROOT=Path("/home/yyx/hzgcode/pom2.0").resolve()
OUT=Path(__file__).resolve().parent
RESOURCE=ROOT/"research/gpu_resource_scheduler_20261022_v1"
JOB="ml1m_A1_dual_seed42_k2"
POLL=60
PYTHON="/home/yyx/anaconda3/envs/hzg_py10/bin/python"

def write(path,value,exclusive=False):
 path=Path(path)
 assert path.parent.resolve().is_relative_to(ROOT)
 text=json.dumps(value,indent=2,allow_nan=False)
 with path.open("x" if exclusive else "w") as f:f.write(text)

def stamp():return time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
def status(state,**values):
 record=dict(state=state,scheduler_pid=os.getpid(),time_utc=stamp(),dataset="ml-1m",**values)
 pending=OUT/".SCHEDULER_STATE.tmp"
 write(pending,record)
 os.replace(pending,OUT/"SCHEDULER_STATE.json")
 print(json.dumps(record),flush=True)

def file_sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(2**20),b""):h.update(b)
 return h.hexdigest()

def observe():
 text=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.free,utilization.gpu",
   "--format=csv,noheader,nounits"],text=True,timeout=15)
 result={}
 for row in csv.reader(io.StringIO(text)):
  if row:result[int(row[0])]=dict(index=int(row[0]),uuid=row[1].strip(),
                                  free_MiB=int(row[2]),utilization=int(row[3]),compute_pids=[])
 apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid",
   "--format=csv,noheader,nounits"],text=True,timeout=15)
 by_uuid={v["uuid"]:v for v in result.values()}
 for row in csv.reader(io.StringIO(apps)):
  if row:
   if len(row)!=2 or not row[0].strip().isdigit():raise RuntimeError("Unrecognized GPU process listing")
   by_uuid[row[1].strip()]["compute_pids"].append(int(row[0]))
 return result

def idle(info):
 # All users' compute PIDs count, regardless of process ownership.
 return not info["compute_pids"] and info["free_MiB"]>=10240 and info["utilization"]<=10

def acquire(path):
 # Never truncate a GPU1 lock: it is a hardlink to Beauty's existing flock inode.
 fd=os.open(path,os.O_RDWR|os.O_CREAT,0o600)
 try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:
  os.close(fd);return None
 return fd

def close(fd):
 if fd is not None:os.close(fd)

def choose_gpu():
 while True:
  inventory=observe()
  for index in (1,):
   if index not in inventory or not idle(inventory[index]):continue
   lease=acquire(RESOURCE/("gpu"+str(index)+".lock"))
   if lease is None:continue
   # A second fresh inventory under the lease avoids transient-idle selection.
   time.sleep(1)
   latest=observe()
   if index in latest and idle(latest[index]):return index,lease,latest[index]["uuid"]
   close(lease)
  status("WAITING_GPU",preferred_GPU=1,poll_seconds=POLL,inventory=inventory)
  time.sleep(POLL)

def validate_success(code,folder):
 if code!=0:raise RuntimeError("Training child failed with exit code "+str(code)+"; see "+str(folder/"stdout.log"))
 p=folder/"result.json"
 if not p.is_file():raise RuntimeError("Child exited without a result; refusing to start next arm")
 result=json.loads(p.read_text())
 if not result.get("train_complete") or not result.get("Dev_best_selected") or result.get("Test_evaluations")!=0:
  raise RuntimeError("Invalid child completion state; queue stopped")

def verify():
 plan=json.loads((OUT/"RUN_PLAN.json").read_text())
 for name,sha in plan["source_sha256"].items():
  if file_sha(OUT/name)!=sha:raise RuntimeError("Source changed: "+name)
 initial=json.loads((OUT/"INITIAL_STATE.json").read_text())
 for name,sha in initial["protected_sha256"].items():
  if file_sha(ROOT/name)!=sha:raise RuntimeError("Protected source or asset changed: "+name)
 check=json.loads((OUT/"CHECK_PASS.json").read_text())
 if not check["checks_passed"]:raise RuntimeError("Preflight not passed")

def proc_identity(pid):
 try:
  raw=Path("/proc")/str(pid)/"stat"
  return raw.read_text().rsplit(")",1)[1].split()[19]
 except FileNotFoundError:return None

def beauty_active():
 adopted=json.loads((RESOURCE/"ADOPTED_BEAUTY.json").read_text())
 if proc_identity(adopted["queue_pid"])==adopted["proc_start_ticks"]:return True
 # Explicitly account for surviving queue children in addition to inherited flock.
 marker=str(ROOT/"research/elector_sil_dual_beauty_20261022_v1").encode()
 for entry in Path("/proc").iterdir():
  if not entry.name.isdigit():continue
  try:
   command=(entry/"cmdline").read_bytes()
   if marker in command and any(token in command for token in (b"train_arm.py",b"queue.sh")):return True
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 return False

def wait_prerequisites():
 while beauty_active():
  status("WAITING_BEAUTY",GPU=1,poll_seconds=POLL)
  time.sleep(POLL)

def main():
 job_lease=None;gpu_lease=None;gpu=None;child=None
 try:
  job_lease=acquire(RESOURCE/(JOB+".lock"))
  if job_lease is None:raise RuntimeError("Duplicate ML-1M scheduler: dataset lease is already held")
  registry=RESOURCE/(JOB+".json")
  if registry.exists():raise RuntimeError("This exact experiment is already registered; no automatic resubmission")
  verify()
  write(registry,dict(queue_pid=os.getpid(),run_root=str(OUT),created_utc=stamp()),True)
  write(OUT/"SCHEDULER_STARTED.json",dict(pid=os.getpid(),time_utc=stamp(),job_lock=True,
         gpu_preference=[1],poll_seconds=POLL,fail_fast=True),True)
  wait_prerequisites()
  status("SELECTING_GPU",preferred_GPU=1)
  gpu,gpu_lease,uuid=choose_gpu()
  for arm in ("B0","C1","M1"):
   verify()
   # Keep the lease for the whole dataset queue. If an unrelated external job
   # appears between arms, wait for it rather than sharing or preempting.
   while True:
    inventory=observe()
    if inventory[gpu]["uuid"]!=uuid:raise RuntimeError("Physical GPU identity changed")
    if idle(inventory[gpu]):break
    status("WAITING_GPU",GPU=gpu,current_arm=arm,poll_seconds=POLL,inventory=inventory)
    time.sleep(POLL)
   folder=OUT/"arms"/arm
   env=os.environ.copy()
   env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE="1",
              OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",
              CUDA_CACHE_PATH=str(OUT/"cuda_cache"),TORCHINDUCTOR_CACHE_DIR=str(OUT/"torch_cache"))
   log=(folder/"stdout.log").open("x")
   child=subprocess.Popen(["nohup",PYTHON,"-u","-B",str(OUT/"train_arm.py"),"--arm",arm],
     cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,
     close_fds=True,pass_fds=(job_lease,gpu_lease),start_new_session=True)
   log.close()
   with (folder/"train.pid").open("x") as f:f.write(str(child.pid)+"\n")
   status("RUNNING",GPU=gpu,GPU_uuid=uuid,current_arm=arm,train_pid=child.pid,
          train_log=str(folder/"train.log"),stdout_log=str(folder/"stdout.log"))
   code=child.wait()
   write(folder/"exit_status.json",dict(code=code,time_utc=stamp(),pid=child.pid),True)
   validate_success(code,folder)
   child=None
  status("COMPLETED",GPU=gpu,Test_evaluations=0)
 except BaseException as exc:
  record=dict(error=repr(exc),traceback=traceback.format_exc(),time_utc=stamp(),
              scheduler_pid=os.getpid(),GPU=gpu,train_pid=child.pid if child else None,
              following_arms_not_started=True,no_automatic_retry_or_config_change=True)
  write(OUT/"SCHEDULER_FAILURE.json",record,True)
  status("FAILED",**{k:record[k] for k in ("error","GPU","train_pid")})
  raise
 finally:
  close(gpu_lease);close(job_lease)

if __name__=="__main__":main()
