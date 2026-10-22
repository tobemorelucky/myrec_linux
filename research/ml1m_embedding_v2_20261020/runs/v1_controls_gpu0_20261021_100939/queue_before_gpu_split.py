"""One GPU0 queue, A -> B -> C, original Trainer/Dev-only evaluation."""
import argparse,datetime,hashlib,json,os,re,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path("/home/yyx/hzgcode/pom2.0");WORK=ROOT/"research/ml1m_embedding_v2_20261020"
ENTRY=WORK/"ops/v1_controls_entry_20261021.py";PYTHON="/home/yyx/anaconda3/envs/hzg_py10/bin/python"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 assert Path(p).resolve().is_relative_to(WORK)
 with Path(p).open("x") as f:json.dump(x,f,indent=2,allow_nan=False)
def now():return datetime.datetime.now().astimezone().isoformat()
def idle():
 pid=subprocess.check_output(["nvidia-smi","--id=0","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
 mem=int(subprocess.check_output(["nvidia-smi","--id=0","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True).strip())
 return not pid and mem<512,dict(compute_pids=pid,memory_used_MiB=mem)
def run(cmd,out,limit):
 with Path(out).open("xb") as f:
  child=subprocess.Popen(["timeout","--signal=TERM","--kill-after=60",str(limit)]+cmd,cwd=ROOT,
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="0",
             OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4"),
    stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
  save(Path(out).with_suffix(".pid.json"),dict(pid=child.pid,command=cmd,start=now(),max_seconds=limit))
  rc=child.wait()
 return rc
def main():
 p=argparse.ArgumentParser();p.add_argument("--queue",required=True);a=p.parse_args()
 q=Path(a.queue).resolve();assert q.is_relative_to(WORK);info=json.loads((q/"queue.json").read_text())
 save(q/"QUEUE_RUNNING.json",dict(pid=os.getpid(),started=now(),gpu=0,order=["A","B","C"]))
 results=[];started=time.monotonic()
 for job in info["jobs"]:
  out=Path(job["output_dir"]);smoke=out/"SMOKE_COMPLETE.json"
  if not smoke.is_file():
   result=dict(label=job["label"],status="BLOCKED_PREFLIGHT",reason="No passed smoke",test_called=False)
   save(out/"BLOCKED.json",result);results.append(result);continue
  passed=json.loads(smoke.read_text());assert passed["passed"] and passed["embedding_sha256"]==job["embedding_sha256"]
  for name,digest in json.loads((q/"SOURCES.json").read_text())["files"].items():assert sha(ROOT/name)==digest,"Source changed after validation: "+name
  okay,gpu=idle()
  if not okay:
   save(q/"QUEUE_BLOCKED_GPU.json",dict(time=now(),next_job=job["label"],gpu=gpu,finished=results));return
  save(out/"RUNNING.json",dict(queue_pid=os.getpid(),started=now(),gpu=0,job_sha256=sha(out/"job.json"),
     entry_sha256=sha(ENTRY),parameter_count=passed["parameter_count"],embedding_sha256=job["embedding_sha256"]))
  begin=time.monotonic()
  rc=run([PYTHON,"-u","-B",str(ENTRY),"--job",str(out/"job.json"),"--mode","train"],out/"train.out",job["max_wall_seconds"])
  train_seconds=time.monotonic()-begin
  text=(out/"train.log").read_text(errors="replace") if (out/"train.log").exists() else ""
  outer=(out/"train.out").read_text(errors="replace")
  result=dict(label=job["label"],name=job["name"],loss=job["loss"],exit_code=rc,training_seconds=train_seconds,
    parameter_count=passed["parameter_count"],embedding_sha256=job["embedding_sha256"],end=now(),test_called=False)
  assert "Test After Training:" not in text
  epochs=re.findall(r"Epoch\s+(\d+)\s+loss=([\d.eE+-]+)",text)
  best=re.findall(r"Best Iter\(dev\)=\s*(\d+)\s+dev=\((.*?)\)",text)
  valid=rc==0 and bool(best) and " END: " in text and (out/"best.pt").is_file() and not re.search(r"loss=(?:nan|inf)",text,re.I)
  if valid:
   result.update(best_epoch=int(best[-1][0]),end_epoch=int(epochs[-1][0]),best_train_log_metrics=best[-1][1],
                 checkpoint_sha256=sha(out/"best.pt"),parameter_count=passed["parameter_count"])
   ev=run([PYTHON,"-u","-B",str(ENTRY),"--job",str(out/"job.json"),"--mode","dev"],out/"dev_full.out",600)
   if ev==0 and (out/"BEST_DEV_FULL.json").is_file():
    full=json.loads((out/"BEST_DEV_FULL.json").read_text())
    assert full["checkpoint_sha256"]==result["checkpoint_sha256"]
    result.update(status="COMPLETE",Dev=full["metrics"])
   else:result.update(status="FAILED_DEV",dev_exit_code=ev)
  else:result.update(status="FAILED",reason="budget timeout" if rc==124 else "original training failed or incomplete",error_tail=outer[-3000:])
  save(out/(result["status"]+".json"),result);save(out/"exit.json",result);results.append(result)
  if "out of memory" in outer.lower():
   save(q/"QUEUE_STOPPED_OOM.json",dict(time=now(),results=results));break
 # Read-only summary of original completed ASPCF run; no retraining.
 baseline=Path(info["baseline_dir"])
 if (baseline/"COMPLETE.json").exists():
  b=json.loads((baseline/"COMPLETE.json").read_text())
  result=dict(label="ASPCF_reference",status="COMPLETE",loss="filtered Batch Softmax + relation .01",training_seconds=b["elapsed_seconds"],checkpoint_sha256=b["checkpoint_sha256"],best_line=b["best_line"],embedding_sha256=info["jobs"][0]["embedding_sha256"])
  # Reuse original model/Runner directly for its best-Dev six metrics, GPU0 only.
  dst=q/"ASPCF_reference";dst.mkdir()
  cfg=json.loads((baseline/"config.json").read_text());cfg.update(gpu="0")
  job=dict(info["jobs"][0],label="R",name="ASPCF reference",config=cfg,output_dir=str(dst))
  save(dst/"job.json",job)
  okay,gpu=idle()
  if okay:
   rc=run([PYTHON,"-u","-B",str(ENTRY),"--job",str(dst/"job.json"),"--mode","dev"],dst/"dev_full.out",600)
   if rc==0:
    f=json.loads((dst/"BEST_DEV_FULL.json").read_text());result.update(Dev=f["metrics"],parameter_count=f["parameters"])
   else:result.update(status="FAILED_DEV",dev_exit_code=rc)
  results.append(result)
 save(q/"RESULTS.json",dict(finished=now(),elapsed_seconds=time.monotonic()-started,results=results,test_called=False))
 save(q/"QUEUE_FINISHED.json",dict(finished=now(),statuses={r["label"]:r["status"] for r in results},test_called=False))
if __name__=="__main__":
 try:main()
 except BaseException:
  traceback.print_exc();raise
