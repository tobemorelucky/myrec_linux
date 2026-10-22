"""Detached stage execution and read-only 1800s D monitor. Never auto-chains/restarts."""
import argparse,ast,datetime,hashlib,json,math,os,re,shutil,signal,subprocess,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RENTAL=Path("/root/autodl-tmp/pomrec/research/ml1m_embedding_v2_20261020")
PYTHON="/root/miniconda3/envs/py10/bin/python"
def now():return datetime.datetime.now().astimezone().isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def new_json(p,obj):
    with Path(p).open("x",encoding="utf-8") as f:json.dump(obj,f,indent=2,allow_nan=False)
def append(p,obj):
    with Path(p).open("a",encoding="utf-8") as f:f.write(json.dumps(obj,allow_nan=False)+"\n")
def runtime_sha():
    files={str(p.relative_to(ROOT)):sha(p) for d in ("adapter","configs","scripts","upstream")
           for p in sorted((ROOT/d).rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
    return hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
def cmd_output(cmd):
    try:return subprocess.check_output(cmd,text=True,stderr=subprocess.STDOUT,timeout=10).strip()
    except Exception as e:return str(e)
def progress(log):
    if not log.exists():return {"step":0,"loss":None,"nonfinite_loss":False}
    with log.open("rb") as f:f.seek(max(0,log.stat().st_size-262144));s=f.read().decode(errors="replace")
    steps=[int(x) for x in re.findall(r"(\d+)\s*/\s*1000\b",s)]
    losses=re.findall(r"['\"]loss['\"]\s*:\s*(-?(?:\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|nan|inf))",s,re.I)
    values=[float(x) for x in losses]
    last=values[-1] if values else None
    return {"step":max(steps,default=0),"loss":last if last is None or math.isfinite(last) else str(last),
            "nonfinite_loss":any(not math.isfinite(x) for x in values)}
def health(job,child,previous=None,exit_code=None):
    stamp=time.time();p=progress(job/"stage.log")
    rate=None
    if previous and stamp>previous["unix_time"]:
        rate=(p["step"]-previous["step"])/(stamp-previous["unix_time"])
    memory={}
    for name,path in [("cgroup_limit","/sys/fs/cgroup/memory/memory.limit_in_bytes"),
                      ("cgroup_usage","/sys/fs/cgroup/memory/memory.usage_in_bytes"),
                      ("cgroup2_limit","/sys/fs/cgroup/memory.max"),
                      ("cgroup2_usage","/sys/fs/cgroup/memory.current")]:
        if Path(path).is_file():memory[name]=Path(path).read_text().strip()
    for limit_key,used_key in [("cgroup_limit","cgroup_usage"),("cgroup2_limit","cgroup2_usage")]:
        if str(memory.get(limit_key,"")).isdigit() and str(memory.get(used_key,"")).isdigit():
            memory["container_available_bytes"]=max(0,int(memory[limit_key])-int(memory[used_key]))
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemAvailable:","MemTotal:")):memory[line.split(":")[0]]=line.split(":")[1].strip()
    checkpoints=[{"name":q.name,"trainer_state_exists":(q/"trainer_state.json").exists()}
                 for q in (ROOT/"runs/scft_rich_full1024_seed42").glob("checkpoint-*")]
    return {"time":now(),"unix_time":stamp,"pid":child.pid,"pid_alive":child.poll() is None,
            **p,"recent_steps_per_second":rate,
            "remaining_seconds_estimate":(1000-p["step"])/rate if rate and rate>0 else None,
            "gpu":cmd_output(["nvidia-smi","--query-gpu=index,name,memory.used,memory.total,utilization.gpu","--format=csv,noheader,nounits"]),
            "memory":memory,"disk_free_bytes":shutil.disk_usage(ROOT).free,
            "checkpoints":checkpoints,"exit_code":exit_code}
def validate(stage):
    outputs={"A":"admission_a_old_ml_lora_full1024/SAMPLE_COMPLETE.json",
             "C":"admission_c_rich_full1024/ADMISSION_COMPLETE.json",
             "D":"scft_rich_full1024_seed42/SCFT_COMPLETE.json",
             "B":"v1_rebuilt_raw_full1024/COMPLETE.json"}
    if stage=="E":
        pca=ROOT/"runs/v2_common_pca1536_full1024/manifest.json";q=json.load(pca.open())
        assert q["nearzero_valid_rows_base"]==q["nearzero_valid_rows_rich"]==0
        return {"stage_qa_passed":True,"pca_manifest":str(pca),"sha256":sha(pca)}
    if stage not in outputs:return {"existing_stage_command_succeeded":True}
    marker=ROOT/"runs"/outputs[stage]
    if stage=="C":marker=Path(os.environ["V2_ADMISSION_OUTPUT_DIR"])/"ADMISSION_COMPLETE.json"
    m=json.load(marker.open())
    observed=m.get("code_sha256")
    if stage=="B":observed=json.load((marker.parent/"manifest.json").open())["code"]["code_sha256"]
    if observed!=runtime_sha():raise ValueError("Completion/source SHA mismatch")
    if stage=="A":
        assert m["rows"]==8 and m["quality"]["dim"]==4096
    if stage=="C":
        assert m["optimizer_steps"]==m["actual_optimizer_step_calls"]==3
        assert len(m["microbatches"])==24 and m["adapter_round_trip"] and not m["full_base_saved"]
        assert all(math.isfinite(x["loss_divided_by_accumulation"]) for x in m["microbatches"])
    if stage=="D":
        assert m["optimizer_steps"]==1000
        cp=ROOT/"runs/scft_rich_full1024_seed42/checkpoint-1000"
        for n in ("adapter_model.bin","adapter_config.json","v2_extra_trainable.pt","optimizer.pt",
                  "scheduler.pt","trainer_state.json","rng_state.pth"):
            assert (cp/n).is_file(),"Missing "+n
        assert json.load((cp/"trainer_state.json").open())["global_step"]==1000
        import torch
        for n in ("adapter_model.bin","v2_extra_trainable.pt"):
            values=torch.load(cp/n,map_location="cpu",weights_only=True)
            assert values and all(torch.isfinite(x).all() for x in values.values()),"Nonfinite "+n
        assert not list(cp.glob("model-*.safetensors")) and not (cp/"pytorch_model.bin").exists()
        for entry in m.get("log_history",[]):
            if "loss" in entry:assert math.isfinite(entry["loss"])
    if stage=="B":
        pca=ROOT/"runs/v1_rebuilt_pca1536_full1024/manifest.json"
        q=json.load(pca.open())
        assert q["nearzero_valid_rows_base"]==0 and q["quality_base"]["rows"]==3706
    return {"marker":str(marker),"sha256":sha(marker),"stage_qa_passed":True}
def supervise(stage,job):
    assert ROOT.resolve()==RENTAL and not job.is_symlink() and ROOT in job.resolve().parents
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1",V2_PROFILE="full1024")
    if stage=="C":
        target=ROOT/"runs"/("admission_c_full1024_"+job.name)
        env["V2_ADMISSION_OUTPUT_DIR"]=str(target)
        os.environ["V2_ADMISSION_OUTPUT_DIR"]=str(target)
    if stage=="D":
        pointer=json.load((ROOT/"ops_runs/C_PASS_POINTER.json").open())
        if not (Path(pointer["job"])/"COMPLETE.json").is_file():raise ValueError("C job not validated")
        if pointer["runtime_code_sha256"]!=runtime_sha():raise ValueError("Code changed since C")
        env["V2_ADMISSION_PROOF"]=pointer["proof"]
    begin=time.monotonic();start=now()
    commands=[["bash","scripts/gpu_stages.sh",stage]]
    quality=[PYTHON,"-B","adapter/quality_pca.py"]
    mapping="/root/autodl-tmp/pomrec/data/ml-1m/item2id.json"
    train="/root/autodl-tmp/pomrec/data/ml-1m/train.csv"
    if stage=="B":
        commands.append(quality+["--base-dir","runs/v1_rebuilt_raw_full1024",
            "--mapping",mapping,"--train",train,"--out","runs/v1_rebuilt_pca1536_full1024","--execute"])
    if stage=="E":
        commands=[[PYTHON,"-B","adapter/stage_gate.py","--stage","E","--profile","full1024"],
            [PYTHON,"-B","adapter/run_encode.py","--config","configs/encode_base_full1024.json","--execute"],
            [PYTHON,"-B","adapter/run_encode.py","--config","configs/encode_rich_full1024.json","--execute"],
            quality+["--base-dir","runs/encode_base_full1024","--rich-dir","runs/encode_rich_full1024",
                "--mapping",mapping,"--train",train,"--out","runs/v2_common_pca1536_full1024","--execute"]]
    with (job/"stage.log").open("x",buffering=1) as log:
        child=subprocess.Popen(commands[0],cwd=ROOT,env=env,
              stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        new_json(job/"process.json",{"supervisor_pid":os.getpid(),"child_pid":child.pid,"start_time":start,
                 "stage":stage,"command":["bash","scripts/gpu_stages.sh",stage],"runtime_code_sha256":runtime_sha(),
                 "scft_config_sha256":sha(ROOT/"configs/scft_rich_full1024_frozen.json")})
        stop=threading.Event();last=[None]
        def take(code=None):
            sample=health(job,child,last[0],code);append(job/"monitor.jsonl",sample);last[0]=sample
        def monitor():
            take()
            while not stop.wait(1800):take()
        thread=None
        if stage=="D":
            thread=threading.Thread(target=monitor,daemon=True);thread.start()
        rc=child.wait()
        for extra in commands[1:]:
            if rc:break
            child=subprocess.Popen(extra,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                  stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            append(job/"subprocesses.jsonl",{"pid":child.pid,"command":extra,"time":now()})
            rc=child.wait()
        stop.set()
        if thread:thread.join(timeout=15)
        take(rc)
    result={"stage":stage,"start_time":start,"end_time":now(),"elapsed_seconds":time.monotonic()-begin,
            "exit_code":rc,"supervisor_pid":os.getpid(),"child_pid":child.pid}
    try:
        if rc:raise RuntimeError("Stage process failed with exit code "+str(rc))
        result["validation"]=validate(stage)
        result["qa_passed"]=True
        new_json(job/"COMPLETE.json",result)
        if stage=="C":
            new_json(ROOT/"ops_runs/C_PASS_POINTER.json",{"job":str(job),
                "proof":result["validation"]["marker"],"runtime_code_sha256":runtime_sha()})
    except Exception as e:
        result["validation_error"]=str(e);new_json(job/"FAILED.json",result)
    result["supervisor_exit_code"]=0 if "validation_error" not in result else 1
    append(job/"monitor.jsonl",{"event":"FINAL_QA","time":now(),"stage":stage,
        "pid_alive":False,"stage_exit_code":rc,"supervisor_exit_code":result["supervisor_exit_code"],
        "qa_passed":result.get("qa_passed",False),"error":result.get("validation_error")})
    new_json(job/"exit.json",result)
    return result["supervisor_exit_code"]
def main():
    p=argparse.ArgumentParser();p.add_argument("action",choices=["launch","supervise","selftest"])
    p.add_argument("--stage",choices=["A","B","C","D","E","F"]);p.add_argument("--tag")
    p.add_argument("--approved-wall-seconds",type=int)
    a=p.parse_args()
    if a.action=="selftest":
        assert ROOT.name=="ml1m_embedding_v2_20261020"
        assert progress(ROOT/"ops/nonexistent")["step"]==0
        print("Operational static selftest passed; no stage executed");return
    if ROOT.resolve()!=RENTAL:raise RuntimeError("7B stages rental only")
    if not a.tag or not re.fullmatch(r"[A-Za-z0-9_-]+",a.tag):raise ValueError("Safe unique tag required")
    if not a.stage:raise ValueError("One explicit stage required")
    if a.stage=="D":
        if not a.approved_wall_seconds or a.approved_wall_seconds<1:raise ValueError("New approved paid budget required")
        os.environ["V2_SCFT_APPROVED"]="YES";os.environ["V2_WALL_SECONDS_APPROVED"]=str(a.approved_wall_seconds)
    job=ROOT/"ops_runs"/a.tag
    if a.action=="supervise":raise SystemExit(supervise(a.stage,job))
    job.mkdir(parents=True,exist_ok=False)
    args=[PYTHON,"-u","-B",str(Path(__file__).resolve()),"supervise","--stage",a.stage,"--tag",a.tag]
    if a.approved_wall_seconds:args+=["--approved-wall-seconds",str(a.approved_wall_seconds)]
    with (job/"supervisor.log").open("x") as log:
        process=subprocess.Popen(args,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
              stderr=subprocess.STDOUT,start_new_session=True)
    new_json(job/"launch.json",{"pid":process.pid,"launch_time":now(),"stage":a.stage,
             "ops_script_sha256":sha(__file__),"command":args})
    print(json.dumps({"pid":process.pid,"stage":a.stage,"log":str(job/"stage.log"),
                     "monitor":str(job/"monitor.jsonl"),"exit":str(job/"exit.json")},indent=2))
if __name__=="__main__":main()
