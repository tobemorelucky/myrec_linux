#!/usr/bin/env bash
# Prepared only. Default is dry-run; no Test command exists.
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/yyx/anaconda3/etc/profile.d/conda.sh
conda activate hzg_py10
export PYTHONDONTWRITEBYTECODE=1
python -B - "$@" <<'PY'
import argparse,hashlib,json,os,re,shlex,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("dataset",choices=["beauty","ml-1m"])
p.add_argument("gpu",choices=["1"])
p.add_argument("mode",choices=["C0","C1"])
p.add_argument("--execute",action="store_true")
p.add_argument("--stage",choices=["all","prepare","train"],default="all")
a=p.parse_args()
root=Path.cwd().resolve()
source=root/"new_log/aspcf_refine_controls"/a.dataset/"user_refine_k4_h20_seed42_20261013T020946Z_attribution/summary.json"
s=json.loads(source.read_text());base=(root/s["checkpoint"]).resolve()
assert base.is_relative_to(root) and base.is_file()
assert hashlib.sha256(base.read_bytes()).hexdigest()==s["checkpoint_sha256"]
args=list(s["common_command"])
def value(k):return args[args.index(k)+1]
def setarg(k,v):
    if k in args:args[args.index(k)+1]=str(v)
    else:args.extend([k,str(v)])
assert value("--K")=="4" and value("--history_max")=="20" and value("--batch_size")=="1024"
assert value("--random_seed")=="42" and value("--sampled_softmax_temperature")=="1"
rid=os.environ.get("RUN_ID",time.strftime("%Y%m%dT%H%M%SZ",time.gmtime()))
if not re.fullmatch("[A-Za-z0-9_.-]+",rid):raise ValueError("Unsafe run ID")
name=a.mode.lower()+"_k"+value("--K")+"_h20_seed42_"+rid
log=root/"new_log/relative_semantic_continuation"/a.dataset/name
model=root/"new_model/relative_semantic_continuation"/a.dataset/name
setarg("--model_name","LLMMIRecASPCFRelativeSemantic")
setarg("--gpu",a.gpu);setarg("--model_path",model/"best.pt")
setarg("--epoch",20);setarg("--early_stop",0) # fixed matched update budget, not final full-training protocol
setarg("--topk","5,10,20");setarg("--dev_only",1)
setarg("--relative_semantic_enabled",int(a.mode=="C1"))
setarg("--relative_semantic_base_checkpoint",base)
def command(train,load,name):
    return ["python","-B","tools/run_relative_semantic_matched.py","main","--"]+args+["--train",str(train),"--load",str(load),"--log_file",str(log/name)]
commands=[command(0,0,"initial_dev.log"),command(1,0,"train.log"),command(0,1,"best_dev.log")]
for commandline in commands:print(shlex.join(commandline),flush=True)
if not a.execute:
    print("DRY RUN ONLY: no training/evaluation/output directories; both controls reset Adam identically.")
    raise SystemExit()
config=dict(model_version="relative-semantic-userrefine-v1",mode=a.mode,K=int(value("--K")),
            seed=42,continuation_epochs=20,early_stop=0,optimizer_state="fresh Adam in both arms",
            base_checkpoint=str(base),base_sha256=s["checkpoint_sha256"],
            base_historical_dev=s["dev_full"],commands=commands,test_allowed=False,
            initialization="exact original A tensors; one alpha=0 scalar; no optimizer state in historical checkpoint")
config["source_sha256"]={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in [
 "models/sequential/LLMMIRecASPCFRelativeSemantic.py","models/sequential/LLMMIRecASPCFRefineControls.py",
 "models/sequential/LLMMIRecASPCFSampledSoftmax.py","models/sequential/llmmi_components.py",
 "models/sequential/LLMMIRecASPCF.py","helpers/BaseRunner.py","main.py",
 "tools/run_relative_semantic_matched.py","tools/test_llmmirec_relative_semantic.py",
 "new_bash/run_llmmirec_relative_semantic_continuation.sh"]}
def metrics(text):
    line=[x for x in text.splitlines() if "Dev Evaluation:" in x][-1]
    return {k:float(v) for k,v in re.findall(r"((?:HR|NDCG)@(?:5|10|20)):([0-9.]+)",line)}
for rel in ["train.csv","dev.csv","SeqReader.pkl","item2id.json","handled/llm_table_pca1536.pkl"]:
    f=root/"data"/a.dataset/rel
    digest=hashlib.sha256()
    with f.open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""):digest.update(block)
    config.setdefault("asset_sha256",{})[str(f.relative_to(root))]=digest.hexdigest()
def run(commandline):
    out=Path(commandline[-1]+".out");t=time.time()
    with out.open("xb") as f:subprocess.run(commandline,check=True,stdout=f,stderr=subprocess.STDOUT,cwd=root)
    return time.time()-t
if a.stage in ("all","prepare"):
    for d in [log,model]:
        assert d.resolve().is_relative_to(root) and not d.exists(),"Output collision/outside project"
        d.mkdir(parents=True)
    (log/"config.json").write_text(json.dumps(config,indent=2))
    initial_seconds=run(commands[0])
    info=json.loads((log/"initial_dev.log.runtime.json").read_text())
    assert info["success"] and len(info["evaluations"])==1
    dev0=info["evaluations"][0]
    expected={k:float(v) for k,v in re.findall(r"((?:HR|NDCG)@(?:5|10|20)):([0-9.]+)",s["dev_full"])}
    assert len(expected)==6 and all(float(format(dev0[k],".4f"))==v for k,v in expected.items()),(dev0,expected)
    assert hashlib.sha256(base.read_bytes()).hexdigest()==s["checkpoint_sha256"]
    (log/"epoch0_gate.json").write_text(json.dumps(dict(status="ready",dev=dev0,
         expected_historical=expected,seconds=initial_seconds,ready_unix=time.time()),indent=2))
    if a.stage=="prepare":
        print("EPOCH0 VERIFIED "+a.dataset+" "+a.mode,flush=True)
        raise SystemExit()
else:
    assert log.resolve().is_relative_to(root) and model.resolve().is_relative_to(root)
    assert json.loads((log/"config.json").read_text())==config,"Prepared config/source/assets changed"
gate=json.loads((log/"epoch0_gate.json").read_text())
assert gate["status"]=="ready"
assert not (log/"train.log").exists() and not (model/"best.pt").exists(),"Never overwrite a training attempt"
train_seconds=run(commands[1])
runtime=json.loads((log/"train.log.runtime.json").read_text())
assert runtime["success"] and len(runtime["epochs"])==20
checkpoint=model/"best.pt";digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
dev_seconds=run(commands[2])
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest,"Dev changed checkpoint"
assert hashlib.sha256(base.read_bytes()).hexdigest()==s["checkpoint_sha256"]
import torch
weights=torch.load(checkpoint,map_location="cpu",weights_only=True)
initial=(log/"initial_dev.log").read_text();dev=(log/"best_dev.log").read_text();train=(log/"train.log").read_text()
assert "Test After Training" not in initial+dev+train
summary=dict(status="complete",config=config,initial_dev=gate["dev"],best_dev=metrics(dev),
             checkpoint_sha256=digest,checkpoint_unchanged=True,
             alpha_at_best=float(weights["extractor.alpha"]),alpha_final=runtime["alpha_final"],
             phase_seconds=[gate["seconds"],train_seconds,dev_seconds],
             total_active_seconds=gate["seconds"]+train_seconds+dev_seconds,
             runtime=runtime,test_executed=False,
             best_reported_epoch=int(re.findall(r"Best Iter\(dev\)=\s*(\d+)",train)[-1]),
             checkpoint_saved_epoch=int(re.findall(r"Epoch\s+(\d+).* \*$",train,re.M)[-1]),
             end_epoch=int(re.findall(r"Epoch\s+(\d+)\s+loss=",train)[-1]))
assert summary["end_epoch"]==20
(log/"summary.json").write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ("config","runtime")}))

PY
