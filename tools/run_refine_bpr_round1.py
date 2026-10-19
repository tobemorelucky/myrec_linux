"""Exactly four authorized from-scratch A/Full BPR Train/Dev runs, no Test."""
import os, sys, json, time, hashlib, pickle, random, argparse, runpy
import subprocess, re, traceback, logging
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCES = ["models/sequential/LLMMIRecASPCFRefineBPR.py",
 "models/sequential/LLMMIRecASPCF.py",
 "models/sequential/LLMMIRecASPCFDualViewRefine.py",
 "models/sequential/LLMMIRecASPCFRefineControls.py",
 "models/sequential/llmmi_components.py", "models/BaseModel.py",
 "helpers/BaseRunner.py", "helpers/SeqReader.py", "helpers/BaseReader.py",
 "main.py", "tools/run_refine_bpr_round1.py", "tools/test_aspcf_refine_bpr.py"]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()
def bh(b):return hashlib.sha256(b).hexdigest()
def write_new(p, obj):
    p=Path(p);assert p.resolve().is_relative_to(ROOT) and not p.exists(), str(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as f:json.dump(obj,f,indent=2,allow_nan=False)
def parser():
    from helpers.SeqReader import SeqReader
    from helpers.BaseRunner import BaseRunner
    from models.sequential.LLMMIRecASPCFRefineBPR import LLMMIRecASPCFRefineBPR
    globals_=runpy.run_path(str(ROOT/"main.py"),run_name="prepare_only")
    p=globals_["parse_global_args"](argparse.ArgumentParser())
    p=SeqReader.parse_data_args(p);p=BaseRunner.parse_runner_args(p)
    return LLMMIRecASPCFRefineBPR.parse_model_args(p)
def arguments(ds,mode,log,weight):
    p=parser();names={a.dest for a in p._actions};old={}
    hist=ROOT/"new_log/llmmirec_aspcf_phase2"/ds/"seed42/LLMMIRecASPCF_seed42.log"
    for line in hist.read_text().splitlines():
        if "|" in line:
            parts=line.split("|");k,v=parts[0].strip(),parts[1].strip()
            if k in names and "..." not in v:old[k]=v
    assert old["K"]=="4" and old["lr"]==("0.004" if ds=="beauty" else "0.001")
    assert old["l2"]=="1e-06" and old["optimizer"]=="Adam" and old["batch_size"]=="1024"
    values=dict(old)
    values.update(dataset=ds,refine_mode=mode,path=str(ROOT/"data"),gpu="1",
        random_seed="42",history_max="20",num_neg="1",lambda_relation=".01",
        test_all="0",test_epoch="-1",dev_only="1",load="0",train="1",regenerate="0",
        topk="5,10,20",metric="NDCG,HR",pin_memory="1",num_workers="5",
        model_path=str(weight),log_file=str(log),
        llm_emb_path=str(ROOT/"data"/ds/"handled/llm_table_pca1536.pkl"))
    args=p.parse_args([x for k,v in values.items() for x in ("--"+k,str(v))])
    return args, dict(path=str(hist),sha256=sha(hist),logged_values=old)
def rng():
    import torch,numpy as np
    return (torch.get_rng_state().clone(),[x.clone() for x in torch.cuda.get_rng_state_all()],
            np.random.get_state(),random.getstate())
def restore(s):
    import torch,numpy as np
    torch.set_rng_state(s[0]);torch.cuda.set_rng_state_all(s[1]);np.random.set_state(s[2]);random.setstate(s[3])
def rh():
    s=rng();h=hashlib.sha256(s[0].numpy().tobytes())
    for x in s[1]:h.update(x.cpu().numpy().tobytes())
    h.update(pickle.dumps(s[2]));h.update(pickle.dumps(s[3]));return h.hexdigest()
def digest(model, shared=False):
    h=hashlib.sha256()
    for name,t in model.state_dict().items():
        if shared and name.startswith("refiner."):continue
        h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def fixture(corpus,device,n=64):
    import numpy as np,torch
    good=corpus.data_df["train"];good=good[good.position>0]
    groups=good.groupby("user_id",sort=False).indices;local=np.random.default_rng(42)
    users=local.choice(list(groups),n,replace=False);hs=[];ls=[];items=[];us=[]
    for u in users:
        row=good.iloc[int(local.choice(groups[u]))];p=int(row.position)
        seq=corpus.user_his[int(u)][max(0,p-20):p]
        assert seq and all(t<=row.time for _,t in seq)
        ids=[int(j) for j,_ in seq];neg=int(local.integers(1,corpus.n_items))
        while neg in corpus.train_clicked_set[int(u)]:neg=int(local.integers(1,corpus.n_items))
        hs.append(ids+[0]*(20-len(ids)));ls.append(len(ids));items.append([int(row.item_id),neg]);us.append(int(u))
    return {k:torch.tensor(v,device=device) for k,v in dict(history_items=hs,lengths=ls,item_id=items,user_id=us).items()}
def probe(model,batch,label):
    import torch
    from models.BaseModel import GeneralModel
    before=rng();hash_before=rh();was=model.training;checked=model._first_batch_checked
    try:
        model.eval();out=model(batch,True);base=out["base_user_vector"];user=out["user_vector"]
        residual=user-base;v=out["interest_vectors"]
        def stats(x):
            x=x.detach().float()
            return dict(mean=float(x.mean()),median=float(x.median()),max=float(x.max()),min=float(x.min()))
        rec=dict(epoch=label,rows=len(base),margin=stats(out["prediction"][:,0]-out["prediction"][:,1]),
            BPR=float(GeneralModel.loss(model,out)),
            residual_to_base=stats(residual.norm(dim=-1)/base.norm(dim=-1).clamp_min(1e-12)),
            residual_norm=stats(residual.norm(dim=-1)),
            original_interest_head_difference=stats((v[:,0]-v[:,1]).norm(dim=-1)/v.norm(dim=-1).mean(-1).clamp_min(1e-12)))
        if model.refine_mode=="full":
            vf=out["refined_interest_vectors"];w=out["interest_weights"]
            rec.update(S_norm=stats(out["interest_semantic"].norm(dim=-1)),
                C_norm=stats(out["interest_complement"].norm(dim=-1)),
                S_head_difference=stats((out["interest_semantic"][:,0]-out["interest_semantic"][:,1]).norm(dim=-1)),
                C_head_difference=stats((out["interest_complement"][:,0]-out["interest_complement"][:,1]).norm(dim=-1)),
                refined_head_difference=stats((vf[:,0]-vf[:,1]).norm(dim=-1)/vf.norm(dim=-1).mean(-1).clamp_min(1e-12)))
            # Frozen train-only ablation, OFF-DISTRIBUTION: repeat weighted-mean
            # features at all slots; not a retrained user MLP or causal proof.
            f=out["refinement_features"];meanf=(f*w[:,:,None]).sum(1)
            repeated=model.refiner(meanf[:,None].expand_as(f))
            alt=base+(repeated*w[:,:,None]).sum(1)
            alt_scores=(alt[:,None]*out["candidate_vectors"]).sum(-1)
            rec["frozen_mean_feature_replacement_prediction_MAE"]=float((alt_scores-out["prediction"]).abs().mean())
            rec["frozen_mean_feature_replacement_note"]="Train-only out-of-distribution probe, not a fair retraining or causal attribution"
        rec["length_groups"]={}
        for name,mask in [("1-3",batch["lengths"]<=3),("4-9",(batch["lengths"]>=4)&(batch["lengths"]<=9)),("10-20",batch["lengths"]>=10)]:
            if mask.any():
                margin=out["prediction"][mask,0]-out["prediction"][mask,1]
                rec["length_groups"][name]=dict(n=int(mask.sum()),mean_margin=float(margin.mean()),
                    BPR=float(GeneralModel.loss(model,dict(prediction=out["prediction"][mask]))))
        assert torch.isfinite(out["prediction"]).all()
        return rec
    finally:
        model.train(was);model._first_batch_checked=checked
        assert rh()==hash_before,"Probe changed random state"
        restore(before)
def prepare(rid):
    assert re.fullmatch("[A-Za-z0-9_.-]+",rid)
    assert all((ROOT/f).exists() for f in SOURCES)
    root=ROOT/"new_log/refine_bpr_round1"/rid
    assert not root.exists()
    source={f:sha(ROOT/f) for f in SOURCES};jobs=[]
    for ds in ["beauty","ml-1m"]:
        assets={str((ROOT/"data"/ds/f).relative_to(ROOT)):sha(ROOT/"data"/ds/f)
                for f in ["train.csv","dev.csv","SeqReader.pkl","item2id.json","handled/llm_table_pca1536.pkl"]}
        for arm,mode in [("B1","user"),("B2","full")]:
            log=root/ds/arm;weight=ROOT/"new_model/refine_bpr_round1"/rid/ds/arm/"best.pt"
            assert not log.exists() and not weight.parent.exists()
            args,hist=arguments(ds,mode,log/"train.log",weight)
            conf=dict(dataset=ds,arm=arm,run_id=rid,effective_args=vars(args),historical_config=hist,
                source_sha256=source,asset_sha256=assets,test_allowed=False,
                command=[sys.executable,"-B",str(Path(__file__).resolve()),"run",str(log/"config.json")])
            jobs.append(conf)
    for conf in jobs:
        log=Path(conf["effective_args"]["log_file"]).parent
        log.mkdir(parents=True,exist_ok=False)
        Path(conf["effective_args"]["model_path"]).parent.mkdir(parents=True,exist_ok=False)
        write_new(log/"config.json",conf)
    write_new(root/"manifest.json",dict(run_id=rid,jobs=jobs,test_allowed=False))
    print(json.dumps(dict(root=str(root),jobs=[j["effective_args"]["log_file"] for j in jobs])))
def run(confpath):
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    import numpy as np,torch
    import helpers.BaseRunner as rm
    from helpers.BaseRunner import BaseRunner
    from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF
    from models.sequential.LLMMIRecASPCFRefineBPR import LLMMIRecASPCFRefineBPR as Model
    from utils import utils
    conf=json.loads(Path(confpath).read_text());a=argparse.Namespace(**conf["effective_args"])
    log=Path(a.log_file).parent
    assert not Path(a.log_file).exists() and not Path(a.model_path).exists()
    assert all(sha(ROOT/f)==h for f,h in conf["source_sha256"].items())
    assert a.K==4 and a.num_neg==1 and a.epoch==200 and a.test_epoch==-1 and a.dev_only==1 and a.load==0
    logging.basicConfig(filename=a.log_file,level=logging.INFO,format="%(asctime)s %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    start=time.monotonic();state=dict(success=False,test_executed=False,epochs=[],probes=[])
    try:
        a.device=torch.device("cuda");utils.init_seed(42)
        with (ROOT/"data"/a.dataset/"SeqReader.pkl").open("rb") as f:corpus=pickle.load(f)
        model=Model(a,corpus).to(a.device)
        state.update(parameters=model.count_variables(),added_parameters=sum(p.numel() for p in model.refiner.parameters()),
            initial_shared_sha256=digest(model,True),initial_state_sha256=digest(model),initial_rng=rh())
        saved=rng();utils.init_seed(42);baseline=LLMMIRecASPCF(a,corpus).to(a.device)
        assert digest(baseline,True)==state["initial_shared_sha256"]
        del baseline;restore(saved)
        assert rh()==state["initial_rng"]
        state["base_initialization_verified"]=True
        b=fixture(corpus,a.device);state["probe_fixture_sha256"]=bh(b"".join(x.cpu().numpy().tobytes() for x in b.values()))
        state["probes"].append(probe(model,b,0))
        data={p:Model.Dataset(model,corpus,p) for p in ["train","dev"]}
        for d in data.values():d.prepare()
        state.update(train_rows=len(data["train"]),dev_rows=len(data["dev"]))
        active=False;relations=[];OriginalLoader=rm.DataLoader
        class RecordedLoader(OriginalLoader):
            def __iter__(self):
                if getattr(self.dataset,"phase",None)!="train":
                    yield from super().__iter__();return
                order=hashlib.sha256();rows=0;steps=0
                for batch in super().__iter__():
                    order.update(batch["user_id"].numpy().tobytes());order.update(batch["item_id"].numpy().tobytes())
                    rows+=len(batch["user_id"]);steps+=1;yield batch
                state["epochs"][-1].update(train_order_sha256=order.hexdigest(),rows=rows,steps=steps)
        rm.DataLoader=RecordedLoader
        def hook(m,inputs,out):
            if active:
                assert not any(k.startswith("_ss") for k in out)
                if "_relation_ids" in out:relations.append(out["_relation_ids"].detach())
        handle=model.register_forward_hook(hook)
        class DevOnlyRunner(BaseRunner):
            def fit(self,dataset,epoch=-1):
                nonlocal active,relations
                assert dataset.phase=="train"
                rec=dict(epoch=epoch,rng_start=rh());state["epochs"].append(rec)
                active=True;relations=[];t=time.monotonic()
                try:loss=super().fit(dataset,epoch)
                finally:active=False
                rec.update(loss=float(loss),fit_seconds=time.monotonic()-t,rng_after_fit=rh(),
                    negatives_sha256=bh(np.ascontiguousarray(dataset.data["neg_items"]).tobytes()),
                    relation_sha256=bh(torch.cat(relations).cpu().numpy().tobytes()))
                assert np.isfinite(loss)
                if epoch in [1,50,100,200]:state["probes"].append(probe(model,b,epoch))
                if time.monotonic()-start>4*3600:raise TimeoutError("Four-hour per-run budget")
                return loss
            def predict(self,dataset):
                assert dataset.phase=="dev","Test prohibited"
                return super().predict(dataset)
            def evaluate(self,dataset,topks,metrics):
                assert dataset.phase=="dev"
                t=time.monotonic();result=super().evaluate(dataset,[5,10,20],metrics)
                assert all(np.isfinite(v) for v in result.values())
                rec=state["epochs"][-1];rec.update(dev={k:float(v) for k,v in result.items()},
                    dev_seconds=time.monotonic()-t,rng_after_dev=rh())
                write_new(log/("epoch_%03d.json"%rec["epoch"]),rec)
                print("DEV_FULL "+json.dumps(dict(epoch=rec["epoch"],**rec["dev"])),flush=True)
                if time.monotonic()-start>4*3600:raise TimeoutError("Four-hour per-run budget")
                return result
        runner=DevOnlyRunner(a);assert runner.main_metric=="NDCG@5"
        torch.cuda.reset_peak_memory_stats()
        write_new(log/"initialization.json",{k:v for k,v in state.items() if k!="epochs"})
        t=time.monotonic();runner.train(data);state["training_dev_seconds"]=time.monotonic()-t
        values=[e["dev"]["NDCG@5"] for e in state["epochs"]];best=max(values)
        first=values.index(best)+1;last=max(i+1 for i,v in enumerate(values) if v==best)
        state.update(best_reported_epoch=first,checkpoint_saved_epoch=last,end_epoch=len(values),
            dev=state["epochs"][last-1]["dev"],checkpoint_sha256=sha(a.model_path))
        loaded=torch.load(a.model_path,map_location=a.device,weights_only=True)
        assert all(torch.equal(v,loaded[k]) for k,v in model.state_dict().items())
        state["best_checkpoint_loaded_verified"]=True
        state["probes"].append(probe(model,b,"best_"+str(last)))
        assert sha(a.model_path)==state["checkpoint_sha256"]
        state["probe_checkpoint_unchanged"]=True;state["success"]=True
        handle.remove()
    except BaseException as e:
        state.update(error=repr(e),traceback=traceback.format_exc());raise
    finally:
        state.update(total_seconds=time.monotonic()-start,
            peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
            source_unchanged={f:sha(ROOT/f)==h for f,h in conf["source_sha256"].items()})
        write_new(log/"summary.json",state)
def orchestrate(rid):
    root=ROOT/"new_log/refine_bpr_round1"/rid
    manifest=json.loads((root/"manifest.json").read_text());results=[]
    for job in manifest["jobs"]:
        while True:
            row=subprocess.check_output(["nvidia-smi","--id=1","--query-gpu=uuid,memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True).strip().split(",")
            apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"],text=True)
            if row[0].strip() not in apps and float(row[1])>=7000:break
            print("GPU1 occupied: safely waiting; no other processes modified",flush=True);time.sleep(60)
        log=Path(job["effective_args"]["log_file"]).parent
        assert not (log/"train.out").exists()
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES="1",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
        with (log/"train.out").open("xb") as f:
            child=subprocess.Popen(job["command"],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
            write_new(log/"process.json",dict(pid=child.pid,started_unix=time.time(),gpu=1))
            code=child.wait()
        result=dict(dataset=job["dataset"],arm=job["arm"],exit_code=code,summary=str(log/"summary.json"))
        results.append(result);write_new(root/(job["dataset"]+"_"+job["arm"]+"_finished.json"),result)
        if code:write_new(root/"blocked.json",dict(results=results));raise RuntimeError(result)
        s=json.loads((log/"summary.json").read_text());assert s["success"] and all(s["source_unchanged"].values())
        # Immediately validate pairing before starting another dataset.
        if job["arm"]=="B2":
            ds=job["dataset"];x,y=[json.loads((root/ds/arm/"summary.json").read_text()) for arm in ["B1","B2"]]
            checks=dict(shared_initial=x["initial_shared_sha256"]==y["initial_shared_sha256"],
                initial_rng=x["initial_rng"]==y["initial_rng"],fixture=x["probe_fixture_sha256"]==y["probe_fixture_sha256"])
            for field in ["rng_start","rng_after_fit","rng_after_dev","negatives_sha256","train_order_sha256","relation_sha256","steps"]:
                checks[field]=all(e[field]==f[field] for e,f in zip(x["epochs"],y["epochs"]))
            write_new(root/(ds+"_fairness.json"),dict(checks=checks,common_epochs=min(x["end_epoch"],y["end_epoch"])))
            assert all(checks.values()),"Pairing failed; stop remaining runs"
        print("RUN_COMPLETE "+json.dumps({**result,"dev":s["dev"]}),flush=True)
    write_new(root/"complete.json",dict(status="complete",results=results,test_executed=False))
if __name__=="__main__":
    action=sys.argv[1]
    if action=="prepare":prepare(sys.argv[2])
    elif action=="run":run(sys.argv[2])
    elif action=="orchestrate":orchestrate(sys.argv[2])
    else:raise ValueError(action)

