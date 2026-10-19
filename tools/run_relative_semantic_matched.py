"""Isolated runtime records for the four authorized RelativeSemantic continuations.
Original model/Reader/Runner source files and training mathematics are unchanged.
"""
import os,sys,json,time,hashlib,random,pickle,math,runpy,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def write_new(path,value):
    p=Path(path);assert p.resolve().is_relative_to(ROOT) and not p.exists()
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2))
def get(args,key):return args[args.index(key)+1]
def runtime(args):
    assert args.pop(0)=="--"
    assert get(args,"--dev_only")=="1" and get(args,"--test_epoch")=="-1"
    assert get(args,"--model_name")=="LLMMIRecASPCFRelativeSemantic"
    os.environ["CUDA_VISIBLE_DEVICES"]=get(args,"--gpu")
    import numpy as np
    import torch
    from helpers.BaseRunner import BaseRunner
    from models.sequential.LLMMIRecASPCFSampledSoftmax import candidate_mask
    log=Path(get(args,"--log_file"));training=get(args,"--train")=="1"
    state=dict(success=False,test_executed=False,epochs=[],probes=[],evaluations=[])
    active=False;fixture=None;started=time.time()
    def rng():
        return dict(cpu=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                    cuda=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
                    numpy=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
                    python=hashlib.sha256(pickle.dumps(random.getstate())).hexdigest())
    def probe(model,corpus,epoch):
        nonlocal fixture
        before=rng();was=model.training;checked=model._first_batch_checked
        if fixture is None:
            # A single fixed Train-only prefix set, local RNG; no Dev/Test rows used.
            df=corpus.data_df["train"];local=np.random.default_rng(42)
            selected=local.choice(np.flatnonzero(df.position.to_numpy()>0),256,replace=False)
            hs=[];lens=[];users=[];ids=[]
            for i in selected:
                row=df.iloc[int(i)];u=int(row.user_id);pos=int(row.position)
                seq=corpus.user_his[u][max(0,pos-20):pos]
                assert seq and all(t<=row.time for _,t in seq)
                h=[int(j) for j,_ in seq];neg=int(local.integers(1,corpus.n_items))
                while neg in corpus.train_clicked_set[u]:neg=int(local.integers(1,corpus.n_items))
                hs.append(h+[0]*(20-len(h)));lens.append(len(h));users.append(u);ids.append([int(row.item_id),neg])
            fixture={k:torch.tensor(v,device=model.device) for k,v in dict(history_items=hs,lengths=lens,user_id=users,item_id=ids).items()}
        try:
            with torch.random.fork_rng(devices=[0]):
                model.eval();out=model(fixture,True)
                allowed,_,_=candidate_mask(model.train_membership,fixture["user_id"],fixture["item_id"][:,0])
                scores=out["user_vector"]@out["candidate_vectors"][:,0,:].T
                rank,_=model._ranking_loss(scores,allowed)
                gradients=torch.autograd.grad(rank,[model.extractor.alpha,model.extractor.query,model.extractor.Wk.weight])
                valid=(torch.arange(20,device=model.device)[None]<fixture["lengths"][:,None])
                selected=valid[:,None,:].expand(-1,model.K,-1)
                logits=out["attention_logits"];corr=out["attention_correction"]
                center=lambda x:x-(x*selected).sum(-1,keepdim=True)/selected.sum(-1,keepdim=True).clamp_min(1)
                rms=lambda x:float(x[selected].square().mean().sqrt())
                raw=rms(logits);centered=rms(center(logits));cr=rms(corr);cc=rms(center(corr))
                record=dict(epoch=epoch,rows=256,mode="fixed Train eval/dropout off, pure Batch Softmax task gradient",
                            alpha=float(model.extractor.alpha),alpha_task_gradient=float(gradients[0]),
                            query_task_gradient=float(gradients[1].norm()),key_task_gradient=float(gradients[2].norm()),
                            ranking_loss=float(rank),raw_logit_rms=raw,centered_logit_rms=centered,
                            correction_rms=cr,centered_correction_rms=cc,
                            correction_to_raw_rms=cr/max(raw,1e-12),
                            correction_to_centered_rms=cc/max(centered,1e-12),
                            attention_mean_L1=float((out["attention_maps"]-out["original_attention"]).abs().sum(-1).mean()),
                            propagation_rms=rms(out["semantic_propagation"]))
                assert all(torch.isfinite(g).all() for g in gradients)
        finally:
            model.train(was);model._first_batch_checked=checked
        assert before==rng(),"Probe contaminated training RNG"
        record["rng_unchanged"]=True
        state["probes"].append(record)
    original_fit=BaseRunner.fit;original_evaluate=BaseRunner.evaluate;original_train=BaseRunner.train
    def fit(self,dataset,epoch=-1):
        entry=dict(epoch=epoch,rng_start=rng());t=time.perf_counter()
        loss=original_fit(self,dataset,epoch)
        entry["fit_seconds"]=time.perf_counter()-t;entry["loss"]=float(loss)
        entry["rng_after_fit"]=rng()
        model=dataset.model;entry["alpha"]=float(model.extractor.alpha.detach())
        if not model.relative_semantic_enabled:assert entry["alpha"]==0,"C0 alpha moved"
        neg=np.ascontiguousarray(dataset.data["neg_items"])
        entry["train_negative_sha256"]=hashlib.sha256(neg.tobytes()).hexdigest()
        entry["optimizer_steps_total"]=int(model.optimizer.state[model.extractor.query]["step"])
        assert math.isfinite(entry["loss"]) and math.isfinite(entry["alpha"])
        state["epochs"].append(entry)
        if epoch in (1,10,20):probe(model,dataset.corpus,epoch)
        if epoch==20:
            final=Path(model.model_path).with_name("final_epoch20.pt")
            assert not final.exists();model.save_model(str(final))
        return loss
    def evaluate(self,dataset,topks,metrics):
        assert dataset.phase=="dev","This runtime never evaluates Test"
        # Same original predictions/ranking once, report all requested cutoffs.
        result=original_evaluate(self,dataset,self.topk,self.metrics)
        record={k:float(v) for k,v in result.items()}
        if active:
            state["epochs"][-1]["dev"]=record
            write_new(log.parent/("epoch_%02d.json"%state["epochs"][-1]["epoch"]),state["epochs"][-1])
        else:state["evaluations"].append(record)
        return result
    def train(self,data):
        nonlocal active
        model=data["train"].model
        digest=hashlib.sha256()
        for name,tensor in model.state_dict().items():
            if name=="extractor.alpha":continue
            digest.update(name.encode());digest.update(tensor.detach().cpu().numpy().tobytes())
        state["initial_shared_sha256"]=digest.hexdigest();state["initial_rng"]=rng()
        state["train_rows"]=len(data["train"]);state["batch_size"]=self.batch_size
        assert self.epoch==20 and self.early_stop==0 and self.test_epoch==-1
        assert float(model.extractor.alpha)==0
        probe(model,data["train"].corpus,0)
        active=True
        try:original_train(self,data)
        finally:active=False
        assert len(state["epochs"])==20
        state["alpha_best_loaded"]=float(model.extractor.alpha)
        state["alpha_final"]=state["epochs"][-1]["alpha"]
        state["parameters"]=model.count_variables()
    BaseRunner.fit=fit;BaseRunner.evaluate=evaluate;BaseRunner.train=train
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    sys.argv=["main.py"]+args
    try:
        runpy.run_path(str(ROOT/"main.py"),run_name="__main__")
        state["success"]=True
    finally:
        state["seconds"]=time.time()-started
        state["peak_allocated_MiB"]=torch.cuda.max_memory_allocated()/2**20
        state["peak_reserved_MiB"]=torch.cuda.max_memory_reserved()/2**20
        write_new(str(log)+".runtime.json",state)
def orchestrate(args):
    assert len(args)==1
    rid=args[0];root=ROOT/"new_log/relative_semantic_continuation"/rid
    root.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env["RUN_ID"]=rid;env["PYTHONDONTWRITEBYTECODE"]="1"
    def call(ds,mode,stage):
        command=["bash","new_bash/run_llmmirec_relative_semantic_continuation.sh",ds,"1",mode,"--execute","--stage",stage]
        path=root/(ds+"_"+mode+"_"+stage+".out")
        f=path.open("xb")
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        f.close();return child
    def directory(ds,mode):return ROOT/"new_log/relative_semantic_continuation"/ds/(mode.lower()+"_k4_h20_seed42_"+rid)
    def wait_gpu():
        while True:
            info=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True)
            row=[x.split(",") for x in info.splitlines() if x.split(",")[0].strip()=="1"][0]
            apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"],text=True)
            if float(row[2])>=7000 and (row[1].strip() not in apps or float(row[3])<=60):return
            print("GPU1 resources unavailable; waiting without touching other tasks",flush=True)
            time.sleep(60)
    wait_gpu()
    # All four epoch0 checks must finish before any training starts.
    for ds in ["beauty","ml-1m"]:
        for mode in ["C0","C1"]:
            p=call(ds,mode,"prepare");assert p.wait()==0,(ds,mode,"epoch0 failed")
        x=[json.loads((directory(ds,m)/"epoch0_gate.json").read_text()) for m in ["C0","C1"]]
        assert x[0]["dev"]==x[1]["dev"],"Two arms' epoch0 Dev differ"
    write_new(root/"all_epoch0_passed.json",dict(status="ready",test_executed=False))
    # Only two processes on GPU1; same-domain arms run together.
    for ds in ["beauty","ml-1m"]:
        wait_gpu()
        children=[call(ds,m,"train") for m in ["C0","C1"]]
        write_new(root/(ds+"_training_started.json"),dict(pids=[p.pid for p in children],dataset=ds,started_unix=time.time()))
        statuses=[p.wait() for p in children]
        assert statuses==[0,0],(ds,statuses)
        ss=[json.loads((directory(ds,m)/"summary.json").read_text()) for m in ["C0","C1"]]
        x,y=ss
        checks=dict(shared_initial_weights=x["runtime"]["initial_shared_sha256"]==y["runtime"]["initial_shared_sha256"],
                    initial_rng=x["runtime"]["initial_rng"]==y["runtime"]["initial_rng"],
                    epoch0=x["initial_dev"]==y["initial_dev"],
                    epoch_rng=all(a["rng_start"]==b["rng_start"] and a["rng_after_fit"]==b["rng_after_fit"] for a,b in zip(x["runtime"]["epochs"],y["runtime"]["epochs"])),
                    negatives=all(a["train_negative_sha256"]==b["train_negative_sha256"] for a,b in zip(x["runtime"]["epochs"],y["runtime"]["epochs"])),
                    update_counts=all(a["optimizer_steps_total"]==b["optimizer_steps_total"] for a,b in zip(x["runtime"]["epochs"],y["runtime"]["epochs"])))
        assert all(checks.values()),checks
        pair=dict(dataset=ds,checks=checks,C0=x,C1=y,epoch_differences=[
            dict(epoch=a["epoch"],ndcg5=b["dev"]["NDCG@5"]-a["dev"]["NDCG@5"]) for a,b in zip(x["runtime"]["epochs"],y["runtime"]["epochs"])])
        write_new(root/(ds+"_paired_comparison.json"),pair)
    write_new(root/"complete.json",dict(status="complete",datasets=["beauty","ml-1m"],test_executed=False))
if __name__=="__main__":
    action=sys.argv[1]
    if action=="main":runtime(sys.argv[2:])
    elif action=="orchestrate":orchestrate(sys.argv[2:])
    else:raise ValueError(action)
