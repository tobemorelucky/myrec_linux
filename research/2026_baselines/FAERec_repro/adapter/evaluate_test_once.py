"""One authorized frozen Test pass for a locked delayed-validation run."""
import argparse, json, os, time, subprocess
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
from run_faerec import BASE, ROOT, official, predict, sha, write_json

def evaluate(run):
    start=time.monotonic()
    assert run.resolve()==run and run.is_relative_to(BASE/"runs")
    assert not (run/"test_once.started.json").exists()
    s=json.loads((run/"summary.json").read_text())
    cfg=json.loads((run/"config.json").read_text())
    assert s["status"]=="complete" and s["star_test"]==60 and s["first_dev_epoch_index"]==61
    assert s["best_epoch"]>=62 and s["checkpoint_unchanged_by_dev"]
    assert s["best_dev"]==s["reloaded_dev"] and s["seed"]==42
    assert cfg["star_test"]==60 and cfg["seed"]==42 and cfg["batch_size"]==64
    assert cfg["shared_adapter_sha256"]==sha(Path(__file__).with_name("run_faerec.py"))
    checkpoint=Path(s["checkpoint"]);assert checkpoint.parent==run
    before=sha(checkpoint);assert before==s["checkpoint_sha256"]
    gpu=subprocess.check_output(["nvidia-smi","--id=1","--query-gpu=uuid","--format=csv,noheader"],text=True).strip()
    apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"],text=True)
    assert not any(row.strip().startswith(gpu+",") for row in apps.splitlines()),"GPU1 occupied"
    assert os.environ.get("CUDA_VISIBLE_DEVICES")=="1"
    prov=json.loads((BASE/"adapter/provenance.json").read_text())
    assert all(sha(BASE/"upstream"/p)==h for p,h in prov["source_sha256"].items())
    ds=s["dataset"];assert ds==cfg["data_name"]
    meta=json.loads((run/"data_protocol.json").read_text())
    binding=json.loads((BASE/"adapter/unified_data_manifest.json").read_text())[ds]["asset_sha256"]
    asset_dir=BASE/"runs/assets"/ds/"llm_emb"/ds
    assert all((asset_dir/name).is_file() and sha(asset_dir/name)==value for name,value in binding.items())
    folder=ROOT/"data"/ds
    assert sha(folder/"train.csv")==meta["train_sha256"]
    assert sha(folder/"dev.csv")==meta["dev_sha256"]
    assert sha(Path(meta["source_embedding"]))==meta["embedding_sha256"]
    models,_,utils,Runner=official()
    utils.set_seed(cfg["seed"])
    os.chdir(BASE/"runs/assets"/ds)
    args=SimpleNamespace(**cfg)
    model=models.SASRecModel(args).cuda()
    model.load_state_dict(torch.load(checkpoint,map_location="cuda",weights_only=True),strict=True)
    model.eval()
    snapshot={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    assert all(torch.isfinite(v).all() for v in snapshot.values() if torch.is_floating_point(v))
    # Test data is opened only after normal training and strict best-weight loading.
    train=pd.read_csv(folder/"train.csv",sep="\t")
    dev=pd.read_csv(folder/"dev.csv",sep="\t")
    test=pd.read_csv(folder/"test.csv",sep="\t")
    assert test.groupby("user_id").size().max()==1
    past=pd.concat([train,dev],ignore_index=True).sort_values(["time","user_id"],kind="mergesort")
    last=past.groupby("user_id").time.max()
    assert not test.user_id.map(last).isna().any()
    assert (test.user_id.map(last)<=test.time).all()
    sequences={int(u):g.item_id.to_numpy(np.int64) for u,g in past.groupby("user_id",sort=False)}
    history=np.zeros((len(test),20),dtype=np.int64)
    for i,u in enumerate(test.user_id):
        h=sequences[int(u)][-20:]
        history[i,-len(h):]=h
    negatives=np.stack([np.fromstring(v.strip()[1:-1],sep=",",dtype=np.int64) for v in test.neg_items])
    assert negatives.shape==(len(test),1000)
    candidates=np.concatenate([test.item_id.to_numpy(np.int64)[:,None],negatives],axis=1)
    assert candidates.min()>0 and candidates.max()<cfg["item_size"]-1
    marker=dict(dataset=ds,seed=42,checkpoint_sha256=before,
        started_utc=datetime.now(timezone.utc).isoformat(),test_csv_sha256=sha(folder/"test.csv"),
        selection="fixed delayed60 best-dev, not selected using Test",candidates=1001)
    with (run/"test_once.started.json").open("x") as f:json.dump(marker,f,indent=2)
    try:
        pred=predict(model,history,candidates,cfg["eval_batch_size"])
        metrics=Runner.evaluate_method(pred,[5,10,20],["HR","NDCG"])
        assert sha(checkpoint)==before
        assert all(torch.equal(snapshot[k],v.detach().cpu()) for k,v in model.state_dict().items())
        result=dict(status="complete",dataset=ds,seed=42,model="FAERec-SASRec",classification="LLM-enhanced single-vector sequential recommendation; not explicit K interests",
            test=metrics,dev=s["best_dev"],checkpoint=str(checkpoint),checkpoint_sha256=before,
            checkpoint_unchanged=True,in_memory_weights_unchanged=True,
            best_epoch=s["best_epoch"],best_epoch_index=s["best_epoch_index"],
            star_test=60,test_users=len(test),candidate_slots=1001,duplicate_slots_preserved=True,
            history_max=20,history_sources="Train+chronologically prior Dev, excluding current Test target",
            evaluator="helpers.BaseRunner.BaseRunner.evaluate_method (unchanged)",
            train_csv_sha256=meta["train_sha256"],dev_csv_sha256=meta["dev_sha256"],
            test_csv_sha256=marker["test_csv_sha256"],embedding_sha256=meta["embedding_sha256"],
            elapsed_seconds=time.monotonic()-start,
            research_visible_test=True,not_used_for_selection=True,upstream_commit=cfg["upstream_commit"])
        with (run/"test_once.json").open("x") as f:json.dump(result,f,indent=2)
        print(json.dumps(result),flush=True)
    except Exception as exc:
        with (run/"test_once.failure.json").open("x") as f:
            json.dump(dict(error=str(exc),type=type(exc).__name__,do_not_retry_without_authorization=True),f)
        raise

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run-dir",type=Path,required=True)
    o=p.parse_args();evaluate(o.run_dir.resolve())
