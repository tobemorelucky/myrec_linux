"""Official zero-based epoch > star_test gate; no Test imports or calls."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from run_faerec import BASE, setup, objective, predict, write_json, sha

def dev_allowed(epoch_index,star_test=60):
    return epoch_index > star_test

def train(o):
    start=time.monotonic()
    model,tr,loader,h,c,Runner,cfg=setup(o)
    schedule=dict(star_test=o.star_test,epoch_index_base=0,validation_condition="epoch_index > star_test",
        first_dev_epoch_index=61,first_dev_completed_epoch=62,training_test_enabled=False)
    cfg.update(schedule)
    cfg["controller_source_sha256"]=sha(Path(__file__))
    cfg["shared_adapter_sha256"]=sha(Path(__file__).with_name("run_faerec.py"))
    write_json(o.run_dir/"config.json",cfg);print("DELAYED_CONFIG "+json.dumps(cfg),flush=True)
    best=-float("inf");best_epoch=0;best_dev=None;bad=0;end=0;reason="max_epoch"
    durations=[];checkpoint=o.run_dir/"best.pt";first_dev=None
    for ep in range(o.max_epoch):
        model.train();accum=torch.zeros(4,device="cuda");steps=0;t=time.monotonic();interrupted=False
        for batch in loader:
            if time.monotonic()-start>=o.max_hours*3600:
                reason="time_budget";interrupted=True;break
            tr.optim.zero_grad()
            parts=objective(model,tr,batch,ep)
            parts[0].backward();tr.optim.step()
            accum+=torch.stack([v.detach() for v in parts]);steps+=1
        if interrupted:break
        torch.cuda.synchronize();train_seconds=time.monotonic()-t
        mean=(accum/max(steps,1)).cpu().tolist()
        if not np.isfinite(mean).all():raise RuntimeError("nonfinite training loss")
        end=ep+1;dev=None
        if dev_allowed(ep,o.star_test):
            if first_dev is None:first_dev=ep
            scores=predict(model,h,c,o.eval_batch_size)
            dev=Runner.evaluate_method(scores,[5,10,20],["HR","NDCG"])
            if dev["NDCG@5"]>best:
                best=dev["NDCG@5"];best_epoch=end;best_dev=dev;bad=0
                torch.save(model.state_dict(),checkpoint)
                write_json(o.run_dir/"best.json",dict(epoch=end,epoch_index=ep,dev=dev))
            else:bad+=1
        else:
            assert bad==0 and best_epoch==0 and not checkpoint.exists()
        seconds=time.monotonic()-t;durations.append(seconds)
        row=dict(epoch=end,epoch_index=ep,loss=mean,dev=dev,dev_evaluated=dev is not None,
            train_seconds=train_seconds,epoch_seconds=seconds,elapsed_seconds=time.monotonic()-start,
            best_epoch=best_epoch or None,bad_epochs=bad,steps=steps,
            projected_200_epoch_seconds=float(np.mean(durations))*200)
        with (o.run_dir/"epochs.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
        print(json.dumps(row),flush=True)
        if dev is not None and bad>=o.early_stop:reason="early_stop";break
        if end==3 and np.mean(durations)*o.max_epoch>o.max_hours*3600:
            reason="projected_over_budget";break
    if not checkpoint.exists():
        write_json(o.run_dir/"summary.json",dict(status="budget_stopped",stop_reason=reason,
            dataset=o.dataset,end_epoch=end,best_epoch=None,first_dev_epoch_index=first_dev,
            test_evaluated=False,total_seconds=time.monotonic()-start))
        return
    before=sha(checkpoint)
    model.load_state_dict(torch.load(checkpoint,map_location="cuda",weights_only=True),strict=True)
    final=Runner.evaluate_method(predict(model,h,c,o.eval_batch_size),[5,10,20],["HR","NDCG"])
    assert all(abs(final[k]-best_dev[k])<1e-12 for k in final)
    assert sha(checkpoint)==before and first_dev==61
    result=dict(status="complete" if reason in ("max_epoch","early_stop") else "budget_stopped",
        stop_reason=reason,dataset=o.dataset,seed=o.seed,best_epoch=best_epoch,best_epoch_index=best_epoch-1,
        end_epoch=end,end_epoch_index=end-1,first_dev_epoch_index=first_dev,first_dev_completed_epoch=first_dev+1,
        star_test=o.star_test,best_dev=best_dev,reloaded_dev=final,checkpoint=str(checkpoint),
        checkpoint_sha256=before,checkpoint_unchanged_by_dev=True,
        parameters=cfg["trainable_parameters"],total_seconds=time.monotonic()-start,
        mean_epoch_seconds=float(np.mean(durations)),projected_200_epoch_seconds=float(np.mean(durations))*200,
        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,test_evaluated=False,
        config=str(o.run_dir/"config.json"))
    write_json(o.run_dir/"summary.json",result);print("SUMMARY "+json.dumps(result),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",choices=["beauty","ml-1m"],required=True)
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--eval-batch-size",type=int,default=128)
    p.add_argument("--max-epoch",type=int,default=200)
    p.add_argument("--early-stop",type=int,default=10)
    p.add_argument("--max-hours",type=float,default=4)
    p.add_argument("--star-test",type=int,default=60)
    o=p.parse_args();o.run_dir=o.run_dir.resolve()
    assert o.run_dir.is_relative_to(BASE/"runs") and o.star_test==60
    assert o.seed==42 and o.batch_size==64 and o.max_epoch==200 and o.early_stop==10
    assert [dev_allowed(i) for i in [59,60,61,62]]==[False,False,True,True]
    o.run_dir.mkdir(parents=True,exist_ok=False)
    try:train(o)
    except Exception as exc:
        write_json(o.run_dir/"failure.json",dict(type=type(exc).__name__,error=str(exc),test_evaluated=False))
        raise
