"""Bounded paired Train/Dev correctness screen, never Test or full training."""
import os
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("MKL_NUM_THREADS","4")
import sys,json,time,signal,pickle,copy,csv,argparse,re,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path("/home/yyx/hzgcode/pom2.0").resolve()
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(BASE))
from prepare import reuse
from models.sequential.LLMMIRecASPCFAdaptiveSIL import LLMMIRecASPCFAdaptiveSIL as Model,file_sha
old=reuse.old
MODES={"B0":"baseline","C1":"fixed","M1":"semantic"}
PREP=BASE/"prep_v2"
OUT=None


def write(name,data):
    text=json.dumps(data,indent=2,allow_nan=False)
    path=OUT/name
    assert path.parent.resolve()==OUT
    with path.open("x") as f:f.write(text)


def snapshot():return torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()
def restore(states):torch.set_rng_state(states[0]);torch.cuda.set_rng_state(states[1])


def negative(model,users,generator):
    item=torch.randint(1,model.item_num,(len(users),),device=model.device,generator=generator)
    while True:
        invalid=((model.train_membership[users,item//64]>>(item%64))&1).bool()
        if not invalid.any():return item
        item[invalid]=torch.randint(1,model.item_num,(int(invalid.sum()),),device=model.device,generator=generator)


def feed(history,lengths,users,positive,negative_items,device):
    return dict(history_items=torch.as_tensor(history,device=device),lengths=torch.as_tensor(lengths,device=device),
                user_id=torch.as_tensor(users,device=device),
                item_id=torch.as_tensor(np.column_stack([positive,negative_items]),device=device))


def init_model(args,corpus,state,mode):
    aa=copy.copy(args)
    aa.sil_mode=mode
    model=Model(aa,corpus)
    missing,unexpected=model.load_state_dict(state,strict=False)
    expected=["sil_log_frequency"]
    if model.sil_block is not None:
        expected += ["sil_block."+name for name in model.sil_block.state_dict()]
    assert set(missing)==set(expected) and not unexpected,(missing,unexpected)
    return model.to(aa.device)


def real_checks(models,args,corpus,state,probe):
    # Exact original ASPCF forward/loss/query-gradient identity at the zero output.
    reference=old.Model(args,corpus).to(args.device)
    reference.load_state_dict(state,strict=True)
    reference.train()
    start=snapshot()
    output=reference(probe,True)
    loss=reference.loss(output)
    grad=torch.autograd.grad(loss,reference.extractor.query)[0]
    after=snapshot()
    checks={}
    for name,model in models.items():
        restore(start);model.train()
        new=model(probe,True);new_loss=model.loss(new)
        new_grad=torch.autograd.grad(new_loss,model.extractor.query)[0]
        assert torch.equal(output["prediction"],new["raw_prediction"])
        assert torch.equal(loss,new_loss) and torch.equal(grad,new_grad)
        assert all(torch.equal(a,b) for a,b in zip(snapshot(),after))
        model.eval()
        with torch.no_grad():
            original=model(probe,True)
            changed=model({**probe,"item_id":probe["item_id"].flip(1)},True)
            assert torch.equal(original["interest_vectors"],changed["interest_vectors"])
        checks[name]=dict(zero_initial_raw_score_loss_query_gradient_exact=True,
                          candidate_invariant_interests=True,unchanged_dropout_and_relation_RNG=True)
        model._ss_stats=None
    restore(after)
    del reference
    # K is configurable in the actual complete model, independently of head count.
    aa=copy.copy(args);aa.K=1;aa.sil_mode="semantic"
    k1=Model(aa,corpus).to(args.device)
    k1.train()
    result=k1(probe,True)
    assert result["interest_vectors"].shape==(len(probe["lengths"]),1,64)
    k1.loss(result).backward()
    assert all(torch.isfinite(p.grad).all() for p in k1.parameters() if p.grad is not None)
    # Empty histories get exact zero V and finite backward, including inherited parts.
    empty={**probe,"history_items":torch.zeros_like(probe["history_items"]),
           "lengths":torch.zeros_like(probe["lengths"])}
    k1.zero_grad()
    result=k1(empty,True)
    assert torch.count_nonzero(result["interest_vectors"])==0
    k1.loss(result).backward()
    assert all(torch.isfinite(p.grad).all() for p in k1.parameters() if p.grad is not None)
    del k1
    restore(after)
    checks["complete_K1_and_empty_history_forward_backward"]=True
    return checks


@torch.no_grad()
def encode_users(model,table,history,lengths):
    result=[]
    for off in range(0,len(history),512):
        h=torch.as_tensor(history[off:off+512],device=model.device)
        ll=torch.as_tensor(lengths[off:off+512],device=model.device)
        raw=table[h]
        position=(ll[:,None]-torch.arange(h.shape[1],device=h.device)[None])*(h>0)
        v,_=model.extractor(raw+model.position_emb(position),ll)
        v,_=model.recompose_interests(h,raw,ll,v)
        w=model.aggregator(raw,ll)
        result.append((v*w[:,:,None]).sum(1))
    return torch.cat(result)


@torch.no_grad()
def evaluate(model,corpus,history,lengths,candidates):
    started=time.perf_counter()
    model.eval()
    table=torch.cat([model.item_encoder(torch.arange(off,min(off+2048,corpus.n_items),device=model.device))
                     for off in range(0,corpus.n_items,2048)])
    users=encode_users(model,table,history,lengths)
    ranks_raw=[];ranks_pop=[]
    for off in range(0,len(history),256):
        sl=slice(off,off+256)
        items=torch.as_tensor(candidates[sl],device=model.device)
        raw=(users[sl,None]*table[items]).sum(-1)
        corrected=model.corrected_scores(raw,items)
        ranks_raw.append(reuse.ranks(raw.double().cpu().numpy()))
        ranks_pop.append(reuse.ranks(corrected.cpu().numpy()))
    return dict(raw=reuse.metrics(np.concatenate(ranks_raw)),
                locked_pop=reuse.metrics(np.concatenate(ranks_pop)),
                seconds=time.perf_counter()-started)


def run_dataset(ds,before):
    started=time.perf_counter();old.utils.init_seed(42)
    paths={k:Path(v["path"]) for k,v in before["inputs"][ds].items()}
    assert all(file_sha(p)==before["inputs"][ds][k]["sha256"] for k,p in paths.items())
    lock=json.loads(paths["lock"].read_text())
    with paths["cache"].open("rb") as f:corpus=pickle.load(f)
    corpus.data_df={p:corpus.data_df[p] for p in ("train","dev")}
    tr=corpus.data_df["train"];dev=corpus.data_df["dev"]
    counts=tr.groupby("user_id").size()
    assert np.all(tr.position.to_numpy()<tr.user_id.map(counts).to_numpy())
    assert np.all(dev.position.to_numpy()==dev.user_id.map(counts).to_numpy())
    for u in corpus.user_his:corpus.user_his[u]=corpus.user_his[u][:int(counts.get(u,0))]
    eligible=np.flatnonzero(tr.position.to_numpy()>0)
    indices=np.random.default_rng(42).choice(eligible,size=16*1024,replace=False)
    train=tr.iloc[indices]
    th,tl=reuse.construct_hist(corpus,train);dh,dl=reuse.construct_hist(corpus,dev)
    candidates=np.concatenate([dev.item_id.to_numpy(np.int64)[:,None],np.stack(dev.neg_items.to_numpy())],1)
    assert candidates.shape==(len(dev),1001)
    args=old.arguments(ds,corpus);args.lr=.0001;args.dataset=ds
    args.sil_asset_path=str(PREP/(ds+"_train_asset.json"))
    args.sil_heads=2;args.sil_recent=3;args.sil_use_recent=1;args.sil_use_local=1
    state=torch.load(paths["checkpoint"],map_location="cpu",weights_only=True)
    models={name:init_model(args,corpus,state,mode) for name,mode in MODES.items()}
    parameters={name:sum(p.numel() for p in m.parameters()) for name,m in models.items()}
    assert parameters["C1"]==parameters["M1"]==parameters["B0"]+37504
    assert reuse.statehash(models["C1"].sil_block)==reuse.statehash(models["M1"].sil_block)
    user=train.user_id.to_numpy(np.int64);positive=train.item_id.to_numpy(np.int64)
    generator=torch.Generator(device=args.device).manual_seed(20261022)
    ng=negative(models["B0"],torch.as_tensor(user[:32],device=args.device),generator).cpu().numpy()
    probe=feed(th[:32],tl[:32],user[:32],positive[:32],ng,args.device)
    checks=real_checks(models,args,corpus,state,probe)
    initial=evaluate(models["B0"],corpus,dh,dl,candidates)
    assert all(abs(initial["locked_pop"][k]-lock["dev_metrics"][k])<1e-10 for k in reuse.METRICS),initial
    # Validate fast cached evaluation against the actual full new-model forward.
    for model in models.values():
        model.eval()
        with torch.no_grad():
            table=model.item_encoder(torch.arange(corpus.n_items,device=args.device))
            cu=encode_users(model,table,dh[:32],dl[:32])
            ids=torch.as_tensor(candidates[:32],device=args.device)
            f=dict(history_items=torch.as_tensor(dh[:32],device=args.device),
                   lengths=torch.as_tensor(dl[:32],device=args.device),item_id=ids)
            actual=model(f,True)
            assert torch.allclose(cu,actual["user_vector"],atol=1e-6,rtol=1e-5)
            assert torch.allclose(model.corrected_scores((cu[:,None]*table[ids]).sum(-1),ids),
                                  actual["prediction"],atol=2e-5,rtol=1e-5)
    print("INITIAL_VERIFIED "+json.dumps(dict(dataset=ds,parameters=parameters,Dev=initial)),flush=True)
    torch.cuda.reset_peak_memory_stats()
    opts={name:torch.optim.Adam(m.parameters(),lr=.0001,weight_decay=1e-6) for name,m in models.items()}
    timings={name:[] for name in models};gradients={name:[] for name in models};losses=[]
    for step in range(16):
        sl=slice(step*1024,(step+1)*1024)
        uu=torch.as_tensor(user[sl],device=args.device)
        ng=negative(models["B0"],uu,generator).cpu().numpy()
        f=feed(th[sl],tl[sl],user[sl],positive[sl],ng,args.device)
        random_before=snapshot();random_after=None;row={"step":step+1}
        for name,model in models.items():
            model.train();restore(random_before);opts[name].zero_grad(set_to_none=True)
            torch.cuda.synchronize();t=time.perf_counter()
            output=model(f,True);loss=model.loss(output)
            assert torch.isfinite(loss);loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            qgrad=float(model.extractor.query.grad.norm())
            assert qgrad>0
            record=dict(query=qgrad)
            if model.sil_block is not None:
                block=model.sil_block
                for label,param in dict(output=block.output.weight,qkv=block.local_qkv.weight,
                                        local_out=block.local_out.weight,read_query=block.read_query.weight,
                                        recent_query=block.recent_query.weight,read_key=block.read_key.weight,
                                        read_value=block.read_value.weight).items():
                    assert param.grad is not None and torch.isfinite(param.grad).all()
                    record[label]=float(param.grad.norm())
                assert record["output"]>0
                if step>=1:assert all(record[k]>0 for k in ("qkv","local_out","read_query","recent_query","read_key","read_value"))
                record["delta_RMS"]=float(output["sil_delta"].detach().square().mean().sqrt())
            gradients[name].append(record)
            opts[name].step()
            torch.cuda.synchronize();timings[name].append(time.perf_counter()-t)
            row[name]=float(loss.detach())
            current=snapshot()
            if random_after is None:random_after=current
            else:assert all(torch.equal(a,b) for a,b in zip(current,random_after))
        restore(random_after);losses.append(row)
    final={name:evaluate(m,corpus,dh,dl,candidates) for name,m in models.items()}
    parameter_change={name:float((m.extractor.query.detach().cpu()-state["extractor.query"]).norm())
                      for name,m in models.items()}
    assert all(v>0 for v in parameter_change.values())
    assert all(file_sha(p)==before["inputs"][ds][k]["sha256"] for k,p in paths.items())
    full_cost={}
    for name,times in timings.items():
        # Excludes loader/IO; deliberately an engineering estimate, not a measured full run.
        batches=int(np.ceil(len(eligible)/1024))
        seconds=batches*float(np.median(times[2:]))+final[name]["seconds"]
        full_cost[name]=dict(eligible_rows=len(eligible),batches_per_epoch=batches,
                             estimated_compute_plus_Dev_seconds_per_epoch=seconds,
                             excludes_loader_and_IO=True)
    result=dict(dataset=ds,initial_Dev=initial,after16_Dev=final,parameters=parameters,
                added_parameters={name:p-parameters["B0"] for name,p in parameters.items()},
                fit_16steps_seconds={name:float(sum(v)) for name,v in timings.items()},
                median_warm_step_seconds={name:float(np.median(v[2:])) for name,v in timings.items()},
                step_seconds=timings,step_gradients=gradients,step_losses=losses,
                query_parameter_change_L2=parameter_change,checks=checks,
                peak_shared_process_GPU_MiB=torch.cuda.max_memory_allocated()/2**20,
                seconds=time.perf_counter()-started,estimated_full_cost=full_cost,
                Train_subset_sha256=reuse.datahash(indices),Dev_candidates_sha256=reuse.datahash(candidates),
                model_sha256=file_sha(ROOT/"models/sequential/LLMMIRecASPCFAdaptiveSIL.py"),
                train_asset_sha256=file_sha(Path(args.sil_asset_path)),checkpoint_sha256=file_sha(paths["checkpoint"]),
                Test_evaluations=0,checkpoints_written=0,training_steps_per_arm=16)
    write(ds+".json",result)
    with (OUT/(ds+"_dev_metrics.csv")).open("x",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["dataset","arm","scoring",*reuse.METRICS])
        writer.writeheader()
        for name,scores in final.items():
            for scoring in ("raw","locked_pop"):
                writer.writerow(dict(dataset=ds,arm=name,scoring=scoring,**scores[scoring]))
    print("DATASET_COMPLETE "+json.dumps({k:result[k] for k in
          ("dataset","after16_Dev","parameters","fit_16steps_seconds","seconds","peak_shared_process_GPU_MiB","estimated_full_cost")}),flush=True)
    del output,loss,models,opts,corpus,state
    torch.cuda.empty_cache()
    return result


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-tag",default="screen_v1")
    arg=parser.parse_args()
    assert re.fullmatch("screen_v[1-9][0-9]*",arg.output_tag)
    OUT=BASE/arg.output_tag
    assert OUT.parent.resolve().is_relative_to(ROOT)
    OUT.mkdir(exist_ok=False)
    signal.alarm(170);torch.set_num_threads(4);start=time.perf_counter()
    before=json.loads((BASE/"INITIAL_STATE.json").read_text())
    prereg=json.loads((PREP/"PREREG.json").read_text())
    assert (PREP/"PREPARE_COMPLETE.json").is_file()
    assert file_sha(ROOT/"models/sequential/LLMMIRecASPCFAdaptiveSIL.py")==prereg["model_sha256"]
    for name,sha in before["source_sha256"].items():
        if name!="main.py":assert file_sha(ROOT/name)==sha
    import_line="import models.sequential.LLMMIRecASPCFAdaptiveSIL as LLMMIRecASPCFAdaptiveSIL\n"
    import hashlib
    assert hashlib.sha256((ROOT/"main.py").read_text().replace(import_line,"").encode()).hexdigest()==before["source_sha256"]["main.py"]
    write("SCREEN_PLAN.json",dict(config=prereg["config"],screen_sha256=file_sha(__file__),
                                   head=before["head"],GPU=0,no_Test=True,no_full_training=True))
    results=[run_dataset(ds,before) for ds in ("beauty","ml-1m")]
    for name,sha in before["source_sha256"].items():
        if name!="main.py":assert file_sha(ROOT/name)==sha
    write("COMPLETE.json",dict(seconds=time.perf_counter()-start,datasets=[r["dataset"] for r in results],
                                Test_evaluations=0,checkpoints_written=0,old_sources_and_assets_unchanged=True))
    print("ALL_COMPLETE",flush=True)


if __name__=="__main__":
    try:main()
    except Exception as exc:
        if OUT is not None and OUT.is_dir() and not (OUT/"FAILURE.json").exists():
            write("FAILURE.json",dict(error=repr(exc),traceback=traceback.format_exc(),no_Test=True))
        raise
