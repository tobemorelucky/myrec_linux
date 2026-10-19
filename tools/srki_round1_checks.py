"""Bounded SRKI round-1 logging only; never alters model/Reader/Runner mathematics."""
import os,sys,time,json,hashlib,runpy,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def value(args,key):
    return args[args.index(key)+1]
def write(path,data):
    p=Path(path);resolved=p.resolve()
    assert resolved.is_relative_to(ROOT) and not p.exists(), "outside project or output collision"
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,indent=2))
mode=sys.argv[1]
if mode=="main":
    args=sys.argv[2:];assert args.pop(0)=="--"
    assert value(args,"--dev_only")=="1" and value(args,"--test_epoch")=="-1"
    assert value(args,"--model_name")=="LLMMIRecASPCFSRKI"
    os.environ["CUDA_VISIBLE_DEVICES"]=value(args,"--gpu")
    import torch
    started=time.time();success=False
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    sys.argv=["main.py"]+args
    try:
        runpy.run_path(str(ROOT/"main.py"),run_name="__main__");success=True
    finally:
        write(value(args,"--log_file")+".resources.json",dict(success=success,seconds=time.time()-started,
              peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
              peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
              GPU=torch.cuda.get_device_name(0),test_executed=False))
    raise SystemExit
assert mode in ("preflight","best")
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--gpu",default="1");parser.add_argument("--output",required=True)
parser.add_argument("--config")
args=parser.parse_args(sys.argv[2:]);assert args.gpu=="1"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
import torch,numpy as np
from test_llmmirec_srki import make,fixture,probe,residual_support
signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError("170s bounded probe")))
signal.alarm(170)
started=time.time()
def coverage(m,b):
    m.eval()
    with torch.no_grad():
        o=m(b,True);_,z,valid=m.encode_history(b["history_items"],b["lengths"])
        full=residual_support(z,valid,z.shape[1],True)
        p2=o["base_attention"].sum(1)>0
        pf=full["base_attention"].sum(1)>0
        # Signed support tolerance fixed in advance; report near-zero ambiguity separately.
        s2=o["factors"].abs().amax(1)>1e-7
        sf=full["factors"].abs().amax(1)>1e-7
        unc=valid&~p2
        categories=dict(positive_truncation_only=unc&~pf&sf,
                        K_cap_only=unc&pf&~s2,
                        combined=unc&pf&s2,
                        zero_or_numeric=unc&~pf&~sf)
        assert sum(int(x.sum()) for x in categories.values())==int(unc.sum())
        row=torch.arange(valid.shape[0],device=valid.device);last=(b["lengths"]-1).clamp_min(0)
        latest=b["lengths"]>0
        norms=z.norm(dim=-1)
        return dict(valid_history_positions=int(valid.sum()),uncovered_positions=int(unc.sum()),
                    uncovered_fraction=float(unc.sum()/valid.sum()),
                    attribution_counts={k:int(v.sum()) for k,v in categories.items()},
                    attribution_definition="U=valid and no positive support at K2; full=exhaust rank using same pivot rule, <=H20. sign-only=U & not Pfull & Sfull; K-only=U & Pfull & not S2; both=U & Pfull & S2; remaining zero/numeric. Signed support abs(q)>1e-7; positive support exact>0. Operational, not causal.",
                    latest_valid_users=int(latest.sum()),latest_uncovered=int((unc[row,last]&latest).sum()),
                    latest_zero_semantic=int(((norms[row,last]<=1e-8)&latest).sum()),
                    active_slots={str(k):int((o["interest_mask"].sum(1)==k).sum()) for k in range(m.K+1)},
                    fallback_users=int(o["fallback_used"].sum()),
                    learned_uncovered_positions=int((valid&~(o["attention_maps"].sum(1)>0)).sum()),
                    learned_mass_below_1e8=int((valid&(o["attention_maps"].sum(1)<1e-8)).sum()),
                    kernel_finite=bool(torch.isfinite(full["kernel"]).all()))
def batch_sha(b):
    h=hashlib.sha256()
    for k,v in sorted(b.items()):h.update(k.encode());h.update(v.numpy().tobytes())
    return h.hexdigest()
def one(dataset,checkpoint=None,meta=None):
    corpus,cpu,sampling=fixture(dataset,1024)
    b={k:v.cuda() for k,v in cpu.items()}
    m=make(K=meta["K"] if meta else 2,source=meta["kernel_source"] if meta else "llm",corpus=corpus,dataset=dataset,device="cuda:0")
    rng=torch.get_rng_state().clone();crng=torch.cuda.get_rng_state().clone()
    result=dict(dataset=dataset,sampling=sampling,fixed_batch_sha256=batch_sha(cpu),
                parameters=m.count_variables())
    if checkpoint:
        m.llm_emb_path=meta["llm_emb_path"];m.load_model(str(checkpoint))
    else:
        shuffle=make(source="permuted_llm",corpus=corpus,dataset=dataset,device="cuda:0")
        result["Real_Shuffle_equal_parameters"]=all(torch.equal(x,y) for x,y in zip(m.parameters(),shuffle.parameters()))
        result["Real_Shuffle_same_parameter_count"]=m.count_variables()==shuffle.count_variables()
        result["Real_Shuffle_same_CPU_RNG"]=torch.equal(rng,torch.get_rng_state())
        result["Real_Shuffle_same_CUDA_RNG"]=torch.equal(crng,torch.cuda.get_rng_state())
        assert all(result[k] for k in list(result) if k.startswith("Real_Shuffle"))
        del shuffle
    result["coverage"]=coverage(m,b);result["task_gradient_probe"]=probe(m,b)
    assert all(torch.isfinite(p).all() for p in m.parameters())
    if not checkpoint:
        optimizer=torch.optim.Adam(m.customize_parameters(),lr=.004 if dataset=="beauty" else .001,weight_decay=1e-6)
        m.train()
        def step():
            optimizer.zero_grad();o=m(b);loss=m.loss(o);assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
            optimizer.step()
        step();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
        for _ in range(3):step()
        torch.cuda.synchronize()
        result["temporary_steps_not_saved"]=4
        result["step_seconds"]=(time.perf_counter()-t)/3
        result["peak_allocated_MiB"]=torch.cuda.max_memory_allocated()/2**20
        result["peak_reserved_MiB"]=torch.cuda.max_memory_reserved()/2**20
        result["train_examples"]=int((corpus.data_df["train"].position>0).sum())
        result["steps_per_epoch"]=(result["train_examples"]+1023)//1024
        result["pure_train_200h"]=200*result["steps_per_epoch"]*result["step_seconds"]/3600
    return result
if mode=="preflight":
    result=dict(protocol="Train-only seed42 fixed1024, H20 K2, no checkpoint saved, no Dev/Test",
                results=[one("beauty"),one("ml-1m")])
else:
    config=json.loads(Path(args.config).read_text());cmd=config["command"]
    ckpt=Path(value(cmd,"--model_path"));before=hashlib.sha256(ckpt.read_bytes()).hexdigest()
    metadata=json.loads(Path(str(ckpt)+".srki.json").read_text())
    result=one(value(cmd,"--dataset"),ckpt,metadata)
    result["checkpoint_sha256"]=before
    assert before==hashlib.sha256(ckpt.read_bytes()).hexdigest()
    result["checkpoint_unchanged"]=True
result["seconds"]=time.time()-started
write(args.output,result)
print(json.dumps(result))
