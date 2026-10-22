"""Authorized two-run Clean ID Softmax experiment; delegates training/evaluation to BaseRunner."""
import os
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
import sys, argparse, copy, gc, hashlib, json, logging, pickle, re, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from helpers.SeqReader import SeqReader
from helpers.BaseRunner import BaseRunner
from models.sequential.LLMMIRec import LLMMIRec
from models.sequential.LLMMIRecCleanSoftmax import LLMMIRecCleanSoftmax as Clean
from models.sequential.LLMMIRecASPCFSampledSoftmax import (
    LLMMIRecASPCFSampledSoftmax as ExistingID, candidate_mask, masked_softmax_loss)
from utils import utils

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def save(path, data):
    with path.open("x", encoding="utf8") as f:json.dump(data,f,ensure_ascii=False,indent=2)

def state_sha(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def settings(ds, checkpoint):
    p=SeqReader.parse_data_args(argparse.ArgumentParser())
    p=BaseRunner.parse_runner_args(p)
    p=Clean.parse_model_args(p)
    a=p.parse_args([])
    a.dataset=ds;a.path="./data/";a.device=torch.device("cuda")
    a.model_path=str(checkpoint);a.random_seed=42
    a.K=4;a.history_max=20;a.emb_size=64;a.attn_size=64
    a.item_encoder="id";a.interest_query_mode="learnable";a.lambda_relation=0.
    a.llm_emb_path="";a.lr=.001;a.l2=1e-6;a.optimizer="Adam"
    a.batch_size=1024;a.eval_batch_size=256;a.num_neg=1;a.dropout=.1
    a.epoch=200;a.early_stop=10;a.test_epoch=-1
    a.topk="5,10,20";a.metric="NDCG,HR";a.num_workers=5
    a.pin_memory=1;a.grad_clip=0.;a.test_all=0;a.sampled_softmax_temperature=1.
    return a

def corpus(ds):
    p=ROOT/"data"/ds/"SeqReader.pkl"
    assert p.resolve().is_relative_to(ROOT) and p.is_file()
    with p.open("rb") as f:return pickle.load(f)

def preflight(ds, out):
    c=corpus(ds);a=settings(ds,out/"unused.pt")
    utils.init_seed(42);base=LLMMIRec(a,c).to(a.device);rng0=torch.get_rng_state().clone()
    utils.init_seed(42);ss=Clean(a,c).to(a.device)
    assert torch.equal(rng0,torch.get_rng_state())
    assert state_sha(base)==state_sha(ss)
    assert base.Dataset is Clean.Dataset
    assert base.count_variables()==ss.count_variables()
    assert not hasattr(ss.item_encoder,"llm_table")
    train=Clean.Dataset(ss,c,"train")
    utils.init_seed(137);train.actions_before_epoch()
    indices=np.random.default_rng(42).choice(len(train),size=min(32,len(train)),replace=False)
    batch=utils.batch_to_gpu(train.collate_batch([train[int(i)] for i in indices]),a.device)
    base.train();ss.train()
    utils.init_seed(91);ob=base(batch,True);rng_b=torch.cuda.get_rng_state()
    utils.init_seed(91);os_=ss(batch,True)
    assert torch.equal(rng_b,torch.cuda.get_rng_state())
    assert torch.equal(ob["prediction"],os_["prediction"])
    assert "_relation_ids" not in os_
    mask,_,_=candidate_mask(ss.train_membership,batch["user_id"],batch["item_id"][:,0])
    assert torch.equal(mask,os_["_ss_allowed"])
    torch.testing.assert_close(os_["_ss_scores"].diagonal(),os_["prediction"][:,0],rtol=1e-5,atol=1e-7)
    expected,_=masked_softmax_loss(os_["_ss_scores"],mask,1.)
    actual=ss.loss(os_)
    assert torch.equal(expected,actual)
    original=base.loss(ob)
    assert torch.equal(original,LLMMIRec.loss(ss,os_))
    actual.backward()
    assert torch.isfinite(actual)
    for module in (ss.item_encoder,ss.extractor,ss.aggregator):
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads)>0
    # Existing ASPCF-class ID path is an equivalence check, not an ASPCF semantic model.
    utils.init_seed(42);eid=ExistingID(a,c).to(a.device)
    assert state_sha(eid)==state_sha(base)
    utils.init_seed(91);oe=eid(batch,True)
    assert torch.equal(oe["prediction"],os_["prediction"])
    assert torch.equal(eid.loss(oe),actual)
    # Synthetic duplicate and known-click cases, using the same shared implementation.
    users=batch["user_id"][:4];pos=batch["item_id"][:4,0].clone();pos[1]=pos[0]
    ma,_,_=candidate_mask(ss.train_membership,users,pos)
    assert ma.diagonal().all() and not ma[0,1] and not ma[1,0]
    # No relation/LLM parameters, no new trainable capacity.
    init_sha=state_sha(base)
    old=ROOT/"new_model/llmmirec_phase0"/ds/"id/LLMMIRec_id_seed42.pt"
    base.load_state_dict(torch.load(old,map_location=a.device,weights_only=True),strict=True)
    base.eval();dev=base.Dataset(base,c,"dev");dev.prepare()
    runner=BaseRunner(a);metrics=runner.evaluate(dev,[5,10,20],["HR","NDCG"])
    oldlog=ROOT/"new_log/llmmirec_phase0"/ds/"id/LLMMIRec_id_seed42.log"
    text=oldlog.read_text()
    line=re.findall(r"Best Iter\(dev\)=.*",text)[-1]
    expected_n=float(re.search(r"NDCG@5:([0-9.]+)",line).group(1))
    expected_h=float(re.search(r"HR@5:([0-9.]+)",line).group(1))
    assert abs(metrics["NDCG@5"]-expected_n)<=.000051,(metrics,line)
    assert abs(metrics["HR@5"]-expected_h)<=.000051
    # Full original Dev batch preserves candidate slots and duplicate item scores.
    b=utils.batch_to_gpu(dev.collate_batch([dev[i] for i in range(8)]),a.device)
    assert b["item_id"].shape[1]==1001
    with torch.no_grad():
        p=base(b)["prediction"]
        b2=copy.deepcopy(b);b2["item_id"][:,2]=b2["item_id"][:,1]
        p2=base(b2)["prediction"]
    assert torch.equal(p2[:,1],p2[:,2])
    rec=dict(dataset=ds,status="passed",parameters=ss.count_variables(),initial_state_sha256=init_sha,
        old_bpr_checkpoint=str(old),old_bpr_checkpoint_sha256=sha(old),
        old_bpr_dev=metrics,old_logged_best=line,old_log_sha256=sha(oldlog),
        initial_prediction_exact=True,initial_rng_exact=True,bpr_original_exact=True,
        shared_softmax_loss_mask_exact=True,existing_aspcf_id_path_exact=True,
        no_llm_no_relation=True,original_dataset_reused=True,dev_candidate_slots=1001,
        inherited_dropout=.1,old_bpr_batch=256,new_softmax_batch=1024,lr=.001)
    save(out/(ds+"_preflight.json"),rec)
    logging.info("PREFLIGHT %s",json.dumps(rec))
    del train,dev,base,ss,eid,c,batch;gc.collect();torch.cuda.empty_cache()

def train_one(ds,out):
    logdir=out/ds;logdir.mkdir()
    checkpoint=logdir/"best.pt"
    handler=logging.FileHandler(logdir/"train.log",encoding="utf8")
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logging.getLogger().addHandler(handler)
    a=settings(ds,checkpoint);save(logdir/"config.json",{k:str(v) if isinstance(v,torch.device) else v for k,v in vars(a).items()})
    utils.init_seed(42);c=corpus(ds);model=Clean(a,c).to(a.device)
    initial_sha=state_sha(model)
    prior=json.loads((out/(ds+"_preflight.json")).read_text())
    assert initial_sha==prior["initial_state_sha256"]
    logging.info("EXPERIMENT_CONFIG %s",vars(a))
    logging.info("INITIAL_STATE_SHA %s",initial_sha)
    data={p:Clean.Dataset(model,c,p) for p in ["train","dev"]}
    for d in data.values():d.prepare()
    for phase in ["dev"]:
        assert all(len(x)==1000 for x in data[phase].data["neg_items"])
    runner=BaseRunner(a);assert runner.main_metric=="NDCG@5" and runner.test_epoch==-1
    torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    runner.train(data) # original full 200-cap training, original early stop, best reload
    elapsed=time.monotonic()-start
    model.actions_after_train()
    checkpoint_sha=sha(checkpoint)
    # One full Dev report on selected checkpoint, not a second selection.
    dev=runner.evaluate(data["dev"],[5,10,20],["HR","NDCG"])
    logging.info("BEST_DEV_FULL %s",json.dumps(dev))
    # Explicitly pre-authorized single Test evaluation, only after best was reloaded.
    test_data=Clean.Dataset(model,c,"test");test_data.prepare()
    assert all(len(x)==1000 for x in test_data.data["neg_items"])
    test=runner.evaluate(test_data,[5,10,20],["HR","NDCG"])
    assert sha(checkpoint)==checkpoint_sha
    logging.info("AUTHORIZED_TEST_ONCE %s",json.dumps(test))
    handler.flush();txt=(logdir/"train.log").read_text()
    epoch_rows=re.findall(r"Epoch\s+(\d+)\s+loss=([0-9.eE+-]+).*?dev=\(([^)]+)\)",txt)
    best_line=re.findall(r"Best Iter\(dev\)=.*",txt)[-1]
    best_epoch=int(re.search(r"Best Iter\(dev\)=\s*(\d+)",best_line).group(1))
    manifest={str(p.relative_to(ROOT)):sha(p) for p in [
        ROOT/"models/sequential/LLMMIRec.py",ROOT/"models/sequential/LLMMIRecCleanSoftmax.py",
        ROOT/"models/sequential/LLMMIRecASPCFSampledSoftmax.py",ROOT/"models/sequential/llmmi_components.py",
        ROOT/"models/BaseModel.py",ROOT/"helpers/BaseRunner.py",ROOT/"helpers/SeqReader.py",
        ROOT/"tools/run_clean_id_softmax.py"]}
    assets={str((ROOT/"data"/ds/n).relative_to(ROOT)):sha(ROOT/"data"/ds/n) for n in ["train.csv","dev.csv","test.csv","SeqReader.pkl"]}
    rec=dict(dataset=ds,status="complete",dev=dev,test=test,test_evaluations=1,
        parameters=model.count_variables(),best_epoch=best_epoch,end_epoch=int(epoch_rows[-1][0]),
        train_dev_seconds=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),initial_state_sha256=initial_sha,
        checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,
        source_sha256=manifest,data_sha256=assets,epochs=[dict(epoch=int(e),loss=float(l),dev=s) for e,l,s in epoch_rows],
        dev_samples=len(data["dev"]),test_samples=len(test_data),train_samples=len(data["train"]),
        config=vars(a)|{"device":str(a.device)},best_line=best_line)
    save(logdir/"summary.json",rec)
    logging.info("COMPLETE %s",json.dumps({k:v for k,v in rec.items() if k not in ["epochs","config","source_sha256","data_sha256"]}))
    logging.getLogger().removeHandler(handler);handler.close()
    del data,test_data,model,c;gc.collect();torch.cuda.empty_cache()
    return rec

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--phase",choices=["preflight","train"],required=True)
    opt=p.parse_args();out=Path(opt.output).resolve()
    assert Path.cwd().resolve()==ROOT and out.is_relative_to(ROOT)
    assert torch.cuda.is_available()
    out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(levelname)s:%(name)s:%(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    logging.info("GPU %s; Python=%s Torch=%s CUDA=%s",torch.cuda.get_device_name(0),sys.version,torch.__version__,torch.version.cuda)
    if opt.phase=="preflight":
        for ds in ["beauty","ml-1m"]:
            if not (out/(ds+"_preflight.json")).exists():preflight(ds,out)
    else:
        assert all((out/(ds+"_preflight.json")).is_file() for ds in ["beauty","ml-1m"])
        for ds in ["beauty","ml-1m"]:train_one(ds,out)
        save(out/"all_complete.json",dict(status="complete",datasets=["beauty","ml-1m"],test_evaluations_per_dataset=1))
if __name__=="__main__":main()
