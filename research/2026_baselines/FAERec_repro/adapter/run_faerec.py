"""FAERec-SASRec official-core reproduction. Train/Dev only; no Test reader."""
import argparse, hashlib, importlib, json, os, pickle, random, signal, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[2]
UP = BASE / "upstream"
assert BASE == ROOT / "research/2026_baselines/FAERec_repro"
torch.set_num_threads(2)

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1048576), b""): h.update(block)
    return h.hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

def official():
    # Both repositories call their modules models/utils; scope only these imports.
    sys.path.insert(0, str(ROOT))
    from helpers.BaseRunner import BaseRunner
    names = ("models", "utils", "modules", "llm", "trainers")
    saved = {n: sys.modules.get(n) for n in names}
    for n in names: sys.modules.pop(n, None)
    sys.path.insert(0, str(UP / "FAERec/src"))
    try:
        models = importlib.import_module("models")
        trainers = importlib.import_module("trainers")
        utils = importlib.import_module("utils")
    finally:
        sys.path.pop(0)
        for n in names:
            if saved[n] is None: sys.modules.pop(n, None)
            else: sys.modules[n] = saved[n]
    return models, trainers, utils, BaseRunner

def read_data(ds):
    folder = ROOT / "data" / ds
    train = pd.read_csv(folder / "train.csv", sep="\t")
    dev = pd.read_csv(folder / "dev.csv", sep="\t")
    assert train.item_id.gt(0).all() and dev.item_id.gt(0).all()
    assert dev.groupby("user_id").size().max() == 1
    train = train.sort_values(["time", "user_id"], kind="mergesort")
    seqs = {int(u): g.item_id.to_numpy(np.int64) for u,g in train.groupby("user_id", sort=False)}
    last = train.groupby("user_id").time.max()
    assert not dev.user_id.map(last).isna().any()
    assert (dev.user_id.map(last) <= dev.time).all()
    emb_path = folder / "handled/llm_table_pca1536.pkl"
    with emb_path.open("rb") as f: table = np.asarray(pickle.load(f), dtype=np.float32)
    assert table.ndim == 2 and table.shape[1] == 1536
    assert np.isfinite(table).all() and np.count_nonzero(table[0]) == 0
    assert int(train.item_id.max()) < len(table) and int(dev.item_id.max()) < len(table)
    negs = np.stack([np.fromstring(v.strip()[1:-1], dtype=np.int64, sep=",") for v in dev.neg_items])
    assert negs.shape == (len(dev), 1000)
    candidates = np.concatenate([dev.item_id.to_numpy(np.int64)[:,None], negs], axis=1)
    assert candidates.min() > 0 and candidates.max() < len(table)
    history = np.zeros((len(dev),20), dtype=np.int64)
    for i,u in enumerate(dev.user_id):
        h = seqs[int(u)][-20:]
        history[i,-len(h):] = h
    counts = train.item_id.value_counts().to_dict()
    metadata = dict(train_rows=len(train), train_users=len(seqs), dev_users=len(dev),
        train_sequence_users=sum(len(s)>=2 for s in seqs.values()),
        dev_history_min=int((history>0).sum(1).min()), dev_history_max=int((history>0).sum(1).max()),
        candidates=1001, candidate_duplicates_preserved=True, test_read=False,
        source_embedding=str(emb_path), embedding_sha256=sha(emb_path),
        train_sha256=sha(folder/"train.csv"), dev_sha256=sha(folder/"dev.csv"),
        embedding_shape=list(table.shape), time_violations=0)
    return seqs, counts, history, candidates, table, metadata

class TrainSequences(Dataset):
    def __init__(self, seqs, n_items, sample_negative):
        self.seqs = [s for s in seqs.values() if len(s)>=2]
        self.clicked = [set(map(int,s)) for s in self.seqs]
        self.n_items = n_items
        self.sample_negative = sample_negative
        assert all(len(s)<n_items-1 for s in self.clicked)
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx):
        s = self.seqs[idx][-21:]
        x, y = s[:-1], s[1:]
        neg = [self.sample_negative(self.clicked[idx], self.n_items) for _ in x]
        out = []
        for value in (x,y,neg):
            z = np.zeros(20, dtype=np.int64)
            z[-len(value):] = value
            out.append(torch.from_numpy(z))
        return tuple(out)

def setup(opts):
    models, trainers, utils, Runner = official()
    utils.set_seed(opts.seed)
    seqs, counts, hist, cand, table, meta = read_data(opts.dataset)
    asset_root = BASE / "runs/assets" / opts.dataset
    asset_dir = asset_root / "llm_emb" / opts.dataset
    asset_dir.mkdir(parents=True, exist_ok=True)
    converted = asset_dir / "itm_emb_np.npy"
    if converted.exists():
        check = np.load(converted, mmap_mode="r")
        assert check.shape == table[1:].shape and np.array_equal(check, table[1:])
    else:
        with converted.open("xb") as f: np.save(f, table[1:])
    # Official loaders prepend padding and append a reserved dummy row.
    # CWD keeps their optional PCA cache writes strictly inside the new directory.
    os.chdir(asset_root)
    cfg = dict(data_name=opts.dataset, hidden_size=128, num_hidden_layers=2,
        num_attention_heads=2, hidden_act="gelu", attention_probs_dropout_prob=.4,
        hidden_dropout_prob=.4, initializer_range=.02, max_seq_length=20,
        no_filters=True, gate_type="dimension", item_size=len(table)+1,
        cuda_condition=torch.cuda.is_available(), no_cuda=False, do_eval=False,
        lr=.001, weight_decay=0., item_tou=.05, cl_weight=.2,
        w_min=.9, w_max=1., warmup_epochs=0, period=30,
        log_freq=1, log_file=str(opts.run_dir/"official.log"),
        item_popularity={int(k):int(v) for k,v in counts.items()},
        head_items=set(), tail_items=set())
    assert cfg["cuda_condition"] and os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    args = SimpleNamespace(**cfg, seed=opts.seed)
    model = models.SASRecModel(args)
    trainer = trainers.SASRecTrainer(model, None, None, None, args)
    assert torch.equal(model.llm_item_emb[:len(table)].cpu(), torch.from_numpy(table))
    assert model.llm_item_emb.shape[0] == args.item_size
    assert model.llm_item_emb[-1].count_nonzero().item() == 0
    dataset = TrainSequences(seqs, len(table), utils.neg_sample)
    loader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=True,
                        num_workers=0, drop_last=False, pin_memory=True)
    config = {k:v for k,v in cfg.items() if k not in ("head_items","tail_items","item_popularity")}
    config.update(seed=opts.seed, batch_size=opts.batch_size, eval_batch_size=opts.eval_batch_size,
        max_epoch=opts.max_epoch, early_stop=opts.early_stop, max_hours=opts.max_hours,
        loss="official_sequence_BCE+item_InfoNCE+head_tail_feature_alignment",
        train_unit="one_train_user_sequence_last_21_items_to_20_causal_pairs",
        selection="Dev NDCG@5 strict improvement, first maximum",
        gpu_physical=1, test_enabled=False,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        upstream_commit=json.loads((BASE/"adapter/provenance.json").read_text())["commit"], optimizer="Adam", adam_betas=[.9,.999], adam_eps=1e-8)
    print("CONFIG "+json.dumps(config),flush=True)
    write_json(opts.run_dir/"config.json",config)
    write_json(opts.run_dir/"data_protocol.json",meta)
    return model,trainer,loader,hist,cand,Runner,config

def objective(model, trainer, batch, epoch):
    x,p,n = [v.cuda(non_blocking=True) for v in batch]
    out = model.transformer_encoder(x)
    rec = trainer.cross_entropy(out,p,n,x)
    items = torch.cat([x,n],dim=1)
    wi,wf = trainer.get_curriculum_weights(epoch, trainer.args.warmup_epochs,
        trainer.args.period, trainer.args.w_min, trainer.args.w_max)
    item = trainer.cl_loss(items)
    feature = trainer.fcl_loss(items)
    loss = rec + trainer.args.cl_weight*(wi*item+wf*feature)
    return loss,rec,item,feature

@torch.no_grad()
def predict(model, hist, candidates, batch_size):
    model.eval()
    # Exact official predict_full candidate representation, computed once in eval.
    item = model.LayerNorm(model.item_embeddings.weight + model.llm_mapper(model.llm_item_emb))
    predictions = np.empty(candidates.shape, dtype=np.float32)
    for lo in range(0,len(hist),batch_size):
        x = torch.from_numpy(hist[lo:lo+batch_size]).cuda()
        c = torch.from_numpy(candidates[lo:lo+batch_size]).cuda()
        h = model.transformer_encoder(x)[:,-1,:]
        predictions[lo:lo+len(x)] = (h[:,None,:]*item[c]).sum(-1).cpu().numpy()
    assert np.isfinite(predictions).all()
    return predictions

def smoke(opts):
    signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TimeoutError("smoke >175s")))
    signal.alarm(175)
    start=time.monotonic()
    model,tr,loader,h,c,Runner,cfg = setup(opts)
    model.train(); batch=next(iter(loader)); tr.optim.zero_grad()
    losses=objective(model,tr,batch,0)
    assert all(torch.isfinite(v).all() for v in losses)
    component_gradient_l2=[]
    for component in losses[1:]:
        gg=torch.autograd.grad(component,tuple(model.parameters()),retain_graph=True,allow_unused=True)
        gg=[g for g in gg if g is not None]
        assert gg and all(torch.isfinite(g).all() for g in gg)
        norm=float(torch.sqrt(sum((g.detach()**2).sum() for g in gg)).item())
        assert norm>0
        component_gradient_l2.append(norm)
    losses[0].backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    gradnorm=float(torch.sqrt(sum((g.detach()**2).sum() for g in grads)).item())
    tr.optim.step()
    # Original duplicate-item off-diagonal masking, including diagonal positives.
    a=torch.randn(3,128,device="cuda");b=torch.randn(3,128,device="cuda");ids=torch.tensor([1,1,2],device="cuda")
    sim=torch.nn.functional.normalize(a,dim=1)@torch.nn.functional.normalize(b,dim=1).T/tr.args.item_tou
    mask=(ids[:,None]==ids[None,:]) & ~torch.eye(3,device="cuda",dtype=torch.bool)
    expected=torch.nn.functional.cross_entropy(sim.masked_fill(mask,-torch.inf),torch.arange(3,device="cuda"))
    assert torch.allclose(tr.item_contrast_loss(a,b,ids), expected)
    pred=predict(model,h[:64],c[:64],opts.eval_batch_size)
    with torch.no_grad():
        small=torch.from_numpy(h[:2]).cuda()
        native=tr.predict_full(model.transformer_encoder(small)[:,-1,:])
        chosen=native.gather(1,torch.from_numpy(c[:2]).cuda()).cpu().numpy()
    assert np.allclose(pred[:2],chosen,rtol=1e-5,atol=1e-4)
    cp=opts.run_dir/"smoke_state.pt";torch.save(model.state_dict(),cp)
    with torch.no_grad(): next(model.parameters()).add_(.123)
    model.load_state_dict(torch.load(cp,map_location="cuda",weights_only=True),strict=True)
    after=predict(model,h[:64],c[:64],opts.eval_batch_size)
    assert np.array_equal(pred,after)
    assert np.allclose(predict(model,h[:1],c[:1],1), pred[:1], rtol=1e-5, atol=1e-5)
    report=dict(status="passed",dataset=opts.dataset,losses=[float(v.detach()) for v in losses],
        gradient_l2=gradnorm,component_gradient_l2=component_gradient_l2,checks=["train_dev_time","embedding_full_row_equality","all_original_losses_backward",
        "duplicate_item_contrastive_mask","strict_checkpoint_roundtrip","single_user_eval_independence"],
        partial_dev_users=len(pred),partial_dev_metrics=Runner.evaluate_method(pred,[5,10,20],["HR","NDCG"]),
        elapsed_seconds=time.monotonic()-start,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,checkpoint_sha256=sha(cp),
        parameters=cfg["trainable_parameters"],gpu_name=torch.cuda.get_device_name(0),test_read=False)
    write_json(opts.run_dir/"smoke.json",report); print(json.dumps(report),flush=True)
    signal.alarm(0)

def train(opts):
    start=time.monotonic()
    model,tr,loader,h,c,Runner,cfg=setup(opts)
    best=-float("inf");best_epoch=0;best_dev=None;bad=0;end=0;reason="max_epoch"
    epoch_seconds=[];checkpoint=opts.run_dir/"best.pt"
    for ep in range(opts.max_epoch):
        model.train(); accum=torch.zeros(4,device="cuda");steps=0;ep_start=time.monotonic()
        interrupted=False
        for batch in loader:
            if time.monotonic()-start >= opts.max_hours*3600:
                reason="time_budget";interrupted=True;break
            tr.optim.zero_grad()
            losses=objective(model,tr,batch,ep); losses[0].backward()
            tr.optim.step()
            accum+=torch.stack([v.detach() for v in losses]);steps+=1
        if interrupted: break
        torch.cuda.synchronize();train_seconds=time.monotonic()-ep_start
        scores=predict(model,h,c,opts.eval_batch_size)
        dev=Runner.evaluate_method(scores,[5,10,20],["HR","NDCG"])
        mean=(accum/max(steps,1)).cpu().tolist()
        if not np.isfinite(mean).all(): raise RuntimeError("nonfinite training loss")
        end=ep+1
        if dev["NDCG@5"]>best:
            best=dev["NDCG@5"];best_epoch=end;best_dev=dev;bad=0
            torch.save(model.state_dict(),checkpoint)
            write_json(opts.run_dir/"best.json",dict(epoch=end,dev=dev))
        else: bad+=1
        seconds=time.monotonic()-ep_start;epoch_seconds.append(seconds)
        row=dict(epoch=end,loss=mean,dev=dev,train_seconds=train_seconds,
            epoch_seconds=seconds,elapsed_seconds=time.monotonic()-start,
            best_epoch=best_epoch,bad_epochs=bad,steps=steps,
            projected_200_epoch_seconds=float(np.mean(epoch_seconds))*200)
        with (opts.run_dir/"epochs.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
        print(json.dumps(row),flush=True)
        if bad>=opts.early_stop: reason="early_stop";break
        if end==3 and np.mean(epoch_seconds)*opts.max_epoch > opts.max_hours*3600:
            reason="projected_over_budget";break
    if not checkpoint.exists(): raise RuntimeError("Stopped before a complete epoch/best checkpoint")
    before=sha(checkpoint)
    model.load_state_dict(torch.load(checkpoint,map_location="cuda",weights_only=True),strict=True)
    final=Runner.evaluate_method(predict(model,h,c,opts.eval_batch_size),[5,10,20],["HR","NDCG"])
    assert all(abs(final[k]-best_dev[k])<1e-12 for k in final)
    assert sha(checkpoint)==before
    report=dict(status="complete" if reason in ("max_epoch","early_stop") else "budget_stopped",
        stop_reason=reason,dataset=opts.dataset,seed=opts.seed,best_epoch=best_epoch,end_epoch=end,
        best_dev=best_dev,reloaded_dev=final,checkpoint=str(checkpoint),checkpoint_sha256=before,
        checkpoint_unchanged_by_dev=True,parameters=cfg["trainable_parameters"],
        total_seconds=time.monotonic()-start,mean_epoch_seconds=float(np.mean(epoch_seconds)),
        projected_200_epoch_seconds=float(np.mean(epoch_seconds))*200,
        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
        test_evaluated=False,config=str(opts.run_dir/"config.json"))
    write_json(opts.run_dir/"summary.json",report);print("SUMMARY "+json.dumps(report),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",choices=["beauty","ml-1m"],required=True)
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--eval-batch-size",type=int,default=128)
    p.add_argument("--max-epoch",type=int,default=200)
    p.add_argument("--early-stop",type=int,default=10)
    p.add_argument("--max-hours",type=float,default=4)
    o=p.parse_args();o.run_dir=o.run_dir.resolve()
    assert o.run_dir.is_relative_to(BASE/"runs")
    o.run_dir.mkdir(parents=True,exist_ok=False)
    try:
        smoke(o) if o.smoke else train(o)
    except Exception as exc:
        write_json(o.run_dir/"failure.json",dict(type=type(exc).__name__,error=str(exc),test_evaluated=False))
        raise
