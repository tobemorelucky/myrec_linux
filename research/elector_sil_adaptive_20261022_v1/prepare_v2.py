"""Train-only asset preparation and mathematical checks; no optimizer on real data."""
import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import sys, json, time, signal, pickle, importlib.util
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT = Path("/home/yyx/hzgcode/pom2.0").resolve()
OUT = Path(__file__).resolve().parent / "prep_v2"
sys.path.insert(0, str(ROOT))
from models.sequential.LLMMIRecASPCFAdaptiveSIL import (
    SILRecomposer, segment_ids, array_sha, file_sha)
PREV = ROOT / "research/frozen_readout_softmax_20261022_v1/probe.py"
assert file_sha(PREV) == "fae7c3155477448075e58d94212ad637bc420d26449b3137bbcc953336e4c81d"
spec = importlib.util.spec_from_file_location("sil_readonly_helpers", PREV)
reuse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reuse)
CONFIG = dict(seed=42, threshold_quantile=.65, semantic_dimension=1536, history_max=20,
              min_segment_length=2, max_segment_length=6, fixed_segment_length=4,
              recent=3, heads=2, steps=16, batch_size=1024, lr=.0001,
              optimizer="Adam", weight_decay=1e-6, lambda_pop=.30,
              threshold_selection="one prespecified Train-only adjacent-gap quantile; no Dev search",
              loss="unchanged original raw filtered batch Softmax + relation .01",
              prior="original locked Train-RMS .30, applied only to exported scoring as previously",
              screen="same existing best-Dev initialization, all original parameters unfrozen except LLM buffer",
              formal_training="not authorized; paired fresh initialization required for structure attribution",
              no_Test=True, no_full_training=True)


def write(name, value):
    path = OUT / name
    assert path.parent.resolve() == OUT
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def math_tests():
    torch.manual_seed(42)
    records = []
    for mode in ("fixed", "semantic"):
        block = SILRecomposer().double()
        assert sum(p.numel() for p in block.parameters()) == 37504
        lengths = torch.tensor([0, 1, 4, 20])
        x = torch.randn(4, 20, 64, dtype=torch.float64)
        gaps = torch.rand(4, 20, dtype=torch.float64) * 2
        ids, valid = segment_ids(gaps, lengths, mode, 1.)
        for K in (1, 2, 4, 7):
            v = torch.randn(4, K, 64, dtype=torch.float64)
            delta, info = block(x, v, ids, valid, lengths)
            assert delta.shape == v.shape and torch.count_nonzero(delta) == 0
            assert torch.isfinite(info["memory_attention"]).all()
        empty, _ = block(x[:, :0], v, ids[:, :0], valid[:, :0], lengths * 0)
        assert torch.count_nonzero(empty) == 0
        with torch.no_grad():
            block.output.weight.normal_(0, .02)
        v = torch.randn(4, 4, 64, dtype=torch.float64)
        delta, info = block(x, v, ids, valid, lengths)
        changed = x.clone()
        changed[~valid] = 1000
        other, _ = block(changed, v, ids, valid, lengths)
        assert torch.equal(delta, other) and torch.count_nonzero(delta[:2]) == 0
        perm = torch.tensor([2, 0, 3, 1])
        other, _ = block(x, v[:, perm], ids, valid, lengths)
        assert torch.allclose(delta[:, perm], other, atol=1e-12, rtol=1e-12)
        # A suffix cannot change prior boundaries or local hidden states.
        xx = torch.randn(1, 20, 64, dtype=torch.float64, requires_grad=True)
        gg = torch.rand(1, 20, dtype=torch.float64) * 2
        full_ids, full_valid = segment_ids(gg, torch.tensor([20]), mode, 1.)
        prefix_ids, prefix_valid = segment_ids(gg[:, :8], torch.tensor([8]), mode, 1.)
        assert torch.equal(prefix_ids, full_ids[:, :8])
        bank = block.local_bank(xx, full_ids, full_valid)
        prefix = block.local_bank(xx[:, :8], prefix_ids, prefix_valid)
        assert torch.allclose(bank[:, :8], prefix, atol=1e-12, rtol=1e-12)
        gradient = torch.autograd.grad(bank[:, 7].square().sum(), xx)[0]
        assert torch.count_nonzero(gradient[:, 8:]) == 0
        yy = xx.detach().clone()
        yy[:, 8:] += 100
        altered_bank = block.local_bank(yy, full_ids, full_valid)
        assert torch.equal(bank[:, :8], altered_bank[:, :8])
        # Encoded positions cannot read any other segment.
        independent = block.local_bank(xx.detach(), torch.arange(20)[None], full_valid)
        changed_x = xx.detach().clone()
        changed_x[:, 0] += 100
        independent2 = block.local_bank(changed_x, torch.arange(20)[None], full_valid)
        assert torch.equal(independent[:, 1:], independent2[:, 1:])
        # All new active parameter groups must receive finite nonzero gradients.
        block.zero_grad()
        delta, _ = block(x, v, ids, valid, lengths)
        (delta * torch.randn_like(delta)).sum().backward()
        gradients = {n: float(p.grad.norm()) for n, p in block.named_parameters()}
        assert all(np.isfinite(a) and a > 0 for a in gradients.values()), gradients
        # Verify that recent guidance and the local encoder change the active read.
        block.use_recent = False
        no_recent, _ = block(x, v, ids, valid, lengths)
        block.use_recent = True
        block.use_local = False
        no_local, _ = block(x, v, ids, valid, lengths)
        assert (no_recent - delta).abs().max() > 1e-8
        assert (no_local - delta).abs().max() > 1e-8
        records.append(dict(mode=mode,added_parameters=37504,K=[1,2,4,7],
                            zero_output_identity=True,padding_and_empty_safe=True,
                            short_length0or1_exact_identity=True,prefix_boundary_invariance=True,
                            local_future_gradient_exact_zero=True,no_cross_segment_read=True,
                            head_permutation_equivariance=True,all_new_parameters_active=True))
    return records


def build_dataset(ds, before):
    start = time.perf_counter()
    paths = {k: Path(v["path"]) for k, v in before["inputs"][ds].items()}
    assert all(file_sha(p) == before["inputs"][ds][k]["sha256"] for k, p in paths.items())
    with paths["cache"].open("rb") as f:
        corpus = pickle.load(f)
    corpus.data_df = {p: corpus.data_df[p] for p in ("train", "dev")}
    frame = corpus.data_df["train"]
    tr = frame.sort_values(["user_id", "position"])
    counts = tr.groupby("user_id").size().reindex(range(corpus.n_users), fill_value=0).to_numpy()
    offsets = np.r_[0, np.cumsum(counts[:-1])]
    items = tr.item_id.to_numpy(np.int64)
    users = tr.user_id.to_numpy(np.int64)
    positions = tr.position.to_numpy(np.int64)
    assert np.array_equal(offsets[users] + positions, np.arange(len(tr)))
    for u in corpus.user_his:
        corpus.user_his[u] = corpus.user_his[u][:int(counts[u])]
    with paths["embedding"].open("rb") as f:
        table = np.asarray(pickle.load(f), dtype=np.float32)
    assert table.shape == (corpus.n_items, 1536) and np.isfinite(table).all()
    assert (np.linalg.norm(table[1:], axis=1) > 0).all()
    # Same normalization order as the model constructor: CPU normalize then move.
    unit = F.normalize(torch.as_tensor(table), dim=-1).to("cuda:0")
    gaps = np.zeros(len(items), dtype=np.float32)
    for off in range(0, len(items), 8192):
        end = min(off + 8192, len(items))
        now = torch.as_tensor(items[off:end], device=unit.device)
        prev = torch.as_tensor(items[np.maximum(np.arange(off, end) - 1, 0)], device=unit.device)
        gaps[off:end] = (1 - (unit[now] * unit[prev]).sum(-1)).clamp(0, 2).cpu().numpy()
    edge = gaps[positions > 0]
    threshold = float(np.quantile(edge, CONFIG["threshold_quantile"]))
    lock = json.loads(paths["lock"].read_text())
    asset = dict(schema="adaptive_sil_train_v1",dataset=ds,n_items=int(corpus.n_items),
                 train_frame_sha256=array_sha(frame[["user_id","item_id","position"]].to_numpy(np.int64)),
                 train_csv_sha256=file_sha(paths["train"]),
                 embedding_sha256=file_sha(paths["embedding"]),semantic_dimension=1536,history_max=20,
                 threshold=threshold,threshold_quantile=CONFIG["threshold_quantile"],
                 threshold_source="distinct adjacent edges from Train, each once; no Dev/Test targets",
                 min_segment_length=2,max_segment_length=6,fixed_segment_length=4,
                 train_item_counts=np.bincount(frame.item_id.to_numpy(np.int64),minlength=corpus.n_items).tolist(),
                 lambda_pop=.30,raw_pop_lambda=lock["raw_lambda"],locked_scale=lock["scale_factor"],
                 locked_prior_sha256=file_sha(paths["lock"]))
    # Freeze the Train asset BEFORE inspecting any Dev segmentation.
    write(ds+"_train_asset.json", asset)
    logs = {}
    for phase, data in (("Train", tr[tr.position > 0]), ("Dev", corpus.data_df["dev"])):
        us = data.user_id.to_numpy(np.int64)
        pos = data.position.to_numpy(np.int64)
        assert (pos <= counts[us]).all()
        if phase == "Dev":
            assert np.array_equal(pos, counts[us])
        lengths = np.minimum(pos, 20)
        totals = {m: np.zeros(6) for m in ("fixed","semantic")}
        for off in range(0, len(data), 8192):
            end = min(off + 8192, len(data))
            ll = torch.as_tensor(lengths[off:end], device=unit.device)
            index = offsets[us[off:end],None] + (pos[off:end]-lengths[off:end])[:,None] + np.arange(20)[None]
            index = np.minimum(index,len(items)-1)
            dd = torch.as_tensor(gaps[index],device=unit.device)
            for mode in totals:
                ids,valid = segment_ids(dd,ll,mode,threshold)
                sizes = dd.new_zeros((len(ll),20)).scatter_add_(1,ids,valid.float())
                segs = (sizes>0).sum(1)
                meaningful = (sizes>=2).sum(1)
                totals[mode] += np.array([len(ll),float(segs.sum()),int((segs>=2).sum()),
                                         int((meaningful>=2).sum()),int((sizes==1).sum()),int((sizes>0).sum())])
        summaries = {}
        for mode,v in totals.items():
            summaries[mode] = dict(mean_segments=float(v[1]/v[0]),two_or_more=float(v[2]/v[0]),
                                   two_non_singleton=float(v[3]/v[0]),singleton_segment_share=float(v[4]/v[5]),
                                   two_non_singleton_given_length4plus=float(v[3]/max((lengths>=4).sum(),1)))
        logs[phase] = dict(rows=len(data),length_mean=float(lengths.mean()),
                          length_quantiles=np.quantile(lengths,[0,.25,.5,.75,1]).tolist(),
                          length_under4=float((lengths<4).mean()),segments=summaries)
    result = dict(dataset=ds,threshold=threshold,adjacent_gap_quantiles=np.quantile(edge,[.1,.5,.65,.9]).tolist(),
                  distribution=logs,seconds=time.perf_counter()-start,
                  short_history_design="segment all observed items; parallel overlapping recent3; no forced fake split",
                  semantic_boundary_validity="content-distance heuristic, not proven behavioral intent boundaries")
    write(ds+"_segmentation.json",result)
    print("SEGMENTATION "+json.dumps(result),flush=True)
    del corpus,unit
    torch.cuda.empty_cache()
    return result


def main():
    signal.alarm(170)
    torch.set_num_threads(4)
    start=time.perf_counter()
    OUT.mkdir(exist_ok=False)
    before=json.loads((OUT.parent/"INITIAL_STATE.json").read_text())
    write("PREREG.json",dict(config=CONFIG,model_sha256=file_sha(ROOT/"models/sequential/LLMMIRecASPCFAdaptiveSIL.py"),
                             prepare_sha256=file_sha(__file__)))
    tests=math_tests()
    print("MATH_PASS",flush=True)
    datasets=[build_dataset(ds,before) for ds in ("beauty","ml-1m")]
    write("PREPARE_COMPLETE.json",dict(math_tests=tests,datasets=[r["dataset"] for r in datasets],
                                       seconds=time.perf_counter()-start))
    print("PREPARE_COMPLETE",flush=True)


if __name__=="__main__":
    main()
