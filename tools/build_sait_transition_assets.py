"""Build only a small, ordered train-only transition asset; never regenerate PCA/prototypes."""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def inside(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Asset path must resolve inside this project")
    return path

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def normalized_centers(centers):
    centers = np.asarray(centers, dtype=np.float32)
    return centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)

def validate_prototypes(proto):
    q = np.asarray(proto["soft_assignments"], dtype=np.float32)
    centers = np.asarray(proto["centers"], dtype=np.float32)
    if (q.ndim != 2 or centers.shape != (q.shape[1], 512)
            or proto["semantic_rank"] != 512 or q.shape[1] != 32
            or not np.isfinite(q).all() or not np.isfinite(centers).all()
            or (q < 0).any() or np.any(q[0] != 0)
            or not np.allclose(q[1:].sum(1), 1, atol=1e-5)):
        raise ValueError("Invalid aligned 32-state semantic prototype asset")
    return q, centers

def semantic_prior(centers, temperature=.1):
    c = normalized_centers(centers).astype(np.float64)
    logits = c @ c.T / temperature
    logits -= logits.max(1, keepdims=True)
    prior = np.exp(logits)
    return prior / prior.sum(1, keepdims=True)

def smooth(counts, prior, strength=10.):
    return ((counts + strength * prior) /
            (counts.sum(-1, keepdims=True) + strength)).astype(np.float32)

def build(train, proto, strength=10., chunk=16384):
    if not np.isfinite(strength) or strength <= 0 or chunk < 1:
        raise ValueError("Invalid smoothing/chunk")
    q, centers = validate_prototypes(proto)
    data = train.loc[:, ["user_id", "item_id", "time"]].sort_values(
        ["user_id", "time"], kind="mergesort")
    uid = data.user_id.to_numpy(dtype=np.int64)
    ids = data.item_id.to_numpy(dtype=np.int64)
    if len(ids) and ((ids < 1).any() or (ids >= len(q)).any() or (uid < 0).any()):
        raise ValueError("Training IDs do not match semantic asset")
    same_user = uid[1:] == uid[:-1]
    left, right, fold = ids[:-1][same_user], ids[1:][same_user], uid[:-1][same_user] % 3
    counts = np.zeros((3, 32, 32), dtype=np.float64)
    # Three folds and bounded BLAS chunks, no per-record Python loop.
    for f in range(3):
        source, target = left[fold == f], right[fold == f]
        for start in range(0, len(source), chunk):
            a = q[source[start:start+chunk]].astype(np.float64)
            b = q[target[start:start+chunk]].astype(np.float64)
            counts[f] += a.T @ b
    total = counts.sum(0)
    prior = semantic_prior(centers)
    return dict(version="sait_v1", assignments=q, centers=centers,
                counts_by_fold=counts, semantic_prior=prior.astype(np.float32),
                train_transition=np.stack([smooth(total-counts[f], prior, strength)
                                           for f in range(3)]),
                eval_transition=smooth(total, prior, strength),
                meta=dict(folds=3, fold_rule="user_id % 3", states=32,
                          semantic_rank=512, smoothing=strength, prior_temperature=.1,
                          train_rows=len(ids), adjacent_pairs=len(left),
                          train_users=int(len(np.unique(uid))),
                          pairs_by_fold=np.bincount(fold, minlength=3).tolist(),
                          inputs="train.csv + frozen PCA-only prototypes; no dev/test"))

def construct(dataset, output):
    base = inside(ROOT / "data" / dataset)
    train_path = inside(base / "train.csv")
    proto_path = inside(base / "handled/llmmi_proto32_sr512.pkl")
    pca_path = inside(base / "handled/llm_table_pca1536.pkl")
    out = inside(output)
    if out.exists():
        raise FileExistsError("Never overwrite a transition asset")
    with proto_path.open("rb") as stream:
        proto = pickle.load(stream)
    q, centers = validate_prototypes(proto)
    with pca_path.open("rb") as stream:
        table = np.asarray(pickle.load(stream), dtype=np.float32)
    if table.shape != (len(q), 1536) or not np.isfinite(table).all():
        raise ValueError("PCA item mapping mismatch")
    c = normalized_centers(centers)
    temperature = float(proto["temperature"])
    # Check every assignment against the frozen item-only semantic asset, in bounded chunks.
    for start in range(1, len(q), 2048):
        z = table[start:start+2048, :512]
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
        logits = z @ (np.asarray(centers) /
                 (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)).T / temperature
        logits -= logits.max(1, keepdims=True)
        probabilities = np.exp(logits); probabilities /= probabilities.sum(1, keepdims=True)
        if not np.allclose(probabilities, q[start:start+2048], atol=2e-5, rtol=1e-4):
            raise ValueError("Prototype assignments do not match frozen PCA item IDs")
    train = pd.read_csv(train_path, sep="\t", usecols=["user_id", "item_id", "time"])
    asset = build(train, proto)
    asset["meta"].update(train_sha256=digest(train_path), prototype_sha256=digest(proto_path),
                         pca_sha256=digest(pca_path), builder_sha256=digest(__file__),
                         dataset=dataset)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as stream:
        pickle.dump(asset, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps(dict(path=str(out), **asset["meta"]), ensure_ascii=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["beauty", "ml-1m"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    construct(args.dataset, args.output)
