"""Isolated V2 utilities. Never sanitize nonfinite vectors into usable assets."""
import hashlib, json, os, random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def new_path(path):
    p = Path(path).resolve()
    if ROOT not in p.parents:
        raise ValueError("Outputs must stay inside independent V2 root")
    if p.exists():
        raise FileExistsError(p)
    return p

def write_new(path, data):
    p = new_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("xb") as f:
        f.write(data)

def json_new(path, obj):
    write_new(path, (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode())

def load_mapping(path):
    m = {str(k):int(v) for k,v in json.load(open(path, encoding="utf-8")).items()}
    if sorted(m.values()) != list(range(1, len(m)+1)):
        raise ValueError("Mapping must be bijective contiguous 1..N")
    return m

def load_prompts(path, mapping):
    rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    seen = set()
    for r in rows:
        i = int(r["item_id"])
        if i in seen or mapping.get(str(r["item"])) != i:
            raise ValueError("Duplicate or mismapped movie ID")
        if not r["input"].strip() or r.get("target") != "":
            raise ValueError("Expect item metadata prompts with empty target")
        seen.add(i)
    if seen != set(mapping.values()):
        raise ValueError("Missing catalog item")
    return sorted(rows, key=lambda r:int(r["item_id"]))

def read_vectors(path, mapping, dim=4096):
    table = np.zeros((len(mapping)+1, dim), dtype=np.float32)
    seen = set()
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        i = int(r["item_id"])
        if i not in mapping.values() or i in seen:
            raise ValueError("Duplicate/out-of-range vector ID")
        if "item" not in r or mapping.get(str(r["item"])) != i:
            raise ValueError("Vector needs matching raw movie ID")
        v = np.asarray(r["hidden_states"], dtype=np.float32)
        if v.shape != (dim,) or not np.isfinite(v).all() or not np.any(v):
            raise ValueError("Invalid/nonfinite/zero vector for item "+str(i))
        table[i] = v
        seen.add(i)
    if seen != set(mapping.values()):
        raise ValueError("Missing vectors: no zero filling")
    return table

def dropout_feature(text, ratio, rng=random):
    if not 0 <= ratio < 1:
        raise ValueError("Invalid attribute dropout")
    instruction, body = text.split("\n", 1)
    attrs = [x.strip() for x in body.split(";") if x.strip()]
    if not attrs:
        raise ValueError("No real attribute")
    keep = max(1, len(attrs)-int(len(attrs)*ratio))
    result = []
    for _ in range(2):
        copy = attrs.copy()
        rng.shuffle(copy)
        result.append(instruction+"\n"+"; ".join(copy[:keep])+";")
    return tuple(result)

def masked_mean(hidden, mask):
    import torch
    if mask.shape != hidden.shape[:2] or not (mask.sum(1)>0).all():
        raise ValueError("Empty or mismatched pooling mask")
    if not torch.isfinite(hidden).all():
        raise FloatingPointError("Nonfinite token hidden states; do not mask/sanitize")
    # FP32 reduction avoids FP16 overflow. No padding-derived denominator.
    fmask = mask.to(torch.float32).unsqueeze(-1)
    result = (hidden.float()*fmask).sum(1)/fmask.sum(1)
    if not torch.isfinite(result).all() or (result.norm(dim=1)==0).any():
        raise FloatingPointError("Invalid pooled vectors")
    return result

def snapshot_paths(paths):
    return {str(p):{"bytes":Path(p).stat().st_size,"sha256":sha256(p)} for p in paths}

def seed_all(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_config(path):
    c=json.load(open(path,encoding="utf-8"))
    return {k:v.replace("@V2_ROOT@",str(ROOT)) if isinstance(v,str) else v for k,v in c.items()}
