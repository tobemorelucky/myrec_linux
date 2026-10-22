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
        if i not in mapping.values() or i in seen or i != len(seen)+1:
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
    table_quality(table,name="raw")
    return table

def dropout_feature(text,ratio,rng=random):
    if not 0 <= ratio < 1:raise ValueError("Invalid attribute dropout")
    instruction,body=text.split("\n",1)
    attrs=[x.strip() for x in body.split(";") if x.strip()]
    if not attrs:raise ValueError("No real attribute")
    title=[x for x in attrs if x.lower().startswith("title ")]
    if len(title)>1:raise ValueError("Ambiguous title")
    others=[x for x in attrs if x not in title]
    keep=max(min(2,len(attrs)),len(attrs)-int(len(attrs)*ratio))
    result=[]
    for _ in range(2):
        rest=others.copy();rng.shuffle(rest)
        selected=title+rest[:max(0,keep-len(title))]
        if not title:
            rest=attrs.copy();rng.shuffle(rest);selected=rest[:keep]
        result.append(instruction+"\n"+"; ".join(selected)+";")
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

def table_quality(table, min_norm=1e-4, name="table"):
    a=np.asarray(table)
    if a.ndim!=2 or len(a)<2 or not np.isfinite(a).all():raise ValueError(name+": invalid shape/nonfinite")
    if np.any(a[0]!=0):raise ValueError(name+": padding row must be zero")
    norms=np.linalg.norm(a[1:].astype(np.float64),axis=1)
    bad=(np.flatnonzero(norms<=min_norm)+1).tolist()
    if bad:raise ValueError(name+": unexplained zero/nearzero IDs "+str(bad[:20]))
    hashes=[hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for v in a[1:]]
    duplicates=len(hashes)-len(set(hashes))
    if duplicates:raise ValueError(name+": unexplained duplicate vectors "+str(duplicates))
    return {"rows":len(a)-1,"dim":a.shape[1],"min_norm":float(norms.min()),
            "median_norm":float(np.median(norms)),"max_norm":float(norms.max()),
            "near_zero_threshold":min_norm,"duplicate_vectors":duplicates}

def environment_manifest():
    import platform, importlib.metadata as md
    versions={}
    for name in ("torch","transformers","tokenizers","datasets","deepspeed","numpy","scikit-learn","peft","accelerate"):
        try:versions[name]=md.version(name)
        except md.PackageNotFoundError:versions[name]=None
    return {"python":platform.python_version(),"packages":versions}

def code_manifest():
    paths=[]
    for directory in ("adapter","configs","scripts","upstream"):
        base=ROOT/directory
        if base.exists():paths.extend(p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    files={str(p.relative_to(ROOT)):sha256(p) for p in sorted(paths)}
    digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    return {"code_sha256":digest,"files":files}

def encoder_artifacts(config,full_weights=True):
    model=Path(config["model_path"]);token=Path(config.get("tokenizer_path",config["model_path"]))
    index=model/"model.safetensors.index.json"
    if not index.is_file():raise FileNotFoundError("Require local safetensors index")
    names=sorted(set(json.load(index.open())["weight_map"].values()))
    if any(Path(n).name!=n for n in names):raise ValueError("Unsafe weight shard path")
    paths=[model/"config.json",index,token/"tokenizer.json",token/"tokenizer.model",
           token/"tokenizer_config.json",token/"special_tokens_map.json"]
    if full_weights:paths += [model/n for n in names]
    if config.get("adapter_path"):
        ad=Path(config["adapter_path"]);paths += [ad/"adapter_config.json",ad/"adapter_model.bin"]
        if (ad/"v2_extra_trainable.pt").exists():paths.append(ad/"v2_extra_trainable.pt")
    for p in paths:
        if not p.is_file():raise FileNotFoundError(p)
    result=snapshot_paths(paths)
    if full_weights:
        lock=json.load(open(config["weight_lock"]))["files"]
        for n in names:
            if result[str(model/n)]["sha256"]!=lock[n]["sha256"]:raise ValueError("Base weight SHA mismatch "+n)
    return result

def encode_prompt(text,tokenizer,max_length,field_caps=None):
    if max_length<2:raise ValueError("max_length must allow text+EOS")
    clips={}
    patched=text
    for attr in text.split("\n",1)[-1].split(";"):
        value=attr.strip()
        for key,cap in (field_caps or {}).items():
            if value.startswith(key+" is "):
                ids=tokenizer.encode(value,add_special_tokens=False)
                if len(ids)>cap:
                    replacement=tokenizer.decode(ids[:cap],skip_special_tokens=True)
                    patched=patched.replace(value,replacement,1);clips[key]=len(ids)-cap
    ids=tokenizer.encode(patched,add_special_tokens=False)
    return ids[:max_length-1]+[tokenizer.eos_token_id],{
        "tokens_before_total_truncation":len(ids)+1,"total_truncated":len(ids)+1>max_length,
        "field_tokens_clipped":clips}

def restore_extras(model,adapter,allow_legacy=False):
    import torch
    extra=Path(adapter)/"v2_extra_trainable.pt"
    if not extra.exists():
        if allow_legacy:return "legacy_tau_absent_inference_only"
        raise FileNotFoundError("V2 non-LoRA state missing")
    saved=torch.load(extra,map_location="cpu",weights_only=True)
    params=dict(model.named_parameters())
    if not any("tau" in n for n in saved):raise ValueError("Saved tau missing")
    for n,v in saved.items():
        if n not in params or not torch.isfinite(v).all():raise ValueError("Invalid extra state "+n)
        params[n].data.copy_(v.to(device=params[n].device,dtype=params[n].dtype))
    return "restored"

def verify_adapter_loaded(model,adapter):
    import torch
    saved=torch.load(Path(adapter)/"adapter_model.bin",map_location="cpu",weights_only=True)
    checked=0
    for n,p in model.named_parameters():
        if "lora_" not in n:continue
        key=n.replace(".default.",".")
        if key not in saved:raise ValueError("Missing LoRA checkpoint key "+key)
        if not torch.equal(p.detach().cpu(),saved[key].to(dtype=p.dtype)):
            raise ValueError("LoRA weight not restored "+key)
        checked+=1
    if checked!=len(saved) or checked==0:raise ValueError("Unexpected/non-LoRA checkpoint contents")
    return checked
