"""Fail-closed raw QA and ONE Base-fitted Train-catalog PCA for both variants."""
import argparse,json,csv
from pathlib import Path
import numpy as np
from common import load_mapping,read_vectors,sha256,json_new,new_path,snapshot_paths,table_quality,code_manifest,environment_manifest
from sklearn.decomposition import PCA

def train_ids(path,mapping):
    with open(path,encoding="utf-8") as f:
        ids=sorted({int(r["item_id"]) for r in csv.DictReader(f,delimiter="\t")})
    if not ids or not set(ids)<=set(mapping.values()):raise ValueError("Bad Train IDs")
    return ids

def fit_common(base,rich,ids,dim):
    # No labels: fixed item content. Fit on Base Train catalog only.
    model=PCA(n_components=dim,svd_solver="full",whiten=False,random_state=42)
    model.fit(base[ids].astype(np.float64))
    def apply(a):
        out=np.zeros((len(a),dim),np.float32)
        out[1:]=model.transform(a[1:].astype(np.float64)).astype(np.float32)
        table_quality(out,name="PCA",min_norm=1e-4)
        return out
    return model,apply(base),apply(rich)

def main():
    p=argparse.ArgumentParser();p.add_argument("--base-dir",required=True)
    p.add_argument("--rich-dir");p.add_argument("--mapping",required=True)
    p.add_argument("--train",required=True);p.add_argument("--out",required=True)
    p.add_argument("--dim",type=int,default=1536);p.add_argument("--execute",action="store_true")
    a=p.parse_args()
    if not a.execute:
        print("DRY RUN: validation/PCA require explicit later authorization via --execute");return
    out=new_path(a.out);mapping=load_mapping(a.mapping)
    dirs=[Path(a.base_dir),Path(a.rich_dir or a.base_dir)]
    complete=[json.load(open(x/"COMPLETE.json")) for x in dirs]
    manifest=[json.load(open(x/"manifest.json")) for x in dirs]
    if any(c.get("kind")!="full_catalog" for c in complete):raise ValueError("Only complete full-catalog encodings accepted")
    if complete[0]["encoder_key"]!=complete[1]["encoder_key"]:
        raise ValueError("Base/Rich use different encoders/protocol")
    if any(x["inputs"][str(Path(a.mapping).resolve())]["sha256"]!=sha256(a.mapping) for x in manifest):
        raise ValueError("Mapping changed")
    for d,c in zip(dirs,complete):
        if sha256(d/"raw.jsonl")!=c["raw_sha256"]:raise ValueError("Raw changed")
    base,rich=[read_vectors(x/"raw.jsonl",mapping) for x in dirs]
    ids=train_ids(a.train,mapping)
    if a.dim>min(len(ids),base.shape[1]):raise ValueError("PCA dim exceeds Train catalog rank bound")
    model,b,r=fit_common(base,rich,ids,a.dim)
    out.mkdir(parents=True)
    # No unsafe pickle, and exact common coordinate system saved with explicit mean.
    np.savez(out/"pca_model.npz",mean=model.mean_,components=model.components_,
             explained_variance=model.explained_variance_,train_item_ids=ids)
    np.save(out/"base_pca.npy",b,allow_pickle=False)
    if a.rich_dir:np.save(out/"rich_pca.npy",r,allow_pickle=False)
    import sklearn
    json_new(out/"manifest.json",{"fit":"Base metadata, Train catalog IDs only","code":code_manifest(),"environment":environment_manifest(),
        "quality_base":table_quality(b,name="base_pca"),"quality_rich":table_quality(r,name="rich_pca"),
        "pca_model_sha256":sha256(out/"pca_model.npz"),"base_sha256":sha256(out/"base_pca.npy"),
        "rich_sha256":sha256(out/"rich_pca.npy") if a.rich_dir else None,
        "sklearn":sklearn.__version__,"numpy":np.__version__,"svd_solver":"full",
        "whiten":False,"random_state":42,"dim":a.dim,
        "inputs":snapshot_paths([dirs[0]/"raw.jsonl",dirs[1]/"raw.jsonl",a.mapping,a.train]),
        "nearzero_valid_rows_base":int((np.linalg.norm(b[1:],axis=1)<1e-4).sum()),
        "nearzero_valid_rows_rich":int((np.linalg.norm(r[1:],axis=1)<1e-4).sum())})
if __name__=="__main__":main()
