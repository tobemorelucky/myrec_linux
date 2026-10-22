"""Import ONLY a QA-passed V1 release tar; exact hashes, canonical IDs, lossless pkl format."""
import argparse,hashlib,json,pickle,shutil,sys,tarfile
from pathlib import Path
import numpy as np
ROOT=Path("/home/yyx/hzgcode/pom2.0")
WORK=ROOT/"research/ml1m_embedding_v2_20261020"
DEST=WORK/"imported_assets/v1_rebuilt_20261020"
sys.path.insert(0,str(WORK/"adapter"))
from common import sha256,read_vectors,load_mapping,table_quality
p=argparse.ArgumentParser();p.add_argument("--archive",required=True);p.add_argument("--sha256",required=True);a=p.parse_args()
archive=Path(a.archive).resolve()
assert archive.parent==DEST and DEST.resolve()==DEST and ROOT.resolve()==ROOT
assert sha256(archive)==a.sha256
assert not (DEST/"IMPORT_QA_COMPLETE.json").exists()
with tarfile.open(archive,"r:gz") as tar:
    members=tar.getmembers()
    assert len({m.name for m in members})==len(members)
    for m in members:
        assert m.isfile() and Path(m.name).name==m.name and m.name not in (".","..")
        target=DEST/m.name
        assert target.resolve().parent==DEST and not target.exists()
        with tar.extractfile(m) as src,target.open("xb") as dst:shutil.copyfileobj(src,dst,1<<20)
sums={}
for line in (DEST/"SHA256SUMS").read_text().splitlines():
    digest,name=line.split("  ",1)
    assert len(digest)==64 and Path(name).name==name and name not in sums
    assert sha256(DEST/name)==digest
    sums[name]=digest
release=json.load((DEST/"release.json").open())
assert release["status"]=="QA_PASSED" and release["kind"]=="V1-Rebuilt"
for name,info in release["artifacts"].items():
    assert name in sums and info["sha256"]==sums[name]
    assert (DEST/name).stat().st_size==info["bytes"]
mapping=load_mapping(DEST/"item2id.json")
assert len(mapping)==3706 and sha256(DEST/"item2id.json")==sha256(ROOT/"data/ml-1m/item2id.json")
assert set(mapping.values())==set(range(1,3707))
raw=np.load(DEST/"table_raw.npy",allow_pickle=False)
again=read_vectors(DEST/"raw.jsonl",mapping)
assert np.array_equal(raw,again);del again
pca=np.load(DEST/"base_pca.npy",allow_pickle=False)
assert raw.shape==(3707,4096) and pca.shape==(3707,1536)
qa={"raw":table_quality(raw),"pca":table_quality(pca)}
ids=np.asarray(release["QA"]["original_129"]["ids"],dtype=int)
assert len(set(ids.tolist()))==129
assert np.isfinite(raw[ids]).all() and np.isfinite(pca[ids]).all()
assert (np.linalg.norm(raw[ids].astype(np.float64),axis=1)>1e-4).all()
assert (np.linalg.norm(pca[ids].astype(np.float64),axis=1)>1e-4).all()
# Existing model loader supports pickle only: format conversion, no new PCA fit.
pkl=DEST/"base_pca.pkl"
with pkl.open("xb") as f:pickle.dump(pca,f,protocol=4)
with pkl.open("rb") as f:assert np.array_equal(pickle.load(f),pca)
receipt={"passed":True,"canonical_items":3706,"archive_sha256":a.sha256,
"source_release":release,"imported_sha256":sums,"raw_sha256":sha256(DEST/"table_raw.npy"),
"pca_npy_sha256":sha256(DEST/"base_pca.npy"),"pkl_sha256":sha256(pkl),
"mapping_sha256":sha256(DEST/"item2id.json"),
"pca_array_sha256":hashlib.sha256(np.ascontiguousarray(pca).tobytes()).hexdigest(),
"pkl_is_lossless_format_conversion":True,"original_129_all_finite_nonzero":True,"quality":qa}
with (DEST/"IMPORT_QA_COMPLETE.json").open("x") as f:json.dump(receipt,f,indent=2)
print(json.dumps({k:v for k,v in receipt.items() if k not in ("source_release","imported_sha256")}))
