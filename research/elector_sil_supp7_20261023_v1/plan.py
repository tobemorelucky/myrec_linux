"""Immutable experiment plan and Dev-only K selection helpers; no torch import."""
import hashlib,json
from pathlib import Path
ROOT=Path('/home/yyx/hzgcode/pom2.0').resolve()
OUT=Path(__file__).resolve().parent
SEEDS=(0,1,2,3,41,42)
MAIN=('SS-B0','SS-C0','SS-C1')
ARMS=MAIN+('SS-A1','SS-A2')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(2**20),b''):h.update(block)
 return h.hexdigest()
def write(p,value):
 p=Path(p);assert p.parent.resolve().is_relative_to(OUT)
 with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False,default=str)
def manifest():return read(OUT/'MANIFEST.json')
def verify():
 m=manifest()
 for name,digest in m['protected_sha256'].items():assert sha(ROOT/name)==digest,name
 for name,digest in read(OUT/'SOURCE_LOCK.json').items():assert sha(OUT/name)==digest,name
 return m
def task_id(ds,K,seed,arm):
 assert dict(dataset=ds,K=K,seed=seed,arm=arm) in manifest()['tasks'],'Task outside seven-job whitelist'
 return '%s_k%d_s%d_%s'%(ds,K,seed,arm)
def folder(ds,K,seed,arm):return OUT/'runs'/task_id(ds,K,seed,arm)
def expected_parameters(ds,K,arm):
 baseline=(943174+(K-4)*129) if ds=='beauty' else (405636+(K-2)*129)
 return baseline+(0 if arm=='SS-B0' else 37504)
def archived(ds,K,seed,arm):
 return next((r for r in manifest()['reused'] if (r['dataset'],r['K'],r['seed'],r['arm'])==(ds,K,seed,arm)),None)
def check_initialization(ds,K,seed,arm,digest):
 task_id(ds,K,seed,arm)
 path=OUT/'initialization'/('%s_k%d_s%d.json'%(ds,K,seed))
 assert path.is_file() and read(path)['common_initial_sha256']==digest,'Common ASPCF initialization mismatch'
def acceptance(path):
 r=read(path/'result.json');d=read(path/'TEST_COMPLETE.json')
 assert r['train_complete'] and r['Dev_best_selected'] and r['Test_accepted'] and r['Test_passes']==1
 assert d['one_Test_pass'] and d['result_sha256']==sha(path/'result.json')
 assert d['checkpoint_sha256']==r['checkpoint_sha256']==sha(path/'best.pt')
 assert not (path/'FAILURE.json').exists()
 return dict(checkpoint_sha256=r['checkpoint_sha256'],best_epoch=r['best']['epoch'])
