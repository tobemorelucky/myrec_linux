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
 assert ds in ('beauty','ml-1m') and K in ((2,4,8) if ds=='beauty' else (2,))
 assert seed in SEEDS and arm in ARMS
 return '%s_k%d_s%d_%s'%(ds,K,seed,arm)
def folder(ds,K,seed,arm):return OUT/'runs'/task_id(ds,K,seed,arm)
def expected_parameters(ds,K,arm):
 baseline=(943174+(K-4)*129) if ds=='beauty' else (405636+(K-2)*129)
 return baseline+(0 if arm=='SS-B0' else 37504)
def archived(ds,K,seed,arm):
 return next((r for r in manifest()['reused'] if (r['dataset'],r['K'],r['seed'],r['arm'])==(ds,K,seed,arm)),None)
def check_initialization(ds,K,seed,arm,digest):
 path=OUT/'initialization'/('%s_k%d_s%d.json'%(ds,K,seed))
 if path.exists():assert read(path)['common_initial_sha256']==digest,'Common ASPCF initialization mismatch'
 else:
  assert arm=='SS-B0','First task of a new (dataset,K,seed) must be SS-B0'
  write(path,dict(dataset=ds,K=K,seed=seed,common_initial_sha256=digest))
def acceptance(path):
 r=read(path/'result.json');d=read(path/'TEST_COMPLETE.json')
 assert r['train_complete'] and r['Dev_best_selected'] and r['Test_accepted'] and r['Test_passes']==1
 assert d['one_Test_pass'] and d['result_sha256']==sha(path/'result.json')
 assert d['checkpoint_sha256']==r['checkpoint_sha256']==sha(path/'best.pt')
 assert not (path/'FAILURE.json').exists()
 return dict(checkpoint_sha256=r['checkpoint_sha256'],best_epoch=r['best']['epoch'])
def select_by_dev(scores):return max(sorted(scores),key=lambda K:(scores[K],-K))
def select_beauty_K():
 target=OUT/'BEAUTY_K_LOCKED.json'
 assert not target.exists(),'Never reselect K'
 scores={};evidence={}
 for K in (2,4,8):
  source=archived('beauty',K,42,'SS-C1')
  if source:
   path=Path(source['train_result']);r=read(path);best=r['best']
  else:
   path=folder('beauty',K,42,'SS-C1')/'train_result.json';r=read(path);best=r['best']
  assert r['train_complete'] and r['Dev_best_selected']
  scores[K]=best['metrics']['NDCG@5']
  evidence[K]=dict(train_result=str(path),sha256=sha(path),best_epoch=best['epoch'],Dev_NDCG5=scores[K])
 K=select_by_dev(scores)
 write(target,dict(K=K,selection='seed42 C1 locked-prior best Dev NDCG@5; exact ties smaller K',
                   evidence=evidence,Test_used=False))
 return K
