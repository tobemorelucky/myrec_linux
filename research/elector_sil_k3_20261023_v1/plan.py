import hashlib,json
from pathlib import Path
ROOT=Path('/home/yyx/hzgcode/pom2.0').resolve()
OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(2**20),b''):h.update(block)
 return h.hexdigest()
def write(p,value):
 assert Path(p).parent.resolve().is_relative_to(OUT)
 with Path(p).open('x') as f:json.dump(value,f,indent=2,allow_nan=False,default=str)
def verify():
 m=read(OUT/'MANIFEST.json')
 for name,h in m['protected_sha256'].items():assert sha(ROOT/name)==h,name
 for name,h in read(OUT/'SOURCE_LOCK.json').items():assert sha(OUT/name)==h,name
 return m
def task_id(ds,K,seed,arm):
 assert ds in ('beauty','ml-1m') and K==3 and seed==42 and arm=='SS-C1'
 return ds+'_k3_s42_SS-C1'
def folder(ds,K,seed,arm):return OUT/'runs'/task_id(ds,K,seed,arm)
def expected_parameters(ds,K,arm):
 task_id(ds,K,42,arm)
 return (943174+(K-4)*129 if ds=='beauty' else 405636+(K-2)*129)+37504
def archived(ds,K,seed,arm):return None
def check_initialization(ds,K,seed,arm,digest):
 task_id(ds,K,seed,arm)
 assert read(OUT/'initialization'/(ds+'_k3_s42.json'))['common_initial_sha256']==digest
def acceptance(p):
 r=read(p/'result.json');d=read(p/'TEST_COMPLETE.json')
 assert r['train_complete'] and r['Dev_best_selected'] and r['Test_accepted'] and r['Test_passes']==1
 assert d['one_Test_pass'] and d['result_sha256']==sha(p/'result.json')
 assert d['checkpoint_sha256']==r['checkpoint_sha256']==sha(p/'best.pt')
 assert not (p/'FAILURE.json').exists()
 return dict(best_epoch=r['best']['epoch'],checkpoint_sha256=r['checkpoint_sha256'])
