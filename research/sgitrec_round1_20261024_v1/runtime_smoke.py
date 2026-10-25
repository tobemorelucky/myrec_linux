"""CPU two-batch regression of the exact launch Runner/hook, without Dev/Test."""
import ast,sys,json,time,logging,signal
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import torch
from helpers.BaseRunner import BaseRunner
from models.sequential.SGITRec import SGITRec
from gict_data import args_from_config,train_corpus
from utils import utils
signal.alarm(170);torch.set_num_threads(4)
a=args_from_config(ROOT/'research/elector_sil_k3_20261023_v1/runs/ml-1m_k3_s42_SS-C1/config.json')
a.device=torch.device('cpu');a.sgit_variant='G3';a.sgit_graph_path=str(ROOT/'research/sgitrec_v1_20261024/graphs/ml-1m_v2.pt')
a.batch_size=4;a.num_workers=0;a.pin_memory=0
c=train_corpus(a);utils.init_seed(42);m=SGITRec(a,c)
d=m.Dataset(m,c,'train')
for key in d.data:d.data[key]=d.data[key][:8]
d.prepare()
out=OUT/'checks/runner_cpu_smoke';out.mkdir()
logging.basicConfig(filename=out/'train.log',filemode='x',level=logging.INFO,force=True)
tree=ast.parse((ROOT/'tools/run_sgitrec.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
cls=next(n for n in main.body if isinstance(n,ast.ClassDef) and n.name=='Runner')
hook=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='encoding_hook')
ns=dict(BaseRunner=BaseRunner,torch=torch,time=time,json=json,m=m,out=out,ROOT=ROOT,a=a)
exec(compile(ast.Module(body=[cls],type_ignores=[]),'runner_actual_class','exec'),ns)
runner=ns['Runner'](a);ns['runner']=runner
exec(compile(ast.Module(body=[hook],type_ignores=[]),'runner_actual_hook','exec'),ns)
m.register_forward_hook(ns['encoding_hook'])
with patch.object(torch.cuda,'reset_peak_memory_stats'),patch.object(torch.cuda,'synchronize'),patch.object(torch.cuda,'max_memory_allocated',return_value=0):
 value=runner.fit(d,epoch=1)
assert torch.isfinite(torch.tensor(value)) and runner.last_cost['batches']==2
assert runner.last_cost['mean_unique_neighbors']>0 and runner.last_cost['neighbor_slots_read']>0
logging.info('RUNNER_CPU_SMOKE_PASS');logging.shutdown()
assert 'RUNNER_CPU_SMOKE_PASS' in (out/'train.log').read_text()
result=dict(passed=True,two_actual_Train_batches=True,original_Runner_class_tested=True,encoding_hook_counts=runner.last_cost,logging_writable=True,Dev=False,Test=False,no_saved_checkpoint=True)
with (OUT/'checks/runner_cpu_smoke.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
