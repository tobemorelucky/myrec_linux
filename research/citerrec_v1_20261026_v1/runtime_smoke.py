"""Two actual Train batches and synthetic evaluator regression; no real Dev/Test inference."""
import ast,sys,json,time,logging,signal,importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch,numpy as np
from helpers.BaseRunner import BaseRunner
from models.sequential.CITERRec import CITERRec
from gict_data import args_from_config,train_corpus
import citer_dev_protocol as protocol
from utils import utils
signal.alarm(170);torch.set_num_threads(4);protocol.install_guard()
a=args_from_config(OUT/'configs/ml-1m_s42.json');a.device=torch.device('cpu')
a.v3_graph_path=str(ROOT/'research/sgitrec_v3_predictions_20261025_v1/graphs/ml-1m_predictions_v3.pt');a.batch_size=4;a.num_workers=0;a.pin_memory=0
c=train_corpus(a);utils.init_seed(42);m=CITERRec(a,c)
d=m.Dataset(m,c,'train')
for key in d.data:d.data[key]=d.data[key][:8]
d.prepare();out=OUT/'checks/runner_cpu_smoke';out.mkdir()
logging.basicConfig(filename=out/'train.log',filemode='x',level=logging.INFO,force=True)
tree=ast.parse((ROOT/'tools/run_citerrec_dev.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
cls=next(n for n in main.body if isinstance(n,ast.ClassDef) and n.name=='Runner');hook=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='encoding_hook')
initial_branch={n:p.detach().clone() for n,p in m.prediction_scorer.named_parameters()}
ns=dict(BaseRunner=BaseRunner,torch=torch,time=time,json=json,m=m,out=out,ROOT=ROOT,a=a,protocol=protocol,initial_branch=initial_branch)
exec(compile(ast.Module(body=[cls],type_ignores=[]),'actual_runner','exec'),ns)
runner=ns['Runner'](a);ns['runner']=runner
exec(compile(ast.Module(body=[hook],type_ignores=[]),'actual_hook','exec'),ns)
m.register_forward_hook(ns['encoding_hook'])
with patch.object(torch.cuda,'reset_peak_memory_stats'),patch.object(torch.cuda,'synchronize'),patch.object(torch.cuda,'max_memory_allocated',return_value=0):
 value=runner.fit(d,epoch=1)
assert np.isfinite(value) and runner.last_cost['batches']==2
# Exercise actual evaluate logger with synthetic result, no real heldout inputs.
ref=json.load(open(ROOT/'research/gictrec_round1_20261024_v1/ml-1m_dev_PROTOCOL.json'))
fake=dict(metrics={s:{k:.1 for k in protocol.KEYS} for s in ['raw','locked_prior']},input_hash=ref['input_hash'],rows=3)
with patch.object(protocol,'score_views',return_value=fake):runner.evaluate(SimpleNamespace(phase='dev'),[5],['NDCG','HR'])
assert 'interest_gate.weight' in runner.latest['branch_update_norms']
# Original NumPy metric/candidate ordering regression on a tiny synthetic Dev dataset.
spec=importlib.util.spec_from_file_location('citer_original_evaluator',ROOT/'research/gictrec_round1_20261024_v1/common.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
class ToyModel(torch.nn.Module):
 def __init__(self):
  super().__init__();self.device=torch.device('cpu')
 def forward(self,feed,return_intermediate=False):
  ids=feed['item_id'];raw=(ids%13).float()*.1
  return dict(raw_prediction=raw,prediction=raw.double()+.3*torch.log1p((ids%7).double()))
class ToyDataset(torch.utils.data.Dataset):
 phase='dev';corpus=SimpleNamespace(data_df={'train':None,'dev':None})
 def __init__(self):self.model=ToyModel()
 def __len__(self):return 3
 def __getitem__(self,i):
  ids=torch.arange(1,1002);ids[0]=5+i;ids[5]=ids[0];ids[-1]=ids[0]
  return dict(user_id=torch.tensor(i+1),history_items=torch.tensor([1,2]),lengths=torch.tensor(2),item_id=ids)
 @staticmethod
 def collate_batch(rows):return {k:torch.stack([x[k] for x in rows]) for k in rows[0]}
toy=ToyDataset();new=protocol.score_views(toy);previous=old.score_views(toy)
assert new==previous,'Evaluator or input fingerprint changed'
assert set(c.data_df)=={'train'}
logging.info('DEV_ONLY_RUNNER_PASS');logging.shutdown()
result=dict(passed=True,two_actual_Train_batches=True,original_Runner_hook=True,evaluate_update_logger=True,
 evaluator_and_input_hash_exact_original=True,synthetic_1001_duplicate_slots=True,no_checkpoint_saved=True,real_Dev=False,Test=False)
with (OUT/'checks/runtime_smoke.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
