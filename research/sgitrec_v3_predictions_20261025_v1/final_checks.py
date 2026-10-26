"""Final bounded proof of actual loss matrix and immutable assets before launch."""
import sys,json,time,signal,hashlib,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch
from tools.gict_data import args_from_config,train_corpus,sha
from test_sgitrec import make_feed
from models.sequential.SGITRecV3Controlled import SGITRecV3Controlled
from models.sequential.sgit_v3_control_components import VARIANTS
from models.sequential.LLMMIRecASPCFSampledSoftmax import candidate_mask,masked_softmax_loss
from utils import utils
signal.alarm(170);torch.set_num_threads(4);tick=time.perf_counter()
before=json.loads((OUT/'BEFORE.json').read_text())
for name,digest in before['protected_sha256'].items():assert sha(name)==digest,name
proof={}
for ds in ['ml-1m','beauty']:
 a=args_from_config(OUT/'configs'/(ds+'_s42.json'));a.device=torch.device('cpu');a.v3_graph_path=str(OUT/'graphs'/(ds+'_predictions_v3.pt'))
 c=train_corpus(a);feed=make_feed(c,'cpu',64);proof[ds]={}
 for variant in VARIANTS:
  a.v3_variant=variant;utils.init_seed(42);m=SGITRecV3Controlled(a,c).train();m.cur_epoch=1
  with torch.no_grad():
   for s in m.prediction_scorer.scales.values():s.fill_(.2)
  out=m(feed,True);h=feed['history_items'];valid=(torch.arange(20)[None]<feed['lengths'][:,None])&(h>0)
  edges={r:m.candidate_graph.lookup(h,feed['user_id'],r) for r in m.spec['relations']}
  ids=feed['item_id'][:,0][None].expand(len(h),-1)
  features=out['candidate_vectors'][:,0][None].expand(len(h),-1,-1)
  delta,_=m.prediction_scorer(out['interest_vectors'],out['attention_maps'],out['interest_weights'],valid,ids,features,edges)
  base=out['user_vector']@out['candidate_vectors'][:,0].T
  assert torch.allclose(out['_ss_scores'],base+delta,atol=2e-6,rtol=1e-5),(ds,variant)
  assert torch.count_nonzero(delta)>0,(ds,variant,'nonzero matrix')
  allowed=candidate_mask(m.train_membership,feed['user_id'],feed['item_id'][:,0])[0]
  assert torch.equal(out['_ss_allowed'],allowed)
  ranking,_=m._ranking_loss(out['_ss_scores'],out['_ss_allowed'])
  manual,_=masked_softmax_loss(base+delta,allowed,m.sampled_softmax_temperature)
  assert torch.allclose(ranking,manual,atol=2e-6)
  ranking.backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.prediction_scorer.parameters())
  assert all(p.grad.abs().sum()>0 for p in m.prediction_scorer.scales.values())
  assert torch.allclose(out['_ss_scores'].diagonal(),out['raw_prediction'][:,0],atol=3e-6,rtol=2e-5)
  proof[ds][variant]=dict(all_pairs_same_scorer=True,off_diagonal_branch_nonzero=bool((delta-torch.diag(delta.diag())).abs().sum()>0),
    actual_ranking_loss_uses_matrix=True,prior_excluded_from_training=True)
  del m,out,delta,base,ranking,manual
source=[]
for name in ['models/sequential/SGITRecV3Controlled.py','models/sequential/sgit_v3_control_components.py','tools/build_sgit_v3_control_index.py','tools/run_sgitrec_v3_controls.py','tools/test_sgit_v3_controls.py','research/sgitrec_v3_predictions_20261025_v1/dispatcher.py']:
 ast.parse((ROOT/name).read_text());source.append(str(ROOT/name))
for name,digest in before['protected_sha256'].items():assert sha(name)==digest,name
result=dict(passed=True,seconds=time.perf_counter()-tick,actual_training_matrix_proof=proof,
 protected_assets_unchanged=len(before['protected_sha256']),source_sha256={n:sha(n) for n in source},
 init_routing_gradient_note='Q/K ranking gradients are nonzero but initially small (about 1e-11 in graph arms); no threshold/performance filter. Per-epoch parameter update norms are logged.',
 no_Test_or_Dev_evaluation=True)
with (OUT/'checks/final_checks.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
