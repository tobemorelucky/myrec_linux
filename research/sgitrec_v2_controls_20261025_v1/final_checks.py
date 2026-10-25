"""Seed-zero constructors, old assets, typed graph folds and original user-future invariance."""
import sys,json,time,signal,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch,numpy as np
from gict_data import args_from_config,train_corpus,sha
from run_sgitrec import core_hash
from models.sequential.LLMMIRecASPCFAdaptiveSIL import LLMMIRecASPCFAdaptiveSIL as C1
from models.sequential.SGITRec import SGITRec
from models.sequential.SGITRecV2Controlled import SGITRecV2Controlled
from sgit_graph import transition_graph,build_graphs
from sgit_v2_graph import build_bridge
from utils import utils
def main():
 signal.alarm(170);torch.set_num_threads(4);tick=time.perf_counter();result={}
 before=json.loads((OUT/'BEFORE.json').read_text())
 for p,h in before['protected_sha256'].items():assert sha(p)==h,p
 for ds in ['ml-1m','beauty']:
  se=json.loads((OUT/'checks'/(ds+'_seed.json')).read_text())['seeds']['0']
  a=args_from_config(se['source_config']);c=train_corpus(a)
  utils.init_seed(0);ref=C1(a,c);expected=core_hash(ref);rng=torch.get_rng_state()
  a.sgit_control='V2-Full';a.sgit_graph_path=str(ROOT/'research/sgitrec_v2_20261025_v1/graphs'/(ds+'_bridge_v2.pt'))
  utils.init_seed(0);m=SGITRecV2Controlled(a,c)
  assert core_hash(m)==expected==se['C1_initial_sha256'] and torch.equal(torch.get_rng_state(),rng)
  payload=torch.load(a.sgit_graph_path,map_location='cpu',weights_only=True)
  for f in range(3):
   fresh,cat,_=transition_graph(c.user_his,c.n_items,f,3,8)
   for k,v in fresh.items():assert torch.equal(torch.as_tensor(v),payload['transition_'+k][f])
   ids=payload['semantic_ids'][f].numpy()
   assert np.isin(ids[ids>0],cat).all()
  a.sgit_variant='G2';a.sgit_graph_path=str(ROOT/'research/sgitrec_v1_20261024/graphs'/(ds+'_v2.pt'))
  utils.init_seed(0);g=SGITRec(a,c)
  assert core_hash(g)==expected and torch.equal(torch.get_rng_state(),rng)
  result[ds]=dict(seed0_full_and_V1_initial_C1_exact=True,C1_initial_sha256=expected,
   rng_after_init_same=True,all_three_folds_direct_exact=True,semantic_candidates_only_other_fold_Train=True)
  del m,g,ref,c
 histories={1:[(1,1),(2,2),(3,3)],2:[(4,1),(5,2),(6,3)],3:[(1,1),(7,2),(8,3)],4:[(4,1),(5,2),(1,3)]}
 torch.manual_seed(10);table=torch.randn(10,7);table[0]=0
 base,meta=build_graphs(histories,table)
 g=build_bridge(dict(metadata=dict(meta,schema='sgit_crossfit_item_graph_v1',scope='full_crossfit_train'),**base))
 changed=copy.deepcopy(histories);changed[3]=[(9,1),(8,2),(1,3),(4,4)]
 b,m=build_graphs(changed,table)
 again=build_bridge(dict(metadata=dict(meta,schema='sgit_crossfit_item_graph_v1',scope='full_crossfit_train'),**b))
 for key,value in g.items():
  if torch.is_tensor(value):assert torch.equal(value[0],again[key][0]),key
 got=dict(passed=True,datasets=result,excluded_own_fold_future_mutation_invariant=True,
  protected_files=len(before['protected_sha256']),protected_SHA_unchanged=True,
  Dev=False,Test=False,seconds=time.perf_counter()-tick)
 with (OUT/'checks/final_checks.json').open('x') as f:json.dump(got,f,indent=2)
 print(json.dumps(got,indent=2))
if __name__=='__main__':main()
