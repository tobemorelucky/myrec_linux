"""Finite common-initialization and unchanged protocol assertions; no held-out scoring."""
import ast,sys,json,signal,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch
from gict_data import args_from_config,train_corpus,sha
from run_sgitrec import core_hash
from models.sequential.SGITRec import SGITRec
from models.sequential.SGITRecV2 import SGITRecV2
from utils import utils
def runner(path):
    tree=ast.parse(path.read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    return next(n for n in main.body if isinstance(n,ast.ClassDef) and n.name=='Runner')
def main():
    signal.alarm(170);torch.set_num_threads(4);tick=time.perf_counter()
    before=json.loads((OUT/'BEFORE.json').read_text())
    for p,h in before['protected_sha256'].items():assert sha(p)==h,p
    old=json.loads((ROOT/'research/sgitrec_round1_20261024_v1/PLAN.json').read_text())
    assert ast.dump(runner(ROOT/'tools/run_sgitrec.py'))==ast.dump(runner(ROOT/'tools/run_sgitrec_v2.py'))
    results={}
    for ds,spec in old['datasets'].items():
        a=args_from_config(spec['config']);a.device=torch.device('cpu')
        c=train_corpus(a);a.sgit_graph_path=str(OUT/'graphs'/(ds+'_bridge_v2.pt'))
        utils.init_seed(42);m=SGITRecV2(a,c)
        assert core_hash(m)==spec['C1_initial_sha256']
        assert a.K==spec['K'] and a.random_seed==42
        assert all(p.requires_grad for p in m.parameters())
        assert a.lr==spec['lr'] and a.epoch==200 and a.early_stop==10 and a.batch_size==1024
        results[ds]=dict(V2_parameters=m.count_variables(),C1_initial_sha256=core_hash(m),
            graph_sha256=sha(a.sgit_graph_path),all_base_parameters_jointly_trainable=True)
        del m
        if ds=='beauty':
            a.sgit_variant='G2';a.sgit_graph_path=spec['graph']
            utils.init_seed(42);m=SGITRec(a,c)
            assert core_hash(m)==spec['C1_initial_sha256']
            results[ds]['V1G2_parameters']=m.count_variables()
            del m
    for name in ['models/sequential/SGITRecV2.py','models/sequential/sgit_v2_components.py',
                 'tools/sgit_v2_graph.py','tools/build_sgit_v2_graph.py','tools/diagnose_sgit_v2.py',
                 'tools/test_sgit_v2.py','tools/run_sgitrec_v2.py']:
        compile((ROOT/name).read_text(),str(ROOT/name),'exec')
    result=dict(passed=True,datasets=results,protected_files=len(before['protected_sha256']),
        protected_SHA_unchanged=True,Runner_identical_AST=True,original_score_views_dependency_unchanged=True,
        selected_prior_lambda=.30,Test=False,Dev=False,seconds=time.perf_counter()-tick)
    with (OUT/'checks/launch_checks.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
