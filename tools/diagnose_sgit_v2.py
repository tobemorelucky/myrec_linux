"""Bounded V1 trained-routing diagnosis and label-isolated bridge build/coverage. No Test."""
import sys,json,time,signal,argparse,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'research/sgitrec_round1_20261024_v1')]
import numpy as np,pandas as pd,torch
from torch.nn import functional as F
from gict_data import args_from_config,train_corpus,sha
from preflight import CONFIGS,summary,routing,pairs
from models.sequential.SGITRec import SGITRec
from sgit_v2_graph import build_bridge,coverage
from utils import utils
OUT=ROOT/'research/sgitrec_v2_20261025_v1'
def feeds(c,n=128):
    rng=np.random.default_rng(20261025)
    users=np.array(sorted(u for u,h in c.user_his.items() if len(h)>=3))
    users=users[np.linspace(0,len(users)-1,n,dtype=int)]
    h=np.zeros((n,20),np.int64);lengths=[];items=[]
    for row,u in enumerate(users):
        seq=c.user_his[int(u)];p=int(rng.integers(2,len(seq)));pre=seq[max(0,p-20):p]
        h[row,:len(pre)]=[x[0] for x in pre];lengths.append(len(pre));items.append([seq[p][0],1])
    return {k:torch.as_tensor(v) for k,v in dict(user_id=users,history_items=h,lengths=lengths,item_id=items).items()}
def diagnostic(ds):
    tick=time.perf_counter();a=args_from_config(CONFIGS[ds]);a.device=torch.device('cpu')
    a.sgit_graph_path=str(ROOT/'research/sgitrec_v1_20261024/graphs'/(ds+'_v2.pt'))
    c=train_corpus(a);base=torch.load(a.sgit_graph_path,map_location='cpu',weights_only=True)
    graph=build_bridge(base)
    # Entire build completes BEFORE Dev CSV/labels are even opened.
    graph['metadata'].update(base_graph_sha256=sha(a.sgit_graph_path),builder_sha256=sha(ROOT/'tools/sgit_v2_graph.py'))
    graphpath=OUT/'graphs'/(ds+'_bridge_v2.pt')
    with graphpath.open('xb') as fp:torch.save(graph,fp)
    train={'Train_nodes':base['metadata']['Train_nodes'],'folds':graph['metadata']['bridge_stats'],
      'base_tensor_bytes':sum(x.numel()*x.element_size() for x in base.values() if torch.is_tensor(x)),
      'V2_tensor_bytes':sum(x.numel()*x.element_size() for x in graph.values() if torch.is_tensor(x)),
      'graph_sha256':sha(graphpath),'path':str(graphpath)}
    dev=pd.read_csv(Path(a.path)/ds/'dev.csv',sep='\t',usecols=['user_id','item_id'])
    dev=dev.iloc[np.linspace(0,len(dev)-1,min(2048,len(dev)),dtype=int)]
    h=np.zeros((len(dev),20),np.int64);lengths=[]
    for row,u in enumerate(dev.user_id):
        seq=c.user_his[int(u)][-20:];h[row,:len(seq)]=[x[0] for x in seq];lengths.append(len(seq))
    cov=coverage(graph,dev.user_id.to_numpy(),h,lengths,dev.item_id.to_numpy())
    result=dict(dataset=ds,Train_graph=train,Dev_coverage=cov,coverage_only_not_ranking=True,Test=False)
    if ds=='ml-1m':
        feed=feeds(c);reports={}
        for variant in ['G2','G3']:
            a.sgit_variant=variant;utils.init_seed(42);m=SGITRec(a,c).eval()
            cp=ROOT/'research/sgitrec_round1_20261024_v1/runs'/('ml-1m_k3_s42_'+variant)/'best.pt'
            m.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True),strict=True)
            with torch.no_grad():o=m(feed,True)
            edges=m.graph_cache.lookup(feed['history_items'],feed['user_id'],'transition')
            v=o['c1_interest_vectors'];q=m.graph_memory.query(m.graph_memory.norm(v));A=o['attention_maps']
            def cos_pair(x):
                return summary(torch.stack([F.cosine_similarity(x[:,i],x[:,j],dim=-1) for i,j in itertools.combinations(range(x.shape[1]),2)],1))
            aa=A.double();ad=aa/aa.sum(-1,keepdim=True).clamp_min(1e-30);js=[]
            for i,j in itertools.combinations(range(a.K),2):
                p,r=ad[:,i],ad[:,j];mid=(p+r)/2
                js.append(.5*((p*(p.clamp_min(1e-30).log()-mid.clamp_min(1e-30).log())).sum(-1)+(r*(r.clamp_min(1e-30).log()-mid.clamp_min(1e-30).log())).sum(-1)))
            r=dict(checkpoint_sha256=sha(cp),samples=len(feed['user_id']),
                original_A_pair_JS=summary(torch.stack(js)),original_A_pair_absdiff=summary(pairs(A).abs()),
                C1_interest_pair_cosine=cos_pair(v),graph_query_pair_cosine=cos_pair(q),
                transition_routing=routing(o['transition_route'],edges['ids'],c.n_items),
                graph_memory_pair_cosine=cos_pair(o['collaborative_memory']),
                graph_support=summary(o['graph_support']),
                delta_to_C1_norm=summary(o['graph_delta'].norm(dim=-1)/v.norm(dim=-1).clamp_min(1e-8)))
            if variant=='G3':
                mc,ms=o['collaborative_memory'],o['semantic_memory']
                mixed=o['relation_weights'][...,0,None]*mc+o['relation_weights'][...,1,None]*ms
                with torch.no_grad():direct=m.graph_memory.output(mc);mixdelta=m.graph_memory.output(mixed)
                r.update(collaborative_weight=summary(o['relation_weights'][...,0]),
                    semantic_weight=summary(o['relation_weights'][...,1]),
                    two_memory_cosine=summary(F.cosine_similarity(mc,ms,dim=-1)),
                    collaborative_norm=summary(mc.norm(dim=-1)),semantic_norm=summary(ms.norm(dim=-1)),
                    blended_delta_to_direct_delta_norm=summary(mixdelta.norm(dim=-1)/direct.norm(dim=-1).clamp_min(1e-8)))
            reports[variant]=r
            del m
        result['V1_trained_diagnostics']=reports
    result['seconds']=time.perf_counter()-tick
    dest=OUT/'checks'/(ds+'_diagnostic.json')
    with dest.open('x') as fp:json.dump(result,fp,indent=2)
    print(json.dumps(result,indent=2))
def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['ml-1m','beauty'],required=True);x=p.parse_args()
    signal.alarm(170);torch.set_num_threads(4);diagnostic(x.dataset)
if __name__=='__main__':main()
