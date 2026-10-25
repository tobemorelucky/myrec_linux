"""Bounded V2 graph isolation, mathematics, real gradients and resource tests; no Dev/Test scoring."""
import sys,os,json,time,signal,copy,argparse,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'research/sgitrec_round1_20261024_v1')]
import numpy as np,torch
from torch.nn import functional as F
from gict_data import args_from_config,train_corpus,sha
from preflight import CONFIGS,routing
from test_sgitrec import make_feed
from sgit_graph import build_graphs,transition_graph
from sgit_v2_graph import build_bridge
from run_sgitrec import core_hash
from models.sequential.SGITRecV2 import SGITRecV2
from models.sequential.sgit_v2_components import InterestBridgeMemory
from models.sequential.LLMMIRecASPCFAdaptiveSIL import LLMMIRecASPCFAdaptiveSIL as C1,segment_ids
from utils import utils
OUT=ROOT/'research/sgitrec_v2_20261025_v1'
def math_check():
    torch.manual_seed(28);reports=[]
    for K in [1,2,3,4]:
        B,L,D,R=4,5,64,8
        v=torch.randn(B,K,D,requires_grad=True);h=torch.randn(B,L,D,requires_grad=True)
        z=torch.randn(B,K,L,requires_grad=True);A=z.softmax(-1)
        valid=torch.arange(L)[None]<torch.tensor([0,1,3,5])[:,None]
        edge=dict(ids=torch.ones(B,L,R,dtype=torch.long),prob=torch.rand(B,L,R).softmax(-1),support=torch.ones(B,L,R))
        other={k:t.clone() for k,t in edge.items()}
        cv=torch.randn(B,L,R,D,requires_grad=True);bv=torch.randn_like(cv,requires_grad=True)
        m=InterestBridgeMemory(D);opt=torch.optim.Adam(m.parameters(),lr=.01)
        y,info=m(v,A,valid,edge,cv,other,bv,h)
        assert torch.equal(y,v)
        assert torch.count_nonzero(info['calibrated_attention'][0])==0
        for step in range(4):
            opt.zero_grad();y,info=m(v,A,valid,edge,cv,other,bv,h)
            loss=(y*torch.linspace(-1,1,D)).sum()+.01*y.square().sum()
            loss.backward(retain_graph=True)
            assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
            opt.step()
        for name in ['output.weight','bridge_output.weight','query.weight','key.weight','value.weight',
                     'responsibility_q.weight','responsibility_h.weight','bridge_match.weight']:
            p=dict(m.named_parameters())[name]
            assert p.grad is not None and p.grad.abs().sum()>0,(K,name)
        assert torch.equal(y[0],v[0])
        low=m(v,A,valid,edge,cv,other,bv,h)[1]['bridge_beta']
        high={**edge,'support':torch.full_like(edge['support'],100.)}
        hb=m(v,A,valid,high,cv,other,bv,h)[1]['bridge_beta']
        assert (hb<=low+1e-7).all() and (hb[1:]<low[1:]).all()
        empty={k:torch.zeros_like(t) for k,t in edge.items()}
        absent,_=m(v,A,valid,empty,cv,empty,bv,h);assert torch.equal(absent,v)
        direct,di=m(v,A,valid,edge,cv,other,bv,h,bridge_enabled=False)
        no_bridge,_=m(v,A,valid,edge,cv,empty,bv,h)
        assert torch.equal(direct,no_bridge)
        mt,*_=m.relation(v,di['calibrated_attention'],cv,edge,valid)
        assert torch.equal(direct,v+m.output(mt))
        raw=m.responsibilities(v,A,h,valid,False)
        ref=A*valid[:,None];ref=ref/ref.sum(-1,keepdim=True).clamp_min(1e-8)
        assert torch.equal(raw,ref)
        if K>1:assert (info['calibrated_attention'][3,0]-info['calibrated_attention'][3,1]).abs().max()>0
        reports.append(dict(K=K,gradients=True,zero_exact=True,empty=True,short_history=True,
          support_monotone=True,bridge_off_direct_exact=True,calibration_toggle=True))
    histories={1:[(1,1),(2,2),(3,3)],2:[(4,1),(5,2),(6,3)],3:[(1,1),(7,2),(8,3)],
               4:[(1,1),(2,2),(3,3)],5:[(4,1),(5,2),(6,3)],6:[(7,1),(8,2),(1,3)]}
    table=torch.randn(10,5);table[0]=0
    g,meta=build_graphs(histories,table);base=dict(metadata=dict(meta,schema='sgit_crossfit_item_graph_v1',scope='full_crossfit_train'),**g)
    bridge=build_bridge(base)
    changed=copy.deepcopy(histories);changed[3]=[(9,1),(6,2),(4,3)];changed[6]=[(5,1),(2,2)]
    x,xm=build_graphs(changed,table);xb=build_bridge(dict(metadata=dict(base['metadata']),**x))
    for name,value in bridge.items():
        if torch.is_tensor(value):assert torch.equal(value[0],xb[name][0]),name
    # Exactly one semantic bridge 1->4 followed by REAL directed 4->5.
    toy={k:v.clone() if torch.is_tensor(v) else dict(v) for k,v in base.items()}
    for name,value in toy.items():
        if torch.is_tensor(value):value.zero_()
    toy['semantic_ids'][:,1,0]=4;toy['semantic_prob'][:,1,0]=1.
    toy['transition_ids'][:,4,0]=5;toy['transition_prob'][:,4,0]=1.;toy['transition_support'][:,4,0]=2
    t=build_bridge(toy)
    assert (t['bridge_ids'][:,1,0]==5).all() and not (t['bridge_ids'][:,1]==4).any()
    assert (t['bridge_witness'][:,1,0]==4).all()
    assert (t['bridge_support'][:,1,0]==2).all()
    return dict(math=reports,excluded_user_future_invariant=True,bridge_direction_and_provenance=True)
def verify_real_graph(ds,c,g):
    oldpath=ROOT/'research/sgitrec_v1_20261024/graphs'/(ds+'_v2.pt')
    old=torch.load(oldpath,map_location='cpu',weights_only=True)
    assert sha(oldpath)==g['metadata']['base_graph_sha256']
    for name,value in old.items():
        if torch.is_tensor(value):assert torch.equal(value,g[name]),name
    for f in range(3):
        fresh,catalog,_=transition_graph(c.user_his,c.n_items,f,3,8)
        for k,value in fresh.items():assert torch.equal(torch.as_tensor(value),g['transition_'+k][f])
        sem=g['semantic_ids'][f].numpy();assert np.isin(sem[sem>0],catalog).all()
        loc=(g['bridge_ids'][f]>0).nonzero()
        for i,r in loc[np.linspace(0,len(loc)-1,min(256,len(loc)),dtype=int)].tolist():
            d=int(g['bridge_ids'][f,i,r]);j=int(g['bridge_witness'][f,i,r])
            assert j in g['semantic_ids'][f,i] and d in g['transition_ids'][f,j]
    return True
def real(ds,gpu):
    spec=importlib.util.spec_from_file_location('v2_resource',ROOT/'research/elector_sil_objective_full_20261023_v2/scheduler.py')
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    lease=None;device='cpu'
    if gpu is not None:
        inv=r.observe();assert r.idle(inv[gpu]),'Busy GPU: do not preempt'
        lease=r.acquire(ROOT/('research/gpu_resource_scheduler_20261022_v1/gpu%d.lock'%gpu));assert lease is not None
        assert r.idle(r.observe()[gpu])
        os.environ['CUDA_VISIBLE_DEVICES']=inv[gpu]['uuid']
        cache=OUT/'checks'/('cuda_'+ds);cache.mkdir(exist_ok=True)
        os.environ['CUDA_CACHE_PATH']=str(cache)
        torch.cuda.set_device(0);device='cuda:0'
    try:
        a=args_from_config(CONFIGS[ds]);a.device=torch.device(device)
        a.sgit_graph_path=str(OUT/'graphs'/(ds+'_bridge_v2.pt'))
        c=train_corpus(a);g=torch.load(a.sgit_graph_path,map_location='cpu',weights_only=True)
        verify_real_graph(ds,c,g)
        feed=make_feed(c,device,32);params={}
        for k in [1,2,3,4]:
            a.K=k;utils.init_seed(42);ref=C1(a,c).to(device).eval()
            utils.init_seed(42);m=SGITRecV2(a,c).to(device).eval()
            assert core_hash(ref)==core_hash(m)
            with torch.no_grad():p=ref(feed,True);q=m(feed,True)
            assert torch.equal(p['prediction'],q['prediction']) and q['interest_vectors'].shape==(32,k,64)
            params[str(k)]=m.count_variables();del ref,m
        a.K=2 if ds=='beauty' else 3
        utils.init_seed(42);ref=C1(a,c).to(device)
        utils.init_seed(42);m=SGITRecV2(a,c).to(device)
        init=core_hash(m)
        m.train();ref.train();rng=torch.get_rng_state();crng=torch.cuda.get_rng_state(0) if gpu is not None else None
        m.cur_epoch=1;ref.cur_epoch=1
        q=m(feed,True);loss=m.loss(q);loss.backward()
        torch.set_rng_state(rng)
        if crng is not None:torch.cuda.set_rng_state(crng,0)
        p=ref(feed,True);oldloss=ref.loss(p);oldloss.backward()
        assert torch.equal(loss,oldloss)
        rd=dict(ref.named_parameters())
        for name,x in m.named_parameters():
            if name in rd and rd[name].grad is not None:
                assert x.grad is not None and torch.allclose(x.grad,rd[name].grad,atol=2e-7,rtol=2e-5),name
        opt=torch.optim.Adam(m.customize_parameters(),lr=a.lr,weight_decay=a.l2)
        for step in range(6):
            m.train();opt.zero_grad();q=m(feed,True);loss=m.loss(q)
            assert torch.isfinite(loss);loss.backward()
            assert all(torch.isfinite(x.grad).all() for x in m.parameters() if x.grad is not None)
            opt.step()
        for name in ['output.weight','bridge_output.weight','responsibility_q.weight','responsibility_h.weight','bridge_match.weight']:
            x=dict(m.graph_memory.named_parameters())[name]
            assert x.grad is not None and x.grad.abs().sum()>0,name
        assert core_hash(m)!=init
        assert m.extractor.query.grad.abs().sum()>0 and m.sil_block.local_qkv.weight.grad.abs().sum()>0
        m.eval()
        ref.load_state_dict({k:v for k,v in m.state_dict().items() if not k.startswith('graph_memory.')},strict=True);ref.eval()
        with torch.no_grad():
            q=m(feed,True);assert q['graph_delta'].norm()>0
            m.sgit_enabled=False
            off=m(feed,True);p=ref(feed,True);assert torch.equal(off['prediction'],p['prediction'])
            m.sgit_enabled=True
            other=dict(feed);other['item_id']=feed['item_id'].flip(0)
            otherout=m(other,True)
            assert torch.allclose(q['interest_vectors'],otherout['interest_vectors'],atol=3e-6,rtol=3e-6)
            dirty=dict(feed);dirty['history_items']=feed['history_items'].clone()
            invalid=torch.arange(20,device=device)[None]>=feed['lengths'][:,None]
            dirty['history_items'][invalid]=3
            pad=m(dirty,True);assert torch.allclose(q['interest_vectors'],pad['interest_vectors'],atol=3e-6,rtol=3e-6)
            empty=dict(feed);empty['history_items']=feed['history_items'][:,:0];empty['lengths']=feed['lengths']*0
            eo=m(empty,True);assert torch.count_nonzero(eo['interest_vectors'])==0 and torch.isfinite(eo['prediction']).all()
            # Empty bridge and explicitly disabled bridge are exactly equivalent.
            m.bridge_enabled=False;direct=m(feed,True);m.bridge_enabled=True
            saved={n:b.clone() for n,b in m.graph_cache.named_buffers()}
            for n,b in m.graph_cache.named_buffers():
                if n.startswith('bridge_'):b.zero_()
            absent=m(feed,True);assert torch.equal(absent['prediction'],direct['prediction'])
            for n,b in m.graph_cache.named_buffers():b.zero_()
            nog=m(feed,True);assert torch.equal(nog['prediction'],p['prediction'])
            for n,b in m.graph_cache.named_buffers():b.copy_(saved[n])
        # Local causal gradient remains zero to future positions.
        x=torch.randn(1,8,64,device=device,requires_grad=True)
        ids,valid=segment_ids(x.new_zeros(1,8),torch.tensor([8],device=device),'fixed',0.,fixed_len=4)
        y=m.sil_block.local_bank(x,ids,valid)
        dx=torch.autograd.grad(y[0,2].square().sum(),x)[0]
        assert torch.count_nonzero(dx[0,3:])==0
        large=make_feed(c,device,1024);m.train()
        for _ in range(2):
            opt.zero_grad();loss=m.loss(m(large,True));loss.backward();opt.step()
        if gpu is not None:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        tick=time.perf_counter()
        for _ in range(3):
            opt.zero_grad();o=m(large,True);loss=m.loss(o);loss.backward();opt.step()
        if gpu is not None:torch.cuda.synchronize()
        elapsed=(time.perf_counter()-tick)/3
        result=dict(dataset=ds,passed=True,actual_K=a.K,parameters_by_K=params,
            initial_C1_outputs_loss_base_gradients_exact=True,Train_graph_matches_V1_all_folds=True,
            bridge_witness_768_edges_valid=True,short_history_padding_empty=True,local_causality=True,
            joint_base_and_bridge_gradients=True,graph_off_exact=True,empty_bridge_direct_exact=True,
            candidate_independent=True,projection_T_norm=float(m.graph_memory.output.weight.norm()),
            projection_B_norm=float(m.graph_memory.bridge_output.weight.norm()),
            step_seconds=elapsed,batch=1024,rows_per_second=1024/elapsed,
            gpu_peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20 if gpu is not None else None,
            graph_gpu_buffer_bytes=m.graph_cache.bytes(),graph_file_bytes=Path(a.sgit_graph_path).stat().st_size,
            neighbor_encoding=o['graph_encoding_cost'],Dev=False,Test=False)
        return result
    finally:
        if lease is not None:os.close(lease)
def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['math','real'],required=True)
    p.add_argument('--dataset',choices=['beauty','ml-1m']);p.add_argument('--gpu',type=int)
    x=p.parse_args();signal.alarm(170);torch.set_num_threads(4);tick=time.perf_counter()
    result=math_check() if x.mode=='math' else real(x.dataset,x.gpu)
    result.update(seconds=time.perf_counter()-tick,passed=True)
    target=OUT/'checks'/((x.dataset+'_structure') if x.mode=='real' else 'math')
    with target.with_suffix('.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
