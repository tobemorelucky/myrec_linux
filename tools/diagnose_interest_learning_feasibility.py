"""Isolated CPU T0-T3 Train mechanism diagnostic, not a registered model.
Same A1 recurrence; anchor/head two-factor comparison. No Dev/Test/checkpoint.
"""
import argparse, json, math, pickle, signal, sys, time, types
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from diagnose_prefix_evidence_symmetry import Model,GeneralModel,num,norm,cos,stats

def fixture(ds):
    c=pickle.load(open(ROOT/'data'/ds/'SeqReader.pkl','rb'));df=c.data_df['train'];eligible=np.flatnonzero(df.position.to_numpy()>0)
    rng=np.random.default_rng(20261016);idx=rng.choice(eligible,1024,replace=False)[:256];hs=[];ls=[];cs=[]
    for i in idx:
        r=df.iloc[int(i)];u=int(r.user_id);pos=int(r.position);hist=c.user_his[u][max(0,pos-20):pos]
        assert len(hist) and all(v[1]<=r.time for v in hist)
        neg=int(rng.integers(1,c.n_items))
        while neg in c.train_clicked_set[u]:neg=int(rng.integers(1,c.n_items))
        h=[int(v[0]) for v in hist];hs.append(h+[0]*(20-len(h)));ls.append(len(h));cs.append([int(r.item_id),neg])
    batch={k:torch.tensor(v) for k,v in dict(history_items=hs,lengths=ls,item_id=cs).items()}
    return c,batch,dict(samples=256,users=int(df.iloc[idx].user_id.nunique()),length_counts={str(i):ls.count(i) for i in sorted(set(ls))},eligible=len(eligible))

def anchor_bias(z,valid,K):
    # No learned anchor parameters, labels, external history or random tie breaks.
    norms=z.norm(dim=-1);sv=valid&(norms>1e-8);unit=z/norms.clamp_min(1e-8)[:,:,None]
    rel=torch.bmm(unit,unit.transpose(1,2)).clamp(-1,1);B,L=valid.shape
    pos=torch.arange(L)[None].expand(B,-1);first=pos.masked_fill(~sv,-1).max(-1).values.clamp_min(0)
    used=torch.zeros_like(sv);maxsim=z.new_full((B,L),-2.);indices=[];rows=torch.arange(B)
    for _ in range(K):
        if not indices:pick=first
        else:
            candidate=maxsim.masked_fill(~sv|used,float('inf'));pick=candidate.argmin(-1)
            exhausted=(~(sv&~used)).all(-1);pick=torch.where(exhausted,first,pick)
        indices.append(pick);used[rows,pick]=True
        maxsim=torch.maximum(maxsim,rel[rows,:,pick])
    ix=torch.stack(indices,-1);anchors=unit[rows[:,None],ix]
    bias=torch.bmm(unit,anchors.transpose(1,2));bias=bias-bias.mean(-1,keepdim=True)
    bias=bias*sv[:,:,None];return bias.detach(),ix,rel,sv

def route(self,e,lengths,z,use_evidence=False):
    B,L,D=e.shape;valid=torch.arange(L)[None]<lengths[:,None];values=self.Wv(e)
    content=torch.einsum('bld,kd->blk',self.We(e),self.query)/math.sqrt(D)
    bias,ix,rel,sv=anchor_bias(z,valid,self.K) if self.diag_anchor else (torch.zeros_like(content),None,None,None)
    state=e.new_full((B,self.K),1/self.K);ps=[];zs=[];states=[]
    for t in range(L):
        logits=content[:,t]+bias[:,t]+self.inertia*((1-self.mixing)*state+self.mixing/self.K).log()
        p=logits.softmax(-1);active=valid[:,t,None];state=torch.where(active,p,state)
        ps.append(torch.where(active,p,torch.zeros_like(p)));zs.append(logits);states.append(state)
    p=torch.stack(ps,1);mass=p.sum(1);att=p.transpose(1,2)/mass.clamp_min(self.eps)[:,:,None]
    v=torch.bmm(att,values)
    self.trace=dict(p=p,logits=zs,values=values,bias=bias,anchor_indices=ix,valid=valid,interests=v)
    return dict(interests=v,attention=att,assignments=p,evidence=torch.zeros_like(p),mass=mass,
                states=torch.stack(states,1),final_state=state)

def scores(v,w,c):
    a=torch.bmm(v,c.transpose(1,2));linear=(w[:,:,None]*a).sum(1)
    # Equivalent for normalized softmax w; preserve small Jensen correction without subtracting large logs.
    lse=linear+8*torch.log1p((w[:,:,None]*torch.expm1((a-linear[:,None,:])/8)).sum(1))
    posterior=(w.log()[:,:,None]+a/8).softmax(1)
    return a,linear,lse,posterior

class Diagnostic(Model):
    def __init__(self,args,corpus,variant):
        super().__init__(args,corpus);self.variant=variant;self.extractor.diag_anchor=variant in ['T2','T3']
        self.extractor.forward=types.MethodType(route,self.extractor)
    def forward(self,b,return_intermediate=False):
        # Provide original semantics even for A1; no evidence D is computed.
        ex=self.extractor;old=ex.forward
        z=self.item_encoder.llm_table[b['history_items'],:512]
        def with_z(this,e,lens,unused,use_evidence=False):return route(this,e,lens,z,False)
        ex.forward=types.MethodType(with_z,ex)
        try:o=super().forward(b,True)
        finally:ex.forward=old
        a,lin,lse,post=scores(o['interest_vectors'],o['interest_weights'],o['candidate_vectors'])
        o.update(prediction=lse if self.variant in ['T1','T3'] else lin,matches=a,linear=lin,lse=lse,posterior=post)
        return o

def synthetic():
    gen=torch.Generator().manual_seed(7);h=torch.randn(2,5,7,generator=gen,dtype=torch.double);e=torch.randn(2,2,7,generator=gen,dtype=torch.double)
    w=torch.full((2,4),.25,dtype=torch.double);z=torch.zeros(2,5,4,dtype=torch.double,requires_grad=True);p=z.softmax(-1)
    v=torch.bmm(p.transpose(1,2)/p.sum(1)[:,:,None],h);a,lin,lse,pi=scores(v,w,e)
    glin=torch.autograd.grad((lin[:,0]-lin[:,1]).sum(),z,retain_graph=True)[0]
    glse=torch.autograd.grad((lse[:,0]-lse[:,1]).sum(),z,retain_graph=True)[0]
    assert glin.abs().max()<1e-14 and glse.abs().max()<1e-14
    gv=torch.autograd.grad(lse[:,0].sum(),v,retain_graph=True)[0]
    torch.testing.assert_close(gv,pi[:,:,0,None]*e[:,None,0],atol=1e-14,rtol=1e-14)
    torch.testing.assert_close(lin,lse,atol=1e-14,rtol=1e-14)
    vv=v[:,:1];ww=torch.ones(2,1,dtype=torch.double);_,x,y,_=scores(vv,ww,e);torch.testing.assert_close(x,y,atol=1e-14,rtol=1e-14)
    zz=torch.randn(3,5,9,generator=gen,dtype=torch.double);valid=torch.tensor([[1,0,0,0,0],[1,1,1,0,0],[1,1,1,1,1]],dtype=torch.bool)
    bias,ix,_,_=anchor_bias(zz,valid,4);zz2=zz.clone();zz2[~valid]=1e6
    bias2,ix2,_,_=anchor_bias(zz2,valid,4)
    torch.testing.assert_close(bias,bias2,atol=0,rtol=0);assert torch.equal(ix,ix2)
    assert bias[0].abs().max()==0
    zeros=anchor_bias(torch.zeros_like(zz),valid,4)[0];assert zeros.abs().max()==0
    return dict(uniform_linear_route_gradient_max=norm(glin.abs().max()),uniform_lse_route_gradient_max=norm(glse.abs().max()),
                score_derivative_formula=True,K1_equivalent=True,padding_and_zero_semantics=True,length1_no_forced_distinction=True)

def probe(m,b,p_initial):
    m.eval();o=m(b);t=m.extractor.trace;p=t['p'];v=o['interest_vectors'];w=o['interest_weights'];valid=t['valid'];mass=p.sum(1)
    rank=GeneralModel.loss(m,o);q=m.extractor.query;We=m.extractor.We.weight
    losses=[GeneralModel.loss(m,{'prediction':o[key]}) for key in ['linear','lse']]
    qheads=[norm(torch.autograd.grad(x,q,retain_graph=True)[0]) for x in losses]
    target=[q,We,v,p]+t['logits'];gg=torch.autograd.grad(rank,target,retain_graph=True)
    gq,gWe,gv,gp=gg[:4];gz=torch.stack(gg[4:],1);projected=gp-gp.mean(-1,keepdim=True);direct=p*(gp-(p*gp).sum(-1,keepdim=True))
    gm=torch.autograd.grad((o['prediction'][:,0]-o['prediction'][:,1]).sum(),p,retain_graph=True)[0]
    margin_route=p*(gm-(p*gm).sum(-1,keepdim=True));pm=mass/b['lengths'][:,None]
    vm=v.mean(1,keepdim=True);gscaled=gv/w[:,:,None];poster=o['posterior']
    ans=dict(rank_loss=num(rank),query_task_norm=norm(gq),query_norm=norm(q),query_decay_norm=1e-6*norm(q),
        decay_to_task=1e-6*norm(q)/max(norm(gq),1e-30),We_task_norm=norm(gWe),value_task_norm=norm(gv),
        p_entropy=num(-(p[valid]*p[valid].clamp_min(1e-30).log()).sum(-1).mean()),p_temporal_std=stats((p-pm[:,None])[valid])['std'],
        p_delta_from_initial_RMS=0. if p_initial is None else stats((p.detach()-p_initial)[valid])['rms'],
        V_relative_slot_spread=norm(v-vm)/max(norm(v),1e-30),w_entropy=num(-(w*w.log()).sum(-1).mean()),
        p_task_norm=norm(gp),p_tangent_task_norm=norm(projected),p_tangent_fraction=norm(projected)/max(norm(gp),1e-30),
        logit_direct_gradient=norm(direct),logit_full_recurrence_gradient=norm(gz[valid]),
        margin_route_gradient_RMS=stats(margin_route[valid])['rms'],
        gradient_div_w_head_difference=norm(gscaled-gscaled.mean(1,keepdim=True))/max(norm(gscaled),1e-30),
        posterior_pos_minus_neg_RMS=stats(poster[:,:,0]-poster[:,:,1])['rms'],posterior_minus_w_RMS=stats(poster-w[:,:,None])['rms'],
        interest_match_centered_RMS=stats(o['matches']-o['matches'].mean(1,keepdim=True))['rms'],
        fixed_graph_query_grad_linear=qheads[0],fixed_graph_query_grad_lse=qheads[1],
        lse_minus_linear_margin_RMS=stats((o['lse'][:,0]-o['lse'][:,1])-(o['linear'][:,0]-o['linear'][:,1]))['rms'],
        score_margin=stats(o['prediction'][:,0]-o['prediction'][:,1]),semantic_bias=stats(t['bias'][valid]))
    if t['anchor_indices'] is not None:
        ix=t['anchor_indices'];z=m.item_encoder.llm_table[b['history_items'],:512];unit=torch.nn.functional.normalize(z,dim=-1)
        anchors=unit[torch.arange(len(z))[:,None],ix];dist=1-torch.bmm(anchors,anchors.transpose(1,2))
        tri=torch.ones(4,4,dtype=torch.bool).triu(1);ans['anchor_pair_cosine_distance']=stats(dist[:,tri])
    # Small candidate replacement test: history-derived route independent of labels/candidates/batch peers.
    with torch.no_grad():
        first={k:x[:8] for k,x in b.items()};x=m(first)
        changed={**first,'item_id':first['item_id'].flip(1)};y=m(changed)
        assert torch.equal(x['prefix_assignments'],y['prefix_assignments'])
        torch.testing.assert_close(x['prediction'].flip(1),y['prediction'],atol=0,rtol=0)
        single=m({k:z[:1] for k,z in first.items()});torch.testing.assert_close(x['prediction'][:1],single['prediction'],atol=1e-7,rtol=1e-5)
    ans['target_isolation_and_candidate_batch_check']=True
    assert all(math.isfinite(ans[k]) for k in ['rank_loss','query_task_norm','We_task_norm','p_entropy'])
    return ans,p.detach().clone()

def main():
    torch.set_num_threads(2);signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError('diagnostic >175s')));signal.alarm(175)
    start=time.monotonic();r=dict(synthetic=synthetic(),datasets={},device='cpu',threads=2,dev_or_test=False,full_training=False)
    for ds in ['beauty','ml-1m']:
        c,b,meta=fixture(ds);data=dict(fixture=meta,models={});shared=None
        for variant in ['T0','T1','T2','T3']:
            args=Model.parse_model_args(argparse.ArgumentParser()).parse_args([]);args.device=torch.device('cpu');args.buffer=0
            args.model_path='unused';args.prefix_variant='A1';args.llm_emb_path=str(ROOT/'data'/ds/'handled/llm_table_pca1536.pkl')
            torch.manual_seed(42);m=Diagnostic(args,c,variant)
            if shared is None:
                shared={k:p.clone() for k,p in m.state_dict().items()}
                ref=Model(args,c);ref.load_state_dict(shared);m.eval();ref.eval()
                with torch.no_grad():
                    small={k:x[:8] for k,x in b.items()};aa=m(small);rr=ref(small,True)
                    assert torch.equal(aa['prefix_assignments'],rr['prefix_assignments'])
                    torch.testing.assert_close(aa['prediction'],rr['prediction'],atol=1e-9,rtol=1e-5)
                data['T0_matches_original_A1_route_exactly']=True
                data['T0_prediction_vs_A1_max_roundoff']=float((aa['prediction']-rr['prediction']).abs().max())
                del ref
            else:assert all(torch.equal(p,shared[k]) for k,p in m.state_dict().items())
            opt=torch.optim.Adam(m.customize_parameters(),lr=.004 if ds=='beauty' else .001,weight_decay=1e-6)
            torch.manual_seed(200);mr=dict(parameters=m.count_variables(),snapshots={},updates=[]);p0=None
            for step in range(16):
                if step in [0,5,15]:
                    mr['snapshots'][str(step)],p=probe(m,b,p0)
                    if p0 is None:p0=p
                if step==15:break
                m.train();opt.zero_grad();o=m(b);rank=GeneralModel.loss(m,o);loss=m.loss(o)
                task=torch.autograd.grad(rank,m.extractor.query,retain_graph=True)[0] if step in [0,4,14] else None
                loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
                if task is not None:
                    q=m.extractor.query;old=q.detach().clone();decay=1e-6*old
                    row=dict(update=step+1,task_norm=norm(task),decay_norm=norm(decay),loss=num(rank))
                opt.step()
                if task is not None:
                    du=q.detach()-old;row.update(update_norm=norm(du),update_cos_negative_task=cos(du,-task),update_cos_negative_decay=cos(du,-decay));mr['updates'].append(row)
            # Same bounded optimization/timing budget for every group, no new state saved.
            m.train()
            for _ in range(2):opt.zero_grad();o=m(b);loss=m.loss(o);loss.backward();opt.step()
            times=[]
            for _ in range(6):
                ts=time.perf_counter();opt.zero_grad();o=m(b);loss=m.loss(o);loss.backward();opt.step();times.append(time.perf_counter()-ts)
            mr['CPU_step_seconds_mean']=float(np.mean(times));mr['CPU_step_seconds_std']=float(np.std(times));mr['updates_total']=23
            data['models'][variant]=mr
            print('PROGRESS '+ds+' '+variant,flush=True)
            del m,opt
        r['datasets'][ds]=data
    r['elapsed_seconds']=time.monotonic()-start;signal.alarm(0);print('RESULT_JSON '+json.dumps(r,allow_nan=False))
if __name__=='__main__':main()