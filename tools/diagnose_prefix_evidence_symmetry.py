"""Bounded Train-only attribution; no model/source/asset/checkpoint edits.
Exact forward trace exists solely to retain actual recurrence nodes for autograd.
"""
import argparse, json, math, pickle, signal, sys, time, types, subprocess, os
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models.BaseModel import GeneralModel
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF as Base
from models.sequential.LLMMIRecASPCFPrefixEvidence import LLMMIRecASPCFPrefixEvidence as Model

def num(x):return float(x.detach())
def norm(x):return 0. if x is None else num(x.norm())
def cos(x,y):return num(torch.nn.functional.cosine_similarity(x.flatten(),y.flatten(),dim=0,eps=1e-30))
def stats(x):
    a=x.detach().double().flatten().cpu().numpy()
    if not a.size:return dict(n=0)
    return dict(n=int(a.size),mean=float(a.mean()),std=float(a.std()),rms=float(np.sqrt(np.mean(a*a))),
        p01=float(np.quantile(a,.01)),p50=float(np.median(a)),p95=float(np.quantile(a,.95)),p99=float(np.quantile(a,.99)),max_abs=float(np.abs(a).max()))
def centered(x):return x-x.mean(-1,keepdim=True)

def traced(self,e,lengths,z,use_evidence=True):
    B,L,D=e.shape;valid=torch.arange(L,device=e.device)[None]<lengths[:,None]
    values=self.Wv(e);content=torch.einsum('bld,kd->blk',self.We(e),self.query)/math.sqrt(D)
    kernel,pairs=self.semantic_kernel(z,valid) if use_evidence else (None,None)
    state=e.new_full((B,self.K),1/self.K);ps=[];ds=[];ss=[];zs=[];prs=[];rawps=[];rawds=[]
    for t in range(L):
        d=self.evidence(torch.stack(ps,1),kernel[:,t,:t],pairs[:,t,:t]) if use_evidence and t else torch.zeros_like(state)
        prior=self.inertia*torch.log((1-self.mixing)*state+self.mixing/self.K)
        logits=content[:,t]+self.strength*d+prior;p=logits.softmax(-1)
        if self.K==1:p=torch.ones_like(p)
        active=valid[:,t,None];state=torch.where(active,p,state)
        ps.append(torch.where(active,p,torch.zeros_like(p)));ds.append(torch.where(active,d,torch.zeros_like(d)));ss.append(state)
        zs.append(logits);prs.append(prior);rawps.append(p);rawds.append(d)
    p=torch.stack(ps,1);d=torch.stack(ds,1);mass=p.sum(1)
    attention=p.transpose(1,2)/mass.clamp_min(self.eps).unsqueeze(-1)
    interests=torch.bmm(attention,values)
    self.trace=dict(values=values,content=content,prior=torch.stack(prs,1),logits=zs,rawps=rawps,rawds=rawds,
                    p=p,d=d,valid=valid,kernel=kernel,pairs=pairs,interests=interests)
    return dict(interests=interests,attention=attention,assignments=p,evidence=d,mass=mass,states=torch.stack(ss,1),final_state=state)

def fixture(ds):
    c=pickle.load(open(ROOT/'data'/ds/'SeqReader.pkl','rb'));df=c.data_df['train']
    eligible=np.flatnonzero(df.position.to_numpy()>0);rng=np.random.default_rng(20261016)
    indices=rng.choice(eligible,1024,replace=False);hs=[];ls=[];cs=[]
    for idx in indices:
        r=df.iloc[int(idx)];u=int(r.user_id);pos=int(r.position);hist=c.user_his[u][max(0,pos-20):pos]
        assert len(hist)>0 and all(v[1]<=r.time for v in hist)
        neg=int(rng.integers(1,c.n_items))
        while neg in c.train_clicked_set[u]:neg=int(rng.integers(1,c.n_items))
        h=[int(v[0]) for v in hist];hs.append(h+[0]*(20-len(h)));ls.append(len(h));cs.append([int(r.item_id),neg])
    batch={k:torch.tensor(v,device='cuda') for k,v in dict(history_items=hs,lengths=ls,item_id=cs).items()}
    return c,batch,dict(samples=1024,users=int(df.iloc[indices].user_id.nunique()),train_batches=math.ceil(len(eligible)/1024),
        length_counts={str(i):int(np.sum(np.array(ls)==i)) for i in sorted(set(ls))},sample_rng=20261016,model_seed=42,update_rng=200)

def math_tests():
    g=torch.Generator().manual_seed(8);h=torch.randn(3,5,7,generator=g,dtype=torch.double)
    logits=torch.zeros(3,5,4,dtype=torch.double,requires_grad=True);p=logits.softmax(-1)
    w=torch.full((3,4),.25,dtype=torch.double);mass=p.sum(1)
    v=torch.bmm(p.transpose(1,2)/mass[:,:,None],h);u=(w[:,:,None]*v).sum(1)
    gu=torch.randn(3,7,generator=g,dtype=torch.double)
    gz=torch.autograd.grad((u*gu).sum(),logits,retain_graph=True)[0]
    assert gz.abs().max()<1e-14
    J=w[:,None,:,None]/mass[:,None,:,None]*(h[:,:,None,:]-v[:,None,:,:])
    Jz=p[:,:,:,None]*(J-(p[:,:,:,None]*J).sum(2,keepdim=True))
    expected=(Jz*gu[:,None,None,:]).sum(-1)
    torch.testing.assert_close(gz,expected,atol=1e-14,rtol=1e-14)
    wu=torch.tensor([[.1,.2,.3,.4]],dtype=torch.double).expand(3,-1)
    u2=(wu[:,:,None]*v).sum(1);g2=torch.autograd.grad((u2*gu).sum(),logits)[0]
    assert g2.norm()>1e-4
    return dict(uniform_logit_gradient_max=norm(gz.abs().max()),nonuniform_w_control_gradient_norm=norm(g2),analytic_jacobian_pass=True)

def probe(m,batch,variant):
    m.eval();orig=m.extractor.forward;is_prefix=variant!='C0'
    if is_prefix:m.extractor.forward=types.MethodType(traced,m.extractor)
    out=m(batch,True);rank=GeneralModel.loss(m,out)
    q=m.extractor.query;v=out['interest_vectors'];u=out['user_vector'];att=out['attention_maps'];w=out['interest_weights']
    margin=out['prediction'][:,0]-out['prediction'][:,1];valid=batch['history_items']>0;lens=batch['lengths']
    split=[]
    for sl in [slice(0,512),slice(512,None)]:
        splitloss=GeneralModel.loss(m,{'prediction':out['prediction'][sl]})
        split.append(norm(torch.autograd.grad(splitloss,q,retain_graph=True)[0]))
    targets=[q,v,u,att]
    if is_prefix:
        t=m.extractor.trace;targets += [t['p'],m.extractor.We.weight]+t['logits']
    else:targets += [m.extractor.Wq.weight,m.extractor.Wk.weight]
    gradients=torch.autograd.grad(rank,targets,retain_graph=True,allow_unused=True)
    gq,gv,gu,ga=gradients[:4];vm=v.mean(1,keepdim=True)
    ans=dict(rank_loss=num(rank),margin=stats(margin),sigmoid_negative_margin=stats(torch.sigmoid(-margin)),
        query_task_norm=norm(gq),query_norm=norm(q),query_decay_norm=1e-6*norm(q),split512_query_task_norms=split,query_gradient_relative_to_user=norm(gq)/max(norm(gu),1e-30),
        value_task_norm=norm(gv),user_task_norm=norm(gu),attention_task_norm=norm(ga),
        w=stats(w),w_entropy=num(-(w*w.clamp_min(1e-30).log()).sum(-1).mean()),
        V_relative_slot_spread=norm(v-vm)/max(norm(v),1e-30),V_absolute_slot_spread=stats(v-vm),
        user_norm=norm(u),prediction=stats(out['prediction']))
    if is_prefix:
        p=t['p'];mass=p.sum(1);gp=gradients[4];gWe=gradients[5];gz=torch.stack(gradients[6:],1)
        hraw=out['history_vectors'];hmean=hraw.sum(1)/lens[:,None]
        values=t['values'];vmean=(values*valid[:,:,None]).sum(1)/lens[:,None]
        ans['history_raw_temporal_relative_spread']=norm((hraw-hmean[:,None])[valid])/max(norm(hraw[valid]),1e-30)
        ans['values_temporal_relative_spread']=norm((values-vmean[:,None])[valid])/max(norm(values[valid]),1e-30)
        cc=centered(t['content']);cm=(cc*valid[:,:,None]).sum(1)/lens[:,None]
        ans['content_temporal_centered_rms']=stats((cc-cm[:,None])[valid])['rms']
        direct=p*(gp-(p*gp).sum(-1,keepdim=True));projected=gp-gp.mean(-1,keepdim=True)
        pm=mass/lens[:,None];temporal=p-pm[:,None,:];ratio=w/mass.clamp_min(1e-30)
        ans.update(p_entropy=num(-(p[valid]*p[valid].clamp_min(1e-30).log()).sum(-1).mean()),
            p_effective_occupancy=num(torch.exp(-(pm*pm.clamp_min(1e-30).log()).sum(-1)).mean()),
            p=stats(p[valid]),p_temporal_variation=stats(temporal[valid]),normalized_attention_slot_spread=stats(att-att.mean(1,keepdim=True)),
            p_direct_task_norm=norm(gp),p_row_tangent_task_norm=norm(projected),
            p_row_tangent_to_unconstrained_norm=norm(projected)/max(norm(gp),1e-30),
            logit_direct_pool_task_norm=norm(direct),logit_with_recurrence_task_norm=norm(gz[valid]),
            We_task_norm=norm(gWe),We_norm=norm(m.extractor.We.weight),
            w_over_mass_relative_slot_spread=norm(ratio-ratio.mean(-1,keepdim=True))/max(norm(ratio),1e-30),
            content_centered=stats(centered(t['content'])[valid]),prior_centered=stats(centered(t['prior'])[valid]),
            D=stats(t['d'][valid]),D_centered=stats(centered(t['d'])[valid]),groups={})
        # Double precision pooling-only check, independent of recurrence and float32 cancellation.
        pp=p.detach().double().requires_grad_();hh=t['values'].detach().double();ww=w.detach().double()
        vv=torch.bmm(pp.transpose(1,2)/pp.sum(1)[:,:,None],hh);uu=(vv*ww[:,:,None]).sum(1)
        coeff=gu.detach().double();gpd=torch.autograd.grad((uu*coeff).sum(),pp)[0]
        gd=pp*(gpd-(pp*gpd).sum(-1,keepdim=True))
        ans['pooling_double_logit_task_norm']=norm(gd)
        ans['pooling_double_vs_float_direct_relative_error']=norm(gd-direct.double())/max(norm(gd),1e-30)
        J=ww[:,None,:,None]/pp.detach().sum(1)[:,None,:,None]*(hh[:,:,None,:]-vv.detach()[:,None,:,:])
        Jz=pp.detach()[:,:,:,None]*(J-(pp.detach()[:,:,:,None]*J).sum(2,keepdim=True))
        ans['analytic_logit_jacobian_norm']=norm(Jz)
        ans['analytic_logit_jacobian_relative_to_unconstrained']=norm(Jz)/max(norm(J),1e-30)
        for lo,hi in [(1,2),(3,5),(6,10),(11,20)]:
            users=(lens>=lo)&(lens<=hi);mask=valid&users[:,None]
            if not int(users.sum()):continue
            ans['groups'][f'{lo}-{hi}']=dict(users=int(users.sum()),p_entropy=num(-(p[mask]*p[mask].clamp_min(1e-30).log()).sum(-1).mean()),
                p_temporal_std=stats(temporal[mask])['std'],D=stats(t['d'][mask]),
                D_centered_rms=stats(centered(t['d'])[mask])['rms'],content_rms=stats(centered(t['content'])[mask])['rms'],
                V_relative_slot_spread=norm((v-vm)[users])/max(norm(v[users]),1e-30))
        # Verify instrumentation has not changed original mathematical forward.
        m.extractor.forward=orig
        with torch.no_grad():reference=m(batch,True)
        assert torch.equal(reference['prediction'],out['prediction'])
        assert torch.equal(reference['prefix_assignments'],p)
        ans['trace_matches_original_prediction_and_p_exactly']=True
        evidence=t['d'].detach().cpu();assignments=p.detach().cpu();vs=v.detach().cpu();us=u.detach().cpu()
        m.extractor.trace=None
    else:
        ans['Wq_task_norm']=norm(gradients[4]);ans['Wk_task_norm']=norm(gradients[5])
        # C0 attention is normalized across time, not across interests.
        ans['attention_logit_task_norm']=norm(att*(ga-(att*ga).sum(-1,keepdim=True)))
        evidence=assignments=vs=us=None
    return ans,(evidence,assignments,vs,us)

def raw_semantics(m,b):
    with torch.no_grad():
        z=m.item_encoder.llm_table[b['history_items'],:512];unit=torch.nn.functional.normalize(z,dim=-1,eps=1e-8)
        cosine=torch.bmm(unit,unit.transpose(1,2));valid=(b['history_items']>0)&(z.norm(dim=-1)>1e-8)
        pairs=valid[:,:,None]&valid[:,None,:]&torch.ones(20,20,device='cuda',dtype=torch.bool).tril(-1)
        r=torch.exp((cosine-1)/.2);counts=pairs.sum(-1);mean=(r*pairs).sum(-1)/counts.clamp_min(1)
        var=(((r-mean[:,:,None])**2)*pairs).sum(-1)/counts.clamp_min(1)
        return dict(cosine=stats(cosine[pairs]),kernel=stats(r[pairs]),
            within_prefix_kernel_CV=stats((var.sqrt()/mean.clamp_min(1e-30))[counts>1]))

def timing(m,b,opt):
    m.train()
    for _ in range(2):opt.zero_grad();loss=m.loss(m(b));loss.backward();opt.step()
    torch.cuda.synchronize();t=time.perf_counter()
    for _ in range(6):opt.zero_grad();loss=m.loss(m(b));loss.backward();opt.step()
    torch.cuda.synchronize();seconds=(time.perf_counter()-t)/6
    # Eight bounded timing updates after snapshot15; no saved model, same existing Adam state.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        opt.zero_grad();loss=m.loss(m(b));loss.backward();torch.cuda.synchronize()
    rows=sorted(prof.key_averages(),key=lambda e:e.self_device_time_total,reverse=True)
    keys=['aten::stack','aten::sum','aten::mul','aten::log','aten::softmax','aten::_softmax','aten::bmm','aten::exp','aten::_local_scalar_dense','aten::item','cudaDeviceSynchronize']
    selected={e.key:dict(calls=e.count,cpu_us=e.self_cpu_time_total,gpu_us=e.self_device_time_total) for e in rows if e.key in keys}
    m.eval()
    with torch.no_grad():
        sem=m.item_encoder.llm_table[b['history_items'],:512];valid=b['history_items']>0
        for _ in range(2):m.extractor.semantic_kernel(sem,valid)
        torch.cuda.synchronize();kt=time.perf_counter()
        for _ in range(12):m.extractor.semantic_kernel(sem,valid)
        torch.cuda.synchronize();kernel_seconds=(time.perf_counter()-kt)/12
    return dict(kernel_only_forward_seconds=kernel_seconds,warmed_train_step_including_optimizer_seconds=seconds,top_gpu=[dict(op=e.key,calls=e.count,gpu_us=e.self_device_time_total) for e in rows[:8]],selected=selected,
                total_cpu_operator_calls=sum(e.count for e in rows if e.key.startswith('aten::')))

def main():
    torch.set_num_threads(2);signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError('bounded diagnostic >175s')));signal.alarm(175)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
    gpu=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert not any(r.startswith(gpu+',') for r in apps.splitlines()),'GPU1 occupied'
    start=time.monotonic();result=dict(math=math_tests(),datasets={},test_read=False,dev_read=False,full_training=False,optimizer_correction='bias decay excluded, matching BaseRunner; prior diagnostic used all params',parameters_changed_on_disk=False)
    for ds in ['beauty','ml-1m']:
        corpus,b,meta=fixture(ds);r=dict(fixture=meta,models={});paired={};shared=None
        for variant in ['C0','A1','A2']:
            cls=Base if variant=='C0' else Model;args=cls.parse_model_args(argparse.ArgumentParser()).parse_args([])
            args.device=torch.device('cpu');args.buffer=0;args.model_path='unused';args.llm_emb_path=str(ROOT/'data'/ds/'handled/llm_table_pca1536.pkl')
            if variant!='C0':args.prefix_variant=variant
            torch.manual_seed(42);m=cls(args,corpus)
            if variant=='A1':shared={n:p.clone() for n,p in m.state_dict().items()}
            if variant=='A2':assert all(torch.equal(p,shared[n]) for n,p in m.state_dict().items())
            m.cuda();torch.cuda.reset_peak_memory_stats();opt=torch.optim.Adam(m.customize_parameters(),lr=.004 if ds=='beauty' else .001,weight_decay=1e-6)
            torch.manual_seed(200);mr=dict(snapshots={},updates=[],optimizer=dict(lr=opt.param_groups[0]['lr'],weight_decay=1e-6,betas=[.9,.999],eps=1e-8,bias_weight_decay=0,groups_match_BaseRunner=True),parameters=m.count_variables())
            if variant=='A2':r['raw_semantics']=raw_semantics(m,b)
            for step in range(16):
                if step in [0,5,15]:
                    mr['snapshots'][str(step)],pair=probe(m,b,variant)
                    if step==15 and variant!='C0':paired[variant]=pair
                if step==15:break
                m.train();opt.zero_grad();out=m(b);rank=GeneralModel.loss(m,out);loss=m.loss(out)
                task=torch.autograd.grad(rank,m.extractor.query,retain_graph=True)[0] if step in [0,4,14] else None
                loss.backward()
                if step in [0,4,14]:
                    q=m.extractor.query;old=q.detach().clone();decay=1e-6*old
                    row=dict(step=step+1,ranking=num(rank),total_loss=num(loss),query_task_norm=norm(task),decay_norm=norm(decay),
                        task_decay_cosine=cos(task,decay),actual_backward_norm=norm(q.grad),
                        relation_adds_no_query_gradient=bool(torch.equal(task,q.grad)))
                opt.step()
                if step in [0,4,14]:
                    update=q.detach()-old;row.update(actual_update_norm=norm(update),update_cos_negative_task=cos(update,-task),update_cos_negative_decay=cos(update,-decay));mr['updates'].append(row)
            if variant!='C0' and ds=='ml-1m':mr['timing']=timing(m,b,opt)
            mr['peak_MiB']=torch.cuda.max_memory_allocated()/2**20;r['models'][variant]=mr
            del m,opt;torch.cuda.empty_cache()
        a,bp=paired['A1'],paired['A2'];r['A2_minus_A1_step15']=dict(p_RMS=stats(bp[1]-a[1])['rms'],V_relative_difference=norm(bp[2]-a[2])/max(norm(a[2]),1e-30),user_relative_difference=norm(bp[3]-a[3])/max(norm(a[3]),1e-30))
        result['datasets'][ds]=r
    result['elapsed_seconds']=time.monotonic()-start;signal.alarm(0);print('DIAGNOSTIC_JSON '+json.dumps(result,allow_nan=False))
if __name__=='__main__':main()