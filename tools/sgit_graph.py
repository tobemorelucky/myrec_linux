"""Train-only, leave-user-fold-out sparse directed item graphs; no held-out inputs."""
import time
import numpy as np
import torch
from torch.nn import functional as F

def transition_graph(histories,n_items,fold,folds=3,R=8):
    """Adjacent counts and distinct-user support, excluding ALL users in query fold."""
    source=[];target=[];users=[];catalog=[]
    for user,seq in histories.items():
        if user%folds==fold:continue
        ids=np.asarray([x[0] for x in seq],np.int64)
        if len(ids):catalog.append(ids)
        if len(ids)>1:
            source.append(ids[:-1]);target.append(ids[1:]);users.append(np.full(len(ids)-1,user,np.int64))
    ids=np.zeros((n_items,R),np.int64);counts=np.zeros((n_items,R),np.int32)
    support=np.zeros((n_items,R),np.int32);prob=np.zeros((n_items,R),np.float32)
    eligible=np.unique(np.concatenate(catalog)) if catalog else np.empty(0,np.int64)
    if not source:return dict(ids=ids,counts=counts,support=support,prob=prob),eligible,dict(raw_edges=0,events=0)
    a,b,u=map(np.concatenate,[source,target,users]);key=a*n_items+b
    keys,c=np.unique(key,return_counts=True)
    n_users=max(histories)+1
    pair_user=np.unique(key*n_users+u)//n_users
    sk,su=np.unique(pair_user,return_counts=True);assert np.array_equal(keys,sk)
    src,dst=keys//n_items,keys%n_items
    row_sum=np.bincount(src,weights=c,minlength=n_items)
    order=np.lexsort((dst,-c,src));s=src[order];j=np.arange(len(s))
    rank=j-np.maximum.accumulate(np.where(np.r_[True,s[1:]!=s[:-1]],j,0))
    take=rank<R;pick=order[take];rr=rank[take];ss=src[pick]
    ids[ss,rr]=dst[pick];counts[ss,rr]=c[pick];support[ss,rr]=su[pick]
    prob[ss,rr]=c[pick]/row_sum[ss]
    return dict(ids=ids,counts=counts,support=support,prob=prob),eligible,dict(raw_edges=len(keys),events=len(a))

@torch.no_grad()
def semantic_graph(unit,eligible,R=8,query_block=128,catalog_block=2048,temperature=.1):
    """Exact blocked cosine TopR over OTHER-fold Train catalog; no dense NxN."""
    N=len(unit);ids=torch.zeros(N,R,dtype=torch.long);sims=torch.zeros(N,R);prob=torch.zeros(N,R)
    candidates=torch.as_tensor(eligible,dtype=torch.long)
    candidates=candidates[(candidates>0)&(unit[candidates].norm(dim=-1)>1e-8)]
    if not len(candidates):return dict(ids=ids,sims=sims,prob=prob)
    for start in range(0,N,query_block):
        q=unit[start:start+query_block]
        best=q.new_full((len(q),R),-torch.inf);which=torch.zeros(len(q),R,dtype=torch.long)
        rows=torch.arange(start,start+len(q))[:,None]
        for part in candidates.split(catalog_block):
            values=q@unit[part].T
            values.masked_fill_((rows==part[None])|(rows==0)|(q.norm(dim=-1)<=1e-8)[:,None],-torch.inf)
            both=torch.cat([best,values],1)
            choices=torch.cat([which,part[None].expand(len(q),-1)],1)
            best,rank=both.topk(R,dim=1);which=choices.gather(1,rank)
        valid=torch.isfinite(best)&(best>0)
        logits=(best/temperature).masked_fill(~valid,torch.finfo(best.dtype).min)
        weights=logits.softmax(-1)*valid;weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
        ids[start:start+len(q)]=which.masked_fill(~valid,0)
        sims[start:start+len(q)]=best.masked_fill(~valid,0)
        prob[start:start+len(q)]=weights
    return dict(ids=ids,sims=sims,prob=prob)

def build_graphs(histories,llm_table,R=8,folds=3):
    if R<1 or folds<2:raise ValueError("Invalid sparse graph budget")
    start=time.perf_counter();N=len(llm_table)
    unit=F.normalize(llm_table.detach().float().cpu(),dim=-1);unit[0]=0
    tensors={};stats=[]
    for fold in range(folds):
        c,catalog,raw=transition_graph(histories,N,fold,folds,R)
        s=semantic_graph(unit,catalog,R)
        queries=[np.asarray([v[0] for v in seq],np.int64) for u,seq in histories.items() if u%folds==fold]
        query=np.concatenate(queries) if queries else np.empty(0,np.int64)
        stats.append(dict(fold=fold,excluded_user_rule="user_id % folds == fold",catalog_nodes=len(catalog),**raw,
            retained_transition_edges=int((c['ids']>0).sum()),semantic_edges=int((s['ids']>0).sum()),
            transition_source_nodes=int((c['ids']>0).any(-1).sum()),semantic_source_nodes=int((s['ids']>0).any(-1).sum()),
            Train_query_occurrence_transition_coverage=float((c['ids'][query]>0).any(-1).mean()) if len(query) else 0.,
            Train_query_occurrence_semantic_coverage=float((s['ids'][query]>0).any(-1).float().mean()) if len(query) else 0.))
        for rel,values in [('transition',c),('semantic',s)]:
            for name,value in values.items():tensors.setdefault(rel+'_'+name,[]).append(torch.as_tensor(value))
    tensors={name:torch.stack(value) for name,value in tensors.items()}
    for name,value in tensors.items():
        if value.is_floating_point() and not torch.isfinite(value).all():raise ValueError('Nonfinite graph tensor: '+name)
    return tensors,dict(folds=folds,R=R,n_items=N,build_seconds=time.perf_counter()-start,per_fold=stats,
        tensor_bytes=sum(x.numel()*x.element_size() for x in tensors.values()))
