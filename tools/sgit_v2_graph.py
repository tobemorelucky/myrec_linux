"""Versioned semantic-bridge completion of immutable V1 cross-fit Train graphs."""
import numpy as np
import torch

def build_bridge(base):
    """i --semantic--> j --real directed Train edge--> d; max support is NOT a user sum."""
    result={k:v.clone() if torch.is_tensor(v) else dict(v) for k,v in base.items()}
    F,N,R=base['transition_ids'].shape
    names={'ids':np.int64,'prob':np.float32,'support':np.int32,
           'path_count':np.int32,'witness':np.int64}
    all_values={k:np.zeros((F,N,R),dtype=d) for k,d in names.items()}
    stats=[]
    for f in range(F):
        tid=base['transition_ids'][f].numpy()
        tp=base['transition_prob'][f].numpy()
        ts=base['transition_support'][f].numpy()
        sid=base['semantic_ids'][f].numpy()
        sp=base['semantic_prob'][f].numpy()
        for i in range(1,N):
            j=sid[i]
            targets=tid[j].reshape(-1)
            supports=ts[j].reshape(-1)
            weight=(sp[i,:,None]*tp[j]*(ts[j]/(ts[j]+1.0))).reshape(-1)
            intermediates=np.repeat(j,R)
            valid=(targets>0)&(intermediates>0)&(weight>0)
            if not valid.any():continue
            d=targets[valid];w=weight[valid];s=supports[valid];via=intermediates[valid]
            unique,back=np.unique(d,return_inverse=True)
            mass=np.bincount(back,weights=w)
            sup=np.zeros(len(unique),np.int32);np.maximum.at(sup,back,s)
            count=np.bincount(back)
            best=np.full(len(unique),-np.inf);np.maximum.at(best,back,w)
            # Deterministic strongest-path witness; support is conservative max.
            witness=np.full(len(unique),N,np.int64)
            np.minimum.at(witness,back[w==best[back]],via[w==best[back]])
            pick=np.lexsort((unique,-mass))[:R];n=len(pick)
            for name,x in [('ids',unique),('prob',mass/mass.sum()),('support',sup),
                           ('path_count',count),('witness',witness)]:
                all_values[name][f,i,:n]=x[pick]
        valid=all_values['ids'][f]>0
        direct=base['transition_ids'][f].numpy()
        duplicate=(all_values['ids'][f,:,:,None]==direct[:,None,:]).any(-1)&valid
        stats.append(dict(fold=f,edges=int(valid.sum()),source_nodes=int(valid.any(-1).sum()),
                          extra_edges_vs_direct=int((valid&~duplicate).sum()),
                          overlap_fraction=float(duplicate.sum()/max(valid.sum(),1))))
    for name,value in all_values.items():result['bridge_'+name]=torch.from_numpy(value)
    result['metadata']=dict(base['metadata'],schema='sgit_semantic_bridge_v2',
        base_schema=base['metadata']['schema'],bridge_R=R,bridge_stats=stats,
        bridge_formula='sum_j semantic_prob(i,j)*Train_P(d|j)*support(j,d)/(support(j,d)+1)',
        support_policy='maximum distinct-user support across constituent j->d edges; NOT summed independent users',
        bridge_uses='immutable same-fold V1 semantic and directed Train tables; no held-out labels')
    return result

def coverage(payload,users,history,lengths,target):
    """Offline labels only: does not feed back into or mutate the graph."""
    u=torch.as_tensor(users,dtype=torch.long);h=torch.as_tensor(history,dtype=torch.long)
    length=torch.as_tensor(lengths,dtype=torch.long);y=torch.as_tensor(target,dtype=torch.long)
    valid=torch.arange(h.shape[1])[None]<length[:,None]
    f=u%3
    direct=payload['transition_ids'][f[:,None],h]
    bridge=payload['bridge_ids'][f[:,None],h]
    hitT=((direct==y[:,None,None])&valid[...,None]).any((-1,-2))
    hitB=((bridge==y[:,None,None])&valid[...,None]).any((-1,-2))
    last=h[torch.arange(len(h)),(length-1).clamp_min(0)]
    lastT=(payload['transition_ids'][f,last]==y[:,None]).any(-1)&(length>0)
    lastB=(payload['bridge_ids'][f,last]==y[:,None]).any(-1)&(length>0)
    support=payload['transition_support'][f,last].amax(-1)
    buckets={}
    for label,mask in [('low_le1',support<=1),('mid_2to4',(support>=2)&(support<=4)),('high_ge5',support>=5)]:
        n=int(mask.sum())
        buckets[label]=dict(n=n,direct=float(hitT[mask].float().mean()) if n else None,
            extra=float((hitB[mask]&~hitT[mask]).float().mean()) if n else None,
            last_source_extra=float((lastB[mask]&~lastT[mask]).float().mean()) if n else None)
    return dict(n=len(h),direct=float(hitT.float().mean()),bridge=float(hitB.float().mean()),
        union=float((hitT|hitB).float().mean()),extra=float((hitB&~hitT).float().mean()),
        extra_users=int((hitB&~hitT).sum()),last_direct=float(lastT.float().mean()),
        last_extra=float((lastB&~lastT).float().mean()),support_buckets=buckets)
