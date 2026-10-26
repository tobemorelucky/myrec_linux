"""Sparse candidate ID joins and candidate-conditioned interest selection; no item-universe matrix."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class CandidateTransitionIndex(nn.Module):
    def __init__(self,payload,direct=True,bridge=True):
        super().__init__();self.metadata=payload['metadata']
        if self.metadata['schema']!='sgit_candidate_index_v3' or self.metadata['scope']!='full_crossfit_train':
            raise ValueError('Expected Train-only calibrated candidate index')
        for rel,use in [('transition',direct),('bridge',bridge)]:
            if not use:continue
            for key in ['ids','value']:
                x=payload[rel+'_'+key]
                if x.is_floating_point() and not torch.isfinite(x).all():raise ValueError('Nonfinite '+rel)
                self.register_buffer(rel+'_'+key,x.detach().clone(),persistent=False)
    def lookup(self,history,users,relation):
        f=users.remainder(self.metadata['folds'])
        return {k:getattr(self,relation+'_'+k)[f[:,None],history] for k in ['ids','value']}
    def bytes(self):return sum(t.numel()*t.element_size() for t in self.buffers())

def sparse_candidate_read(A,valid,edges,candidates):
    """O(B*C log C + B*L*R log C + B*K*(L*R+C)); duplicates mapped back exactly."""
    B,K,L=A.shape;C=candidates.shape[1]
    if C==0:return A.new_zeros((B,K,0))
    sorted_ids,order=candidates.sort(-1)
    edge_ids=edges['ids'].flatten(1).contiguous()
    left=torch.searchsorted(sorted_ids.contiguous(),edge_ids)
    at=left.clamp_max(C-1)
    found=(left<C)&(sorted_ids.gather(1,at)==edge_ids)&(edge_ids>0)
    allowed=valid[...,None]&(edges['ids']>0)&(edges['value']!=0)
    found=found&allowed.flatten(1)
    contribution=(A[...,None]*edges['value'][:,None]).reshape(B,K,-1)
    contribution=contribution*found[:,None]
    first=A.new_zeros(B,K,C).scatter_add(2,at[:,None].expand(-1,K,-1),contribution)
    # Repeated candidate slots each get the same mass; never add their copies.
    duplicate_first=torch.searchsorted(sorted_ids.contiguous(),sorted_ids.contiguous())
    filled=first.gather(2,duplicate_first[:,None].expand(-1,K,-1))
    inverse=order.argsort(-1)
    return filled.gather(2,inverse[:,None].expand(-1,K,-1))

class CandidateTransitionScorer(nn.Module):
    def __init__(self,dim=64,chunk=128):
        super().__init__()
        if chunk<1:raise ValueError('Candidate chunk must be positive')
        self.query=nn.Linear(dim,dim,bias=False);self.key=nn.Linear(dim,dim,bias=False)
        self.scale=nn.Parameter(torch.zeros(2));self.chunk=chunk;self.dim=dim
    def alpha(self,v,weights,candidate_vectors):
        q=self.query(F.layer_norm(v,(self.dim,)))
        keys=self.key(F.layer_norm(candidate_vectors,(self.dim,)))
        logits=torch.einsum('bkd,cd->bkc',q,keys) if keys.ndim==2 else torch.einsum('bkd,bcd->bkc',q,keys)
        return (logits/math.sqrt(self.dim)+weights.clamp_min(1e-8).log()[...,None]).softmax(1)
    def forward(self,v,A,weights,valid,candidate_ids,candidate_vectors,transition=None,bridge=None,details=False):
        B,K,D=v.shape
        ids=candidate_ids[None].expand(B,-1) if candidate_ids.ndim==1 else candidate_ids
        mass=A.masked_fill(~valid[:,None],0)
        parts={};raw={}
        for rel,edges in [('transition',transition),('bridge',bridge)]:
            raw[rel]=sparse_candidate_read(mass,valid,edges,ids) if edges is not None else None
        residual=[];alphas=[];rel_scores={'transition':[],'bridge':[]}
        for lo in range(0,ids.shape[1],self.chunk):
            hi=min(lo+self.chunk,ids.shape[1])
            feature=candidate_vectors[lo:hi] if candidate_vectors.ndim==2 else candidate_vectors[:,lo:hi]
            alpha=self.alpha(v,weights,feature)
            scores=[]
            for rel in ['transition','bridge']:
                s=(alpha*raw[rel][...,lo:hi]).sum(1) if raw[rel] is not None else alpha.new_zeros((B,hi-lo))
                scores.append(s)
                if details:rel_scores[rel].append(s)
            delta=self.scale[0]*scores[0]+self.scale[1]*scores[1]
            reachable=(scores[0]!=0)|(scores[1]!=0)
            residual.append(delta.masked_fill(~reachable,0))
            if details:alphas.append(alpha)
        delta=torch.cat(residual,-1) if residual else v.new_zeros(B,0)
        if details:
            parts=dict(candidate_interest_weights=torch.cat(alphas,-1),
                direct_candidate_score=torch.cat(rel_scores['transition'],-1),
                bridge_candidate_score=torch.cat(rel_scores['bridge'],-1))
        return delta,parts
