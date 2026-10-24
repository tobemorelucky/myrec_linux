"""Sparse relation memories conditioned on EACH C1 interest, no new loss."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.sequential.LLMMIRecASPCFAdaptiveSIL import safe_softmax

class CrossFitItemGraph(nn.Module):
    def __init__(self,payload):
        super().__init__()
        self.metadata=payload['metadata']
        if self.metadata['schema']!='sgit_crossfit_item_graph_v1' or self.metadata['scope']!='full_crossfit_train':
            raise ValueError("Expected complete cross-fit Train graph")
        for name,value in payload.items():
            if name=='metadata':continue
            if value.is_floating_point() and not torch.isfinite(value).all():raise ValueError(name)
            self.register_buffer(name,value.detach().clone(),persistent=False)
    def lookup(self,history,users,relation):
        fold=users.remainder(self.metadata['folds'])
        result={name:getattr(self,relation+'_'+name)[fold[:,None],history] for name in
            (['ids','prob','support'] if relation=='transition' else ['ids','prob'])}
        return result
    def bytes(self):return sum(x.numel()*x.element_size() for x in self.buffers())

class InterestGraphMemory(nn.Module):
    def __init__(self,dim=64,semantic=False):
        super().__init__()
        self.norm=nn.LayerNorm(dim)
        self.query=nn.Linear(dim,dim,bias=False);self.key=nn.Linear(dim,dim,bias=False)
        self.value=nn.Linear(dim,dim,bias=False);self.output=nn.Linear(dim,dim,bias=False)
        self.fusion=nn.Linear(dim+2,2) if semantic else None
        nn.init.zeros_(self.output.weight)
    def relation(self,v,A,features,edges,valid_history):
        # [B,K,L,R] neighbor routing, then original interest-specific history mass.
        allowed=(edges['ids']>0)&valid_history[...,None]
        logits=torch.einsum('bkd,blrd->bklr',self.query(self.norm(v)),self.key(self.norm(features)))/math.sqrt(v.shape[-1])
        logits=logits+edges['prob'].clamp_min(1e-8).log()[:,None]
        route=safe_softmax(logits,allowed[:,None])
        attention=A*allowed.any(-1)[:,None]
        attention=attention/attention.sum(-1,keepdim=True).clamp_min(1e-8)
        joint=attention[...,None]*route
        read=torch.einsum('bklr,blrd->bkd',joint,self.value(features))
        support=(joint*edges.get('support',torch.zeros_like(edges['prob']))[:,None]).sum((-1,-2))
        present=allowed.any((-1,-2))[:,None].expand(-1,v.shape[1])
        return read,support,present,joint
    def forward(self,interests,A,valid,collab,collab_features,semantic=None,semantic_features=None,variant='G3',weights=None):
        v=interests
        if variant=='G1':
            v=(interests*weights[...,None]).sum(1,keepdim=True)
            A=(A*weights[...,None]).sum(1,keepdim=True)
        mc,support,hasc,routec=self.relation(v,A,collab_features,collab,valid)
        if variant=='G3':
            ms,_,hass,routes=self.relation(v,A,semantic_features,semantic,valid)
            agreement=F.cosine_similarity(mc,ms,dim=-1,eps=1e-8)
            x=torch.cat([self.norm(v),torch.log1p(support)[...,None],agreement[...,None]],-1)
            relation_weights=safe_softmax(self.fusion(x),torch.stack([hasc,hass],-1))
            memory=relation_weights[...,0,None]*mc+relation_weights[...,1,None]*ms
        else:
            ms=torch.zeros_like(mc);routes=None
            relation_weights=torch.stack([hasc.to(mc.dtype),torch.zeros_like(hasc,dtype=mc.dtype)],-1)
            memory=mc;agreement=torch.zeros_like(support)
        delta=self.output(memory)
        if variant=='G1':delta=delta.expand_as(interests)
        return interests+delta,dict(graph_delta=delta,collaborative_memory=mc,semantic_memory=ms,
            relation_weights=relation_weights,graph_support=support,relation_agreement=agreement,
            transition_route=routec,semantic_route=routes)

def cached_neighbor_vectors(encoder,ids,history_ids,history_vectors,candidate_ids,candidate_vectors):
    """Encode only missing UNIQUE neighbors, never keep embeddings across optimizer steps."""
    needed,inverse=torch.unique(ids.reshape(-1),sorted=True,return_inverse=True)
    known_ids=torch.cat([history_ids.reshape(-1),candidate_ids.reshape(-1)])
    known_vec=torch.cat([history_vectors.reshape(-1,history_vectors.shape[-1]),candidate_vectors.reshape(-1,candidate_vectors.shape[-1])])
    unique,back=torch.unique(known_ids,sorted=True,return_inverse=True)
    first=torch.full_like(unique,len(known_ids))
    first.scatter_reduce_(0,back,torch.arange(len(known_ids),device=ids.device),reduce='amin')
    where=torch.searchsorted(unique,needed);safe=where.clamp_max(len(unique)-1)
    found=(where<len(unique))&(unique[safe]==needed)
    positions=found.nonzero(as_tuple=True)[0];missing=(~found).nonzero(as_tuple=True)[0]
    values=known_vec.new_zeros((len(needed),known_vec.shape[-1]))
    values=values.index_copy(0,positions,known_vec[first[safe[positions]]])
    if len(missing):values=values.index_copy(0,missing,encoder(needed[missing]))
    return values[inverse].reshape(*ids.shape,values.shape[-1]),dict(unique_neighbor_items=len(needed),newly_encoded_items=len(missing))
