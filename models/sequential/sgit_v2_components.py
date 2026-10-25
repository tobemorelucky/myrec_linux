"""Interest responsibility calibration and additive trustworthy bridge memory."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.sequential.sgit_components import CrossFitItemGraph,InterestGraphMemory
from models.sequential.LLMMIRecASPCFAdaptiveSIL import safe_softmax

class CrossFitBridgeGraph(CrossFitItemGraph):
    def __init__(self,payload):
        meta=payload['metadata']
        if meta['schema']!='sgit_semantic_bridge_v2' or meta['scope']!='full_crossfit_train':
            raise ValueError('Expected complete Train-only cross-fit bridge graph')
        # Only lookup tables used in training are placed on device.
        needed={k:v for k,v in payload.items() if k.startswith(('transition_','bridge_'))}
        needed['metadata']=dict(meta,schema='sgit_crossfit_item_graph_v1')
        super().__init__(needed)
        self.metadata=meta
    def lookup(self,history,users,relation):
        if relation not in ('transition','bridge'):raise ValueError(relation)
        f=users.remainder(self.metadata['folds'])
        return {name:getattr(self,relation+'_'+name)[f[:,None],history] for name in ['ids','prob','support']}

class InterestBridgeMemory(InterestGraphMemory):
    def __init__(self,dim=64):
        super().__init__(dim,semantic=False)
        self.responsibility_q=nn.Linear(dim,dim,bias=False)
        self.responsibility_h=nn.Linear(dim,dim,bias=False)
        self.bridge_output=nn.Linear(dim,dim,bias=False)
        self.bridge_match=nn.Linear(2*dim,1)
        nn.init.zeros_(self.bridge_output.weight)
    def responsibilities(self,v,A,history,valid,enabled=True):
        if not enabled:
            # Padding is always masked, including when calibration is disabled.
            z=A*valid[:,None]
            return z/z.sum(-1,keepdim=True).clamp_min(1e-8)
        sim=torch.einsum('bkd,bld->bkl',
            self.responsibility_q(self.norm(v)),self.responsibility_h(self.norm(history)))/math.sqrt(v.shape[-1])
        return safe_softmax(A.clamp_min(1e-8).log()+sim,valid[:,None])
    def forward(self,v,A,valid,collab,cv,bridge,bv,history,
                responsibility=True,direct_enabled=True,bridge_enabled=True):
        calibrated=self.responsibilities(v,A,history,valid,responsibility)
        mt,support,_,rt=self.relation(v,calibrated,cv,collab,valid)
        mb,bridge_support,_,rb=self.relation(v,calibrated,bv,bridge,valid)
        # Source support is max edge distinct-user support, then interest-weighted.
        # Never sum overlapping edge users and call it independent-user support.
        source_support=(calibrated*collab['support'].amax(-1)[:,None]).sum(-1)
        has_bridge=((bridge['ids']>0)&valid[...,None]).any((-1,-2))[:,None]
        match=self.bridge_match(torch.cat([self.norm(v),self.norm(mb)],-1)).squeeze(-1)
        beta=match.sigmoid()/(1+torch.log1p(source_support))
        beta=beta*has_bridge
        dt=self.output(mt) if direct_enabled else torch.zeros_like(v)
        db=beta[...,None]*self.bridge_output(mb) if bridge_enabled else torch.zeros_like(v)
        delta=dt+db
        return v+delta,dict(graph_delta=delta,direct_delta=dt,bridge_delta=db,
            collaborative_memory=mt,bridge_memory=mb,transition_route=rt,bridge_route=rb,
            graph_support=support,bridge_support=bridge_support,source_support=source_support,
            bridge_beta=beta,calibrated_attention=calibrated)
