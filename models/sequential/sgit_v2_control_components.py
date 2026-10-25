"""Versioned R/B/S factors. Immutable V2 attention, projections and graph weights."""
import torch
from models.sequential.sgit_components import CrossFitItemGraph
from models.sequential.sgit_v2_components import InterestBridgeMemory
FACTORS={
    'V2-R':dict(R=True,B=False,S=False),
    'V2-B':dict(R=False,B=True,S=True),
    'V2-NoS':dict(R=True,B=True,S=False),
    'V2-Full':dict(R=True,B=True,S=True),
    # Interface only: NOT an additional authorized training task.
    'V2-Direct':dict(R=False,B=False,S=False),
}
class DirectOnlyGraph(CrossFitItemGraph):
    def __init__(self,payload):
        if payload['metadata']['schema']!='sgit_crossfit_item_graph_v1':raise ValueError('Use original direct cache')
        small={k:v for k,v in payload.items() if k.startswith('transition_')}
        small['metadata']=payload['metadata']
        super().__init__(small)
    def lookup(self,history,users,relation):
        if relation!='transition':raise RuntimeError('Bridge lookup forbidden in direct-only control')
        return super().lookup(history,users,relation)

class ControlledBridgeMemory(InterestBridgeMemory):
    def __init__(self,dim,factors):
        # Allocate canonical Full first: shared parameter values/RNG are unchanged
        # when factor-specific unused modules are subsequently removed.
        super().__init__(dim)
        self.factors=dict(factors)
        if not factors['B']:
            self.bridge_output=None;self.bridge_match=None
        if not factors['R']:
            self.responsibility_q=None;self.responsibility_h=None
    def responsibilities(self,v,A,history,valid,enabled=True):
        if not enabled:
            # Exactly the old G2 historical mass; relation() does its own
            # eligible-source normalization. Do not normalize twice.
            return A.masked_fill(~valid[:,None],0)
        return super().responsibilities(v,A,history,valid,True)
    def forward(self,v,A,valid,collab,cv,bridge,bv,history,
                responsibility=None,direct_enabled=True,bridge_enabled=None):
        use_R=self.factors['R'] if responsibility is None else bool(responsibility)
        use_B=self.factors['B'] and (bridge_enabled is None or bool(bridge_enabled))
        calibrated=self.responsibilities(v,A,history,valid,use_R)
        mt,support,_,rt=self.relation(v,calibrated,cv,collab,valid)
        dt=self.output(mt) if direct_enabled else torch.zeros_like(v)
        if not use_B:
            # No bridge read, no bridge parameters, no bridge gradient/RNG.
            return v+dt,dict(graph_delta=dt,direct_delta=dt,bridge_delta=torch.zeros_like(v),
                collaborative_memory=mt,transition_route=rt,graph_support=support,
                calibrated_attention=calibrated)
        if bridge is None or bv is None:raise ValueError('Missing enabled bridge')
        mb,bridge_support,_,rb=self.relation(v,calibrated,bv,bridge,valid)
        source_support=(calibrated*collab['support'].amax(-1)[:,None]).sum(-1)
        has_bridge=((bridge['ids']>0)&valid[...,None]).any((-1,-2))[:,None]
        match=self.bridge_match(torch.cat([self.norm(v),self.norm(mb)],-1)).squeeze(-1)
        beta=match.sigmoid()
        if self.factors['S']:beta=beta/(1+torch.log1p(source_support))
        beta=beta*has_bridge
        db=beta[...,None]*self.bridge_output(mb)
        delta=dt+db
        return v+delta,dict(graph_delta=delta,direct_delta=dt,bridge_delta=db,
            collaborative_memory=mt,bridge_memory=mb,transition_route=rt,bridge_route=rb,
            graph_support=support,bridge_support=bridge_support,source_support=source_support,
            bridge_beta=beta,calibrated_attention=calibrated)
