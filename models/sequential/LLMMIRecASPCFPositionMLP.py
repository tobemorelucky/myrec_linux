"""Capacity-matched pointwise GLU control for TIA; no cross-position inputs."""
import logging
import torch
from torch import nn
import torch.nn.functional as F
from models.sequential.LLMMIRecASPCFTIA import LLMMIRecASPCFTIA

class PositionwiseAdapter(nn.Module):
    def __init__(self,dim):
        super().__init__()
        target=7*dim*dim+3*dim
        hidden=(target-dim)//(3*dim+2)
        self.hidden=hidden
        self.transition=nn.Linear(dim,2*hidden)
        self.output=nn.Linear(hidden,dim)
        nn.init.normal_(self.transition.weight,std=.01)
        nn.init.normal_(self.transition.bias,std=.01)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
    def forward(self,x,lengths):
        positions=torch.arange(x.shape[1],device=x.device)[None,:]
        valid=positions<lengths[:,None]
        current=x.masked_fill(~valid[:,:,None],0)
        update=self.output(F.glu(self.transition(current),dim=-1))
        update=update.masked_fill(~(valid & (positions>0))[:,:,None],0)
        return (current+update).masked_fill(~valid[:,:,None],0)

class LLMMIRecASPCFPositionMLP(LLMMIRecASPCFTIA):
    def __init__(self,args,corpus):
        super().__init__(args,corpus)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261012)
            adapter=PositionwiseAdapter(self.emb_size)
        # Replace; never retain unused TIA parameters.
        self.extractor.tia=adapter
        logging.info("[PositionMLP] local-only GLU hidden=%d added=%d total=%d; TIA constructor count is superseded",
                     adapter.hidden,sum(p.numel() for p in adapter.parameters()),self.count_variables())
