"""Exact small coalition game and history-only, permutation-equivariant correction."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class CoalitionTeacher(nn.Module):
    def __init__(self, interests=4, temperature=.1):
        super().__init__()
        if interests != 4 or temperature != .1:
            raise ValueError("Round 1 fixed K=4 and teacher temperature=.1")
        self.temperature=temperature
        masks=torch.tensor([[bool(s & (1<<k)) for k in range(interests)]
                            for s in range(1<<interests)],dtype=torch.float32)
        normalized=masks/masks.sum(-1,keepdim=True).clamp_min(1)
        coeff=torch.zeros(1<<interests,interests)
        for k in range(interests):
            for s in range(1<<interests):
                if s & (1<<k):continue
                n=int(masks[s].sum())
                factor=math.factorial(n)*math.factorial(interests-n-1)/math.factorial(interests)
                coeff[s|(1<<k),k]+=factor
                coeff[s,k]-=factor
        self.register_buffer("coalition",normalized,persistent=False)
        self.register_buffer("coeff",coeff,persistent=False)

    @torch.no_grad()
    def forward(self, scores):
        if scores.ndim!=3 or scores.shape[1]!=4 or scores.shape[2]!=2:
            raise ValueError("CICL teacher requires original positive plus one BPR negative")
        margins=scores.detach()[:,:,0]-scores.detach()[:,:,1]
        coalition_margin=margins @ self.coalition.T
        utility=torch.sigmoid(coalition_margin)-.5
        phi=utility @ self.coeff
        target=torch.softmax(phi/self.temperature,dim=-1)
        return phi,target,utility

class ContributionPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(322,elementwise_affine=False)
        self.hidden=nn.Linear(322,64)
        self.output=nn.Linear(64,1,bias=False)
        nn.init.normal_(self.hidden.weight,0,.01)
        nn.init.normal_(self.hidden.bias,0,.01)
        nn.init.zeros_(self.output.weight)

    def forward(self, vectors, attention, history, lengths, semantic_q):
        # Detached features: contribution supervision trains only the predictor.
        vectors,attention,history=vectors.detach(),attention.detach(),history.detach()
        B,K,D=vectors.shape;L=history.shape[1]
        mask=torch.arange(L,device=lengths.device)[None,:]<lengths[:,None]
        hist_mean=(history*mask[:,:,None]).sum(1)/lengths[:,None].clamp_min(1)
        recent=history[torch.arange(B,device=lengths.device),(lengths-1).clamp_min(0)]
        profile=attention @ semantic_q.detach()
        sem_mean=(semantic_q*mask[:,:,None]).sum(1)/lengths[:,None].clamp_min(1)
        entropy=-(attention*attention.clamp_min(1e-12).log()).sum(-1)
        entropy=entropy/lengths.float().log().clamp_min(1)[:,None]
        parts=[vectors,vectors.mean(1,keepdim=True).expand(-1,K,-1),
               hist_mean[:,None].expand(-1,K,-1),recent[:,None].expand(-1,K,-1),
               profile,sem_mean[:,None].expand(-1,K,-1),
               lengths.float().log1p()[:,None,None].expand(-1,K,-1),entropy[:,:,None]]
        return torch.tanh(self.output(F.gelu(self.hidden(self.norm(torch.cat(parts,-1))))).squeeze(-1))

def corrected_weights(base, correction):
    # Algebraically normalized base*exp(correction); anchored denominator is exactly 1 at zero.
    factor=torch.exp(correction)
    denominator=1+(base*(factor-1)).sum(-1,keepdim=True)
    return base*factor/denominator
