"""Two independent prediction experts and candidate-level observable reliability."""
import math,torch
from torch import nn
from models.sequential.sgit_v3_control_components import PredictionScorer,PredictionIndex
from models.sequential.sgit_v3_components import sparse_candidate_read

def path_statistics(valid,edges,candidates):
 """Max constituent edge distinct-user support (NOT union users), event coverage."""
 B,C=candidates.shape
 if C==0:return candidates.new_zeros(B,0),valid.new_zeros(B,0,dtype=torch.float32)
 sorted_ids,order=candidates.sort(-1);edge=edges['ids'].flatten(1).contiguous()
 at=torch.searchsorted(sorted_ids.contiguous(),edge);safe=at.clamp_max(C-1)
 found=(at<C)&(sorted_ids.gather(1,safe)==edge)&(edge>0)&valid[...,None].expand_as(edges['ids']).flatten(1)
 supports=edges['support'].flatten(1).long().masked_fill(~found,0)
 maximum=candidates.new_zeros(B,C).scatter_reduce(1,safe,supports,reduce='amax',include_self=True)
 events=torch.zeros(B,C,device=candidates.device).scatter_add(1,safe,found.float())
 duplicate_first=torch.searchsorted(sorted_ids.contiguous(),sorted_ids.contiguous());inverse=order.argsort(-1)
 maximum=maximum.gather(1,duplicate_first).gather(1,inverse)
 events=events.gather(1,duplicate_first).gather(1,inverse)
 return maximum,events/valid.sum(-1,keepdim=True).clamp_min(1)

class DirectReliabilityIndex(PredictionIndex):
 def __init__(self,payload):
  super().__init__(payload,('transition',))
  self.register_buffer('transition_support',payload['transition_support'].detach().clone(),persistent=False)
 def lookup(self,history,users,relation='transition'):
  assert relation=='transition'
  fold=users.remainder(self.metadata['folds'])
  return {k:getattr(self,'transition_'+k)[fold[:,None],history] for k in ['ids','value','support']}

class CITERScorer(PredictionScorer):
 def __init__(self,dim,chunk=128):
  super().__init__(dim,'TargetInterest',chunk)
  self.scales['transition']=nn.Parameter(torch.zeros(()))
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261026)
   self.interest_gate=nn.Linear(3,1)
   self.transition_gate=nn.Linear(3,1)
   with torch.no_grad():
    self.interest_gate.weight.fill_(1.);self.interest_gate.bias.fill_(-1.)
    self.transition_gate.weight.copy_(torch.tensor([[1.,1.,1.]]));self.transition_gate.bias.fill_(-1.)
 def forward(self,v,A,w,valid,ids,features,edges,details=False,interest=True,transition=True,reliability=True):
  B,K,D=v.shape;candidates=ids[None].expand(B,-1) if ids.ndim==1 else ids
  mass=A.masked_fill(~valid[:,None],0)
  if transition:
   per_interest=sparse_candidate_read(mass,valid,edges['transition'],candidates)
   # Delay the fixed-w reduction until each candidate chunk, exactly as legacy B.
   support,path_fraction=path_statistics(valid,edges['transition'],candidates)
  else:
   per_interest=None;support=candidates.new_zeros(candidates.shape);path_fraction=v.new_zeros(B,candidates.shape[1])
  deltas=[];logs={k:[] for k in ['delta_I','transition_score','gate_I','gate_T','confidence','support']}
  for lo in range(0,candidates.shape[1],self.chunk):
   hi=min(lo+self.chunk,candidates.shape[1]);e=features[lo:hi] if features.ndim==2 else features[:,lo:hi]
   scoreT=(w[...,None].expand(-1,-1,hi-lo)*per_interest[...,lo:hi]).sum(1) if transition else v.new_zeros(B,hi-lo)
   if interest:
    alpha=self.alpha(v,w,e)
    dot=torch.einsum('bkd,cd->bkc',v,e) if e.ndim==2 else torch.einsum('bkd,bcd->bkc',v,e)
    deltaI=((alpha-w[...,None])*dot).sum(1)
    deltaI=deltaI.masked_fill(~valid.any(-1)[:,None]|(candidates[:,lo:hi]==0),0)
    if K>1:
     confidence=(1+(alpha.clamp_min(1e-8)*alpha.clamp_min(1e-8).log()).sum(1)/math.log(K)).clamp(0,1)
     top=alpha.topk(2,dim=1).values;margin=top[:,0]-top[:,1]
     dispersion=dot.std(dim=1,unbiased=False)/(dot.square().mean(1)+1e-12).sqrt()
    else:confidence=deltaI*0;margin=deltaI*0;dispersion=deltaI*0
    gateI=self.interest_gate(torch.stack([confidence,margin,dispersion.clamp(0,1)],-1)).squeeze(-1).sigmoid() if reliability else torch.ones_like(deltaI)
   else:
    deltaI=scoreT*0;confidence=deltaI;gateI=deltaI
   sup=support[:,lo:hi]
   if transition:
    exist=(sup>0)&(candidates[:,lo:hi]>0)
    length=valid.sum(-1).float().clamp_max(20)/20
    f=torch.stack([sup.float()/(sup.float()+1),path_fraction[:,lo:hi],length[:,None].expand_as(scoreT)],-1)
    gateT=self.transition_gate(f).squeeze(-1).sigmoid()*exist if reliability else exist.float()
   else:gateT=scoreT*0
   delta=self.scales['target']*gateI*deltaI+self.scales['transition']*gateT*scoreT
   deltas.append(delta)
   if details:
    for key,z in [('delta_I',deltaI),('transition_score',scoreT),('gate_I',gateI),('gate_T',gateT),('confidence',confidence),('support',sup)]:
     logs[key].append(z)
  delta=torch.cat(deltas,-1) if deltas else v.new_zeros(B,0)
  return delta,{k:torch.cat(z,-1) for k,z in logs.items()} if details else {}
