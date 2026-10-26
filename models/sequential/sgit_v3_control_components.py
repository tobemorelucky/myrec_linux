"""Versioned six-direction scoring, with sparse joins reused unchanged from V3."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.sequential.sgit_v3_components import sparse_candidate_read

VARIANTS={
 'TargetInterest':dict(target=True,condition=True,relations=()),
 'Direct-Fixed':dict(target=False,condition=False,relations=('transition',)),
 'Direct-Target':dict(target=False,condition=True,relations=('transition',)),
 'Bridge-ST':dict(target=False,condition=True,relations=('transition','bridge')),
 'Smooth-TS':dict(target=False,condition=True,relations=('transition','smooth')),
 'DualPath':dict(target=False,condition=True,relations=('transition','bridge','smooth')),
}
class PredictionIndex(nn.Module):
 def __init__(self,payload,relations):
  super().__init__();self.metadata=payload['metadata']
  assert self.metadata['schema']=='sgit_candidate_predictions_v3'
  assert self.metadata['scope']=='full_crossfit_train'
  for rel in relations:
   for key in ['ids','value']:
    x=payload[rel+'_'+key]
    if x.is_floating_point():assert torch.isfinite(x).all()
    self.register_buffer(rel+'_'+key,x.detach().clone(),persistent=False)
 def lookup(self,history,users,relation):
  fold=users.remainder(self.metadata['folds'])
  return {k:getattr(self,relation+'_'+k)[fold[:,None],history] for k in ['ids','value']}
 def bytes(self):return sum(x.numel()*x.element_size() for x in self.buffers())

class PredictionScorer(nn.Module):
 def __init__(self,dim,variant,chunk=128):
  super().__init__()
  self.spec=VARIANTS[variant];self.dim=dim;self.chunk=chunk
  # Same Q/K tensors in every candidate-conditioned arm. No learned Q/K in B.
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261025)
   self.query=nn.Linear(dim,dim,bias=False)
   self.key=nn.Linear(dim,dim,bias=False)
  if not self.spec['condition']:
   self.query=None;self.key=None
  self.scales=nn.ParameterDict({r:nn.Parameter(torch.zeros(())) for r in self.spec['relations']})
  if self.spec['target']:self.scales['target']=nn.Parameter(torch.zeros(()))
 def alpha(self,v,w,e):
  if not self.spec['condition']:
   return w[...,None].expand(-1,-1,e.shape[-2])
  q=self.query(F.layer_norm(v,(self.dim,)))
  keys=self.key(F.layer_norm(e,(self.dim,)))
  logits=torch.einsum('bkd,cd->bkc',q,keys) if e.ndim==2 else torch.einsum('bkd,bcd->bkc',q,keys)
  return (logits/math.sqrt(self.dim)+w.clamp_min(1e-8).log()[...,None]).softmax(1)
 def forward(self,v,A,w,valid,ids,features,edges,details=False):
  B,K,D=v.shape
  candidates=ids[None].expand(B,-1) if ids.ndim==1 else ids
  mass=A.masked_fill(~valid[:,None],0)
  reads={r:sparse_candidate_read(mass,valid,edges[r],candidates) for r in self.spec['relations']}
  chunks=[];alphas=[];scores={r:[] for r in self.scales}
  for lo in range(0,candidates.shape[1],self.chunk):
   hi=min(lo+self.chunk,candidates.shape[1])
   e=features[lo:hi] if features.ndim==2 else features[:,lo:hi]
   alpha=self.alpha(v,w,e)
   delta=v.new_zeros(B,hi-lo)
   for rel in self.spec['relations']:
    value=(alpha*reads[rel][...,lo:hi]).sum(1)
    delta=delta+self.scales[rel]*value
    if details:scores[rel].append(value)
   if self.spec['target']:
    dot=torch.einsum('bkd,cd->bkc',v,e) if e.ndim==2 else torch.einsum('bkd,bcd->bkc',v,e)
    value=((alpha-w[...,None])*dot).sum(1)
    value=value.masked_fill(~valid.any(-1)[:,None]|(candidates[:,lo:hi]==0),0)
    delta=delta+self.scales['target']*value
    if details:scores['target'].append(value)
   chunks.append(delta)
   if details:alphas.append(alpha)
  result=torch.cat(chunks,-1) if chunks else v.new_zeros(B,0)
  info={}
  if details:
   info['candidate_interest_weights']=torch.cat(alphas,-1)
   info.update({rel+'_candidate_score':torch.cat(x,-1) for rel,x in scores.items()})
  return result,info
