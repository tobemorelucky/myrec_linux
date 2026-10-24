"""Independent local multi-scale recomposition, standard CNN/cross-attention components."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.sequential.LLMMIRecASPCFSampledSoftmax import LLMMIRecASPCFSampledSoftmax as SS

class LocalRecomposer(nn.Module):
 def __init__(self,dim=64,mode="interest"):
  super().__init__()
  if mode not in ("local","interest"):raise ValueError(mode)
  self.dim=dim;self.mode=mode;self.dilations=(1,3)
  self.norm=nn.LayerNorm(dim)
  self.convs=nn.ModuleList([nn.Conv1d(dim,dim,3,dilation=d,groups=dim,bias=False) for d in self.dilations])
  self.point=nn.Linear(dim,dim)
  self.Wq=nn.Linear(dim,dim,bias=False);self.Wk=nn.Linear(dim,dim,bias=False)
  self.out=nn.Linear(dim,dim,bias=False)
  nn.init.zeros_(self.out.weight)
 def bank(self,x,lengths):
  B,L,D=x.shape;valid=torch.arange(L,device=x.device)[None]<lengths[:,None]
  h=self.norm(x).masked_fill(~valid[:,:,None],0).transpose(1,2)
  values=[]
  for conv,d in zip(self.convs,self.dilations):
   y=conv(F.pad(h,(2*d,0))).transpose(1,2)
   y=F.gelu(self.point(y)).masked_fill(~valid[:,:,None],0)
   values.append(y)
  return torch.stack(values,1),valid  # [B,S,L,D]; strictly left padded, no reverse positions
 def forward(self,x,V,A,lengths):
  C,valid=self.bank(x,lengths);B,S,L,D=C.shape
  if self.mode=="local":
   # Ordinary convolutional residual values, pooled by unchanged ASPCF attention.
   f=F.gelu(self.Wq(C.mean(1))+self.Wk(self.norm(x)))
   f=f.masked_fill(~valid[:,:,None],0)
   delta=torch.bmm(A,self.out(f))
   weights=None
  else:
   memory=C.reshape(B,S*L,D);mask=valid[:,None].expand(-1,S,-1).reshape(B,S*L)
   q=self.Wq(F.layer_norm(V,(D,)))
   key=self.Wk(F.layer_norm(memory,(D,)))
   logits=torch.bmm(q,key.transpose(1,2))/math.sqrt(D)
   weights=F.softmax(logits.masked_fill(~mask[:,None],-1e9),-1)*mask[:,None]
   weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
   local=torch.bmm(weights,memory)
   delta=self.out(local)
  delta=delta.masked_fill((lengths==0)[:,None,None],0)
  return delta,weights

class RecomposeModel(SS):
 def __init__(self,args,corpus,mode="interest"):
  super().__init__(args,corpus)
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261022)
   self.local_block=LocalRecomposer(self.emb_size,mode)
 def forward(self,feed_dict,return_intermediate=False):
  out=super().forward(feed_dict,True)
  original=out["interest_vectors"]
  delta,read=self.local_block(out["history_vectors"],original,out["attention_maps"],feed_dict["lengths"])
  V=original+delta
  u=(V*out["interest_weights"][:,:,None]).sum(1)
  out["interest_vectors"]=V;out["user_vector"]=u
  out["prediction"]=(u[:,None]*out["candidate_vectors"]).sum(-1)
  if self.training:out["_ss_scores"]=u@out["candidate_vectors"][:,0].T
  if return_intermediate:
   out["base_interest_vectors"]=original;out["local_delta"]=delta;out["local_read_attention"]=read
  else:
   out={k:v for k,v in out.items() if k in ("prediction","_relation_ids") or k.startswith("_ss_")}
  return out

