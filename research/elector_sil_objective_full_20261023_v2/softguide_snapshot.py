"""Shared C1, exact-capacity ordinary history control, and soft semantic memory read."""
import importlib.util,math,hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
HERE=Path(__file__).resolve().parent
p=HERE/'base_snapshot.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='27e2fbea40fec6f065a02fb03d6edf5e83d614b28f120693175abbb1b269c1e6'
spec=importlib.util.spec_from_file_location('softguide_original_sil',p)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

class MemoryRead(nn.Module):
 def __init__(self,original,variant,K,alpha_initial=.1):
  super().__init__()
  if variant not in ('C0','C1','M2'):raise ValueError(variant)
  for attr in ('dim','heads','recent','use_recent','use_local'):setattr(self,attr,getattr(original,attr))
  for name,child in original.named_children():self.add_module(name,child)
  self.variant=variant;self.semantic_max=1.
  if variant=='M2':
   if not 0<alpha_initial<1:raise ValueError('alpha_initial must be in (0,1)')
   self.semantic_alpha=nn.Parameter(torch.full((K,),math.log(alpha_initial/(1-alpha_initial))))
 local_bank=base.SILRecomposer.local_bank

 def forward(self,x,interests,ids,valid,lengths,semantic=None,original_attention=None):
  B,L,D=x.shape
  if L==0:return torch.zeros_like(interests),{}
  # The control changes only the memory organization, not its parameter budget.
  local_ids=torch.zeros_like(ids) if self.variant=='C0' else ids
  local=self.local_bank(x,local_ids,valid)
  order=torch.arange(L,device=x.device)[None]
  recent_valid=valid & (order>=(lengths-self.recent).clamp_min(0)[:,None])
  masked=x.masked_fill(~recent_valid[:,:,None],0)
  recent=masked.sum(1)/recent_valid.sum(1).clamp_min(1)[:,None]
  last=x[torch.arange(B,device=x.device),(lengths-1).clamp_min(0)]
  recent=(recent+last).masked_fill((lengths==0)[:,None],0)
  query=self.read_query(self.interest_norm(interests))
  if self.use_recent:query=query+self.recent_query(self.recent_norm(recent))[:,None]
  details={}
  if self.variant=='C0':
   joint_memory=local;joint_valid=valid
  else:
   membership=F.one_hot(ids,num_classes=L).to(x.dtype)*valid[:,:,None]
   sizes=membership.sum(1)
   memory=torch.bmm(membership.transpose(1,2),local)/sizes.clamp_min(1)[:,:,None]
   recent_membership=membership*recent_valid[:,:,None]
   singleton_recent=(sizes==1)&(recent_membership.sum(1)>0)
   segment_valid=(sizes>0)&~singleton_recent
   fine_positions=lengths[:,None]-self.recent+torch.arange(self.recent,device=x.device)[None]
   fine_valid=(fine_positions>=0)&(fine_positions<lengths[:,None])
   gather=fine_positions.clamp(0,L-1)
   fine=x.gather(1,gather[:,:,None].expand(-1,-1,D)).masked_fill(~fine_valid[:,:,None],0)
   joint_memory=torch.cat([memory,fine],1);joint_valid=torch.cat([segment_valid,fine_valid],1)
   details.update(segment_sizes=sizes,segment_memory_valid=segment_valid,behavior_memory_valid=fine_valid,
                  behavior_positions=fine_positions.masked_fill(~fine_valid,-1))
  logits=torch.bmm(query,self.read_key(self.memory_norm(joint_memory)).transpose(1,2))/math.sqrt(D)
  if self.variant=='M2':
   if semantic is None or original_attention is None:raise ValueError('M2 requires history-only frozen semantics and original attention')
   z=semantic.detach().masked_fill(~valid[:,:,None],0)
   A=original_attention.masked_fill(~valid[:,None,:],0)
   A=A/A.sum(-1,keepdim=True).clamp_min(1e-8)
   profile=F.normalize(torch.bmm(A,z),dim=-1,eps=1e-6)
   sem_segment=torch.bmm(membership.transpose(1,2),z)/sizes.clamp_min(1)[:,:,None]
   sem_fine=z.gather(1,gather[:,:,None].expand(-1,-1,z.shape[-1])).masked_fill(~fine_valid[:,:,None],0)
   sem_memory=F.normalize(torch.cat([sem_segment,sem_fine],1),dim=-1,eps=1e-6)
   similarity=torch.bmm(profile,sem_memory.transpose(1,2))
   mask=joint_valid[:,None].to(similarity.dtype)
   centered=(similarity-(similarity*mask).sum(-1,keepdim=True)/mask.sum(-1,keepdim=True).clamp_min(1))*mask
   alpha=self.semantic_max*self.semantic_alpha.sigmoid()
   semantic_bias=alpha[None,:,None]*centered
   logits=logits+semantic_bias
   details.update(semantic_profile=profile,semantic_memory=sem_memory,semantic_similarity=centered,
                  semantic_bias=semantic_bias,semantic_alpha=alpha)
  attention=base.safe_softmax(logits,joint_valid[:,None])
  read=torch.bmm(attention,self.read_value(joint_memory))
  delta=self.output(read).masked_fill((lengths<2)[:,None,None],0)
  details.update(memory_attention=attention,joint_memory_valid=joint_valid)
  return delta,details

class SoftGuideSIL(base.LLMMIRecASPCFAdaptiveSIL):
 def __init__(self,args,corpus):
  self.variant=getattr(args,'sil_variant','M2')
  if self.variant not in ('B0','C0','C1','M2'):raise ValueError(self.variant)
  args.sil_mode='baseline' if self.variant=='B0' else 'fixed'
  super().__init__(args,corpus)
  if self.sil_block is not None:
   self.sil_block=MemoryRead(self.sil_block,self.variant,self.K,getattr(args,'sil_alpha_initial',.1))
  if self.variant=='M2':
   self.register_buffer('m2_unit_semantics',F.normalize(self.item_encoder.llm_table.detach(),dim=-1,eps=1e-8),persistent=False)

 def recompose(self,history,raw,lengths,interests,attention):
  if self.sil_block is None:return interests,{}
  ids,valid=base.segment_ids(raw.new_zeros(history.shape),lengths,'fixed',self.sil_threshold,
                            self.sil_min_len,self.sil_max_len,self.sil_fixed_len)
  semantic=self.m2_unit_semantics[history] if self.variant=='M2' else None
  delta,info=self.sil_block(raw,interests,ids,valid,lengths,semantic,attention)
  return interests+delta,dict(sil_delta=delta,sil_segment_ids=ids,**info)

 def forward(self,feed_dict,return_intermediate=False):
  lengths=feed_dict['lengths'];safe=dict(feed_dict)
  if feed_dict['history_items'].shape[1]==0:safe['history_items']=feed_dict['history_items'].new_zeros((len(lengths),1))
  safe['lengths']=lengths.clamp_min(1)
  out=base.SS.forward(self,safe,True)
  original=out['interest_vectors'].masked_fill((lengths==0)[:,None,None],0)
  attention=out['attention_maps'].masked_fill((lengths==0)[:,None,None],0)
  updated,info=self.recompose(safe['history_items'],out['history_vectors'],lengths,original,attention)
  user=(updated*out['interest_weights'][:,:,None]).sum(1)
  raw=(user[:,None]*out['candidate_vectors']).sum(-1)
  out.update(interest_vectors=updated,attention_maps=attention,user_vector=user,
             prediction=self.corrected_scores(raw,feed_dict['item_id']))
  if self.training:out['_ss_scores']=user@out['candidate_vectors'][:,0].T
  if return_intermediate:out.update(base_interest_vectors=original,raw_prediction=raw,**info)
  else:out={k:v for k,v in out.items() if k in ('prediction','_relation_ids') or k.startswith('_ss_')}
  return out
