"""Objective adapter only: original raw BPR or original filtered batch Softmax."""
import importlib.util
from pathlib import Path
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF as ASPCF
p=Path(__file__).resolve().parent/'softguide_snapshot.py'
spec=importlib.util.spec_from_file_location('objective_softguide_snapshot',p)
soft=importlib.util.module_from_spec(spec);spec.loader.exec_module(soft)
base=soft.base
class ObjectiveSIL(soft.SoftGuideSIL):
 def __init__(self,args,corpus):
  self.objective=args.objective
  if self.objective not in ('BPR','SS'):raise ValueError(self.objective)
  super().__init__(args,corpus)
 def forward(self,feed_dict,return_intermediate=False):
  lengths=feed_dict['lengths'];safe=dict(feed_dict)
  if feed_dict['history_items'].shape[1]==0:safe['history_items']=feed_dict['history_items'].new_zeros((len(lengths),1))
  safe['lengths']=lengths.clamp_min(1)
  out=ASPCF.forward(self,safe,True) if self.objective=='BPR' else base.SS.forward(self,safe,True)
  original=out['interest_vectors'].masked_fill((lengths==0)[:,None,None],0)
  attention=out['attention_maps'].masked_fill((lengths==0)[:,None,None],0)
  updated,info=self.recompose(safe['history_items'],out['history_vectors'],lengths,original,attention)
  user=(updated*out['interest_weights'][:,:,None]).sum(1)
  raw=(user[:,None]*out['candidate_vectors']).sum(-1)
  out.update(interest_vectors=updated,attention_maps=attention,user_vector=user,prediction=self.corrected_scores(raw,feed_dict['item_id']))
  if self.training:
   if self.objective=='SS':out['_ss_scores']=user@out['candidate_vectors'][:,0].T
   else:out['_bpr_scores']=raw
  if return_intermediate:out.update(base_interest_vectors=original,raw_prediction=raw,**info)
  else:out={k:v for k,v in out.items() if k in ('prediction','_relation_ids','_bpr_scores') or k.startswith('_ss_')}
  return out
 def loss(self,out):
  if self.objective=='SS':return super().loss(out)
  view=dict(out);view['prediction']=out['_bpr_scores']
  value=ASPCF.loss(self,view)
  if 'loss_relation' in view:out['loss_relation']=view['loss_relation']
  return value
