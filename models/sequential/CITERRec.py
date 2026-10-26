"""C1 with independent candidate-interest and fixed-interest transition experts."""
import torch,numpy as np
from models.sequential.LLMMIRecASPCFAdaptiveSIL import LLMMIRecASPCFAdaptiveSIL as C1,array_sha,file_sha
from models.sequential.citer_components import DirectReliabilityIndex,CITERScorer
class CITERRec(C1):
 @staticmethod
 def parse_model_args(parser):
  parser=C1.parse_model_args(parser)
  parser.add_argument('--v3_graph_path',required=True)
  parser.add_argument('--v3_candidate_chunk',type=int,default=128)
  return parser
 def __init__(self,args,corpus):
  assert args.sil_mode=='fixed'
  super().__init__(args,corpus)
  self.citer_enabled=True;self.interest_enabled=True;self.transition_enabled=True;self.reliability_enabled=True
  payload=torch.load(args.v3_graph_path,map_location='cpu',weights_only=True);meta=payload['metadata']
  train=corpus.data_df['train'][['user_id','item_id','position']].to_numpy(np.int64)
  assert meta['train_frame_sha256']==array_sha(train) and meta['embedding_sha256']==file_sha(args.llm_emb_path)
  assert meta['scope']=='full_crossfit_train' and meta['folds']==3 and meta['R']==8 and meta['n_items']==self.item_num
  self.candidate_graph=DirectReliabilityIndex(payload)
  self.graph_sha256=file_sha(args.v3_graph_path)
  self.prediction_scorer=CITERScorer(self.emb_size,getattr(args,'v3_candidate_chunk',128))
 def forward(self,feed_dict,return_intermediate=False):
  feed=dict(feed_dict);h=feed['history_items'];lengths=feed['lengths']
  valid=torch.arange(h.shape[1],device=h.device)[None]<lengths[:,None]
  feed['history_items']=h.masked_fill(~valid,0)
  out=super().forward(feed,True)
  self.last_graph_encoding_cost=dict(unique_neighbor_items=0,newly_encoded_items=0,history_slots=h.numel(),neighbor_slots=0)
  if self.citer_enabled and (self.interest_enabled or self.transition_enabled):
   history=feed['history_items']
   if history.shape[1]==0:history=history.new_zeros((len(lengths),1))
   valid=(torch.arange(history.shape[1],device=h.device)[None]<lengths[:,None])&(history>0)
   edges={'transition':self.candidate_graph.lookup(history,feed['user_id'])} if self.transition_enabled else {}
   arguments=(out['interest_vectors'],out['attention_maps'],out['interest_weights'],valid)
   flags=dict(interest=self.interest_enabled,transition=self.transition_enabled,reliability=self.reliability_enabled)
   delta,info=self.prediction_scorer(*arguments,feed['item_id'],out['candidate_vectors'],edges,return_intermediate,**flags)
   raw=out['raw_prediction']+delta
   out.update(raw_prediction=raw,prediction=self.corrected_scores(raw,feed['item_id']),candidate_graph_delta=delta,**info)
   if self.training:
    shared,_=self.prediction_scorer(*arguments,feed['item_id'][:,0],out['candidate_vectors'][:,0],edges,False,**flags)
    out['_ss_scores']=out['_ss_scores']+shared
   self.last_graph_encoding_cost['neighbor_slots']=sum(e['ids'].numel() for e in edges.values())
  if not return_intermediate:
   out={k:v for k,v in out.items() if k in ('prediction','_relation_ids') or k.startswith('_ss_')}
  return out
