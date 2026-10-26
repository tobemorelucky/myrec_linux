"""Fresh C1 plus six independently controlled candidate prediction mechanisms."""
import torch,numpy as np
from models.sequential.LLMMIRecASPCFAdaptiveSIL import LLMMIRecASPCFAdaptiveSIL as C1,array_sha,file_sha
from models.sequential.sgit_v3_control_components import PredictionIndex,PredictionScorer,VARIANTS

class SGITRecV3Controlled(C1):
 @staticmethod
 def parse_model_args(parser):
  parser=C1.parse_model_args(parser)
  parser.add_argument('--v3_variant',choices=list(VARIANTS),required=True)
  parser.add_argument('--v3_graph_path',default='')
  parser.add_argument('--v3_candidate_chunk',type=int,default=128)
  return parser
 def __init__(self,args,corpus):
  assert args.sil_mode=='fixed'
  super().__init__(args,corpus)
  self.v3_enabled=True;self.variant=args.v3_variant;self.spec=VARIANTS[self.variant]
  self.candidate_graph=None;self.graph_sha256=None
  if self.spec['relations']:
   payload=torch.load(args.v3_graph_path,map_location='cpu',weights_only=True);meta=payload['metadata']
   train=corpus.data_df['train'][['user_id','item_id','position']].to_numpy(np.int64)
   assert meta['train_frame_sha256']==array_sha(train) and meta['embedding_sha256']==file_sha(args.llm_emb_path)
   assert meta['folds']==3 and meta['R']==8 and meta['n_items']==self.item_num
   self.candidate_graph=PredictionIndex(payload,self.spec['relations'])
   self.graph_sha256=file_sha(args.v3_graph_path)
  self.prediction_scorer=PredictionScorer(self.emb_size,self.variant,getattr(args,'v3_candidate_chunk',128))
 def forward(self,feed_dict,return_intermediate=False):
  feed=dict(feed_dict);h=feed['history_items'];lengths=feed['lengths']
  valid=torch.arange(h.shape[1],device=h.device)[None]<lengths[:,None]
  feed['history_items']=h.masked_fill(~valid,0)
  out=super().forward(feed,True)
  self.last_graph_encoding_cost=dict(unique_neighbor_items=0,newly_encoded_items=0,history_slots=h.numel(),neighbor_slots=0)
  if self.v3_enabled:
   history=feed['history_items']
   if history.shape[1]==0:history=history.new_zeros((len(lengths),1))
   valid=(torch.arange(history.shape[1],device=h.device)[None]<lengths[:,None])&(history>0)
   edges={r:self.candidate_graph.lookup(history,feed['user_id'],r) for r in self.spec['relations']}
   arguments=(out['interest_vectors'],out['attention_maps'],out['interest_weights'],valid)
   delta,info=self.prediction_scorer(*arguments,feed['item_id'],out['candidate_vectors'],edges,return_intermediate)
   raw=out['raw_prediction']+delta
   out.update(raw_prediction=raw,prediction=self.corrected_scores(raw,feed['item_id']),candidate_graph_delta=delta,**info)
   if self.training:
    # EXACT SAME scorer, ALL B positives, preserving repeated slots and inherited mask.
    shared,_=self.prediction_scorer(*arguments,feed['item_id'][:,0],out['candidate_vectors'][:,0],edges,False)
    out['_ss_scores']=out['_ss_scores']+shared
   self.last_graph_encoding_cost['neighbor_slots']=sum(e['ids'].numel() for e in edges.values())
  if not return_intermediate:
   out={k:v for k,v in out.items() if k in ('prediction','_relation_ids') or k.startswith('_ss_')}
  return out
