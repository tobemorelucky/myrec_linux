"""C1 plus sparse candidate-conditioned transition scoring, no interest-vector residual."""
import torch,numpy as np
from models.sequential.LLMMIRecASPCFAdaptiveSIL import LLMMIRecASPCFAdaptiveSIL as C1,array_sha,file_sha
from models.sequential.sgit_v3_components import CandidateTransitionIndex,CandidateTransitionScorer
class SGITRecV3(C1):
    @staticmethod
    def parse_model_args(parser):
        parser=C1.parse_model_args(parser)
        parser.add_argument('--v3_graph_path',required=True)
        parser.add_argument('--v3_direct',type=int,choices=[0,1],default=1)
        parser.add_argument('--v3_bridge',type=int,choices=[0,1],default=1)
        parser.add_argument('--v3_candidate_chunk',type=int,default=128)
        return parser
    def __init__(self,args,corpus):
        if args.sil_mode!='fixed':raise ValueError('V3 inherits fixed C1')
        super().__init__(args,corpus)
        self.v3_enabled=True;self.direct_enabled=bool(getattr(args,'v3_direct',1));self.bridge_enabled=bool(getattr(args,'v3_bridge',1))
        payload=torch.load(args.v3_graph_path,map_location='cpu',weights_only=True);meta=payload['metadata']
        train=corpus.data_df['train'][['user_id','item_id','position']].to_numpy(np.int64)
        if meta['train_frame_sha256']!=array_sha(train) or meta['embedding_sha256']!=file_sha(args.llm_emb_path):
            raise ValueError('Train/embedding mismatch')
        if meta['R']!=8 or meta['folds']!=3 or meta['n_items']!=self.item_num:raise ValueError('Index budget mismatch')
        self.graph_sha256=file_sha(args.v3_graph_path)
        self.candidate_graph=CandidateTransitionIndex(payload,self.direct_enabled,self.bridge_enabled)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261025)
            self.v3_scorer=CandidateTransitionScorer(self.emb_size,getattr(args,'v3_candidate_chunk',128))
    def forward(self,feed_dict,return_intermediate=False):
        feed=dict(feed_dict);h=feed['history_items'];lengths=feed['lengths']
        valid=torch.arange(h.shape[1],device=h.device)[None]<lengths[:,None]
        feed['history_items']=h.masked_fill(~valid,0)
        out=super().forward(feed,True)
        if self.v3_enabled and (self.direct_enabled or self.bridge_enabled):
            history=feed['history_items']
            if history.shape[1]==0:history=history.new_zeros((len(lengths),1))
            valid=(torch.arange(history.shape[1],device=h.device)[None]<lengths[:,None])&(history>0)
            t=self.candidate_graph.lookup(history,feed['user_id'],'transition') if self.direct_enabled else None
            b=self.candidate_graph.lookup(history,feed['user_id'],'bridge') if self.bridge_enabled else None
            args=(out['interest_vectors'],out['attention_maps'],out['interest_weights'],valid)
            delta,info=self.v3_scorer(*args,feed['item_id'],out['candidate_vectors'],t,b,return_intermediate)
            raw=out['raw_prediction']+delta
            out.update(raw_prediction=raw,prediction=self.corrected_scores(raw,feed['item_id']),candidate_graph_delta=delta,**info)
            if self.training:
                shared,_=self.v3_scorer(*args,feed['item_id'][:,0],out['candidate_vectors'][:,0],t,b,False)
                out['_ss_scores']=out['_ss_scores']+shared
            self.last_graph_encoding_cost=dict(unique_neighbor_items=0,newly_encoded_items=0,
                history_slots=history.numel(),neighbor_slots=sum(e['ids'].numel() for e in [t,b] if e is not None))
        if not return_intermediate:
            out={k:v for k,v in out.items() if k in ('prediction','_relation_ids') or k.startswith('_ss_')}
        return out
