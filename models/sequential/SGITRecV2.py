"""SGITRec-V2 inherits unmodified fourth-chapter C1, not a copied backbone."""
import torch,numpy as np
from models.sequential.LLMMIRecASPCFAdaptiveSIL import LLMMIRecASPCFAdaptiveSIL as C1,array_sha,file_sha
from models.sequential.sgit_components import cached_neighbor_vectors
from models.sequential.sgit_v2_components import CrossFitBridgeGraph,InterestBridgeMemory

class SGITRecV2(C1):
    @staticmethod
    def parse_model_args(parser):
        parser=C1.parse_model_args(parser)
        parser.add_argument('--sgit_graph_path',required=True)
        parser.add_argument('--sgit_responsibility',type=int,default=1)
        parser.add_argument('--sgit_bridge',type=int,default=1)
        return parser
    def __init__(self,args,corpus):
        if args.sil_mode!='fixed':raise ValueError('V2 inherits fixed C1')
        super().__init__(args,corpus)
        self.sgit_enabled=True
        self.responsibility_enabled=bool(getattr(args,'sgit_responsibility',1))
        self.bridge_enabled=bool(getattr(args,'sgit_bridge',1));self.direct_enabled=True
        payload=torch.load(args.sgit_graph_path,map_location='cpu',weights_only=True)
        meta=payload['metadata'];train=corpus.data_df['train'][['user_id','item_id','position']].to_numpy(np.int64)
        if meta['train_frame_sha256']!=array_sha(train) or meta['embedding_sha256']!=file_sha(args.llm_emb_path):
            raise ValueError('Train/embedding mismatch')
        if meta['R']!=8 or meta['folds']!=3 or meta['n_items']!=self.item_num:raise ValueError('Graph budget mismatch')
        self.graph_sha256=file_sha(args.sgit_graph_path)
        self.graph_cache=CrossFitBridgeGraph(payload)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261025)
            self.graph_memory=InterestBridgeMemory(self.emb_size)
    def forward(self,feed_dict,return_intermediate=False):
        feed=dict(feed_dict);h=feed['history_items'];lengths=feed['lengths']
        valid=torch.arange(h.shape[1],device=h.device)[None]<lengths[:,None]
        feed['history_items']=h.masked_fill(~valid,0)
        out=super().forward(feed,True)
        if self.sgit_enabled:
            history=feed['history_items']
            if history.shape[1]==0:history=history.new_zeros((len(lengths),1))
            valid=(torch.arange(history.shape[1],device=h.device)[None]<lengths[:,None])&(history>0)
            history=history.masked_fill(~valid,0)
            t=self.graph_cache.lookup(history,feed['user_id'],'transition')
            b=self.graph_cache.lookup(history,feed['user_id'],'bridge')
            ids=torch.cat([t['ids'],b['ids']],-1)
            # Candidates are never read to choose routes. Encode unique graph items
            # using only history reuse, so Train dropout cannot make interest output
            # depend on whether a neighbor happens to be the current target.
            empty_ids=history[:,:0];empty_vec=out['history_vectors'][:,:0]
            cv,tcost=cached_neighbor_vectors(self.item_encoder,t['ids'],history,out['history_vectors'],empty_ids,empty_vec)
            known_ids=torch.cat([history,t['ids'].flatten(1)],-1)
            known_vectors=torch.cat([out['history_vectors'],cv.flatten(1,2)],1)
            bv,bcost=cached_neighbor_vectors(self.item_encoder,b['ids'],known_ids,known_vectors,empty_ids,empty_vec)
            cost=dict(unique_neighbor_items=len(torch.unique(ids)),
                newly_encoded_items=tcost['newly_encoded_items']+bcost['newly_encoded_items'])
            self.last_graph_encoding_cost=dict(cost,history_slots=history.numel(),neighbor_slots=ids.numel())
            base=out['interest_vectors']
            updated,info=self.graph_memory(base,out['attention_maps'],valid,t,cv,
                b,bv,out['history_vectors'],self.responsibility_enabled,
                self.direct_enabled,self.bridge_enabled)
            user=(updated*out['interest_weights'][...,None]).sum(1)
            raw=(user[:,None]*out['candidate_vectors']).sum(-1)
            out.update(c1_interest_vectors=base,interest_vectors=updated,user_vector=user,
                raw_prediction=raw,prediction=self.corrected_scores(raw,feed['item_id']),
                graph_encoding_cost=cost,**info)
            if self.training:out['_ss_scores']=user@out['candidate_vectors'][:,0].T
        if not return_intermediate:
            out={k:v for k,v in out.items() if k in ('prediction','_relation_ids') or k.startswith('_ss_')}
        return out
