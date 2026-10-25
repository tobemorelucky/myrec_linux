"""SGITRec v1: fresh end-to-end C1 with cross-fit typed sparse item memories."""
import torch
import numpy as np
from models.sequential.LLMMIRecASPCFAdaptiveSIL import LLMMIRecASPCFAdaptiveSIL as C1,array_sha,file_sha
from models.sequential.sgit_components import CrossFitItemGraph,InterestGraphMemory,cached_neighbor_vectors

class SGITRec(C1):
    @staticmethod
    def parse_model_args(parser):
        parser=C1.parse_model_args(parser)
        parser.add_argument('--sgit_variant',choices=['B0','G1','G2','G3'],default='G3')
        parser.add_argument('--sgit_graph_path',default='')
        return parser
    def __init__(self,args,corpus):
        if args.sil_mode!='fixed':raise ValueError("SGITRec inherits fixed-segment C1")
        super().__init__(args,corpus)
        self.sgit_variant=getattr(args,'sgit_variant','G3');self.sgit_enabled=True
        if self.sgit_variant not in ('B0','G1','G2','G3'):raise ValueError(self.sgit_variant)
        self.graph_cache=None;self.graph_memory=None
        if self.sgit_variant!='B0':
            p=args.sgit_graph_path
            payload=torch.load(p,map_location='cpu',weights_only=True);meta=payload['metadata']
            train=corpus.data_df['train'][['user_id','item_id','position']].to_numpy(np.int64)
            if meta['train_frame_sha256']!=array_sha(train) or meta['embedding_sha256']!=file_sha(args.llm_emb_path):
                raise ValueError("Graph Train/embedding mismatch")
            if meta['n_items']!=self.item_num or meta['R']!=8 or meta['folds']!=3:raise ValueError("Graph budget mismatch")
            self.graph_sha256=file_sha(p);self.graph_cache=CrossFitItemGraph(payload)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20261024)
                self.graph_memory=InterestGraphMemory(self.emb_size,semantic=self.sgit_variant=='G3')
    def forward(self,feed_dict,return_intermediate=False):
        # The inherited ASPCF position lookup assumes zero padding. Sanitize by
        # observed lengths BEFORE calling it, without touching original C1 code.
        feed_dict=dict(feed_dict)
        h=feed_dict['history_items']
        prefix=torch.arange(h.shape[1],device=h.device)[None]<feed_dict['lengths'][:,None]
        feed_dict['history_items']=h.masked_fill(~prefix,0)
        out=super().forward(feed_dict,True)
        if self.graph_memory is not None and self.sgit_enabled:
            history,lengths=feed_dict['history_items'],feed_dict['lengths']
            if history.shape[1]==0:history=history.new_zeros((len(lengths),1))
            valid=(torch.arange(history.shape[1],device=history.device)[None]<lengths[:,None])&(history>0)
            history=history.masked_fill(~valid,0)
            collab=self.graph_cache.lookup(history,feed_dict['user_id'],'transition')
            sem=self.graph_cache.lookup(history,feed_dict['user_id'],'semantic') if self.sgit_variant=='G3' else None
            ids=torch.cat([collab['ids'],sem['ids']],-1) if sem is not None else collab['ids']
            vectors,cost=cached_neighbor_vectors(self.item_encoder,ids,history,out['history_vectors'],
                feed_dict['item_id'],out['candidate_vectors'])
            self.last_graph_encoding_cost=dict(cost,history_slots=history.numel(),neighbor_slots=ids.numel())  # Counts only.
            R=collab['ids'].shape[-1];cvec=vectors[:,:,:R]
            svec=vectors[:,:,R:] if sem is not None else None
            baseline=out['interest_vectors']
            updated,info=self.graph_memory(baseline,out['attention_maps'],valid,collab,cvec,sem,svec,
                variant=self.sgit_variant,weights=out['interest_weights'])
            user=(updated*out['interest_weights'][:,:,None]).sum(1)
            raw=(user[:,None]*out['candidate_vectors']).sum(-1)
            out.update(interest_vectors=updated,c1_interest_vectors=baseline,user_vector=user,raw_prediction=raw,
                prediction=self.corrected_scores(raw,feed_dict['item_id']),graph_encoding_cost=cost,**info)
            if self.training:out['_ss_scores']=user@out['candidate_vectors'][:,0].T
        if not return_intermediate:
            out={k:v for k,v in out.items() if k in ('prediction','_relation_ids') or k.startswith('_ss_')}
        return out
