"""Training-only mode of GICTRec v1: immutable deterministic C1, same transfer math."""
import torch
from torch import nn
from models.sequential.GICTRec import GICTRec
from models.sequential.LLMMIRecASPCFAdaptiveSIL import LLMMIRecASPCFAdaptiveSIL
from models.sequential.gict_components import interest_profiles

class GICTRecFrozen(GICTRec):
    def __init__(self,args,corpus):
        super().__init__(args,corpus)
        if self.gict_variant not in ("G1","G2","G3"):
            raise ValueError("Frozen-branch experiment permits G1/G2/G3 only")
        for name,p in self.named_parameters():
            p.requires_grad_(name.startswith("transfer."))
        assert [n for n,p in self.named_parameters() if p.requires_grad]==["transfer.output.weight"]
        self.train(False)

    def train(self,mode=True):
        # Frozen Dropout/LayerNorm behavior is identical to the C1 evaluation anchor.
        nn.Module.train(self,False)
        self.training=mode
        self.transfer.train(mode)
        return self

    def forward(self,feed_dict,return_intermediate=False):
        # self.training retains original Filtered Softmax/relation-ID collection.
        # All C1 submodules stay in eval mode and create no autograd graph.
        with torch.no_grad():
            out=LLMMIRecASPCFAdaptiveSIL.forward(self,feed_dict,True)
            bank=self.semantic_index
            if bank is None:raise RuntimeError("Attach the verified full Train teacher bank")
            history,lengths=feed_dict["history_items"],feed_dict["lengths"]
            if history.shape[1]==0:history=history.new_zeros((len(lengths),1))
            profiles,support=interest_profiles(bank.item_probabilities,history,out["attention_maps"],lengths)
            ids,valid=bank.retrieve(history,lengths,feed_dict["user_id"],self.gict_neighbors)
        baseline=out["interest_vectors"]
        updated,info=self.transfer(baseline,profiles,support,bank.interest_vectors[ids],
            bank.interest_profiles[ids],bank.interest_support[ids],bank.user_vectors[ids],valid,self.gict_variant)
        user=(updated*out["interest_weights"][:,:,None]).sum(1)
        raw=(user[:,None]*out["candidate_vectors"]).sum(-1)
        out.update(interest_vectors=updated,c1_interest_vectors=baseline,user_vector=user,raw_prediction=raw,
            prediction=self.corrected_scores(raw,feed_dict["item_id"]),semantic_interest_profiles=profiles,
            gict_neighbor_users=bank.user_ids[ids],gict_neighbor_valid=valid,**info)
        if self.training:out["_ss_scores"]=user@out["candidate_vectors"][:,0].T
        if not return_intermediate:
            out={k:v for k,v in out.items() if k in ("prediction","_relation_ids") or k.startswith("_ss_")}
        return out
