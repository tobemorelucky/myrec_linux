"""Negative-pair-specific responsibilities, a standard soft-mixture objective adaptation."""
import math
import torch
from torch.nn import functional as F
from models.sequential.LLMMIRecASPCFSampledSoftmax import LLMMIRecASPCFSampledSoftmax

def pair_role_loss(vectors, positives, allowed, alpha=.5, temperature=1.):
    if not 0<=alpha<=1 or temperature<=0 or not math.isfinite(temperature):
        raise ValueError("Invalid responsibility parameters")
    B,K,D=vectors.shape
    match=torch.matmul(vectors,positives.T)  # [B,K,B]
    positive=match.diagonal(dim1=0,dim2=2).T  # [B,K]
    margin=positive[:,:,None]-match
    ell=F.softplus(-margin)
    uniform=ell.mean(1)
    softmin=-temperature*(torch.logsumexp(-ell/temperature,dim=1)-math.log(K))
    combined=(1-alpha)*uniform+alpha*softmin
    legal=allowed & ~torch.eye(B,device=allowed.device,dtype=torch.bool)
    count=legal.sum(1);valid=count>0
    per=(combined*legal).sum(1)/count.clamp_min(1)
    loss=(per*valid).sum()/valid.sum().clamp_min(1)
    with torch.no_grad():
        responsibility=(1-alpha)/K+alpha*F.softmax(-ell/temperature,dim=1)
    return loss,dict(responsibility=responsibility,legal=legal,valid=valid,per_head_loss=ell.detach())

class PairRoleModel(LLMMIRecASPCFSampledSoftmax):
    # No added parameters, buffers, teachers, serving logic or Dataset changes.
    def __init__(self,args,corpus,role_weight=.1,role_alpha=.5,role_temperature=1.):
        super().__init__(args,corpus)
        if not math.isfinite(role_weight) or role_weight<0:raise ValueError("role_weight")
        self.role_weight=role_weight;self.role_alpha=role_alpha;self.role_temperature=role_temperature
    def forward(self,feed_dict,return_intermediate=False):
        if not self.training or self.role_weight==0:
            return super().forward(feed_dict,return_intermediate)
        out=super().forward(feed_dict,True)
        auxiliary,detail=pair_role_loss(out["interest_vectors"],out["candidate_vectors"][:,0],
            out["_ss_allowed"],self.role_alpha,self.role_temperature)
        out["_pair_role_loss"]=auxiliary
        if return_intermediate:out["_pair_role_detail"]=detail
        if not return_intermediate:
            out={k:v for k,v in out.items() if k in ("prediction","_relation_ids","_pair_role_loss") or k.startswith("_ss_")}
        return out
    def loss(self,out):
        base=super().loss(out)
        if "_pair_role_loss" not in out:return base
        out["loss_pair_role"]=out["_pair_role_loss"].detach()
        return base+self.role_weight*out["_pair_role_loss"]

