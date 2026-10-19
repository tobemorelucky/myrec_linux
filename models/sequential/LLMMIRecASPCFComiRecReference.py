"""ASPCF item space + Query/ComiRec-DR x history/max reference (not a new method).
DR source: THUDM/ComiRec a576eed8b605a531f2971136ce6ae87739d47693,
src/model.py:119-170,205 (bilinear_type=2, stop_grad=True, 3 routing rounds).
Adaptation: retain ASPCF position/dropout, original single-negative BPR/relation,
and score EVERY candidate with its own max. No positive-conditioned readout.
"""
import logging
import torch
import torch.nn as nn
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF


def squash(s):
    norm2=s.square().sum(-1,keepdim=True)
    return s*(norm2/(1+norm2)/torch.sqrt(norm2+1e-9))


class ComiRecDR(nn.Module):
    def __init__(self,K,dim,history_max=20,iterations=3,seed=42):
        super().__init__()
        if K<1 or history_max<1 or iterations<1:raise ValueError('Invalid DR shape/rounds')
        self.K=int(K);self.dim=int(dim);self.history_max=int(history_max);self.iterations=int(iterations)
        # TF random_normal_initializer defaults stddev=1. Local CPU RNG preserves base RNG.
        generator=torch.Generator(device='cpu').manual_seed(int(seed))
        self.vote_weight=nn.Parameter(torch.randn(history_max,K,dim,dim,generator=generator))

    def forward(self,history,lengths):
        B,L,D=history.shape
        if L>self.history_max or D!=self.dim:raise ValueError('DR history exceeds configured shape')
        valid=torch.arange(L,device=history.device)[None,:]<lengths[:,None]
        votes=torch.einsum('bld,lkod->bklo',history,self.vote_weight[:L])
        # Original implementation blocks early agreement gradients and uses live votes last.
        detached=votes.detach();logits=history.new_zeros(B,self.K,L)
        for turn in range(self.iterations):
            route=logits.softmax(dim=1)*valid[:,None,:].to(history.dtype)
            current=votes if turn==self.iterations-1 else detached
            v=squash((route[...,None]*current).sum(dim=2))
            if turn<self.iterations-1:logits=logits+(detached*v[:,:,None,:]).sum(-1)
        return v,route


def independent_max(interests,candidates):
    per_head=torch.bmm(interests,candidates.transpose(1,2))
    scores,heads=per_head.max(dim=1)
    return scores,per_head,heads


class LLMMIRecASPCFComiRecReference(LLMMIRecASPCF):
    extra_log_args=LLMMIRecASPCF.extra_log_args+['reference_extractor','reference_scoring','dr_iterations']

    @staticmethod
    def parse_model_args(parser):
        parser=LLMMIRecASPCF.parse_model_args(parser)
        parser.add_argument('--reference_extractor',choices=['query','dr'],default='query')
        parser.add_argument('--reference_scoring',choices=['history','max'],default='max')
        parser.add_argument('--dr_iterations',type=int,default=3)
        return parser

    def __init__(self,args,corpus):
        if args.reference_extractor=='query' and args.reference_scoring=='history':
            raise ValueError('A0 must use original LLMMIRecASPCF, not this reference')
        # Construct and initialize EXACT original base first. Shared tensors/RNG stay equal.
        super().__init__(args,corpus)
        self.reference_extractor=args.reference_extractor;self.reference_scoring=args.reference_scoring
        self.dr_iterations=int(args.dr_iterations)
        if self.reference_extractor=='dr':
            self.extractor=ComiRecDR(self.K,self.emb_size,self.max_his,self.dr_iterations,getattr(args,'random_seed',42))
        if self.reference_scoring=='max':
            del self.aggregator  # No unused trainable aggregator or Adam state/decay.
        logging.info('[ComiRecReference] extractor=%s scoring=%s effective_params=%s DR_rounds=%s; inherited ItemEncoder trainable',self.reference_extractor,self.reference_scoring,self.count_variables(),self.dr_iterations)

    def forward(self,feed_dict,return_intermediate=False):
        history=feed_dict['history_items'];lengths=feed_dict['lengths'];ids=feed_dict['item_id']
        if history.shape[1]==0:
            history=history.new_zeros(history.shape[0],1)
        B,L=history.shape;device=history.device
        if L>self.max_his:raise ValueError('H20/configured history limit exceeded')
        raw=self.item_encoder(history);candidate=self.item_encoder(ids)
        valid=(history>0).long();position=(lengths[:,None]-torch.arange(L,device=device)[None,:])*valid
        encoded=self.dropout(raw+self.position_emb(position))
        pre_dropout,attention=self.extractor(encoded,lengths)
        interests=self.dropout(pre_dropout)
        mask=(lengths>0)[:,None].expand(-1,self.K)
        out={}
        if self.reference_scoring=='max':
            prediction,head_scores,selected=independent_max(interests,candidate)
        else:
            weights=self.aggregator(raw,lengths)
            user=(interests*weights[:,:,None]).sum(1)
            prediction=(user[:,None,:]*candidate).sum(-1)
        out['prediction']=prediction
        # Same original IDs, sampler, phase conditions and random draw as frozen ASPCF.
        if self.training and self.lambda_relation>0 and self.item_encoder_mode in ('aspcf','llm_replace'):
            unique=torch.unique(torch.cat([history.reshape(-1),ids.reshape(-1)]));unique=unique[unique!=0]
            if unique.numel()>self.relation_sample_size:
                unique=unique[torch.randperm(unique.numel(),device=device)[:self.relation_sample_size]]
            out['_relation_ids']=unique
        if return_intermediate:
            out.update(interest_vectors=interests,interest_vectors_before_dropout=pre_dropout,interest_mask=mask,attention_maps=attention,history_vectors=raw,candidate_vectors=candidate)
            if self.reference_scoring=='max':out.update(interest_candidate_scores=head_scores,selected_interest=selected)
            else:out.update(interest_weights=weights,user_vector=user)
        return out
    # Inherit original loss (GeneralModel BPR + relation) and original Dataset unmodified.
