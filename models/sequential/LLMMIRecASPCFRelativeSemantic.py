"""History-only relative semantic attention adaptation on the archived UserRefine A.
Known graph/attention reweighting components; no claim of established novelty.
"""
import hashlib,json,logging,math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from models.sequential.LLMMIRecASPCFRefineControls import LLMMIRecASPCFUserRefine
from models.sequential.LLMMIRecASPCFSampledSoftmax import candidate_mask
from models.sequential.llmmi_utils import check_nan_inf

def history_semantic_kernel(semantic,valid):
    # Fixed raw first512 LLM values, never the trainable ASPCF semantic branch.
    x=semantic.detach().masked_fill(~valid[...,None],0)
    good=valid & (x.norm(dim=-1)>1e-8)
    z=F.normalize(x,dim=-1,eps=1e-8)
    L=x.shape[1]
    edges=good[:,:,None] & good[:,None,:] & ~torch.eye(L,dtype=torch.bool,device=x.device)[None]
    return torch.bmm(z,z.transpose(1,2)).masked_fill(~edges,0)

def safe_history_softmax(logits,valid):
    x=logits.masked_fill(~valid[:,None,:],-torch.inf)
    x=torch.where(valid.any(-1)[:,None,None],x,torch.zeros_like(x))
    return F.softmax(x,dim=-1).masked_fill(~valid[:,None,:],0)

class RelativeSemanticAttention(nn.Module):
    def __init__(self,original,enabled=True):
        super().__init__()
        self.K,self.emb_size,self.attn_size=original.K,original.emb_size,original.attn_size
        # Preserve checkpoint parameter names, initialized values and RNG streams.
        self.query=original.query
        self.Wq,self.Wk,self.Wv=original.Wq,original.Wk,original.Wv
        self.alpha=nn.Parameter(torch.zeros(()))
        self.enabled=bool(enabled)

    def forward(self,history,lengths,semantic,semantic_valid=None,return_details=False):
        B,L,_=history.shape
        valid=torch.arange(L,device=history.device)[None]<lengths[:,None]
        Q=self.Wq(self.query).unsqueeze(0).expand(B,-1,-1)
        keys,values=self.Wk(history),self.Wv(history)
        logits=torch.bmm(Q,keys.transpose(1,2))/math.sqrt(self.attn_size)
        attention=safe_history_softmax(logits,valid)
        kernel=history_semantic_kernel(semantic,valid if semantic_valid is None else valid&semantic_valid)
        relative=attention-attention.mean(dim=1,keepdim=True)
        propagated=torch.bmm(relative,kernel.transpose(1,2))
        # A remains differentiable; only the source semantic table is detached.
        # Both controls carry the same scalar; disabled mode has exactly zero task gradient.
        correction=self.alpha*propagated*int(self.enabled)
        updated=safe_history_softmax(logits+correction,valid)
        vectors=torch.bmm(updated,values)
        if return_details:
            return vectors,updated,dict(original_attention=attention,relative_attention=relative,
                        semantic_kernel=kernel,semantic_propagation=propagated,
                        attention_correction=correction,attention_logits=logits)
        return vectors,updated

class LLMMIRecASPCFRelativeSemantic(LLMMIRecASPCFUserRefine):
    VERSION="relative-semantic-userrefine-v1"
    extra_log_args=LLMMIRecASPCFUserRefine.extra_log_args+["relative_semantic_enabled"]
    @staticmethod
    def parse_model_args(parser):
        parser=LLMMIRecASPCFUserRefine.parse_model_args(parser)
        parser.add_argument("--relative_semantic_enabled",type=int,choices=[0,1],default=1)
        parser.add_argument("--relative_semantic_base_checkpoint",type=str,default="")
        return parser

    def __init__(self,args,corpus):
        if args.semantic_rank!=512 or args.K<1:
            raise ValueError("Requires existing first512 semantic source and positive configurable K")
        super().__init__(args,corpus)
        self.relative_semantic_enabled=bool(args.relative_semantic_enabled)
        self.extractor=RelativeSemanticAttention(self.extractor,self.relative_semantic_enabled)
        self.base_checkpoint_sha256=None
        if args.relative_semantic_base_checkpoint:
            self.load_base_checkpoint(args.relative_semantic_base_checkpoint)
        logging.info("[RelativeSemantic] version=%s enabled=%s K=%d added=1 alpha_init=0 total=%d",
                     self.VERSION,self.relative_semantic_enabled,self.K,self.count_variables())

    def load_base_checkpoint(self,path):
        p=Path(path).resolve();root=Path(__file__).resolve().parents[2]
        if not p.is_relative_to(root) or not p.is_file():
            raise ValueError("Base checkpoint must be an existing project asset")
        state=torch.load(p,map_location="cpu",weights_only=True)
        expected=set(self.state_dict())-{"extractor.alpha"}
        if set(state)!=expected:
            raise ValueError("Expected exact original UserRefine A state; no partial loading allowed")
        # load_state_dict checks every tensor shape, including K.
        result=self.load_state_dict(state,strict=False)
        if result.missing_keys!=["extractor.alpha"] or result.unexpected_keys:
            raise ValueError(str(result))
        with torch.no_grad():self.extractor.alpha.zero_()
        self.base_checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest()
        logging.info("[RelativeSemantic] loaded exact base checkpoint SHA256=%s; Adam state not stored in original asset",
                     self.base_checkpoint_sha256)

    def forward(self,feed_dict,return_intermediate=False):
        history,lengths=feed_dict["history_items"],feed_dict["lengths"]
        ids=feed_dict["item_id"];B,L=history.shape;device=history.device
        # Match archived UserRefine A's original ASPCF.forward(..., True) path.
        hist_out=self.item_encoder(history,return_components=True)
        cand_out=self.item_encoder(ids,return_components=True)
        raw,candidates=hist_out["emb"],cand_out["emb"]
        valid_his=(history>0).long()
        position=(lengths[:,None]-torch.arange(self.max_his,device=device)[None,:L])*valid_his
        positioned=self.dropout(raw+self.position_emb(position))
        semantic=self.item_encoder.llm_table[history][...,:512]
        result=self.extractor(positioned,lengths,semantic,history>0,return_intermediate)
        vectors,attention=result[:2];vectors=self.dropout(vectors)
        weights=self.aggregator(raw,lengths)
        base_user=(vectors*weights[:,:,None]).sum(1)
        residual=self.refiner(base_user)
        user=base_user+residual
        out=dict(prediction=(user[:,None,:]*candidates).sum(-1))
        # Exact inherited relation ID set and CUDA RNG sampling.
        if self.training and self.lambda_relation>0:
            unique=torch.unique(torch.cat([history.reshape(-1),ids.reshape(-1)]))
            unique=unique[unique!=0]
            if unique.numel()>self.relation_sample_size:
                unique=unique[torch.randperm(unique.numel(),device=device)[:self.relation_sample_size]]
            out["_relation_ids"]=unique
        if not self._first_batch_checked:
            self._first_batch_checked=True
            for name,t in [("history",raw),("interests",vectors),("weights",weights),
                           ("candidates",candidates),("prediction",out["prediction"])]:
                check_nan_inf(t,"relative_semantic_"+name)
        if self.training:
            allowed,duplicate,clicked=candidate_mask(self.train_membership,feed_dict["user_id"],ids[:,0])
            out.update(_ss_scores=user@candidates[:,0,:].T,_ss_allowed=allowed,
                       _ss_duplicate=duplicate.sum(),_ss_clicked=clicked.sum())
        if return_intermediate:
            out.update(interest_vectors=vectors,attention_maps=attention,interest_weights=weights,
                       interest_mask=(lengths>0)[:,None].expand(-1,self.K),
                       user_vector=user,base_user_vector=base_user,user_residual=residual,
                       refinement_features=base_user,history_vectors=raw,candidate_vectors=candidates)
            out.update(result[2])
            for name in ["semantic","complement","alpha_sem","alpha_comp"]:
                out["history_"+name]=hist_out[name];out["candidate_"+name]=cand_out[name]
        return out

    def checkpoint_metadata(self):
        return dict(version=self.VERSION,K=self.K,H=self.max_his,
                    enabled=self.relative_semantic_enabled,llm_emb_path=self.llm_emb_path,
                    semantic_rank=self.semantic_rank,temperature=self.sampled_softmax_temperature)
    def save_model(self,model_path=None):
        path=model_path or self.model_path
        super().save_model(path)
        Path(str(path)+".relative_semantic.json").write_text(json.dumps(self.checkpoint_metadata(),indent=2))
    def load_model(self,model_path=None):
        path=model_path or self.model_path
        meta=json.loads(Path(str(path)+".relative_semantic.json").read_text())
        if meta!=self.checkpoint_metadata():raise ValueError("RelativeSemantic checkpoint/config mismatch")
        super().load_model(path)
