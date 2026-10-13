"""PoMRec structure-preserving BPR / filtered in-batch Softmax control."""
import logging,math
import torch
from models.sequential.PoMRec import PoMRec
# Reuse ONLY loss/filter utilities, never any ASPCF encoder/extractor/aggregator.
from models.sequential.LLMMIRecASPCFSampledSoftmax import build_train_membership,candidate_mask,masked_softmax_loss

class PoMRecBatchObjective(PoMRec):
    extra_log_args=PoMRec.extra_log_args+["ranking_objective","sampled_softmax_temperature"]
    @staticmethod
    def parse_model_args(parser):
        parser=PoMRec.parse_model_args(parser)
        parser.add_argument("--ranking_objective",choices=["bpr","softmax"],default="softmax")
        parser.add_argument("--sampled_softmax_temperature",type=float,default=1.)
        return parser
    def __init__(self,args,corpus):
        if int(args.history_max)!=20 or int(args.use_llmemb)!=0:
            raise ValueError("This control requires H20 and use_llmemb=0")
        super().__init__(args,corpus)
        self.ranking_objective=args.ranking_objective
        self.sampled_softmax_temperature=float(args.sampled_softmax_temperature)
        if not math.isfinite(self.sampled_softmax_temperature) or self.sampled_softmax_temperature<=0:
            raise ValueError("Invalid Softmax temperature")
        # Construct both modes identically, RNG-free and nonpersistent.
        self.register_buffer("train_membership",build_train_membership(corpus),persistent=False)
        self._batch_stats=None;self._batch_epoch=None
    def forward(self,feed_dict):
        if not self.training or self.ranking_objective=="bpr":
            return super().forward(feed_dict)
        # Exactly the active plain-PoMRec forward path, using its own modules.
        # User/item vectors are computed once and shared by service and train scores.
        self.global_step += 1
        V,distribution=self.interest_extractor(feed_dict["history_items"],feed_dict["lengths"])
        E=self.interest_extractor.get_item_emb(feed_dict["item_id"])
        w=self.proj(distribution).softmax(-1)
        u=(V*w[:,:,None]).sum(-2)
        out={"prediction":(u[:,None,:]*E).sum(-1)}
        positives=feed_dict["item_id"][:,0]
        allowed,duplicate,clicked=candidate_mask(self.train_membership,feed_dict["user_id"],positives)
        out["_batch_scores"]=u@E[:,0].T
        out["_batch_allowed"]=allowed
        out["_batch_duplicate"]=duplicate.sum();out["_batch_clicked"]=clicked.sum()
        return out
    def loss(self,out):
        if "_batch_scores" not in out:return super().loss(out)
        ranking,valid=masked_softmax_loss(out["_batch_scores"],out["_batch_allowed"],self.sampled_softmax_temperature)
        total=ranking
        # Preserve parent auxiliary semantics if present; plain mode produces none.
        if self.use_llmemb and "align_loss" in out:
            total=total+self._alpha_t()*out["align_loss"]
        n=out["_batch_allowed"].sum(1)-1;B=n.numel()
        with torch.no_grad():
            stats=torch.stack([n.new_tensor(B),valid.sum(),n.sum(),n.new_tensor(B*(B-1)),
                               out["_batch_duplicate"],out["_batch_clicked"],ranking.detach()*B]).to(torch.float64)
            if self._batch_stats is None:self._batch_stats=stats
            else:self._batch_stats+=stats
        out["loss_ranking"]=ranking.detach()
        return total
    def _flush(self):
        if self._batch_stats is not None:
            rows,valid,neg,slots,dup,clicked,loss=self._batch_stats.cpu().tolist()
            logging.info("[PoMRecBatchStats] epoch=%s valid_ratio=%.6f mean_neg=%.3f duplicate_mask=%.6f clicked_mask=%.6f ranking_loss=%.6f",
                         self._batch_epoch,valid/max(rows,1),neg/max(rows,1),dup/max(slots,1),clicked/max(slots,1),loss/max(rows,1))
            self._batch_stats=None
    def actions_before_epoch(self):
        self._flush();self._batch_epoch=self.cur_epoch
    def actions_after_train(self):self._flush()
