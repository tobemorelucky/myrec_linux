"""Loss-only Clean LLMMIRec ID adapter; original architecture/Dataset are inherited."""
import math
from models.sequential.LLMMIRec import LLMMIRec
from models.sequential.LLMMIRecASPCFSampledSoftmax import (
    LLMMIRecASPCFSampledSoftmax as Shared,
    build_train_membership, candidate_mask,
)

class LLMMIRecCleanSoftmax(LLMMIRec):
    extra_log_args = LLMMIRec.extra_log_args + ["sampled_softmax_temperature"]

    @staticmethod
    def parse_model_args(parser):
        parser = LLMMIRec.parse_model_args(parser)
        parser.set_defaults(item_encoder="id", lambda_relation=0.0,
                            interest_query_mode="learnable")
        parser.add_argument("--sampled_softmax_temperature", type=float, default=1.0)
        return parser

    def __init__(self, args, corpus):
        if args.item_encoder != "id" or args.lambda_relation != 0:
            raise ValueError("Clean adapter requires item_encoder=id, lambda_relation=0")
        if args.interest_query_mode != "learnable":
            raise ValueError("Clean adapter requires the original learned Query")
        if not math.isfinite(args.sampled_softmax_temperature) or args.sampled_softmax_temperature <= 0:
            raise ValueError("Softmax temperature must be positive and finite")
        super().__init__(args, corpus)
        self.sampled_softmax_temperature = args.sampled_softmax_temperature
        self.register_buffer("train_membership", build_train_membership(corpus), persistent=False)
        self._ss_stats = None
        self._ss_epoch = None

    def forward(self, feed_dict, return_intermediate=False):
        if not self.training:
            return super().forward(feed_dict, return_intermediate)
        # Exactly one original LLMMIRec forward, including its original dropout.
        out = super().forward(feed_dict, return_intermediate=True)
        allowed, duplicate, clicked = candidate_mask(
            self.train_membership, feed_dict["user_id"], feed_dict["item_id"][:, 0])
        out["_ss_scores"] = out["user_vector"] @ out["candidate_vectors"][:, 0, :].T
        out["_ss_allowed"] = allowed
        out["_ss_duplicate"] = duplicate.sum()
        out["_ss_clicked"] = clicked.sum()
        if not return_intermediate:
            out = {k: v for k, v in out.items() if k == "prediction" or k.startswith("_ss_")}
        return out

    def loss(self, out):
        if "_ss_scores" not in out:
            return super().loss(out)
        # Shared implementation calls the original shared masked_softmax_loss.
        return Shared.loss(self, out)

    _ranking_loss = Shared._ranking_loss
    _flush_stats = Shared._flush_stats
    actions_before_epoch = Shared.actions_before_epoch
    actions_after_train = Shared.actions_after_train
