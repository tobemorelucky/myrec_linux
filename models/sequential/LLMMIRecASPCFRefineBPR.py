"""BPR-only attribution adapters. Original ASPCF and Softmax classes are unchanged."""
import torch
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF
from models.sequential.LLMMIRecASPCFDualViewRefine import (
    DualViewInterestRefiner, LLMMIRecASPCFDualViewRefine)
from models.sequential.LLMMIRecASPCFRefineControls import UserVectorRefiner


class LLMMIRecASPCFRefineBPR(LLMMIRecASPCF):
    extra_log_args = LLMMIRecASPCF.extra_log_args + ["refine_mode"]

    @staticmethod
    def parse_model_args(parser):
        parser = LLMMIRecASPCF.parse_model_args(parser)
        parser.add_argument("--refine_mode", choices=["user", "full"], required=True)
        return parser

    def __init__(self, args, corpus):
        if (int(args.history_max) != 20 or args.item_encoder != "aspcf"
                or int(args.emb_size) != 64 or int(args.semantic_dim) != 32
                or int(args.complement_dim) != 32 or int(args.num_neg) != 1):
            raise ValueError("Requires original ASPCF H20 D64/S32/C32 and single-negative BPR")
        super().__init__(args, corpus)
        self.refine_mode = args.refine_mode
        # Same historical network values, but restore BOTH CPU and CUDA RNG:
        # torch.manual_seed in the old CPU-only fork also changed CUDA RNG.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(20261012)
            self.refiner = (UserVectorRefiner() if self.refine_mode == "user"
                            else DualViewInterestRefiner())

    def forward(self, feed_dict, return_intermediate=False):
        # One baseline forward: exact same history/candidate encoding,
        # positions, attention, Value, two dropout draws and relation sample.
        out = LLMMIRecASPCF.forward(self, feed_dict, return_intermediate=True)
        base_user = out["user_vector"]
        if self.refine_mode == "user":
            residual = self.refiner(base_user)
            user = base_user + residual
            if return_intermediate:
                out.update(user_residual=residual, refinement_features=base_user)
        else:
            # Reuse the precise historical Full computation: same attention
            # pools RAW, ungated S/C, same aggregator LN(mean+last).
            user, refined, residual, S, C, context, features = (
                LLMMIRecASPCFDualViewRefine._refine(
                    self, out, feed_dict["history_items"], feed_dict["lengths"]))
            if return_intermediate:
                out.update(refined_interest_vectors=refined,
                           interest_residual=residual, interest_semantic=S,
                           interest_complement=C, refinement_context=context,
                           refinement_features=features)
        out["user_vector"] = user
        out["prediction"] = (user[:, None, :] * out["candidate_vectors"]).sum(-1)
        if return_intermediate:
            out["base_user_vector"] = base_user
        else:
            out = {k: v for k, v in out.items()
                   if k in ("prediction", "_relation_ids")}
        return out
    # loss, Dataset, optimizer groups and all sampling inherited DIRECTLY from
    # original ASPCF. No SampledSoftmax mixin, buffers, BxB scores or temperature.

