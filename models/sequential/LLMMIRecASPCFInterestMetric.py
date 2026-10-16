"""C/I diagonal metric attribution on frozen-architecture ASPCF; no novelty claim."""
import torch
from torch import nn
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF


class LLMMIRecASPCFInterestMetric(LLMMIRecASPCF):
    extra_log_args = LLMMIRecASPCF.extra_log_args + ["metric_mode"]

    @staticmethod
    def parse_model_args(parser):
        parser = LLMMIRecASPCF.parse_model_args(parser)
        parser.add_argument("--metric_mode", choices=["C", "I"], default="I")
        return parser

    def __init__(self, args, corpus):
        if args.item_encoder != "aspcf" or args.num_neg != 1 or args.history_max != 20:
            raise ValueError("Requires original ASPCF, single-negative BPR and H20")
        if args.emb_size != 64 or args.K not in (1, 4):
            raise ValueError("Registered D64, K1/K4 only")
        super().__init__(args, corpus)
        self.metric_mode = args.metric_mode
        # No additional RNG draw or multiplicative small-initialization factors.
        self.a = nn.Parameter(torch.zeros(self.K, self.emb_size))

    @staticmethod
    def correction(V, w, u, a, mode):
        """Algebraically C: abar*u; I: sum_k w_k*a_k*V_k."""
        weighted_a = w.unsqueeze(-1) * a.unsqueeze(0)
        if mode == "C":
            return weighted_a.sum(dim=1) * u
        if mode == "I":
            return (weighted_a * V).sum(dim=1)
        raise ValueError("Unknown metric mode")

    def forward(self, feed_dict, return_intermediate=False):
        # Reuse the original history/candidate encodings and all original loss IDs.
        # return_components is a view of existing computations, not another encoding.
        out = super().forward(feed_dict, return_intermediate=True)
        V, w, u = out["interest_vectors"], out["interest_weights"], out["user_vector"]
        delta = self.correction(V, w, u, self.a, self.metric_mode)
        user = u + delta
        prediction = (user[:, None, :] * out["candidate_vectors"]).sum(-1)
        if return_intermediate:
            out["base_user_vector"] = u
            out["metric_correction"] = delta
            out["user_vector"] = user
            out["prediction"] = prediction
            return out
        result = {"prediction": prediction}
        if "_relation_ids" in out:
            result["_relation_ids"] = out["_relation_ids"]
        return result
