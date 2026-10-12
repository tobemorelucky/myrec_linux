"""CIGR: train-only surrogate routing; unchanged ASPCF service and Softmax."""
import math
import logging
import torch
from models.sequential.LLMMIRecASPCFSampledSoftmax import LLMMIRecASPCFSampledSoftmax

def route_interest_gradient(scores, interests, weights, candidates, allowed, beta):
    # scores [B,C], V [B,K,D], w [B,K], E [C,D].
    # No extra direct candidate/aggregator gradient: ALL responsibility inputs detach.
    if beta == 0 or interests.shape[1] == 1:
        return scores, None, None
    a_hist = torch.matmul(interests, candidates.detach().T)
    with torch.no_grad():
        z = torch.log(weights.detach().unsqueeze(-1) + 1e-12)
        z = z + (a_hist.detach() - scores.detach().unsqueeze(1)) / math.sqrt(interests.shape[-1])
        pi = torch.softmax(z, dim=1)
        correction = beta * (pi - weights.detach().unsqueeze(-1))
        correction = correction * allowed.unsqueeze(1)
        q = (1-beta) * weights.detach().unsqueeze(-1) + beta * pi
    delta = (correction * (a_hist - a_hist.detach())).sum(dim=1)
    return scores + delta, pi, q

class LLMMIRecASPCFCIGR(LLMMIRecASPCFSampledSoftmax):
    extra_log_args = LLMMIRecASPCFSampledSoftmax.extra_log_args + ["cigr_beta"]
    @staticmethod
    def parse_model_args(parser):
        parser = LLMMIRecASPCFSampledSoftmax.parse_model_args(parser)
        parser.add_argument("--cigr_beta", type=float, default=.5)
        return parser
    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.cigr_beta = float(args.cigr_beta)
        if not math.isfinite(self.cigr_beta) or not 0 <= self.cigr_beta <= 1:
            raise ValueError("cigr_beta must be finite in [0,1]")
        self._cigr_stats = None
    def forward(self, feed_dict, return_intermediate=False):
        if not self.training or self.cigr_beta == 0 or self.K == 1:
            return super().forward(feed_dict, return_intermediate)
        out = super().forward(feed_dict, True)
        original = out["_ss_scores"]
        routed, pi, q = route_interest_gradient(
            original, out["interest_vectors"], out["interest_weights"],
            out["candidate_vectors"][:,0], out["_ss_allowed"], self.cigr_beta)
        out["_ss_scores"] = routed
        with torch.no_grad():
            legal = out["_ss_allowed"].unsqueeze(1)
            diff = (pi - out["interest_weights"].unsqueeze(-1)).abs()
            stats = torch.stack([legal.sum() * self.K, (diff * legal).sum(),
                                 ((pi * (pi.clamp_min(1e-12).log())).sum(1).neg() * legal[:,0]).sum(),
                                 legal[:,0].sum()]).to(torch.float64)
            if self._cigr_stats is None:self._cigr_stats=stats
            else:self._cigr_stats += stats
        if return_intermediate:
            out["cigr_original_scores"]=original
            out["cigr_pi"]=pi
            out["cigr_q"]=q
        else:
            for key in list(out):
                if key not in ("prediction","_relation_ids") and not key.startswith("_ss_"):
                    del out[key]
        return out
    def _flush_stats(self):
        super()._flush_stats()
        if self._cigr_stats is not None:
            slots,diff,entropy,candidates=self._cigr_stats.cpu().tolist()
            logging.info("[CIGRStats] epoch=%s mean_abs_pi_minus_w=%.9g mean_pi_entropy=%.9g beta=%.3f",
                         self._ss_epoch,diff/max(slots,1),entropy/max(candidates,1),self.cigr_beta)
            self._cigr_stats=None
