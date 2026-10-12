"""Controlled uniform pairwise BPR on exactly the SampledSoftmax in-batch mask."""
import torch
import torch.nn.functional as F
from models.sequential.LLMMIRecASPCFSampledSoftmax import LLMMIRecASPCFSampledSoftmax


def masked_bpr_loss(scores, allowed):
    negative = allowed & ~torch.eye(scores.shape[0], device=scores.device, dtype=torch.bool)
    counts = negative.sum(1)
    valid = counts > 0
    pair_loss = F.softplus(scores - scores.diagonal()[:, None])
    row_loss = (pair_loss * negative).sum(1) / counts.clamp_min(1)
    return (row_loss * valid).sum() / valid.sum().clamp_min(1), valid


class LLMMIRecASPCFBatchBPR(LLMMIRecASPCFSampledSoftmax):
    def _ranking_loss(self, scores, allowed):
        return masked_bpr_loss(scores, allowed)
