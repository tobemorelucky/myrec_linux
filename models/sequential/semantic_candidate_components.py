"""Training-only semantic responsibilities and weighted interest candidate competition."""
import math
import torch
from torch.nn import functional as F


@torch.no_grad()
def semantic_responsibility(attention, history_q, candidate_q, lengths, temperature):
    B, K, L = attention.shape
    valid = torch.arange(L, device=attention.device)[None, :] < lengths[:, None]
    weights = attention.detach() * valid[:, None, :]
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    profile = weights @ history_q.detach()
    similarity = torch.einsum("bkd,jd->bkj", F.normalize(profile, dim=-1, eps=1e-8),
                              F.normalize(candidate_q.detach(), dim=-1, eps=1e-8))
    responsibility = torch.softmax(similarity / temperature, dim=1)
    if K == 1:
        confidence = torch.zeros_like(similarity[:, 0])
    else:
        # Normalized squared deviation from uniform: 0 uniform, 1 one-hot.
        confidence = K / (K - 1) * ((responsibility - 1 / K) ** 2).sum(1)
        confidence = confidence.clamp(0, 1)
    return profile, responsibility, confidence


def interest_competition(scores, allowed, responsibility, confidence):
    # scores [B,K,B]; original diagonal targets and shared legal candidates.
    B, K, C = scores.shape
    if C != B:
        raise ValueError("In-batch candidates must match rows")
    diagonal = torch.eye(B, dtype=torch.bool, device=scores.device)
    negative = allowed & ~diagonal
    r = responsibility.detach()
    c = confidence.detach()
    negative_weight = K * r * c[:, None, :] * negative[:, None, :]
    # Avoid log(0) even in a hypothetical differentiated weight path.
    log_weight = negative_weight.clamp_min(1e-30).log().masked_fill(negative_weight == 0, -torch.inf)
    weighted_logits = scores + log_weight
    weighted_logits = torch.where(diagonal[:, None, :], scores, weighted_logits)
    positive_scores = scores.diagonal(dim1=0, dim2=2).T  # [B,K]
    head_loss = torch.logsumexp(weighted_logits, dim=-1) - positive_scores
    positive_r = r.diagonal(dim1=0, dim2=2).T
    positive_c = c.diagonal()
    active = (negative_weight.sum(-1) > 0).any(1) & (allowed.sum(1) > 1) & (positive_c > 0)
    # Fixed row denominator retains confidence attenuation; do not renormalize it away.
    row_loss = positive_c * (positive_r * head_loss).sum(1)
    loss = (row_loss * active).sum() / (allowed.sum(1) > 1).sum().clamp_min(1)
    return loss, dict(positive_r=positive_r, positive_confidence=positive_c,
                      active=active, negative_mass=negative_weight.sum(-1))
