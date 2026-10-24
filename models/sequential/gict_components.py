"""Train-only semantic interest retrieval and detached collaborative transfer.

No target/candidate argument is accepted by retrieval. Query memory is bounded by
query_block * user_block; no all-user pair matrix is constructed.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def probability(x, eps=1e-8):
    x = x.clamp_min(0)
    return x / x.sum(-1, keepdim=True).clamp_min(eps)


def js_distance(p, q):
    p, q = probability(p), probability(q)
    m = (p + q) * .5
    return .5 * ((p * (p.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(-1)
                 + (q * (q.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(-1))


def interest_profiles(item_probabilities, history, attention, lengths):
    """Original C1 extraction attention -> shared-anchor distributions/support."""
    valid = (history > 0) & (torch.arange(history.shape[1], device=history.device)[None] < lengths[:, None])
    weights = attention * valid[:, None].to(attention.dtype)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
    profiles = probability(weights @ item_probabilities[history])
    support = 1. / weights.square().sum(-1).clamp_min(1e-8)
    support = support.masked_fill((lengths == 0)[:, None], 0)
    return profiles, support


@torch.no_grad()
def semantic_anchors(llm_table, train_item_ids, anchors=32, seed=0, iterations=8,
                     max_fit_items=4096, block=1024, temperature=.1):
    """Spherical k-means fit ONLY on unique Train items, bounded sample/workspace."""
    ids = torch.unique(train_item_ids.long())
    ids = ids[ids > 0]
    if not len(ids) or anchors < 1 or temperature <= 0:
        raise ValueError("Empty Train catalog or invalid anchor configuration")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    take = torch.randperm(len(ids), generator=rng)[:max_fit_items].to(ids.device)
    fit = F.normalize(llm_table[ids[take]].float(), dim=-1)
    count = min(anchors, len(fit))
    centers = fit[torch.linspace(0, len(fit)-1, count, device=fit.device).long()].clone()
    for _ in range(iterations):
        totals = torch.zeros_like(centers)
        sizes = fit.new_zeros(count)
        for x in fit.split(block):
            assigned = (x @ centers.T).argmax(-1)
            totals.index_add_(0, assigned, x)
            sizes.index_add_(0, assigned, torch.ones_like(assigned, dtype=fit.dtype))
        centers = torch.where((sizes > 0)[:, None], F.normalize(totals, dim=-1), centers)
    rows = []
    for x in llm_table.float().split(block):
        rows.append((F.normalize(x, dim=-1) @ centers.T / temperature).softmax(-1))
    table = torch.cat(rows)
    table[0] = 0
    return centers, table


class SemanticInterestIndex(nn.Module):
    """Immutable frozen-teacher bank. All buffers are reconstructible, nonpersistent."""
    def __init__(self, payload, query_block=128, user_block=1024):
        super().__init__()
        if payload["metadata"]["schema"] != "gict_train_bank_v1":
            raise ValueError("Unsupported bank")
        self.metadata = payload["metadata"]
        self.query_block, self.user_block = query_block, user_block
        if query_block < 1 or user_block < 1:
            raise ValueError("Invalid block sizes")
        for name in ("anchors", "item_probabilities", "user_ids", "user_profiles",
                     "interest_profiles", "interest_vectors", "interest_support", "user_vectors"):
            value = payload[name].detach().clone()
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError("Nonfinite bank: " + name)
            self.register_buffer(name, value, persistent=False)
        if self.user_ids.numel() == 0:
            for name in ("user_ids","user_profiles","interest_profiles","interest_vectors","interest_support","user_vectors"):
                value = getattr(self,name)
                setattr(self,name, value.new_full((1,*value.shape[1:]), -1 if name=="user_ids" else 0))
        if self.user_ids.unique().numel() != self.user_ids.numel():
            raise ValueError("Duplicate bank user IDs")
        self.requires_grad_(False)

    def bytes(self):
        return sum(t.numel()*t.element_size() for t in self.buffers())

    def prefix_profile(self, history, lengths):
        valid = (history > 0) & (torch.arange(history.shape[1], device=history.device)[None] < lengths[:, None])
        return probability((self.item_probabilities[history] * valid[..., None]).sum(1))

    @torch.no_grad()
    def retrieve(self, history, lengths, user_ids, neighbors=8):
        if neighbors < 1:
            raise ValueError("neighbors must be positive")
        query = self.prefix_profile(history, lengths).detach()
        result, masks = [], []
        catalog = self.user_profiles.clamp_min(0).sqrt()
        N = len(self.user_ids)
        for offset in range(0, len(query), self.query_block):
            q = query[offset:offset+self.query_block].clamp_min(0).sqrt()
            users = user_ids[offset:offset+len(q)]
            best = q.new_full((len(q), neighbors), -torch.inf)
            positions = torch.zeros((len(q), neighbors), dtype=torch.long, device=q.device)
            for start in range(0, N, self.user_block):
                sims = q @ catalog[start:start+self.user_block].T
                excluded = users[:, None] == self.user_ids[start:start+self.user_block][None]
                sims = sims.masked_fill(excluded | (lengths[offset:offset+len(q)] == 0)[:, None]
                                        | (self.user_profiles[start:start+self.user_block].sum(-1) <= 0)[None], -torch.inf)
                ids = torch.arange(start, start+sims.shape[1], device=q.device)[None].expand(len(q), -1)
                both = torch.cat([best, sims], -1)
                all_ids = torch.cat([positions, ids], -1)
                best, selected = both.topk(neighbors, -1)
                positions = all_ids.gather(1, selected)
            valid = torch.isfinite(best)
            result.append(positions.masked_fill(~valid, 0))
            masks.append(valid)
        return torch.cat(result), torch.cat(masks)


class ReliableInterestTransfer(nn.Module):
    """G1 mean-user enhancement; G2 JS matching; G3 adds support reliability.

    All variants share exactly one D x D zero-initialized projection. G1 is an
    enhancement control, not a claim to reproduce SAGE's full distillation loss.
    """
    def __init__(self, dim=64, temperature=.1):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.output = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, interests, profiles, support, neighbor_vectors,
                neighbor_profiles, neighbor_support, neighbor_users, valid, variant="G3"):
        if variant not in ("G1", "G2", "G3"):
            raise ValueError(variant)
        neighbor_vectors = neighbor_vectors.detach()
        neighbor_profiles = neighbor_profiles.detach()
        neighbor_support = neighbor_support.detach()
        neighbor_users = neighbor_users.detach()
        if variant == "G1":
            weights = valid.to(interests.dtype)
            read = (neighbor_users * weights[..., None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(1)
            read = read[:, None].expand_as(interests)
            details = {}
        else:
            distance = js_distance(profiles[:, :, None, None, :],
                                   neighbor_profiles[:, None, :, :, :])
            allowed = valid[:, None, :, None] & (neighbor_support[:, None] > 0)
            allowed = allowed & (support[:, :, None, None] > 0)
            logits = (-distance/self.temperature).flatten(2)
            mask = allowed.expand_as(distance).flatten(2)
            weights = logits.masked_fill(~mask, torch.finfo(logits.dtype).min).softmax(-1) * mask
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
            weights = weights.reshape_as(distance)
            reliability = torch.ones_like(distance)
            if variant == "G3":
                reliability = ((neighbor_support[:, None] / (neighbor_support[:, None] + 2))
                               * torch.exp(-distance/self.temperature)
                               * (2 / (support[:, :, None, None] + 2)))
            read = torch.einsum("bknj,bnjd->bkd", weights*reliability, neighbor_vectors)
            details = dict(match_distance=distance, match_weights=weights, reliability=reliability)
        delta = self.output(read)
        delta = delta.masked_fill((support <= 0)[..., None], 0)
        return interests + delta, dict(transfer_delta=delta, **details)
