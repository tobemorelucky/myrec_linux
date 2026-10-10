"""Small semantic-state forecasting operator; no item encoder or item parameters."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class SemanticInterestTransition(nn.Module):
    def __init__(self, dim=64, states=32, rank=8, personalize=True, static=False):
        super().__init__()
        self.rank = rank
        if personalize:
            self.context = nn.Linear(2 * dim, rank)
            self.source = nn.Parameter(torch.empty(states, rank))
            self.destination = nn.Parameter(torch.empty(states, rank))
        self.decoder = nn.Sequential(nn.Linear(2 * dim + 2 * states, dim),
                                     nn.GELU(), nn.Linear(dim, dim))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0., .01)
                nn.init.normal_(layer.bias, 0., .01)
        if personalize:
            nn.init.normal_(self.source, 0., .01)
            nn.init.normal_(self.destination, 0., .01)
        if static:
            # Keep exactly the full-mode decoder initialization, without idle parameters.
            del self.context, self.source, self.destination

    def forward(self, vectors, weights, attention, raw_history, lengths,
                assignments, prior, personalize=True, markov=False, static=False):
        batch, interests, dim = vectors.shape
        width = raw_history.shape[1]
        mask = torch.arange(width, device=lengths.device)[None, :] < lengths[:, None]
        attention = attention * mask[:, None, :]
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
        current = attention @ assignments
        last_index = (lengths - 1).clamp_min(0)
        rows = torch.arange(batch, device=lengths.device)
        recent = raw_history[rows, last_index]
        if markov:
            current = assignments[rows, last_index][:, None, :].expand(-1, interests, -1)
        if not static and prior.ndim == 2:
            prior = prior[None, :, :].expand(batch, -1, -1)
        if static:
            predicted = current
            kernel = None
        elif personalize and not markov:
            condition = torch.tanh(self.context(torch.cat(
                [vectors, recent[:, None, :].expand(-1, interests, -1)], dim=-1)))
            adjustment = torch.einsum("bkr,mr,nr->bkmn",
                                      condition, self.source, self.destination) / math.sqrt(self.rank)
            kernel = torch.softmax(prior.clamp_min(1e-12).log()[:, None] + adjustment, dim=-1)
            predicted = torch.einsum("bkm,bkmn->bkn", current, kernel)
        else:
            kernel = prior[:, None].expand(-1, interests, -1, -1)
            predicted = torch.einsum("bkm,bmn->bkn", current, prior)
        next_vectors = self.decoder(torch.cat(
            [vectors, recent[:, None, :].expand(-1, interests, -1), current, predicted], dim=-1))
        user = torch.sum(weights[:, :, None] * next_vectors, dim=1)
        next_state = torch.sum(weights[:, :, None] * predicted, dim=1)
        return dict(user_vector=user, next_interest_vectors=next_vectors,
                    current_state=current, predicted_state=predicted,
                    next_state=next_state, transition_kernel=kernel)
