"""Tensor-only PoMRec-style interest backbone. No item/LLM/position tables."""
import math

import torch
from torch import nn


class PoMRecInterestBackbone(nn.Module):
    """External positioned history -> K interests and history-only weights.

    Matches PoMRec's five prompt slots, including constant-one padding slots.
    W2/W4 omit biases that cancel exactly in position-wise softmax.
    """
    def __init__(self, emb_size, K, attn_size, prompt_num, n_layers, lambda_disp):
        super().__init__()
        if min(emb_size, K, attn_size, n_layers) < 1:
            raise ValueError("dimensions and n_layers must be positive")
        if not 0 <= prompt_num <= 5:
            raise ValueError("prompt_num must be between 0 and 5")
        if not math.isfinite(lambda_disp) or lambda_disp < 0:
            raise ValueError("lambda_disp must be finite and nonnegative")
        self.emb_size, self.K, self.attn_size = emb_size, K, attn_size
        self.prompt_num, self.n_layers = prompt_num, n_layers
        self.max_prompt = 5
        self.lambda_disp = float(lambda_disp)  # fixed config, no extra parameter
        self.register_buffer("prompt_pad", torch.ones(5 - prompt_num, emb_size))
        self.prompt1 = nn.Embedding(prompt_num, emb_size)
        self.prompt2 = nn.Embedding(prompt_num, emb_size)
        self.W1 = nn.Linear(emb_size, attn_size)
        self.W2 = nn.Linear(attn_size, K, bias=False)
        self.W3 = nn.Linear(emb_size, attn_size)
        self.W4 = nn.Linear(attn_size, 1, bias=False)
        self.proj = nn.Sequential()
        for i in range(n_layers - 1):
            self.proj.add_module(f"proj_{i}", nn.Linear(emb_size, emb_size))
            self.proj.add_module(f"dropout_{i}", nn.Dropout(p=0.5))
            self.proj.add_module(f"relu_{i}", nn.ReLU())
        self.proj.add_module("proj_final", nn.Linear(emb_size, K))

    @staticmethod
    def _attention(logits, mask):
        # Each row has five valid prompts, including for empty history.
        return logits.transpose(1, 2).masked_fill(~mask[:, None, :], -torch.inf).softmax(-1)

    @staticmethod
    def _zero_safe_sqrt(variance):
        # Forward is sqrt(max(var,0)), with zero derivative at exactly zero.
        # Avoid evaluating sqrt(0)'s infinite derivative in an inactive branch.
        variance = variance.clamp_min(0)
        positive = variance > 0
        safe = torch.where(positive, variance, torch.ones_like(variance))
        return torch.where(positive, torch.sqrt(safe), torch.zeros_like(variance))

    def forward(self, history_embeddings, valid_mask, lengths, return_intermediate=False):
        if history_embeddings.ndim != 3 or history_embeddings.size(-1) != self.emb_size:
            raise ValueError("history_embeddings must have shape [B,L,emb_size]")
        B, L, _ = history_embeddings.shape
        if valid_mask.shape != (B, L) or lengths.shape != (B,):
            raise ValueError("mask/lengths shapes do not match history")
        if valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean")
        expected = torch.arange(L, device=lengths.device)[None, :] < lengths[:, None]
        if torch.any(lengths < 0) or torch.any(lengths > L) or not torch.equal(valid_mask, expected):
            raise ValueError("history must have a valid prefix and right padding")
        history = history_embeddings.masked_fill(~valid_mask[:, :, None], 0)
        mask = torch.cat([valid_mask, torch.ones(B, 5, device=valid_mask.device, dtype=torch.bool)], 1)
        p1 = torch.cat([self.prompt_pad, self.prompt1.weight], 0)
        p2 = torch.cat([self.prompt_pad, self.prompt2.weight], 0)
        H1 = torch.cat([history, p1[None].expand(B, -1, -1)], 1)
        H2 = torch.cat([history, p2[None].expand(B, -1, -1)], 1)
        attention = self._attention(self.W2(self.W1(H1).tanh()), mask)
        center = torch.bmm(attention, H1)
        std = None
        vectors = center
        if self.lambda_disp != 0.0:
            # Direct centered RMS; no cancellation-prone E[x²]-E[x]².
            variance = (attention[:, :, :, None] *
                        (H1[:, None, :, :] - center[:, :, None, :]).square()).sum(2)
            std = self._zero_safe_sqrt(variance)
            vectors = center + self.lambda_disp * std
        distribution_attention = self._attention(self.W4(self.W3(H2).tanh()), mask)
        distribution_vector = torch.bmm(distribution_attention, H2).squeeze(1)
        weights = self.proj(distribution_vector).softmax(-1)
        if not return_intermediate:
            return vectors, weights
        return {
            "interest_vectors": vectors,
            "interest_weights": weights,
            "attention_maps": attention,
            "centrality": center,
            "dispersion": std,
            "distribution_attention": distribution_attention,
            "distribution_vector": distribution_vector,
        }
