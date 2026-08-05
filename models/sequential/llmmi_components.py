# -*- coding: UTF-8 -*-
"""
Core components for LLMMIRec.

- ItemEncoder: shared item embedding module (id / llm_replace / residual)
- QueryMultiInterestExtractor: K learnable queries, scaled dot-product attention
- InterestAggregator: history-only interest weight computation
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sequential.llmmi_utils import get_activation


# =========================
#  ItemEncoder
# =========================

class ItemEncoder(nn.Module):
    """Shared item embedding module.

    Three modes:
      - id:           e = id_embedding(item_id)
      - llm_replace:  e = adapter(llm_table[item_id])
      - residual:     e = id_embedding(item_id) + gamma * adapter(llm_table[item_id])

    Padding items (item_id == 0) always produce zero vectors.
    """

    def __init__(
        self,
        item_num: int,
        emb_size: int,
        mode: str = "llm_replace",
        llm_table: torch.Tensor = None,
        adapter_hidden: int = 256,
        adapter_activation: str = "gelu",
        adapter_use_ln: bool = False,
        gamma_init: float = 0.1,
        gamma_trainable: bool = False,
    ):
        super().__init__()
        self.item_num = int(item_num)
        self.emb_size = int(emb_size)
        self.mode = mode

        if mode not in ("id", "llm_replace", "residual"):
            raise ValueError(
                f"Unknown item_encoder mode: '{mode}'. "
                f"Supported: id, llm_replace, residual"
            )

        # --- ID embedding (used in 'id' and 'residual' modes) ---
        self.id_embedding = nn.Embedding(item_num, emb_size)

        # --- LLM table + adapter (used in 'llm_replace' and 'residual' modes) ---
        if mode in ("llm_replace", "residual"):
            if llm_table is None:
                raise ValueError(
                    f"item_encoder='{mode}' requires llm_table, got None"
                )
            if llm_table.shape[0] != item_num:
                raise ValueError(
                    f"llm_table shape[0]={llm_table.shape[0]} != item_num={item_num}"
                )

            self.register_buffer("llm_table", llm_table, persistent=False)
            d_llm = llm_table.size(1)

            # Build adapter
            act = get_activation(adapter_activation)
            layers = [
                nn.Linear(d_llm, adapter_hidden),
                act,
                nn.Linear(adapter_hidden, emb_size),
            ]
            if adapter_use_ln:
                layers.append(nn.LayerNorm(emb_size))
            self.adapter = nn.Sequential(*layers)

            # Gamma for residual mode
            if mode == "residual":
                if gamma_trainable:
                    # softplus^{-1}(gamma_init)
                    self.log_gamma = nn.Parameter(
                        torch.log(torch.exp(torch.tensor(float(gamma_init))) - 1.0)
                    )
                else:
                    self.register_buffer(
                        "gamma", torch.tensor(float(gamma_init))
                    )

        self._mode = mode  # stored for logging

    @property
    def llm_dim(self):
        if hasattr(self, "llm_table"):
            return self.llm_table.size(1)
        return None

    def _gamma_value(self) -> torch.Tensor:
        if hasattr(self, "log_gamma"):
            return F.softplus(self.log_gamma)
        return self.gamma

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Get item embeddings.

        Args:
            item_ids: [*] int tensor

        Returns:
            embeddings: [*, emb_size] float tensor, padding items forced to zero
        """
        if self.mode == "id":
            emb = self.id_embedding(item_ids)
        elif self.mode == "llm_replace":
            emb = self.adapter(self.llm_table[item_ids])
        elif self.mode == "residual":
            e_cf = self.id_embedding(item_ids)
            e_llm = self.adapter(self.llm_table[item_ids])
            emb = e_cf + self._gamma_value() * e_llm
        else:
            raise RuntimeError(f"Unknown mode: {self.mode}")

        # Force padding items to zero
        pad_mask = (item_ids == 0).float().unsqueeze(-1)  # [*, 1]
        emb = emb * (1.0 - pad_mask)

        return emb


# =========================
#  QueryMultiInterestExtractor
# =========================

class QueryMultiInterestExtractor(nn.Module):
    """Extract K interest vectors via learnable queries and scaled dot-product attention.

    Args:
        K: number of interest vectors
        emb_size: item embedding dimension
        attn_size: attention head dimension
    """

    def __init__(self, K: int, emb_size: int, attn_size: int):
        super().__init__()
        self.K = int(K)
        self.emb_size = int(emb_size)
        self.attn_size = int(attn_size)

        # Learnable query embeddings [K, emb_size]
        self.query = nn.Parameter(torch.empty(K, emb_size))
        nn.init.normal_(self.query, mean=0.0, std=0.01)

        # Projections
        self.Wq = nn.Linear(emb_size, attn_size)
        self.Wk = nn.Linear(emb_size, attn_size)
        self.Wv = nn.Linear(emb_size, emb_size)

    def forward(
        self,
        history_emb: torch.Tensor,
        lengths: torch.Tensor,
    ):
        """Extract interest vectors from history embeddings.

        Args:
            history_emb: [B, L, D] history item embeddings (with position encoding)
            lengths: [B] valid lengths per sample

        Returns:
            interest_vectors: [B, K, D]
            attention_maps: [B, K, L]
        """
        B, L, D = history_emb.shape
        device = history_emb.device

        # Valid mask: [B, L]
        valid_mask = (torch.arange(L, device=device)[None, :] < lengths[:, None]).float()

        # Query: [K, attn_size] -> [B, K, attn_size]
        Q = self.Wq(self.query)  # [K, attn_size]
        Q = Q.unsqueeze(0).expand(B, -1, -1)  # [B, K, attn_size]

        # Key, Value: [B, L, attn_size / emb_size]
        K_mat = self.Wk(history_emb)  # [B, L, attn_size]
        V_mat = self.Wv(history_emb)  # [B, L, D]

        # Scaled dot-product attention
        scale = math.sqrt(self.attn_size)
        scores = torch.bmm(Q, K_mat.transpose(1, 2)) / scale  # [B, K, L]

        # Mask padding positions
        attn_mask = (valid_mask == 0).unsqueeze(1)  # [B, 1, L]
        scores = scores.masked_fill(attn_mask, float("-inf"))

        # Softmax with NaN safety
        attn = F.softmax(scores, dim=-1)  # [B, K, L]
        attn = attn.masked_fill(torch.isnan(attn), 0.0)

        # Weighted sum: [B, K, D]
        interest_vectors = torch.bmm(attn, V_mat)

        return interest_vectors, attn


# =========================
#  InterestAggregator
# =========================

class InterestAggregator(nn.Module):
    """Compute interest weights using only history information.

    Combines masked mean history and last valid item:
        context = LayerNorm(mean_history + last_history)
        weights = softmax(MLP(context))
    """

    def __init__(self, emb_size: int, K: int):
        super().__init__()
        self.emb_size = int(emb_size)
        self.K = int(K)

        self.ln = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, K),
        )

    def forward(
        self,
        history_emb: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute interest weights.

        Args:
            history_emb: [B, L, D] raw history embeddings (without position encoding)
            lengths: [B] valid lengths per sample

        Returns:
            interest_weights: [B, K] softmax-normalized weights
        """
        B, L, D = history_emb.shape
        device = history_emb.device

        # Valid mask
        valid_mask = (torch.arange(L, device=device)[None, :] < lengths[:, None]).float()

        # Masked mean: sum(emb * mask) / sum(mask)
        sum_emb = (history_emb * valid_mask.unsqueeze(-1)).sum(dim=1)  # [B, D]
        count = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)       # [B, 1]
        mean_his = sum_emb / count                                         # [B, D]

        # Last valid item
        last_idx = (lengths - 1).clamp(min=0).long()  # [B]
        last_his = history_emb[torch.arange(B, device=device), last_idx]  # [B, D]

        # Context
        context = mean_his + last_his          # [B, D]
        context = self.ln(context)             # [B, D]

        # MLP → logits → softmax weights
        logits = self.mlp(context)             # [B, K]
        weights = F.softmax(logits, dim=-1)     # [B, K]

        return weights
