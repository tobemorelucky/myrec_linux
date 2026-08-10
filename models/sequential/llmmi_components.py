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

    Four modes:
      - id:           e = id_embedding(item_id)
      - llm_replace:  e = adapter(llm_table[item_id])
      - residual:     e = id_embedding(item_id) + gamma * adapter(llm_table[item_id])
      - aspcf:        e = concat(sqrt(α_s) * s, sqrt(α_c) * c)
                        where s=semantic_branch(z_high), c=complement_branch(z_low, id),
                        [α_s,α_c]=softmax(gate([s;c]))

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
        # ── ASPCF params ──
        semantic_rank: int = 512,
        semantic_dim: int = 32,
        semantic_hidden: int = 128,
        complement_dim: int = 32,
        tail_hidden: int = 64,
        complement_hidden: int = 64,
        gate_hidden: int = 64,
        aspcf_gate_mode: str = "basic",
    ):
        super().__init__()
        self.item_num = int(item_num)
        self.emb_size = int(emb_size)
        self.mode = mode

        if mode not in ("id", "llm_replace", "residual", "aspcf"):
            raise ValueError(
                f"Unknown item_encoder mode: '{mode}'. "
                f"Supported: id, llm_replace, residual, aspcf"
            )

        # --- ID embedding (used in 'id' and 'residual' modes) ---
        if mode in ("id", "residual"):
            self.id_embedding = nn.Embedding(item_num, emb_size)

        # --- LLM-based modes (llm_replace, residual, aspcf) ---
        if mode in ("llm_replace", "residual", "aspcf"):
            if llm_table is None:
                raise ValueError(f"item_encoder='{mode}' requires llm_table, got None")
            if llm_table.shape[0] != item_num:
                raise ValueError(
                    f"llm_table shape[0]={llm_table.shape[0]} != item_num={item_num}"
                )

            self.register_buffer("llm_table", llm_table, persistent=False)
            d_llm = llm_table.size(1)

        # --- llm_replace / residual adapter ---
        if mode in ("llm_replace", "residual"):
            act = get_activation(adapter_activation)
            layers = [
                nn.Linear(d_llm, adapter_hidden),
                act,
                nn.Linear(adapter_hidden, emb_size),
            ]
            if adapter_use_ln:
                layers.append(nn.LayerNorm(emb_size))
            self.adapter = nn.Sequential(*layers)

            if mode == "residual":
                if gamma_trainable:
                    self.log_gamma = nn.Parameter(
                        torch.log(torch.exp(torch.tensor(float(gamma_init))) - 1.0)
                    )
                else:
                    self.register_buffer("gamma", torch.tensor(float(gamma_init)))

        # --- ASPCF ---
        if mode == "aspcf":
            if semantic_dim + complement_dim != emb_size:
                raise ValueError(
                    f"semantic_dim({semantic_dim}) + complement_dim({complement_dim}) "
                    f"!= emb_size({emb_size})"
                )
            self.semantic_rank = int(semantic_rank)
            self.semantic_dim = int(semantic_dim)
            self.complement_dim = int(complement_dim)

            # Semantic branch: z_high → s
            self.semantic_branch = nn.Sequential(
                nn.Linear(semantic_rank, semantic_hidden),
                nn.GELU(),
                nn.Linear(semantic_hidden, semantic_dim),
            )

            # Complement: z_low processing
            self.complement_tail = nn.Sequential(
                nn.Linear(d_llm - semantic_rank, tail_hidden),
                nn.GELU(),
            )

            # Complement: trainable ID embedding
            self.complement_id_emb = nn.Embedding(item_num, emb_size)

            # Complement: fusion MLP
            self.complement_mlp = nn.Sequential(
                nn.Linear(emb_size + tail_hidden, complement_hidden),
                nn.GELU(),
                nn.Linear(complement_hidden, complement_dim),
            )

            # Gate: basic=[s;c], conflict=[s;c;|s-c|;s*c]
            self.aspcf_gate_mode = aspcf_gate_mode
            if aspcf_gate_mode == "basic":
                gate_in_dim = semantic_dim + complement_dim  # 64
            elif aspcf_gate_mode == "conflict":
                gate_in_dim = (semantic_dim + complement_dim) * 2  # 128
            else:
                raise ValueError(f"Unknown aspcf_gate_mode: {aspcf_gate_mode}")
            self.gate = nn.Sequential(
                nn.Linear(gate_in_dim, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, 2),
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

    def forward(self, item_ids: torch.Tensor, return_components: bool = False):
        """Get item embeddings.

        Args:
            item_ids: [*] int tensor
            return_components: if True and mode='aspcf', also return
                (emb, s, c, alpha_s, alpha_c) dict

        Returns:
            If return_components=False: embeddings [*, emb_size]
            If return_components=True (aspcf only):
                {'emb': [*,emb_size], 'semantic': [*,semantic_dim],
                 'complement': [*,complement_dim],
                 'alpha_sem': [*], 'alpha_comp': [*]}
        """
        if self.mode == "id":
            emb = self.id_embedding(item_ids)
        elif self.mode == "llm_replace":
            emb = self.adapter(self.llm_table[item_ids])
        elif self.mode == "residual":
            e_cf = self.id_embedding(item_ids)
            e_llm = self.adapter(self.llm_table[item_ids])
            emb = e_cf + self._gamma_value() * e_llm
        elif self.mode == "aspcf":
            return self._forward_aspcf(item_ids, return_components=return_components)
        else:
            raise RuntimeError(f"Unknown mode: {self.mode}")

        # Force padding items to zero
        pad_mask = (item_ids == 0).float().unsqueeze(-1)  # [*, 1]
        emb = emb * (1.0 - pad_mask)

        return emb

    def _forward_aspcf(self, item_ids: torch.Tensor, return_components: bool = False):
        """ASPCF forward pass."""
        z = self.llm_table[item_ids]          # [* , d_llm]
        z_high = z[..., :self.semantic_rank]   # [* , semantic_rank]
        z_low = z[..., self.semantic_rank:]    # [* , d_llm-semantic_rank]

        # Semantic branch
        s = self.semantic_branch(z_high)       # [* , semantic_dim]

        # Complement branch
        id_emb = self.complement_id_emb(item_ids)           # [* , emb_size]
        low_feat = self.complement_tail(z_low)              # [* , tail_hidden]
        comp_input = torch.cat([id_emb, low_feat], dim=-1)  # [* , emb_size+tail_hidden]
        c = self.complement_mlp(comp_input)                 # [* , complement_dim]

        # Gate
        if self.aspcf_gate_mode == "basic":
            gate_input = torch.cat([s, c], dim=-1)                     # [* , 64]
        else:  # conflict
            gate_input = torch.cat([s, c, torch.abs(s - c), s * c], dim=-1)  # [* , 128]
        gate_weights = F.softmax(self.gate(gate_input), dim=-1)  # [* , 2]
        alpha_sem = gate_weights[..., 0]                      # [*]
        alpha_comp = gate_weights[..., 1]                     # [*]

        # Final embedding
        eps = 1e-8
        e = torch.cat([
            torch.sqrt(alpha_sem.unsqueeze(-1) + eps) * s,
            torch.sqrt(alpha_comp.unsqueeze(-1) + eps) * c,
        ], dim=-1)  # [* , emb_size]

        # Force padding items to zero
        pad_mask = (item_ids == 0).float().unsqueeze(-1)
        e = e * (1.0 - pad_mask)

        if return_components:
            return {
                "emb": e,
                "semantic": s * (1.0 - pad_mask),
                "complement": c * (1.0 - pad_mask),
                "alpha_sem": alpha_sem * (1.0 - pad_mask.squeeze(-1)),
                "alpha_comp": alpha_comp * (1.0 - pad_mask.squeeze(-1)),
            }
        return e


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
        external_query: torch.Tensor = None,
        attention_prior: torch.Tensor = None,
        prior_strength: float = 0.0,
    ):
        """Extract interest vectors from history embeddings.

        Args:
            history_emb: [B, L, D] history item embeddings (with position encoding)
            lengths: [B] valid lengths per sample
            external_query: [B, K, D] optional external query seeds.
            attention_prior: [B, K, L] optional prior for attention logits.
            prior_strength: weight for attention_prior in log space.

        Returns:
            interest_vectors: [B, K, D]
            attention_maps: [B, K, L]
            (when prior is active, also returns logits_before_prior)
        """
        B, L, D = history_emb.shape
        device = history_emb.device

        # Valid mask: [B, L]
        valid_mask = (torch.arange(L, device=device)[None, :] < lengths[:, None]).float()

        # Query: use external if provided, else learned
        if external_query is not None:
            Q = self.Wq(external_query)  # [B, K, attn_size]
        else:
            Q = self.Wq(self.query)  # [K, attn_size]
            Q = Q.unsqueeze(0).expand(B, -1, -1)  # [B, K, attn_size]

        # Key, Value: [B, L, attn_size / emb_size]
        K_mat = self.Wk(history_emb)  # [B, L, attn_size]
        V_mat = self.Wv(history_emb)  # [B, L, D]

        # Scaled dot-product attention
        scale = math.sqrt(self.attn_size)
        scores = torch.bmm(Q, K_mat.transpose(1, 2)) / scale  # [B, K, L]
        logits_before_prior = scores.clone()  # stash for diagnostics

        # Optional routing prior
        if attention_prior is not None and prior_strength > 0:
            scores = scores + prior_strength * torch.log(attention_prior + 1e-8)

        # Mask padding positions
        attn_mask = (valid_mask == 0).unsqueeze(1)  # [B, 1, L]
        scores = scores.masked_fill(attn_mask, float("-inf"))

        # Softmax with NaN safety
        attn = F.softmax(scores, dim=-1)  # [B, K, L]
        attn = attn.masked_fill(torch.isnan(attn), 0.0)

        # Weighted sum: [B, K, D]
        interest_vectors = torch.bmm(attn, V_mat)

        if attention_prior is not None and prior_strength > 0:
            return interest_vectors, attn, logits_before_prior
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


# =========================
#  DualViewInterestExtractor
# =========================

class DualViewInterestExtractor(nn.Module):
    """Dual-view interest extraction: semantic + collaborative attention
    fused via per-interest adaptive routing gate.

    Args:
        K: number of interest vectors
        semantic_dim: semantic query/history dim
        complement_dim: collaborative query/history dim
        attn_dim: attention projection dim
        gate_hidden: routing gate hidden dim
        emb_size: full item embedding dim (for value projection)
        rho_mode: "learned" (MLP gate) or "fixed" (constant rho)
        rho_value: fixed rho value when rho_mode="fixed"
    """

    def __init__(self, K: int, semantic_dim: int = 32, complement_dim: int = 32,
                 attn_dim: int = 32, gate_hidden: int = 32, emb_size: int = 64,
                 rho_mode: str = "learned", rho_value: float = 0.5):
        super().__init__()
        self.K = int(K)
        self.attn_dim = int(attn_dim)
        self.emb_size = int(emb_size)
        self.rho_mode = rho_mode
        self.rho_value = float(rho_value)

        # Semantic attention
        self.Wq_sem = nn.Linear(semantic_dim, attn_dim)
        self.Wk_sem = nn.Linear(semantic_dim, attn_dim)

        # Collaborative attention
        self.Wq_comp = nn.Linear(complement_dim, attn_dim)
        self.Wk_comp = nn.Linear(complement_dim, attn_dim)

        # Value projection (from full fused history embedding)
        self.Wv = nn.Linear(emb_size, emb_size)

        # Per-interest routing gate (only for learned mode)
        if rho_mode == "learned":
            self.routing_gate = nn.Sequential(
                nn.Linear(semantic_dim + complement_dim, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, 1),
                nn.Sigmoid(),
            )
        else:
            self.routing_gate = None

    def forward(
        self,
        history_emb: torch.Tensor,          # [B, L, D] full fused embedding (+pos)
        lengths: torch.Tensor,              # [B]
        history_semantic: torch.Tensor,     # [B, L, 32]
        history_complement: torch.Tensor,   # [B, L, 32]
        semantic_query: torch.Tensor,       # [B, K, 32]
        collaborative_query: torch.Tensor,  # [B, K, 32]
    ):
        B, L, D = history_emb.shape
        device = history_emb.device

        valid_mask = (torch.arange(L, device=device)[None, :] < lengths[:, None]).float()
        attn_mask = (valid_mask == 0).unsqueeze(1)  # [B, 1, L]
        scale = math.sqrt(self.attn_dim)

        # Semantic attention logits
        Q_sem = self.Wq_sem(semantic_query)         # [B, K, attn_dim]
        K_sem = self.Wk_sem(history_semantic)        # [B, L, attn_dim]
        sem_logits = torch.bmm(Q_sem, K_sem.transpose(1, 2)) / scale  # [B, K, L]

        # Collaborative attention logits
        Q_comp = self.Wq_comp(collaborative_query)
        K_comp = self.Wk_comp(history_complement)
        comp_logits = torch.bmm(Q_comp, K_comp.transpose(1, 2)) / scale  # [B, K, L]

        # Per-interest routing gate: rho ∈ [0,1]
        if self.rho_mode == "learned":
            gate_input = torch.cat([semantic_query, collaborative_query], dim=-1)  # [B, K, 64]
            rho = self.routing_gate(gate_input)  # [B, K, 1]
        else:
            rho = torch.full((B, self.K, 1), self.rho_value, device=device)

        # Fused logits
        logits = rho * sem_logits + (1.0 - rho) * comp_logits  # [B, K, L]

        # Mask + softmax
        logits = logits.masked_fill(attn_mask, float("-inf"))
        attn = F.softmax(logits, dim=-1)
        attn = attn.masked_fill(torch.isnan(attn), 0.0)

        # Value from full history embedding
        V = self.Wv(history_emb)  # [B, L, D]
        interest_vectors = torch.bmm(attn, V)  # [B, K, D]

        return {
            "interest_vectors": interest_vectors,
            "attention_maps": attn,
            "semantic_attention_logits": sem_logits,
            "collaborative_attention_logits": comp_logits,
            "routing_rho": rho.squeeze(-1),  # [B, K]
            "semantic_query": semantic_query,
            "collaborative_query": collaborative_query,
        }
