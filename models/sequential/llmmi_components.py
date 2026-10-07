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

    Six modes:
      - id:           e = id_embedding(item_id)
      - llm_replace:  e = adapter(llm_table[item_id])
      - residual:     e = id_embedding(item_id) + gamma * adapter(llm_table[item_id])
      - aspcf:        e = concat(sqrt(α_s) * s, sqrt(α_c) * c)
                        where s=semantic_branch(z_high), c=complement_branch(z_low, id),
                        [α_s,α_c]=softmax(gate([s;c]))
      - cgscd:        same fusion skeleton as aspcf, but the split is data-driven:
                        z_sh = (z - z_mean) @ U_r          (shared subspace, [d_llm, r])
                        z_pv = (z - z_mean) - z_sh @ U_r^T (private residual)
                        s    = shared_branch(z_sh)
                        c    = complement_mlp([tail(z_pv); cf_table[item]])
                      U_r comes from cross-view association between the LLM and
                      collaborative item views (tools/build_cgscd_basis.py).
      - rasrf:        reliability-aware semantic residual fusion (Ch3 Round 2):
                        e_cf  = id_embedding(item_id)            CF main path
                        e_sem = adapter(llm_table[item_id])      semantic correction
                        gate  = sigmoid(MLP(consistency signals))  item-specific
                        e     = e_cf + gate * e_sem
                      Signals: cos(e_cf, e_sem), |e_cf - e_sem|, e_cf * e_sem, and
                      optionally an offline neighbourhood-agreement prior
                      (tools/build_semantic_neighborhood_agreement.py).
                      At init all Linears are ~N(0, 0.01) so e_sem ~ 0 and the
                      model starts as pure CF; the correction is grown only if
                      training finds it useful.

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
        # ── CGSCD params ──
        shared_basis: torch.Tensor = None,
        z_mean: torch.Tensor = None,
        cf_table: torch.Tensor = None,
        shared_dim: int = 32,
        shared_hidden: int = 128,
        compl_dim: int = 32,
        compl_hidden: int = 64,
        cgscd_gate_mode: str = "basic",
        # ── RASRF params ──
        neigh_prior: torch.Tensor = None,
        rasrf_gate_mode: str = "scalar",
        rasrf_gate_input: str = "agree_diff_inter",
        rasrf_gate_hidden: int = 64,
    ):
        super().__init__()
        self.item_num = int(item_num)
        self.emb_size = int(emb_size)
        self.mode = mode

        if mode not in ("id", "llm_replace", "residual", "aspcf", "cgscd", "rasrf"):
            raise ValueError(
                f"Unknown item_encoder mode: '{mode}'. "
                f"Supported: id, llm_replace, residual, aspcf, cgscd, rasrf"
            )

        # --- ID embedding (used in 'id', 'residual' and 'rasrf' modes) ---
        if mode in ("id", "residual", "rasrf"):
            self.id_embedding = nn.Embedding(item_num, emb_size)

        # --- LLM-based modes (llm_replace, residual, aspcf, cgscd, rasrf) ---
        if mode in ("llm_replace", "residual", "aspcf", "cgscd", "rasrf"):
            if llm_table is None:
                raise ValueError(f"item_encoder='{mode}' requires llm_table, got None")
            if llm_table.shape[0] != item_num:
                raise ValueError(
                    f"llm_table shape[0]={llm_table.shape[0]} != item_num={item_num}"
                )

            self.register_buffer("llm_table", llm_table, persistent=False)
            d_llm = llm_table.size(1)

        # --- llm_replace / residual / rasrf adapter (semantic branch) ---
        if mode in ("llm_replace", "residual", "rasrf"):
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

        # --- CGSCD ---
        if mode == "cgscd":
            if shared_dim + compl_dim != emb_size:
                raise ValueError(
                    f"shared_dim({shared_dim}) + compl_dim({compl_dim}) "
                    f"!= emb_size({emb_size})"
                )
            if shared_basis is None or z_mean is None or cf_table is None:
                raise ValueError(
                    "item_encoder='cgscd' requires shared_basis, z_mean and cf_table"
                )
            if shared_basis.dim() != 2 or shared_basis.size(0) != d_llm:
                raise ValueError(
                    f"shared_basis must be [d_llm={d_llm}, r], "
                    f"got {tuple(shared_basis.shape)}"
                )
            if z_mean.numel() != d_llm:
                raise ValueError(f"z_mean must have {d_llm} entries, got {z_mean.numel()}")
            if cf_table.shape[0] != item_num:
                raise ValueError(
                    f"cf_table shape[0]={cf_table.shape[0]} != item_num={item_num}"
                )

            self.shared_dim = int(shared_dim)
            self.compl_dim = int(compl_dim)
            self.cgscd_gate_mode = cgscd_gate_mode
            self.cf_dim = int(cf_table.size(1))

            # The private residual keeps the full d_llm width.
            self.register_buffer("shared_basis", shared_basis, persistent=False)
            self.register_buffer("z_mean", z_mean, persistent=False)
            self.register_buffer("cf_table", cf_table, persistent=False)

            # Shared branch: z_shared -> s  (mirrors ASPCF semantic_branch shape)
            self.shared_branch = nn.Sequential(
                nn.Linear(shared_basis.size(1), shared_hidden),
                nn.GELU(),
                nn.Linear(shared_hidden, shared_dim),
            )

            # Complement: private residual processing (mirrors ASPCF complement_tail)
            self.compl_tail = nn.Sequential(
                nn.Linear(d_llm, compl_hidden),
                nn.GELU(),
            )

            # Complement: fusion with the collaborative item embedding
            self.compl_mlp = nn.Sequential(
                nn.Linear(compl_hidden + self.cf_dim, compl_hidden),
                nn.GELU(),
                nn.Linear(compl_hidden, compl_dim),
            )

            # Gate: basic=[s;c], conflict=[s;c;|s-c|;s*c]
            if cgscd_gate_mode == "basic":
                gate_in_dim = shared_dim + compl_dim
            elif cgscd_gate_mode == "conflict":
                if shared_dim != compl_dim:
                    raise ValueError(
                        f"cgscd_gate_mode='conflict' requires shared_dim == compl_dim "
                        f"(|s-c| and s*c are elementwise), got "
                        f"shared_dim={shared_dim}, compl_dim={compl_dim}"
                    )
                gate_in_dim = (shared_dim + compl_dim) * 2
            else:
                raise ValueError(f"Unknown cgscd_gate_mode: {cgscd_gate_mode}")
            self.gate = nn.Sequential(
                nn.Linear(gate_in_dim, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, 2),
            )

        # --- RASRF ---
        if mode == "rasrf":
            if rasrf_gate_mode not in ("scalar", "vector"):
                raise ValueError(f"Unknown rasrf_gate_mode: {rasrf_gate_mode}")
            if rasrf_gate_input not in ("agree", "agree_diff", "agree_diff_inter"):
                raise ValueError(f"Unknown rasrf_gate_input: {rasrf_gate_input}")

            self.rasrf_gate_mode = rasrf_gate_mode
            self.rasrf_gate_input = rasrf_gate_input

            # Consistency-signal layout
            n_sig = 1                                  # cos(e_cf, e_sem)
            if rasrf_gate_input in ("agree_diff", "agree_diff_inter"):
                n_sig += emb_size                      # |e_cf - e_sem|
            if rasrf_gate_input == "agree_diff_inter":
                n_sig += emb_size                      # e_cf * e_sem

            self.has_neigh = neigh_prior is not None
            if self.has_neigh:
                if neigh_prior.dim() != 2 or neigh_prior.size(0) != item_num:
                    raise ValueError(
                        f"neigh_prior must be [item_num={item_num}, n_signals], "
                        f"got {tuple(neigh_prior.shape)}"
                    )
                self.register_buffer("neigh_prior", neigh_prior, persistent=False)
                self.neigh_dim = int(neigh_prior.size(1))
                n_sig += self.neigh_dim
            else:
                self.neigh_dim = 0

            self.gate_in_dim = n_sig
            self.gate_out_dim = 1 if rasrf_gate_mode == "scalar" else emb_size
            self.rasrf_gate = nn.Sequential(
                nn.Linear(self.gate_in_dim, rasrf_gate_hidden),
                nn.GELU(),
                nn.Linear(rasrf_gate_hidden, self.gate_out_dim),
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
        elif self.mode == "cgscd":
            return self._forward_cgscd(item_ids, return_components=return_components)
        elif self.mode == "rasrf":
            return self._forward_rasrf(item_ids, return_components=return_components)
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

    def _forward_cgscd(self, item_ids: torch.Tensor, return_components: bool = False):
        """CGSCD forward: data-driven shared / private split of the LLM view.

        z_sh = (z - z_mean) @ U_r            shared subspace coordinates
        z_pv = (z - z_mean) - z_sh @ U_r^T   private residual (orthogonal complement)
        """
        z = self.llm_table[item_ids]                       # [*, d_llm]
        z_c = z - self.z_mean                              # [*, d_llm]
        z_sh = z_c @ self.shared_basis                     # [*, r]
        z_pv = z_c - z_sh @ self.shared_basis.t()          # [*, d_llm]

        # Shared branch
        s = self.shared_branch(z_sh)                       # [*, shared_dim]

        # Complement branch: private residual + collaborative item embedding
        c_anchor = self.cf_table[item_ids]                 # [*, d_cf]
        comp_input = torch.cat([self.compl_tail(z_pv), c_anchor], dim=-1)
        c = self.compl_mlp(comp_input)                     # [*, compl_dim]

        # Gate
        if self.cgscd_gate_mode == "basic":
            gate_input = torch.cat([s, c], dim=-1)
        else:  # conflict
            gate_input = torch.cat([s, c, torch.abs(s - c), s * c], dim=-1)
        gate_weights = F.softmax(self.gate(gate_input), dim=-1)
        alpha_sem = gate_weights[..., 0]
        alpha_comp = gate_weights[..., 1]

        eps = 1e-8
        e = torch.cat([
            torch.sqrt(alpha_sem.unsqueeze(-1) + eps) * s,
            torch.sqrt(alpha_comp.unsqueeze(-1) + eps) * c,
        ], dim=-1)                                          # [*, emb_size]

        pad_mask = (item_ids == 0).float().unsqueeze(-1)
        e = e * (1.0 - pad_mask)

        if return_components:
            return {
                "emb": e,
                "semantic": s * (1.0 - pad_mask),
                "complement": c * (1.0 - pad_mask),
                "alpha_sem": alpha_sem * (1.0 - pad_mask.squeeze(-1)),
                "alpha_comp": alpha_comp * (1.0 - pad_mask.squeeze(-1)),
                "z_shared": z_sh * (1.0 - pad_mask),
                "z_private": z_pv * (1.0 - pad_mask),
            }
        return e

    def _forward_rasrf(self, item_ids: torch.Tensor, return_components: bool = False):
        """Reliability-aware semantic residual fusion.

        e = e_cf + gate(consistency(e_cf, e_sem)) * e_sem

        e_cf is the main path (collaborative, learned only from interactions).
        The semantic correction is gated per item by explicit cross-view
        consistency signals, so it can be suppressed where the LLM view does
        not agree with the collaborative view.
        """
        e_cf = self.id_embedding(item_ids)                    # [*, D]
        e_sem = self.adapter(self.llm_table[item_ids])        # [*, D]

        parts = []
        if self.rasrf_gate_input in ("agree", "agree_diff", "agree_diff_inter"):
            cos = F.cosine_similarity(e_cf, e_sem, dim=-1, eps=1e-8)
            parts.append(cos.unsqueeze(-1))                   # [*, 1]
        if self.rasrf_gate_input in ("agree_diff", "agree_diff_inter"):
            parts.append(torch.abs(e_cf - e_sem))             # [*, D]
        if self.rasrf_gate_input == "agree_diff_inter":
            parts.append(e_cf * e_sem)                        # [*, D]
        if self.has_neigh:
            parts.append(self.neigh_prior[item_ids])          # [*, neigh_dim]

        gate_in = torch.cat(parts, dim=-1)                    # [*, gate_in_dim]
        gate = torch.sigmoid(self.rasrf_gate(gate_in))        # [*, 1] or [*, D]

        e = e_cf + gate * e_sem                               # [*, D]

        pad_mask = (item_ids == 0).float().unsqueeze(-1)
        e = e * (1.0 - pad_mask)

        if return_components:
            return {
                "emb": e,
                "semantic": (gate * e_sem) * (1.0 - pad_mask),   # gated correction
                "complement": e_cf * (1.0 - pad_mask),           # CF main path
                "gate": gate * (1.0 - pad_mask),
                "e_cf": e_cf * (1.0 - pad_mask),
                "e_sem_raw": e_sem * (1.0 - pad_mask),
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
        return_route_scores: bool = False,
    ):
        """Extract interest vectors from history embeddings.

        Args:
            history_emb: [B, L, D] history item embeddings (with position encoding)
            lengths: [B] valid lengths per sample
            external_query: [B, K, D] optional external query seeds.
            attention_prior: [B, K, L] optional prior for attention logits.
            prior_strength: weight for attention_prior in log space.
            return_route_scores: if True, also return raw scores [B,K,L]
                (before softmax, after mask+prior). Used for HSDIR routing.

        Returns:
            Default: (interest_vectors, attention_maps)
            With prior: (interest_vectors, attention_maps, logits_before_prior)
            With return_route_scores: (interest_vectors, attention_maps, raw_scores)
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

        # Save raw scores BEFORE mask (for HSDIR routing)
        raw_route_scores = scores.clone() if return_route_scores else None

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

        if return_route_scores:
            # Return raw scores BEFORE padding mask (no -inf, safe for softmax over K)
            return interest_vectors, attn, raw_route_scores
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


# =========================
#  Chapter 4 — ECTIR: transport routing + evidence aggregation
# =========================

# Finite stand-in for -inf: exp(-1e4) underflows to exactly 0, and unlike -inf it
# cannot produce NaN through (-inf) - (-inf) in the Sinkhorn normalisations.
_NEG = -1e4


class TransportInterestRouter(nn.Module):
    """Evidence-Constrained Transport Interest Routing (Chapter 4, Module 1 + 2).

    Replaces the per-interest independent softmax over history with a globally
    coupled entropic optimal-transport assignment:

      - history side: row marginal `a` = uniform over valid positions
      - interest side: column marginal `b` = softmax_k(logsumexp_l S[l,k] / tau_c),
        i.e. capacity is derived from each interest's affinity evidence and is
        NOT forced to be uniform
      - Sinkhorn (log domain) solves for the transport plan T [B, L, K]

    T is then row-normalised and used to form V_k as a convex combination of
    values, so V_k keeps the same form as the baseline attention output; only
    the way the weights are produced changes.

    Module 2 (optional, n_refine > 0) re-derives the interest query from the
    current V_k, making the assignment sequence-specific.

    Shapes: internal affinity/transport use [B, L, K]; returned interest vectors
    use the baseline convention [B, K, D].
    """

    def __init__(self, K: int, emb_size: int, attn_size: int,
                 eps: float = 0.1, tau_c: float = 1.0,
                 n_sinkhorn: int = 5, n_refine: int = 1):
        super().__init__()
        if eps <= 0:
            raise ValueError(f"ectir eps must be > 0, got {eps}")
        if n_sinkhorn < 1:
            raise ValueError(f"ectir n_sinkhorn must be >= 1, got {n_sinkhorn}")
        if n_refine < 0:
            raise ValueError(f"ectir n_refine must be >= 0, got {n_refine}")

        self.K = int(K)
        self.emb_size = int(emb_size)
        self.attn_size = int(attn_size)
        self.eps = float(eps)
        self.tau_c = float(tau_c)
        self.n_sinkhorn = int(n_sinkhorn)
        self.n_refine = int(n_refine)

        # same parameter shapes/semantics as QueryMultiInterestExtractor
        self.query = nn.Parameter(torch.empty(K, emb_size))
        nn.init.normal_(self.query, mean=0.0, std=0.01)
        self.Wq = nn.Linear(emb_size, attn_size)
        self.Wk = nn.Linear(emb_size, attn_size)
        self.Wv = nn.Linear(emb_size, emb_size)

        # Module 2 refinement: near-zero init => q^(1) ~ q at start of training
        self.W_r = nn.Linear(emb_size, emb_size) if n_refine > 0 else None

    def _affinity(self, Kmat: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """Kmat [B, L, d], Q [B, K, d] -> S [B, L, K]."""
        return torch.bmm(Kmat, Q.transpose(1, 2)) / math.sqrt(self.attn_size)

    def _sinkhorn(self, S: torch.Tensor, valid: torch.Tensor):
        """Log-domain entropic transport.

        Args:
            S: [B, L, K] affinity, padding already set to _NEG
            valid: [B, L] float mask (1 = valid)

        Returns:
            T: [B, L, K] transport plan (exactly 0 at padding, row sums = a,
               column sums ~= b)
            b: [B, K] interest capacity (column marginal target)
        """
        B, L, K = S.shape
        mb = valid.bool()

        # history-side mass: uniform over valid positions
        a = valid / valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        log_a = torch.log(a + 1e-12)

        # interest-side capacity: from affinity evidence, NOT uniform
        evidence = torch.logsumexp(S, dim=1)                  # [B, K]
        b = torch.softmax(evidence / self.tau_c, dim=-1)      # [B, K]
        log_b = torch.log(b + 1e-12)

        M = S / self.eps
        f = S.new_zeros(B, L)
        g = S.new_zeros(B, K)
        for _ in range(self.n_sinkhorn):
            f_new = log_a - torch.logsumexp(M + g[:, None, :], dim=2)   # [B, L]
            f = torch.where(mb, f_new, torch.full_like(f_new, _NEG))
            g = log_b - torch.logsumexp(M + f[:, :, None], dim=1)       # [B, K]

        T = torch.exp(M + f[:, :, None] + g[:, None, :]) * valid[:, :, None]
        return T, b

    def _values(self, T: torch.Tensor, Vmat: torch.Tensor):
        """T [B, L, K], Vmat [B, L, D] -> (V [B, K, D], A [B, K, L]).

        A is the row-normalised plan, i.e. the actual attention used to form V.
        It follows the baseline convention (rows sum to 1 over L), which is what
        the downstream diagnostics expect. The raw plan is reported separately
        via `transport` in the info dict.
        """
        Tm = T.transpose(1, 2)                                # [B, K, L]
        denom = Tm.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        A = Tm / denom
        return torch.bmm(A, Vmat), A

    def forward(self, history_emb: torch.Tensor, lengths: torch.Tensor,
                return_intermediate: bool = False):
        B, L, D = history_emb.shape
        valid = (torch.arange(L, device=history_emb.device)[None, :]
                 < lengths[:, None]).float()                  # [B, L]
        # match the baseline: keep raw scores free of -inf before masking
        valid_f = valid
        neg_mask = ~valid.bool()

        Kmat = self.Wk(history_emb)                           # [B, L, d]
        Vmat = self.Wv(history_emb)                           # [B, L, D]

        # ---- step 0 ----
        Q0 = self.Wq(self.query).unsqueeze(0).expand(B, -1, -1)        # [B, K, d]
        S = self._affinity(Kmat, Q0).masked_fill(neg_mask[:, :, None], _NEG)
        T, b_cap = self._sinkhorn(S, valid_f)
        V, A = self._values(T, Vmat)

        # ---- Module 2: sequence-specific refinement ----
        if self.W_r is not None:
            for _ in range(self.n_refine):
                q_t = self.query.unsqueeze(0) + self.W_r(V)            # [B, K, D]
                Q_t = self.Wq(q_t)
                S = self._affinity(Kmat, Q_t).masked_fill(neg_mask[:, :, None], _NEG)
                T, b_cap = self._sinkhorn(S, valid_f)
                V, A = self._values(T, Vmat)

        if return_intermediate:
            return V, A, {"transport": T, "capacity": b_cap, "affinity": S}
        return V, A


class TransportEvidenceAggregator(nn.Module):
    """Transport-Evidence Interest Aggregation (Chapter 4, Module 3).

    Interest weights are produced from three evidence features computed off the
    FINAL transport plan, together with the interest vector and a history
    summary:

      mass_k : total transport mass acquired by interest k
      conc_k : concentration of that mass over history (1 - normalised entropy)
      rec_k  : temporal centroid (how recent the mass is)

    Everything is history-only (no candidate/target information), so the
    aggregation remains leakage-free.

    NOTE: the MLP here is only a local parameterisation of the evidence. The
    contribution is the evidence features themselves, not the gate.
    """

    def __init__(self, K: int, emb_size: int, hidden: int = 64, n_evidence: int = 3):
        super().__init__()
        self.K = int(K)
        self.emb_size = int(emb_size)
        self.n_evidence = int(n_evidence)
        self.ln = nn.LayerNorm(emb_size)
        in_dim = emb_size + n_evidence + emb_size          # V_k ; evidence ; h_sum
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, interest_vectors: torch.Tensor, transport: torch.Tensor,
                history_emb: torch.Tensor, lengths: torch.Tensor,
                position: torch.Tensor):
        """
        Args:
            interest_vectors: [B, K, D]
            transport:        [B, L, K]  final plan
            history_emb:      [B, L, D]  raw (no position) history embeddings
            lengths:          [B]
            position:         [B, L]     baseline position values (larger = newer)
        Returns:
            weights: [B, K] softmax over K
        """
        B, L, K = transport.shape
        device = transport.device
        valid = (torch.arange(L, device=device)[None, :] < lengths[:, None]).float()

        mass = transport.sum(dim=1)                                    # [B, K]
        p = transport / mass[:, None, :].clamp(min=1e-8)               # [B, L, K]
        ent = -(p * torch.log(p + 1e-12)).sum(dim=1)                   # [B, K]
        logL = torch.log(valid.sum(dim=1).clamp(min=2.0))[:, None]     # [B, 1]
        conc = (1.0 - ent / logL.clamp(min=1e-6)).clamp(0.0, 1.0)      # [B, K]

        pos_max = position.max().clamp(min=1.0)
        rec = (transport * position[:, :, None]).sum(dim=1) / mass.clamp(min=1e-8)
        rec = (rec / pos_max).clamp(0.0, 1.0)                          # [B, K]

        # history summary: same construction as InterestAggregator
        sum_emb = (history_emb * valid[:, :, None]).sum(dim=1)
        count = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_his = sum_emb / count
        last_idx = (lengths - 1).clamp(min=0).long()
        last_his = history_emb[torch.arange(B, device=device), last_idx]
        h_sum = self.ln(mean_his + last_his)                           # [B, D]

        ev = torch.stack([mass, conc, rec], dim=-1)                    # [B, K, 3]
        h_rep = h_sum[:, None, :].expand(-1, self.K, -1)               # [B, K, D]
        feat = torch.cat([interest_vectors, ev, h_rep], dim=-1)        # [B, K, 2D+3]
        logits = self.mlp(feat).squeeze(-1)                            # [B, K]
        return torch.softmax(logits, dim=-1)
