# -*- coding: UTF-8 -*-
"""
LLMMIRecContextControl — Chapter 4 Phase 3: history contextualization axis test.

STRUCTURAL CONTROL EXPERIMENT. Not the Chapter 4 final method.

Question it answers:
    Before entering the multi-interest extractor, does letting history items
    contextualize EACH OTHER beat the current per-item independent encoding?

What is (and is not) changed
    ONLY the tensor fed into the extractor:
        history_emb_pos -> [contextualizer] -> QueryMultiInterestExtractor
    Everything downstream is byte-identical to LLMMIRecASPCF:
        interest_vectors = existing QueryMultiInterestExtractor(...)
        interest_weights = existing InterestAggregator(history_emb_raw, lengths)
        user_vector      = sum_k w_k V_k
        prediction       = user_vector . candidate_emb
        loss             = BPR + lambda_relation * Chapter3 relation loss
    ItemEncoder / QueryMultiInterestExtractor / InterestAggregator are imported
    unmodified; LLMMIRecASPCF.py is not touched. No Chapter 4 loss is added.

context_mode
    baseline     : strictly reproduces ASPCF (no extra module is even built)
    ffn_control  : per-position MLP, NO cross-position mixing (capacity control)
    self_attn    : 1-layer Pre-LN self-attention over history positions

MASKING POLICY (explicit, no causal mask in this version)
    The task is "predict the next item from the FULL history". Every history
    position occurs strictly BEFORE the target item, so no position can leak
    the target. A causal triangular mask is therefore NOT used in this first
    version -- only the padding mask. This is stated deliberately: a causal
    mask would be needed only if the sequence contained future items.
    Padding is handled two ways:
        1. keys are masked in attention (key_padding_mask)
        2. block output at padded positions is explicitly zeroed
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import SequentialModel
from models.sequential.llmmi_utils import load_llm_table, check_nan_inf
from models.sequential.llmmi_components import (
    ItemEncoder,
    QueryMultiInterestExtractor,
    InterestAggregator,
)

VALID_CONTEXT_MODES = ("baseline", "ffn_control", "self_attn")


# =========================
#  Self-attention contextual block (Pre-LN, 1 layer)
# =========================

class HistorySelfAttnBlock(nn.Module):
    """One Pre-LN self-attention block over history positions.

        QKV = LN1(X)
        A   = MHSA(QKV, QKV, QKV, key_padding_mask)
        Y   = X + dropout(A)
        H   = Y + dropout(FFN(LN2(Y)))

    Returns (output, attention_weights). attention_weights is [B, h, L, L] and is
    returned for DIAGNOSIS only (entropy / diagonal mass / head similarity).
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, d_ff: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_k = self.d_model // self.n_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn1 = nn.Linear(d_model, d_ff)
        self.ffn2 = nn.Linear(d_ff, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, need_attn=False):
        """
        Args:
            x: [B, L, D]
            key_padding_mask: [B, L] bool, True = valid key (attend allowed)
        Returns:
            (out [B,L,D], attn [B,h,L,L] or None)
        """
        B, L, D = x.shape
        h, dk = self.n_heads, self.d_k

        qkv = self.ln1(x)

        def split(t):
            return t.view(B, L, h, dk).transpose(1, 2)      # [B,h,L,dk]

        q = split(self.q_proj(qkv))
        k = split(self.k_proj(qkv))
        v = split(self.v_proj(qkv))

        scores = torch.matmul(q, k.transpose(-1, -2)) / (dk ** 0.5)   # [B,h,L,L]

        if key_padding_mask is not None:
            # [B,L] valid -> [B,1,1,L] broadcast; invalid keys -> -inf
            scores = scores.masked_fill(
                ~key_padding_mask[:, None, None, :], float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        # all-masked rows (length 0) produce NaN -> 0, same convention as
        # QueryMultiInterestExtractor and utils.layers.MultiHeadAttention
        attn = attn.masked_fill(torch.isnan(attn), 0.0)

        ctx = torch.matmul(attn, v)                          # [B,h,L,dk]
        ctx = ctx.transpose(1, 2).reshape(B, L, D)
        ctx = self.out_proj(ctx)

        y = x + self.dropout1(ctx)                           # residual

        z = self.ln2(y)
        z = self.ffn2(F.gelu(self.ffn1(z)))
        out = y + self.dropout2(z)                           # residual

        return out, (attn if need_attn else None)


# =========================
#  Parameter-matched FFN control (NO cross-position mixing)
# =========================

class HistoryFFNControlBlock(nn.Module):
    """Per-position MLP:  H + dropout(MLP(LN(H))).

    Deliberately contains NO operation that reads another position's value.
    Every position is transformed independently, so perturbing position j
    provably cannot change position i's output. This isolates "added nonlinear
    capacity" from "cross-position interaction".
    """

    def __init__(self, d_model: int = 64, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, need_attn=False):
        z = self.ln(x)
        z = self.fc2(F.gelu(self.fc1(z)))
        return x + self.dropout(z), None


# =========================
#  Model
# =========================

class LLMMIRecContextControl(SequentialModel):
    reader = "SeqReader"
    runner = "BaseRunner"

    extra_log_args = [
        "emb_size", "K", "item_encoder", "adapter_hidden",
        "adapter_activation", "adapter_use_ln",
        "semantic_rank", "lambda_relation", "aspcf_gate_mode",
        "context_mode", "context_dropout", "context_heads",
        "context_ffn_hidden", "context_ffn_control_hidden",
    ]

    # ========================= Args =========================

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--emb_size", type=int, default=64)
        parser.add_argument("--attn_size", type=int, default=64)
        parser.add_argument("--K", type=int, default=4)

        parser.add_argument("--item_encoder", type=str, default="aspcf",
                           choices=["id", "llm_replace", "residual", "aspcf"])
        parser.add_argument("--llm_emb_path", type=str, default="")

        parser.add_argument("--adapter_hidden", type=int, default=256)
        parser.add_argument("--adapter_activation", type=str, default="gelu",
                           choices=["gelu", "relu"])
        parser.add_argument("--adapter_use_ln", type=int, default=0, choices=[0, 1])
        parser.add_argument("--gamma_init", type=float, default=0.1)
        parser.add_argument("--gamma_trainable", type=int, default=0, choices=[0, 1])

        # ASPCF
        parser.add_argument("--semantic_rank", type=int, default=512)
        parser.add_argument("--semantic_dim", type=int, default=32)
        parser.add_argument("--semantic_hidden", type=int, default=128)
        parser.add_argument("--complement_dim", type=int, default=32)
        parser.add_argument("--tail_hidden", type=int, default=64)
        parser.add_argument("--complement_hidden", type=int, default=64)
        parser.add_argument("--gate_hidden", type=int, default=64)
        parser.add_argument("--aspcf_gate_mode", type=str, default="basic",
                           choices=["basic", "conflict"])

        # Relation loss
        parser.add_argument("--lambda_relation", type=float, default=0.01)
        parser.add_argument("--relation_sample_size", type=int, default=128)
        parser.add_argument("--relation_teacher_temp", type=float, default=0.1)
        parser.add_argument("--relation_student_temp", type=float, default=0.1)

        # Contextualization axis (Phase 3)
        parser.add_argument("--context_mode", type=str, default="baseline",
                           choices=list(VALID_CONTEXT_MODES))
        parser.add_argument("--context_dropout", type=float, default=0.1)
        parser.add_argument("--context_heads", type=int, default=4)
        parser.add_argument("--context_ffn_hidden", type=int, default=128,
                           help="FFN hidden width inside the self-attn block")
        parser.add_argument("--context_ffn_control_hidden", type=int, default=256,
                           help="hidden width of the parameter-matched FFN control")

        parser = SequentialModel.parse_model_args(parser)
        parser.set_defaults(dropout=0.1)
        return parser

    # ========================= Init =========================

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)

        self.emb_size = int(args.emb_size)
        self.attn_size = int(args.attn_size)
        self.K = int(args.K)
        self.max_his = int(args.history_max)

        self.item_encoder_mode = str(getattr(args, "item_encoder", "aspcf"))
        self.llm_emb_path = str(getattr(args, "llm_emb_path", ""))
        self.adapter_hidden = int(getattr(args, "adapter_hidden", 256))
        self.adapter_activation = str(getattr(args, "adapter_activation", "gelu"))
        self.adapter_use_ln = bool(int(getattr(args, "adapter_use_ln", 0)))
        self.gamma_init = float(getattr(args, "gamma_init", 0.1))
        self.gamma_trainable = bool(int(getattr(args, "gamma_trainable", 0)))

        self.semantic_rank = int(getattr(args, "semantic_rank", 512))
        self.semantic_dim = int(getattr(args, "semantic_dim", 32))
        self.semantic_hidden = int(getattr(args, "semantic_hidden", 128))
        self.complement_dim = int(getattr(args, "complement_dim", 32))
        self.tail_hidden = int(getattr(args, "tail_hidden", 64))
        self.complement_hidden = int(getattr(args, "complement_hidden", 64))
        self.gate_hidden = int(getattr(args, "gate_hidden", 64))
        self.aspcf_gate_mode = str(getattr(args, "aspcf_gate_mode", "basic"))

        self.lambda_relation = float(getattr(args, "lambda_relation", 0.01))
        self.relation_sample_size = int(getattr(args, "relation_sample_size", 128))
        self.relation_teacher_temp = float(getattr(args, "relation_teacher_temp", 0.1))
        self.relation_student_temp = float(getattr(args, "relation_student_temp", 0.1))

        self.context_mode = str(getattr(args, "context_mode", "baseline"))
        self.context_dropout = float(getattr(args, "context_dropout", 0.1))
        self.context_heads = int(getattr(args, "context_heads", 4))
        self.context_ffn_hidden = int(getattr(args, "context_ffn_hidden", 128))
        self.context_ffn_control_hidden = int(
            getattr(args, "context_ffn_control_hidden", 256))

        if self.context_mode not in VALID_CONTEXT_MODES:
            raise ValueError(f"Unknown context_mode: {self.context_mode}")

        self.dropout_p = float(getattr(args, "dropout", 0.1))

        llm_table = None
        if self.item_encoder_mode in ("llm_replace", "residual", "aspcf"):
            llm_table = load_llm_table(self.llm_emb_path, expected_rows=self.item_num)

        self._define_params(llm_table)
        self.apply(self.init_weights)
        self._first_batch_checked = False

        base = self.count_variables()
        ctx = self.count_context_params()
        logging.info(f"[CtxCtl] initialized: context_mode={self.context_mode} "
                     f"enc={self.item_encoder_mode} K={self.K} "
                     f"heads={self.context_heads} ffn={self.context_ffn_hidden} "
                     f"ctx_dropout={self.context_dropout}")
        logging.info(f"[CtxCtl] #params: {base}  (context block: {ctx})")

    def _define_params(self, llm_table):
        """Module names identical to LLMMIRecASPCF for checkpoint compatibility."""
        ie_kwargs = dict(
            item_num=self.item_num, emb_size=self.emb_size,
            mode=self.item_encoder_mode, llm_table=llm_table,
            adapter_hidden=self.adapter_hidden,
            adapter_activation=self.adapter_activation,
            adapter_use_ln=self.adapter_use_ln,
            gamma_init=self.gamma_init, gamma_trainable=self.gamma_trainable,
        )
        if self.item_encoder_mode == "aspcf":
            ie_kwargs.update(
                semantic_rank=self.semantic_rank, semantic_dim=self.semantic_dim,
                semantic_hidden=self.semantic_hidden, complement_dim=self.complement_dim,
                tail_hidden=self.tail_hidden, complement_hidden=self.complement_hidden,
                gate_hidden=self.gate_hidden, aspcf_gate_mode=self.aspcf_gate_mode,
            )
        self.item_encoder = ItemEncoder(**ie_kwargs)

        self.position_emb = nn.Embedding(self.max_his + 1, self.emb_size)
        self.extractor = QueryMultiInterestExtractor(
            K=self.K, emb_size=self.emb_size, attn_size=self.attn_size)
        self.aggregator = InterestAggregator(emb_size=self.emb_size, K=self.K)
        self.dropout = nn.Dropout(p=self.dropout_p)

        # ---- the ONLY added module (absent in baseline mode) ----
        if self.context_mode == "self_attn":
            self.contextualizer = HistorySelfAttnBlock(
                d_model=self.emb_size, n_heads=self.context_heads,
                d_ff=self.context_ffn_hidden, dropout=self.context_dropout)
        elif self.context_mode == "ffn_control":
            self.contextualizer = HistoryFFNControlBlock(
                d_model=self.emb_size, d_ff=self.context_ffn_control_hidden,
                dropout=self.context_dropout)
        else:
            self.contextualizer = None

    def count_context_params(self):
        if self.contextualizer is None:
            return 0
        return int(sum(p.numel() for p in self.contextualizer.parameters()))

    def base_params(self):
        """Params excluding the context block (i.e. the frozen ASPCF part)."""
        own = sum(p.numel() for p in self.parameters())
        return int(own - self.count_context_params())

    # ========================= Forward =========================

    def forward(self, feed_dict, return_intermediate=False):
        history = feed_dict["history_items"]
        lengths = feed_dict["lengths"]
        i_ids = feed_dict["item_id"]
        B, L = history.shape
        device = history.device

        # 1. Item embeddings — UNCHANGED
        history_emb_raw = self.item_encoder(history)
        candidate_emb = self.item_encoder(i_ids)

        # 2. Position encoding — UNCHANGED
        valid_his = (history > 0).long()
        len_range = torch.arange(self.max_his, device=device)
        position = (lengths[:, None] - len_range[None, :L]) * valid_his
        history_emb_pos = history_emb_raw + self.position_emb(position)
        history_emb_pos = self.dropout(history_emb_pos)

        # 3. ★ THE ONLY INSERTION POINT ★
        ctx_attn = None
        if self.contextualizer is None:
            history_ctx = history_emb_pos
        else:
            key_pad = (torch.arange(L, device=device)[None, :]
                       < lengths[:, None])                       # [B,L] True=valid
            history_ctx, ctx_attn = self.contextualizer(
                history_emb_pos, key_padding_mask=key_pad,
                need_attn=return_intermediate)
            # padded positions explicitly zeroed: a padded query row would
            # otherwise carry a garbage representation into the extractor
            history_ctx = history_ctx * key_pad[:, :, None].to(history_ctx.dtype)

        # 4. Multi-interest extraction — UNCHANGED module
        interest_vectors, attention_maps = self.extractor(history_ctx, lengths)
        interest_vectors = self.dropout(interest_vectors)

        # 5-7. Aggregation / user vector / prediction — UNCHANGED
        # NOTE: aggregator still reads history_emb_raw (pre-context), as specified.
        interest_weights = self.aggregator(history_emb_raw, lengths)
        user_vector = (interest_vectors * interest_weights[:, :, None]).sum(dim=1)
        prediction = (user_vector[:, None, :] * candidate_emb).sum(dim=-1)

        # 8. Relation loss stashing — UNCHANGED
        if (self.training and self.lambda_relation > 0
                and self.item_encoder_mode in ("aspcf", "llm_replace")):
            all_ids = torch.cat([history.reshape(-1), i_ids.reshape(-1)], dim=0)
            unique_ids = torch.unique(all_ids)
            unique_ids = unique_ids[unique_ids != 0]
            if unique_ids.numel() > self.relation_sample_size:
                idx = torch.randperm(unique_ids.numel(), device=device)[:self.relation_sample_size]
                unique_ids = unique_ids[idx]
            out_dict = {"prediction": prediction, "_relation_ids": unique_ids}
        else:
            out_dict = {"prediction": prediction}

        # 9. NaN/Inf check
        if not self._first_batch_checked:
            self._first_batch_checked = True
            for name, t in [("history_vectors", history_emb_raw),
                            ("history_context", history_ctx),
                            ("interest_vectors", interest_vectors),
                            ("candidate_vectors", candidate_emb),
                            ("prediction", prediction)]:
                check_nan_inf(t, name)
            logging.info("[CtxCtl] First-batch NaN/Inf check passed.")

        # 10. Output
        if return_intermediate:
            out_dict["interest_vectors"] = interest_vectors
            out_dict["attention_maps"] = attention_maps
            out_dict["interest_weights"] = interest_weights
            out_dict["user_vector"] = user_vector
            out_dict["history_vectors"] = history_emb_raw
            out_dict["candidate_vectors"] = candidate_emb
            out_dict["history_pre_context"] = history_emb_pos
            out_dict["history_post_context"] = history_ctx
            out_dict["context_attention_maps"] = ctx_attn

        return out_dict

    # ========================= Loss =========================
    # LOSS = BPR + lambda_relation * Chapter 3 relation loss.
    # No Chapter 4 auxiliary loss this round.

    def loss(self, out_dict: dict):
        total = super().loss(out_dict)
        if "_relation_ids" in out_dict and self.lambda_relation > 0:
            rel = self._compute_relation_loss(out_dict["_relation_ids"])
            total = total + self.lambda_relation * rel
            out_dict["loss_relation"] = rel.detach()
        return total

    def _compute_relation_loss(self, item_ids):
        M = item_ids.numel()
        if M < 2:
            return torch.zeros([], device=item_ids.device)
        z = self.item_encoder.llm_table[item_ids]
        teacher = z[:, :self.semantic_rank]
        if self.item_encoder_mode == "aspcf":
            student = self.item_encoder.semantic_branch(teacher)
        else:
            student = self.item_encoder.adapter(z)
        teacher = F.normalize(teacher, dim=-1, eps=1e-8)
        student = F.normalize(student, dim=-1, eps=1e-8)
        t_sim = teacher @ teacher.t() / self.relation_teacher_temp
        s_sim = student @ student.t() / self.relation_student_temp
        mask = ~torch.eye(M, dtype=torch.bool, device=item_ids.device)
        t_sim = t_sim[mask].view(M, M - 1)
        s_sim = s_sim[mask].view(M, M - 1)
        return F.kl_div(F.log_softmax(s_sim, dim=-1),
                        F.softmax(t_sim, dim=-1).detach(), reduction="batchmean")
