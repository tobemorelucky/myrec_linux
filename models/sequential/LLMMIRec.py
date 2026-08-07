# -*- coding: UTF-8 -*-
"""
LLMMIRec: Clean LLM-enhanced Multi-Interest sequential Recommendation baseline.

Phase 0 — minimal working baseline (id / llm_replace / residual).
Phase 1 — ASPCF: Adaptive Semantic-Preserving Subspace Complementary Fusion.
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


class LLMMIRec(SequentialModel):
    reader = "SeqReader"
    runner = "BaseRunner"

    extra_log_args = [
        "emb_size",
        "K",
        "item_encoder",
        "adapter_hidden",
        "adapter_activation",
        "adapter_use_ln",
        "semantic_rank",
        "lambda_relation",
    ]

    # =========================
    #  Args
    # =========================

    @staticmethod
    def parse_model_args(parser):
        # ---- core architecture ----
        parser.add_argument("--emb_size", type=int, default=64)
        parser.add_argument("--attn_size", type=int, default=64)
        parser.add_argument("--K", type=int, default=4)

        # ---- item encoder ----
        parser.add_argument(
            "--item_encoder", type=str, default="llm_replace",
            choices=["id", "llm_replace", "residual", "aspcf"],
        )
        parser.add_argument("--llm_emb_path", type=str, default="",
                           help="Path to LLM embedding pkl table")

        # ---- adapter (llm_replace / residual) ----
        parser.add_argument("--adapter_hidden", type=int, default=256)
        parser.add_argument(
            "--adapter_activation", type=str, default="gelu",
            choices=["gelu", "relu"],
        )
        parser.add_argument("--adapter_use_ln", type=int, default=0,
                           choices=[0, 1])

        # ---- fusion gamma (residual mode only) ----
        parser.add_argument("--gamma_init", type=float, default=0.1)
        parser.add_argument("--gamma_trainable", type=int, default=0,
                           choices=[0, 1])

        # ---- ASPCF ----
        parser.add_argument("--semantic_rank", type=int, default=512)
        parser.add_argument("--semantic_dim", type=int, default=32)
        parser.add_argument("--semantic_hidden", type=int, default=128)
        parser.add_argument("--complement_dim", type=int, default=32)
        parser.add_argument("--tail_hidden", type=int, default=64)
        parser.add_argument("--complement_hidden", type=int, default=64)
        parser.add_argument("--gate_hidden", type=int, default=64)

        # ---- relation preservation loss ----
        parser.add_argument("--lambda_relation", type=float, default=0.0,
                           help="Weight for semantic relation preservation loss")
        parser.add_argument("--relation_sample_size", type=int, default=128)
        parser.add_argument("--relation_teacher_temp", type=float, default=0.1)
        parser.add_argument("--relation_student_temp", type=float, default=0.1)

        # ---- regularization (dropout is defined in GeneralModel.parse_model_args) ----

        parser = SequentialModel.parse_model_args(parser)

        # Override GeneralModel's default (0) to 0.1 for LLMMIRec
        parser.set_defaults(dropout=0.1)

        return parser

    # =========================
    #  Init
    # =========================

    @staticmethod
    def init_weights(m):
        """Safe init using isinstance (parent class uses string matching)."""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)

        # ---- core params ----
        self.emb_size = int(args.emb_size)
        self.attn_size = int(args.attn_size)
        self.K = int(args.K)
        self.max_his = int(args.history_max)

        # ---- item encoder params ----
        self.item_encoder_mode = str(getattr(args, "item_encoder", "llm_replace"))
        self.llm_emb_path = str(getattr(args, "llm_emb_path", ""))
        self.adapter_hidden = int(getattr(args, "adapter_hidden", 256))
        self.adapter_activation = str(getattr(args, "adapter_activation", "gelu"))
        self.adapter_use_ln = bool(int(getattr(args, "adapter_use_ln", 0)))

        # ---- gamma (residual only) ----
        self.gamma_init = float(getattr(args, "gamma_init", 0.1))
        self.gamma_trainable = bool(int(getattr(args, "gamma_trainable", 0)))

        # ---- ASPCF params ----
        self.semantic_rank = int(getattr(args, "semantic_rank", 512))
        self.semantic_dim = int(getattr(args, "semantic_dim", 32))
        self.semantic_hidden = int(getattr(args, "semantic_hidden", 128))
        self.complement_dim = int(getattr(args, "complement_dim", 32))
        self.tail_hidden = int(getattr(args, "tail_hidden", 64))
        self.complement_hidden = int(getattr(args, "complement_hidden", 64))
        self.gate_hidden = int(getattr(args, "gate_hidden", 64))

        # ---- relation loss ----
        self.lambda_relation = float(getattr(args, "lambda_relation", 0.0))
        self.relation_sample_size = int(getattr(args, "relation_sample_size", 128))
        self.relation_teacher_temp = float(getattr(args, "relation_teacher_temp", 0.1))
        self.relation_student_temp = float(getattr(args, "relation_student_temp", 0.1))

        # ---- regularization ----
        self.dropout_p = float(getattr(args, "dropout", 0.1))

        # ---- load LLM table (if needed) ----
        llm_table = None
        if self.item_encoder_mode in ("llm_replace", "residual", "aspcf"):
            llm_table = load_llm_table(self.llm_emb_path, expected_rows=self.item_num)

        # ---- build modules ----
        self._define_params(llm_table)
        self.apply(self.init_weights)

        # ---- diagnostics ----
        self._first_batch_checked = False

        logging.info(f"[LLMMIRec] initialized")
        logging.info(f"[LLMMIRec]   item_encoder: {self.item_encoder_mode}")
        logging.info(f"[LLMMIRec]   emb_size={self.emb_size}, attn_size={self.attn_size}, "
                     f"K={self.K}, max_his={self.max_his}")
        if self.item_encoder_mode == "aspcf":
            logging.info(f"[LLMMIRec]   aspcf: semantic_rank={self.semantic_rank} "
                         f"semantic_dim={self.semantic_dim} complement_dim={self.complement_dim}")
        logging.info(f"[LLMMIRec]   adapter: hidden={self.adapter_hidden}, "
                     f"activation={self.adapter_activation}, ln={self.adapter_use_ln}")
        logging.info(f"[LLMMIRec]   gamma_init={self.gamma_init}, "
                     f"gamma_trainable={self.gamma_trainable}")
        logging.info(f"[LLMMIRec]   relation: lambda={self.lambda_relation} "
                     f"sample_size={self.relation_sample_size}")
        logging.info(f"[LLMMIRec]   dropout={self.dropout_p}")
        logging.info(f"[LLMMIRec]   #params: {self.count_variables()}")

    def _define_params(self, llm_table):
        # Item encoder
        ie_kwargs = dict(
            item_num=self.item_num,
            emb_size=self.emb_size,
            mode=self.item_encoder_mode,
            llm_table=llm_table,
            adapter_hidden=self.adapter_hidden,
            adapter_activation=self.adapter_activation,
            adapter_use_ln=self.adapter_use_ln,
            gamma_init=self.gamma_init,
            gamma_trainable=self.gamma_trainable,
        )
        if self.item_encoder_mode == "aspcf":
            ie_kwargs.update(
                semantic_rank=self.semantic_rank,
                semantic_dim=self.semantic_dim,
                semantic_hidden=self.semantic_hidden,
                complement_dim=self.complement_dim,
                tail_hidden=self.tail_hidden,
                complement_hidden=self.complement_hidden,
                gate_hidden=self.gate_hidden,
            )
        self.item_encoder = ItemEncoder(**ie_kwargs)

        # Position embedding
        self.position_emb = nn.Embedding(self.max_his + 1, self.emb_size)

        # Multi-interest extractor
        self.extractor = QueryMultiInterestExtractor(
            K=self.K,
            emb_size=self.emb_size,
            attn_size=self.attn_size,
        )

        # Interest aggregator
        self.aggregator = InterestAggregator(
            emb_size=self.emb_size,
            K=self.K,
        )

        # Dropout
        self.dropout = nn.Dropout(p=self.dropout_p)

    # =========================
    #  Forward
    # =========================

    def forward(self, feed_dict, return_intermediate=False):
        history = feed_dict["history_items"]   # [B, L] int
        lengths = feed_dict["lengths"]          # [B]
        i_ids = feed_dict["item_id"]            # [B, 1+N]

        B, L = history.shape
        device = history.device

        # ---- 1. Item embeddings (shared encoder for history AND candidates) ----
        aspcf_comps = None
        if self.item_encoder_mode == "aspcf" and return_intermediate:
            hist_out = self.item_encoder(history, return_components=True)
            cand_out = self.item_encoder(i_ids, return_components=True)
            history_emb_raw = hist_out["emb"]
            candidate_emb = cand_out["emb"]
            aspcf_comps = {
                "history_semantic": hist_out["semantic"],
                "history_complement": hist_out["complement"],
                "history_alpha_sem": hist_out["alpha_sem"],
                "history_alpha_comp": hist_out["alpha_comp"],
                "candidate_semantic": cand_out["semantic"],
                "candidate_complement": cand_out["complement"],
                "candidate_alpha_sem": cand_out["alpha_sem"],
                "candidate_alpha_comp": cand_out["alpha_comp"],
            }
        else:
            history_emb_raw = self.item_encoder(history)    # [B, L, D]
            candidate_emb = self.item_encoder(i_ids)         # [B, C, D]

        # ---- 2. Position encoding on history ----
        valid_his = (history > 0).long()  # [B, L]
        len_range = torch.arange(self.max_his, device=device)
        position = (lengths[:, None] - len_range[None, :L]) * valid_his  # [B, L]
        pos_vectors = self.position_emb(position)  # [B, L, D]

        history_emb_pos = history_emb_raw + pos_vectors  # [B, L, D]

        # ---- 3. Dropout ----
        history_emb_pos = self.dropout(history_emb_pos)

        # ---- 4. Multi-interest extraction ----
        interest_vectors, attention_maps = self.extractor(history_emb_pos, lengths)

        interest_vectors = self.dropout(interest_vectors)

        # ---- 5. Interest aggregation (history-only, no position) ----
        interest_weights = self.aggregator(history_emb_raw, lengths)  # [B, K]

        # ---- 6. User vector ----
        user_vector = (interest_vectors * interest_weights[:, :, None]).sum(dim=1)  # [B, D]

        # ---- 7. Prediction ----
        prediction = (user_vector[:, None, :] * candidate_emb).sum(dim=-1)  # [B, C]

        # ---- 8. Stash relation loss inputs ----
        if self.item_encoder_mode == "aspcf" and self.lambda_relation > 0:
            all_ids = torch.cat([history.reshape(-1), i_ids.reshape(-1)], dim=0)
            unique_ids = torch.unique(all_ids)
            unique_ids = unique_ids[unique_ids != 0]  # exclude padding
            if unique_ids.numel() > self.relation_sample_size:
                idx = torch.randperm(unique_ids.numel(), device=device)[:self.relation_sample_size]
                unique_ids = unique_ids[idx]
            out_dict = {"prediction": prediction, "_relation_ids": unique_ids}
        else:
            out_dict = {"prediction": prediction}

        # ---- 9. First-batch NaN/Inf check ----
        if not self._first_batch_checked:
            self._first_batch_checked = True
            diagnostics = [
                ("history_vectors", history_emb_raw),
                ("interest_vectors", interest_vectors),
                ("interest_weights", interest_weights),
                ("candidate_vectors", candidate_emb),
                ("prediction", prediction),
            ]
            for name, tensor in diagnostics:
                check_nan_inf(tensor, name)
            logging.info("[LLMMIRec] First-batch NaN/Inf check passed for all tensors.")

        # ---- 10. Output ----
        if return_intermediate:
            out_dict["interest_vectors"] = interest_vectors
            out_dict["attention_maps"] = attention_maps
            out_dict["interest_weights"] = interest_weights
            out_dict["user_vector"] = user_vector
            out_dict["history_vectors"] = history_emb_raw
            out_dict["candidate_vectors"] = candidate_emb
            if aspcf_comps is not None:
                out_dict.update(aspcf_comps)

        return out_dict

    # =========================
    #  Loss
    # =========================

    def loss(self, out_dict: dict):
        bpr_loss = super().loss(out_dict)
        total = bpr_loss

        # Relation preservation loss (ASPCF only)
        if "_relation_ids" in out_dict and self.lambda_relation > 0:
            rel_loss = self._compute_relation_loss(out_dict["_relation_ids"])
            total = total + self.lambda_relation * rel_loss
            out_dict["loss_relation"] = rel_loss.detach()

        return total

    def _compute_relation_loss(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Semantic relation preservation: KL(teacher || student).

        Teacher: frozen z_high (first semantic_rank PCA dims).
        Student: semantic branch output s.
        Both produce item-item cosine similarity matrices; we enforce the
        student's similarity structure to match the teacher's.
        """
        M = item_ids.numel()
        if M < 2:
            return torch.zeros([], device=item_ids.device)

        # Teacher: frozen z_high
        z = self.item_encoder.llm_table[item_ids]           # [M, d_llm]
        teacher = z[:, :self.semantic_rank]                  # [M, semantic_rank]

        # Student: semantic branch output
        s = self.item_encoder.semantic_branch(teacher)       # [M, semantic_dim]

        # L2 normalize
        teacher = F.normalize(teacher, dim=-1, eps=1e-8)
        student = F.normalize(s, dim=-1, eps=1e-8)

        # Item-item cosine similarity matrices
        teacher_sim = teacher @ teacher.t() / self.relation_teacher_temp   # [M, M]
        student_sim = student @ student.t() / self.relation_student_temp   # [M, M]

        # Remove diagonal
        diag_mask = ~torch.eye(M, dtype=torch.bool, device=item_ids.device)
        teacher_sim = teacher_sim[diag_mask].view(M, M - 1)    # [M, M-1]
        student_sim = student_sim[diag_mask].view(M, M - 1)    # [M, M-1]

        # Row-wise softmax
        teacher_dist = F.softmax(teacher_sim, dim=-1)          # [M, M-1]
        student_log = F.log_softmax(student_sim, dim=-1)       # [M, M-1]

        # KL(teacher || student)
        kl = F.kl_div(student_log, teacher_dist.detach(), reduction="batchmean")

        return kl
