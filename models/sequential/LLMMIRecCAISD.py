# -*- coding: UTF-8 -*-
"""
LLMMIRecCAISD — Collaborative-Anchored Interest Semantic Distillation.

Keeps ASPCF + collaborative multi-interest attention + InterestAggregator
and the recommendation path completely unchanged.

Frozen LLM semantic prototype assignments act as a training-only teacher:
each collaborative interest gets a dynamic semantic profile T built from
detached attention, and the interest vector learns to predict T via a
student head. The student head never participates in recommendation.
"""

import logging
import math
import pickle

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


class LLMMIRecCAISD(SequentialModel):
    reader = "SeqReader"
    runner = "BaseRunner"

    extra_log_args = [
        "emb_size", "K", "item_encoder", "adapter_hidden",
        "adapter_activation", "adapter_use_ln",
        "semantic_rank", "lambda_relation", "aspcf_gate_mode",
        "semantic_distill_mode", "lambda_interest_semantic",
        "semantic_teacher_mode",
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

        # CAISD semantic distillation
        parser.add_argument("--semantic_teacher_path", type=str, default="",
                           help="Path to llmmi_proto32_sr512.pkl")
        parser.add_argument("--semantic_distill_mode", type=str, default="none",
                           choices=["none", "uniform", "confidence"])
        parser.add_argument("--semantic_teacher_mode", type=str, default="attention",
                           choices=["attention", "responsibility"])
        parser.add_argument("--lambda_interest_semantic", type=float, default=0.01)

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

        self.semantic_teacher_path = str(getattr(args, "semantic_teacher_path", ""))
        self.semantic_distill_mode = str(getattr(args, "semantic_distill_mode", "none"))
        self.semantic_teacher_mode = str(getattr(args, "semantic_teacher_mode", "attention"))
        self.lambda_interest_semantic = float(getattr(args, "lambda_interest_semantic", 0.01))

        self.dropout_p = float(getattr(args, "dropout", 0.1))

        llm_table = None
        if self.item_encoder_mode in ("llm_replace", "residual", "aspcf"):
            llm_table = load_llm_table(self.llm_emb_path, expected_rows=self.item_num)

        # Frozen semantic teacher (prototype assignments)
        if self.semantic_distill_mode != "none":
            if not self.semantic_teacher_path:
                raise ValueError("semantic_distill_mode != none requires --semantic_teacher_path")
            proto_data = pickle.load(open(self.semantic_teacher_path, "rb"))
            self.register_buffer("t_semantic_assign",
                torch.tensor(proto_data["soft_assignments"], dtype=torch.float32),
                persistent=False)
            logging.info(f"[CAISD] semantic teacher: {self.t_semantic_assign.shape}")

        self._define_params(llm_table)
        self.apply(self.init_weights)
        self._first_batch_checked = False

        logging.info(f"[CAISD] initialized: enc={self.item_encoder_mode} K={self.K} "
                     f"distill={self.semantic_distill_mode} teacher={self.semantic_teacher_mode} "
                     f"lambda_sem={self.lambda_interest_semantic}")
        logging.info(f"[CAISD] #params: {self.count_variables()}")

    def _define_params(self, llm_table):
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

        # Student semantic head (training-only)
        if self.semantic_distill_mode != "none":
            self.semantic_predictor = nn.Linear(self.emb_size, 32)

    # ========================= Forward =========================

    def forward(self, feed_dict, return_intermediate=False):
        history = feed_dict["history_items"]   # [B, L]
        lengths = feed_dict["lengths"]          # [B]
        i_ids = feed_dict["item_id"]            # [B, 1+N]
        B, L = history.shape
        device = history.device

        # 1. Item embeddings
        history_emb_raw = self.item_encoder(history)
        candidate_emb = self.item_encoder(i_ids)

        # 2. Position encoding
        valid_his = (history > 0).long()
        len_range = torch.arange(self.max_his, device=device)
        position = (lengths[:, None] - len_range[None, :L]) * valid_his
        history_emb_pos = history_emb_raw + self.position_emb(position)
        history_emb_pos = self.dropout(history_emb_pos)

        # 3. Multi-interest extraction (unchanged)
        interest_vectors, attention_maps = self.extractor(history_emb_pos, lengths)
        interest_vectors = self.dropout(interest_vectors)
        V = interest_vectors  # [B, K, D]

        # 4. Dynamic semantic profile (teacher, fully detached)
        # Only computed during training or explicit diagnostic pass.
        need_semantic_distill = (
            self.semantic_distill_mode != "none"
            and (self.training or return_intermediate)
        )
        distill_info = {}
        if need_semantic_distill:
            Q = self.t_semantic_assign[history]              # [B, L, 32]
            A_teacher = attention_maps.detach()              # [B, K, L]
            valid_h = (history > 0).float()                  # [B, L]

            responsibility_w = None
            if self.semantic_teacher_mode == "responsibility":
                # Cross-interest responsibility: R = A / Σ_k A
                A_masked = A_teacher * valid_h.unsqueeze(1)  # [B, K, L]
                R = A_masked / A_masked.sum(dim=1, keepdim=True).clamp_min(1e-8)  # [B, K, L]
                # Responsibility-aware attention, then normalize over history
                W = A_masked * R
                W = W / W.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                responsibility_w = W
                T = torch.bmm(W, Q)                          # [B, K, 32]
            else:
                T = torch.bmm(A_teacher, Q)                  # [B, K, 32]
            T = T / T.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            T = T.detach()

            P_logits = self.semantic_predictor(V)            # [B, K, 32]
            log_P = F.log_softmax(P_logits, dim=-1)

            kl = F.kl_div(log_P, T, reduction="none").sum(dim=-1)  # [B, K]

            if self.semantic_distill_mode == "confidence":
                eps = 1e-8
                H = -(T * torch.log(T + eps)).sum(dim=-1)    # [B, K]
                confidence = (1.0 - H / math.log(32)).clamp(0.0, 1.0).detach()
                L_sem = (confidence * kl).sum() / confidence.sum().clamp(min=1e-8)
            else:  # uniform
                confidence = torch.ones_like(kl)
                L_sem = kl.mean()

            distill_info = {
                "interest_semantic_teacher": T,
                "interest_semantic_prediction": torch.softmax(P_logits, dim=-1),
                "interest_semantic_confidence": confidence,
                "interest_semantic_kl": kl,
                "_sem_loss": L_sem,
            }
            if responsibility_w is not None:
                distill_info["interest_semantic_responsibility"] = responsibility_w

        # 5-7. Aggregation, user vector, prediction (UNCHANGED — no semantic injection)
        interest_weights = self.aggregator(history_emb_raw, lengths)
        user_vector = (V * interest_weights[:, :, None]).sum(dim=1)
        prediction = (user_vector[:, None, :] * candidate_emb).sum(dim=-1)

        # 8. Relation stashing + semantic loss stashing
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

        if distill_info and self.training:
            out_dict["_sem_loss"] = distill_info["_sem_loss"]

        # 9. NaN/Inf check
        if not self._first_batch_checked:
            self._first_batch_checked = True
            for name, t in [("history_vectors", history_emb_raw),
                            ("interest_vectors", V),
                            ("interest_weights", interest_weights),
                            ("candidate_vectors", candidate_emb),
                            ("prediction", prediction)]:
                check_nan_inf(t, name)
            logging.info("[CAISD] First-batch NaN/Inf check passed.")

        # 10. Output
        if return_intermediate:
            out_dict["interest_vectors"] = V
            out_dict["attention_maps"] = attention_maps
            out_dict["interest_weights"] = interest_weights
            out_dict["user_vector"] = user_vector
            out_dict["history_vectors"] = history_emb_raw
            out_dict["candidate_vectors"] = candidate_emb
            if distill_info:
                for k, v in distill_info.items():
                    if not k.startswith("_"):
                        out_dict[k] = v

        return out_dict

    # ========================= Loss =========================

    def loss(self, out_dict: dict):
        total = super().loss(out_dict)

        if "_relation_ids" in out_dict and self.lambda_relation > 0:
            rel = self._compute_relation_loss(out_dict["_relation_ids"])
            total = total + self.lambda_relation * rel
            out_dict["loss_relation"] = rel.detach()

        # CAISD semantic distillation (training-only)
        if "_sem_loss" in out_dict and self.lambda_interest_semantic > 0:
            L_sem = out_dict["_sem_loss"]
            total = total + self.lambda_interest_semantic * L_sem
            out_dict["loss_interest_semantic"] = L_sem.detach()

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
