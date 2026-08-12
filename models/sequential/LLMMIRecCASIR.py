# -*- coding: UTF-8 -*-
"""
LLMMIRecCASIR — Chapter 4: Collaborative-Anchored Semantic Interest Refinement.

Keeps ASPCF + collaborative multi-interest routing unchanged.
Uses detached attention as anchor to build interest-specific semantic
representation from ASPCF semantic component, and adds only the semantic
residual not already expressed by the collaborative interest.

Refine modes:
  none                  : pure ASPCF baseline
  semantic_add          : V_refined = V + gamma * S_interest
  complement            : V_refined = V + gamma * S_residual (orthogonal to V)
  complement_coherence  : V_refined = V + gamma * q * S_residual (coherence-gated)
"""

import logging
import math

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


class LLMMIRecCASIR(SequentialModel):
    reader = "SeqReader"
    runner = "BaseRunner"

    extra_log_args = [
        "emb_size", "K", "item_encoder", "adapter_hidden",
        "adapter_activation", "adapter_use_ln",
        "semantic_rank", "lambda_relation", "aspcf_gate_mode",
        "interest_refine_mode",
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

        # CASIR semantic refinement
        parser.add_argument("--interest_refine_mode", type=str, default="none",
                           choices=["none", "semantic_add", "complement",
                                    "complement_coherence"])
        parser.add_argument("--semantic_refine_gamma_init", type=float, default=0.1)
        parser.add_argument("--semantic_refine_gamma_max", type=float, default=0.5)

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

        # CASIR
        self.interest_refine_mode = str(getattr(args, "interest_refine_mode", "none"))
        self.refine_gamma_init = float(getattr(args, "semantic_refine_gamma_init", 0.1))
        self.refine_gamma_max = float(getattr(args, "semantic_refine_gamma_max", 0.5))

        self.dropout_p = float(getattr(args, "dropout", 0.1))

        llm_table = None
        if self.item_encoder_mode in ("llm_replace", "residual", "aspcf"):
            llm_table = load_llm_table(self.llm_emb_path, expected_rows=self.item_num)

        self._define_params(llm_table)
        self.apply(self.init_weights)
        self._first_batch_checked = False

        logging.info(f"[CASIR] initialized: enc={self.item_encoder_mode} K={self.K} "
                     f"refine={self.interest_refine_mode} gamma_init={self.refine_gamma_init} "
                     f"gamma_max={self.refine_gamma_max}")
        logging.info(f"[CASIR] #params: {self.count_variables()}")

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

        # CASIR semantic interest adapter + bounded learnable gamma
        if self.interest_refine_mode != "none":
            self.semantic_interest_adapter = nn.Linear(self.semantic_dim, self.emb_size)
            # gamma = gamma_max * sigmoid(raw_gamma), init at gamma_init
            ratio = self.refine_gamma_init / self.refine_gamma_max
            ratio = max(min(ratio, 0.999), 0.001)
            raw_init = math.log(ratio / (1.0 - ratio))
            self.raw_refine_gamma = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

    def _refine_gamma_value(self):
        return self.refine_gamma_max * torch.sigmoid(self.raw_refine_gamma)

    # ========================= Forward =========================

    def forward(self, feed_dict, return_intermediate=False):
        history = feed_dict["history_items"]   # [B, L]
        lengths = feed_dict["lengths"]          # [B]
        i_ids = feed_dict["item_id"]            # [B, 1+N]
        B, L = history.shape
        device = history.device

        refine_on = self.interest_refine_mode != "none"
        aspcf_comps = None

        # 1. Item embeddings
        if refine_on and self.item_encoder_mode == "aspcf":
            hist_out = self.item_encoder(history, return_components=True)
            cand_out = self.item_encoder(i_ids, return_components=True)
            history_emb_raw = hist_out["emb"]
            candidate_emb = cand_out["emb"]
            if return_intermediate:
                aspcf_comps = {
                    "history_semantic": hist_out["semantic"],
                    "history_complement": hist_out["complement"],
                    "candidate_semantic": cand_out["semantic"],
                    "candidate_complement": cand_out["complement"],
                }
        else:
            history_emb_raw = self.item_encoder(history)
            candidate_emb = self.item_encoder(i_ids)

        # 2. Position encoding
        valid_his = (history > 0).long()
        len_range = torch.arange(self.max_his, device=device)
        position = (lengths[:, None] - len_range[None, :L]) * valid_his
        history_emb_pos = history_emb_raw + self.position_emb(position)
        history_emb_pos = self.dropout(history_emb_pos)

        # 3. Multi-interest extraction
        interest_vectors, attention_maps = self.extractor(history_emb_pos, lengths)
        interest_vectors = self.dropout(interest_vectors)
        V = interest_vectors  # [B, K, D]

        # 4. CASIR semantic refinement
        refine_info = {}
        if refine_on:
            S = hist_out["semantic"]                      # [B, L, 32]
            A_anchor = attention_maps.detach()            # [B, K, L] (no grad to extractor routing)
            Z_sem = torch.bmm(A_anchor, S)                # [B, K, 32]
            S_interest = self.semantic_interest_adapter(Z_sem)  # [B, K, 64]
            gamma = self._refine_gamma_value()            # scalar

            if self.interest_refine_mode == "semantic_add":
                V_refined = V + gamma * S_interest
                S_residual = None
                q = None

            elif self.interest_refine_mode == "complement":
                V_anchor = V.detach()
                v_hat = F.normalize(V_anchor, dim=-1, eps=1e-8)       # [B, K, 64]
                S_parallel = (S_interest * v_hat).sum(dim=-1, keepdim=True) * v_hat
                S_residual = S_interest - S_parallel                   # [B, K, 64]
                V_refined = V + gamma * S_residual
                q = None

            elif self.interest_refine_mode == "complement_coherence":
                V_anchor = V.detach()
                v_hat = F.normalize(V_anchor, dim=-1, eps=1e-8)
                S_parallel = (S_interest * v_hat).sum(dim=-1, keepdim=True) * v_hat
                S_residual = S_interest - S_parallel
                S_item_norm = F.normalize(S.detach(), dim=-1, eps=1e-8)  # [B, L, 32]
                q = (torch.bmm(A_anchor, S_item_norm)).norm(dim=-1).clamp(0.0, 1.0)  # [B, K]
                V_refined = V + gamma * q.unsqueeze(-1) * S_residual
            else:
                V_refined = V

            refine_info = {
                "original_interest_vectors": V,
                "semantic_interest_vectors": S_interest,
                "semantic_residual_vectors": S_residual,
                "semantic_coherence": q,
                "refined_interest_vectors": V_refined,
                "semantic_refine_gamma": gamma,
            }
        else:
            V_refined = V

        # 5-7. Aggregation, user vector, prediction
        interest_weights = self.aggregator(history_emb_raw, lengths)
        user_vector = (V_refined * interest_weights[:, :, None]).sum(dim=1)
        prediction = (user_vector[:, None, :] * candidate_emb).sum(dim=-1)

        # 8. Relation loss stashing
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
                            ("interest_vectors", V_refined),
                            ("interest_weights", interest_weights),
                            ("candidate_vectors", candidate_emb),
                            ("prediction", prediction)]:
                check_nan_inf(t, name)
            logging.info("[CASIR] First-batch NaN/Inf check passed.")

        # 10. Output
        if return_intermediate:
            out_dict["interest_vectors"] = V_refined
            out_dict["attention_maps"] = attention_maps
            out_dict["interest_weights"] = interest_weights
            out_dict["user_vector"] = user_vector
            out_dict["history_vectors"] = history_emb_raw
            out_dict["candidate_vectors"] = candidate_emb
            if aspcf_comps is not None:
                out_dict.update(aspcf_comps)
            if refine_info:
                out_dict.update(refine_info)

        return out_dict

    # ========================= Loss =========================

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
