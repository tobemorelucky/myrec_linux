# -*- coding: UTF-8 -*-
"""
LLMMIRecHSDIR — Chapter 4: Hierarchical Semantic Distillation for Interest Routing.

Training-only hierarchical semantic routing supervision.
Recommendation backbone: ASPCF + relation + learnable queries (same as Ch.3).

HSDIR adds NO new parameters at inference time.
The LLM semantic structure acts only as a teacher during training.
"""

import logging
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


class LLMMIRecHSDIR(SequentialModel):
    reader = "SeqReader"
    runner = "BaseRunner"

    extra_log_args = [
        "emb_size", "K", "item_encoder", "adapter_hidden",
        "adapter_activation", "adapter_use_ln",
        "semantic_rank", "lambda_relation", "aspcf_gate_mode",
        "lambda_hsr", "hsr_teacher_mode",
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

        # Relation loss (same as ASPCF)
        parser.add_argument("--lambda_relation", type=float, default=0.01)
        parser.add_argument("--relation_sample_size", type=int, default=128)
        parser.add_argument("--relation_teacher_temp", type=float, default=0.1)
        parser.add_argument("--relation_student_temp", type=float, default=0.1)

        # HSDIR
        parser.add_argument("--lambda_hsr", type=float, default=0.0)
        parser.add_argument("--hsr_teacher_mode", type=str, default="hierarchical",
                           choices=["fine", "coarse", "hierarchical"])
        parser.add_argument("--hsr_student_temp", type=float, default=1.0)
        parser.add_argument("--teacher_path", type=str, default="")

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

        self.lambda_hsr = float(getattr(args, "lambda_hsr", 0.0))
        self.hsr_teacher_mode = str(getattr(args, "hsr_teacher_mode", "hierarchical"))
        self.hsr_student_temp = float(getattr(args, "hsr_student_temp", 1.0))
        self.teacher_path = str(getattr(args, "teacher_path", ""))

        self.dropout_p = float(getattr(args, "dropout", 0.1))

        llm_table = None
        if self.item_encoder_mode in ("llm_replace", "residual", "aspcf"):
            llm_table = load_llm_table(self.llm_emb_path, expected_rows=self.item_num)

        # Hierarchical teacher (training-only, frozen)
        if self.lambda_hsr > 0:
            if not self.teacher_path:
                raise ValueError("lambda_hsr>0 requires --teacher_path")
            teacher_data = pickle.load(open(self.teacher_path, "rb"))
            self.register_buffer("t_fine_assign",
                torch.tensor(teacher_data["fine_assignments"], dtype=torch.float32),
                persistent=False)
            self.register_buffer("t_coarse_assign",
                torch.tensor(teacher_data["coarse_assignments"], dtype=torch.float32),
                persistent=False)
            logging.info(f"[HSDIR] Teacher loaded: fine={self.t_fine_assign.shape}, "
                         f"coarse={self.t_coarse_assign.shape}")

        self._define_params(llm_table)
        self.apply(self.init_weights)
        self._first_batch_checked = False

        logging.info(f"[HSDIR] initialized: enc={self.item_encoder_mode} K={self.K} "
                     f"lambda_relation={self.lambda_relation} lambda_hsr={self.lambda_hsr} "
                     f"teacher_mode={self.hsr_teacher_mode}")
        logging.info(f"[HSDIR] #params: {self.count_variables()}")

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

    # ========================= Forward =========================

    def forward(self, feed_dict, return_intermediate=False):
        history = feed_dict["history_items"]   # [B, L]
        lengths = feed_dict["lengths"]          # [B]
        i_ids = feed_dict["item_id"]            # [B, 1+N]
        B, L = history.shape
        device = history.device

        # 1. Item embeddings
        need_comp = return_intermediate and self.item_encoder_mode == "aspcf"
        aspcf_comps = None
        if need_comp:
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
            history_emb_raw = self.item_encoder(history)
            candidate_emb = self.item_encoder(i_ids)

        # 2. Position encoding
        valid_his = (history > 0).long()
        len_range = torch.arange(self.max_his, device=device)
        position = (lengths[:, None] - len_range[None, :L]) * valid_his
        history_emb_pos = history_emb_raw + self.position_emb(position)
        history_emb_pos = self.dropout(history_emb_pos)

        # 3. Multi-interest extraction (learnable queries)
        route_scores = None  # raw scores for HSDIR
        need_route = (self.training and self.lambda_hsr > 0)
        extractor_out = self.extractor(
            history_emb_pos, lengths, return_route_scores=need_route)
        if need_route:
            interest_vectors, attention_maps, route_scores = extractor_out
        else:
            interest_vectors, attention_maps = extractor_out
        interest_vectors = self.dropout(interest_vectors)

        # 4-6. Aggregation, user vector, prediction
        interest_weights = self.aggregator(history_emb_raw, lengths)
        user_vector = (interest_vectors * interest_weights[:, :, None]).sum(dim=1)
        prediction = (user_vector[:, None, :] * candidate_emb).sum(dim=-1)

        # 7. Relation loss stashing
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

        # 8. HSDIR stashing (training-only)
        if need_route:
            out_dict["_hsdr_route_scores"] = route_scores       # [B, K, L]
            out_dict["_hsdr_history_ids"] = history              # [B, L]

        # 9. NaN/Inf check
        if not self._first_batch_checked:
            self._first_batch_checked = True
            for name, t in [("history_vectors", history_emb_raw),
                            ("interest_vectors", interest_vectors),
                            ("interest_weights", interest_weights),
                            ("candidate_vectors", candidate_emb),
                            ("prediction", prediction)]:
                check_nan_inf(t, name)
            logging.info("[HSDIR] First-batch NaN/Inf check passed.")

        # 10. Output
        if return_intermediate:
            out_dict["interest_vectors"] = interest_vectors
            out_dict["attention_maps"] = attention_maps
            out_dict["interest_weights"] = interest_weights
            out_dict["user_vector"] = user_vector
            out_dict["history_vectors"] = history_emb_raw
            out_dict["candidate_vectors"] = candidate_emb
            if aspcf_comps is not None:
                out_dict.update(aspcf_comps)
            if route_scores is not None:
                # Student route membership
                R = F.softmax(
                    route_scores.transpose(1, 2) / self.hsr_student_temp, dim=-1)  # [B,L,K]
                G_route = R @ R.transpose(-1, -2)  # [B,L,L]  permutation-invariant
                out_dict["route_scores"] = route_scores
                out_dict["route_membership"] = R
                out_dict["route_comembership"] = G_route
                # Teacher relations (frozen lookup)
                fine = self.t_fine_assign[history]      # [B, L, 32]
                coarse = self.t_coarse_assign[history]   # [B, L, 8]
                out_dict["teacher_fine_relation"] = fine @ fine.transpose(-1, -2)
                out_dict["teacher_coarse_relation"] = coarse @ coarse.transpose(-1, -2)

        return out_dict

    # ========================= Loss =========================

    def loss(self, out_dict: dict):
        total = super().loss(out_dict)

        # ASPCF relation loss
        if "_relation_ids" in out_dict and self.lambda_relation > 0:
            rel = self._compute_relation_loss(out_dict["_relation_ids"])
            total = total + self.lambda_relation * rel
            out_dict["loss_relation"] = rel.detach()

        # HSDIR loss (training-only)
        if "_hsdr_route_scores" in out_dict and self.lambda_hsr > 0:
            L_coh, L_sep = self._compute_hsdr_loss(
                out_dict["_hsdr_route_scores"],
                out_dict["_hsdr_history_ids"])
            hsr = L_coh + L_sep
            total = total + self.lambda_hsr * hsr
            out_dict["loss_hsr"] = hsr.detach()
            out_dict["loss_hsr_coh"] = L_coh.detach()
            out_dict["loss_hsr_sep"] = L_sep.detach()

        return total

    # ---------- ASPCF relation loss (unchanged) ----------

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

    # ---------- HSDIR loss ----------

    def _compute_hsdr_loss(self, route_scores, history_ids):
        """Hierarchical Semantic Distillation routing loss.

        Student: route_membership R from raw attention scores.
        Teacher: frozen fine/coarse semantic assignments → G_fine, G_coarse.

        ASPCF relation: item-level representation preservation.
        HSDIR:           history-level interest co-membership supervision.
        HSDIR uses ONLY history_items (no candidate/target leakage).
        """
        B, K, L = route_scores.shape
        device = route_scores.device

        # Student: route membership [B, L, K]
        R = F.softmax(route_scores.transpose(1, 2) / self.hsr_student_temp, dim=-1)

        # Student co-membership: G_route = R @ R^T [B, L, L]
        # Permutation-invariant: independent of interest slot numbering.
        G_route = (R @ R.transpose(-1, -2)).clamp(1e-6, 1 - 1e-6)

        # Teacher: frozen semantic relation graphs
        fine = self.t_fine_assign[history_ids]     # [B, L, 32]
        coarse = self.t_coarse_assign[history_ids]  # [B, L, 8]
        G_fine = fine @ fine.transpose(-1, -2)      # [B, L, L]
        G_coarse = coarse @ coarse.transpose(-1, -2)  # [B, L, L]

        # Valid pairs: non-padding, off-diagonal
        valid = (history_ids > 0).float()  # [B, L]
        valid_pair = valid.unsqueeze(-1) * valid.unsqueeze(-2)  # [B, L, L]
        diag = torch.eye(L, dtype=torch.bool, device=device).unsqueeze(0)
        valid_pair = valid_pair * (~diag).float()

        # Confidence weights
        if self.hsr_teacher_mode == "hierarchical":
            W_pos = G_fine          # fine-level semantic agreement
            W_neg = 1.0 - G_coarse  # coarse-level semantic disagreement
        elif self.hsr_teacher_mode == "fine":
            W_pos = G_fine
            W_neg = 1.0 - G_fine
        else:  # coarse
            W_pos = G_coarse
            W_neg = 1.0 - G_coarse

        # Cohesion: same-interest pairs should have high G_route
        pos_sum = (valid_pair * W_pos).sum()
        if pos_sum > 1e-8:
            L_coh = -(valid_pair * W_pos * torch.log(G_route)).sum() / pos_sum
        else:
            L_coh = torch.zeros([], device=device)

        # Separation: different-interest pairs should have low G_route
        neg_sum = (valid_pair * W_neg).sum()
        if neg_sum > 1e-8:
            L_sep = -(valid_pair * W_neg * torch.log(1.0 - G_route)).sum() / neg_sum
        else:
            L_sep = torch.zeros([], device=device)

        return L_coh, L_sep
