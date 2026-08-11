# -*- coding: UTF-8 -*-
"""
LLMMIRecHSDIR — Chapter 4: Hierarchical Semantic Distillation for Interest Routing.

Training-only hierarchical semantic routing supervision.
Recommendation backbone: ASPCF + relation + learnable queries (same as Ch.3).

HSDIR adds NO new parameters at inference time.
The LLM semantic structure acts only as a teacher during training.
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


class LLMMIRecHSDIR(SequentialModel):
    reader = "SeqReader"
    runner = "BaseRunner"

    extra_log_args = [
        "emb_size", "K", "item_encoder", "adapter_hidden",
        "adapter_activation", "adapter_use_ln",
        "semantic_rank", "lambda_relation", "aspcf_gate_mode",
        "lambda_hsr", "hsr_teacher_mode", "hsr_loss_mode",
        "hsr_confidence_mode", "hsr_route_source", "aggregation_mode",
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
        parser.add_argument("--hsr_loss_mode", type=str, default="absolute",
                           choices=["absolute", "relative", "pair_selective"])
        parser.add_argument("--hsr_margin", type=float, default=0.1)
        parser.add_argument("--hsr_pair_margin", type=float, default=0.1,
                           help="Margin for pair_selective HSR loss")
        parser.add_argument("--hsr_confidence_mode", type=str, default="semantic",
                           choices=["semantic", "agreement"])
        parser.add_argument("--hsr_route_source", type=str, default="raw",
                           choices=["raw", "attention_contribution"])
        parser.add_argument("--teacher_path", type=str, default="")

        # Aggregation calibration
        parser.add_argument("--aggregation_mode", type=str, default="base",
                           choices=["base", "support_confidence"])
        parser.add_argument("--support_beta", type=float, default=1.0)

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
        self.hsr_loss_mode = str(getattr(args, "hsr_loss_mode", "absolute"))
        self.hsr_margin = float(getattr(args, "hsr_margin", 0.1))
        self.hsr_pair_margin = float(getattr(args, "hsr_pair_margin", 0.1))
        self.hsr_confidence_mode = str(getattr(args, "hsr_confidence_mode", "semantic"))
        self.teacher_path = str(getattr(args, "teacher_path", ""))
        self.hsr_route_source = str(getattr(args, "hsr_route_source", "raw"))
        self.aggregation_mode = str(getattr(args, "aggregation_mode", "base"))
        self.support_beta = float(getattr(args, "support_beta", 1.0))

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
                     f"teacher_mode={self.hsr_teacher_mode} "
                     f"confidence={self.hsr_confidence_mode} "
                     f"loss={self.hsr_loss_mode} "
                     f"agg={self.aggregation_mode}")
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
        need_route = ((self.training and self.lambda_hsr > 0) or return_intermediate
                      or self.aggregation_mode == "support_confidence")
        extractor_out = self.extractor(
            history_emb_pos, lengths, return_route_scores=need_route)
        if need_route:
            interest_vectors, attention_maps, route_scores = extractor_out
        else:
            interest_vectors, attention_maps = extractor_out
        interest_vectors = self.dropout(interest_vectors)

        # 4-6. Aggregation, user vector, prediction
        base_weights = self.aggregator(history_emb_raw, lengths)  # [B, K]
        support_dist = None
        routing_conf = None

        if self.aggregation_mode == "support_confidence" and route_scores is not None:
            # R: route membership [B, L, K]
            valid_h = (history > 0).float().unsqueeze(-1)
            R = F.softmax(route_scores.transpose(1, 2) / self.hsr_student_temp, dim=-1)
            R = R * valid_h

            # Support distribution over K interests
            support = R.sum(dim=1)                                        # [B, K]
            support = support / support.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            support_dist = support

            # Routing confidence: 1 - mean normalized entropy over valid items
            eps = 1e-8
            ent = -(R * torch.log(R + eps)).sum(dim=-1)                   # [B, L]
            logK = math.log(self.K)
            norm_ent = (ent / logK) * valid_h.squeeze(-1)                 # [B, L]
            count = valid_h.squeeze(-1).sum(dim=-1).clamp(min=1)          # [B]
            mean_ent = norm_ent.sum(dim=-1) / count                       # [B]
            routing_conf = (1.0 - mean_ent).unsqueeze(-1)                 # [B, 1]

            # Calibrate: boost interests with higher support, gated by confidence
            calibrated = base_weights * (support + eps).pow(
                self.support_beta * routing_conf)
            interest_weights = calibrated / calibrated.sum(dim=-1, keepdim=True).clamp_min(eps)
        else:
            interest_weights = base_weights

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
            if self.hsr_route_source == "attention_contribution":
                out_dict["_hsdr_attention_maps"] = attention_maps  # [B, K, L]
            else:
                out_dict["_hsdr_route_scores"] = route_scores     # [B, K, L]
            out_dict["_hsdr_history_ids"] = history                # [B, L]
            if self.hsr_confidence_mode == "agreement":
                out_dict["_hsdr_history_emb"] = history_emb_raw    # [B, L, D] (no pos)

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
            out_dict["base_interest_weights"] = base_weights
            if support_dist is not None:
                out_dict["support_distribution"] = support_dist
                out_dict["routing_confidence"] = routing_conf.squeeze(-1)
            out_dict["user_vector"] = user_vector
            out_dict["history_vectors"] = history_emb_raw
            out_dict["candidate_vectors"] = candidate_emb
            if aspcf_comps is not None:
                out_dict.update(aspcf_comps)
            if route_scores is not None or (
                    self.hsr_route_source == "attention_contribution" and need_route):
                # Student membership for HSDIR
                valid_h = (history > 0).float().unsqueeze(-1)
                if self.hsr_route_source == "attention_contribution":
                    C = attention_maps.transpose(1, 2)  # [B, L, K]
                    C = C * valid_h
                    C = C / C.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    hsr_membership = C
                else:
                    R = F.softmax(
                        route_scores.transpose(1, 2) / self.hsr_student_temp, dim=-1)
                    hsr_membership = R * valid_h
                hsr_comembership = hsr_membership @ hsr_membership.transpose(-1, -2)
                out_dict["route_scores"] = route_scores
                out_dict["route_membership"] = hsr_membership
                out_dict["route_comembership"] = hsr_comembership
                out_dict["hsr_student_membership"] = hsr_membership
                out_dict["hsr_student_comembership"] = hsr_comembership
                # Teacher relations (if teacher is loaded)
                if hasattr(self, "t_fine_assign"):
                    fine = self.t_fine_assign[history]
                    coarse = self.t_coarse_assign[history]
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
        has_hsr = (("_hsdr_route_scores" in out_dict or "_hsdr_attention_maps" in out_dict)
                    and self.lambda_hsr > 0)
        if has_hsr:
            L_coh, L_sep, hsr_gap = self._compute_hsdr_loss(
                out_dict.get("_hsdr_route_scores", None),
                out_dict["_hsdr_history_ids"],
                out_dict.get("_hsdr_history_emb", None),
                out_dict.get("_hsdr_attention_maps", None))
            if self.hsr_loss_mode == "relative":
                if not torch.isfinite(L_coh):
                    raise RuntimeError(f"[HSDIR] Non-finite relative HSR loss: {L_coh}")
                hsr = L_coh
                total = total + self.lambda_hsr * hsr
                out_dict["loss_hsr"] = hsr.detach()
                if hsr_gap is not None:
                    out_dict["loss_hsr_gap"] = hsr_gap.detach()
            elif self.hsr_loss_mode == "pair_selective":
                if not torch.isfinite(L_coh):
                    raise RuntimeError(f"[HSDIR] Non-finite pair_selective HSR loss: {L_coh}")
                hsr = L_coh
                total = total + self.lambda_hsr * hsr
                out_dict["loss_hsr"] = hsr.detach()
                if hsr_gap is not None:
                    out_dict["loss_hsr_gap"] = hsr_gap.detach()
            else:
                if not (torch.isfinite(L_coh) and torch.isfinite(L_sep)):
                    raise RuntimeError(
                        f"[HSDIR] Non-finite HSR loss! L_coh={L_coh:.4f} L_sep={L_sep:.4f}")
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

    def _compute_hsdr_loss(self, route_scores, history_ids, history_emb=None,
                           attention_maps=None):
        """Hierarchical Semantic Distillation routing loss.

        Student:
          raw: route_membership R from raw attention scores.
          attention_contribution: C from attention_maps (true routing weights).
        Teacher: frozen fine/coarse semantic assignments → G_fine, G_coarse.

        ASPCF relation: item-level representation preservation.
        HSDIR:           history-level interest co-membership supervision.
        HSDIR uses ONLY history_items (no candidate/target leakage).
        """
        if route_scores is not None:
            B, K, L = route_scores.shape
            device = route_scores.device
        else:
            B, K, L = attention_maps.shape
            device = attention_maps.device

        valid_mask = (history_ids > 0).float().unsqueeze(-1)  # [B, L, 1]

        # Student membership
        if self.hsr_route_source == "attention_contribution":
            C = attention_maps.transpose(1, 2)               # [B, L, K]
            C = C * valid_mask
            C = C / C.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            student_membership = C
        else:
            R = F.softmax(route_scores.transpose(1, 2) / self.hsr_student_temp, dim=-1)
            student_membership = R * valid_mask

        # Student co-membership
        G_route = student_membership @ student_membership.transpose(-1, -2)
        G_route = G_route.clamp(1e-6, 1 - 1e-6)

        # Safety
        if not torch.isfinite(G_route).all():
            raise RuntimeError(
                f"[HSDIR] G_route contains NaN/Inf! "
                f"student: nan={torch.isnan(student_membership).any().item()} "
                f"inf={torch.isinf(student_membership).any().item()}"
            )

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

        # Confidence weights (from hierarchical teacher)
        if self.hsr_teacher_mode == "hierarchical":
            W_pos_sem = G_fine
            W_neg_sem = 1.0 - G_coarse
        elif self.hsr_teacher_mode == "fine":
            W_pos_sem = G_fine
            W_neg_sem = 1.0 - G_fine
        else:  # coarse
            W_pos_sem = G_coarse
            W_neg_sem = 1.0 - G_coarse

        # Collaborative agreement modulation
        if self.hsr_confidence_mode == "agreement" and history_emb is not None:
            # Collaborative cosine similarity C ∈ [0,1] (detached — no grad to ASPCF)
            he_detach = history_emb.detach()
            he_norm = F.normalize(he_detach, dim=-1, eps=1e-8)
            C = (he_norm @ he_norm.transpose(-1, -2))     # [B, L, L], in [-1, 1]
            C = (C + 1.0) / 2.0                            # [0, 1]
            W_pos = W_pos_sem * C
            W_neg = W_neg_sem * (1.0 - C)
        else:
            W_pos = W_pos_sem
            W_neg = W_neg_sem

        if self.hsr_loss_mode == "pair_selective":
            # Anchor-level selective positive/negative pair ranking.
            M = self.hsr_pair_margin
            valid_anchor_mask = valid_pair.sum(dim=-1) > 0  # [B, L]

            pos_scores = G_fine.clone()
            pos_scores[valid_pair == 0] = float("-inf")
            p_idx = pos_scores.argmax(dim=-1)  # [B, L]

            batch_idx = torch.arange(B, device=device)[:, None].expand(-1, L)
            L_arange = torch.arange(L, device=device).unsqueeze(0)
            neg_scores = G_coarse.clone()
            neg_scores[valid_pair == 0] = float("inf")
            neg_scores[batch_idx, L_arange, p_idx] = float("inf")
            n_idx = neg_scores.argmin(dim=-1)  # [B, L]

            pos_valid = valid_pair[batch_idx, L_arange, p_idx].bool()
            neg_valid = valid_pair[batch_idx, L_arange, n_idx].bool()
            anchor_ok = valid_anchor_mask & pos_valid & neg_valid & (p_idx != n_idx)

            if anchor_ok.sum() < 1:
                return torch.zeros([], device=device), torch.zeros([], device=device), None

            c_i = G_fine[batch_idx, L_arange, p_idx] * \
                  (1.0 - G_coarse[batch_idx, L_arange, n_idx])
            c_i = c_i.detach()

            r_pos = G_route[batch_idx, L_arange, p_idx]
            r_neg = G_route[batch_idx, L_arange, n_idx]

            loss_per_anchor = c_i * F.relu(M - r_pos + r_neg)
            total_c = (c_i * anchor_ok.float()).sum().clamp(min=1e-8)
            L_hsr = (loss_per_anchor * anchor_ok.float()).sum() / total_c

            gap_val = (r_pos - r_neg)[anchor_ok].detach().mean()
            return L_hsr, torch.zeros([], device=device), gap_val

        if self.hsr_loss_mode == "relative":
            # Per-user g_pos and g_neg
            pos_num = (valid_pair * W_pos * G_route).sum(dim=[1, 2])    # [B]
            pos_den = (valid_pair * W_pos).sum(dim=[1, 2]).clamp(min=1e-8)
            g_pos = pos_num / pos_den

            neg_num = (valid_pair * W_neg * G_route).sum(dim=[1, 2])
            neg_den = (valid_pair * W_neg).sum(dim=[1, 2]).clamp(min=1e-8)
            g_neg = neg_num / neg_den

            valid_user = (pos_den > 1e-8) & (neg_den > 1e-8)
            if valid_user.sum() < 1:
                return torch.zeros([], device=device), torch.zeros([], device=device), None

            gap = (g_pos - g_neg)[valid_user]  # [B']
            relu_gap = F.relu(self.hsr_margin - gap)
            L_hsr = (relu_gap / self.hsr_margin).mean()
            return L_hsr, torch.zeros([], device=device), gap.detach().mean()

        # --- absolute mode (original) ---
        pos_sum = (valid_pair * W_pos).sum()
        if pos_sum > 1e-8:
            L_coh = -(valid_pair * W_pos * torch.log(G_route)).sum() / pos_sum
        else:
            L_coh = torch.zeros([], device=device)

        neg_sum = (valid_pair * W_neg).sum()
        if neg_sum > 1e-8:
            L_sep = -(valid_pair * W_neg * torch.log(1.0 - G_route)).sum() / neg_sum
        else:
            L_sep = torch.zeros([], device=device)

        return L_coh, L_sep, None
