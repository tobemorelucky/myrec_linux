# -*- coding: UTF-8 -*-
"""
LLMMIRecECTIR — Chapter 4: Evidence-Constrained Transport Interest Routing.

Core problem (see CHAPTER4_DESIGN.md §3): the baseline extractor applies an
independent softmax over history FOR EACH interest, so the K interests are
never jointly allocated and drift into redundant representations
(measured: interest pairwise cosine 0.87 / 0.79, effective rank 1.11 / 1.62
against an upper bound of K-1 = 3).

This model replaces the routing operator with a globally coupled entropic
optimal-transport assignment:

    Module 1  TransportInterestRouter      history -> K interests via Sinkhorn
    Module 2  sequence-specific refinement (<= 2 steps, inside the same module)
    Module 3  TransportEvidenceAggregator  interest weights from transport evidence

Chapter 3's ASPCF ItemEncoder is used UNCHANGED. "Frozen" refers to its
structure, not its gradients: this model is trained end-to-end exactly like
ASPCF and keeps lambda_relation = 0.01.

Training objective (unchanged, no Chapter 4 auxiliary loss):
    LOSS = BPR + lambda_relation * L_relation

`--ectir_mode baseline` reproduces ASPCF bit-for-bit (same module names, so an
ASPCF checkpoint loads directly); this is asserted in the unit tests.
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
    TransportInterestRouter,
    TransportEvidenceAggregator,
)

VALID_MODES = ("baseline", "transport", "transport_refine", "full")


class LLMMIRecECTIR(SequentialModel):
    reader = "SeqReader"
    runner = "BaseRunner"

    extra_log_args = [
        "emb_size", "K", "item_encoder", "semantic_rank", "lambda_relation",
        "aspcf_gate_mode", "ectir_mode", "ectir_eps", "ectir_tau_c",
        "ectir_n_sinkhorn", "ectir_n_refine", "ectir_evi_hidden",
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

        # ASPCF (unchanged; Chapter 3 frozen)
        parser.add_argument("--semantic_rank", type=int, default=512)
        parser.add_argument("--semantic_dim", type=int, default=32)
        parser.add_argument("--semantic_hidden", type=int, default=128)
        parser.add_argument("--complement_dim", type=int, default=32)
        parser.add_argument("--tail_hidden", type=int, default=64)
        parser.add_argument("--complement_hidden", type=int, default=64)
        parser.add_argument("--gate_hidden", type=int, default=64)
        parser.add_argument("--aspcf_gate_mode", type=str, default="basic",
                           choices=["basic", "conflict"])

        # relation loss (Chapter 3 existing; kept at 0.01)
        parser.add_argument("--lambda_relation", type=float, default=0.01)
        parser.add_argument("--relation_sample_size", type=int, default=128)
        parser.add_argument("--relation_teacher_temp", type=float, default=0.1)
        parser.add_argument("--relation_student_temp", type=float, default=0.1)

        # ---- Chapter 4: ECTIR ----
        parser.add_argument("--ectir_mode", type=str, default="full",
                           choices=list(VALID_MODES),
                           help="baseline = exact ASPCF; transport = Module 1; "
                                "transport_refine = M1+M2; full = M1+M2+M3")
        parser.add_argument("--ectir_eps", type=float, default=0.1,
                           help="entropic regularisation for Sinkhorn")
        parser.add_argument("--ectir_tau_c", type=float, default=1.0,
                           help="temperature for the evidence-derived capacity")
        parser.add_argument("--ectir_n_sinkhorn", type=int, default=5)
        parser.add_argument("--ectir_n_refine", type=int, default=1)
        parser.add_argument("--ectir_evi_hidden", type=int, default=64)

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

        self.ectir_mode = str(getattr(args, "ectir_mode", "full"))
        if self.ectir_mode not in VALID_MODES:
            raise ValueError(f"Unknown ectir_mode: {self.ectir_mode}")
        self.ectir_eps = float(getattr(args, "ectir_eps", 0.1))
        self.ectir_tau_c = float(getattr(args, "ectir_tau_c", 1.0))
        self.ectir_n_sinkhorn = int(getattr(args, "ectir_n_sinkhorn", 5))
        self.ectir_n_refine = int(getattr(args, "ectir_n_refine", 1))
        self.ectir_evi_hidden = int(getattr(args, "ectir_evi_hidden", 64))

        self.dropout_p = float(getattr(args, "dropout", 0.1))

        llm_table = None
        if self.item_encoder_mode in ("llm_replace", "residual", "aspcf"):
            llm_table = load_llm_table(self.llm_emb_path, expected_rows=self.item_num)

        self._define_params(llm_table)
        self.apply(self.init_weights)
        self._first_batch_checked = False

        logging.info(f"[ECTIR] initialized: enc={self.item_encoder_mode} K={self.K} "
                     f"mode={self.ectir_mode} eps={self.ectir_eps} tau_c={self.ectir_tau_c} "
                     f"n_sinkhorn={self.ectir_n_sinkhorn} n_refine={self.ectir_n_refine} "
                     f"lambda_relation={self.lambda_relation}")
        logging.info(f"[ECTIR] #params: {self.count_variables()}")

    def _uses_transport(self):
        return self.ectir_mode != "baseline"

    def _uses_evidence_aggregation(self):
        return self.ectir_mode == "full"

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

        if self._uses_transport():
            self.transport_router = TransportInterestRouter(
                K=self.K, emb_size=self.emb_size, attn_size=self.attn_size,
                eps=self.ectir_eps, tau_c=self.ectir_tau_c,
                n_sinkhorn=self.ectir_n_sinkhorn,
                n_refine=self.ectir_n_refine if self.ectir_mode != "transport" else 0,
            )
        else:
            # baseline: identical module name/type to LLMMIRecASPCF so that an
            # ASPCF state_dict loads with matching keys.
            self.extractor = QueryMultiInterestExtractor(
                K=self.K, emb_size=self.emb_size, attn_size=self.attn_size)

        if self._uses_evidence_aggregation():
            self.evidence_aggregator = TransportEvidenceAggregator(
                K=self.K, emb_size=self.emb_size, hidden=self.ectir_evi_hidden)
        else:
            self.aggregator = InterestAggregator(emb_size=self.emb_size, K=self.K)

        self.dropout = nn.Dropout(p=self.dropout_p)

    # ========================= Forward =========================

    def forward(self, feed_dict, return_intermediate=False):
        history = feed_dict["history_items"]
        lengths = feed_dict["lengths"]
        i_ids = feed_dict["item_id"]
        B, L = history.shape
        device = history.device

        # 1. Item embeddings (UNCHANGED: Chapter 3 ASPCF encoder)
        aspcf_comps = None
        if return_intermediate and self.item_encoder_mode == "aspcf":
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

        # 2. Position encoding + dropout (UNCHANGED)
        valid_his = (history > 0).long()
        len_range = torch.arange(self.max_his, device=device)
        position = (lengths[:, None] - len_range[None, :L]) * valid_his
        history_emb_pos = history_emb_raw + self.position_emb(position)
        history_emb_pos = self.dropout(history_emb_pos)

        # 3. Interest extraction — the part Chapter 4 replaces
        transport_info = None
        if self._uses_transport():
            interest_vectors, attention_maps, transport_info = self.transport_router(
                history_emb_pos, lengths, return_intermediate=True)
        else:
            interest_vectors, attention_maps = self.extractor(history_emb_pos, lengths)
        interest_vectors = self.dropout(interest_vectors)

        # 4. Aggregation
        if self._uses_evidence_aggregation():
            interest_weights = self.evidence_aggregator(
                interest_vectors, transport_info["transport"],
                history_emb_raw, lengths, position)
        else:
            interest_weights = self.aggregator(history_emb_raw, lengths)

        # 5-7. User vector + prediction (UNCHANGED)
        user_vector = (interest_vectors * interest_weights[:, :, None]).sum(dim=1)
        prediction = (user_vector[:, None, :] * candidate_emb).sum(dim=-1)

        # 8. Relation loss stashing (UNCHANGED)
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
                            ("interest_vectors", interest_vectors),
                            ("interest_weights", interest_weights),
                            ("candidate_vectors", candidate_emb),
                            ("prediction", prediction)]:
                check_nan_inf(t, name)
            logging.info("[ECTIR] First-batch NaN/Inf check passed.")

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
            if transport_info is not None:
                out_dict["transport"] = transport_info["transport"]
                out_dict["transport_capacity"] = transport_info["capacity"]

        return out_dict

    # ========================= Loss =========================
    # LOSS = BPR + Chapter 3 existing relation loss. No Chapter 4 auxiliary loss.

    def loss(self, out_dict: dict):
        total = super().loss(out_dict)
        if "_relation_ids" in out_dict and self.lambda_relation > 0:
            rel = self._compute_relation_loss(out_dict["_relation_ids"])
            total = total + self.lambda_relation * rel
            out_dict["loss_relation"] = rel.detach()
        return total

    def _compute_relation_loss(self, item_ids):
        """Chapter 3 relation-preservation loss, copied verbatim from ASPCF."""
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
