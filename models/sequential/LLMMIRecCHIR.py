# -*- coding: UTF-8 -*-
"""
LLMMIRecCHIR — Chapter 4: Collaborative-calibrated Hierarchical Interest Routing.

Builds on ASPCF + relation preservation, adds prototype-guided interest query
with collaborative calibration (global / prototype-specific).
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
    DualViewInterestExtractor,
    InterestAggregator,
)


class LLMMIRecCHIR(SequentialModel):
    reader = "SeqReader"
    runner = "BaseRunner"

    extra_log_args = [
        "emb_size", "K", "item_encoder", "adapter_hidden",
        "adapter_activation", "adapter_use_ln",
        "semantic_rank", "lambda_relation", "aspcf_gate_mode",
        "interest_query_mode", "collab_calibration",
        "prototype_prior_strength", "routing_mode",
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

        # Prototype query
        parser.add_argument("--interest_query_mode", type=str, default="learnable",
                           choices=["learnable", "prototype"])
        parser.add_argument("--prototype_path", type=str, default="")
        parser.add_argument("--collab_calibration", type=str, default="global",
                           choices=["global", "prototype"],
                           help="global: single collab context; "
                                "prototype: per-prototype weighted collab context")
        parser.add_argument("--prototype_prior_strength", type=float, default=0.0,
                           help="Strength of prototype history weights as attention routing prior")
        parser.add_argument("--routing_mode", type=str, default="single",
                           choices=["single", "dual"],
                           help="single: QueryMultiInterestExtractor; dual: DualViewInterestExtractor")
        parser.add_argument("--routing_gate_hidden", type=int, default=32,
                           help="Hidden dim for dual-view routing gate MLP")

        # Relation loss
        parser.add_argument("--lambda_relation", type=float, default=0.01)
        parser.add_argument("--relation_sample_size", type=int, default=128)
        parser.add_argument("--relation_teacher_temp", type=float, default=0.1)
        parser.add_argument("--relation_student_temp", type=float, default=0.1)

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

        self.interest_query_mode = str(getattr(args, "interest_query_mode", "learnable"))
        self.prototype_path = str(getattr(args, "prototype_path", ""))
        self.collab_calibration = str(getattr(args, "collab_calibration", "global"))
        self.prototype_prior_strength = float(getattr(args, "prototype_prior_strength", 0.0))
        self.routing_mode = str(getattr(args, "routing_mode", "single"))
        self.routing_gate_hidden = int(getattr(args, "routing_gate_hidden", 32))

        self.lambda_relation = float(getattr(args, "lambda_relation", 0.01))
        self.relation_sample_size = int(getattr(args, "relation_sample_size", 128))
        self.relation_teacher_temp = float(getattr(args, "relation_teacher_temp", 0.1))
        self.relation_student_temp = float(getattr(args, "relation_student_temp", 0.1))

        self.dropout_p = float(getattr(args, "dropout", 0.1))

        llm_table = None
        if self.item_encoder_mode in ("llm_replace", "residual", "aspcf"):
            llm_table = load_llm_table(self.llm_emb_path, expected_rows=self.item_num)

        # Prototype data
        if self.interest_query_mode == "prototype":
            if not self.prototype_path:
                raise ValueError("interest_query_mode=prototype requires --prototype_path")
            proto_data = pickle.load(open(self.prototype_path, "rb"))
            self.register_buffer("proto_centers",
                torch.tensor(proto_data["centers"], dtype=torch.float32), persistent=False)
            self.register_buffer("proto_assignments",
                torch.tensor(proto_data["soft_assignments"], dtype=torch.float32), persistent=False)
            self.proto_num = int(proto_data["prototype_num"])
            logging.info(f"[CHIR] prototype: num={self.proto_num}")

        self._define_params(llm_table)
        self.apply(self.init_weights)
        self._first_batch_checked = False

        logging.info(f"[CHIR] initialized: enc={self.item_encoder_mode} K={self.K} "
                     f"query={self.interest_query_mode} calibration={self.collab_calibration} "
                     f"routing={self.routing_mode} gate={self.aspcf_gate_mode} "
                     f"relation_lambda={self.lambda_relation}")
        logging.info(f"[CHIR] #params: {self.count_variables()}")

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
        if self.routing_mode == "dual":
            self.extractor = DualViewInterestExtractor(
                K=self.K, semantic_dim=self.semantic_dim,
                complement_dim=self.complement_dim,
                attn_dim=self.attn_size, gate_hidden=self.routing_gate_hidden,
                emb_size=self.emb_size)
        else:
            self.extractor = QueryMultiInterestExtractor(
                K=self.K, emb_size=self.emb_size, attn_size=self.attn_size)
        self.aggregator = InterestAggregator(emb_size=self.emb_size, K=self.K)
        self.dropout = nn.Dropout(p=self.dropout_p)

    # ========================= Forward =========================

    def forward(self, feed_dict, return_intermediate=False):
        history = feed_dict["history_items"]
        lengths = feed_dict["lengths"]
        i_ids = feed_dict["item_id"]
        B, L = history.shape
        device = history.device

        # 1. Item embeddings
        need_comp = (return_intermediate or
                    (self.interest_query_mode == "prototype"
                     and self.item_encoder_mode == "aspcf"))
        aspcf_comps = None
        hist_out = None
        if need_comp and self.item_encoder_mode == "aspcf":
            hist_out = self.item_encoder(history, return_components=True)
            cand_out = self.item_encoder(i_ids, return_components=True)
            history_emb_raw = hist_out["emb"]
            candidate_emb = cand_out["emb"]
            if return_intermediate:
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

        # 3. Multi-interest extraction
        external_query = None
        proto_mass = None
        selected_proto_ids = None
        query_seeds = None
        proto_hist_weights = None
        proto_collab_context = None
        sem_query = None
        collab_ctx = None

        if self.interest_query_mode == "prototype":
            proto_dist = self.proto_assignments[history]           # [B, L, P]
            valid_his_f = (history > 0).float().unsqueeze(-1)      # [B, L, 1]
            user_mass = ((proto_dist * valid_his_f).sum(dim=1) /
                          valid_his_f.sum(dim=1).clamp(min=1))     # [B, P]
            proto_mass = user_mass
            _, topk_idx = torch.topk(user_mass, k=self.K, dim=-1)  # [B, K]
            selected_proto_ids = topk_idx

            # Semantic query from prototype centers
            centers = self.proto_centers[topk_idx]                  # [B, K, 512]
            sem_query = self.item_encoder.semantic_branch(centers)   # [B, K, 32]

            # Collaborative context
            hist_comp = hist_out["complement"]                      # [B, L, 32]

            if self.collab_calibration == "global":
                collab_ctx = ((hist_comp * valid_his_f).sum(dim=1) /
                               valid_his_f.sum(dim=1).clamp(min=1))  # [B, 32]
                collab_ctx = collab_ctx.unsqueeze(1).expand(-1, self.K, -1)  # [B, K, 32]
            else:  # prototype-specific
                # proto_weight[j,k] = proto_assignment[history_j, p_k], normalized over j
                proto_w = proto_dist.gather(
                    2, topk_idx.unsqueeze(1).expand(-1, L, -1))     # [B, L, K]
                proto_w = proto_w * (history > 0).float().unsqueeze(-1)
                proto_w = proto_w / proto_w.sum(
                    dim=1, keepdim=True
                ).clamp_min(1e-8)
                proto_hist_weights = proto_w                         # stash
                collab_ctx = torch.einsum("bld,blk->bkd", hist_comp, proto_w)  # [B, K, 32]
                proto_collab_context = collab_ctx                    # stash

            query_seeds = torch.cat([sem_query, collab_ctx], dim=-1)  # [B, K, 64]
            external_query = query_seeds

        # Dual-view routing extras
        dual_extras = {}
        if self.routing_mode == "dual":
            # DualViewInterestExtractor needs semantic_query and collaborative_query
            # from the prototype path above, plus history components
            if hist_out is None or "semantic" not in hist_out:
                hist_out = self.item_encoder(history, return_components=True)
            history_sem = hist_out["semantic"]      # [B, L, 32]
            history_comp = hist_out["complement"]    # [B, L, 32]
            extractor_out = self.extractor(
                history_emb_pos, lengths,
                history_semantic=history_sem,
                history_complement=history_comp,
                semantic_query=sem_query,
                collaborative_query=collab_ctx,
            )
            interest_vectors = extractor_out["interest_vectors"]
            attention_maps = extractor_out["attention_maps"]
            dual_extras = {
                "semantic_attention_logits": extractor_out["semantic_attention_logits"],
                "collaborative_attention_logits": extractor_out["collaborative_attention_logits"],
                "routing_rho": extractor_out["routing_rho"],
                "semantic_query": extractor_out["semantic_query"],
                "collaborative_query": extractor_out["collaborative_query"],
            }
        else:
            # Single-view: existing QueryMultiInterestExtractor
            attn_prior = None
            prior_strength = 0.0
            if (self.interest_query_mode == "prototype"
                    and self.collab_calibration == "prototype"
                    and proto_hist_weights is not None
                    and self.prototype_prior_strength > 0):
                attn_prior = proto_hist_weights.transpose(1, 2)
                prior_strength = self.prototype_prior_strength

            extractor_out = self.extractor(
                history_emb_pos, lengths, external_query=external_query,
                attention_prior=attn_prior, prior_strength=prior_strength,
            )
            logits_before_prior = None
            if attn_prior is not None and prior_strength > 0:
                interest_vectors, attention_maps, logits_before_prior = extractor_out
            else:
                interest_vectors, attention_maps = extractor_out
            if logits_before_prior is not None:
                dual_extras["attention_logits_before_prior"] = logits_before_prior
                dual_extras["attention_prior"] = attn_prior

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

        # 8. NaN/Inf check
        if not self._first_batch_checked:
            self._first_batch_checked = True
            for name, t in [("history_vectors", history_emb_raw),
                            ("interest_vectors", interest_vectors),
                            ("interest_weights", interest_weights),
                            ("candidate_vectors", candidate_emb),
                            ("prediction", prediction)]:
                check_nan_inf(t, name)
            logging.info("[CHIR] First-batch NaN/Inf check passed.")

        # 9. Output
        if return_intermediate:
            out_dict["interest_vectors"] = interest_vectors
            out_dict["attention_maps"] = attention_maps
            out_dict["interest_weights"] = interest_weights
            out_dict["user_vector"] = user_vector
            out_dict["history_vectors"] = history_emb_raw
            out_dict["candidate_vectors"] = candidate_emb
            if aspcf_comps is not None:
                out_dict.update(aspcf_comps)
            if self.interest_query_mode == "prototype":
                out_dict["prototype_mass"] = proto_mass
                out_dict["selected_prototype_ids"] = selected_proto_ids
                out_dict["query_seeds"] = query_seeds
                if proto_hist_weights is not None:
                    out_dict["prototype_history_weights"] = proto_hist_weights
                    out_dict["prototype_collab_context"] = proto_collab_context
            if dual_extras:
                out_dict.update(dual_extras)

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
