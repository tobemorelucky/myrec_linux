# -*- coding: UTF-8 -*-
"""
LLMMIRecCGSCD — Chapter 3 (Architecture-First): Collaborative-Guided
Shared/Complementary representation.

Scope of this round: validate the REPRESENTATION ARCHITECTURE itself.

  LLM embedding
    -> shared projection (frozen U_r from cross-view association)
    -> shared branch
  LLM private residual + collaborative item embedding
    -> complementary branch
  (both) -> adaptive gate -> fused item embedding
  fused embedding -> unchanged multi-interest backbone

Deliberately NOT added in this round (see THESIS_MASTER_PLAN.md §0.5):
  shared relation loss / private relation loss / redundancy loss /
  orthogonality loss / contrastive loss.

Only the pre-existing ASPCF relation-preservation loss is kept, as a switchable
legacy term (--lambda_relation 0.0 => PURE_BPR, 0.01 => BPR + existing relation),
so we can tell whether any gain comes from the new encoder or from the old
auxiliary supervision.
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


class LLMMIRecCGSCD(SequentialModel):
    reader = "SeqReader"
    runner = "BaseRunner"

    # NOTE: main.py resolves these via eval('args.<name>'), so every entry must
    # be a real CLI arg. Derived values (method/rank, read from the basis pkl)
    # are logged explicitly in __init__ instead.
    extra_log_args = [
        "emb_size", "K", "item_encoder",
        "cgscd_shared_dim", "cgscd_compl_dim",
        "cgscd_gate_mode", "lambda_relation",
    ]

    # ========================= Args =========================

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--emb_size", type=int, default=64)
        parser.add_argument("--attn_size", type=int, default=64)
        parser.add_argument("--K", type=int, default=4)

        parser.add_argument("--item_encoder", type=str, default="cgscd",
                           choices=["id", "llm_replace", "residual", "aspcf", "cgscd"])
        parser.add_argument("--llm_emb_path", type=str, default="")

        # ---- CGSCD ----
        parser.add_argument("--cgscd_basis_path", type=str, default="",
                           help="pkl from tools/build_cgscd_basis.py")
        parser.add_argument("--cgscd_cf_path", type=str, default="",
                           help="collaborative item embedding pkl (itm_emb_pomrec.pkl)")
        parser.add_argument("--cgscd_shared_dim", type=int, default=32)
        parser.add_argument("--cgscd_shared_hidden", type=int, default=128)
        parser.add_argument("--cgscd_compl_dim", type=int, default=32)
        parser.add_argument("--cgscd_compl_hidden", type=int, default=64)
        parser.add_argument("--cgscd_gate_mode", type=str, default="basic",
                           choices=["basic", "conflict"])

        # ---- relation preservation loss (legacy ASPCF term, switchable) ----
        parser.add_argument("--lambda_relation", type=float, default=0.0,
                           help="0.0 => PURE_BPR; 0.01 => BPR + existing relation loss")
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

        self.item_encoder_mode = str(getattr(args, "item_encoder", "cgscd"))
        self.llm_emb_path = str(getattr(args, "llm_emb_path", ""))
        self.cgscd_basis_path = str(getattr(args, "cgscd_basis_path", ""))
        self.cgscd_cf_path = str(getattr(args, "cgscd_cf_path", ""))
        self.cgscd_shared_dim = int(getattr(args, "cgscd_shared_dim", 32))
        self.cgscd_shared_hidden = int(getattr(args, "cgscd_shared_hidden", 128))
        self.cgscd_compl_dim = int(getattr(args, "cgscd_compl_dim", 32))
        self.cgscd_compl_hidden = int(getattr(args, "cgscd_compl_hidden", 64))
        self.cgscd_gate_mode = str(getattr(args, "cgscd_gate_mode", "basic"))

        self.lambda_relation = float(getattr(args, "lambda_relation", 0.0))
        self.relation_sample_size = int(getattr(args, "relation_sample_size", 128))
        self.relation_teacher_temp = float(getattr(args, "relation_teacher_temp", 0.1))
        self.relation_student_temp = float(getattr(args, "relation_student_temp", 0.1))

        self.dropout_p = float(getattr(args, "dropout", 0.1))

        # ---- LLM table ----
        llm_table = None
        if self.item_encoder_mode in ("llm_replace", "residual", "aspcf", "cgscd"):
            llm_table = load_llm_table(self.llm_emb_path, expected_rows=self.item_num)

        # ---- CGSCD basis + collaborative table ----
        basis = z_mean = cf_table = None
        self.cgscd_method = "none"
        self.cgscd_rank = 0
        if self.item_encoder_mode == "cgscd":
            if not self.cgscd_basis_path:
                raise ValueError("item_encoder=cgscd requires --cgscd_basis_path")
            if not self.cgscd_cf_path:
                raise ValueError("item_encoder=cgscd requires --cgscd_cf_path")

            bd = pickle.load(open(self.cgscd_basis_path, "rb"))
            basis = torch.tensor(bd["U_r"], dtype=torch.float32)          # [d_llm, r]
            z_mean = torch.tensor(bd["z_mean"], dtype=torch.float32)      # [d_llm]
            self.cgscd_method = str(bd.get("method", "unknown"))
            self.cgscd_rank = int(bd.get("rank", basis.size(1)))

            cf_table = load_llm_table(self.cgscd_cf_path, expected_rows=self.item_num)

            if basis.size(0) != llm_table.size(1):
                raise ValueError(
                    f"basis d_llm {basis.size(0)} != llm_table d_llm {llm_table.size(1)}; "
                    f"basis was built from a different LLM table"
                )
            log = bd.get("diagnostics", {})
            logging.info(f"[CGSCD] basis: method={self.cgscd_method} rank={self.cgscd_rank} "
                         f"U_r={tuple(basis.shape)}")
            logging.info(f"[CGSCD]   source z_sha1={bd.get('source', {}).get('z_sha1', '')[:12]} "
                         f"c_sha1={bd.get('source', {}).get('c_sha1', '')[:12]}")
            if log:
                logging.info(f"[CGSCD]   z_var_share_r={log.get('z_var_share_r', float('nan')):.4f} "
                             f"cf_r2_total={log.get('cf_r2_total', float('nan')):.4f} "
                             f"pred_share_in_shared={log.get('pred_share_in_shared', float('nan')):.4f}")
            logging.info(f"[CGSCD] cf_table: {tuple(cf_table.shape)}")

        self._define_params(llm_table, basis, z_mean, cf_table)
        self.apply(self.init_weights)
        self._first_batch_checked = False

        logging.info(f"[CGSCD] initialized: enc={self.item_encoder_mode} K={self.K} "
                     f"shared={self.cgscd_shared_dim} compl={self.cgscd_compl_dim} "
                     f"gate={self.cgscd_gate_mode} lambda_relation={self.lambda_relation}")
        logging.info(f"[CGSCD] #params: {self.count_variables()}")

    def _define_params(self, llm_table, basis, z_mean, cf_table):
        ie_kwargs = dict(
            item_num=self.item_num, emb_size=self.emb_size,
            mode=self.item_encoder_mode, llm_table=llm_table,
        )
        if self.item_encoder_mode == "cgscd":
            ie_kwargs.update(
                shared_basis=basis, z_mean=z_mean, cf_table=cf_table,
                shared_dim=self.cgscd_shared_dim,
                shared_hidden=self.cgscd_shared_hidden,
                compl_dim=self.cgscd_compl_dim,
                compl_hidden=self.cgscd_compl_hidden,
                gate_hidden=64,
                cgscd_gate_mode=self.cgscd_gate_mode,
            )
        self.item_encoder = ItemEncoder(**ie_kwargs)

        self.position_emb = nn.Embedding(self.max_his + 1, self.emb_size)
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

        # 1. Item embeddings (shared encoder for history AND candidates)
        comps = None
        if return_intermediate:
            hist_out = self.item_encoder(history, return_components=True)
            cand_out = self.item_encoder(i_ids, return_components=True)
            history_emb_raw = hist_out["emb"]
            candidate_emb = cand_out["emb"]
            comps = {}
            for k in ("semantic", "complement", "alpha_sem", "alpha_comp"):
                comps[f"history_{k}"] = hist_out[k]
                comps[f"candidate_{k}"] = cand_out[k]
        else:
            history_emb_raw = self.item_encoder(history)
            candidate_emb = self.item_encoder(i_ids)

        # 2. Position encoding
        valid_his = (history > 0).long()
        len_range = torch.arange(self.max_his, device=device)
        position = (lengths[:, None] - len_range[None, :L]) * valid_his
        history_emb_pos = history_emb_raw + self.position_emb(position)
        history_emb_pos = self.dropout(history_emb_pos)

        # 3. Multi-interest extraction (UNCHANGED backbone)
        interest_vectors, attention_maps = self.extractor(history_emb_pos, lengths)
        interest_vectors = self.dropout(interest_vectors)

        # 4-6. Aggregation, user vector, prediction (UNCHANGED)
        interest_weights = self.aggregator(history_emb_raw, lengths)
        user_vector = (interest_vectors * interest_weights[:, :, None]).sum(dim=1)
        prediction = (user_vector[:, None, :] * candidate_emb).sum(dim=-1)

        # 7. Relation loss stashing
        if self.training and self.lambda_relation > 0:
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
            logging.info("[CGSCD] First-batch NaN/Inf check passed.")

        if return_intermediate:
            out_dict["interest_vectors"] = interest_vectors
            out_dict["attention_maps"] = attention_maps
            out_dict["interest_weights"] = interest_weights
            out_dict["user_vector"] = user_vector
            out_dict["history_vectors"] = history_emb_raw
            out_dict["candidate_vectors"] = candidate_emb
            if comps is not None:
                out_dict.update(comps)

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
        """Legacy ASPCF relation-preservation term, re-pointed at the CGSCD split.

        Teacher: frozen shared-subspace coordinates z_shared (the CGSCD analogue
                 of ASPCF's z_high).
        Student: shared_branch(z_shared).

        This is the SAME objective ASPCF uses, only the teacher slice changes.
        """
        M = item_ids.numel()
        if M < 2:
            return torch.zeros([], device=item_ids.device)

        ie = self.item_encoder
        z = ie.llm_table[item_ids]
        z_sh = (z - ie.z_mean) @ ie.shared_basis
        teacher = z_sh
        student = ie.shared_branch(z_sh)

        teacher = F.normalize(teacher, dim=-1, eps=1e-8)
        student = F.normalize(student, dim=-1, eps=1e-8)
        t_sim = teacher @ teacher.t() / self.relation_teacher_temp
        s_sim = student @ student.t() / self.relation_student_temp
        mask = ~torch.eye(M, dtype=torch.bool, device=item_ids.device)
        t_sim = t_sim[mask].view(M, M - 1)
        s_sim = s_sim[mask].view(M, M - 1)
        return F.kl_div(F.log_softmax(s_sim, dim=-1),
                        F.softmax(t_sim, dim=-1).detach(), reduction="batchmean")
