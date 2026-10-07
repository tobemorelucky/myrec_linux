# -*- coding: UTF-8 -*-
"""
LLMMIRecPPCIM — Chapter 4 Round 1: Prior–Posterior Candidate Interest Matching.

Question this round answers (ONE question only):

    is candidate-specific latent-interest scoring more effective than
    candidate-independent single-user-vector scoring?

Baseline scoring (all 7 existing models, verified identical):

    w            = HistoryOnlyAggregator(H)              # [B, K]
    u            = Σ_k w_k V_k                           # [B, D]  candidate-independent
    score_j      = uᵀ e_j                                # [B, C]

PPCIM scoring:

    p     = HistoryOnlyAggregator(H)                     # [B, K]   prior (history-only)
    m     = bmm(V, Eᵀ) / sqrt(D)                         # [B, K, C]
    z     = log(p + eps).unsqueeze(-1) + m / tau         # [B, K, C]
    score = sqrt(D) * tau * logsumexp(z, dim=1)          # [B, C]   (marginal)

The `sqrt(D)` factor is REQUIRED, not cosmetic: m is already divided by sqrt(D),
so sqrt(D)·tau·logsumexp → Σ_k p_k <V_k, e_j> as tau → inf, which is exactly the
baseline score. Without it the ranking limit is unchanged but the BPR score scale
shrinks by ~sqrt(D)=8, which would change the sigmoid gradients and break the
"only the scoring operator changed" fairness claim.

ROUND 1 HARD CONSTRAINTS — everything else is frozen ASPCF:
    ItemEncoder / position encoding / QueryMultiInterestExtractor /
    InterestAggregator / relation loss / BPR / K=4 / dropout /
    all dataset-specific hyper-parameters.
    Added trainable parameters: ZERO.

LOSS (unchanged) = BPR + lambda_relation * Chapter3 relation loss.
No Chapter 4 auxiliary loss is implemented or enabled this round.
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

VALID_MODES = ("baseline", "marginal", "posterior_mean")
VALID_PRIORS = ("history", "uniform")


class LLMMIRecPPCIM(SequentialModel):
    reader = "SeqReader"
    runner = "BaseRunner"

    extra_log_args = [
        "emb_size", "K", "item_encoder", "semantic_rank", "lambda_relation",
        "aspcf_gate_mode", "ppcim_mode", "ppcim_prior", "ppcim_tau",
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

        # ASPCF (frozen, unchanged)
        parser.add_argument("--semantic_rank", type=int, default=512)
        parser.add_argument("--semantic_dim", type=int, default=32)
        parser.add_argument("--semantic_hidden", type=int, default=128)
        parser.add_argument("--complement_dim", type=int, default=32)
        parser.add_argument("--tail_hidden", type=int, default=64)
        parser.add_argument("--complement_hidden", type=int, default=64)
        parser.add_argument("--gate_hidden", type=int, default=64)
        parser.add_argument("--aspcf_gate_mode", type=str, default="basic",
                           choices=["basic", "conflict"])

        # relation loss (Chapter 3, kept at 0.01)
        parser.add_argument("--lambda_relation", type=float, default=0.01)
        parser.add_argument("--relation_sample_size", type=int, default=128)
        parser.add_argument("--relation_teacher_temp", type=float, default=0.1)
        parser.add_argument("--relation_student_temp", type=float, default=0.1)

        # ---- Chapter 4 Round 1: PPCIM scoring ----
        parser.add_argument("--ppcim_mode", type=str, default="marginal",
                           choices=list(VALID_MODES),
                           help="baseline = exact ASPCF scoring; marginal = "
                                "latent-interest marginalisation; posterior_mean = "
                                "posterior-weighted mean (ablation)")
        parser.add_argument("--ppcim_prior", type=str, default="history",
                           choices=list(VALID_PRIORS))
        parser.add_argument("--ppcim_tau", type=float, default=1.0)
        parser.add_argument("--ppcim_eps", type=float, default=1e-8)

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

        self.ppcim_mode = str(getattr(args, "ppcim_mode", "marginal"))
        self.ppcim_prior = str(getattr(args, "ppcim_prior", "history"))
        self.ppcim_tau = float(getattr(args, "ppcim_tau", 1.0))
        self.ppcim_eps = float(getattr(args, "ppcim_eps", 1e-8))

        if self.ppcim_mode not in VALID_MODES:
            raise ValueError(f"Unknown ppcim_mode: {self.ppcim_mode}")
        if self.ppcim_prior not in VALID_PRIORS:
            raise ValueError(f"Unknown ppcim_prior: {self.ppcim_prior}")
        if self.ppcim_tau <= 0:
            raise ValueError(f"ppcim_tau must be > 0, got {self.ppcim_tau}")

        self.dropout_p = float(getattr(args, "dropout", 0.1))

        llm_table = None
        if self.item_encoder_mode in ("llm_replace", "residual", "aspcf"):
            llm_table = load_llm_table(self.llm_emb_path, expected_rows=self.item_num)

        self._define_params(llm_table)
        self.apply(self.init_weights)
        self._first_batch_checked = False

        logging.info(f"[PPCIM] initialized: enc={self.item_encoder_mode} K={self.K} "
                     f"mode={self.ppcim_mode} prior={self.ppcim_prior} "
                     f"tau={self.ppcim_tau} lambda_relation={self.lambda_relation}")
        logging.info(f"[PPCIM] #params: {self.count_variables()}  "
                     f"(added trainable params vs ASPCF: 0)")

    def _define_params(self, llm_table):
        """Identical to LLMMIRecASPCF — same module names for checkpoint compat."""
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

    # ========================= Scoring =========================

    def _score(self, interest_vectors, candidate_emb, history_emb_raw, lengths,
               need_posterior=False):
        """Returns (prediction [B,C], user_vector [B,D], extra dict).

        baseline       : u = Σ_k p_k V_k ; score = uᵀ e_j
        marginal       : sqrt(D)·tau·logsumexp_k(log p_k + m/τ)
        posterior_mean : sqrt(D)·Σ_k π_k m_k     (== <Σ_k π_k V_k, e_j>)

        `user_vector` is always the prior-aggregated vector u = Σ_k p_k V_k.
        In baseline mode it IS the scoring vector; in PPCIM modes it is not used
        for scoring but is returned for diagnostics/contract compatibility.
        """
        extra = {}
        p = self.aggregator(history_emb_raw, lengths)                    # [B,K]
        if self.ppcim_mode != "baseline" and self.ppcim_prior == "uniform":
            p = torch.full_like(p, 1.0 / p.size(-1))
        extra["history_interest_prior"] = p

        user_vector = (interest_vectors * p[:, :, None]).sum(dim=1)      # [B,D]

        if self.ppcim_mode == "baseline":
            prediction = (user_vector[:, None, :] * candidate_emb).sum(dim=-1)  # [B,C]
            return prediction, user_vector, extra

        # ---- candidate-interest match: m = <V_k, e_j>/sqrt(D)  [B,K,C] ----
        sqrtD = math.sqrt(self.emb_size)
        m = torch.bmm(interest_vectors,
                      candidate_emb.transpose(1, 2)) / sqrtD             # [B,K,C]

        # ---- z = log p + m/tau  [B,K,C] ----
        z = torch.log(p + self.ppcim_eps).unsqueeze(-1) + m / self.ppcim_tau

        if self.ppcim_mode == "marginal":
            # sqrt(D) is REQUIRED so that tau->inf recovers the baseline score scale
            prediction = sqrtD * self.ppcim_tau * torch.logsumexp(z, dim=1)   # [B,C]
        elif self.ppcim_mode == "posterior_mean":
            pi = torch.softmax(z, dim=1)                                  # [B,K,C]
            prediction = sqrtD * (pi * m).sum(dim=1)                      # [B,C]
            extra["candidate_interest_posterior"] = pi
        else:
            raise RuntimeError(f"unknown ppcim_mode {self.ppcim_mode}")

        extra["candidate_interest_match"] = m
        if need_posterior and "candidate_interest_posterior" not in extra:
            extra["candidate_interest_posterior"] = torch.softmax(z, dim=1)
        return prediction, user_vector, extra

    # ========================= Forward =========================

    def forward(self, feed_dict, return_intermediate=False):
        history = feed_dict["history_items"]
        lengths = feed_dict["lengths"]
        i_ids = feed_dict["item_id"]
        B, L = history.shape
        device = history.device

        # 1. Item embeddings (UNCHANGED)
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

        # 2. Position encoding (UNCHANGED)
        valid_his = (history > 0).long()
        len_range = torch.arange(self.max_his, device=device)
        position = (lengths[:, None] - len_range[None, :L]) * valid_his
        history_emb_pos = history_emb_raw + self.position_emb(position)
        history_emb_pos = self.dropout(history_emb_pos)

        # 3. Multi-interest extraction (UNCHANGED)
        interest_vectors, attention_maps = self.extractor(history_emb_pos, lengths)
        interest_vectors = self.dropout(interest_vectors)

        # 4-6. SCORING — the only part Chapter 4 Round 1 changes
        prediction, user_vector, extra = self._score(
            interest_vectors, candidate_emb, history_emb_raw, lengths,
            need_posterior=return_intermediate)

        # 7. Relation loss stashing (UNCHANGED)
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
                            ("candidate_vectors", candidate_emb),
                            ("prediction", prediction)]:
                check_nan_inf(t, name)
            logging.info("[PPCIM] First-batch NaN/Inf check passed.")

        # 9. Output
        if return_intermediate:
            out_dict["interest_vectors"] = interest_vectors
            out_dict["attention_maps"] = attention_maps
            out_dict["user_vector"] = user_vector
            out_dict["interest_weights"] = extra.get("history_interest_prior")
            out_dict["history_vectors"] = history_emb_raw
            out_dict["candidate_vectors"] = candidate_emb
            out_dict.update({k: v for k, v in extra.items() if v is not None})
            if aspcf_comps is not None:
                out_dict.update(aspcf_comps)

        return out_dict

    # ========================= Loss =========================
    # LOSS = BPR + Chapter 3 relation loss. No Chapter 4 auxiliary loss this round.

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
