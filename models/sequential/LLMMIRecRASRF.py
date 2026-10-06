# -*- coding: UTF-8 -*-
"""
LLMMIRecRASRF — Chapter 3 Round 2: Reliability-Aware Semantic Residual Fusion.

Core question (replaces Round 1's "global shared/private decomposition"):

  how do we stop LLM semantics from producing NEGATIVE transfer across
  datasets and across items?

Structure (structural test only — no new auxiliary losses):

  e_cf   = id_embedding[item]              CF main path, learned from
                                           recommendation interactions only
  e_sem  = adapter(llm_table[item])         semantic correction
  gate   = sigmoid(MLP(consistency(e_cf, e_sem)))   item-specific
  e_final = e_cf + gate * e_sem

The gate sees explicit cross-view consistency evidence (cosine agreement,
element-wise absolute difference / interaction, and optionally an offline
local-neighbourhood agreement prior) rather than a single global learnable
gamma.

LOSS = BPR only. Deliberately NOT included in this round:
alignment / relation / contrastive / orthogonal / entropy losses.

`--item_encoder residual` is supported so the same file can run the
global-gamma control that isolates whether ITEM-SPECIFIC gating is what helps.
"""

import logging
import pickle

import torch
import torch.nn as nn

from models.BaseModel import SequentialModel
from models.sequential.llmmi_utils import load_llm_table, check_nan_inf
from models.sequential.llmmi_components import (
    ItemEncoder,
    QueryMultiInterestExtractor,
    InterestAggregator,
)


class LLMMIRecRASRF(SequentialModel):
    reader = "SeqReader"
    runner = "BaseRunner"

    extra_log_args = [
        "emb_size", "K", "item_encoder",
        "rasrf_gate_mode", "rasrf_gate_input", "rasrf_gate_hidden",
        "adapter_hidden", "adapter_activation", "adapter_use_ln",
        "gamma_init", "gamma_trainable",
    ]

    # ========================= Args =========================

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--emb_size", type=int, default=64)
        parser.add_argument("--attn_size", type=int, default=64)
        parser.add_argument("--K", type=int, default=4)

        parser.add_argument("--item_encoder", type=str, default="rasrf",
                           choices=["id", "llm_replace", "residual", "rasrf"])
        parser.add_argument("--llm_emb_path", type=str, default="")

        # ---- semantic adapter ----
        parser.add_argument("--adapter_hidden", type=int, default=256)
        parser.add_argument("--adapter_activation", type=str, default="gelu",
                           choices=["gelu", "relu"])
        parser.add_argument("--adapter_use_ln", type=int, default=0, choices=[0, 1])

        # ---- global gamma (residual control only) ----
        parser.add_argument("--gamma_init", type=float, default=0.1)
        parser.add_argument("--gamma_trainable", type=int, default=1, choices=[0, 1])

        # ---- RASRF ----
        parser.add_argument("--rasrf_neigh_path", type=str, default="",
                           help="pkl from tools/build_semantic_neighborhood_agreement.py")
        parser.add_argument("--rasrf_gate_mode", type=str, default="scalar",
                           choices=["scalar", "vector"])
        parser.add_argument("--rasrf_gate_input", type=str, default="agree_diff_inter",
                           choices=["agree", "agree_diff", "agree_diff_inter"])
        parser.add_argument("--rasrf_gate_hidden", type=int, default=64)
        parser.add_argument("--rasrf_zero_init_correction", type=int, default=0,
                           choices=[0, 1],
                           help="1 = zero-init the adapter's last layer so the semantic "
                                "correction starts at exactly 0 (model is pure CF at init). "
                                "Default 0 = repo-standard init, so the main runs isolate "
                                "the gating mechanism rather than the initialisation.")

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

        self.item_encoder_mode = str(getattr(args, "item_encoder", "rasrf"))
        self.llm_emb_path = str(getattr(args, "llm_emb_path", ""))
        self.adapter_hidden = int(getattr(args, "adapter_hidden", 256))
        self.adapter_activation = str(getattr(args, "adapter_activation", "gelu"))
        self.adapter_use_ln = bool(int(getattr(args, "adapter_use_ln", 0)))
        self.gamma_init = float(getattr(args, "gamma_init", 0.1))
        self.gamma_trainable = bool(int(getattr(args, "gamma_trainable", 1)))

        self.rasrf_neigh_path = str(getattr(args, "rasrf_neigh_path", ""))
        self.rasrf_gate_mode = str(getattr(args, "rasrf_gate_mode", "scalar"))
        self.rasrf_gate_input = str(getattr(args, "rasrf_gate_input", "agree_diff_inter"))
        self.rasrf_gate_hidden = int(getattr(args, "rasrf_gate_hidden", 64))

        self.dropout_p = float(getattr(args, "dropout", 0.1))

        llm_table = None
        if self.item_encoder_mode in ("llm_replace", "residual", "rasrf"):
            llm_table = load_llm_table(self.llm_emb_path, expected_rows=self.item_num)

        # ---- optional neighbourhood-agreement prior ----
        neigh_prior = None
        if self.item_encoder_mode == "rasrf" and self.rasrf_neigh_path:
            nd = pickle.load(open(self.rasrf_neigh_path, "rb"))
            arr = nd["agreement"]
            if arr.shape[0] != self.item_num:
                raise ValueError(
                    f"neigh prior rows {arr.shape[0]} != item_num {self.item_num}"
                )
            neigh_prior = torch.tensor(arr, dtype=torch.float32)
            logging.info(f"[RASRF] neigh prior: {tuple(neigh_prior.shape)} "
                         f"k={nd.get('k')} from {self.rasrf_neigh_path}")
            st = nd.get("stats", {})
            if st:
                logging.info(f"[RASRF]   a1(llm->cf) mean={st.get('a1_llm2cf_mean')} "
                             f"a2(cf->llm) mean={st.get('a2_cf2llm_mean')} "
                             f"corr={st.get('corr_a1_a2')}")

        self._define_params(llm_table, neigh_prior)
        self.apply(self.init_weights)

        # Optional: make the semantic correction start at exactly zero so the
        # model begins as a pure collaborative model. Must run AFTER init_weights.
        # Only meaningful for modes that have an adapter.
        self.rasrf_zero_init_correction = bool(
            int(getattr(args, "rasrf_zero_init_correction", 0)))
        if self.rasrf_zero_init_correction and hasattr(self.item_encoder, "adapter"):
            last = self.item_encoder.adapter[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
                logging.info("[RASRF] zero-init: semantic correction starts at 0 "
                             "(model is pure CF at init)")

        self._first_batch_checked = False

        logging.info(f"[RASRF] initialized: enc={self.item_encoder_mode} K={self.K} "
                     f"gate_mode={self.rasrf_gate_mode} gate_input={self.rasrf_gate_input} "
                     f"neigh={'yes' if neigh_prior is not None else 'no'}")
        if self.item_encoder_mode == "residual":
            logging.info(f"[RASRF]   residual control: gamma_init={self.gamma_init} "
                         f"gamma_trainable={self.gamma_trainable}")
        logging.info(f"[RASRF] #params: {self.count_variables()}")

    def _define_params(self, llm_table, neigh_prior):
        ie_kwargs = dict(
            item_num=self.item_num, emb_size=self.emb_size,
            mode=self.item_encoder_mode, llm_table=llm_table,
            adapter_hidden=self.adapter_hidden,
            adapter_activation=self.adapter_activation,
            adapter_use_ln=self.adapter_use_ln,
            gamma_init=self.gamma_init,
            gamma_trainable=self.gamma_trainable,
        )
        if self.item_encoder_mode == "rasrf":
            ie_kwargs.update(
                neigh_prior=neigh_prior,
                rasrf_gate_mode=self.rasrf_gate_mode,
                rasrf_gate_input=self.rasrf_gate_input,
                rasrf_gate_hidden=self.rasrf_gate_hidden,
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
            for k in hist_out:
                if k == "emb":
                    continue
                comps[f"history_{k}"] = hist_out[k]
                comps[f"candidate_{k}"] = cand_out.get(k)
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

        out_dict = {"prediction": prediction}

        # 7. NaN/Inf check
        if not self._first_batch_checked:
            self._first_batch_checked = True
            for name, t in [("history_vectors", history_emb_raw),
                            ("interest_vectors", interest_vectors),
                            ("interest_weights", interest_weights),
                            ("candidate_vectors", candidate_emb),
                            ("prediction", prediction)]:
                check_nan_inf(t, name)
            logging.info("[RASRF] First-batch NaN/Inf check passed.")

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
    # LOSS = plain BPR (inherited). No auxiliary terms in this round.
