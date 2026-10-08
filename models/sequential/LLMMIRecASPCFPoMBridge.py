"""Chapter 4 backbone attribution: unchanged ASPCF + tensor-only PoMRec."""
import logging

import torch

from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF
from models.sequential.llmmi_utils import check_nan_inf
from models.sequential.pomrec_backbone import PoMRecInterestBackbone


class LLMMIRecASPCFPoMBridge(LLMMIRecASPCF):
    extra_log_args = LLMMIRecASPCF.extra_log_args + [
        "bridge_mode", "bridge_attn_size", "bridge_prompt_num",
        "bridge_n_layers", "bridge_lambda_disp",
    ]

    @staticmethod
    def parse_model_args(parser):
        parser = LLMMIRecASPCF.parse_model_args(parser)
        parser.add_argument("--bridge_mode", choices=["clean", "pom_centrality", "pom_full"], default="clean")
        parser.add_argument("--bridge_attn_size", type=int, default=8)
        parser.add_argument("--bridge_prompt_num", type=int, default=3)
        parser.add_argument("--bridge_n_layers", type=int, default=2)
        parser.add_argument("--bridge_lambda_disp", type=float, default=1.0)
        return parser

    def __init__(self, args, corpus):
        # Execute the exact parent construction AND initialization order.
        # Shared encoder/position initial values and post-init RNG match clean.
        super().__init__(args, corpus)
        self.bridge_mode = args.bridge_mode
        if self.bridge_mode not in ("clean", "pom_centrality", "pom_full"):
            raise ValueError("invalid bridge_mode")
        if self.item_encoder_mode != "aspcf":
            raise ValueError("Bridge requires the original ASPCF ItemEncoder")
        if self.bridge_mode != "clean":
            if self.emb_size != 64:
                raise ValueError("registered Bridge requires 64-dimensional input")
            dispersion = 0.0 if self.bridge_mode == "pom_centrality" else args.bridge_lambda_disp
            # New module initialization must not move the shared training RNG.
            # Modules are constructed on CPU before main.py moves the model.
            with torch.random.fork_rng(devices=[]):
                backbone = PoMRecInterestBackbone(
                    self.emb_size, self.K, args.bridge_attn_size,
                    args.bridge_prompt_num, args.bridge_n_layers, dispersion)
                backbone.apply(self.init_weights)
            del self.extractor
            del self.aggregator
            self.backbone = backbone
        logging.info("[ASPCFPoMBridge] mode=%s bridge_A=%s p=%s n=%s effective_disp=%s #params: %s",
                     self.bridge_mode, args.bridge_attn_size, args.bridge_prompt_num,
                     args.bridge_n_layers,
                     0.0 if self.bridge_mode != "pom_full" else args.bridge_lambda_disp,
                     self.count_variables())

    def forward(self, feed_dict, return_intermediate=False):
        if self.bridge_mode == "clean":
            return super().forward(feed_dict, return_intermediate=return_intermediate)
        history, lengths, i_ids = (feed_dict["history_items"],
                                  feed_dict["lengths"], feed_dict["item_id"])
        B, L = history.shape
        if L > self.max_his:
            raise ValueError("history exceeds history_max")
        valid = history > 0
        if lengths.shape != (B,):
            raise ValueError("lengths shape does not match history")
        self.backbone._check_history_layout(valid, lengths)
        comps = {}
        if return_intermediate:
            hist_out = self.item_encoder(history, return_components=True)
            cand_out = self.item_encoder(i_ids, return_components=True)
            Hraw, candidates = hist_out["emb"], cand_out["emb"]
            for prefix, output in (("history", hist_out), ("candidate", cand_out)):
                for name in ("semantic", "complement", "alpha_sem", "alpha_comp"):
                    comps[f"{prefix}_{name}"] = output[name]
        else:
            Hraw = self.item_encoder(history)
            candidates = self.item_encoder(i_ids)
        position = (lengths[:, None] - torch.arange(self.max_his, device=history.device)[None, :L]) * valid.long()
        H = self.dropout(Hraw + self.position_emb(position))
        if return_intermediate:
            details = self.backbone(H, valid, lengths, return_intermediate=True, validate_history=False)
            V, w = details["interest_vectors"], details["interest_weights"]
        else:
            V, w = self.backbone(H, valid, lengths, validate_history=False)
        V = self.dropout(V)
        u = (V * w[:, :, None]).sum(1)
        prediction = (u[:, None, :] * candidates).sum(-1)
        out = {"prediction": prediction}
        # Exact ASPCF relation-ID collection/sampling; loss method is inherited.
        if self.training and self.lambda_relation > 0:
            unique_ids = torch.unique(torch.cat([history.reshape(-1), i_ids.reshape(-1)], 0))
            unique_ids = unique_ids[unique_ids != 0]
            if unique_ids.numel() > self.relation_sample_size:
                idx = torch.randperm(unique_ids.numel(), device=history.device)[:self.relation_sample_size]
                unique_ids = unique_ids[idx]
            out["_relation_ids"] = unique_ids
        if not self._first_batch_checked:
            self._first_batch_checked = True
            for name, value in (("history_vectors", Hraw), ("interest_vectors", V),
                                ("interest_weights", w), ("candidate_vectors", candidates),
                                ("prediction", prediction)):
                check_nan_inf(value, name)
        if return_intermediate:
            out.update(details)
            out.update(comps)
            out.update(interest_vectors=V, interest_weights=w, user_vector=u,
                       history_vectors=Hraw, candidate_vectors=candidates)
        return out
