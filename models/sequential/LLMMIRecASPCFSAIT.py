"""SAIT candidate: history-only semantic state forecasting participates in scoring."""
import logging
import math
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF
from models.sequential.sait_components import SemanticInterestTransition

MODES = ("baseline", "full", "semantic_only", "behavior_only", "markov", "static")

class LLMMIRecASPCFSAIT(LLMMIRecASPCF):
    extra_log_args = LLMMIRecASPCF.extra_log_args + [
        "sait_mode", "sait_rank", "lambda_transition", "sait_asset_path"]

    @staticmethod
    def parse_model_args(parser):
        parser = LLMMIRecASPCF.parse_model_args(parser)
        parser.add_argument("--sait_mode", choices=MODES, default="full")
        parser.add_argument("--sait_rank", type=int, default=8)
        parser.add_argument("--lambda_transition", type=float, default=.01)
        parser.add_argument("--sait_asset_path", default="")
        return parser

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.sait_mode = args.sait_mode
        self.lambda_transition = float(args.lambda_transition)
        if (args.item_encoder != "aspcf" or args.aspcf_gate_mode != "basic"
                or args.sait_rank != 8 or self.K != 4 or self.emb_size != 64
                or not math.isfinite(self.lambda_transition) or self.lambda_transition < 0):
            raise ValueError("SAIT prototype requires frozen ASPCF and registered settings")
        if self.sait_mode == "static" and self.lambda_transition != 0:
            raise ValueError("Static decoder ablation requires lambda_transition=0")
        if self.sait_mode == "baseline":
            return
        self._load_asset(args.sait_asset_path)
        # New parameters must not consume the native model/dropout/global RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.forecaster = SemanticInterestTransition(self.emb_size, 32, args.sait_rank,
                                                         personalize=self.sait_mode != 'markov',
                                                         static=self.sait_mode == 'static')
        logging.info("[SAIT] added_params=%d total_params=%d mode=%s lambda_transition=%s",
                     sum(p.numel() for p in self.forecaster.parameters()),
                     self.count_variables(), self.sait_mode, self.lambda_transition)

    def _load_asset(self, path):
        resolved = Path(path).resolve()
        root = Path(__file__).resolve().parents[2]
        if not resolved.is_relative_to(root):
            raise ValueError("SAIT asset must resolve within project")
        with resolved.open("rb") as stream:
            asset = pickle.load(stream)
        if asset.get("version") != "sait_v1":
            raise ValueError("Unknown transition asset schema")
        q = np.asarray(asset["assignments"], dtype=np.float32)
        if (q.shape != (self.item_num, 32) or not np.isfinite(q).all()
                or np.any(q < 0) or np.any(q[0] != 0)
                or not np.allclose(q[1:].sum(1), 1, atol=1e-5)):
            raise ValueError("Invalid item-aligned state assignments")
        meta = asset["meta"]
        if (meta.get("folds") != 3 or meta.get("fold_rule") != "user_id % 3"
                or meta.get("semantic_rank") != self.semantic_rank):
            raise ValueError("Incorrect train-only/fold schema")
        self.register_buffer("sait_q", torch.from_numpy(q))
        for name, key, shape in [
            ("sait_train_prior", "train_transition", (3,32,32)),
            ("sait_eval_prior", "eval_transition", (32,32)),
            ("sait_semantic_prior", "semantic_prior", (32,32))]:
            value = np.asarray(asset[key], dtype=np.float32)
            if (value.shape != shape or not np.isfinite(value).all() or np.any(value < 0)
                    or not np.allclose(value.sum(-1), 1, atol=1e-5)):
                raise ValueError("Invalid stochastic transition table")
            self.register_buffer(name, torch.from_numpy(value))
        counts = np.asarray(asset["counts_by_fold"], dtype=np.float64)
        if counts.shape != (3,32,32) or not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError("Invalid fold counts")
        behavior = np.stack([counts.sum(0)-counts[f] for f in range(3)])
        row_sum = behavior.sum(-1, keepdims=True)
        behavior = np.divide(behavior, row_sum, out=np.full_like(behavior,1/32), where=row_sum>0)
        evaluation = counts.sum(0)
        rows = evaluation.sum(-1, keepdims=True)
        evaluation = np.divide(evaluation, rows, out=np.full_like(evaluation,1/32), where=rows>0)
        self.register_buffer("sait_behavior_train", torch.tensor(behavior,dtype=torch.float32))
        self.register_buffer("sait_behavior_eval", torch.tensor(evaluation,dtype=torch.float32))

    def forward(self, feed_dict, return_intermediate=False):
        if self.sait_mode == "baseline":
            return super().forward(feed_dict, return_intermediate)
        out = super().forward(feed_dict, True)
        history, lengths = feed_dict["history_items"], feed_dict["lengths"]
        assignments = self.sait_q[history]
        prior = (self.sait_train_prior[feed_dict["user_id"].long() % 3]
                 if self.training else self.sait_eval_prior)
        if self.sait_mode == "semantic_only":
            prior = self.sait_semantic_prior
        elif self.sait_mode == "behavior_only":
            prior = (self.sait_behavior_train[feed_dict["user_id"].long() % 3]
                     if self.training else self.sait_behavior_eval)
        state = self.forecaster(out["interest_vectors"], out["interest_weights"],
                                out["attention_maps"], out["history_vectors"], lengths,
                                assignments, prior, markov=self.sait_mode=="markov",
                                static=self.sait_mode=="static")
        out["prediction"] = (state["user_vector"][:, None] * out["candidate_vectors"]).sum(-1)
        out["_sait_next_state"] = state["next_state"]
        if self.training and self.lambda_transition > 0:
            out["_sait_target_state"] = self.sait_q[feed_dict["item_id"][:, 0]].detach()
        if return_intermediate:
            out.update(state)
        else:
            out = {k:v for k,v in out.items() if k=="prediction" or k.startswith("_")}
        return out

    def loss(self, out_dict):
        total = super().loss(out_dict)
        if self.training and self.sait_mode != "baseline" and self.lambda_transition > 0:
            transition = F.kl_div(out_dict["_sait_next_state"].clamp_min(1e-12).log(),
                                  out_dict["_sait_target_state"], reduction="batchmean")
            total = total + self.lambda_transition * transition
            out_dict["loss_transition"] = transition.detach()
        return total
