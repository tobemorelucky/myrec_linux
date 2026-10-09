"""Controlled train-only auxiliary-negative attribution on unchanged ASPCF."""
import logging
import math
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF


class AuxiliaryNegativeSampler:
    def __init__(self, n_items, train_clicked, mode, seed, bank=None):
        if mode not in ("random_aux", "shnc"):
            raise ValueError("Unknown auxiliary source")
        self.n_items, self.train_clicked = int(n_items), train_clicked
        self.mode, self.seed, self.bank = mode, int(seed), bank
        if mode == "shnc":
            if (not isinstance(bank, np.ndarray) or bank.shape != (n_items, 100)
                    or not np.issubdtype(bank.dtype, np.integer)
                    or np.any(bank < 0) or np.any(bank >= n_items)
                    or np.any(bank[0] != 0)):
                raise ValueError("Semantic bank must be aligned integer [n_items,100], row0=0")

    def sample(self, user, positive, history, original_negatives, epoch, index):
        if not 0 < int(positive) < self.n_items:
            raise ValueError("Invalid positive item")
        forbidden = {0, int(positive)}
        forbidden.update(map(int, self.train_clicked.get(int(user), ())))
        forbidden.update(map(int, history))
        forbidden.update(map(int, original_negatives))
        forbidden = {x for x in forbidden if 0 <= x < self.n_items}
        if len(forbidden) >= self.n_items:
            raise ValueError("No legal auxiliary negative exists")
        rng = np.random.default_rng(np.random.SeedSequence(
            [self.seed, int(epoch), int(index), 1701]))
        fallback, ratio = False, 0.
        if self.mode == "shnc":
            pool = list(dict.fromkeys(int(x) for x in self.bank[int(positive)]
                                     if int(x) not in forbidden))
            ratio = len(pool) / 100.
            if pool:
                return int(pool[int(rng.integers(len(pool)))]), False, ratio
            fallback = True
        for _ in range(64):
            item = int(rng.integers(1, self.n_items))
            if item not in forbidden:
                return item, fallback, ratio
        allowed = [x for x in range(1, self.n_items) if x not in forbidden]
        return int(allowed[int(rng.integers(len(allowed)))]), fallback, ratio


class LLMMIRecASPCFAuxNeg(LLMMIRecASPCF):
    extra_log_args = LLMMIRecASPCF.extra_log_args + ["aux_mode", "lambda_aux", "aux_warmup_epochs"]

    @staticmethod
    def parse_model_args(parser):
        parser = LLMMIRecASPCF.parse_model_args(parser)
        parser.add_argument("--aux_mode", choices=["random_aux", "shnc"], default="random_aux")
        parser.add_argument("--lambda_aux", type=float, default=.005)
        parser.add_argument("--aux_warmup_epochs", type=int, default=5)
        parser.add_argument("--aux_bank_path", type=str, default="")
        return parser

    def __init__(self, args, corpus):
        if args.item_encoder != "aspcf" or args.aspcf_gate_mode != "basic":
            raise ValueError("Controlled experiment requires clean basic ASPCF")
        super().__init__(args, corpus)
        self.aux_mode = args.aux_mode
        self.lambda_aux = float(args.lambda_aux)
        self.aux_warmup_epochs = int(args.aux_warmup_epochs)
        if not math.isfinite(self.lambda_aux) or self.lambda_aux < 0 or self.aux_warmup_epochs < 0:
            raise ValueError("Invalid auxiliary loss settings")
        self._aux_batch_size = int(args.batch_size)
        self._aux_steps_per_epoch = None
        self._aux_epoch_batch = self._aux_count = self._aux_total_steps = 0
        self._aux_stats = None
        self.aux_sampler = None
        if self.lambda_aux > 0:
            bank = None
            if self.aux_mode == "shnc":
                with open(args.aux_bank_path, "rb") as f:
                    bank = pickle.load(f)
            self.aux_sampler = AuxiliaryNegativeSampler(
                self.item_num, corpus.train_clicked_set, self.aux_mode, args.random_seed, bank)

    def _flush_aux_stats(self):
        if self._aux_count:
            values = (self._aux_stats / self._aux_count).detach().cpu().tolist()
            logging.info("[AuxNeg] epoch=%s source=%s samples=%d train_steps_total=%d "
                         "aux_loss=%.8f score_gap=%.8f valid_negative_rate=%.6f "
                         "semantic_fallback_rate=%.6f pool_valid_ratio=%.6f mean_lambda=%.8f",
                         getattr(self, "_cur_epoch", 0), self.aux_mode, self._aux_count,
                         self._aux_total_steps, *values)
        self._aux_stats, self._aux_count = None, 0

    def actions_before_epoch(self):
        self._flush_aux_stats()
        self._aux_epoch_batch = 0

    def forward(self, feed_dict, return_intermediate=False):
        if not self.training:
            self._flush_aux_stats()
        if not self.training or self.lambda_aux == 0:
            return super().forward(feed_dict, return_intermediate)
        # Parent forward owns all dropout, prediction and original relation-ID sampling.
        out = super().forward(feed_dict, True)
        ids = feed_dict["aux_neg_items"]
        if ids.shape != (out["prediction"].shape[0], 1):
            raise ValueError("Exactly one auxiliary negative per record is required")
        out["_aux_scores"] = (out["user_vector"][:, None, :] * self.item_encoder(ids)).sum(-1)
        out["_aux_fallback"] = feed_dict["aux_fallback"]
        out["_aux_pool_ratio"] = feed_dict["aux_pool_ratio"]
        if not return_intermediate:
            out = {k: v for k, v in out.items()
                   if k == "prediction" or k == "_relation_ids" or k.startswith("_aux_")}
        return out

    def loss(self, out_dict):
        total = super().loss(out_dict)
        if not self.training or self.lambda_aux == 0:
            return total
        if not self._aux_steps_per_epoch:
            raise RuntimeError("Train Dataset must establish steps per epoch")
        gap = out_dict["prediction"][:, :1] - out_dict["_aux_scores"]
        per_sample = F.softplus(-gap).reshape(-1)
        progress = (getattr(self, "_cur_epoch", 1) - 1
                    + (self._aux_epoch_batch + 1) / self._aux_steps_per_epoch)
        scale = 1. if self.aux_warmup_epochs == 0 else min(progress / self.aux_warmup_epochs, 1.)
        weight = self.lambda_aux * scale
        aux_loss = per_sample.mean()
        total = total + weight * aux_loss
        count = per_sample.numel()
        stats = torch.stack((per_sample.detach().sum(), gap.detach().sum(),
                             gap.new_tensor(float(count)),
                             out_dict["_aux_fallback"].detach().sum(),
                             out_dict["_aux_pool_ratio"].detach().sum(),
                             gap.new_tensor(weight * count)))
        self._aux_stats = stats if self._aux_stats is None else self._aux_stats + stats
        self._aux_count += count
        self._aux_epoch_batch += 1
        self._aux_total_steps += 1
        out_dict["loss_aux"] = aux_loss.detach()
        out_dict["aux_weight"] = weight
        return total

    class Dataset(LLMMIRecASPCF.Dataset):
        def __init__(self, model, corpus, phase):
            super().__init__(model, corpus, phase)
            if phase == "train":
                model._aux_steps_per_epoch = math.ceil(len(self) / model._aux_batch_size)

        def _get_feed_dict(self, index):
            feed = super()._get_feed_dict(index)
            if self.phase == "train" and self.model.lambda_aux > 0:
                negative, fallback, ratio = self.model.aux_sampler.sample(
                    feed["user_id"], feed["item_id"][0], feed["history_items"],
                    feed["item_id"][1:], getattr(self.model, "_cur_epoch", 1), index)
                feed["aux_neg_items"] = np.array([negative], dtype=np.int64)
                feed["aux_fallback"] = np.float32(fallback)
                feed["aux_pool_ratio"] = np.float32(ratio)
            return feed
