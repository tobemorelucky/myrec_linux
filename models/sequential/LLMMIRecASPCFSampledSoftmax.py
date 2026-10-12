"""Loss-only ASPCF baseline: filtered in-batch positive-target softmax."""
import logging
import numpy as np
import torch
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF


def build_train_membership(corpus):
    # Compact train-only bitset, constructed once; no RNG and no per-batch CPU queries.
    words = (corpus.n_items + 63) // 64
    packed = np.zeros((corpus.n_users, words), dtype=np.uint64)
    for user, items in corpus.train_clicked_set.items():
        ids = np.asarray(list(items), dtype=np.int64)
        if ids.size:
            if user < 0 or user >= corpus.n_users or (ids <= 0).any() or (ids >= corpus.n_items).any():
                raise ValueError("Invalid train user/item IDs")
            np.bitwise_or.at(packed[user], ids // 64,
                             np.left_shift(np.uint64(1), (ids % 64).astype(np.uint64)))
    return torch.from_numpy(packed.view(np.int64))


def candidate_mask(membership, users, positives):
    B = positives.numel()
    same = positives[:, None] == positives[None, :]
    clicked = ((membership[users[:, None], positives[None, :] // 64]
                >> (positives[None, :] % 64)) & 1).bool()
    diagonal = torch.eye(B, dtype=torch.bool, device=positives.device)
    duplicate = same & ~diagonal
    clicked_only = clicked & ~same & ~diagonal
    allowed = ~(same | clicked | (positives[None, :] == 0))
    allowed = allowed | diagonal
    return allowed, duplicate, clicked_only


def masked_softmax_loss(scores, allowed, temperature):
    logits = scores / temperature
    masked = logits.masked_fill(~allowed, -torch.inf)
    per_row = torch.logsumexp(masked, dim=1) - logits.diagonal()
    valid = allowed.sum(dim=1) > 1
    # All-invalid batch yields graph-connected zero; relation loss still applies.
    return (per_row * valid).sum() / valid.sum().clamp_min(1), valid


class LLMMIRecASPCFSampledSoftmax(LLMMIRecASPCF):
    extra_log_args = LLMMIRecASPCF.extra_log_args + ["sampled_softmax_temperature"]

    @staticmethod
    def parse_model_args(parser):
        parser = LLMMIRecASPCF.parse_model_args(parser)
        parser.add_argument("--sampled_softmax_temperature", type=float, default=1.0)
        return parser

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        if args.sampled_softmax_temperature <= 0 or not np.isfinite(args.sampled_softmax_temperature):
            raise ValueError("Temperature must be positive and finite")
        self.sampled_softmax_temperature = args.sampled_softmax_temperature
        self.register_buffer("train_membership", build_train_membership(corpus), persistent=False)
        self._ss_stats = None
        self._ss_epoch = None
        logging.info("[SampledSoftmax] train-only membership bytes=%d; no new parameters",
                     self.train_membership.numel() * self.train_membership.element_size())

    def forward(self, feed_dict, return_intermediate=False):
        if not self.training:
            return super().forward(feed_dict, return_intermediate)
        # Preserve original candidates and original relation ID collection/RNG.
        out = super().forward(feed_dict, return_intermediate=True)
        positives = feed_dict["item_id"][:, 0]
        allowed, duplicate, clicked = candidate_mask(self.train_membership, feed_dict["user_id"], positives)
        out["_ss_scores"] = out["user_vector"] @ out["candidate_vectors"][:, 0, :].T
        out["_ss_allowed"] = allowed
        out["_ss_duplicate"] = duplicate.sum()
        out["_ss_clicked"] = clicked.sum()
        if not return_intermediate:
            for key in list(out):
                if key not in ("prediction", "_relation_ids") and not key.startswith("_ss_"):
                    del out[key]
        return out

    def _ranking_loss(self, scores, allowed):
        return masked_softmax_loss(scores, allowed, self.sampled_softmax_temperature)

    def loss(self, out):
        if "_ss_scores" not in out:
            return super().loss(out)
        ranking, valid = self._ranking_loss(out["_ss_scores"], out["_ss_allowed"])
        total = ranking
        if "_relation_ids" in out and self.lambda_relation > 0:
            rel = self._compute_relation_loss(out["_relation_ids"])
            total = total + self.lambda_relation * rel
            out["loss_relation"] = rel.detach()
        out["loss_ranking"] = ranking.detach()
        negatives = out["_ss_allowed"].sum(dim=1) - 1
        B = negatives.numel()
        stats = torch.stack([
            negatives.new_tensor(B), valid.sum(), negatives.sum(),
            negatives.new_tensor(B * (B - 1)), out["_ss_duplicate"], out["_ss_clicked"],
            negatives.min(), negatives.max(), ranking.detach() * B
        ]).detach().to(torch.float64)
        if self._ss_stats is None:
            self._ss_stats = stats
        else:
            self._ss_stats[:6] += stats[:6]
            self._ss_stats[6] = torch.minimum(self._ss_stats[6], stats[6])
            self._ss_stats[7] = torch.maximum(self._ss_stats[7], stats[7])
            self._ss_stats[8] += stats[8]
        return total

    def _flush_stats(self):
        if self._ss_stats is not None:
            rows, valid, negatives, slots, duplicate, clicked, minimum, maximum, loss = self._ss_stats.cpu().tolist()
            logging.info("[SampledSoftmaxStats] epoch=%s rows=%d valid_ratio=%.6f mean_neg=%.3f min_neg=%d max_neg=%d duplicate_mask=%.6f clicked_only_mask=%.6f ranking_loss=%.6f",
                         self._ss_epoch, rows, valid / max(rows, 1), negatives / max(rows, 1),
                         minimum, maximum, duplicate / max(slots, 1), clicked / max(slots, 1), loss / max(rows, 1))
            self._ss_stats = None

    def actions_before_epoch(self):
        self._flush_stats()
        self._ss_epoch = self.cur_epoch

    def actions_after_train(self):
        self._flush_stats()
