"""Train-only two-future-label BPR attribution; unchanged ASPCF serving path.
Known multi-positive exclusive matching adapted from prior work, not a novelty claim.
"""
import math
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF


def ordered_head_pairs(K, device=None):
    heads = torch.arange(K, device=device)
    pairs = torch.cartesian_prod(heads, heads).reshape(-1, 2)
    return pairs[pairs[:, 0] != pairs[:, 1]]


def exclusive_assignment(margins, keys):
    """Exact maximum TOTAL margin, not minimum sum-softplus; no matching gradient.
    Exact ties use a deterministic row-key rotation, without score noise or RNG.
    """
    B, K, M = margins.shape
    if M != 2 or K < 2:
        raise ValueError("Two labels and at least two heads required")
    pairs = ordered_head_pairs(K, margins.device)
    scores = margins.detach()[:, pairs[:, 0], 0] + margins.detach()[:, pairs[:, 1], 1]
    best = scores.max(-1, keepdim=True).values
    ties = scores == best
    n = pairs.shape[0]
    priority = (torch.arange(n, device=margins.device)[None] - keys[:, None].remainder(n)).remainder(n)
    chosen = priority.masked_fill(~ties, n).argmin(-1)
    return pairs[chosen], ties.sum(-1) > 1


def auxiliary_bpr(vectors, user, positives, negative, valid, keys, mode):
    """Both controls use the SAME two targets, one reused negative and active rows."""
    B, K, _ = vectors.shape
    if mode not in ("extra", "exclusive"):
        raise ValueError(mode)
    if K == 1:
        zero = vectors.sum() * 0
        return zero, dict(valid=torch.zeros_like(valid), assignment=None, tied=torch.zeros_like(valid))
    delta = positives - negative[:, None, :]
    margins = torch.einsum("bkd,bmd->bkm", vectors, delta)
    if mode == "exclusive":
        assignment, tied = exclusive_assignment(margins, keys)
        selected = margins.gather(1, assignment[:, None, :]).squeeze(1)
    else:
        assignment, tied = None, torch.zeros_like(valid)
        selected = torch.einsum("bd,bmd->bm", user, delta)
    per_row = F.softplus(-selected).mean(-1)
    loss = per_row.masked_fill(~valid, 0).sum() / valid.sum().clamp_min(1)
    return loss, dict(valid=valid, assignment=assignment, tied=tied, margins=margins,
                      selected=selected)


class LLMMIRecASPCFMultiPositive(LLMMIRecASPCF):
    extra_log_args = LLMMIRecASPCF.extra_log_args + ["mp_mode", "lambda_aux"]

    @staticmethod
    def parse_model_args(parser):
        parser = LLMMIRecASPCF.parse_model_args(parser)
        parser.add_argument("--mp_mode", choices=["off", "extra", "exclusive"], default="off")
        parser.add_argument("--lambda_aux", type=float, default=0.)
        return parser

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.mp_mode = args.mp_mode
        self.lambda_aux = float(args.lambda_aux)
        if self.num_neg != 1:
            raise ValueError("This attribution requires original single-negative BPR")
        if not math.isfinite(self.lambda_aux) or self.lambda_aux < 0:
            raise ValueError("lambda_aux must be finite and nonnegative")
        # No new parameters, modules, random draws, buffers or persistent assets.

    def forward(self, feed_dict, return_intermediate=False):
        if (not self.training or self.mp_mode == "off" or self.lambda_aux == 0 or self.K < 2):
            return super().forward(feed_dict, return_intermediate)
        # Collect original history/candidates/relation IDs ONCE, before any extra target.
        out = super().forward(feed_dict, return_intermediate=True)
        extra = feed_dict["mp_extra_item"].long()
        ids = feed_dict["item_id"]
        valid = (feed_dict["mp_extra_valid"].bool() & (feed_dict["lengths"] > 0)
                 & (ids[:, 0] > 0) & (extra > 0) & (extra != ids[:, 0])
                 & (ids[:, 1] > 0) & (ids[:, 1] != ids[:, 0]) & (ids[:, 1] != extra))
        extra_vector = self.item_encoder(extra)  # Encode only the extra B items; no extractor rerun.
        positives = torch.stack([out["candidate_vectors"][:, 0], extra_vector], dim=1)
        aux, detail = auxiliary_bpr(out["interest_vectors"], out["user_vector"], positives,
                                    out["candidate_vectors"][:, 1], valid,
                                    feed_dict["mp_assignment_key"].long(), self.mp_mode)
        out["_mp_aux"] = aux
        out["_mp_valid"] = valid
        out["_mp_tied"] = detail["tied"]
        if detail["assignment"] is not None:
            out["_mp_assignment"] = detail["assignment"]
        if not return_intermediate:
            out = {k: v for k, v in out.items() if k in ("prediction", "_relation_ids") or k.startswith("_mp_")}
        return out

    def loss(self, out):
        # Exact original GeneralModel BPR + original relation sampling/computation.
        original = super().loss(out)
        if "_mp_aux" not in out:
            return original
        out["loss_auxiliary"] = out["_mp_aux"].detach()
        return original + self.lambda_aux * out["_mp_aux"]

    class Dataset(LLMMIRecASPCF.Dataset):
        def __init__(self, model, corpus, phase):
            super().__init__(model, corpus, phase)
            if phase != "train":
                return  # No future fields or target-window construction at evaluation.
            # Original row order and main labels stay untouched. Only Train rows read.
            u = np.asarray(self.data["user_id"], dtype=np.int64)
            pos = np.asarray(self.data["position"], dtype=np.int64)
            item = np.asarray(self.data["item_id"], dtype=np.int64)
            frame = pd.DataFrame(dict(user=u, position=pos, item=item, row=np.arange(len(u))))
            ordered = frame.sort_values(["user", "position"], kind="mergesort")
            if ordered.duplicated(["user", "position"]).any():
                raise ValueError("Ambiguous Train sequence positions")
            nxt = ordered.groupby("user", sort=False)[["item", "position"]].shift(-1)
            # Adjacent events only: never scan ahead to find a more convenient distinct item.
            ok = ((nxt.position == ordered.position + 1) & (nxt.item > 0)
                  & (ordered.item > 0) & (nxt.item != ordered.item)).to_numpy()
            extra = nxt.item.fillna(0).to_numpy(dtype=np.int64)
            self.mp_extra = np.zeros(len(u), dtype=np.int64)
            self.mp_valid = np.zeros(len(u), dtype=np.int64)
            rows = ordered.row.to_numpy(dtype=np.int64)
            self.mp_extra[rows] = np.where(ok, extra, 0)
            self.mp_valid[rows] = ok.astype(np.int64)
            self.mp_keys = u * np.int64(1103515245) + pos * np.int64(12345)

        def _get_feed_dict(self, index):
            feed = super()._get_feed_dict(index)
            if self.phase == "train":
                feed["mp_extra_item"] = self.mp_extra[index]
                feed["mp_extra_valid"] = self.mp_valid[index]
                feed["mp_assignment_key"] = self.mp_keys[index]
            return feed
