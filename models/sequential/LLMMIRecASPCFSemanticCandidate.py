"""Candidate prototype: train-only LLM-conditioned interest competition on frozen ASPCF paths."""
import logging,math,pickle
from pathlib import Path
import numpy as np
import torch
from models.sequential.LLMMIRecASPCFSampledSoftmax import LLMMIRecASPCFSampledSoftmax
from models.sequential.semantic_candidate_components import semantic_responsibility, interest_competition


class LLMMIRecASPCFSemanticCandidate(LLMMIRecASPCFSampledSoftmax):
    extra_log_args = LLMMIRecASPCFSampledSoftmax.extra_log_args + [
        "semantic_candidate_path", "lambda_interest_candidate", "responsibility_temperature"]

    @staticmethod
    def parse_model_args(parser):
        parser = LLMMIRecASPCFSampledSoftmax.parse_model_args(parser)
        parser.add_argument("--semantic_candidate_path", default="")
        parser.add_argument("--lambda_interest_candidate", type=float, default=.01)
        parser.add_argument("--responsibility_temperature", type=float, default=.1)
        return parser

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.lambda_interest_candidate = float(args.lambda_interest_candidate)
        self.responsibility_temperature = float(args.responsibility_temperature)
        if (not math.isfinite(self.lambda_interest_candidate) or self.lambda_interest_candidate < 0
                or not math.isfinite(self.responsibility_temperature) or self.responsibility_temperature <= 0):
            raise ValueError("Invalid semantic competition settings")
        path = Path(args.semantic_candidate_path).resolve()
        if not path.is_relative_to(Path(__file__).resolve().parents[2]):
            raise ValueError("Prototype asset outside project")
        with path.open("rb") as f:
            asset = pickle.load(f)
        q = np.asarray(asset["soft_assignments"], dtype=np.float32)
        if (q.shape != (self.item_num, 32) or not np.isfinite(q).all() or np.any(q < 0)
                or np.any(q[0] != 0) or not np.allclose(q[1:].sum(1), 1, atol=1e-5)
                or asset.get("semantic_rank") != self.semantic_rank):
            raise ValueError("Invalid item-aligned semantic asset")
        self.register_buffer("semantic_candidate_q", torch.from_numpy(q), persistent=False)
        self._candidate_stats = None
        self._candidate_gradient_epoch = None
        self._candidate_query_grad = None
        logging.info("[SemanticCandidate] no new parameters; lambda=%s responsibility_temperature=%s score_temperature=1",
                     self.lambda_interest_candidate, self.responsibility_temperature)

    def forward(self, feed_dict, return_intermediate=False):
        if not self.training or self.lambda_interest_candidate == 0 or self.K == 1:
            return super().forward(feed_dict, return_intermediate)
        out = super().forward(feed_dict, return_intermediate=True)
        candidate_q = self.semantic_candidate_q[feed_dict["item_id"][:, 0]]
        profile, responsibility, confidence = semantic_responsibility(
            out["attention_maps"], self.semantic_candidate_q[feed_dict["history_items"]],
            candidate_q, feed_dict["lengths"], self.responsibility_temperature)
        scores = torch.einsum("bkd,jd->bkj", out["interest_vectors"], out["candidate_vectors"][:, 0])
        aux, details = interest_competition(scores, out["_ss_allowed"], responsibility, confidence)
        out["_candidate_aux"] = aux
        out["_candidate_details"] = details
        if return_intermediate:
            out.update(semantic_profile=profile, candidate_responsibility=responsibility,
                       responsibility_confidence=confidence)
        else:
            for key in list(out):
                if key not in ("prediction", "_relation_ids") and not key.startswith("_"):
                    del out[key]
        return out

    def loss(self, out):
        total = super().loss(out)
        if "_candidate_aux" not in out:
            return total
        aux = out["_candidate_aux"]
        total = total + self.lambda_interest_candidate * aux
        out["loss_interest_candidate"] = aux.detach()
        details = out["_candidate_details"]
        r = details["positive_r"]
        B = r.shape[0]
        values = torch.stack([
            aux.detach(), details["positive_confidence"].mean(),
            -(r * r.clamp_min(1e-12).log()).sum(1).mean(), r.max(1).values.mean(),
            details["active"].float().mean(), details["negative_mass"].mean()
        ]).detach()
        if self._candidate_stats is None:
            self._candidate_stats = torch.zeros(7, device=values.device, dtype=torch.float64)
        self._candidate_stats[:6] += values * B
        self._candidate_stats[6] += B
        # One extra partial backward per epoch; no optimizer or RNG changes, no CPU sync.
        epoch = getattr(self, "cur_epoch", 0)
        if self._candidate_gradient_epoch != epoch and torch.is_grad_enabled():
            grad = torch.autograd.grad(aux, self.extractor.query, retain_graph=True)[0]
            self._candidate_query_grad = grad.detach().norm()
            self._candidate_gradient_epoch = epoch
        return total

    def _flush_candidate(self):
        if self._candidate_stats is not None:
            s = self._candidate_stats.cpu().tolist()
            grad = float(self._candidate_query_grad.cpu()) if self._candidate_query_grad is not None else 0.
            logging.info("[SemanticCandidateStats] epoch=%s rows=%d aux=%.9g pos_confidence=%.9g pos_entropy=%.7f pos_max=%.7f active_ratio=%.7f negative_mass=%.7f first_batch_aux_query_grad=%.9g weighted_grad=%.9g",
                         self._ss_epoch, int(s[6]), *[v / max(s[6], 1) for v in s[:6]],
                         grad, self.lambda_interest_candidate * grad)
            self._candidate_stats = None
            self._candidate_query_grad = None

    def actions_before_epoch(self):
        self._flush_candidate()
        super().actions_before_epoch()

    def actions_after_train(self):
        self._flush_candidate()
        super().actions_after_train()
