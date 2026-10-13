"""A/B attribution controls; original full dual-view model remains unchanged."""
import logging
import torch
from torch import nn
import torch.nn.functional as F
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF
from models.sequential.LLMMIRecASPCFDualViewRefine import LLMMIRecASPCFDualViewRefine
from models.sequential.LLMMIRecASPCFSampledSoftmax import candidate_mask


class UserVectorRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        # Closest integer GLU width to 28864: 28776 active parameters.
        self.input = nn.Linear(64, 296)
        self.output = nn.Linear(148, 64)
        nn.init.normal_(self.input.weight, std=.01)
        nn.init.normal_(self.input.bias, std=.01)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, user):
        return self.output(F.glu(self.input(user), dim=-1))


class LLMMIRecASPCFUserRefine(LLMMIRecASPCFDualViewRefine):
    """A: only the already aggregated 64-D user vector enters the new network."""
    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261012)
            self.refiner = UserVectorRefiner()
        logging.info("[RefineControl A] user-only64->296 GLU148->64 added=%d total=%d; "
                     "constructor full-model parameter log superseded",
                     sum(p.numel() for p in self.refiner.parameters()), self.count_variables())

    def forward(self, feed_dict, return_intermediate=False):
        out = LLMMIRecASPCF.forward(self, feed_dict, return_intermediate=True)
        base_user = out["user_vector"]
        residual = self.refiner(base_user)
        user = base_user + residual
        out["user_vector"] = user
        out["prediction"] = (user[:, None, :] * out["candidate_vectors"]).sum(-1)
        if self.training:
            positives = feed_dict["item_id"][:, 0]
            allowed, duplicate, clicked = candidate_mask(
                self.train_membership, feed_dict["user_id"], positives)
            out["_ss_scores"] = user @ out["candidate_vectors"][:, 0, :].T
            out["_ss_allowed"] = allowed
            out["_ss_duplicate"] = duplicate.sum()
            out["_ss_clicked"] = clicked.sum()
        if return_intermediate:
            out.update(base_user_vector=base_user, user_residual=residual,
                       refinement_features=base_user)
        else:
            for key in list(out):
                if key not in ("prediction", "_relation_ids") and not key.startswith("_ss_"):
                    del out[key]
        return out


class LLMMIRecASPCFSliceRefine(LLMMIRecASPCFDualViewRefine):
    """B: exact full-model network; explicit S/C replaced by V[:32]/V[32:]."""
    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        logging.info("[RefineControl B] explicit raw S/C replaced by transformed V slices; "
                     "same 192-D features/network/initialization/parameters")

    def _refine(self, out, history, lengths):
        raw = out["history_vectors"]
        B, L, _ = raw.shape
        valid = ((torch.arange(L, device=raw.device)[None, :] < lengths[:, None])
                 & (history != 0))
        masked_raw = raw.masked_fill(~valid[:, :, None], 0)
        mean = masked_raw.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        last = masked_raw[torch.arange(B, device=raw.device), (lengths-1).clamp_min(0)]
        context = self.aggregator.ln(mean + last)
        V = out["interest_vectors"]
        slot_S = V[..., :32]
        slot_C = V[..., 32:]
        features = torch.cat([V, slot_S, slot_C,
                              context[:, None, :].expand(-1, self.K, -1)], dim=-1)
        residual = self.refiner(features).masked_fill((lengths == 0)[:, None, None], 0)
        refined = V + residual
        user = (refined * out["interest_weights"][:, :, None]).sum(1)
        # Parent intermediate interest_semantic/complement fields denote control
        # slots here, not actual semantic/complement spaces of dense Wv.
        return user, refined, residual, slot_S, slot_C, context, features
