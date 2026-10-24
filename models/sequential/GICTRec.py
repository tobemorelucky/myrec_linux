"""Chapter 5 GICTRec: inherits the frozen Chapter 4 C1 backbone."""
from pathlib import Path
import numpy as np
from models.sequential.LLMMIRecASPCFAdaptiveSIL import array_sha, file_sha
import torch
from models.sequential.LLMMIRecASPCFAdaptiveSIL import LLMMIRecASPCFAdaptiveSIL
from models.sequential.gict_components import SemanticInterestIndex, ReliableInterestTransfer, interest_profiles


class GICTRec(LLMMIRecASPCFAdaptiveSIL):
    @staticmethod
    def parse_model_args(parser):
        parser = LLMMIRecASPCFAdaptiveSIL.parse_model_args(parser)
        parser.add_argument("--gict_variant", choices=["G0", "G1", "G2", "G3"], default="G3")
        parser.add_argument("--gict_neighbors", type=int, default=8)
        parser.add_argument("--gict_temperature", type=float, default=.1)
        return parser

    def __init__(self, args, corpus):
        if getattr(args, "sil_mode", "fixed") != "fixed":
            raise ValueError("GICTRec requires the unchanged fixed-segment C1")
        super().__init__(args, corpus)
        self.gict_variant = getattr(args, "gict_variant", "G3")
        self.gict_neighbors = int(getattr(args, "gict_neighbors", 8))
        if self.gict_variant not in ("G0", "G1", "G2", "G3") or self.gict_neighbors < 1:
            raise ValueError("Invalid GICT configuration")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261024)
            self.transfer = ReliableInterestTransfer(self.emb_size, float(getattr(args, "gict_temperature", .1)))
        self.semantic_index = None
        self._gict_train_sha = array_sha(corpus.data_df["train"][["user_id","item_id","position"]].to_numpy(np.int64))
        self._embedding_sha = file_sha(args.llm_emb_path)

    def attach_bank(self, payload, allow_sample=False):
        meta = payload["metadata"]
        if meta.get("scope") != "full_train" and not allow_sample:
            raise ValueError("Sample bank is for structural checks only")
        if meta["K"] != self.K or meta["history_max"] != self.max_his:
            raise ValueError("Teacher/student K or history cap mismatch")
        if meta["train_frame_sha256"] != self._train_frame_sha():
            raise ValueError("Bank not from this Train frame")
        if meta["embedding_sha256"] != self._embedding_sha:
            raise ValueError("Bank semantic table mismatch")
        self.semantic_index = SemanticInterestIndex(payload).to(self.device)

    def _train_frame_sha(self):
        return self._gict_train_sha

    def load_c1(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing != ["transfer.output.weight"] or unexpected:
            raise ValueError(("Not an exact C1 state", missing, unexpected))
        torch.nn.init.zeros_(self.transfer.output.weight)

    def forward(self, feed_dict, return_intermediate=False):
        out = super().forward(feed_dict, True)
        if self.gict_variant != "G0":
            bank = self.semantic_index
            if bank is None:
                raise RuntimeError("Attach a verified frozen Train-only teacher bank")
            history, lengths = feed_dict["history_items"], feed_dict["lengths"]
            if history.shape[1] == 0:
                history = history.new_zeros((len(lengths), 1))
            profiles, support = interest_profiles(bank.item_probabilities, history, out["attention_maps"], lengths)
            ids, valid = bank.retrieve(history, lengths, feed_dict["user_id"], self.gict_neighbors)
            # An empty library is represented by one fully masked sentinel row.
            updated, info = self.transfer(out["interest_vectors"], profiles, support,
                bank.interest_vectors[ids], bank.interest_profiles[ids], bank.interest_support[ids],
                bank.user_vectors[ids], valid, self.gict_variant)
            user = (updated * out["interest_weights"][:, :, None]).sum(1)
            raw = (user[:, None] * out["candidate_vectors"]).sum(-1)
            out.update(interest_vectors=updated, user_vector=user, raw_prediction=raw,
                       prediction=self.corrected_scores(raw, feed_dict["item_id"]),
                       semantic_interest_profiles=profiles, gict_neighbor_users=bank.user_ids[ids],
                       gict_neighbor_valid=valid, **info)
            if self.training:
                out["_ss_scores"] = user @ out["candidate_vectors"][:, 0].T
        if not return_intermediate:
            out = {k:v for k,v in out.items()
                   if k in ("prediction", "_relation_ids") or k.startswith("_ss_")}
        return out
