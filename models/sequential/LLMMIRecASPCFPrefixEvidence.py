"""Prefix-only semantic evidence routing; independent Chapter-4 research prototype.

A1/A2 share all parameters and initialization. Evidence is a history kernel
contrast, NOT an auxiliary contrastive loss, calibrated likelihood or causal label.
No cross-window state, candidate-conditioned routing, teacher or new loss.
"""
import copy
import logging
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF


class PrefixEvidenceTracker(nn.Module):
    def __init__(self, K=4, dim=64, tau=.2, kappa=1., strength=1.,
                 inertia=.5, mixing=.1, eps=1e-8, semantic_eps=1e-8):
        super().__init__()
        if K < 1 or dim < 1 or tau <= 0 or kappa <= 0:
            raise ValueError("Positive K, dim, tau and kappa required")
        if not 0 < mixing <= 1 or inertia < 0 or strength < 0 or eps <= 0:
            raise ValueError("Invalid recurrence constants")
        self.K, self.dim = K, dim
        self.tau, self.kappa = tau, kappa
        self.strength, self.inertia, self.mixing = strength, inertia, mixing
        self.eps, self.semantic_eps = eps, semantic_eps
        self.query = nn.Parameter(torch.empty(K, dim))
        self.We = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        # Avoid the old product of multiple tiny initialized routing factors.
        nn.init.normal_(self.query, std=1.)
        nn.init.xavier_uniform_(self.We.weight)
        nn.init.zeros_(self.We.bias)

    def semantic_kernel(self, semantics, valid):
        # Original LLM buffer is fixed; no learned semantic teacher/online LLM.
        z = semantics.detach()
        norms = z.norm(dim=-1)
        semantic_valid = valid & (norms > self.semantic_eps)
        unit = z / norms.clamp_min(self.semantic_eps).unsqueeze(-1)
        cosine = torch.bmm(unit, unit.transpose(1, 2)).clamp(-1., 1.)
        L = z.shape[1]
        causal = torch.ones(L, L, device=z.device, dtype=torch.bool).tril(-1)
        pairs = semantic_valid.unsqueeze(2) & semantic_valid.unsqueeze(1) & causal
        kernel = torch.exp((cosine-1.)/self.tau) * pairs.to(z.dtype)
        return kernel, pairs

    def evidence(self, past, kernel_row, pair_valid):
        """past [B,t,K]; kernel_row/pair_valid [B,t]; original formula."""
        mask = pair_valid.to(past.dtype)
        count = mask.sum(-1, keepdim=True)
        r = kernel_row * mask
        mu = r.sum(-1, keepdim=True) / count.clamp_min(1.)
        plus_mass = (past * mask.unsqueeze(-1)).sum(1)
        minus_mass = ((1.-past) * mask.unsqueeze(-1)).sum(1)
        # Algebraically identical centered form. Constant/uniform assignments
        # yield exact zero covariance instead of cancellation of two ratios.
        mean_p = plus_mass / count.clamp_min(1.)
        covariance_sum = ((past-mean_p.unsqueeze(1))*r.unsqueeze(-1)).sum(1)
        plus = mu + covariance_sum/(self.kappa+plus_mass)
        minus = mu - covariance_sum/(self.kappa+minus_mass)
        plus, minus = plus.clamp_min(0.), minus.clamp_min(0.)
        d = torch.log(plus+self.eps)-torch.log(minus+self.eps)
        # With 0 or 1 semantic predecessor evidence is mathematically zero.
        # K1 is also exactly neutral; avoid roundoff masquerading as evidence.
        enabled = (count > 1) & (self.K > 1)
        return torch.where(enabled, d, torch.zeros_like(d))

    def forward(self, embeddings, lengths, semantics, use_evidence=True):
        B, L, D = embeddings.shape
        if D != self.dim or semantics.shape[:2] != (B,L) or lengths.shape != (B,):
            raise ValueError("Expected embeddings [B,L,D], semantics [B,L,S], lengths [B]")
        valid = torch.arange(L, device=embeddings.device)[None] < lengths[:,None]
        values = self.Wv(embeddings)
        scores = torch.einsum("bld,kd->blk", self.We(embeddings), self.query) / math.sqrt(D)
        kernel, pairs = (self.semantic_kernel(semantics, valid) if use_evidence
                         else (None,None))
        state = embeddings.new_full((B,self.K), 1./self.K)
        assignments, evidence_rows, state_rows = [], [], []
        for t in range(L):  # H<=20; vectorized over users, slots and predecessors.
            if use_evidence and t:
                past = torch.stack(assignments, dim=1)
                evidence = self.evidence(past, kernel[:,t,:t], pairs[:,t,:t])
            else:
                evidence = torch.zeros_like(state)
            prior = (1.-self.mixing)*state + self.mixing/self.K
            logits = scores[:,t] + self.strength*evidence + self.inertia*torch.log(prior)
            p = torch.softmax(logits, dim=-1)
            if self.K == 1:
                p = torch.ones_like(p)  # Exact A1/A2 K1 routing degeneration.
            active = valid[:,t,None]
            state = torch.where(active, p, state)
            assignments.append(torch.where(active,p,torch.zeros_like(p)))
            evidence_rows.append(torch.where(active,evidence,torch.zeros_like(evidence)))
            state_rows.append(state)
        if L:
            p = torch.stack(assignments,1)
            evidence = torch.stack(evidence_rows,1)
            states = torch.stack(state_rows,1)
        else:
            p = embeddings.new_zeros((B,0,self.K))
            evidence = p
            states = p
        mass = p.sum(1)
        # Exact normalized pooling for nonempty slots; zero for empty slots.
        attention = p.transpose(1,2) / mass.clamp_min(self.eps).unsqueeze(-1)
        interests = torch.bmm(attention,values)
        return dict(interests=interests, attention=attention, assignments=p,
                    evidence=evidence, mass=mass, states=states, final_state=state)


class LLMMIRecASPCFPrefixEvidence(LLMMIRecASPCF):
    extra_log_args = LLMMIRecASPCF.extra_log_args + ["prefix_variant"]

    @staticmethod
    def parse_model_args(parser):
        parser = LLMMIRecASPCF.parse_model_args(parser)
        parser.add_argument("--prefix_variant", choices=["A1","A2"], default="A2")
        return parser

    def __init__(self,args,corpus):
        if (args.item_encoder != "aspcf" or args.aspcf_gate_mode != "basic"
            or args.num_neg != 1 or args.history_max != 20
            or args.emb_size != 64 or args.attn_size != 64):
            raise ValueError("Requires frozen ASPCF definition, single BPR, D64/A64/H20")
        super().__init__(args,corpus)
        self.prefix_variant = args.prefix_variant
        # Construct the unmodified base first. Preserve shared cold-start tensors
        # and both CPU/CUDA RNG states; only the new routing parameters differ.
        original_value = copy.deepcopy(self.extractor.Wv.state_dict())
        with torch.random.fork_rng(devices=[]):
            tracker = PrefixEvidenceTracker(self.K,self.emb_size)
            tracker.Wv.load_state_dict(original_value)
        self.extractor = tracker
        # Reverse positions depend on complete-window length. This prefix tracker
        # deliberately uses raw e_t and recurrence, with no unused position table.
        del self.position_emb
        logging.info("[PrefixEvidence] variant=%s params=%d constants=%s",
                     self.prefix_variant,self.count_variables(),
                     dict(tau=.2,kappa=1.,strength=1.,inertia=.5,mixing=.1,
                          query_std=1.,key_init="Xavier",state_scope="H20 only"))

    def forward(self,feed_dict,return_intermediate=False):
        history, lengths = feed_dict["history_items"], feed_dict["lengths"]
        candidate_ids = feed_dict["item_id"]
        valid = torch.arange(history.shape[1],device=history.device)[None] < lengths[:,None]
        raw = self.item_encoder(history) * valid.unsqueeze(-1)
        candidates = self.item_encoder(candidate_ids)
        semantics = (self.item_encoder.llm_table[:, :512][history]
                     if self.prefix_variant=="A2" else raw[..., :1].detach())
        tracked = self.extractor(self.dropout(raw),lengths,semantics,
                                 use_evidence=self.prefix_variant=="A2")
        interests = self.dropout(tracked["interests"])
        # Base aggregator supports length zero if at least one masked position.
        agg_raw = raw if raw.shape[1] else raw.new_zeros(raw.shape[0],1,64)
        weights = self.aggregator(agg_raw,lengths)
        user = (interests*weights.unsqueeze(-1)).sum(1)
        prediction = (user[:,None,:]*candidates).sum(-1)
        out = {"prediction":prediction}
        # Same original IDs, sampling and inherited relation loss.
        if self.training and self.lambda_relation > 0:
            ids = torch.unique(torch.cat([history.reshape(-1),candidate_ids.reshape(-1)]))
            ids = ids[ids!=0]
            if ids.numel()>self.relation_sample_size:
                ids = ids[torch.randperm(ids.numel(),device=history.device)[:self.relation_sample_size]]
            out["_relation_ids"] = ids
        if return_intermediate:
            out.update(interest_vectors=interests,attention_maps=tracked["attention"],
                       interest_weights=weights,user_vector=user,history_vectors=raw,
                       candidate_vectors=candidates,prefix_assignments=tracked["assignments"],
                       semantic_evidence=tracked["evidence"],interest_mass=tracked["mass"],
                       prefix_states=tracked["states"],final_state=tracked["final_state"])
        return out
