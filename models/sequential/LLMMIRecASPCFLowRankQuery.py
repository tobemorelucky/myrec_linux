"""Controlled static/dynamic low-rank Query adaptation; no novelty claim."""
import logging
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.sequential.llmmi_components import QueryMultiInterestExtractor
from models.sequential.LLMMIRecASPCFSampledSoftmax import LLMMIRecASPCFSampledSoftmax


class LowRankQueryAdapter(nn.Module):
    def __init__(self, dim=64, rank=8):
        super().__init__()
        # Explicit CPU generator: no torch.manual_seed and no CUDA RNG reset.
        generator = torch.Generator(device="cpu").manual_seed(20261013)
        self.U = nn.Parameter(torch.zeros(dim, rank, device="cpu"))
        self.V = nn.Parameter(torch.empty(rank, dim, device="cpu"))
        self.context = nn.Linear(dim, rank, device="cpu")
        nn.init.normal_(self.V, std=.01, generator=generator)
        nn.init.normal_(self.context.weight, std=.01, generator=generator)
        nn.init.zeros_(self.context.bias)

    def correction(self, query, context):
        coordinates = F.linear(query, self.V)
        modulation = self.context(context).tanh()
        return F.linear(coordinates[None] * modulation[:, None, :], self.U)

    def forward(self, query, context):
        return query[None] + self.correction(query, context)


class LowRankQueryExtractor(QueryMultiInterestExtractor):
    def __init__(self, original, mode, rank):
        nn.Module.__init__(self)
        self.K, self.emb_size, self.attn_size = original.K, original.emb_size, original.attn_size
        self.query = original.query
        self.Wq, self.Wk, self.Wv = original.Wq, original.Wk, original.Wv
        self.mode = mode
        # nn.Linear construction uses CPU RNG, restored here. Local generator
        # initializes the added parameters; CUDA RNG remains untouched.
        with torch.random.fork_rng(devices=[]):
            self.low_rank = LowRankQueryAdapter(self.emb_size, rank)

    def query_context(self, history_emb, lengths):
        B, L, D = history_emb.shape
        if self.mode == "static":
            context = F.layer_norm(self.query.mean(0), (D,))[None].expand(B, -1)
        else:
            if L == 0:
                context = history_emb.new_zeros(B, D)
            else:
                valid = torch.arange(L, device=lengths.device)[None] < lengths[:, None]
                masked = history_emb.masked_fill(~valid[..., None], 0)
                mean = masked.sum(1) / lengths[:, None].clamp_min(1)
                last = masked[torch.arange(B, device=lengths.device), (lengths - 1).clamp_min(0)]
                context = F.layer_norm(mean + last, (D,))
        return context

    def make_query(self, history_emb, lengths):
        return self.low_rank(self.query, self.query_context(history_emb, lengths))

    def forward(self, history_emb, lengths):
        B, L, _ = history_emb.shape
        delta = self.low_rank.correction(self.query, self.query_context(history_emb, lengths))
        # Preserve the original shared Query projection/reduction path exactly.
        # Wq(q + delta) = Wq(q) + Wq.weight @ delta; bias appears once.
        projected = self.Wq(self.query)[None].expand(B, -1, -1)
        projected = projected + F.linear(delta, self.Wq.weight, bias=None)
        keys, values = self.Wk(history_emb), self.Wv(history_emb)
        scores = torch.bmm(projected, keys.transpose(1, 2)) / math.sqrt(self.attn_size)
        valid = torch.arange(L, device=lengths.device)[None] < lengths[:, None]
        attention = scores.masked_fill(~valid[:, None, :], -torch.inf).softmax(-1)
        attention = attention.masked_fill(torch.isnan(attention), 0)
        return torch.bmm(attention, values), attention


class LLMMIRecASPCFLowRankQuery(LLMMIRecASPCFSampledSoftmax):
    extra_log_args = LLMMIRecASPCFSampledSoftmax.extra_log_args + [
        "query_adaptation", "query_rank"]

    @staticmethod
    def parse_model_args(parser):
        parser = LLMMIRecASPCFSampledSoftmax.parse_model_args(parser)
        parser.add_argument("--query_adaptation", choices=["static", "dynamic"], default="dynamic")
        parser.add_argument("--query_rank", type=int, default=8)
        return parser

    def __init__(self, args, corpus):
        if args.history_max != 20 or args.item_encoder != "aspcf":
            raise ValueError("Requires unchanged ASPCF encoder and H20")
        if not 0 < args.query_rank <= args.emb_size:
            raise ValueError("Invalid query rank")
        super().__init__(args, corpus)
        self.extractor = LowRankQueryExtractor(
            self.extractor, args.query_adaptation, args.query_rank)
        logging.info("[LowRankQuery] mode=%s rank=%d added=%d total=%d U zero init; "
                     "CPU/CUDA RNG preserved; no teacher/new loss",
                     args.query_adaptation, args.query_rank,
                     sum(p.numel() for p in self.extractor.low_rank.parameters()),
                     self.count_variables())
