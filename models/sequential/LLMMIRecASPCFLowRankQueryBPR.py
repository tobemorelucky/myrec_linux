"""A0/A1 Query-only candidate on the original single-negative ASPCF BPR."""
import logging
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF
from models.sequential.LLMMIRecASPCFLowRankQuery import LowRankQueryExtractor


class LLMMIRecASPCFLowRankQueryBPR(LLMMIRecASPCF):
    extra_log_args = LLMMIRecASPCF.extra_log_args + ["query_adaptation", "query_rank"]

    @staticmethod
    def parse_model_args(parser):
        parser = LLMMIRecASPCF.parse_model_args(parser)
        parser.add_argument("--query_adaptation", choices=["static", "dynamic"], default="dynamic")
        parser.add_argument("--query_rank", type=int, default=8)
        return parser

    def __init__(self, args, corpus):
        if args.num_neg != 1 or args.history_max != 20 or args.item_encoder != "aspcf":
            raise ValueError("Requires original ASPCF / single-negative BPR / H20")
        if not 0 < args.query_rank <= args.emb_size:
            raise ValueError("Invalid query rank")
        super().__init__(args, corpus)
        self.extractor = LowRankQueryExtractor(self.extractor, args.query_adaptation, args.query_rank)
        logging.info("[LowRankQueryBPR] mode=%s rank=%d added=%d total=%d; "
                     "original GeneralModel.loss + ASPCF relation; no in-batch scores/membership",
                     args.query_adaptation, args.query_rank,
                     sum(p.numel() for p in self.extractor.low_rank.parameters()),
                     self.count_variables())
