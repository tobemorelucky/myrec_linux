"""Thin joint/frozen construction adapter; all readout math stays in legacy Probe."""
import torch
from tools.run_frozen_interest_readout import Probe, Base

class LLMMIRecASPCFReadoutBPR(Probe):
    reader, runner = Base.reader, Base.runner
    Dataset = Base.Dataset

    @staticmethod
    def parse_model_args(parser):
        parser=Base.parse_model_args(parser)
        parser.add_argument('--readout_mode',choices=['P0','P1'],required=True)
        parser.add_argument('--backbone_mode',choices=['frozen','joint'],default='joint')
        return parser

    def __init__(self,args,corpus):
        # No checkpoint loading. All original initialization is done once.
        base=Base(args,corpus)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(42)
            super().__init__(base,args.readout_mode,
                             backbone_mode=getattr(args,'backbone_mode','joint'))
