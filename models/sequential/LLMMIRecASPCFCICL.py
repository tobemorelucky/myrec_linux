"""CICL candidate: train coalition credit distilled to history-only serving weights."""
import logging,math,pickle
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF
from models.sequential.cicl_components import CoalitionTeacher,ContributionPredictor,corrected_weights

class LLMMIRecASPCFCICL(LLMMIRecASPCF):
    extra_log_args=LLMMIRecASPCF.extra_log_args+["lambda_contribution","cicl_proto_path"]
    @staticmethod
    def parse_model_args(parser):
        parser=LLMMIRecASPCF.parse_model_args(parser)
        parser.add_argument("--lambda_contribution",type=float,default=.01)
        parser.add_argument("--cicl_proto_path",default="")
        return parser

    def __init__(self,args,corpus):
        super().__init__(args,corpus)
        self.lambda_contribution=float(args.lambda_contribution)
        if (self.K!=4 or self.emb_size!=64 or self.semantic_rank!=512 or
            self.item_encoder_mode!="aspcf" or self.aspcf_gate_mode!="basic" or
            self.num_neg!=1 or not math.isfinite(self.lambda_contribution) or self.lambda_contribution<0):
            raise ValueError("CICL requires frozen ASPCF K4/D64/rank512/one original BPR negative")
        path=Path(args.cicl_proto_path).resolve()
        if not path.is_relative_to(Path(__file__).resolve().parents[2]):
            raise ValueError("Prototype must resolve inside project")
        with path.open("rb") as f:asset=pickle.load(f)
        q=np.asarray(asset["soft_assignments"],dtype=np.float32)
        if (q.shape!=(self.item_num,32) or not np.isfinite(q).all() or np.any(q<0)
            or np.any(q[0]!=0) or not np.allclose(q[1:].sum(1),1,atol=1e-5)
            or asset.get("semantic_rank")!=512):
            raise ValueError("Invalid item-aligned PCA semantic prototype")
        self.register_buffer("cicl_q",torch.from_numpy(q),persistent=False)
        with torch.random.fork_rng(devices=[]):
            self.contribution_predictor=ContributionPredictor()
            self.coalition_teacher=CoalitionTeacher()
        self._cicl_stats=None;self._cicl_epoch=0
        logging.info("[CICL] extra_params=%d total_params=%d lambda=%s teacher_temp=.1 correction_bound=1",
                     sum(p.numel() for p in self.contribution_predictor.parameters()),self.count_variables(),self.lambda_contribution)

    def forward(self,feed_dict,return_intermediate=False):
        out=super().forward(feed_dict,True)
        base=out["interest_weights"]
        correction=self.contribution_predictor(out["interest_vectors"],out["attention_maps"],
                    out["history_vectors"],feed_dict["lengths"],self.cicl_q[feed_dict["history_items"]])
        weights=corrected_weights(base,correction)
        user=(weights[:,:,None]*out["interest_vectors"]).sum(1)
        out["prediction"]=(user[:,None,:]*out["candidate_vectors"]).sum(-1)
        if self.training and self.lambda_contribution>0:
            with torch.no_grad():
                scores=torch.einsum("bkd,bcd->bkc",out["interest_vectors"].detach(),out["candidate_vectors"].detach())
                phi,target,utility=self.coalition_teacher(scores)
            # Aux path has no gradient into original aggregator/encoder/extractor.
            auxiliary=corrected_weights(base.detach(),correction)
            out["_cicl_aux_weights"]=auxiliary
            out["_cicl_teacher"]=target
            out["_cicl_phi"]=phi
            out["_cicl_correction"]=correction.detach()
        if return_intermediate:
            out.update(base_interest_weights=base,interest_weights=weights,user_vector=user,contribution_correction=correction)
        else:
            out={k:v for k,v in out.items() if k=="prediction" or k.startswith("_")}
        return out

    def loss(self,out_dict):
        total=super().loss(out_dict)
        if self.training and "_cicl_teacher" in out_dict:
            target=out_dict["_cicl_teacher"]
            aux=F.kl_div(out_dict["_cicl_aux_weights"].clamp_min(1e-12).log(),target,reduction="batchmean")
            total=total+self.lambda_contribution*aux
            out_dict["loss_contribution"]=aux.detach()
            B=target.shape[0]
            values=torch.stack([aux.detach(),
                -(target*target.clamp_min(1e-12).log()).sum(-1).mean(),
                out_dict["_cicl_phi"].std(-1,unbiased=False).mean(),
                out_dict["_cicl_phi"].abs().mean(),
                out_dict["_cicl_correction"].abs().mean(),
                -(out_dict["_cicl_aux_weights"].detach()*out_dict["_cicl_aux_weights"].detach().clamp_min(1e-12).log()).sum(-1).mean()])
            if self._cicl_stats is None:self._cicl_stats=torch.zeros(7,device=values.device)
            self._cicl_stats[:6]+=values*B;self._cicl_stats[6]+=B
        return total

    def _flush_cicl(self):
        if self._cicl_stats is not None:
            stats=self._cicl_stats.detach().cpu().tolist()
            if stats[6]>0:
                logging.info("[CICL epoch=%s] n=%d aux_kl=%.6f teacher_entropy=%.6f phi_std=%.6f phi_abs=%.6f correction_abs=%.6f serving_entropy=%.6f",
                             self._cicl_epoch,int(stats[6]),*[x/stats[6] for x in stats[:6]])
            self._cicl_stats=None

    def actions_before_epoch(self):
        self._flush_cicl()
        self._cicl_epoch=int(self.cur_epoch)

    def actions_after_train(self):
        self._flush_cicl()
        return super().actions_after_train()
