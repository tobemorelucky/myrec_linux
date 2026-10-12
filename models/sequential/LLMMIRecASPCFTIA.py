"""TIA: causal adjacent-history adapter; same H20 and ASPCF Softmax."""
import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.BaseModel import GeneralModel
from models.sequential.llmmi_components import QueryMultiInterestExtractor
from models.sequential.LLMMIRecASPCFSampledSoftmax import LLMMIRecASPCFSampledSoftmax as SS

class TemporalInteractionAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.transition = nn.Linear(3*dim, 2*dim)
        self.output = nn.Linear(dim, dim)
        nn.init.normal_(self.transition.weight, std=.01)
        nn.init.normal_(self.transition.bias, std=.01)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
    def forward(self, x, lengths):
        valid = torch.arange(x.shape[1],device=x.device)[None,:] < lengths[:,None]
        current = x.masked_fill(~valid[:,:,None], 0)
        previous = torch.cat([torch.zeros_like(current[:,:1]),current[:,:-1]],dim=1)
        previous_valid = torch.cat([torch.zeros_like(valid[:,:1]),valid[:,:-1]],dim=1)
        edge = valid & previous_valid
        features = torch.cat([current, previous, current-previous],dim=-1)
        update = self.output(F.glu(self.transition(features),dim=-1))
        update = update.masked_fill(~edge[:,:,None],0)
        return (current + update).masked_fill(~valid[:,:,None],0)

class TemporalQueryExtractor(QueryMultiInterestExtractor):
    def __init__(self, original, adapter):
        nn.Module.__init__(self)
        self.K=original.K;self.emb_size=original.emb_size;self.attn_size=original.attn_size
        self.query=original.query
        self.Wq=original.Wq;self.Wk=original.Wk;self.Wv=original.Wv
        self.tia=adapter
    def forward(self, history_emb, lengths, **kwargs):
        return super().forward(self.tia(history_emb,lengths),lengths,**kwargs)

class LLMMIRecASPCFTIA(SS):
    def __init__(self,args,corpus):
        if int(args.history_max)!=20:
            raise ValueError("TIA experiment requires exactly history_max=20")
        super().__init__(args,corpus)
        # Additional module initialization cannot change baseline init or dropout RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261012)
            adapter=TemporalInteractionAdapter(self.emb_size)
        self.extractor=TemporalQueryExtractor(self.extractor,adapter)
        self._tia_grad_stats=None
        logging.info("[TIA] H=20 causal adjacent GLU, zero output, added parameters=%d; no new loss",
                     sum(p.numel() for p in adapter.parameters()))
    def loss(self,out):
        loss=super().loss(out)
        if self.training and out.get("_ss_scores") is not None and not self._tia_grad_stats:
            # One optimizer backward hook/epoch: read only existing gradient, no extra backward.
            self._tia_grad_stats={}
            for name,parameter in [("output",self.extractor.tia.output.weight),
                                   ("transition",self.extractor.tia.transition.weight)]:
                def capture(grad,key=name):
                    if key not in self._tia_grad_stats:
                        self._tia_grad_stats[key]=grad.detach().norm()
                    return grad
                handle=parameter.register_hook(capture)
                self._tia_grad_stats[name+"_handle"]=handle
        return loss
    def _flush_stats(self):
        super()._flush_stats()
        if self._tia_grad_stats:
            values=[]
            for key in ["output","transition"]:
                self._tia_grad_stats[key+"_handle"].remove()
                values.append(float(self._tia_grad_stats[key].cpu()) if key in self._tia_grad_stats else float("nan"))
            logging.info("[TIAStats] epoch=%s first_batch_output_grad=%.9g first_batch_transition_grad=%.9g output_weight_norm=%.9g",
                         self._ss_epoch,values[0],values[1],float(self.extractor.tia.output.weight.detach().norm().cpu()))
            self._tia_grad_stats=None
    class Dataset(SS.Dataset):
        def _get_feed_dict(self,index):
            feed=GeneralModel.Dataset._get_feed_dict(self,index)
            pos=int(self.data["position"][index])
            # Do not materialize an older prefix: identical final IDs to baseline H20.
            seq=self.corpus.user_his[feed["user_id"]][max(0,pos-20):pos]
            feed["history_items"]=np.array([x[0] for x in seq])
            feed["history_times"]=np.array([x[1] for x in seq])
            feed["lengths"]=len(seq)
            return feed
