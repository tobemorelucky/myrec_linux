"""Thin fixes for official Trainer saving; no training framework reimplementation."""
import json,random,sys
from pathlib import Path
import numpy as np
import torch
from common import ROOT
sys.path.insert(0,str(ROOT/"upstream"))
from llm.trainer_seq2seq import MedRecTrainer

def save_adapter_complete(model,path,tokenizer=None,args=None):
    p=Path(path).resolve()
    if ROOT not in p.parents:raise ValueError("Checkpoint outside isolated root")
    p.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(str(p))
    extra={n:v.detach().cpu() for n,v in model.named_parameters() if v.requires_grad and "lora_" not in n}
    if not any("tau" in n for n in extra):raise ValueError("Trainable tau missing")
    if not all(torch.isfinite(v).all() for v in extra.values()):raise FloatingPointError("Nonfinite extra state")
    torch.save(extra,p/"v2_extra_trainable.pt")
    if tokenizer is not None:tokenizer.save_pretrained(str(p))
    if args is not None:
        torch.save(args,p/"training_args.bin")
        (p/"training_args.json").write_text(args.to_json_string())
    rng={"python":random.getstate(),"numpy":np.random.get_state(),"torch_cpu":torch.get_rng_state()}
    if torch.cuda.is_available():rng["torch_cuda"]=torch.cuda.get_rng_state_all()
    torch.save(rng,p/"v2_rng_state.pth")

class V2Trainer(MedRecTrainer):
    def _save(self,output_dir=None,state_dict=None):
        # Generic Trainer can mistake vendored PEFT for a plain nn.Module.
        save_adapter_complete(self.model,output_dir or self.args.output_dir,self.tokenizer,self.args)
    def create_optimizer(self):
        out=super().create_optimizer()
        def finite_before_step(optimizer,args,kwargs):
            grads=[p.grad for g in optimizer.param_groups for p in g["params"] if p.grad is not None]
            if grads:
                norms=torch._foreach_norm(grads)
                if not torch.isfinite(torch.stack([n.float() for n in norms])).all():
                    raise FloatingPointError("Nonfinite optimizer task gradients")
        if not getattr(self.optimizer,"_v2_guard",False):
            self.optimizer.register_step_pre_hook(finite_before_step)
            self.optimizer._v2_guard=True
        return out
