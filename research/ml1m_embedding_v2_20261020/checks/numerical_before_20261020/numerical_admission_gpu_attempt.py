"""Opt-in BF16 and bounded first-nonfinite tracing. No loss/model rewrite."""
import json,random,time
from pathlib import Path
import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from model_patch import LlamaRSEmbV2

class LlamaRSEmbBF16(LlamaRSEmbV2):
    @classmethod
    def from_pretrained(cls,*args,**kwargs):
        kwargs["torch_dtype"]=torch.bfloat16
        return super().from_pretrained(*args,**kwargs)
    def half(self):
        # Intercept official .half().cuda() ONLY in independent BF16 class.
        self.to(dtype=torch.bfloat16)
        self.tau.data=self.tau.data.float()
        if any(p.dtype==torch.float16 for p in self.parameters()):
            raise TypeError("FP16 parameter survived native BF16 selection")
        return self

def tensors(obj):
    if isinstance(obj,torch.Tensor):yield obj
    elif isinstance(obj,dict):
        for value in obj.values():yield from tensors(value)
    elif isinstance(obj,(tuple,list)):
        for value in obj:yield from tensors(value)

def stats(t):
    t=t.detach()
    out={"shape":list(t.shape),"dtype":str(t.dtype),"device":str(t.device)}
    if not t.numel():return out
    if t.is_floating_point():
        good=torch.isfinite(t)
        out.update(nan=int(torch.isnan(t).sum().item()),
                   posinf=int(torch.isposinf(t).sum().item()),
                   neginf=int(torch.isneginf(t).sum().item()))
        if good.any():
            finite=t[good].float()
            out.update(finite_min=float(finite.min().item()),finite_max=float(finite.max().item()))
    else:out.update(min=int(t.min().item()),max=int(t.max().item()))
    return out

def peaks():
    if not torch.cuda.is_available():return {}
    return {"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes":torch.cuda.max_memory_reserved()}

class FirstNonfinite(TorchDispatchMode):
    """Probe-only matmuls/softmax + module boundaries. Never sanitize.
    Deliberate -Inf additive masks are not activation bugs. Matmul/module
    outputs must be entirely finite. Detection is limited to observed points.
    """
    def __init__(self,model,path):
        super().__init__();self.model=model;self.path=Path(path)
        self.stack=[];self.handles=[];self.checked_ops=0;self.checked_modules=0
    def fail(self,where,inputs,outputs):
        record={"first_observed_failure":where,
                "module_stack":[name for name,_ in self.stack],
                "operator_inputs":[stats(x) for x in tensors(inputs)],
                "outputs":[stats(x) for x in tensors(outputs)],
                "module_inputs":[{"module":name,"tensors":[stats(x) for x in tensors(values)]}
                                 for name,values in self.stack[-2:]],
                "checked_ops":self.checked_ops,"checked_modules":self.checked_modules,**peaks()}
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open("x") as f:json.dump(record,f,indent=2,allow_nan=False)
        raise FloatingPointError("First nonfinite activation at "+where+"; "+str(self.path))
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        result=func(*args,**(kwargs or {}))
        name=str(func)
        if any(word in name for word in ("aten.mm.","aten.bmm.","aten.addmm.","aten._softmax.","aten._safe_softmax.")):
            self.checked_ops+=1
            if any(x.is_floating_point() and not torch.isfinite(x).all() for x in tensors(result)):
                self.fail(name,args,result)
        return result
    def __enter__(self):
        def before(name):
            def hook(module,args,kwargs):self.stack.append((name,(args,kwargs)))
            return hook
        def after(name):
            def hook(module,args,kwargs,out):
                self.checked_modules+=1
                if any(x.is_floating_point() and not torch.isfinite(x).all() for x in tensors(out)):
                    self.fail(name,(args,kwargs),out)
                if self.stack and self.stack[-1][0]==name:self.stack.pop()
            return hook
        for name,module in self.model.named_modules():
            if "Llama" in type(module).__name__ or isinstance(module,(torch.nn.Linear,torch.nn.Embedding)):
                self.handles.append(module.register_forward_pre_hook(before(name),with_kwargs=True))
                self.handles.append(module.register_forward_hook(after(name),with_kwargs=True))
        return super().__enter__()
    def __exit__(self,*exc):
        for handle in self.handles:handle.remove()
        self.stack.clear()
        return super().__exit__(*exc)

def precision_report(model,fix_trainable=False):
    counts={};bad=[]
    for name,p in model.named_parameters():
        if not p.is_floating_point():continue
        if p.requires_grad:
            if "lora_" not in name and not name.endswith("tau"):raise TypeError("Unexpected trainable base "+name)
            if fix_trainable:p.data=p.data.float()
            expected=torch.float32
        else:expected=torch.bfloat16
        counts[str(p.dtype)]=counts.get(str(p.dtype),0)+p.numel()
        if p.dtype!=expected:bad.append((name,str(p.dtype),str(expected)))
    if bad:raise TypeError("BF16 base / FP32 LoRA,tau mismatch: "+str(bad[:8]))
    return counts

def numerical_trainer(base,config,run):
    """Reuse official Trainer/loss/schedule; opt-in short-test diagnostics only."""
    class NumericalTrainer(base):
        numerical_report=None
        probes_done=False
        def create_optimizer(self):
            self.parameter_dtypes=precision_report(self.model,fix_trainable=True)
            result=super().create_optimizer()
            def record_grad(optimizer,args,kwargs):
                gs=[p.grad.detach().float().norm() for g in optimizer.param_groups
                    for p in g["params"] if p.grad is not None]
                norm=float(torch.stack(gs).norm().item()) if gs else 0.
                if not np.isfinite(norm) or norm==0:raise FloatingPointError("Invalid/zero trainable gradient")
                type(self).numerical_report["optimizer_gradient_norms"].append(norm)
            self.optimizer.register_step_pre_hook(record_grad)
            return result
        def probe_forwards(self,model):
            # Same already augmented dataset; restore RNG after diagnostic passes.
            rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
            report={"precision":"BF16 base/autocast; FP32 LoRA master weights,tau,pooling,SCFT",
                    "parameter_elements_by_dtype":self.parameter_dtypes,
                    "probe_batches":[],"optimizer_gradient_norms":[]}
            type(self).numerical_report=report
            try:
                for number in (0,1):
                    indices=self.admission_indices[number*16:(number+1)*16]
                    batch=self._prepare_inputs(self.data_collator([self.train_dataset[i] for i in indices]))
                    mask=batch["attention_mask"]
                    if mask.ndim!=2 or not (mask.sum(1)>0).all():raise ValueError("Empty/misaligned real-token mask")
                    if not torch.all((mask==0)|(mask==1)):raise ValueError("Non-binary token mask")
                    if (mask[:,1:]>mask[:,:-1]).any():raise ValueError("Expected right dynamic padding")
                    before=time.monotonic();torch.cuda.synchronize()
                    with FirstNonfinite(model,Path(run)/("FIRST_NONFINITE_probe"+str(number)+".json")) as trace:
                        with self.compute_loss_context_manager():
                            # Train-mode LoRA dropout/checkpoint path, no backward.
                            out=model(**batch)
                        if not torch.isfinite(out.loss).all():raise FloatingPointError("Nonfinite probe loss")
                        loss=float(out.loss.detach().float().item());del out
                    torch.cuda.synchronize()
                    report["probe_batches"].append({"number":number,"shape":list(batch["input_ids"].shape),
                        "real_tokens":int(mask.sum().item()),"min_length":int(mask.sum(1).min().item()),
                        "max_length":int(mask.sum(1).max().item()),"loss":loss,
                        "seconds_with_diagnostics":time.monotonic()-before,
                        "checked_ops":trace.checked_ops,"checked_modules":trace.checked_modules,**peaks()})
                    del batch
                from common import json_new
                json_new(Path(run)/"FORWARD_PROBES_COMPLETE.json",report)
            finally:
                random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
        def training_step(self,model,inputs):
            if not self.probes_done:
                self.probe_forwards(model);self.probes_done=True
            return super().training_step(model,inputs)
    return NumericalTrainer
