"""Thin fixes for official Trainer saving; no training framework reimplementation."""
import json,random,sys
from pathlib import Path
import numpy as np
import torch
from common import ROOT,cap_text_fields
sys.path.insert(0,str(ROOT/"upstream"))
from llm.trainer_seq2seq import MedRecTrainer

def save_adapter_complete(model,path,tokenizer=None,args=None):
    p=Path(path).resolve()
    if ROOT not in p.parents:raise ValueError("Checkpoint outside isolated root")
    from llm.lora_cls import PeftModelForCLS
    if not isinstance(model,PeftModelForCLS):raise TypeError("Refuse non-PEFT full-base save")
    unexpected=[n for n,v in model.named_parameters() if v.requires_grad and
                "lora_" not in n and not (n=="tau" or n.endswith(".tau"))]
    if unexpected:raise ValueError("Unexpected non-LoRA trainable parameters: "+str(unexpected))
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
    def _checkpoint_path(self,checkpoint):
        p=Path(checkpoint).resolve()
        if ROOT not in p.parents or p.is_symlink():
            raise ValueError("Resume checkpoint outside isolated project")
        required=("adapter_model.bin","v2_extra_trainable.pt","optimizer.pt",
                  "scheduler.pt","trainer_state.json","rng_state.pth")
        from common import sha256
        record=json.loads((p/"V2_RESUME_MANIFEST.json").read_text())
        for name in required:
            if not (p/name).is_file() or sha256(p/name)!=record["files"][name]:
                raise ValueError("Checkpoint state missing/changed: "+name)
        return p
    def _save_checkpoint(self,model,trial,metrics=None):
        result=super()._save_checkpoint(model,trial,metrics)
        from common import sha256
        # Single-process production only; original HF saves optimizer/scheduler,
        # TrainerState and RNG. This manifest never contains the 7B base.
        if self.args.world_size!=1:raise ValueError("Only single-process V2 is validated")
        p=Path(self._get_output_dir(trial=trial))/("checkpoint-"+str(self.state.global_step))
        files={f.name:sha256(f) for f in p.iterdir() if f.is_file()}
        (p/"V2_RESUME_MANIFEST.json").write_text(json.dumps(
            {"global_step":self.state.global_step,"files":files,"adapter_only":True},indent=2))
        return result
    def _load_from_checkpoint(self,resume_from_checkpoint,model=None):
        p=self._checkpoint_path(resume_from_checkpoint)
        target=self.model if model is None else model
        from common import verify_adapter_loaded,restore_extras
        state=torch.load(p/"adapter_model.bin",map_location="cpu",weights_only=True)
        parameters=dict(target.named_parameters())
        expected={n:n.replace(".default.",".") for n in parameters if "lora_" in n}
        if set(expected.values())!=set(state):
            raise ValueError("Adapter architecture/parameter keys changed")
        with torch.no_grad():
            for name,key in expected.items():
                if parameters[name].shape!=state[key].shape:raise ValueError("Adapter shape changed")
                parameters[name].copy_(state[key].to(parameters[name]))
            restore_extras(target,p)
        verify_adapter_loaded(target,p)
    def _load_rng_state(self,checkpoint):
        if checkpoint is None:return
        p=self._checkpoint_path(checkpoint)
        if self.args.world_size!=1:raise ValueError("Only single-process RNG resume is validated")
        # Only our SHA-verified checkpoint. NumPy/Python RNG requires pickle;
        # never change torch.load defaults or load untrusted external files.
        state=torch.load(p/"rng_state.pth",map_location="cpu",weights_only=False)
        random.setstate(state["python"]);np.random.set_state(state["numpy"])
        torch.random.set_rng_state(state["cpu"])
        if torch.cuda.is_available():
            if "cuda" not in state:raise ValueError("GPU checkpoint has no CUDA RNG")
            torch.cuda.random.set_rng_state(state["cuda"])
    admission_indices=None
    admission_batches=None
    admission_optimizer_updates=0
    def _get_train_sampler(self):
        if self.admission_indices is None:return super()._get_train_sampler()
        return FixedAdmissionSampler(self.admission_indices,len(self.train_dataset))
    def training_step(self,model,inputs):
        if self.admission_indices is None:return super().training_step(model,inputs)
        type(self).last_admission_input={"shape":list(inputs["input_ids"].shape),
             "nonpadding_tokens":int(inputs["attention_mask"].sum().item())}
        result=super().training_step(model,inputs)
        if not torch.isfinite(result).all():raise FloatingPointError("Nonfinite admission loss")
        self.admission_batches.append({"shape":list(inputs["input_ids"].shape),
            "nonpadding_tokens":int(inputs["attention_mask"].sum().item()),
            "loss_divided_by_accumulation":float(result.detach().float().cpu())})
        return result
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
            if self.admission_indices is not None:
                def count_real_step(optimizer,args,kwargs):
                    type(self).admission_optimizer_updates+=1
                self.optimizer.register_step_post_hook(count_real_step)
            self.optimizer._v2_guard=True
        return out


from llm.data_processor.llama import llama_train_mask
class V2TrainMask(llama_train_mask):
    field_caps={}
    fail_on_truncation=False
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.fail_on_truncation:
            import llm.data_processor.llama as processor
            previous=processor.dropout_feature
            def checked_views(text,ratio):
                pair=previous(text,ratio)
                from common import encode_prompt
                for view in pair:
                    encode_prompt(view,self.tokenizer,self.data_args.max_source_length,strict=True)
                return pair
            processor.dropout_feature=checked_views
    def __call__(self,examples):
        copied=dict(examples)
        if self.fail_on_truncation:
            if self.field_caps:raise ValueError("Full-text mode requires empty field caps")
            from common import encode_prompt
            for text in examples[self.prompt_column]:
                encode_prompt(text,self.tokenizer,self.data_args.max_source_length,strict=True)
        copied[self.prompt_column]=[cap_text_fields(s,self.tokenizer,self.field_caps)
                                    for s in examples[self.prompt_column]]
        return super().__call__(copied)

def verify_save_restore(model,path):
    """Admission-only in-place round trip; no second 7B model or optimizer step."""
    from common import verify_adapter_loaded,restore_extras
    verify_adapter_loaded(model,path)
    saved=torch.load(Path(path)/"adapter_model.bin",map_location="cpu",weights_only=True)
    params=dict(model.named_parameters())
    first=next(p for n,p in params.items() if "lora_" in n)
    with torch.no_grad():
        first.add_(1)
        for n,p in params.items():
            if "lora_" in n:p.copy_(saved[n.replace(".default.",".")].to(p.device,p.dtype))
        restore_extras(model,path)
    verify_adapter_loaded(model,path)
    return {"adapter_round_trip":True,"full_base_saved":False}


def scft_text_only(datasets):
    """Keep provenance in the immutable JSONL, not in Torch training batches."""
    return type(datasets)({name: data.select_columns(["input", "target"])
                           for name, data in datasets.items()})

class FixedAdmissionSampler(torch.utils.data.Sampler):
    """C only: exactly 24 microbatches, fixed longest/mixed schedule."""
    def __init__(self,indices,dataset_length):
        self.indices=list(indices)
        if not self.indices or min(self.indices)<0 or max(self.indices)>=dataset_length:
            raise ValueError("Invalid admission index")
    def __iter__(self):return iter(self.indices)
    def __len__(self):return len(self.indices)

def validate_admission_plan(c):
    from common import sha256,load_prompts,load_mapping
    plan=json.load(open(c["admission_plan"]))
    if sha256(c["admission_plan"])!=c["expected_admission_plan_sha256"]:
        raise ValueError("Admission plan changed")
    if plan["prompt_sha256"]!=sha256(c["prompt"]):raise ValueError("Plan prompt changed")
    rows=load_prompts(c["prompt"],load_mapping(c["mapping"]))
    if len(rows)!=plan["dataset_length"]:raise ValueError("Plan catalog changed")
    if [rows[i]["item_id"] for i in plan["indices"]]!=plan["item_ids"]:
        raise ValueError("Plan item order changed")
    if len(plan["indices"])!=3*c["batch_size"]*c["gradient_accumulation_steps"]:
        raise ValueError("C must use exactly three real accumulated updates")
    return plan

def local_scft_dataset(loader,*args,**kwargs):
    """HF datasets v4 removed the legacy None authentication keyword."""
    legacy=kwargs.pop("use_auth_token",None)
    if legacy not in (None,False):
        raise ValueError("Only local offline item metadata is allowed")
    return scft_text_only(loader(*args,**kwargs))
