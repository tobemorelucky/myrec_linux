"""Thin SCFT entrypoint. Default dry-run. Reuses pinned main_llm/Trainer/PEFT."""
import argparse, ast, hashlib, json, os, sys, types
from pathlib import Path
from common import load_config, ROOT, new_path, sha256, seed_all, json_new, load_mapping, load_prompts, environment_manifest, code_manifest, encoder_artifacts

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True)
    p.add_argument("--execute",action="store_true")
    a=p.parse_args(); c=load_config(a.config)
    if c["stage"]!="scft": raise ValueError("Wrong stage")
    print(json.dumps(c,indent=2))
    if not a.execute: return
    from preflight import check
    ready=check(c,require_gpu=True,framework_imports=True)
    if not ready["ready"]:raise RuntimeError(json.dumps(ready))
    import torch
    from model_patch import LlamaRSEmbV2
    if not torch.cuda.is_available(): raise RuntimeError("Approved GPU allocation required")
    if c["precision"]=="bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unavailable: do not silently switch precision")
    seed_all(c["seed"])
    rows=load_prompts(c["prompt"],load_mapping(c["mapping"]))
    if len({r["input"] for r in rows}) != len(rows):raise ValueError("Duplicate item prompts: SCFT false-negative risk")
    run=new_path(c["output_dir"]); run.mkdir(parents=True)
    json_new(run/"invocation.json",{"config":c,"config_sha256":sha256(a.config),
        "inputs":encoder_artifacts(c,full_weights=True),"prompt_sha256":sha256(c["prompt"]),
        "mapping_sha256":sha256(c["mapping"]),"environment":environment_manifest(),"code":code_manifest(),
        "augmentation":"title_preserving_attribute_dropout_once_before_training",
        "contrastive_items_per_microbatch_max":c["batch_size"],
        "tail_microbatch_items":len(rows)%c["batch_size"] or c["batch_size"],
        "dataloader_drop_last":False,
        "optimizer_effective_batch_max":c["batch_size"]*c["gradient_accumulation_steps"]})
    up=ROOT/"upstream"; sys.path.insert(0,str(up))
    import llm.data_processor.llama as processor
    from common import dropout_feature
    processor.dropout_feature=dropout_feature
    # No fork of training framework: import the official main after removing only
    # its hard-coded CUDA_VISIBLE_DEVICES assignment, with an AST-scoped change.
    src=(up/"main_llm.py").read_text()
    tree=ast.parse(src); removed=0; body=[]
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) and
            isinstance(t.value,ast.Attribute) and t.value.attr=="environ" and
            isinstance(t.slice,ast.Constant) and t.slice.value=="CUDA_VISIBLE_DEVICES"
            for t in n.targets):
            removed+=1
        else: body.append(n)
    if removed!=1: raise RuntimeError("Pinned main changed")
    tree.body=body
    mod=types.ModuleType("v2_official_main"); mod.__file__=str(up/"main_llm.py")
    exec(compile(tree,mod.__file__,"exec"),mod.__dict__)
    mod.LlamaRSEmb=LlamaRSEmbV2
    from training_support import V2Trainer,save_adapter_complete
    mod.MedRecTrainer=V2Trainer
    from transformers import TrainerCallback
    class SaveV2(TrainerCallback):
        def on_train_end(self,args,state,control,**kwargs):
            if state.is_world_process_zero:
                save_adapter_complete(kwargs["model"],Path(args.output_dir)/"lora_final",
                                      kwargs.get("tokenizer"),args)
            return control
    mod.SavePeftModelCallback=SaveV2
    args=["--do_train","--model_name_or_path",c["model_path"],
          "--train_file",c["prompt"],"--cache_dir",str(run/"cache"),
          "--output_dir",str(run),"--max_source_length",str(c["max_source_length"]),
          "--max_target_length","196","--per_device_train_batch_size",str(c["batch_size"]),
          "--gradient_accumulation_steps",str(c["gradient_accumulation_steps"]),
          "--max_steps",str(c["max_steps"]),"--learning_rate",str(c["learning_rate"]),
          "--lora_rank","8","--lora_alpha","32","--lora_dropout","0.1",
          "--trainable",c["target_modules"],"--modules_to_save","null",
          "--pool_type","avg","--dropout_ratio","0.4","--seed",str(c["seed"]),
          "--save_steps",str(c["max_steps"]),"--logging_steps","100",
          "--lr_scheduler_type","linear","--warmup_steps","0","--max_grad_norm","1.0",
          "--logging_nan_inf_filter","false","--save_safetensors","false",
          "--preprocessing_num_workers","1","--report_to","none",
          "--dataloader_drop_last","false","--weight_decay","0",
          "--adam_beta1","0.9","--adam_beta2","0.999","--adam_epsilon","1e-8"]
    if c["precision"] in ("bf16","fp16"): args+=["--"+c["precision"]]
    if c["gradient_checkpointing"]: args+=["--gradient_checkpointing","true"]
    sys.argv=["v2_scft"]+args
    mod.train()

if __name__=="__main__": main()
