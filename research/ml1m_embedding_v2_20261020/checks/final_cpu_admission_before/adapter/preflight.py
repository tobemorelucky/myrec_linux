"""Read-only readiness checks; never instantiate or execute a pretrained LLM."""
import argparse,json,shutil,sys
from pathlib import Path
from common import ROOT,load_config,load_mapping,load_prompts,sha256,new_path,environment_manifest

def check(c,require_gpu=False,framework_imports=False):
    blockers=[];facts={"stage":c["stage"],"environment":environment_manifest()}
    def record(name,fn):
        try:facts[name]=fn()
        except Exception as e:blockers.append(name+": "+type(e).__name__+": "+str(e))
    record("output_new",lambda:str(new_path(c["output_dir"])))
    record("free_disk_gib",lambda:shutil.disk_usage(ROOT).free/1024**3)
    if facts.get("free_disk_gib",0)<c.get("min_disk_free_gib",8):blockers.append("Insufficient isolated-workspace disk reserve")
    def data():
        mapping=load_mapping(c["mapping"]);rows=load_prompts(c["prompt"],mapping)
        if len(mapping)!=3706:raise ValueError("Expected exactly 3706 ML-1M items")
        for key in ("mapping","prompt"):
            expected=c.get("expected_"+key+"_sha256")
            if not expected:raise ValueError(key+" identity not frozen")
            if sha256(c[key])!=expected:raise ValueError(key+" SHA differs from registered input")
        if len({r["input"] for r in rows})!=len(rows):raise ValueError("Duplicate prompts: SCFT collision")
        if not Path(c["source_manifest"]).is_file():raise FileNotFoundError("Original metadata provenance")
        if c.get("variant")=="V2-Rich" and not any(r.get("sources") for r in rows):
            raise ValueError("Rich input contains no verified added content")
        return {"items":len(mapping),"prompts":len(rows),"prompt_sha256":sha256(c["prompt"]),
                "mapping_sha256":sha256(c["mapping"])}
    record("data",data)
    def weights():
        model=Path(c["model_path"]);idx=json.load((model/"model.safetensors.index.json").open())
        names=sorted(set(idx["weight_map"].values()))
        from safetensors import safe_open
        keys=set()
        for n in names:
            if Path(n).name!=n:raise ValueError("Unsafe shard index")
            p=model/n
            with safe_open(p,framework="np") as h:keys.update(h.keys())
        if keys!=set(idx["weight_map"]):raise ValueError("Base shard keys/index mismatch")
        lock=json.load(open(c["weight_lock"]))["files"]
        if set(lock)!=set(names):raise ValueError("Weight-lock/index shard names differ")
        if any((model/n).stat().st_size!=lock[n]["bytes"] for n in names):raise ValueError("Weight-lock shard size mismatch")
        return {"selected":"safetensors","shards":[{"name":n,"bytes":(model/n).stat().st_size} for n in names],
                "tensor_keys":len(keys),"full_weight_lock_present":Path(c["weight_lock"]).is_file()}
    record("model_header",weights)
    def tokenizer():
        from transformers import AutoTokenizer
        t=AutoTokenizer.from_pretrained(c.get("tokenizer_path",c["model_path"]),use_fast=True,local_files_only=True)
        t.pad_token=t.unk_token
        if t.eos_token_id is None or t.pad_token_id is None:raise ValueError("Missing EOS/pad")
        if "data" not in facts:return {"loaded":True}
        rows=load_prompts(c["prompt"],load_mapping(c["mapping"]))
        lengths=[len(t.encode(r["input"],add_special_tokens=False))+1 for r in rows]
        return {"loaded":True,"min_tokens":min(lengths),"max_tokens":max(lengths),
                "over_max_before_field_caps":sum(n>c["max_source_length"] for n in lengths)}
    record("tokenizer",tokenizer)
    if c.get("adapter_path"):
        def adapter():
            ad=Path(c["adapter_path"])
            ac=json.load((ad/"adapter_config.json").open())
            path=ad/"adapter_model.bin"
            if not path.is_file():raise FileNotFoundError(path)
            if c.get("expected_adapter_sha256") and sha256(path)!=c["expected_adapter_sha256"]:
                raise ValueError("LoRA SHA differs")
            if not c.get("allow_legacy_adapter_without_tau") and not (ad/"v2_extra_trainable.pt").is_file():
                raise FileNotFoundError("Fresh V2 adapter/tau not yet trained")
            return {"rank":ac["r"],"alpha":ac["lora_alpha"],"bytes":path.stat().st_size,
                    "sha256":sha256(path),"tau_required_for_inference":False}
        record("adapter",adapter)
    if framework_imports:
        def framework():
            sys.path.insert(0,str(ROOT/"upstream"))
            from training_support import V2Trainer
            from llm.lora_cls import PeftModelForCLS
            from transformers import Seq2SeqTrainingArguments
            return {"trainer_import":True,"adapter_only_save_override":V2Trainer._save.__module__,
                    "required_args_available":all(k in Seq2SeqTrainingArguments.__dataclass_fields__
                        for k in ("save_safetensors","logging_nan_inf_filter","gradient_checkpointing"))}
        record("framework",framework)
    import torch
    facts["gpu"]={"available":torch.cuda.is_available(),"count":torch.cuda.device_count(),
                  "torch_cuda":torch.version.cuda}
    if require_gpu:
        if not torch.cuda.is_available():blockers.append("No allocated CUDA GPU")
        else:
            free,total=torch.cuda.mem_get_info()
            facts["gpu"].update(name=torch.cuda.get_device_name(),free_gib=free/1024**3,total_gib=total/1024**3)
            if free/1024**3<c.get("min_gpu_free_gib",14):blockers.append("GPU free memory below registered requirement")
            if c.get("precision")=="bf16" and not torch.cuda.is_bf16_supported():blockers.append("BF16 not supported")
    return {"ready":not blockers,"gpu_required":require_gpu,"blockers":blockers,"facts":facts}

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True)
    p.add_argument("--gpu",action="store_true");p.add_argument("--framework",action="store_true")
    p.add_argument("--strict",action="store_true");a=p.parse_args()
    report=check(load_config(a.config),a.gpu,a.framework)
    print(json.dumps(report,indent=2,ensure_ascii=False))
    if a.strict and not report["ready"]:raise SystemExit(2)
if __name__=="__main__":main()
