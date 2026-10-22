"""One 12-item V1 BF16 probe; reuse existing run_encode, never full B/D/E."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"adapter"),str(ROOT/"upstream")]
IDS=[1,2,49,1248,1249,1250,1251,1252,1258,2655,3646,3706]
def main():
    p=argparse.ArgumentParser();p.add_argument("--execute",action="store_true")
    a=p.parse_args()
    from common import load_config,new_path,json_new,require_rental_execution
    config=ROOT/"ops/v1_recovery/bf16_probe.json";c=load_config(config)
    print(json.dumps({"config":c,"sample_ids":IDS,"expected_groups":[IDS[:4],IDS[4:8],IDS[8:]],"execute":a.execute},indent=2))
    if not a.execute:return
    if os.environ.get("V1_GPU_PROBE_APPROVED")!="YES":raise RuntimeError("Separate GPU probe approval required")
    require_rental_execution()
    report_path=new_path(c["probe_runtime_report"])
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Approved native BF16 rental GPU required")
    from llm.lora_cls import PeftModelForCLS
    original=PeftModelForCLS.from_pretrained
    descriptor=PeftModelForCLS.__dict__.get("from_pretrained")
    report={"sample_ids":IDS,"batches":[],"optimizer_updates":0,"status":"started"}
    def capture(cls,*args,**kwargs):
        model=original(*args,**kwargs)
        counts={};base=set()
        for name,value in model.named_parameters():
            kind="LoRA" if "lora_" in name else ("tau" if name.endswith("tau") else "base")
            key=kind+":"+str(value.dtype);counts[key]=counts.get(key,0)+value.numel()
            if kind=="base" and value.is_floating_point():base.add(value.dtype)
            if value.dtype==torch.float16:raise TypeError("FP16 parameter survived BF16 V1 loading")
        if base!={torch.bfloat16}:raise TypeError("Base must actually be BF16")
        report["actual_parameter_elements_by_dtype"]=counts
        def before(module,args,kw):
            mask=kw["attention_mask"]
            assert not module.training and not torch.is_grad_enabled()
            assert torch.all((mask==0)|(mask==1)) and torch.all(mask.sum(1)>0)
            assert not (mask[:,1:]>mask[:,:-1]).any()
            report["batches"].append({"shape":list(kw["input_ids"].shape),
                "mask_lengths":mask.sum(1).cpu().tolist(),"mask_dtype":str(mask.dtype),
                "autocast_enabled":torch.is_autocast_enabled(),"eval_mode":not module.training,
                "grad_enabled":torch.is_grad_enabled()})
        model.register_forward_pre_hook(before,with_kwargs=True)
        return model
    PeftModelForCLS.from_pretrained=classmethod(capture)
    try:
        import run_encode
        sys.argv=["run_encode","--config",str(config),"--sample-ids",",".join(map(str,IDS)),"--execute"]
        run_encode.main()
        complete=json.load((Path(c["output_dir"])/"SAMPLE_COMPLETE.json").open())
        assert complete["canonical_ids"]==IDS and complete["rows"]==12
        assert len(report["batches"])==3
        report.update(status="passed",sample_complete=complete,full_catalog_generated=False)
    except BaseException as exc:
        report.update(status="failed",error_type=type(exc).__name__,error=str(exc));raise
    finally:
        if descriptor is None:delattr(PeftModelForCLS,"from_pretrained")
        else:PeftModelForCLS.from_pretrained=descriptor
        if torch.cuda.is_available():report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
        json_new(report_path,report)
if __name__=="__main__":main()
