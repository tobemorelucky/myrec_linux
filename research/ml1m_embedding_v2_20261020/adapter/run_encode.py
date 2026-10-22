"""Production encoding; default dry-run, reuses the same corrected LLMEmb class."""
import argparse,json
from pathlib import Path
from common import (load_config,ROOT,new_path,load_mapping,load_prompts,seed_all,sha256,json_new,
    snapshot_paths,encoder_artifacts,environment_manifest,code_manifest,encode_prompt,read_vectors,
    restore_extras,verify_adapter_loaded,table_quality)

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--sample-ids",help="Admission only: canonical IDs; never a full-catalog asset")
    a=p.parse_args();c=load_config(a.config)
    if c["stage"]!="encode":raise ValueError("Wrong stage")
    print(json.dumps(c,indent=2))
    if not a.execute:return
    from common import require_rental_execution
    require_rental_execution()
    from preflight import check
    ready=check(c,require_gpu=True,framework_imports=False)
    if not ready["ready"]:raise RuntimeError(json.dumps(ready))
    out=new_path(c["output_dir"])
    mapping=load_mapping(c["mapping"]);rows=load_prompts(c["prompt"],mapping)
    if a.sample_ids:
        selected=[int(i) for i in a.sample_ids.split(",")]
        if len(selected)>16 or len(set(selected))!=len(selected) or not set(selected)<=set(mapping.values()):
            raise ValueError("Admission supports <=16 unique canonical IDs")
        rows=[r for r in rows if r["item_id"] in selected]
    import torch
    import numpy as np
    from transformers import AutoTokenizer
    from model_patch import LlamaRSEmbV2
    from llm.lora_cls import PeftModelForCLS
    seed_all(c["seed"])
    dtype={"fp16":torch.float16,"bf16":torch.bfloat16,"fp32":torch.float32}[c["precision"]]
    token=AutoTokenizer.from_pretrained(c["tokenizer_path"],use_fast=True,local_files_only=True)
    token.pad_token=token.unk_token;token.padding_side="right"
    identity={"artifacts":encoder_artifacts(c,full_weights=True),"pooling":"masked_mean_fp32",
        "special_tokens":"no_BOS_truncate_Lminus1_append_EOS","max_length":c["max_source_length"],
        "precision":c["precision"],"seed":c["seed"],"field_caps":c.get("field_caps",{}),
        "code_sha256":code_manifest()["code_sha256"]}
    key=__import__("hashlib").sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    import time
    model_load_started=time.monotonic()
    model=LlamaRSEmbV2.from_pretrained(c["model_path"],pool_type="avg",tau=3.,
              torch_dtype=dtype,local_files_only=True,use_safetensors=True).cuda()
    model.config.pad_token_id=token.pad_token_id
    adapter=Path(c["adapter_path"])
    model=PeftModelForCLS.from_pretrained(model,str(adapter),is_trainable=False)
    extra_status=restore_extras(model,adapter,c.get("allow_legacy_adapter_without_tau",False))
    restored=verify_adapter_loaded(model,adapter);model.eval()
    torch.cuda.synchronize();model_load_seconds=time.monotonic()-model_load_started
    out.mkdir(parents=True)
    json_new(out/"manifest.json",{"config":c,"encoder_key":key,"encoder_identity":identity,
        "code":code_manifest(),"environment":environment_manifest(),"adapter_tensors_restored":restored,
        "extra_state":extra_status,"inputs":snapshot_paths([c["prompt"],c["mapping"],c["source_manifest"]]),
        "status":"started"})
    import time
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    lengths=[];truncated=0;field_clips={};sample_vectors=[]
    with (out/"raw.jsonl").open("x",encoding="utf-8") as writer,torch.inference_mode():
        for start in range(0,len(rows),c["batch_size"]):
            batch=rows[start:start+c["batch_size"]];seq=[]
            for r in batch:
                ids,stats=encode_prompt(r["input"],token,c["max_source_length"],c.get("field_caps"),strict=c.get("fail_on_truncation",False))
                seq.append(ids);lengths.append(stats["tokens_before_total_truncation"])
                truncated+=int(stats["total_truncated"])
                for k,v in stats["field_tokens_clipped"].items():field_clips[k]=field_clips.get(k,0)+v
            ids=torch.full((len(seq),max(map(len,seq))),token.pad_token_id,dtype=torch.long,device="cuda")
            mask=torch.zeros_like(ids)
            for j,s in enumerate(seq):
                ids[j,:len(s)]=torch.tensor(s,device="cuda");mask[j,:len(s)]=1
            v=model(input_ids=ids,attention_mask=mask,labels=None).logits.float().cpu()
            for r,vec in zip(batch,v):
                if a.sample_ids:sample_vectors.append(vec.numpy())
                writer.write(json.dumps({"item_id":r["item_id"],"item":r["item"],
                    "hidden_states":vec.tolist()},allow_nan=False)+"\n")
    elapsed=time.monotonic()-started
    if a.sample_ids:
        table=np.vstack([np.zeros(4096,dtype=np.float32)]+sample_vectors)
        json_new(out/"SAMPLE_COMPLETE.json",{"kind":"sample_admission","canonical_ids":[r["item_id"] for r in rows],
            "rows":len(rows),"quality":table_quality(table),"encoder_key":key,
            "config_sha256":sha256(a.config),"code_sha256":code_manifest()["code_sha256"],"prompt_sha256":sha256(c["prompt"]),
            "adapter_sha256":sha256(adapter/"adapter_model.bin"),
            "forward_seconds":elapsed,"model_load_seconds":model_load_seconds,"peak_gpu_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),
            "not_for_pca":True})
        return
    table=read_vectors(out/"raw.jsonl",mapping)
    np.save(out/"table_raw.npy",table,allow_pickle=False)
    json_new(out/"COMPLETE.json",{"kind":"full_catalog","rows":len(rows),"dim":4096,
        "encoder_key":key,"quality":table_quality(table),"raw_sha256":sha256(out/"raw.jsonl"),
        "table_sha256":sha256(out/"table_raw.npy"),"truncated":truncated,"field_tokens_clipped":field_clips,
        "token_length_min":min(lengths),"token_length_max":max(lengths),
        "forward_seconds":elapsed,"model_load_seconds":model_load_seconds,"peak_gpu_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved()})
if __name__=="__main__":main()
