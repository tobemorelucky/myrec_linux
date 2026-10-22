"""Explicit base+LoRA encoding, no Trainer.predict / generated-token ambiguity."""
import argparse, json, sys
from pathlib import Path
from common import load_config, ROOT, new_path, load_mapping, load_prompts, seed_all, sha256, json_new, snapshot_paths

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True)
    p.add_argument("--execute",action="store_true")
    a=p.parse_args(); c=load_config(a.config)
    if c["stage"]!="encode": raise ValueError("Wrong stage")
    print(json.dumps(c,indent=2))
    if not a.execute: return
    # Reserve new directory; incomplete runs remain labeled, never reused/overwritten.
    out=new_path(c["output_dir"])
    mapping=load_mapping(c["mapping"]); rows=load_prompts(c["prompt"],mapping)
    import torch
    from transformers import AutoTokenizer
    from model_patch import LlamaRSEmbV2
    from llm.lora_cls import PeftModelForCLS
    if not torch.cuda.is_available(): raise RuntimeError("GPU allocation required")
    seed_all(c["seed"])
    dtype={"fp16":torch.float16,"bf16":torch.bfloat16,"fp32":torch.float32}[c["precision"]]
    if dtype==torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unavailable")
    token=AutoTokenizer.from_pretrained(c["tokenizer_path"],use_fast=True,local_files_only=True)
    token.pad_token=token.unk_token; token.padding_side="right"
    model=LlamaRSEmbV2.from_pretrained(c["model_path"],pool_type="avg",tau=3.,
                    torch_dtype=dtype,local_files_only=True).cuda()
    model.config.pad_token_id=token.pad_token_id
    adapter=Path(c["adapter_path"])
    if not adapter.is_dir(): raise FileNotFoundError(adapter)
    model=PeftModelForCLS.from_pretrained(model,str(adapter),is_trainable=False)
    extra=adapter/"v2_extra_trainable.pt"
    if extra.exists():
        saved=torch.load(extra,map_location="cpu",weights_only=True)
        params=dict(model.named_parameters())
        for n,v in saved.items():
            if n not in params: raise ValueError("Unknown saved extra parameter "+n)
            params[n].data.copy_(v.to(device=params[n].device,dtype=params[n].dtype))
    elif not c.get("allow_legacy_adapter_without_tau",False):
        raise FileNotFoundError("V2 training requires saved non-LoRA parameters")
    model.eval()
    out.mkdir(parents=True)
    identity_paths=[Path(c["model_path"])/"config.json",Path(c["model_path"])/"model.safetensors.index.json",
                    Path(c["tokenizer_path"])/"tokenizer.json",Path(c["tokenizer_path"])/"tokenizer.model",
                    Path(c["tokenizer_path"])/"tokenizer_config.json",Path(c["tokenizer_path"])/"special_tokens_map.json",adapter/"adapter_config.json",
                    adapter/"adapter_model.bin"]
    identity_paths=[x for x in identity_paths if x.exists()]
    identity={"artifacts":snapshot_paths(identity_paths),
              "pooling":"masked_mean_fp32","special_tokens":"no_BOS_truncate_Lminus1_append_EOS",
              "max_length":c["max_source_length"],"precision":c["precision"],"seed":c["seed"],
              "v2_model_patch_sha256":sha256(Path(__file__).parent/"model_patch.py"),
              "v2_common_sha256":sha256(Path(__file__).parent/"common.py"),
              "upstream_commit":"3458a5e225062e94b4f1a01e41f3ec82089f0407"}
    # Hash actual large model shards in this explicitly approved stage.
    modeldir=Path(c["model_path"])
    shards=sorted(set(modeldir.glob("*.safetensors"))|set(modeldir.glob("pytorch_model*.bin")))
    if not shards: raise FileNotFoundError("Base weight shards")
    identity["base_weights"]=snapshot_paths(shards)
    if extra.exists(): identity["extra"]=snapshot_paths([extra])
    key=__import__("hashlib").sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    json_new(out/"manifest.json",{"config":c,"encoder_key":key,"encoder_identity":identity,
        "inputs":snapshot_paths([c["prompt"],c["mapping"]]),"status":"started"})
    lengths=[]; truncated=0
    with (out/"raw.jsonl").open("x",encoding="utf-8") as f,torch.inference_mode():
        for start in range(0,len(rows),c["batch_size"]):
            batch=rows[start:start+c["batch_size"]]
            seq=[]
            for r in batch:
                ids=token.encode(r["input"],add_special_tokens=False)
                lengths.append(len(ids)+1); truncated+=int(len(ids)+1>c["max_source_length"])
                seq.append(ids[:c["max_source_length"]-1]+[token.eos_token_id])
            ids=torch.full((len(seq),max(map(len,seq))),token.pad_token_id,dtype=torch.long,device="cuda")
            mask=torch.zeros_like(ids)
            for j,s in enumerate(seq):
                ids[j,:len(s)]=torch.tensor(s,device="cuda");mask[j,:len(s)]=1
            vectors=model(input_ids=ids,attention_mask=mask,labels=None).logits.float().cpu()
            for r,v in zip(batch,vectors):
                f.write(json.dumps({"item_id":r["item_id"],"item":r["item"],
                                    "hidden_states":v.tolist()},allow_nan=False)+"\n")
    from common import read_vectors
    table=read_vectors(out/"raw.jsonl",mapping)
    json_new(out/"COMPLETE.json",{"rows":len(rows),"dim":int(table.shape[1]),"encoder_key":key,
         "raw_sha256":sha256(out/"raw.jsonl"),"truncated":truncated,
         "token_length_min":min(lengths),"token_length_max":max(lengths)})
if __name__=="__main__": main()
