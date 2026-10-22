"""Full-text 1024 CPU admission: tokenizer/dataset only; no 7B or GPU."""
import argparse,functools,json,random,sys
from pathlib import Path
import numpy as np
from common import ROOT,load_config,load_mapping,load_prompts,sha256,json_new,encode_prompt,dropout_feature,code_manifest
from audit_inputs_cpu import audit,describe

def main():
    p=argparse.ArgumentParser();p.add_argument("--execute",action="store_true");a=p.parse_args()
    if not a.execute:raise ValueError("Explicit CPU output authorization required")
    c=load_config(ROOT/"configs/scft_rich_full1024_pending.json")
    if c["max_source_length"]!=1024 or c["field_caps"]:raise ValueError("Wrong full-text policy")
    from transformers import AutoTokenizer
    t=AutoTokenizer.from_pretrained(c["model_path"],local_files_only=True,use_fast=True)
    t.pad_token=t.unk_token;t.padding_side="right"
    rows=load_prompts(c["prompt"],load_mapping(c["mapping"]))
    source=json.load(open(ROOT/"inputs/rich_zenodo_20261020/rich_verified.json"))
    by_raw={x["movie_id"]:x for x in source}
    for row in rows:
        if row["item"] in by_raw:
            text=str(by_raw[row["item"]]["overview"]).replace("\n"," ").replace(";"," ").strip()
            assert row["input"].split("overview is ",1)[1]==text+";","Incomplete normalized overview"
    assert sum(bool(x.get("sources")) for x in rows)==3666
    assert [json.loads(x)["item_id"] for x in open(c["prompt"]) if x.strip()]==list(range(1,3707))
    stats=audit(c,t)
    body=[];encoded=[];pair_ids=[];rng=random.Random(c["seed"])
    for row in rows:
        ids,st=encode_prompt(row["input"],t,1024,{},strict=True)
        assert not st["total_truncated"] and not st["field_tokens_clipped"]
        assert ids==t.encode(row["input"],add_special_tokens=False)+[t.eos_token_id]
        body.append(len(ids)-1);encoded.append(ids)
        pair=dropout_feature(row["input"],.4,rng)
        pair_ids.append([encode_prompt(v,t,1024,strict=True)[0] for v in pair])
    # Match the actual official load/map/set_format/collator input pipeline.
    from datasets import load_dataset
    from types import SimpleNamespace
    from training_support import V2TrainMask,scft_text_only,FixedAdmissionSampler
    import llm.data_processor.llama as processor
    from llm.data_processor.collator import PairwiseDataCollatorWithPadding
    processor.dropout_feature=functools.partial(dropout_feature,rng=random.Random(c["seed"]))
    class FullMask(V2TrainMask):
        fail_on_truncation=True
        field_caps={}
    f=FullMask(SimpleNamespace(max_source_length=1024,max_target_length=196,dropout_ratio=.4,ignore_pad_token_for_loss=True),None,t)
    data=scft_text_only(load_dataset("json",data_files={"train":c["prompt"]},
                       cache_dir=str(ROOT/"checks/full1024_dataset_cache")))["train"]
    data=data.map(f,batched=True,num_proc=1,desc="Full1024 CPU preprocessing");data.set_format("torch")
    maximum=0;input_lengths=[];all_equal=True
    for i in range(len(rows)):
        rec=data[i]
        for k,name in enumerate(["chosen","rejected"]):
            seq=rec[name+"_ids"].tolist()
            assert seq==pair_ids[i][k],"Official/encoding EOS or truncation mismatch"
            assert bool(rec[name+"_mask"].all()) and seq[-1]==t.eos_token_id
            input_lengths.append(len(seq));maximum=max(maximum,len(seq))
    collator=PairwiseDataCollatorWithPadding(t)
    # C: 3 updates x 8 microbatches x 16 items; first batch stresses longest views.
    rawlong=max(range(len(rows)),key=lambda i:len(encoded[i]))
    ranked=sorted(range(len(rows)),key=lambda i:max(map(len,pair_ids[i])),reverse=True)
    longest=[rawlong]+[i for i in ranked if i!=rawlong][:15]
    pool=[i for i in range(len(rows)) if i not in longest]
    random.Random(42).shuffle(pool);indices=longest+pool[:368]
    assert len(set(indices))==384
    sampler=FixedAdmissionSampler(indices,len(data));assert list(sampler)==indices
    batches=[]
    for j in range(0,len(indices),16):
        idx=indices[j:j+16];x=collator([data[i] for i in idx])
        assert x["input_ids"].shape[0]==32
        assert x["input_ids"].shape[1]==max(len(s) for i in idx for s in pair_ids[i])
        assert x["input_ids"].shape[1]<1024,"No forced 1024 padding"
        assert (x["input_ids"][~x["attention_mask"].bool()]==t.pad_token_id).all()
        batches.append({"microbatch":j//16+1,"optimizer_step":j//128+1,
                        "item_ids":[rows[i]["item_id"] for i in idx],
                        "shape":list(x["input_ids"].shape),
                        "nonpadding_tokens":int(x["attention_mask"].sum())})
    plan={"prompt_sha256":sha256(c["prompt"]),"dataset_length":len(rows),"indices":indices,
          "item_ids":[rows[i]["item_id"] for i in indices],"batch_size":16,"gradient_accumulation_steps":8,
          "optimizer_steps":3,"longest_raw_item_id":rows[rawlong]["item_id"],
          "maximum_raw_with_eos":max(map(len,encoded)),"maximum_view_with_eos":maximum,
          "first_batch_longest_views_then_random_mixed":True,"microbatches":batches,
          "formal_training_uses_original_sampler":True}
    assert max(x["shape"][1] for x in batches)==maximum
    json_new(c["admission_plan"],plan);c["expected_admission_plan_sha256"]=sha256(c["admission_plan"])
    report={"policy":{"max_source_length":1024,"field_caps":{},"fail_on_truncation":True},
            "items":3706,"complete_normalized_overviews":3666,"base_fallback":40,
            "prompt_sha256":sha256(c["prompt"]),"prompt_file_changed":False,
            "raw_body_tokens":describe(body),"raw_body_plus_eos_tokens":describe([n+1 for n in body]),
            "official_model_input_tokens":describe(input_lengths),"augmentation":stats,
            "body_eos_and_official_input_equal":True,"full_prompt_retention_fraction":1.0,
            "field_clipped_items":0,"unaugmented_over_1024":0,"augmented_over_1024":0,
            "dynamic_padding_verified":True,"forced_1024_padding":False,
            "admission_first_batch":batches[0],"all_24_admission_microbatches_checked":True,
            "no_pretrained_model_loaded":True,"plan_sha256":c["expected_admission_plan_sha256"],
            "normalization_note":"Only existing newline/semicolon attribute formatting; no plot token clipping. Attribute dropout still intentionally omits attributes in training views."}
    assert stats["truncated_views"]==0 and stats["field_clipped_items"]==0
    assert stats["different_movie_token_collision_pairs"]==0
    cert={"passed":True,"prompt_sha256":sha256(c["prompt"]),"mapping_sha256":sha256(c["mapping"]),
          "length_policy":report["policy"],"coverage":{"verified_overview":3666,"overview_coverage":3666/3706},
          "fulltext_report":report}
    json_new(c["readiness_certificate"],cert)
    json_new(ROOT/"configs/scft_rich_full1024_frozen.json",c)
    json_new(ROOT/"checks/fulltext1024_cpu_20261020.json",report)
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
