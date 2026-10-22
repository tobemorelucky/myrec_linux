"""CPU-only quality of published Rich inputs; tokenizer only, no pretrained model."""
import argparse,json,random,collections,itertools
from pathlib import Path
import numpy as np
from common import ROOT,load_config,load_mapping,load_prompts,sha256,json_new,encode_prompt,cap_text_fields,prompt_attributes,dropout_feature
from audit_inputs_cpu import audit,describe

def inspect(c,token,coverage):
    rows=load_prompts(c["prompt"],load_mapping(c["mapping"]))
    old=load_prompts("/root/autodl-tmp/pomrec/data/ml-1m/item_str.jsonline",load_mapping(c["mapping"]))
    base={r["item_id"]:r for r in old}
    result=audit(c,token);rng=random.Random(c["seed"])
    raw_lengths=[];clipped=0;field_counts=collections.Counter()
    attr_counts=[];title_only=0;views_overview=0;no_overview_pairs=0;owners=collections.defaultdict(set)
    bad_base=[];bad_fields=[];counts=collections.Counter()
    file_order=[json.loads(s) for s in open(c["prompt"],encoding="utf-8") if s.strip()]
    allowed={"title","release year","genres","overview","plot","director","cast","themes"}
    for row in file_order:
        item=int(row["item_id"]);raw_lengths.append(len(token.encode(row["input"],add_special_tokens=False))+1)
        attrs=prompt_attributes(row["input"])
        names=[x.split(" is ",1)[0] if " is " in x else x.split(" are ",1)[0] for x in attrs]
        if set(names)-allowed:bad_fields.append(item)
        original=base[item]["input"]
        if row.get("sources"):
            if not row["input"].startswith(original+" overview is "):bad_base.append(item)
        elif row["input"]!=original:bad_base.append(item)
        _,stat=encode_prompt(row["input"],token,c["max_source_length"],c.get("field_caps"))
        clipped+=bool(stat["field_tokens_clipped"]);field_counts.update(stat["field_tokens_clipped"].keys())
        text=cap_text_fields(row["input"],token,c.get("field_caps"))
        views=dropout_feature(text,.4,rng);has=[]
        for view in views:
            parsed=prompt_attributes(view);attr_counts.append(len(parsed))
            title_only+=len(parsed)==1 and parsed[0].startswith("title ")
            has.append(any(x.startswith(("overview is ","plot is ")) for x in parsed))
            seq=tuple(encode_prompt(view,token,c["max_source_length"])[0]);owners[seq].add(item)
        views_overview+=sum(has);no_overview_pairs+=not any(has)
    pairs=set()
    for ids in owners.values():
        if len(ids)>1:pairs.update(itertools.combinations(sorted(ids),2))
    if len(pairs)!=result["different_movie_token_collision_pairs"]:raise AssertionError("Augmentation audit mismatch")
    details=[{"item_ids":list(pair),"movie_ids":[rows[i-1]["item"] for i in pair],
              "base_inputs":[base[i]["input"] for i in pair]} for pair in sorted(pairs)]
    result.update({
        "uncapped_prompt_token_lengths":describe(raw_lengths),
        "uncapped_prompts_over_256":sum(n>256 for n in raw_lengths),
        "uncapped_prompt_truncation_fraction_at_256":sum(n>256 for n in raw_lengths)/len(rows),
        "items_with_explicit_field_budget_clipping":clipped,
        "field_clipping_fraction":clipped/len(rows),"field_clip_item_counts":dict(field_counts),
        "minimum_attributes_per_view":min(attr_counts),"maximum_attributes_per_view":max(attr_counts),
        "title_only_views":title_only,"views_containing_overview":views_overview,
        "items_neither_view_contains_overview":no_overview_pairs,
        "base_identity_bad_count":len(bad_base),"unexpected_attribute_ids":bad_fields,
        "collision_details":details,"coverage":coverage,
        "ready_for_freeze":coverage["overview_coverage"]>=.9 and result["truncated_fraction"]<=.05 and not pairs
                           and not bad_base and not bad_fields and title_only==0,
        "budget_note":"Raw text remains intact in source; 160-token overview budget is explicitly applied BEFORE BOTH augmentation and encoding. Zero post-budget total truncation does not imply lossless plots."
    })
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--coverage",required=True)
    p.add_argument("--out");p.add_argument("--execute",action="store_true");a=p.parse_args()
    c=load_config(a.config)
    from transformers import AutoTokenizer
    t=AutoTokenizer.from_pretrained(c["model_path"],local_files_only=True,use_fast=True)
    report=inspect(c,t,json.load(open(a.coverage)))
    print(json.dumps(report,indent=2))
    if a.execute:
        if not a.out:raise ValueError("Need new output path")
        json_new(a.out,report)
if __name__=="__main__":main()
