"""CPU tokenizer/augmentation audit only; no pretrained model or GPU."""
import argparse,json,random
from collections import Counter
from common import load_config,load_mapping,load_prompts,encode_prompt,dropout_feature,sha256
def describe(v):
    import numpy as np
    return {"n":len(v),"min":min(v),"p50":float(np.median(v)),"p95":float(np.quantile(v,.95)),"max":max(v)}
def audit(c,token):
    checked=load_prompts(c["prompt"],load_mapping(c["mapping"]));by_id={int(r["item_id"]):r for r in checked}
    file_order=[int(json.loads(s)["item_id"]) for s in open(c["prompt"],encoding="utf-8") if s.strip()]
    rows=[by_id[i] for i in file_order];rng=random.Random(c["seed"])
    lengths=[];view_lengths=[];identical=0;clipped=0;truncated=0;views=[]
    for r in rows:
        ids,stats=encode_prompt(r["input"],token,c["max_source_length"],c.get("field_caps",{}))
        lengths.append(stats["tokens_before_total_truncation"]);clipped+=bool(stats["field_tokens_clipped"])
        # Same field capping then attribute augmentation as V2TrainMask.
        from training_support import cap_text_fields
        query=cap_text_fields(r["input"],token,c.get("field_caps",{}))
        pair=dropout_feature(query,.4,rng);token_pair=[]
        for s in pair:
            seq,st=encode_prompt(s,token,c["max_source_length"])
            token_pair.append(tuple(seq));view_lengths.append(st["tokens_before_total_truncation"])
            truncated+=st["total_truncated"]
        identical+=token_pair[0]==token_pair[1];views.extend(token_pair)
    owners={}
    for i in range(len(rows)):
        for seq in set(views[2*i:2*i+2]):owners.setdefault(seq,set()).add(i)
    collision_pairs=set()
    for ids in owners.values():
        if len(ids)>1:
            import itertools
            collision_pairs.update(itertools.combinations(sorted(ids),2))
    return {"items":len(rows),"prompt_sha256":sha256(c["prompt"]),"seed":c["seed"],"audit_order":"actual SCFT input file order","file_order_is_canonical":file_order==sorted(file_order),
            "identical_two_views":identical,"identical_fraction":identical/len(rows),
            "unaugmented_lengths":describe(lengths),"view_lengths":describe(view_lengths),
            "field_clipped_items":clipped,"truncated_views":truncated,"truncated_fraction":truncated/len(views),
            "different_movie_token_collision_pairs":len(collision_pairs),
            "contrastive_other_items_full_microbatch":c["batch_size"]-1,
            "contrastive_other_items_tail_microbatch":(len(rows)%c["batch_size"] or c["batch_size"])-1,
            "optimizer_accumulation_does_not_expand_negative_pool":True}
def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);a=p.parse_args()
    c=load_config(a.config)
    from transformers import AutoTokenizer
    t=AutoTokenizer.from_pretrained(c.get("tokenizer_path",c["model_path"]),local_files_only=True,use_fast=True)
    print(json.dumps(audit(c,t),indent=2))
if __name__=="__main__":main()
