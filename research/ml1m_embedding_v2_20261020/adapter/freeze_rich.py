"""Freeze a REAL audited Rich prompt/config. Does not train or generate embeddings."""
import argparse,json
from pathlib import Path
from common import ROOT,load_config,load_mapping,load_prompts,sha256,json_new
from audit_inputs_cpu import audit
def main():
    p=argparse.ArgumentParser();p.add_argument("--prompt",required=True);p.add_argument("--coverage",required=True)
    p.add_argument("--execute",action="store_true");a=p.parse_args()
    c=load_config(ROOT/"configs/scft_rich_pending.json");policy=json.load(open(c["readiness_policy"]))
    rows=load_prompts(a.prompt,load_mapping(c["mapping"]));cov=json.load(open(a.coverage))
    if not cov.get("ready_for_rich_scft") or cov["overview_coverage"]<policy["min_overview_coverage"]:
        raise ValueError("Insufficient verified overview coverage; do not fall back to Base SCFT")
    if len(rows)!=cov["items"]:raise ValueError("Coverage/catalog mismatch")
    manifest=json.load(open(str(a.prompt)+".manifest.json"))
    rich_digests=[v["sha256"] for k,v in manifest.get("inputs",{}).items() if k.endswith("rich_verified.json")]
    if cov.get("rich_source_sha256") not in rich_digests:
        raise ValueError("Coverage and prompt refer to different Rich sources")
    if not manifest.get("rich_source_validated") or manifest.get("prompt_sha256")!=sha256(a.prompt):
        raise ValueError("Missing validated prompt-generation manifest")
    from transformers import AutoTokenizer
    c["prompt"]=str(Path(a.prompt).resolve());c["expected_prompt_sha256"]=sha256(a.prompt)
    token=AutoTokenizer.from_pretrained(c["model_path"],local_files_only=True,use_fast=True)
    stats=audit(c,token)
    if stats["truncated_fraction"]>policy["max_truncated_view_fraction"] or stats["different_movie_token_collision_pairs"]:
        raise ValueError("Rich truncation/token-identity gate failed")
    cert={"passed":True,"prompt_sha256":sha256(a.prompt),"mapping_sha256":sha256(c["mapping"]),
          "coverage_sha256":sha256(a.coverage),"coverage":cov,"augmentation":stats}
    print(json.dumps(cert,indent=2))
    if a.execute:
        json_new(c["readiness_certificate"],cert)
        json_new(ROOT/"configs/scft_rich_frozen.json",c)
        enc=load_config(ROOT/"configs/encode_rich_proposed.json")
        enc["prompt"]=c["prompt"];enc["expected_prompt_sha256"]=c["expected_prompt_sha256"]
        json_new(ROOT/"configs/encode_rich_frozen.json",enc)
if __name__=="__main__":main()
