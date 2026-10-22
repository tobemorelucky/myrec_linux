"""Small stage identity/backup manifest checks. No model or automatic transfer."""
import argparse,json
from pathlib import Path
from common import ROOT,load_config,sha256,json_new,code_manifest
def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["B","E","F"],required=True);p.add_argument("--profile",choices=["legacy256","full1024"],default="legacy256");a=p.parse_args()
    full=a.profile=="full1024"
    ar="admission_a_old_ml_lora_full1024" if full else "admission_a_old_ml_lora"
    sr="scft_rich_full1024_seed42" if full else "scft_rich_seed42"
    if a.stage=="B":
        proof=json.load(open(ROOT/"runs"/ar/"SAMPLE_COMPLETE.json"))
        c=load_config(ROOT/"configs"/("encode_v1_rebuilt_full1024.json" if full else "encode_v1_rebuilt.json"))
        if proof["kind"]!="sample_admission" or proof["prompt_sha256"]!=sha256(c["prompt"]):
            raise ValueError("Stage A input proof mismatch")
        if proof["adapter_sha256"]!=sha256(Path(c["adapter_path"])/"adapter_model.bin"):raise ValueError("Adapter changed")
        if proof["code_sha256"]!=code_manifest()["code_sha256"]:raise ValueError("Code changed since A")
    elif a.stage=="E":
        complete=json.load(open(ROOT/"runs"/sr/"SCFT_COMPLETE.json"))
        c=load_config(ROOT/"configs"/("scft_rich_full1024_frozen.json" if full else "scft_rich_frozen.json"))
        if complete["production_config"]!=c or complete["optimizer_steps"]!=1000:raise ValueError("SCFT identity/steps mismatch")
        if complete["code_sha256"]!=code_manifest()["code_sha256"]:raise ValueError("Code changed since SCFT")
    else:
        selected=[]
        for folder in ("v1_rebuilt_raw","v1_rebuilt_pca1536","scft_rich_seed42","encode_base","encode_rich","v2_common_pca1536"):
            folder=(sr if folder=="scft_rich_seed42" else folder+"_full1024") if full else folder
            base=ROOT/"runs"/folder
            if not base.is_dir():raise FileNotFoundError("Missing production stage "+folder)
            selected.extend(p for p in base.rglob("*") if p.is_file() and "cache" not in p.parts)
        manifest={str(p.relative_to(ROOT)):{"bytes":p.stat().st_size,"sha256":sha256(p)} for p in sorted(selected)}
        json_new(ROOT/"runs"/("ASSET_BACKUP_MANIFEST_full1024.json" if full else "ASSET_BACKUP_MANIFEST.json"),manifest)
        print("Manifest only; no compression, deletion, transfer or V1 replacement")
if __name__=="__main__":main()
