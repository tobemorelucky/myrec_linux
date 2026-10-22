"""Metadata only. No interaction labels or LLM calls. Default verifies without writes."""
import argparse, json, re
from pathlib import Path
from common import load_mapping, load_prompts, write_new, json_new, snapshot_paths

def build(movies, mapping, rich=None):
    result = []
    seen = set()
    extra = {} if rich is None else {str(x["movie_id"]):x for x in rich}
    if rich is not None and len(extra) != len(rich):
        raise ValueError("Duplicate rich movie identity")
    for line in Path(movies).read_text(encoding="latin-1").splitlines():
        if not line.strip(): continue
        raw, title, genres = line.split("::")
        if raw not in mapping: continue
        match = re.search(r"\((\d{4})\)\s*$", title)
        if not match: raise ValueError("Missing release year "+raw)
        year = match.group(1)
        clean = title[:match.start()].strip()
        attrs = ["title is "+clean[:120], "release year is "+year,
                 "genres are "+", ".join(genres.split("|")[:8])]
        provenance = None
        if raw in extra:
            x = extra[raw]
            allowed = {"movie_id","title","year","overview","themes","director","cast","sources","verified"}
            if set(x)-allowed: raise ValueError("Unexpected rich fields (ratings/statistics forbidden)")
            if x["title"] != clean or str(x["year"]) != year or x.get("verified") is not True:
                raise ValueError("Unverified title/year identity "+raw)
            if not x.get("sources"): raise ValueError("Missing source provenance")
            for s in x["sources"]:
                for k in ("url","source_id","retrieved_at","license","fields","text_sha256"):
                    if not s.get(k): raise ValueError("Missing per-source "+k)
                if not s["url"].startswith("https://"):
                    raise ValueError("Expected explicit HTTPS source")
            for k in ("overview","themes","director","cast"):
                val = x.get(k)
                if val:
                    if not any(k in s["fields"] for s in x["sources"]):
                        raise ValueError("Field without source "+k)
                    if isinstance(val,list): val=", ".join(val)
                    val=str(val).replace("\n"," ").replace(";"," ").strip()
                    attrs.append(k+" is "+val)
            provenance=x["sources"]
        r={"input":"The movie has the following attributes:\n"+"; ".join(attrs)+";",
           "target":"","item":raw,"item_id":mapping[raw]}
        if rich is not None: r["sources"]=provenance or []
        result.append(r); seen.add(raw)
    if seen != set(mapping): raise ValueError("Missing movies")
    return sorted(result,key=lambda x:x["item_id"])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--movies",required=True); p.add_argument("--mapping",required=True)
    p.add_argument("--rich"); p.add_argument("--verify-existing")
    p.add_argument("--out"); p.add_argument("--execute",action="store_true")
    a=p.parse_args(); m=load_mapping(a.mapping)
    rich=None if a.rich is None else json.load(open(a.rich,encoding="utf-8"))
    rows=build(a.movies,m,rich)
    report={"records":len(rows),"rich_records":sum(bool(r.get("sources")) for r in rows),
            "inputs":snapshot_paths([a.movies,a.mapping]+([a.rich] if a.rich else []))}
    if a.verify_existing:
        old=load_prompts(a.verify_existing,m)
        report["existing_prompts_equal"]=all(
            all(x[k]==y[k] for k in ("input","target","item","item_id")) for x,y in zip(rows,old))
        if not report["existing_prompts_equal"]: raise ValueError("Existing prompts differ")
    print(json.dumps(report))
    if a.execute:
        if not a.out: raise ValueError("--out required")
        write_new(a.out, "".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows).encode())
        json_new(a.out+".manifest.json",report)
if __name__=="__main__": main()
