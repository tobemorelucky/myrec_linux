"""Import the exact public RecSys25/Zenodo movie text release. CPU only.
Never ingest reviews/tags/ratings into training prompts. No fuzzy ID matching.
"""
import argparse,csv,hashlib,html,json,re,collections,time
from pathlib import Path
from common import ROOT,load_mapping,json_new,write_new,sha256
from rich_metadata import catalog,digest_values

EXPECTED={
 "text_ml1m.tsv":{"md5":"a98afbff73f503ce705613ed7e6efe9c","sha256":"4676223139ff78710f4451bcca0cc389cca03303db6f4980d96eefc1f90925a3"},
 "ml1m_full_extended_mapping.tsv":{"md5":"d176119fe9f2f6b5da13f5ca843521fa","sha256":"a3b548c2bfe8fa14155ba22900f82c21419cf38f56176f6ae5f31372af15f46d"}}
SOURCE_URL="https://zenodo.org/records/15403972"
SOURCE_COMMIT="e8c90c3f93ed14753fec4905d37c401de2255160"
NULLS={"","nan","none","null","[]","no abstract found.","no plot available","n/a"}

def checked_rows(path,delimiter,key,expected_name=None):
    p=Path(path)
    if expected_name:
        spec=EXPECTED[expected_name]
        if sha256(p)!=spec["sha256"] or hashlib.md5(p.read_bytes()).hexdigest()!=spec["md5"]:
            raise ValueError("File differs from pinned official release: "+expected_name)
    csv.field_size_limit(20_000_000)
    with p.open(encoding="utf-8-sig",newline="") as stream:
        rows=list(csv.DictReader(stream,delimiter=delimiter))
    ids=[x[key] for x in rows]
    if any(not re.fullmatch(r"[1-9]\d*",i) for i in ids):raise ValueError("Noncanonical published ID")
    counts=collections.Counter(ids)
    duplicates={k:v for k,v in counts.items() if v>1}
    if duplicates:raise ValueError("Duplicate source IDs; manual review: "+str(duplicates))
    if any(None in x for x in rows):raise ValueError("Malformed CSV field count")
    return {x[key]:x for x in rows}

def exact_title_year(c,name):
    expected=c["title"]+" ("+c["year"]+")"
    # Case/HTML/whitespace normalization only, never similarity thresholds.
    normalize=lambda x:re.sub(r"\s+"," ",html.unescape(x)).strip().casefold()
    return normalize(expected)==normalize(name)

def valid_plot(text):
    s=re.sub(r"\s+"," ",text).strip()
    if s.casefold() in NULLS:return None,"empty_plot"
    if len(re.findall(r"\w+",s))<5:return None,"too_short_plot"
    if re.search(r"^\s*(?:error:|traceback|as an ai|i(?:'m| am) sorry)",s,re.I):
        return None,"invalid_or_generated_placeholder"
    if "\x00" in s:return None,"nul_in_text"
    return s,None

def convert(movies,mapping,text,links,verify_release=True):
    cat=catalog(movies,mapping)
    text_sha=sha256(text)
    td=checked_rows(text,",","item","text_ml1m.tsv" if verify_release else None)
    ld=checked_rows(links,"\t","movie_id","ml1m_full_extended_mapping.tsv" if verify_release else None)
    by_external=collections.defaultdict(list)
    for raw,x in ld.items():
        for key in ("dburl","wiki_url"):
            if x.get(key):by_external[(key,x[key])].append(raw)
    shared={key+":"+url:ids for (key,url),ids in by_external.items() if len(ids)>1}
    if shared:raise ValueError("Published external URI maps to multiple movies; manual review required")
    rich=[];mapping_rows=[];missing=[];empty=[];pending=[];genre_warnings=[]
    descriptions=collections.defaultdict(list)
    actual_time=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
    title_verified=0;source_id_only=0;accepted_identity=0
    for raw,c in sorted(cat.items(),key=lambda x:x[1]["item_id"]):
        detail={k:c[k] for k in ("movie_id","item_id","title","year")}
        ext=ld.get(raw);entry=td.get(raw)
        method="published_original_movielens_movie_id"
        status="matched";reason=""
        if entry is None:
            status="missing_text_record";missing.append(detail)
        elif ext and not exact_title_year(c,ext["name"]):
            status="identity_pending";reason="published_title_or_year_mismatch"
            pending.append({**detail,"source_name":ext["name"],"reason":reason})
        else:
            accepted_identity+=1
            if ext:title_verified+=1
            else:source_id_only+=1
            plot,reason=valid_plot(entry.get("description",""))
            if plot is None:
                status="empty_or_invalid_plot";empty.append({**detail,"reason":reason})
            else:
                # These are publisher-supplied plots, NOT independently fact-checked
                # per movie or relabeled as current DBpedia abstracts.
                values={"overview":plot}
                identity={"method":method,"source_id_semantics_confirmed":True,
                          "canonical_item_id_checked":True,"metadata_conflict":False,
                          "source_title_year_crosscheck":bool(ext),
                          "facts_independently_verified":False}
                external={"zenodo_record":"15403972","published_movie_id":raw}
                if ext:
                    if ext.get("dburl"):external["dbpedia"]=ext["dburl"]
                    if ext.get("wiki_url"):external["wikipedia"]=ext["wiki_url"]
                rich.append({"movie_id":raw,"title":c["title"],"year":c["year"],**values,
                    "verified":True,"external_ids":external,"identity":identity,
                    "sources":[{"url":SOURCE_URL,"source_id":"MovieLens MovieID="+raw,
                        "retrieved_at":actual_time,"license":"CC-BY-4.0 (Zenodo deposit); underlying text rights retained",
                        "fields":["overview"],"text_sha256":digest_values(values),
                        "dataset_file_sha256":text_sha,
                        "source_field":"description",
                        "excluded_fields":["tags","reviews","genres"],
                        "provenance_note":"Published movie plot; final per-record crawl provenance is not supplied"}]})
                descriptions[plot].append(raw)
            orig=set(c["genres"]);orig={x.replace("Children's","Children") for x in orig}
            source_genres=set(entry.get("genres","").split())
            if not orig&source_genres:
                genre_warnings.append({**detail,"base_genres":c["genres"],"source_genres":entry.get("genres",""),
                                       "action":"keep original Base genres; no ID remap"})
        mapping_rows.append({**detail,"source_item":raw if entry is not None else "",
                             "source_name":ext["name"] if ext else "",
                             "matching_method":method if entry is not None else "",
                             "status":status,"reason":reason or ""})
    duplicates={s[:100]:ids for s,ids in descriptions.items() if len(ids)>1}
    if duplicates:raise ValueError("Duplicate imported plot text; manual content review required")
    total=len(mapping)
    report={"items":total,"source_text_rows":len(td),"source_mapping_rows":len(ld),
            "source_text_target_id_overlap":len(set(td)&set(mapping)),
            "successful_unique_identity_matches":accepted_identity,
            "title_year_crosschecked":title_verified,"publisher_id_without_external_mapping":source_id_only,
            "duplicate_source_ids":0,"many_to_one_external_ids":0,
            "identity_pending":len(pending),"missing_text_records":len(missing),
            "empty_or_invalid_target_plots":len(empty),
            "verified_rich":len(rich),"verified_overview":len(rich),"overview_coverage":len(rich)/total,
            "overview_missing":total-len(rich),"director_coverage":0.0,"cast_coverage":0.0,
            "pending":total-len(rich),"ambiguous_observed":len(pending),"uninspected":0,
            "ready_for_rich_scft":len(rich)/total>=.9,
            "genre_taxonomy_warnings":len(genre_warnings),
            "excluded_source_columns":["tags","reviews","genres"],
            "llm_generated_claim":"No LLM generation evidence in publisher documentation; per-record crawl chain incomplete",
            "license":{"zenodo_deposit":"cc-by-4.0","attribution_required":True,
                       "underlying_source_rights_not_relicensed":True},
            "source":{"doi":"10.5281/zenodo.15403972","paper_doi":"10.1145/3705328.3748162",
                      "repository":"https://github.com/swapUniba/multimodal_ml1m_dbbook_lfm2k",
                      "commit":SOURCE_COMMIT},
            "inputs":{str(p):{"sha256":sha256(p),"bytes":Path(p).stat().st_size} for p in (movies,text,links)}}
    return rich,mapping_rows,missing,empty,pending,genre_warnings,report

def validate_published_identity(record):
    identity=record.get("identity",{});external=record.get("external_ids",{})
    if identity.get("method")!="published_original_movielens_movie_id":return False
    if not identity.get("source_id_semantics_confirmed") or not identity.get("canonical_item_id_checked"):
        return False
    if identity.get("metadata_conflict") is not False:return False
    if external.get("zenodo_record")!="15403972" or external.get("published_movie_id")!=record["movie_id"]:
        return False
    return bool(record.get("sources")) and all(
        x.get("url")==SOURCE_URL and x.get("dataset_file_sha256")==EXPECTED["text_ml1m.tsv"]["sha256"]
        and x.get("source_field")=="description" for x in record["sources"])

def save_tsv(path,rows,fields):
    import io
    stream=io.StringIO(newline="")
    writer=csv.DictWriter(stream,fieldnames=fields,delimiter="\t",extrasaction="ignore")
    writer.writeheader();writer.writerows(rows)
    write_new(path,stream.getvalue().encode())

def main():
    p=argparse.ArgumentParser()
    for key in ("movies","mapping","text","links","out"):p.add_argument("--"+key,required=True)
    p.add_argument("--execute",action="store_true");a=p.parse_args()
    mapping=load_mapping(a.mapping)
    if len(mapping)!=3706:raise ValueError("Expected exact 3706-item canonical mapping")
    result=convert(a.movies,mapping,a.text,a.links)
    rich,rows,missing,empty,pending,genres,report=result
    report["inputs"][a.mapping]={"sha256":sha256(a.mapping),"bytes":Path(a.mapping).stat().st_size}
    print(json.dumps({k:v for k,v in report.items() if k!="inputs"},indent=2))
    if not a.execute:return
    out=Path(a.out).resolve()
    if ROOT not in out.parents:raise ValueError("Output must stay inside existing independent V2 workspace")
    out.mkdir(parents=True,exist_ok=False)
    json_new(out/"rich_verified.json",rich)
    report["rich_source_sha256"]=sha256(out/"rich_verified.json")
    json_new(out/"coverage.json",report)
    json_new(out/"identity_pending.json",pending);json_new(out/"empty_plots.json",empty)
    json_new(out/"missing_text.json",missing);json_new(out/"genre_warnings.json",genres)
    save_tsv(out/"canonical_alignment.tsv",rows,list(rows[0]))
    save_tsv(out/"missing_text.tsv",missing,["movie_id","item_id","title","year"])
    fallback=missing+empty+pending
    save_tsv(out/"base_fallback.tsv",sorted(fallback,key=lambda x:x["item_id"]),
             ["movie_id","item_id","title","year","reason","source_name"])
if __name__=="__main__":main()
