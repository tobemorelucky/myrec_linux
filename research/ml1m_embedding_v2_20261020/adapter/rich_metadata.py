"""Movie identity + official metadata intake. Offline by default; no generated text."""
import argparse,csv,hashlib,html,json,os,re,time,unicodedata,urllib.request,urllib.error
from pathlib import Path
from common import ROOT,load_mapping,json_new,sha256,prompt_attributes

def digest_values(values):
    return hashlib.sha256(json.dumps(values,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def title_key(title):
    s=re.sub(r"\s*\(\d{4}\)\s*$","",html.unescape(title)).strip()
    s=re.sub(r",\s*(The|A|An)$",lambda m:"",s,flags=re.I)
    s=re.sub(r"^(The|A|An)\s+","",s,flags=re.I)
    return "".join(c for c in unicodedata.normalize("NFKD",s).casefold() if c.isalnum())

def catalog(movies,mapping):
    out={}
    if Path(movies).suffix in (".jsonl",".jsonline"):
        from common import load_prompts
        for r in load_prompts(movies,mapping):
            body=r["input"].split("\n",1)[1]
            parts=prompt_attributes(r["input"])
            attrs={x.split(" is ",1)[0]:x.split(" is ",1)[1] for x in parts if " is " in x}
            if len(parts)!=3 or set(attrs)!={"title","release year"}:
                raise ValueError("Only original title/year/genres Base is allowed; no ratings")
            title=attrs["title"];year=attrs["release year"]
            genres=next(x[len("genres are "):] for x in parts if x.startswith("genres are "))
            raw=str(r["item"]);out[raw]={"movie_id":raw,"item_id":mapping[raw],"title":title,"year":year,"genres":genres.split(", ")}
        return out
    for line in Path(movies).read_text(encoding="latin-1").splitlines():
        if not line.strip():continue
        raw,title,genres=line.split("::")
        if raw not in mapping:continue
        m=re.search(r"\((\d{4})\)\s*$",title)
        if not m:raise ValueError("Missing catalog year")
        out[raw]={"movie_id":raw,"item_id":mapping[raw],"title":title[:m.start()].strip(),
                  "year":m.group(1),"genres":genres.split("|")}
    if set(out)!=set(mapping):raise ValueError("Incomplete canonical catalog")
    return out

def read_links(path,later_movies=None,base=None):
    rows=list(csv.DictReader(open(path,encoding="utf-8-sig",newline="")))
    out={}
    for x in rows:
        raw=x["movieId"]
        if raw in out:raise ValueError("Duplicate MovieLens ID in links")
        imdb=x.get("imdbId","").strip();tmdb=x.get("tmdbId","").strip()
        if not re.fullmatch(r"\d+",imdb or "") or not re.fullmatch(r"\d+",tmdb or ""):continue
        out[raw]={"imdb_id":"tt"+imdb.zfill(7),"tmdb_id":int(tmdb)}
    if later_movies and base is not None:
        later={r["movieId"]:r["title"] for r in csv.DictReader(open(later_movies,encoding="utf-8-sig"))}
        for raw in set(out)&set(base):
            title=later.get(raw,"");m=re.search(r"\((\d{4})\)\s*$",title)
            if not m or m.group(1)!=base[raw]["year"] or title_key(title)!=title_key(base[raw]["title"]):
                # MovieLens IDs are not accepted across releases without title/year verification.
                out.pop(raw)
    return out

GENRE={"Children's":"Family","Sci-Fi":"Science Fiction","Film-Noir":"Crime"}
def verify_response(c,link,response):
    reasons=[]
    if response.get("id")!=link["tmdb_id"] or response.get("imdb_id")!=link["imdb_id"]:
        reasons.append("external_id_mismatch")
    titles=[response.get("title",""),response.get("original_title","")]
    titles += [x.get("title","") for x in response.get("alternative_titles",{}).get("titles",[])]
    if title_key(c["title"]) not in {title_key(t) for t in titles if t}:reasons.append("title_ambiguous")
    if response.get("release_date","")[:4]!=c["year"]:reasons.append("year_ambiguous")
    genres={g["name"].casefold() for g in response.get("genres",[])}
    expected={GENRE.get(g,g).casefold() for g in c["genres"]}
    if not genres or not expected&genres:reasons.append("genre_requires_manual_review")
    return reasons

def make_record(c,link,response,retrieved_at):
    reasons=verify_response(c,link,response)
    if reasons:return None,reasons
    values={}
    overview=response.get("overview","").strip()
    if overview:values["overview"]=overview
    crew=response.get("credits",{}).get("crew",[])
    directors=list(dict.fromkeys(x["name"] for x in crew if x.get("job")=="Director" and x.get("name")))
    cast=[x["name"] for x in sorted(response.get("credits",{}).get("cast",[]),
                                   key=lambda x:x.get("order",999))[:3] if x.get("name")]
    if directors:values["director"]=directors
    if cast:values["cast"]=cast
    if not values:return None,["no_rich_content"]
    rec={"movie_id":c["movie_id"],"title":c["title"],"year":c["year"],**values,
         "verified":True,"external_ids":link,
         "identity":{"title_year_genres_checked":True,"external_ids_checked":True},
         "sources":[{"url":"https://www.themoviedb.org/movie/"+str(link["tmdb_id"]),
                     "source_id":str(link["tmdb_id"]),"retrieved_at":retrieved_at,
                     "license":"TMDB non-commercial API terms; attribution required",
                     "fields":list(values),"text_sha256":digest_values(values)}]}
    return rec,[]

def fetch_samples(cat,links,out,count,approved_bulk=False,resume=False):
    if not 1<=count<=len(cat) or (count>5 and not approved_bulk):
        raise ValueError("More than five requests requires separately approved --approved-bulk and exact item limit")
    token=os.environ.get("TMDB_READ_TOKEN")
    if not token:raise ValueError("Set TMDB_READ_TOKEN privately; never put credentials in config or reports")
    if not out.resolve().is_relative_to(ROOT):raise ValueError("Output outside independent workspace")
    out.mkdir(parents=True,exist_ok=resume)
    selected=[r for r in sorted(cat,key=lambda r:cat[r]["item_id"]) if r in links][:count]
    for raw in selected:
        existing=out/(raw+".json")
        if existing.exists():
            saved=json.load(open(existing))
            if not resume or saved["response"].get("id")!=links[raw]["tmdb_id"]:
                raise ValueError("Existing response identity differs; no overwrite")
            continue
        url="https://api.themoviedb.org/3/movie/"+str(links[raw]["tmdb_id"])+"?language=en-US&append_to_response=credits,alternative_titles"
        req=urllib.request.Request(url,headers={"Authorization":"Bearer "+token,"Accept":"application/json"})
        try:
            with urllib.request.urlopen(req,timeout=20) as h:data=json.load(h)
        except urllib.error.HTTPError as e:raise RuntimeError("TMDB HTTP "+str(e.code)+"; stop without automatic retry") from None
        # Never retain ratings/popularity/vote counts as model inputs.
        allowed=("id","imdb_id","title","original_title","release_date","genres","overview","credits","alternative_titles")
        data={k:data[k] for k in allowed if k in data}
        credits=data.get("credits",{})
        data["credits"]={"crew":[{k:x[k] for k in ("name","job") if k in x} for x in credits.get("crew",[])],
                         "cast":[{k:x[k] for k in ("name","order") if k in x} for x in credits.get("cast",[])]}
        json_new(out/(raw+".json"),{"retrieved_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"response":data})
        time.sleep(.3)

def audit(cat,links,responses):
    records=[];pending=[];external_used={}
    for raw,c in sorted(cat.items(),key=lambda x:x[1]["item_id"]):
        if raw not in links:pending.append({"movie_id":raw,"reason":["missing_or_unverified_links"]});continue
        p=Path(responses)/(raw+".json") if responses else None
        if p is None or not p.is_file():pending.append({"movie_id":raw,"reason":["response_not_collected"]});continue
        envelope=json.load(open(p));r,reason=make_record(c,links[raw],envelope["response"],envelope["retrieved_at"])
        ext=links[raw]["tmdb_id"]
        if r and ext in external_used:raise ValueError("Same external movie mapped to multiple canonical movies: manual review required")
        if r:records.append(r);external_used[ext]=raw
        else:pending.append({"movie_id":raw,"reason":reason})
    n=len(cat);plots=sum(bool(r.get("overview")) for r in records)
    report={"items":n,"verified_rich":len(records),"verified_overview":plots,
            "overview_coverage":plots/n,"overview_missing":n-plots,
            "director_coverage":sum(bool(r.get("director")) for r in records)/n,
            "pending":len(pending),"ambiguous_observed":sum(any("ambiguous" in q or "mismatch" in q or "review" in q for q in r["reason"]) for r in pending),
            "uninspected":sum("response_not_collected" in r["reason"] or "missing_or_unverified_links" in r["reason"] for r in pending),
            "ready_for_rich_scft":plots/n>=.9 and all(r["verified"] for r in records),
            "coverage_gate_note":"Predeclared engineering minimum 90%; not a recommendation-performance threshold"}
    return records,pending,report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--movies",required=True);p.add_argument("--mapping",required=True)
    p.add_argument("--links");p.add_argument("--catalog-movies");p.add_argument("--responses");p.add_argument("--out")
    p.add_argument("--fetch-samples",type=int,default=0);p.add_argument("--confirm-tmdb-noncommercial-terms",action="store_true")
    p.add_argument("--approved-bulk",action="store_true",help="Requires separate user authorization; exact --fetch-samples remains the hard item budget")
    p.add_argument("--resume-existing",action="store_true",help="Reuse verified response IDs; never overwrite or retry HTTP errors")
    p.add_argument("--execute",action="store_true")
    a=p.parse_args();cat=catalog(a.movies,load_mapping(a.mapping))
    if a.links and not a.catalog_movies:raise ValueError("Provide matching MovieLens release movies.csv for identity verification")
    links=read_links(a.links,a.catalog_movies,cat) if a.links else {}
    if a.fetch_samples:
        if not a.execute or not a.confirm_tmdb_noncommercial_terms or not a.out:raise ValueError("Explicit sample collection/terms/output required")
        fetch_samples(cat,links,Path(a.out),a.fetch_samples,a.approved_bulk,a.resume_existing);return
    records,pending,report=audit(cat,links,a.responses)
    report["inputs"]={x:sha256(x) for x in [a.movies,a.mapping,a.links,a.catalog_movies] if x}
    print(json.dumps(report,indent=2))
    if a.execute:
        if not a.out:raise ValueError("Need new output folder")
        out=Path(a.out).resolve()
        if not out.is_relative_to(ROOT):raise ValueError("Outside independent root")
        out.mkdir(parents=True,exist_ok=False)
        json_new(out/"rich_verified.json",records);json_new(out/"pending.json",pending)
        report["rich_source_sha256"]=sha256(out/"rich_verified.json")
        json_new(out/"coverage.json",report)
if __name__=="__main__":main()
