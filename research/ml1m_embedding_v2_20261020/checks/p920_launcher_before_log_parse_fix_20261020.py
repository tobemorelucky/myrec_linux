"""Thin existing-main launcher: ASPCF SS, new V1 table only, Train/Dev only."""
import argparse,datetime,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
ROOT=Path("/home/yyx/hzgcode/pom2.0")
WORK=ROOT/"research/ml1m_embedding_v2_20261020"
ASSET=WORK/"imported_assets/v1_rebuilt_20261020"
PYTHON="/home/yyx/anaconda3/envs/hzg_py10/bin/python"
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()
def save(p,obj):
    with p.open("x") as f:json.dump(obj,f,indent=2)
def main():
    p=argparse.ArgumentParser();p.add_argument("--run-dir",required=True);p.add_argument("--gpu",required=True)
    a=p.parse_args();out=Path(a.run_dir).resolve()
    assert ROOT.resolve()==ROOT and out.is_relative_to(WORK) and out.is_dir()
    assert not (out/"config.json").exists()
    receipt=json.load((ASSET/"IMPORT_QA_COMPLETE.json").open())
    assert receipt["passed"] and receipt["canonical_items"]==3706
    table=ASSET/"base_pca.pkl"
    assert sha(table)==receipt["pkl_sha256"]
    historic=ROOT/"new_log/aspcf_sampled_softmax/ml-1m/k4_tau1_seed42_20261010T112637Z_round1/train.log"
    text=historic.read_text()
    expected={"K":"4","batch_size":"1024","dataset":"ml-1m","dropout":"0.1",
       "early_stop":"10","emb_size":"64","epoch":"200","history_max":"20",
       "item_encoder":"aspcf","l2":"1e-06","lambda_relation":"0.01","lr":"0.001",
       "optimizer":"Adam","random_seed":"42","sampled_softmax_temperature":"1.0",
       "relation_sample_size":"128","relation_teacher_temp":"0.1","relation_student_temp":"0.1"}
    for key,val in expected.items():
        assert re.search(r"^\s*"+re.escape(key)+r"\s+\|\s*"+re.escape(val)+r"\s*\|",text,re.M),(key,val)
    config={
      "model_name":"LLMMIRecASPCFSampledSoftmax","dataset":"ml-1m","path":"./data/",
      "gpu":a.gpu,"random_seed":42,"emb_size":64,"attn_size":64,"K":4,"history_max":20,
      "item_encoder":"aspcf","llm_emb_path":str(table),"semantic_rank":512,"semantic_dim":32,
      "semantic_hidden":128,"complement_dim":32,"tail_hidden":64,"complement_hidden":64,
      "gate_hidden":64,"aspcf_gate_mode":"basic","lambda_relation":.01,"relation_sample_size":128,
      "relation_teacher_temp":.1,"relation_student_temp":.1,"dropout":.1,"lr":.001,"l2":1e-6,
      "optimizer":"Adam","batch_size":1024,"eval_batch_size":256,"num_neg":1,
      "epoch":200,"early_stop":10,"metric":"NDCG,HR","topk":"5,10,20,50","num_workers":5,
      "test_epoch":-1,"dev_only":1,"regenerate":0,"train":1,"load":0,
      "sampled_softmax_temperature":1.,"log_file":str(out/"train.log"),"model_path":str(out/"best.pt")}
    cmd=[PYTHON,"-u","-B","main.py"]
    for key,value in config.items():cmd+=["--"+key,str(value)]
    sources=["main.py","models/sequential/LLMMIRecASPCFSampledSoftmax.py",
             "models/sequential/LLMMIRecASPCF.py","models/sequential/llmmi_components.py",
             "models/sequential/llmmi_utils.py","helpers/BaseRunner.py","helpers/SeqReader.py","models/BaseModel.py"]
    maintext=(ROOT/"main.py").read_text()
    assert "phases = ['train', 'dev'] if args.dev_only" in maintext
    save(out/"config.json",config)
    save(out/"provenance.json",{"historic_log":str(historic),"historic_log_sha256":sha(historic),
         "sources":{s:sha(ROOT/s) for s in sources},"embedding_pkl_sha256":sha(table),
         "raw_sha256":receipt["raw_sha256"],"pca_npy_sha256":receipt["pca_npy_sha256"],
         "mapping_sha256":receipt["mapping_sha256"],"only_scientific_input_change":"LLM table rebuilt in independent PCA coordinates",
         "data_sha256":{s:sha(ROOT/"data/ml-1m"/s) for s in ("train.csv","dev.csv","SeqReader.pkl")},
         "command":cmd,"from_scratch":True,"test_forbidden":True,"max_wall_seconds":14400})
    started=time.monotonic()
    save(out/"RUNNING.json",{"pid":os.getpid(),"start":datetime.datetime.now().astimezone().isoformat(),
                            "gpu":a.gpu,"checkpoint":str(out/"best.pt")})
    # No custom model, loss, Reader or Runner; exact existing main dev-only route.
    with (out/"train.out").open("xb") as log:
        child=subprocess.Popen(["timeout","--signal=TERM","--kill-after=60","14400"]+cmd,
            cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="4",
                    MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4"),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        save(out/"child.json",{"pid":child.pid,"command":cmd})
        rc=child.wait()
    result={"exit_code":rc,"elapsed_seconds":time.monotonic()-started,
            "end":datetime.datetime.now().astimezone().isoformat(),"test_called":False}
    if rc==0:
        s=(out/"train.log").read_text()
        assert "Test After Training:" not in s and "Best Iter(dev)=" in s
        assert (out/"best.pt").is_file()
        result.update(checkpoint_sha256=sha(out/"best.pt"),
                      best_line=re.findall(r"Best Iter\(dev\)=.*",s)[-1],
                      epochs=re.findall(r"Epoch\s+\d+\s+loss=.*",s))
        save(out/"COMPLETE.json",result)
    else:save(out/"FAILED.json",result)
    save(out/"exit.json",result)
    raise SystemExit(rc)
if __name__=="__main__":main()
