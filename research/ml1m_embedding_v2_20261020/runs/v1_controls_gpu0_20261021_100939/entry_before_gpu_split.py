"""Isolated path binding, bounded Train smoke and original-main Dev-only queue."""
import argparse,datetime,gc,hashlib,importlib,json,os,pickle,re,runpy,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path("/home/yyx/hzgcode/pom2.0")
WORK=ROOT/"research/ml1m_embedding_v2_20261020"
ASSET=WORK/"imported_assets/v1_rebuilt_20261020"
PYTHON="/home/yyx/anaconda3/envs/hzg_py10/bin/python"
EXPECTED="5546240cc16beff4666a74a18cb758f41dbf23f21f46a9082778193501c59aad"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()
def save(p,obj):
 p=Path(p).resolve()
 assert p.is_relative_to(WORK)
 with p.open("x") as f:json.dump(obj,f,indent=2,allow_nan=False)
def asset():
 m=json.loads((ASSET/"P920_MANIFEST.json").read_text());p=ASSET/"base_pca.pkl"
 assert p.resolve()==p and sha(p)==m["pkl_sha256"]==EXPECTED
 assert sha(ASSET/"item2id.json")==sha(ROOT/"data/ml-1m/item2id.json")==m["mapping_sha256"]
 return p,m
def bind(job):
 p,m=asset();assert Path(job["config"]["llm_emb_path"]).resolve()==p
 sys.path.insert(0,str(ROOT))
 if job["label"]=="B":
  mod=importlib.import_module("models.sequential.PoMRecLLMEmbLinear")
  def explicit(dataset):
   assert dataset=="ml-1m";assert sha(p)==EXPECTED
   return str(p)
  mod._find_llm_emb_path=explicit
  original=mod._safe_load_emb_table
  def checked(path,expected_rows,name="embedding"):
   assert Path(path).resolve()==p and sha(p)==EXPECTED
   value=original(path,expected_rows,name)
   receipt=Path(job["output_dir"])/("EMBEDDING_"+os.environ.get("V1_ENTRY_PHASE","UNKNOWN")+".json")
   save(receipt,dict(path=str(p),sha256=EXPECTED,shape=list(value.shape),dtype=str(value.dtype),old_autodiscovery_disabled=True))
   return value
  mod._safe_load_emb_table=checked
 else:
  name="LLMMIRecASPCF" if job["label"] in ("A","R") else "LLMMIRecCAISD"
  mod=importlib.import_module("models.sequential."+name);original=mod.load_llm_table
  def checked(path,expected_rows):
   assert Path(path).resolve()==p and sha(p)==EXPECTED
   value=original(path,expected_rows)
   receipt=Path(job["output_dir"])/("EMBEDDING_"+os.environ.get("V1_ENTRY_PHASE","UNKNOWN")+".json")
   save(receipt,dict(path=str(p),sha256=EXPECTED,shape=list(value.shape),dtype=str(value.dtype)))
   return value
  mod.load_llm_table=checked
def flags(config):
 result=[]
 for key,value in config.items():result+=["--"+key,str(value)]
 return result
def build(job):
 import argparse,torch
 import main as original_main
 from helpers.SeqReader import SeqReader
 from helpers.BaseRunner import BaseRunner
 cls=getattr(importlib.import_module("models.sequential."+job["config"]["model_name"]),job["config"]["model_name"])
 p=argparse.ArgumentParser();p=original_main.parse_global_args(p);p=SeqReader.parse_data_args(p);p=BaseRunner.parse_runner_args(p);p=cls.parse_model_args(p)
 args,extra=p.parse_known_args(flags(job["config"]))
 assert extra in ([],["--model_name",job["config"]["model_name"]]),extra
 args.device=torch.device("cuda")
 from utils import utils
 utils.init_seed(args.random_seed)
 with (ROOT/"data/ml-1m/SeqReader.pkl").open("rb") as f:corpus=pickle.load(f)
 model=cls(args,corpus).to(args.device)
 table=model.llm_table if job["label"]=="B" else model.item_encoder.llm_table
 with (ASSET/"base_pca.pkl").open("rb") as f:expected=pickle.load(f)
 import numpy as np
 assert expected.dtype==np.float32 and expected.shape==(3707,1536)
 assert np.isfinite(expected).all() and not expected[0].any()
 assert torch.equal(table.detach().cpu(),torch.from_numpy(expected))
 assert sorted(json.loads((ASSET/"item2id.json").read_text()).values())==list(range(1,3707))
 if job["label"]=="C":
  teacher=Path(args.semantic_teacher_path)
  provenance=json.loads(teacher.with_suffix(".json").read_text())
  assert provenance["embedding_sha256"]==EXPECTED and provenance["teacher_sha256"]==sha(teacher)
  with teacher.open("rb") as f:t=pickle.load(f)
  assert torch.equal(model.t_semantic_assign.detach().cpu(),torch.from_numpy(t["soft_assignments"]))
 return model,corpus,args,cls
def smoke(job):
 import numpy as np,torch
 from utils import utils
 model,corpus,args,cls=build(job);model.train()
 ds=cls.Dataset(model,corpus,"train")
 n=max(args.batch_size,args.eval_batch_size);indices=np.random.RandomState(42).choice(len(ds),n,replace=False)
 ds.data={key:np.asarray(value)[indices] for key,value in ds.data.items()}
 ds.actions_before_epoch()
 feeds=[ds[i] for i in range(n)]
 feed=utils.batch_to_gpu(ds.collate_batch(feeds[:args.batch_size]),model.device)
 assert feed["history_items"].shape[1]<=20 and (feed["item_id"]>0).all()
 torch.cuda.reset_peak_memory_stats()
 if job["label"]=="A":
  from models.sequential.LLMMIRec import LLMMIRec
  plain=LLMMIRec(args,corpus).to(model.device)
  plain.load_state_dict(model.state_dict(),strict=True)
  plain.eval();model.eval()
  with torch.no_grad():torch.testing.assert_close(plain(feed)["prediction"],model(feed)["prediction"],rtol=0,atol=0)
  del plain;model.train()
 times=[];losses=[]
 for step in range(3):
  model.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.monotonic()
  out=model(feed);loss=model.loss(out);assert torch.isfinite(loss)
  loss.backward()
  grads=[p.grad for p in model.parameters() if p.grad is not None]
  assert grads and all(torch.isfinite(g).all() for g in grads)
  torch.cuda.synchronize();times.append(time.monotonic()-start);losses.append(float(loss.detach()))
  if job["label"]=="A":
   from models.sequential.LLMMIRecASPCFSampledSoftmax import masked_softmax_loss
   expected,_=masked_softmax_loss(out["_ss_scores"],out["_ss_allowed"],1.)
   torch.testing.assert_close(loss,expected,rtol=0,atol=0)
   assert "_relation_ids" not in out
  if job["label"]=="C":assert "loss_relation" in out and "loss_interest_semantic" in out
 params=model.count_variables();effective=sum(p.numel() for p in model.parameters() if p.grad is not None)
 model.zero_grad(set_to_none=True);del out,loss,grads
 # Full evaluation batch size, but only legal Train prefixes and artificial
 # repeated legal Train-negative slots. No Dev/Test samples are read here.
 eval_feed=utils.batch_to_gpu(ds.collate_batch(feeds[:args.eval_batch_size]),model.device)
 eval_feed["item_id"]=torch.cat([eval_feed["item_id"][:,:1],eval_feed["item_id"][:,1:2].repeat(1,1000)],1)
 eval_feed["phase"]="dev";model.eval()
 with torch.no_grad():prediction=model(eval_feed)["prediction"]
 assert prediction.shape==(args.eval_batch_size,1001) and torch.isfinite(prediction).all()
 torch.testing.assert_close(prediction[:,1:],prediction[:,1:2].expand(-1,1000),rtol=0,atol=0)
 result=dict(passed=True,model=job["name"],embedding_path=str(ASSET/"base_pca.pkl"),embedding_sha256=EXPECTED,
  parameter_count=params,parameters_with_task_gradient=effective,loss=job["loss"],train_batch=args.batch_size,
  forward_backward_seconds=times,loss_samples=losses,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
  peak_reserved_bytes=torch.cuda.max_memory_reserved(),evaluation_shape=list(prediction.shape),
  duplicate_slots_retained=True,train_prefix_only=True,optimizer_steps=0,
  A_forward_exact_original_LLMMIRec=job["label"]=="A",no_Test=True)
 save(Path(job["output_dir"])/"SMOKE_COMPLETE.json",result)
 print(json.dumps(result,indent=2))
def dev(job):
 from helpers.BaseRunner import BaseRunner
 model,corpus,args,cls=build(job);model.load_model();model.eval()
 ds=cls.Dataset(model,corpus,"dev");ds.prepare()
 runner=BaseRunner(args);metrics=runner.evaluate(ds,[5,10,20],["HR","NDCG"])
 save(Path(job["output_dir"])/"BEST_DEV_FULL.json",dict(metrics=metrics,checkpoint_sha256=sha(args.model_path),parameters=model.count_variables(),embedding_sha256=EXPECTED,phase="Dev",test_called=False))
 print(json.dumps(metrics))
def main():
 p=argparse.ArgumentParser();p.add_argument("--job",required=True);p.add_argument("--mode",choices=["smoke","train","dev"],required=True)
 a=p.parse_args();path=Path(a.job).resolve();assert path.is_relative_to(WORK)
 job=json.loads(path.read_text());os.chdir(ROOT);os.environ["CUDA_VISIBLE_DEVICES"]="0";os.environ["V1_ENTRY_PHASE"]=a.mode.upper()
 bind(job)
 if a.mode=="smoke":smoke(job)
 elif a.mode=="dev":dev(job)
 else:
  assert job["config"]["dev_only"]==1 and job["config"]["test_epoch"]==-1 and job["config"]["train"]==1 and job["config"]["load"]==0
  assert json.loads((Path(job["output_dir"])/"SMOKE_COMPLETE.json").read_text())["passed"]
  sys.argv=["main.py"]+flags(job["config"])
  runpy.run_path(str(ROOT/"main.py"),run_name="__main__")
if __name__=="__main__":
 try:main()
 except BaseException:
  traceback.print_exc();raise
