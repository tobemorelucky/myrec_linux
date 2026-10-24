"""Authorized ML-1M A1 B0/C1/M1 full Train/Dev arms. No Test code path."""
import os
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("MKL_NUM_THREADS","4")
import sys,json,time,pickle,hashlib,logging,importlib.util,argparse,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path("/home/yyx/hzgcode/pom2.0").resolve()
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from helpers.BaseRunner import BaseRunner
from models.sequential.LLMMIRecASPCFAdaptiveSIL import file_sha
def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module
snapshot=load_module("dual_sil_snapshot",OUT/"model_snapshot.py")
Model=snapshot.LLMMIRecASPCFAdaptiveSIL
helper=ROOT/"research/frozen_readout_softmax_20261022_v1/probe.py"
assert file_sha(helper)=="fae7c3155477448075e58d94212ad637bc420d26449b3137bbcc953336e4c81d"
reuse=load_module("dual_sil_original_helpers",helper)
old=reuse.old
MODES=dict(B0="baseline",C1="fixed",M1="semantic")
ASSET=ROOT/"research/elector_sil_adaptive_20261022_v1/prep_v2/ml-1m_train_asset.json"


def write(path,value,exclusive=False):
 path=Path(path)
 assert path.parent.resolve().is_relative_to(OUT)
 text=json.dumps(value,indent=2,allow_nan=False,default=str)
 with path.open("x" if exclusive else "w") as f:f.write(text)


def verify_sources():
 initial=json.loads((OUT/"INITIAL_STATE.json").read_text())
 assert all(file_sha(ROOT/p)==sha for p,sha in initial["protected_sha256"].items())
 change=json.loads((OUT/"SOURCE_CHANGE.json").read_text())
 assert file_sha(OUT/"model_snapshot.py")==change["after"]
 return initial


def read_corpus():
 with (ROOT/"data/ml-1m/SeqReader.pkl").open("rb") as f:c=pickle.load(f)
 c.data_df={phase:c.data_df[phase] for phase in ("train","dev")}
 counts=c.data_df["train"].groupby("user_id").size()
 train,dev=c.data_df["train"],c.data_df["dev"]
 assert np.all(train.position.to_numpy()<train.user_id.map(counts).to_numpy())
 assert np.all(dev.position.to_numpy()==dev.user_id.map(counts).to_numpy())
 for u in c.user_his:c.user_his[u]=c.user_his[u][:int(counts.get(u,0))]
 assert np.stack(dev.neg_items.to_numpy()).shape==(len(dev),1000)
 return c


def arguments(arm,c):
 a=old.arguments("ml-1m",c)
 a.K=8
 a.dataset="ml-1m";a.path=str(ROOT/"data")
 a.model_path=str(OUT/"arms"/arm/"best.pt")
 a.sil_mode=MODES[arm];a.sil_asset_path=str(ASSET)
 a.sil_heads=2;a.sil_recent=3;a.sil_use_recent=1;a.sil_use_local=1
 a.random_seed=42;a.device=torch.device("cuda:0")
 a.lr=.001;a.l2=1e-6;a.optimizer="Adam";a.grad_clip=0.
 a.batch_size=1024;a.eval_batch_size=256;a.num_workers=5;a.pin_memory=1
 a.epoch=200;a.early_stop=10;a.test_epoch=-1;a.check_epoch=0
 a.metric="NDCG,HR";a.topk="5,10,20";a.buffer=0;a.test_all=0
 a.dev_only=1;a.regenerate=0;a.load=0
 assert a.K==8 and a.history_max==20 and a.emb_size==64
 assert a.lambda_relation==.01 and a.sampled_softmax_temperature==1
 return a


def common_hash(model):
 h=hashlib.sha256()
 for name,p in model.named_parameters():
  if not name.startswith("sil_block."):
   h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()


class DevOnlyRunner(BaseRunner):
 def __init__(self,args,folder):
  super().__init__(args);self.folder=folder;self.last_dev=None
 def evaluate(self,dataset,topks,metrics):
  assert dataset.phase=="dev"
  result=super().evaluate(dataset,[5,10,20],["NDCG","HR"])
  self.last_dev=dict(epoch=int(dataset.model.cur_epoch),metrics={k:float(v) for k,v in result.items()})
  with (self.folder/"dev_metrics.jsonl").open("a") as f:f.write(json.dumps(self.last_dev)+"\n")
  return result
 def fit(self,dataset,epoch=-1):
  assert dataset.phase=="train"
  value=super().fit(dataset,epoch)
  if not np.isfinite(value):raise FloatingPointError("Nonfinite full-epoch loss")
  return value


def run(arm):
 folder=OUT/"arms"/arm
 assert folder.resolve().is_relative_to(OUT) and folder.is_dir()
 assert not (folder/"best.pt").exists() and not (folder/"train.log").exists()
 logging.basicConfig(filename=folder/"train.log",filemode="x",level=logging.INFO,
                     format="%(asctime)s %(levelname)s %(message)s",force=True)
 logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
 start=time.perf_counter();initial=verify_sources()
 c=read_corpus();a=arguments(arm,c)
 old.utils.init_seed(42)
 model=Model(a,c)
 init_hash=common_hash(model)
 expected=json.loads((OUT/"CHECK_PASS.json").read_text())
 assert init_hash==expected["common_initial_parameter_sha256"]
 params=model.count_variables()
 assert params==(406410 if arm=="B0" else 443914)
 model=model.to(a.device)
 # These are the only Dataset objects. Reader/cache regeneration and Test are absent.
 data={phase:Model.Dataset(model,c,phase) for phase in ("train","dev")}
 for value in data.values():value.prepare()
 runner=DevOnlyRunner(a,folder)
 config=dict(arm=arm,args=vars(a),head=initial["head"],source_sha256=file_sha(OUT/"model_snapshot.py"),
             train_script_sha256=file_sha(__file__),train_asset_sha256=file_sha(ASSET),
             common_initial_parameter_sha256=init_hash,parameters=params,
             physical_GPU=os.environ.get("CUDA_VISIBLE_DEVICES"),Test_evaluations=0,
             ties="unchanged BaseRunner latest-on-tie checkpoint saving")
 write(folder/"config.json",config,True)
 original_save=model.save_model
 def tracked_save(model_path=None):
  original_save(model_path)
  write(folder/"best_metadata.json",dict(**runner.last_dev,checkpoint="best.pt"))
 model.save_model=tracked_save
 write(folder/"READY.json",dict(pid=os.getpid(),arm=arm,GPU=os.environ.get("CUDA_VISIBLE_DEVICES"),
                                device=torch.cuda.get_device_name(0),Train_rows=len(data["train"]),
                                Dev_rows=len(data["dev"]),parameters=params,initial_sha256=init_hash),True)
 logging.info("TRAIN_START arm=%s seed=42 K=8 lr=.001 batch=1024 max_epoch=200; Dev-only; no checkpoint load",arm)
 write(folder/"RUNNING.json",dict(pid=os.getpid(),arm=arm,train_complete=False,Dev_best_selected=False,Test_accepted=False),True)
 runner.train(data)
 model.actions_after_train()
 best=json.loads((folder/"best_metadata.json").read_text())
 # Save this arm's result only; no cross-arm summary or additional evaluation.
 write(folder/"result.json",dict(arm=arm,train_complete=True,Dev_best_selected=True,
       Test_accepted=False,Test_evaluations=0,best=best,parameters=params,seconds=time.perf_counter()-start,
       checkpoint_sha256=file_sha(folder/"best.pt"),config=config),True)
 verify_sources()
 logging.info("TRAIN_FINISHED arm=%s; Test remains unaccepted",arm)


if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--arm",choices=list(MODES),required=True)
 arg=p.parse_args()
 try:run(arg.arm)
 except Exception as exc:
  write(OUT/"arms"/arg.arm/"FAILURE.json",dict(error=repr(exc),traceback=traceback.format_exc(),
         train_complete=False,Dev_best_selected=False,Test_accepted=False),True)
  raise
