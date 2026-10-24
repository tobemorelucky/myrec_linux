"""Explicit Chapter5 Train/Dev entry; never constructs or evaluates Test."""
import sys,argparse,json,time,logging,os,atexit,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch,pandas as pd
from models.sequential.GICTRec import GICTRec
from models.sequential.LLMMIRecASPCFAdaptiveSIL import file_sha
from helpers.BaseRunner import BaseRunner
from utils import utils
from gict_data import train_corpus,args_from_config


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--init-c1",required=True)
    p.add_argument("--bank",required=True);p.add_argument("--variant",choices=["G0","G1","G2","G3"],required=True)
    p.add_argument("--output-dir",required=True);p.add_argument("--gpu",type=int,default=0)
    x=p.parse_args();root=Path(__file__).resolve().parents[1];out=Path(x.output_dir).resolve()
    if not out.is_relative_to(root) or not out.parent.is_dir() or out.exists():
        raise ValueError("Choose a new project output directory; existing output is protected")
    payload=torch.load(x.bank,map_location="cpu",weights_only=True)
    if payload["metadata"]["scope"]!="full_train":raise ValueError("Sample cache cannot train")
    if payload["metadata"]["teacher_checkpoint_sha256"]!=file_sha(x.init_c1):
        raise ValueError("All arms must share the same C1 teacher/student initialization")
    a=args_from_config(x.config);a.gict_variant=x.variant;a.gict_neighbors=8;a.gict_temperature=.1
    if torch.cuda.is_available():
        spec=importlib.util.spec_from_file_location("gict_training_gpu_lease",root/"research/elector_sil_objective_full_20261023_v2/scheduler.py")
        resource=importlib.util.module_from_spec(spec);spec.loader.exec_module(resource)
        inventory=resource.observe()
        if x.gpu not in inventory or not resource.idle(inventory[x.gpu]):
            raise RuntimeError("Requested GPU is occupied; no training started")
        lease=resource.acquire(root/("research/gpu_resource_scheduler_20261022_v1/gpu%d.lock"%x.gpu))
        if lease is None:raise RuntimeError("GPU lease is held; no training started")
        atexit.register(os.close,lease)
        if not resource.idle(resource.observe()[x.gpu]):raise RuntimeError("GPU became occupied")
        a.device=torch.device("cuda:"+str(x.gpu))
    else:a.device=torch.device("cpu")
    c=train_corpus(a)
    # Dev only, using exactly the historical leave-last-two protocol/prefixes.
    dev=pd.read_csv(Path(a.path)/a.dataset/"dev.csv",sep="\t")
    dev=utils.eval_list_columns(dev);counts=c.data_df["train"].groupby("user_id").size()
    dev["position"]=dev.user_id.map(counts)
    if dev.position.isna().any():raise ValueError("Unknown Dev user")
    c.data_df["dev"]=dev
    utils.init_seed(a.random_seed);m=GICTRec(a,c)
    m.load_c1(torch.load(x.init_c1,map_location="cpu",weights_only=True))
    m.attach_bank(payload);m=m.to(a.device)
    out.mkdir();a.model_path=str(out/"best.pt");m.model_path=a.model_path
    logging.basicConfig(filename=out/"train.log",filemode="x",level=logging.INFO)
    data={phase:m.Dataset(m,c,phase) for phase in ["train","dev"]}
    for d in data.values():d.prepare()
    config=dict(args={k:str(v) if k=="device" else v for k,v in vars(a).items()},init_c1=x.init_c1,
        init_sha256=file_sha(x.init_c1),bank=x.bank,bank_sha256=file_sha(x.bank),
        variants="G0 C1; G1 mean-user enhancement; G2 JS-interest transfer; G3 support reliability",
        full_training=True,Test_authorized=False,cache_metadata=payload["metadata"])
    with (out/"config.json").open("x") as f:json.dump(config,f,indent=2)
    class DevRunner(BaseRunner):
        def evaluate(self,dataset,topks,metrics):
            assert dataset.phase=="dev"
            values=super().evaluate(dataset,[5,10,20],["HR","NDCG"])
            with (out/"dev_metrics.jsonl").open("a") as f:f.write(json.dumps(values)+"\n")
            return values
    runner=DevRunner(a);runner.train(data);m.actions_after_train()
    with (out/"TRAIN_DEV_COMPLETE.json").open("x") as f:json.dump(dict(Test_pending=True,best=str(out/"best.pt")),f)
if __name__=="__main__":main()
