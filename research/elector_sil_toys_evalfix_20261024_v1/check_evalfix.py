"""Synthetic evaluator regression; no real Dev/Test or optimizer."""
import ast,types,json,time,importlib.util,sys,os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
ROOT=Path('/home/yyx/hzgcode/pom2.0').resolve()
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from utils import utils

def load_function(path):
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in ("score_views","metrics_from_ranks")]
    ns=dict(torch=torch,np=np,DataLoader=DataLoader,Path=Path,csv=__import__("csv"),
        hashlib=__import__("hashlib"),KEYS=["HR@5","NDCG@5","HR@10","NDCG@10","HR@20","NDCG@20"],
        OUT=OUT,read=lambda p:json.loads(Path(p).read_text()),utils=utils)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),ns)
    return ns["score_views"]
class Fake(torch.nn.Module):
    def __init__(self,device):super().__init__();self.device=device
    def forward(self,batch,*a):
        # Intentionally many ties: exact original NumPy ordering must survive.
        raw=torch.arange(1001,device=self.device).remainder(7).float()[None].repeat(3,1)
        return dict(prediction=raw,raw_prediction=raw)
class Toy(Dataset):
    def __init__(self,device):self.model=Fake(device)
    def __len__(self):return 3
    def __getitem__(self,i):return dict(user_id=torch.tensor(i+1),history_items=torch.tensor([2,3,0]),
        lengths=torch.tensor(2),item_id=torch.arange(1,1002))
    def collate_batch(self,rows):return {k:torch.stack([r[k] for r in rows]) for k in rows[0]}
def main():
    torch.set_num_threads(2);start=time.perf_counter()
    old=load_function(ROOT/"research/elector_sil_toys_parallel_20261024_v1/common.py")
    fixed=load_function(OUT/"common.py")
    refpath=OUT/"synthetic_cpu_ranks.csv";newpath=OUT/"synthetic_fixed_cpu_ranks.csv"
    reference=old(Toy("cpu"),refpath);cpu=fixed(Toy("cpu"),newpath)
    assert reference==cpu and refpath.read_bytes()==newpath.read_bytes()
    result=dict(passed=True,CPU_exact=True,synthetic_only=True,original_files_untouched=True,
        six_metrics_exact=True,input_hash_exact=True,tied_order_and_duplicate_slots_preserved=True)
    path=ROOT/"research/elector_sil_objective_full_20261023_v2/scheduler.py"
    spec=importlib.util.spec_from_file_location("evalfix_gpu_lease",path);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    inv=r.observe();gpu=next((g for g in [1,0] if g in inv and r.idle(inv[g])),None)
    if gpu is not None:
        fd=r.acquire(ROOT/("research/gpu_resource_scheduler_20261022_v1/gpu%d.lock"%gpu))
        if fd is not None:
            try:
                if r.idle(r.observe()[gpu]):
                    gp=OUT/"synthetic_fixed_gpu_ranks.csv"
                    actual=fixed(Toy("cuda:"+str(gpu)),gp)
                    assert actual==reference and gp.read_bytes()==refpath.read_bytes()
                    result.update(GPU_exact=True,physical_GPU=gpu,frequency_and_rank_export_exercised=True)
            finally:os.close(fd)
    result["seconds"]=time.perf_counter()-start
    with (OUT/"EVAL_FIX_CHECK.json").open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=="__main__":main()
