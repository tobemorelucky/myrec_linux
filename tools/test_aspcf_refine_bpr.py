"""Bounded real-Train equivalence tests; no Dev/Test dataset or evaluation."""
import os,sys,time,pickle,copy,io,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch,numpy as np
from tools.run_refine_bpr_round1 import arguments,rng,restore,rh,digest,fixture,sha,write_new
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF as Base
from models.sequential.LLMMIRecASPCFRefineBPR import LLMMIRecASPCFRefineBPR as Model
from models.sequential.LLMMIRecASPCFDualViewRefine import LLMMIRecASPCFDualViewRefine as OldFull
from models.sequential.LLMMIRecASPCFRefineControls import LLMMIRecASPCFUserRefine as OldA
from models.sequential.LLMMIRecASPCFSampledSoftmax import LLMMIRecASPCFSampledSoftmax as SS
from models.BaseModel import GeneralModel
from helpers.BaseRunner import BaseRunner
from utils import utils
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument("--output",required=True);a=p.parse_args()
 out=(ROOT/a.output).resolve();assert out.is_relative_to(ROOT) and not out.exists()
 gpu=os.popen("nvidia-smi --id=1 --query-gpu=uuid,memory.free --format=csv,noheader,nounits").read().strip().split(",")
 apps=os.popen("nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader").read()
 assert gpu[0].strip() not in apps and float(gpu[1])>7000,"GPU1 occupied"
 torch.set_num_threads(2);start=time.perf_counter();result=dict(success=False,datasets={},dev_test_executed=False)
 for ds in ["beauty","ml-1m"]:
  args,_=arguments(ds,"user",ROOT/"UNUSED_NO_LOG",ROOT/"UNUSED_NO_WEIGHT")
  args.device=torch.device("cuda")
  with (ROOT/"data"/ds/"SeqReader.pkl").open("rb") as f:c=pickle.load(f)
  models=[];states=[]
  for mode in ["base","user","full"]:
   utils.init_seed(42);args.refine_mode=mode if mode!="base" else "user"
   m=(Base(args,c) if mode=="base" else Model(args,c)).to(args.device)
   models.append(m);states.append(rh())
  base,user,full=models
  assert states[0]==states[1]==states[2]
  assert digest(base,True)==digest(user,True)==digest(full,True)
  assert Model.loss is Base.loss and Model.Dataset is Base.Dataset
  assert not issubclass(Model,SS)
  for m in [user,full]:
   assert not hasattr(m,"train_membership") and not hasattr(m,"sampled_softmax_temperature")
   assert all(not n.startswith("_ss") for n,_ in m.named_parameters())
  assert user.count_variables()-base.count_variables()==28776
  assert full.count_variables()-base.count_variables()==28864
  b=fixture(c,args.device,n=16)
  # Same original train batch, two dropout draws and relation subset.
  saved=rng();outs=[];grads=[];losses=[];ranks=[];rel=[];endrng=[];calls=[]
  for m in models:
   restore(saved);m.train();m.zero_grad();count={"item":0,"query":0,"dropout":0}
   def cnt(k):
    def f(*_):count[k]+=1
    return f
   hooks=[m.item_encoder.register_forward_hook(cnt("item")),
          m.extractor.register_forward_hook(cnt("query")),
          m.dropout.register_forward_hook(cnt("dropout"))]
   o=m(b,True);rank=GeneralModel.loss(m,o);loss=m.loss(o);loss.backward()
   for h in hooks:h.remove()
   assert count==dict(item=2,query=1,dropout=2)
   calls.append(count);outs.append(o);losses.append(float(loss));ranks.append(float(rank));rel.append(float(o["loss_relation"]))
   grads.append({n:p.grad.detach().clone() if p.grad is not None else None for n,p in m.named_parameters() if not n.startswith("refiner.")})
   endrng.append(rh())
   assert all(not key.startswith("_ss") for key in o)
  assert endrng[0]==endrng[1]==endrng[2]
  assert torch.equal(outs[0]["prediction"],outs[1]["prediction"]) and torch.equal(outs[0]["prediction"],outs[2]["prediction"])
  assert torch.equal(outs[0]["_relation_ids"],outs[1]["_relation_ids"]) and torch.equal(outs[0]["_relation_ids"],outs[2]["_relation_ids"])
  assert losses[0]==losses[1]==losses[2] and ranks[0]==ranks[1]==ranks[2] and rel[0]==rel[1]==rel[2]
  maxerr=0.
  for n,g in grads[0].items():
   for z in grads[1:]:
    assert (g is None)==(z[n] is None)
    if g is not None:
     maxerr=max(maxerr,float((g-z[n]).abs().max()))
     assert torch.equal(g,z[n]),"Shared initial gradient differs: "+n
  # No unrequested extra ItemEncoder call; raw S/C share exact same attention.
  o=outs[2];valid=(torch.arange(20,device=args.device)[None]<b["lengths"][:,None])&(b["history_items"]!=0)
  att=o["attention_maps"].masked_fill(~valid[:,None],0)
  assert torch.equal(o["interest_semantic"],torch.bmm(att,o["history_semantic"].masked_fill(~valid[:,:,None],0)))
  assert torch.equal(o["interest_complement"],torch.bmm(att,o["history_complement"].masked_fill(~valid[:,:,None],0)))
  # Two real Train updates: output zero init must not permanently block input.
  update=[]
  for m in [user,full]:
   opt=BaseRunner(args)._build_optimizer(m);inner=[]
   for step in [1,2]:
    restore(saved);opt.zero_grad();o=m(b);loss=m.loss(o);loss.backward()
    inner.append(dict(step=step,input_task_gradient=float(m.refiner.input.weight.grad.norm()),
                     output_task_gradient=float(m.refiner.output.weight.grad.norm()),loss=float(loss)))
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    opt.step()
   assert inner[0]["output_task_gradient"]>0 and inner[1]["input_task_gradient"]>0
   update.append(inner)
  # Exact old structure/feature parity even after a nonzero residual exists.
  legacy=[]
  for mode,m,Old in [("user",user,OldA),("full",full,OldFull)]:
   args.refine_mode=mode;args.sampled_softmax_temperature=1.
   old=Old(args,c).to(args.device);old.load_state_dict(m.state_dict(),strict=True)
   old.eval();m.eval();left=old(b,True);right=m(b,True)
   keys=["prediction","user_vector","refinement_features"]
   if mode=="full":keys+=["interest_semantic","interest_complement","refined_interest_vectors","interest_residual"]
   assert all(torch.equal(left[k],right[k]) for k in keys)
   assert old.count_variables()==m.count_variables()
   legacy.append(dict(mode=mode,exact_nonzero_residual_structure_parity=True,parameters=m.count_variables()))
   del old
  # Padding/zero-length/short history and K compatibility; zero history represented
  # by all-padding slots, exactly the original Reader convention.
  edge=[]
  for K in [1,2,4]:
   args.K=K
   for mode in ["user","full"]:
    args.refine_mode=mode;utils.init_seed(42);m=Model(args,c).to(args.device);m.eval()
    q={k:v[:4].clone() for k,v in b.items()};q["lengths"]=torch.tensor([0,1,2,20],device=args.device)
    for i,l in enumerate([0,1,2,20]):
     q["history_items"][i,:l]=b["item_id"][0,0];q["history_items"][i,l:]=0
    z=m(q,True);assert z["interest_vectors"].shape==(4,K,64)
    assert torch.isfinite(z["prediction"]).all()
    assert torch.count_nonzero(z["refiner_output"])==0 if "refiner_output" in z else True
    # 1001 repeated slots: no dedup or different candidate scoring.
    many={k:v.clone() for k,v in q.items()}
    many["item_id"]=q["item_id"][:,:1].expand(-1,1001).clone()
    r=m(many)["prediction"];assert r.shape==(4,1001) and torch.equal(r,r[:,:1].expand_as(r))
    # Candidate or another user's history cannot alter first user's history state.
    other={k:v.clone() for k,v in q.items()};other["item_id"].flip(1)
    other["item_id"]=other["item_id"].flip(1)
    assert torch.equal(z["interest_vectors"],m(other,True)["interest_vectors"])
    other["history_items"][1]=0;other["lengths"][1]=0
    assert torch.equal(z["prediction"][0],m(other)["prediction"][0])
    # In-memory state save/load only; no checkpoint asset.
    mem=io.BytesIO();torch.save(m.state_dict(),mem);mem.seek(0);m.load_state_dict(torch.load(mem,weights_only=True))
    assert torch.equal(z["prediction"],m(q)["prediction"])
    edge.append(dict(K=K,mode=mode,finite=True,candidate_slots=1001))
    del m
  args.K=4
  # Warm GPU cost uses actual B1024 Train prefixes and original BPR+relation.
  times={}
  for mode in ["user","full"]:
   args.refine_mode=mode;utils.init_seed(42);m=Model(args,c).to(args.device);m.train()
   d=Model.Dataset(m,c,"train");d.actions_before_epoch()
   batch=d.collate_batch([d[i] for i in range(1024)])
   batch=utils.batch_to_gpu(batch,args.device)
   opt=BaseRunner(args)._build_optimizer(m)
   torch.cuda.reset_peak_memory_stats()
   for i in range(5):
    if i==2:torch.cuda.synchronize();t=time.perf_counter()
    opt.zero_grad();o=m(batch);loss=m.loss(o);assert torch.isfinite(loss);loss.backward();opt.step()
   torch.cuda.synchronize();times[mode]=dict(step_seconds=(time.perf_counter()-t)/3,
      peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,batch=1024)
   del m,opt,d,batch
  result["datasets"][ds]=dict(shared_state_exact=True,constructor_RNG_exact=True,
    first_train_forward_loss_relation_gradients_exact=True,max_shared_gradient_error=maxerr,
    dropout_relation_RNG_exact=True,encoding_call_counts=calls,losses=losses,BPR=ranks,relation=rel,
    raw_view_same_attention=True,legacy_nonzero_parity=legacy,two_step_gradients=update,
    edges=edge,timings=times,parameters=dict(B0=base.count_variables(),B1=user.count_variables(),B2=full.count_variables()))
  print("DATASET_PASS "+json.dumps(result["datasets"][ds]),flush=True)
  del models,base,user,full,c,outs,grads,b
 result.update(success=True,seconds=time.perf_counter()-start,
    source_sha256={f:sha(ROOT/f) for f in ["models/sequential/LLMMIRecASPCFRefineBPR.py","tools/run_refine_bpr_round1.py","tools/test_aspcf_refine_bpr.py"]})
 write_new(out,result);print("ALL_PASS "+str(out),flush=True)
if __name__=="__main__":main()

