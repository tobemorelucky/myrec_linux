"""CPU-only ML-1M A1 initialization/asset/gradient preflight; no full fit or Test."""
import os
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
os.environ["CUDA_VISIBLE_DEVICES"]=""
import sys,time,json,signal,random
from pathlib import Path
import numpy as np,torch
ROOT=Path("/home/yyx/hzgcode/pom2.0").resolve();OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OUT))
from train_arm import Model,read_corpus,arguments,common_hash,verify_sources,write,old
from models.sequential.LLMMIRecASPCFAdaptiveSIL import file_sha

def main():
 signal.alarm(170);torch.set_num_threads(2);start=time.perf_counter()
 verify_sources();c=read_corpus();models={};hashes={};counts={}
 for arm in ("B0","C1","M1"):
  a=arguments(arm,c);a.device=torch.device("cpu")
  random.seed(42);np.random.seed(42);torch.manual_seed(42)
  m=Model(a,c);models[arm]=m;hashes[arm]=common_hash(m);counts[arm]=m.count_variables()
  assert m.item_encoder.llm_table.shape==(3707,1536)
  assert not m.item_encoder.llm_table.requires_grad
  assert file_sha(a.llm_emb_path)=="03cf9649844da2c172487a0f552a49a4d331d8262135e3b70e48d6e772ad7905"
  assert m.sil_raw_pop_lambda==0.3842694319760107
 assert len(set(hashes.values()))==1 and counts==dict(B0=405894,C1=443398,M1=443398)
 frame=c.data_df["train"].query("position >= 4").sample(n=32,random_state=42)
 hist=np.zeros((32,20),dtype=np.int64);length=[]
 for row,(u,p) in enumerate(zip(frame.user_id,frame.position)):
  seq=c.user_his[int(u)][max(0,int(p)-20):int(p)]
  length.append(len(seq));hist[row,:len(seq)]=[x[0] for x in seq]
 f=dict(history_items=torch.from_numpy(hist),lengths=torch.tensor(length),
   user_id=torch.as_tensor(frame.user_id.to_numpy(np.int64)),
   item_id=torch.as_tensor(np.column_stack([frame.item_id.to_numpy(np.int64),np.ones(32,dtype=np.int64)])))
 before=torch.get_rng_state().clone();reference=None;post=None
 for arm,m in models.items():
  m.train();torch.set_rng_state(before)
  out=m(f,True);loss=m.loss(out);assert torch.isfinite(loss)
  grad=torch.autograd.grad(loss,m.extractor.query)[0]
  values=(out["prediction"],loss,grad)
  if reference is None:reference=[v.detach() for v in values];post=torch.get_rng_state().clone()
  else:
   assert all(torch.equal(v,r) for v,r in zip(values,reference))
   assert torch.equal(post,torch.get_rng_state())
  m.zero_grad();torch.set_rng_state(before)
  out=m(f,True);loss=m.loss(out);loss.backward()
  assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
  assert m.extractor.query.grad.norm()>0
  if arm!="B0":assert m.sil_block.output.weight.grad.norm()>0
 verify_sources()
 result=dict(seconds=time.perf_counter()-start,checks_passed=True,CPU_only=True,
  common_initial_parameter_sha256=next(iter(hashes.values())),parameters=counts,
  A1_embedding_verified=True,locked_lambda_pop=.30,frequency_scale_unchanged=True,
  zero_initial_score_loss_query_gradient_equal=True,finite_backward=True,
  Train_rows=int((c.data_df["train"].position>0).sum()),Dev_rows=len(c.data_df["dev"]),
  no_Test=True,no_full_training_started_by_check=True,checkpoint_input_load=False)
 write(OUT/"CHECK_PASS.json",result,True)
 print(json.dumps(result),flush=True)

if __name__=="__main__":main()
