"""Focused pointwise control tests; no data/asset writes."""
import sys,unittest,argparse,copy
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from tools.test_llmmirec_tia import make,feed,TABLE
from models.sequential.LLMMIRecASPCFPositionMLP import LLMMIRecASPCFPositionMLP as MLP,PositionwiseAdapter
from models.sequential.LLMMIRecASPCFTIA import TemporalInteractionAdapter,LLMMIRecASPCFTIA as TIA
def build():
 args=MLP.parse_model_args(argparse.ArgumentParser()).parse_args([])
 args.device=torch.device("cpu");args.model_path="unused";args.buffer=0;args.llm_emb_path="synthetic"
 c=SimpleNamespace(n_users=4,n_items=60,train_clicked_set={1:{1,2,6},2:{3,4,7},3:{5,8}})
 torch.manual_seed(42)
 with patch("models.sequential.LLMMIRecASPCF.load_llm_table",return_value=TABLE):return MLP(args,c)
class Tests(unittest.TestCase):
 def test_parameters_initial_service_rng_mask_relation(self):
  b=make(base=True);r=torch.get_rng_state();m=build()
  self.assertTrue(torch.equal(r,torch.get_rng_state()))
  for k,v in b.state_dict().items():self.assertTrue(torch.equal(v,m.state_dict()[k]),k)
  self.assertEqual(m.count_variables()-b.count_variables(),28776)
  self.assertEqual(sum(p.numel() for p in m.extractor.tia.parameters()),28776)
  b.train();m.train()
  torch.manual_seed(3);ob=b(feed(),True);lb=b.loss(ob)
  torch.manual_seed(3);om=m(feed(),True);lm=m.loss(om)
  for key in ["prediction","_ss_scores","_ss_allowed","_relation_ids","candidate_vectors"]:
   self.assertTrue(torch.equal(ob[key],om[key]),key)
  self.assertTrue(torch.equal(lb,lm));self.assertTrue(torch.equal(ob["loss_relation"],om["loss_relation"]))
  m.actions_after_train()
 def test_pointwise_vs_neighbor_jacobian_and_boundaries(self):
  for cls,is_local in [(PositionwiseAdapter,True),(TemporalInteractionAdapter,False)]:
   torch.manual_seed(7);a=cls(4)
   with torch.no_grad():a.output.weight.normal_(std=.2);a.output.bias.fill_(.3)
   x=torch.randn(2,5,4,requires_grad=True);lens=torch.tensor([4,1]);y=a(x,lens)
   j=torch.autograd.grad(y[0,2].sum(),x)[0]
   self.assertGreater(float(j[0,2].abs().sum()),0)
   if is_local:self.assertEqual(float(j[0,1].abs().sum()),0)
   else:self.assertGreater(float(j[0,1].abs().sum()),0)
   self.assertEqual(float(j[0,3:].abs().sum()),0)
   self.assertTrue(torch.equal(y[:,0],x[:,0]))
   self.assertTrue(torch.equal(y[1,1:],torch.zeros_like(y[1,1:])))
   xx=x.detach().clone();xx[1,1:]=float("nan");yy=a(xx,lens)
   self.assertTrue(torch.isfinite(yy).all());self.assertTrue(torch.equal(y[1,0],yy[1,0]))
 def test_all_parameters_active_after_projection_update(self):
  a=PositionwiseAdapter(64);opt=torch.optim.SGD(a.parameters(),lr=.1)
  x=torch.randn(4,6,64);lens=torch.tensor([6,5,4,3]);target=torch.randn_like(x)
  for step in range(2):
   opt.zero_grad();l=(a(x,lens)-target).square().sum();l.backward()
   self.assertTrue(torch.isfinite(l));self.assertGreater(float(a.output.weight.grad.abs().sum()),0)
   if step==1:
    for name,p in a.named_parameters():
     self.assertTrue(torch.isfinite(p.grad).all(),name)
     self.assertTrue((p.grad!=0).all(),name)
   opt.step()
 def test_same_dataset_H20_future_and_eval(self):
  t=make();m=build();self.assertIs(m.Dataset,TIA.Dataset)
  d=object.__new__(m.Dataset);d.model=m;d.phase="train"
  d.corpus=SimpleNamespace(user_his={1:[(i+1,i) for i in range(50)]},n_items=60)
  d.data={"user_id":[1],"item_id":[36],"neg_items":[np.array([55])],"position":[35]}
  dt=object.__new__(t.Dataset);dt.model=t;dt.corpus=d.corpus;dt.phase=d.phase;dt.data=d.data
  f=d._get_feed_dict(0);ft=dt._get_feed_dict(0)
  self.assertTrue(np.array_equal(f["history_items"],ft["history_items"]))
  self.assertEqual(f["history_items"].tolist(),list(range(16,36)))
  self.assertEqual(f["lengths"],20);self.assertNotIn(36,f["history_items"])
  m.eval();ff=feed()
  with torch.no_grad():
   a=m(ff,True);ff["item_id"]=ff["item_id"][:,[1,0]];b=m(ff,True)
  self.assertTrue(torch.equal(a["prediction"][:,[1,0]],b["prediction"]))
  self.assertNotIn("_ss_scores",a);self.assertTrue(torch.isfinite(a["prediction"]).all())
if __name__=="__main__":unittest.main(verbosity=2)
