"""Focused TIA causality, same-window, identity and gradient tests."""
import sys,argparse,unittest,copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from models.sequential.LLMMIRecASPCFSampledSoftmax import LLMMIRecASPCFSampledSoftmax as SS
from models.sequential.LLMMIRecASPCFTIA import LLMMIRecASPCFTIA as TIA,TemporalInteractionAdapter
torch.set_num_threads(2)
TABLE=torch.randn(60,1536,generator=torch.Generator().manual_seed(17));TABLE[0]=0
def make(K=4,base=False):
 a=TIA.parse_model_args(argparse.ArgumentParser()).parse_args([])
 a.device=torch.device("cpu");a.model_path="unused";a.buffer=0;a.llm_emb_path="synthetic";a.K=K;a.history_max=20
 c=SimpleNamespace(n_users=4,n_items=60,train_clicked_set={1:{1,2,6},2:{3,4,7},3:{5,8}})
 torch.manual_seed(42)
 with patch("models.sequential.LLMMIRecASPCF.load_llm_table",return_value=TABLE):
  return (SS if base else TIA)(a,c)
def feed():
 return dict(user_id=torch.tensor([1,2,3]),history_items=torch.tensor([[1,2,0],[3,4,0],[5,0,0]]),lengths=torch.tensor([2,2,1]),item_id=torch.tensor([[6,20],[7,21],[8,22]]))
class Tests(unittest.TestCase):
 def test_initial_identity_parameters_rng_loss_relation(self):
  a=make(base=True);rng=torch.get_rng_state();b=make()
  self.assertTrue(torch.equal(rng,torch.get_rng_state()))
  for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[k]),k)
  a.train();b.train()
  torch.manual_seed(6);oa=a(feed(),True);la=a.loss(oa)
  torch.manual_seed(6);ob=b(feed(),True);lb=b.loss(ob)
  for k in ["prediction","candidate_vectors","interest_vectors","interest_weights","_ss_scores","_relation_ids"]:
   self.assertTrue(torch.equal(oa[k],ob[k]),k)
  self.assertTrue(torch.equal(la,lb));self.assertTrue(torch.equal(oa["loss_relation"],ob["loss_relation"]))
  self.assertEqual(b.count_variables()-a.count_variables(),28864)
 def test_causality_first_position_padding_and_cross_dependency(self):
  torch.manual_seed(5);a=TemporalInteractionAdapter(4)
  with torch.no_grad():a.output.weight.normal_(std=.2);a.output.bias.fill_(.3)
  x=torch.randn(2,5,4,requires_grad=True);lens=torch.tensor([4,1]);y=a(x,lens)
  self.assertTrue(torch.equal(y[:,0],x[:,0]));self.assertTrue(torch.equal(y[1,1:],torch.zeros_like(y[1,1:])))
  j=torch.autograd.grad(y[0,2].sum(),x,retain_graph=True)[0]
  self.assertGreater(float(j[0,1].abs().sum()),0)
  self.assertEqual(float(j[0,3:].abs().sum()),0)
  self.assertEqual(float(j[0,0].abs().sum()),0)
  xx=x.detach().clone();xx[0,4]=float("nan");xx[1,1:]=1e5
  yy=a(xx,lens)
  self.assertTrue(torch.equal(y[0,:4],yy[0,:4]));self.assertTrue(torch.equal(y[1,:1],yy[1,:1]))
  self.assertTrue(torch.isfinite(yy).all())
 def test_output_then_inner_gradient_after_update(self):
  a=make();a.train();opt=torch.optim.SGD(a.parameters(),lr=.1)
  losses=[]
  for step in range(2):
   opt.zero_grad();o=a(feed());l=a.loss(o);l.backward();losses.append(float(l))
   self.assertTrue(torch.isfinite(l));self.assertTrue(all(torch.isfinite(p.grad).all() for p in a.parameters() if p.grad is not None))
   self.assertGreater(float(a.extractor.tia.output.weight.grad.abs().sum()),0)
   if step==0:self.assertEqual(float(a.extractor.tia.transition.weight.grad.abs().sum()),0)
   else:self.assertGreater(float(a.extractor.tia.transition.weight.grad.abs().sum()),0)
   opt.step()
  a.actions_after_train()
 def test_K1_length1_eval_batch_isolation(self):
  a=make(K=1);a.eval();f=feed();f["history_items"]=f["history_items"][:,:1];f["lengths"]=torch.ones(3,dtype=torch.long)
  with torch.no_grad():
   o=a(f,True);ff=copy.deepcopy(f);ff["item_id"]=ff["item_id"][:,[1,0]];oo=a(ff,True)
  self.assertTrue(torch.equal(o["prediction"][:,[1,0]],oo["prediction"]))
  self.assertNotIn("_ss_scores",o);self.assertTrue(torch.isfinite(o["prediction"]).all())
 def test_dataset_exact_H20_no_old_or_future(self):
  m=make()
  class OnlyWindow:
   def __getitem__(self,s):
    self.last=s
    assert isinstance(s,slice) and s.start==15 and s.stop==35
    return [(i+1,i) for i in range(s.start,s.stop)]
  seq=OnlyWindow()
  d=object.__new__(TIA.Dataset);d.model=m;d.corpus=SimpleNamespace(user_his={1:seq},n_items=60)
  d.phase="train";d.data={"user_id":[1],"item_id":[36],"neg_items":[np.array([55])],"position":[35]}
  f=d._get_feed_dict(0)
  self.assertEqual(f["history_items"].tolist(),list(range(16,36)));self.assertEqual(f["lengths"],20)
  self.assertNotIn(36,f["history_items"])
  baseline=object.__new__(SS.Dataset);baseline.model=m;baseline.corpus=SimpleNamespace(user_his={1:[(i+1,i) for i in range(50)]},n_items=60)
  baseline.phase="train";baseline.data=d.data
  self.assertTrue(np.array_equal(f["history_items"],baseline._get_feed_dict(0)["history_items"]))
if __name__=="__main__":unittest.main(verbosity=2)
