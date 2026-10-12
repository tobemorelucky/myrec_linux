"""Only CIGR-specific math/path tests, no files or dataset execution."""
import sys, unittest, argparse, copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from models.sequential.LLMMIRecASPCFSampledSoftmax import LLMMIRecASPCFSampledSoftmax as SS, masked_softmax_loss
from models.sequential.LLMMIRecASPCFCIGR import LLMMIRecASPCFCIGR as CG, route_interest_gradient
torch.set_num_threads(2)
TABLE=torch.randn(30,1536,generator=torch.Generator().manual_seed(8));TABLE[0]=0
def make(beta=.5,K=4,base=False):
 a=CG.parse_model_args(argparse.ArgumentParser()).parse_args([])
 a.device=torch.device("cpu");a.model_path="unused";a.buffer=0;a.llm_emb_path="synthetic";a.cigr_beta=beta;a.K=K
 c=SimpleNamespace(n_users=4,n_items=30,train_clicked_set={1:{1,2,6},2:{3,4,7},3:{5,8}})
 torch.manual_seed(42)
 with patch("models.sequential.LLMMIRecASPCF.load_llm_table",return_value=TABLE):
  return (SS if base else CG)(a,c)
def feed():
 return dict(user_id=torch.tensor([1,2,3]),history_items=torch.tensor([[1,2,0],[3,4,0],[5,0,0]]),
             lengths=torch.tensor([2,2,1]),item_id=torch.tensor([[6,20],[7,21],[8,22]]))
def math_case(beta=.5,K=4):
 g=torch.Generator().manual_seed(7)
 v=torch.randn(3,K,6,generator=g,dtype=torch.float64,requires_grad=True)
 w=torch.softmax(torch.randn(3,K,generator=g,dtype=torch.float64),1).detach().requires_grad_()
 e=torch.randn(3,6,generator=g,dtype=torch.float64,requires_grad=True)
 s=(v*w[:,:,None]).sum(1)@e.T
 mask=torch.tensor([[1,1,0],[0,1,1],[1,0,1]],dtype=torch.bool)
 r,pi,q=route_interest_gradient(s,v,w,e,mask,beta)
 return v,w,e,s,mask,r,pi,q
class Tests(unittest.TestCase):
 def test_beta_zero_all_params_loss_rng_exact(self):
  a,b=make(base=True),make(0)
  for k,x in a.state_dict().items():self.assertTrue(torch.equal(x,b.state_dict()[k]),k)
  a.train();b.train()
  torch.manual_seed(33);oa=a(feed(),True);ra=torch.get_rng_state();la=a.loss(oa);la.backward()
  torch.manual_seed(33);ob=b(feed(),True);rb=torch.get_rng_state();lb=b.loss(ob);lb.backward()
  self.assertTrue(torch.equal(ra,rb));self.assertTrue(torch.equal(la,lb))
  self.assertTrue(torch.equal(oa["prediction"],ob["prediction"]))
  for (k,p),(_,z) in zip(a.named_parameters(),b.named_parameters()):
   self.assertEqual(p.grad is None,z.grad is None,k)
   if p.grad is not None:self.assertTrue(torch.equal(p.grad,z.grad),k)
 def test_surrogate_analytic_gradient_and_direct_isolation(self):
  v,w,e,s,mask,r,pi,q=math_case()
  self.assertTrue(torch.equal(s,r))
  self.assertFalse(pi.requires_grad);self.assertFalse(q.requires_grad)
  self.assertTrue(torch.allclose(pi.sum(1),torch.ones_like(s)))
  self.assertTrue((q>=0).all());self.assertTrue(torch.allclose(q.sum(1),torch.ones_like(s)))
  l,_=masked_softmax_loss(s,mask,1);lr,_=masked_softmax_loss(r,mask,1)
  gs=torch.autograd.grad(l,s,retain_graph=True)[0]
  baseline=torch.autograd.grad(l,(v,w,e),retain_graph=True)
  routed=torch.autograd.grad(lr,(v,w,e),retain_graph=True)
  expected=torch.einsum("bj,bkj,jd->bkd",gs,q,e.detach())
  self.assertTrue(torch.allclose(routed[0],expected,atol=1e-12))
  for k in [1,2]:self.assertTrue(torch.equal(baseline[k],routed[k]))
  self.assertGreater(float((baseline[0]-routed[0]).abs().max()),1e-5)
  self.assertTrue(torch.allclose(baseline[0].sum(1),routed[0].sum(1),atol=1e-12))
  self.assertTrue(torch.equal(gs[~mask],torch.zeros_like(gs[~mask])))
  self.assertTrue((gs.diagonal()<0).all()) # positive and allowed negative both routed
 def test_permutation_and_extreme_mask(self):
  v,w,e,s,mask,r,pi,q=math_case();perm=torch.tensor([2,0,3,1])
  vp=v.detach()[:,perm].requires_grad_();wp=w.detach()[:,perm].requires_grad_()
  sp=(vp*wp[:,:,None]).sum(1)@e.detach().T
  rp,pip,_=route_interest_gradient(sp,vp,wp,e.detach(),mask,.5)
  self.assertTrue(torch.allclose(rp,r,atol=1e-12))
  self.assertTrue(torch.allclose(pip,pi[:,perm],atol=1e-12))
  l,_=masked_softmax_loss(r,mask,1);lp,_=masked_softmax_loss(rp,mask,1)
  self.assertTrue(torch.allclose(torch.autograd.grad(l,v)[0][:,perm],torch.autograd.grad(lp,vp)[0],atol=1e-12))
  for k in [1,4]:
   vv=torch.full((2,k,4),1e3,requires_grad=True);ee=torch.full((2,4),-1e3,requires_grad=True)
   ww=torch.zeros(2,k);ww[:,0]=1
   ss=(vv*ww[:,:,None]).sum(1)@ee.T;mm=torch.eye(2,dtype=torch.bool)
   rr,pp,_=route_interest_gradient(ss,vv,ww,ee,mm,.5)
   ll,_=masked_softmax_loss(rr,mm,1);ll.backward()
   self.assertEqual(float(ll),0);self.assertTrue(torch.isfinite(vv.grad).all())
   if pp is not None:self.assertTrue(torch.isfinite(pp).all())
 def test_K1_exact_model_and_eval_isolation(self):
  for K in [1,4]:
   a,b=make(K=K,base=True),make(K=K)
   if K==1:
    a.train();b.train()
    torch.manual_seed(3);oa=a(feed(),True);la=a.loss(oa);la.backward()
    torch.manual_seed(3);ob=b(feed(),True);lb=b.loss(ob);lb.backward()
    self.assertTrue(torch.equal(la,lb))
    for p,z in zip(a.parameters(),b.parameters()):
     if p.grad is not None:self.assertTrue(torch.equal(p.grad,z.grad))
   a.eval();b.eval()
   with patch("models.sequential.LLMMIRecASPCFCIGR.route_interest_gradient",side_effect=AssertionError("eval routing")):
    oa=a(feed(),True);ob=b(feed(),True);self.assertTrue(torch.equal(oa["prediction"],ob["prediction"]))
    ff=feed();ff["item_id"]=ff["item_id"][:,[1,0]]
    self.assertTrue(torch.equal(ob["prediction"][:,[1,0]],b(ff)["prediction"]))
 def test_full_model_forward_relation_direct_grad_and_extractor(self):
  a,b=make(base=True),make()
  # Amplify only test fixture head differences, never production initialization.
  with torch.no_grad():
   a.extractor.query.mul_(100);a.extractor.Wv.weight.mul_(10)
  b.load_state_dict(a.state_dict());a.train();b.train()
  torch.manual_seed(4);oa=a(feed(),True);la=a.loss(oa)
  with patch.object(b.item_encoder,"forward",wraps=b.item_encoder.forward) as enc:
   torch.manual_seed(4);ob=b(feed(),True);self.assertEqual(enc.call_count,2)
  lb=b.loss(ob)
  self.assertTrue(torch.equal(oa["prediction"],ob["prediction"]))
  self.assertTrue(torch.equal(oa["_ss_scores"],ob["_ss_scores"]))
  self.assertTrue(torch.equal(la,lb))
  self.assertTrue(torch.equal(oa["loss_relation"],ob["loss_relation"]))
  ga=torch.autograd.grad(la,(oa["candidate_vectors"],oa["interest_weights"],a.extractor.query),retain_graph=True)
  gb=torch.autograd.grad(lb,(ob["candidate_vectors"],ob["interest_weights"],b.extractor.query),retain_graph=True)
  for k in [0,1]:self.assertTrue(torch.equal(ga[k],gb[k]))
  self.assertGreater(float((ga[2]-gb[2]).abs().max()),0)
  lb.backward()
  self.assertTrue(all(torch.isfinite(p.grad).all() for p in b.parameters() if p.grad is not None))
if __name__=="__main__":unittest.main(verbosity=2)
