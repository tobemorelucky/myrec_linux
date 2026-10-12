"""Only additional multi-negative BPR and K1 checks."""
import sys,unittest,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tools.test_llmmirec_aspcf_sampled_softmax import make,feed
from models.sequential.LLMMIRecASPCFBatchBPR import masked_bpr_loss,LLMMIRecASPCFBatchBPR as BPR
from models.sequential.LLMMIRecASPCFSampledSoftmax import candidate_mask,masked_softmax_loss
class Tests(unittest.TestCase):
 def test_pairwise_reference_stability_and_invalid(self):
  s=torch.tensor([[1000.,1001.,-1000.],[3.,2.,1.],[-10.,8.,2.]],requires_grad=True)
  mask=torch.tensor([[1,1,0],[0,1,0],[1,1,1]],dtype=torch.bool)
  l,v=masked_bpr_loss(s,mask)
  ref=(torch.nn.functional.softplus(s[0,1]-s[0,0])+
       (torch.nn.functional.softplus(s[2,0]-s[2,2])+torch.nn.functional.softplus(s[2,1]-s[2,2]))/2)/2
  self.assertTrue(torch.allclose(l,ref));self.assertEqual(v.tolist(),[True,False,True])
  l.backward();self.assertTrue(torch.isfinite(s.grad).all());self.assertEqual(float(s.grad[1].abs().sum()),0)
  z=torch.randn(3,3,requires_grad=True);lz,vz=masked_bpr_loss(z,torch.eye(3,dtype=torch.bool))
  self.assertEqual(float(lz),0);lz.backward();self.assertTrue(torch.equal(z.grad,torch.zeros_like(z)))
 def test_bpr_same_candidates_relation_and_forward(self):
  a,b=make(),make()
  b.__class__=BPR
  torch.manual_seed(77);oa=a(feed(),True)
  torch.manual_seed(77);ob=b(feed(),True)
  for key in ['prediction','user_vector','_ss_scores','_ss_allowed','_relation_ids']:
   self.assertTrue(torch.equal(oa[key],ob[key]),key)
  la=a.loss(oa);lb=b.loss(ob);lb.backward()
  self.assertTrue(torch.equal(oa['loss_relation'],ob['loss_relation']))
  self.assertTrue(torch.isfinite(lb))
  self.assertEqual(a.count_variables(),b.count_variables())
  for p in b.parameters():
   if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all())
 def test_k1_shapes_normalization_gradients_and_eval_isolation(self):
  m=make(interests=1);m.train();o=m(feed(),True)
  self.assertEqual(tuple(o['interest_vectors'].shape),(3,1,64))
  self.assertTrue(torch.equal(o['interest_weights'],torch.ones(3,1)))
  self.assertTrue(torch.equal(o['user_vector'],o['interest_vectors'][:,0]))
  l=m.loss(o);l.backward();self.assertTrue(torch.isfinite(l))
  for p in m.parameters():
   if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all())
  self.assertGreater(float(m.extractor.query.grad.abs().sum()),0)
  self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.item_encoder.parameters() if p.grad is not None),0)
  for p in m.aggregator.parameters():
   if p.grad is not None:self.assertEqual(float(p.grad.abs().sum()),0)
  m.eval();a=m(feed(),True);f=copy.deepcopy(feed())
  f['item_id'][1:]=torch.tensor([[100,101],[110,111]])
  b=m(f,True);self.assertTrue(torch.equal(a['prediction'][0],b['prediction'][0]))
  self.assertNotIn('_ss_scores',a)
  self.assertEqual(make().count_variables()-m.count_variables(),387)
if __name__=='__main__':unittest.main(verbosity=2)
