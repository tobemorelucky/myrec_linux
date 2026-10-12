"""Focused tests for semantic-conditioned candidate competition only."""
import argparse,copy,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,mock_open
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from models.sequential.LLMMIRecASPCFSampledSoftmax import LLMMIRecASPCFSampledSoftmax as Base
from models.sequential.LLMMIRecASPCFSemanticCandidate import LLMMIRecASPCFSemanticCandidate as New
from models.sequential.semantic_candidate_components import semantic_responsibility,interest_competition
torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parents[1]
TABLE=torch.randn(33,1536,generator=torch.Generator().manual_seed(17));TABLE[0]=0
Q=np.zeros((33,32),dtype=np.float32);Q[1:]=np.random.default_rng(21).dirichlet(np.full(32,.1),size=32).astype(np.float32)
def make(base=False,weight=.01,k=4):
 a=New.parse_model_args(argparse.ArgumentParser()).parse_args([])
 a.device=torch.device('cpu');a.buffer=0;a.model_path='unused';a.llm_emb_path='synthetic'
 a.semantic_candidate_path=str(ROOT/'data/synthetic/proto.pkl');a.lambda_interest_candidate=weight;a.K=k
 a.relation_sample_size=5
 c=SimpleNamespace(n_items=33,n_users=4,train_clicked_set={1:{1,2,6},2:{3,4,7},3:{5,8}})
 torch.manual_seed(42)
 with patch('models.sequential.LLMMIRecASPCF.load_llm_table',return_value=TABLE),patch('models.sequential.LLMMIRecASPCFSemanticCandidate.Path.open',mock_open()),patch('models.sequential.LLMMIRecASPCFSemanticCandidate.pickle.load',return_value=dict(soft_assignments=Q,semantic_rank=512)):
  return (Base if base else New)(a,c)
def feed():
 return dict(user_id=torch.tensor([1,2,3]),history_items=torch.tensor([[1,2,0],[3,4,0],[5,0,0]]),
             lengths=torch.tensor([2,2,1]),item_id=torch.tensor([[6,20],[7,21],[8,22]]))
class Tests(unittest.TestCase):
 def test_disabled_exact_prediction_loss_grad_rng_relation(self):
  a,b=make(True),make(weight=0)
  self.assertEqual(set(a.state_dict()),set(b.state_dict()))
  for n,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[n]))
  make(True);state=torch.get_rng_state();make();self.assertTrue(torch.equal(state,torch.get_rng_state()))
  torch.manual_seed(51);oa=a(feed());torch.manual_seed(51);ob=b(feed())
  self.assertTrue(torch.equal(oa['prediction'],ob['prediction']))
  self.assertTrue(torch.equal(oa['_ss_allowed'],ob['_ss_allowed']))
  la=a.loss(oa);lb=b.loss(ob);self.assertTrue(torch.equal(la,lb))
  la.backward();lb.backward()
  for n,p in a.named_parameters():
   if p.grad is not None:self.assertTrue(torch.equal(p.grad,dict(b.named_parameters())[n].grad),n)
 def test_eval_teacher_and_batch_isolation(self):
  a,b=make(True),make();a.eval();b.eval()
  with patch('models.sequential.LLMMIRecASPCFSemanticCandidate.semantic_responsibility',side_effect=AssertionError('eval teacher')):
   oa=a(feed(),True);ob=b(feed(),True)
   f=copy.deepcopy(feed());f['item_id'][1:]=torch.tensor([[30,29],[27,28]])
   oc=b(f,True)
  self.assertTrue(torch.equal(oa['prediction'],ob['prediction']))
  self.assertTrue(torch.equal(ob['prediction'][0],oc['prediction'][0]))
  self.assertNotIn('_candidate_aux',ob)
 def test_math_reference_mask_and_gradients(self):
  B,K=3,4
  scores=torch.randn(B,K,B,dtype=torch.float64,requires_grad=True)
  r=torch.softmax(torch.randn(B,K,B,dtype=torch.float64),1)
  c=K/(K-1)*((r-1/K)**2).sum(1)
  mask=torch.tensor([[1,1,0],[1,1,1],[0,0,1]],dtype=torch.bool)
  loss,details=interest_competition(scores,mask,r,c)
  ref=scores.new_zeros([])
  for b in range(B):
   if mask[b].sum()<=1:continue
   for k in range(K):
    terms=[scores[b,k,b]]
    for j in range(B):
     if j!=b and mask[b,j]:terms.append(scores[b,k,j]+torch.log(K*r[b,k,j]*c[b,j]))
    ref+=c[b,b]*r[b,k,b]*(torch.logsumexp(torch.stack(terms),0)-scores[b,k,b])
  ref/=2
  self.assertTrue(torch.allclose(loss,ref,atol=1e-12))
  loss.backward();self.assertTrue(torch.isfinite(scores.grad).all())
  self.assertTrue(torch.equal(scores.grad[0,:,2],torch.zeros(K,dtype=torch.float64)))
  self.assertTrue(torch.equal(scores.grad[2],torch.zeros_like(scores.grad[2])))
  huge=(scores.detach()*10000).requires_grad_();l,_=interest_competition(huge,mask,r,c)
  l.backward();self.assertTrue(torch.isfinite(l));self.assertTrue(torch.isfinite(huge.grad).all())
 def test_teacher_stopgrad_padding_permutation_and_uniform(self):
  A=torch.softmax(torch.randn(3,4,4),-1).requires_grad_()
  h=torch.randn(3,4,32).softmax(-1);h[:,3]=0
  q=torch.randn(3,32).softmax(-1);lengths=torch.tensor([3,2,1])
  P,r,c=semantic_responsibility(A,h,q,lengths,.1)
  self.assertFalse(P.requires_grad);self.assertFalse(r.requires_grad);self.assertFalse(c.requires_grad)
  self.assertTrue(torch.allclose(r.sum(1),torch.ones(3,3),atol=1e-6))
  h2=h.clone();h2[:,3]=torch.randn(3,32)
  P2,r2,c2=semantic_responsibility(A,h2,q,lengths,.1)
  self.assertTrue(torch.equal(P,P2));self.assertTrue(torch.equal(r,r2))
  perm=torch.tensor([2,0,3,1]);_,rp,cp=semantic_responsibility(A[:,perm],h,q,lengths,.1)
  self.assertTrue(torch.allclose(r[:,perm],rp,atol=1e-6));self.assertTrue(torch.allclose(c,cp,atol=1e-6))
  _,uniform,zero=semantic_responsibility(A[:,:1].expand(-1,4,-1),h,q,lengths,.1)
  self.assertTrue(torch.equal(uniform,torch.full_like(uniform,.25)));self.assertTrue(torch.equal(zero,torch.zeros_like(zero)))
  s=torch.randn(3,4,3,requires_grad=True);l,_=interest_competition(s,torch.ones(3,3,dtype=torch.bool),uniform,zero)
  self.assertEqual(float(l),0);l.backward();self.assertTrue(torch.equal(s.grad,torch.zeros_like(s)))
 def test_full_forward_aux_gradient_and_encoder_reuse(self):
  m=make()
  with torch.no_grad():
   m.extractor.query.normal_(0,1);m.extractor.Wq.weight.normal_(0,1);m.extractor.Wk.weight.normal_(0,1)
  with patch.object(m.item_encoder,'forward',wraps=m.item_encoder.forward) as enc:o=m(feed(),True)
  self.assertEqual(enc.call_count,2)
  self.assertFalse(o['candidate_responsibility'].requires_grad)
  self.assertTrue(torch.isfinite(o['_candidate_aux']))
  g=torch.autograd.grad(o['_candidate_aux'],m.extractor.query,retain_graph=True)[0]
  self.assertGreater(float(g.abs().sum()),0)
  l=m.loss(o);l.backward();self.assertTrue(torch.isfinite(l))
  for p in m.parameters():
   if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all())
  m._flush_candidate()
 def test_k1_exact_and_history_prefix_no_future(self):
  a,b=make(True,k=1),make(k=1)
  torch.manual_seed(72);oa=a(feed());torch.manual_seed(72);ob=b(feed())
  self.assertTrue(torch.equal(oa['prediction'],ob['prediction']))
  self.assertTrue(torch.equal(a.loss(oa),b.loss(ob)))
  self.assertNotIn('_candidate_aux',ob)
  corpus=SimpleNamespace(data_df={'train':pd.DataFrame(dict(user_id=[1],item_id=[6],time=[3],position=[2]))},
                         user_his={1:[(1,1),(2,2),(6,3),(9,4)]})
  d=b.Dataset(b,corpus,'train');d.data['neg_items']=np.array([[20]])
  f=d._get_feed_dict(0)
  self.assertEqual(f['history_items'].tolist(),[1,2]);self.assertEqual(f['item_id'][0],6)
if __name__=='__main__':unittest.main(verbosity=2)
