"""PoMRec loss-only controlled comparison tests; no asset writes."""
import sys,unittest,argparse,copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from models.sequential.PoMRec import PoMRec
from models.sequential.PoMRecBatchObjective import PoMRecBatchObjective as Control
from models.sequential.LLMMIRecASPCFSampledSoftmax import candidate_mask
torch.set_num_threads(2)
def make(mode="softmax",base=False):
 a=Control.parse_model_args(argparse.ArgumentParser()).parse_args([])
 a.device=torch.device("cpu");a.model_path="unused";a.buffer=0;a.K=4;a.prompt_num=3;a.n_layers=2;a.lamb=4.;a.history_max=20;a.ranking_objective=mode;a.use_llmemb=0
 c=SimpleNamespace(n_users=5,n_items=50,train_clicked_set={1:{1,2,6,7},2:{3,4,7},3:{5,8},4:{9}},residual_clicked_set={1:{8}})
 np.random.seed(42);torch.manual_seed(42)
 return (PoMRec if base else Control)(a,c)
def feed():
 return dict(user_id=torch.tensor([1,2,3,2]),history_items=torch.tensor([[1,2,0],[3,4,0],[5,0,0],[3,0,0]]),
             lengths=torch.tensor([2,2,1,1]),item_id=torch.tensor([[6,20],[7,21],[8,22],[7,23]]))
class Tests(unittest.TestCase):
 def test_BPR_identical_parameters_prediction_loss_gradient_rng(self):
  a,b=make(base=True),make("bpr")
  self.assertEqual(set(a.state_dict()),set(b.state_dict()))
  for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[k]),k)
  self.assertEqual(a.count_variables(),b.count_variables())
  a.train();b.train()
  torch.manual_seed(4);oa=a(feed());la=a.loss(oa);ra=torch.get_rng_state();la.backward()
  torch.manual_seed(4);ob=b(feed());lb=b.loss(ob);rb=torch.get_rng_state();lb.backward()
  self.assertTrue(torch.equal(ra,rb));self.assertTrue(torch.equal(oa["prediction"],ob["prediction"]));self.assertTrue(torch.equal(la,lb))
  for p,z in zip(a.parameters(),b.parameters()):
   self.assertEqual(p.grad is None,z.grad is None)
   if p.grad is not None:self.assertTrue(torch.equal(p.grad,z.grad))
 def test_eval_original_no_batch_dependence_full_candidate_permutation(self):
  a,b=make(base=True),make();a.eval();b.eval();f=feed()
  with torch.no_grad(),patch("models.sequential.PoMRecBatchObjective.candidate_mask",side_effect=AssertionError("eval mask")):
   x=a(f);y=b(f);self.assertTrue(torch.equal(x["prediction"],y["prediction"]))
   ff=copy.deepcopy(f);ff["item_id"]=ff["item_id"][:,[1,0]]
   self.assertTrue(torch.equal(y["prediction"][:,[1,0]],b(ff)["prediction"]))
   single={k:v[:1] for k,v in f.items()}
   # Original value2attn global scalar shift may introduce rounding across batches.
   self.assertTrue(torch.allclose(y["prediction"][:1],b(single)["prediction"],atol=1e-7,rtol=1e-6))
  self.assertNotIn("_batch_scores",y)
 def test_train_single_extractor_reuse_diagonal_exact_mask_finite(self):
  a=make(base=True);m=make();a.train();m.train()
  torch.manual_seed(6);original=a(feed());ra=torch.get_rng_state()
  torch.manual_seed(6);controlled=m(feed());rb=torch.get_rng_state()
  self.assertTrue(torch.equal(original["prediction"],controlled["prediction"]))
  self.assertTrue(torch.equal(ra,rb))
  with patch.object(m.interest_extractor,"forward",wraps=m.interest_extractor.forward) as ext,patch.object(m.interest_extractor.i_embeddings,"forward",wraps=m.interest_extractor.i_embeddings.forward) as enc:
   out=m(feed());self.assertEqual(ext.call_count,1);self.assertEqual(enc.call_count,2)
  self.assertEqual(out["_batch_scores"].shape,(4,4))
  self.assertTrue(torch.allclose(out["_batch_scores"].diagonal(),out["prediction"][:,0],atol=1e-7))
  ref,_,_=candidate_mask(m.train_membership,feed()["user_id"],feed()["item_id"][:,0])
  self.assertTrue(torch.equal(ref,out["_batch_allowed"]))
  self.assertTrue(ref.diagonal().all());self.assertFalse(ref[1,3]);self.assertFalse(ref[0,1])
  self.assertTrue(ref[0,2]) # ID8 held-out for user1 is not train-filtered
  loss=m.loss(out);loss.backward();self.assertTrue(torch.isfinite(loss))
  self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
  for module in [m.interest_extractor.W1,m.interest_extractor.i_embeddings,m.proj]:
   self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None),0)
 def test_aux_structure_prompts_and_all_invalid(self):
  m=make();self.assertEqual(m.interest_extractor.prompt_pad.shape,(2,64))
  self.assertEqual(m.interest_extractor.prompt1.num_embeddings,3)
  self.assertEqual(m.interest_extractor.max_prompt,5);self.assertEqual(m.lamb,4.)
  m.train();f=feed();f["item_id"][:,0]=6
  out=m(f);l=m.loss(out);self.assertEqual(float(l),0);l.backward()
  self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
if __name__=="__main__":unittest.main(verbosity=2)
