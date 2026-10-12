"""Focused tests for loss-only in-batch ASPCF; no dataset writes."""
import argparse, copy, sys, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF as C0
from models.sequential.LLMMIRecASPCFSampledSoftmax import (
    LLMMIRecASPCFSampledSoftmax as SS, build_train_membership, candidate_mask, masked_softmax_loss)
torch.set_num_threads(2)
TABLE=torch.randn(130,1536,generator=torch.Generator().manual_seed(17));TABLE[0]=0
def make(base=False, interests=4):
    args=SS.parse_model_args(argparse.ArgumentParser()).parse_args([])
    args.device=torch.device("cpu");args.model_path="unused";args.buffer=0
    args.llm_emb_path="synthetic";args.relation_sample_size=5;args.K=interests
    c=SimpleNamespace(n_users=5,n_items=130,
                      train_clicked_set={1:{1,2,6,7,63,64,127},2:{3,4,7},3:{5,8}},
                      residual_clicked_set={1:{8}},clicked_set={1:{8}})
    torch.manual_seed(42)
    with patch("models.sequential.LLMMIRecASPCF.load_llm_table",return_value=TABLE):
        return (C0 if base else SS)(args,c)
def feed():
    return dict(user_id=torch.tensor([1,2,3]),history_items=torch.tensor([[1,2,0],[3,4,0],[5,0,0]]),
                lengths=torch.tensor([2,2,1]),item_id=torch.tensor([[6,20],[7,21],[8,22]]))
class Tests(unittest.TestCase):
    def test_bitset_filter_train_only_duplicates_padding_positive(self):
        m=make()
        ids=torch.tensor([6,7,8,7,0,63,64,127])
        users=torch.tensor([1,2,3,2,4,1,1,1])
        allowed,dup,clicked=candidate_mask(m.train_membership,users,ids)
        self.assertTrue(allowed.diagonal().all())
        self.assertFalse(allowed[1,3]);self.assertTrue(dup[1,3])
        self.assertFalse(allowed[0,1]);self.assertTrue(clicked[0,1])
        self.assertTrue(allowed[0,2]) # dev/test-only ID8 must not be filtered
        self.assertFalse(allowed[0,4])
        for j in [5,6,7]:self.assertFalse(allowed[0,j]) # signed bit63 and next word
        for i,u in enumerate(users.tolist()):
            known={1:{1,2,6,7,63,64,127},2:{3,4,7},3:{5,8}}.get(u,set())
            for j,item in enumerate(ids.tolist()):
                expected=(i==j) or (item>0 and item!=ids[i].item() and item not in known)
                self.assertEqual(bool(allowed[i,j]),expected)
    def test_stable_math_gradient_and_invalid_rows(self):
        scores=torch.tensor([[1000.,1001.,-1000.],[3.,2.,1.],[-10.,8.,2.]],requires_grad=True)
        mask=torch.tensor([[1,1,0],[0,1,0],[1,1,1]],dtype=torch.bool)
        loss,valid=masked_softmax_loss(scores,mask,1.)
        ref=(torch.nn.functional.softplus(scores[0,1]-scores[0,0])
             +torch.logsumexp(scores[2],0)-scores[2,2])/2
        self.assertTrue(torch.allclose(loss,ref))
        self.assertEqual(valid.tolist(),[True,False,True])
        loss.backward();self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertEqual(float(scores.grad[1].abs().sum()),0)
        for n in [1,3]:
            s=torch.randn(n,n,requires_grad=True)
            l,v=masked_softmax_loss(s,torch.eye(n,dtype=torch.bool),1.)
            self.assertEqual(float(l),0);self.assertFalse(v.any())
            l.backward();self.assertTrue(torch.equal(s.grad,torch.zeros_like(s)))
    def test_initial_parameters_rng_and_eval_identical(self):
        a,b=make(True),make()
        self.assertEqual(a.count_variables(),b.count_variables())
        for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[k]),k)
        self.assertEqual(set(a.state_dict()),set(b.state_dict()))
        make(True);ra=torch.get_rng_state();make();self.assertTrue(torch.equal(ra,torch.get_rng_state()))
        a.eval();b.eval()
        with patch("models.sequential.LLMMIRecASPCFSampledSoftmax.candidate_mask",side_effect=AssertionError("eval sampler")):
            oa=a(feed(),True);ob=b(feed(),True)
            self.assertTrue(torch.equal(oa["prediction"],ob["prediction"]))
            f=copy.deepcopy(feed());f["item_id"]=f["item_id"][:,[1,0]]
            oc=b(f,True)
        self.assertTrue(torch.equal(ob["prediction"][:,[1,0]],oc["prediction"]))
        self.assertTrue(torch.equal(ob["user_vector"],oc["user_vector"]))
        self.assertNotIn("_ss_scores",ob)
    def test_train_reuse_rng_relation_and_shared_gradients(self):
        a,b=make(True),make()
        torch.manual_seed(91);oa=a(feed(),True);ra=torch.get_rng_state()
        with patch.object(b.item_encoder,"forward",wraps=b.item_encoder.forward) as enc:
            torch.manual_seed(91);ob=b(feed(),True);self.assertEqual(enc.call_count,2)
        self.assertTrue(torch.equal(ra,torch.get_rng_state()))
        self.assertTrue(torch.equal(oa["prediction"],ob["prediction"]))
        self.assertTrue(torch.equal(oa["_relation_ids"],ob["_relation_ids"]))
        self.assertTrue(torch.equal(ob["_ss_scores"],ob["user_vector"]@ob["candidate_vectors"][:,0].T))
        la=a.loss(oa);lb=b.loss(ob)
        self.assertTrue(torch.equal(oa["loss_relation"],ob["loss_relation"]))
        lb.backward();self.assertTrue(torch.isfinite(lb))
        for p in b.parameters():
            if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all())
        for module in [b.item_encoder,b.extractor,b.aggregator]:
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None),0)
    def test_batch_permutation_and_full_invalid(self):
        b=make();b.eval();o=b(feed(),True)
        s=o["user_vector"]@o["candidate_vectors"][:,0].T
        mask,_,_=candidate_mask(b.train_membership,feed()["user_id"],feed()["item_id"][:,0])
        l,_=masked_softmax_loss(s,mask,1.);perm=torch.tensor([2,0,1])
        lp,_=masked_softmax_loss(s[perm][:,perm],mask[perm][:,perm],1.)
        self.assertTrue(torch.allclose(l,lp,atol=1e-7))
        b.train();f=feed();f["item_id"][:,0]=6
        out=b(f);loss=b.loss(out);loss.backward()
        self.assertEqual(float(out["loss_ranking"]),0)
        self.assertTrue(torch.isfinite(loss))
if __name__=="__main__":unittest.main(verbosity=2)
