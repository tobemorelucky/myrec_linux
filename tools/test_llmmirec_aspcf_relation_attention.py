"""Focused CPU synthetic tests; in-memory assets only, no dataset/cache/checkpoint writes."""
import argparse
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF
from models.sequential.LLMMIRecASPCFRelationAttention import (
    LLMMIRecASPCFRelationAttention, relation_difference)

torch.set_num_threads(1)
MODULE = "models.sequential.LLMMIRecASPCFRelationAttention"


class RelationAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = torch.randn(33, 1536, generator=torch.Generator().manual_seed(17))
        cls.table[0] = 0
        cls.corpus = SimpleNamespace(n_items=33, n_users=8)

    def model(self, original=False, eta=0.0):
        args = LLMMIRecASPCFRelationAttention.parse_model_args(argparse.ArgumentParser()).parse_args([])
        args.device, args.model_path = torch.device("cpu"), "unused-no-write.pt"
        args.llm_emb_path, args.history_max = "synthetic://memory", 20
        args.num_neg, args.relation_sample_size, args.relation_eta_init = 2, 5, eta
        torch.manual_seed(42)
        with patch("models.sequential.LLMMIRecASPCF.load_llm_table", return_value=self.table):
            return (LLMMIRecASPCF if original else LLMMIRecASPCFRelationAttention)(args, self.corpus)

    @staticmethod
    def feed():
        return dict(history_items=torch.tensor([[1,2,3,4,0],[5,6,0,0,0],[7,8,9,0,0]]),
                    lengths=torch.tensor([4,2,3]),
                    item_id=torch.tensor([[10,11,12],[13,14,15],[16,17,18]]))

    def exact(self, x, y):
        self.assertTrue(torch.equal(x, y), "Expected torch.equal")

    def compare_baseline(self, original, new):
        for training in (False, True):
            original.train(training); new.train(training)
            original.zero_grad(); new.zero_grad()
            torch.manual_seed(73)
            a = original(self.feed(), True); la = original.loss(a)
            torch.manual_seed(73)
            b = new(self.feed(), True); lb = new.loss(b)
            for key in ("prediction","attention_maps","interest_vectors","interest_weights","user_vector"):
                self.exact(a[key], b[key])
            self.exact(la, lb)
            if training:
                self.exact(a["_relation_ids"], b["_relation_ids"])
                self.exact(a["loss_relation"], b["loss_relation"])
            la.backward(); lb.backward()
            new_params = dict(new.named_parameters())
            for name, p in original.named_parameters():
                self.exact(p.grad, new_params[name].grad)

    def test_eta_zero_strict_baseline_init_rng_predictions_loss_gradients(self):
        original = self.model(True)
        rng = torch.get_rng_state()
        new = self.model()
        self.exact(rng, torch.get_rng_state())
        old = original.state_dict()
        for name, value in old.items():
            self.exact(value, new.state_dict()[name])
        self.assertEqual(set(new.state_dict()) - set(old), {"extractor.eta_raw"})
        self.assertEqual(new.count_variables(), original.count_variables()+1)
        self.compare_baseline(original, new)
        self.assertTrue(torch.isfinite(new.extractor.eta_raw.grad))
        self.assertGreater(new.extractor.eta_raw.grad.abs().item(), 0)

    def test_d_zero_strict_baseline_with_nonzero_eta(self):
        original, new = self.model(True), self.model(eta=0.05)
        def zero_difference(s, c, valid):
            d, edges = relation_difference(s, c, valid)
            return d * 0, edges
        with patch(MODULE + ".relation_difference", side_effect=zero_difference):
            self.compare_baseline(original, new)
        # Actual identical branch relations (not just an injected zero tensor).
        h = torch.randn(3,5,64)
        s = torch.randn(3,5,32)
        v = torch.arange(5)[None] < self.feed()["lengths"][:,None]
        d, _ = relation_difference(s, s, v)
        self.exact(d, torch.zeros_like(d))
        a, aa = original.extractor(h, self.feed()["lengths"])
        b, ba = new.extractor(h, self.feed()["lengths"], s, s)
        self.exact(a,b); self.exact(aa,ba)

    def test_pairwise_math_diagonal_and_effective_mass_normalization(self):
        m = self.model(eta=0.05).double().eval()
        s = torch.tensor([[[1.,0.],[0.,1.],[1.,1.],[9.,9.]]],dtype=torch.float64)
        c = torch.tensor([[[1.,0.],[1.,0.],[-1.,0.],[3.,5.]]],dtype=torch.float64)
        lengths = torch.tensor([3]); valid = torch.arange(4)[None] < lengths[:,None]
        h = torch.randn(1,4,64,dtype=torch.float64)
        _, attn, details = m.extractor(h,lengths,s,c,True)
        d, edges = relation_difference(s,c,valid)
        ref = torch.zeros_like(d)
        for i in range(3):
            for j in range(3):
                if i != j:
                    ref[0,i,j] = (torch.dot(s[0,i],s[0,j])/(s[0,i].norm()*s[0,j].norm())
                                  -torch.dot(c[0,i],c[0,j])/(c[0,i].norm()*c[0,j].norm()))
        torch.testing.assert_close(d,ref,atol=1e-12,rtol=1e-12)
        a0 = details["baseline_attention"]
        e = torch.zeros_like(a0)
        for k in range(4):
            for j in range(3):
                den = sum(a0[0,k,i] for i in range(3) if i != j)
                e[0,k,j] = sum(a0[0,k,i]*ref[0,i,j] for i in range(3) if i != j)/den
        torch.testing.assert_close(details["relation_evidence"],e,atol=1e-12,rtol=1e-12)
        self.exact(d.diagonal(dim1=-2,dim2=-1),torch.zeros(1,4,dtype=d.dtype))
        self.assertLessEqual(details["attention_correction"].abs().max().item(),.2+1e-12)

    def test_padding_contents_and_extension_do_not_change_valid_result(self):
        m = self.model(eta=0.05).eval()
        h,s,c = torch.randn(3,5,64),torch.randn(3,5,32),torch.randn(3,5,32)
        lengths=self.feed()["lengths"]; valid=torch.arange(5)[None]<lengths[:,None]
        v,a = m.extractor(h,lengths,s,c)
        h2,s2,c2 = h.clone(),s.clone(),c.clone()
        h2[~valid]=300; s2[~valid]=-400; c2[~valid]=500
        vv,aa = m.extractor(h2,lengths,s2,c2)
        self.exact(v,vv); self.exact(a,aa)
        ext = lambda x: torch.cat([x,torch.randn(x.shape[0],2,x.shape[2])*100],1)
        vv,aa = m.extractor(ext(h),lengths,ext(s),ext(c))
        torch.testing.assert_close(v,vv,atol=1e-7,rtol=1e-6)
        torch.testing.assert_close(a,aa[...,:5],atol=1e-7,rtol=1e-6)
        self.exact(aa[...,5:],torch.zeros_like(aa[...,5:]))
        feed=self.feed(); out=m(feed,True)
        longer=dict(feed,history_items=torch.cat([feed["history_items"],torch.zeros(3,2,dtype=torch.long)],1))
        torch.testing.assert_close(out["prediction"],m(longer)["prediction"],atol=1e-7,rtol=1e-6)

    def test_zero_norm_singleton_and_empty_attention_finite(self):
        m=self.model(eta=.05)
        h=torch.randn(3,4,64,requires_grad=True)
        s=torch.zeros(3,4,32,requires_grad=True); c=torch.zeros_like(s,requires_grad=True)
        v,a,d=m.extractor(h,torch.tensor([0,1,3]),s,c,True)
        self.assertTrue(torch.isfinite(v).all())
        self.exact(d["relation_evidence"],torch.zeros_like(a))
        self.exact(a[0],torch.zeros_like(a[0]))
        v.square().sum().backward()
        for x in (h,s,c):
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_attention_weights_and_signed_eta_bound(self):
        m=self.model(eta=.05).eval(); feed=self.feed()
        for raw in (-100.,0.,100.):
            with torch.no_grad(): m.extractor.eta_raw.fill_(raw)
            out=m(feed,True); a=out["attention_maps"]
            valid=torch.arange(5)[None]<feed["lengths"][:,None]
            self.assertTrue((a>=0).all())
            self.exact(a.masked_select(~valid[:,None,:]),torch.zeros_like(a.masked_select(~valid[:,None,:])))
            torch.testing.assert_close(a.sum(-1),torch.ones(3,4))
            self.assertLessEqual(abs(out["eta"].item()),.100001)
            self.assertLessEqual(out["attention_correction"].abs().max().item(),.200001)

    def test_forward_backward_finite_and_nonzero_mechanism_gradient(self):
        for eta in (0.,.05):
            m=self.model(eta=eta).train()
            out=m(self.feed(),True); loss=m.loss(out)
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(out["prediction"]).all())
            loss.backward()
            for name,p in m.named_parameters():
                self.assertIsNotNone(p.grad,name)
                self.assertTrue(torch.isfinite(p.grad).all(),name)
            self.assertGreater(m.extractor.eta_raw.grad.abs().item(),0)

    def test_candidate_permutation_and_set_do_not_change_history_routing(self):
        m=self.model(eta=.05).eval(); feed=self.feed()
        a=m(feed,True)
        perm=torch.tensor([2,0,1])
        b=m(dict(feed,item_id=feed["item_id"][:,perm]),True)
        self.exact(a["user_vector"],b["user_vector"])
        self.exact(a["attention_maps"],b["attention_maps"])
        self.exact(a["prediction"][:,perm],b["prediction"])
        extra=dict(feed,item_id=torch.cat([feed["item_id"],torch.tensor([[20],[21],[22]])],1))
        c=m(extra,True)
        self.exact(a["user_vector"],c["user_vector"])
        self.exact(a["prediction"],c["prediction"][:,:3])

    def test_relation_loss_is_original_for_same_ids_with_active_mechanism(self):
        original,new=self.model(True),self.model(eta=.05)
        ids=torch.tensor([1,4,7,9,12])
        self.exact(original._compute_relation_loss(ids),new._compute_relation_loss(ids))
        self.assertIs(type(new).loss,LLMMIRecASPCF.loss)
        self.assertIs(type(new)._compute_relation_loss,LLMMIRecASPCF._compute_relation_loss)
        torch.manual_seed(19); a=original(self.feed())
        torch.manual_seed(19); b=new(self.feed())
        self.exact(a["_relation_ids"],b["_relation_ids"])
        la=original.loss(a); lb=new.loss(b)
        self.exact(a["loss_relation"],b["loss_relation"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
