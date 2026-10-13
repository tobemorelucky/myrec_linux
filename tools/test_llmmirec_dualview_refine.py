"""Focused dual-view interest-refinement tests; synthetic tensors, no asset writes."""
import argparse, copy, sys, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from models.sequential.LLMMIRecASPCFSampledSoftmax import LLMMIRecASPCFSampledSoftmax as SS
from models.sequential.LLMMIRecASPCFDualViewRefine import LLMMIRecASPCFDualViewRefine as DV, DualViewInterestRefiner
torch.set_num_threads(2)
TABLE=torch.randn(60,1536,generator=torch.Generator().manual_seed(17));TABLE[0]=0
def make(base=False,K=4):
    a=DV.parse_model_args(argparse.ArgumentParser()).parse_args([])
    a.device=torch.device("cpu");a.model_path="unused";a.buffer=0
    a.llm_emb_path="synthetic";a.K=K;a.history_max=20;a.relation_sample_size=5
    c=SimpleNamespace(n_users=4,n_items=60,train_clicked_set={1:{1,2,6},2:{3,4,7},3:{5,8}})
    torch.manual_seed(42)
    with patch("models.sequential.LLMMIRecASPCF.load_llm_table",return_value=TABLE):
        return (SS if base else DV)(a,c)
def feed():
    return dict(user_id=torch.tensor([1,2,3]),
                history_items=torch.tensor([[1,2,0],[3,4,0],[5,0,0]]),
                lengths=torch.tensor([2,2,1]),item_id=torch.tensor([[6,20],[7,21],[8,22]]))
class Tests(unittest.TestCase):
    def test_initial_identity_rng_loss_relation_base_gradients(self):
        b=make(True);r=torch.get_rng_state();m=make()
        self.assertTrue(torch.equal(r,torch.get_rng_state()))
        self.assertEqual(m.count_variables()-b.count_variables(),28864)
        for k,v in b.state_dict().items():self.assertTrue(torch.equal(v,m.state_dict()[k]),k)
        for train in [False,True]:
            b.train(train);m.train(train)
            torch.manual_seed(7);ob=b(feed(),True)
            torch.manual_seed(7);om=m(feed(),True)
            keys=["prediction","candidate_vectors","attention_maps","interest_vectors","interest_weights"]
            if train:keys+=["_ss_scores","_ss_allowed","_relation_ids"]
            for k in keys:self.assertTrue(torch.equal(ob[k],om[k]),k)
            if train:
                lb=b.loss(ob);lm=m.loss(om)
                self.assertTrue(torch.equal(lb,lm))
                self.assertTrue(torch.equal(ob["loss_relation"],om["loss_relation"]))
                lb.backward();lm.backward()
                for n,p in b.named_parameters():
                    self.assertTrue(torch.equal(p.grad,dict(m.named_parameters())[n].grad),n)
        m.actions_after_train();b.actions_after_train()

    def test_views_same_attention_ID_mask_encoder_reuse(self):
        m=make();m.eval()
        with patch.object(m.item_encoder,"forward",wraps=m.item_encoder.forward) as enc:
            o=m(feed(),True)
        self.assertEqual(enc.call_count,2)
        h=m.item_encoder(feed()["history_items"],True)
        for name in ["semantic","complement"]:
            self.assertTrue(torch.equal(o["history_"+name],h[name]))
            pooled=torch.bmm(o["attention_maps"],h[name])
            self.assertTrue(torch.equal(o["interest_"+name],pooled))
        self.assertEqual(tuple(o["refinement_features"].shape),(3,4,192))
        oo={k:v.clone() if torch.is_tensor(v) else v for k,v in o.items()}
        for key in ["history_semantic","history_complement","history_vectors"]:
            valid=torch.arange(3)[None,:]<feed()["lengths"][:,None]
            oo[key].masked_fill_(~valid[:,:,None],float("nan"))
        u=m._refine(oo,feed()["history_items"],feed()["lengths"])[0]
        self.assertTrue(torch.equal(u,o["user_vector"]));self.assertTrue(torch.isfinite(u).all())

    def test_two_updates_all_refinement_parameters_active(self):
        m=make();m.train();opt=torch.optim.SGD(m.parameters(),lr=.1)
        for step in range(2):
            opt.zero_grad();o=m(feed());loss=m.loss(o);loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
            self.assertGreater(float(m.refiner.output.weight.grad.abs().sum()),0)
            if step==0:self.assertEqual(float(m.refiner.input.weight.grad.abs().sum()),0)
            else:
                for n,p in m.refiner.named_parameters():
                    self.assertGreater(float(p.grad.abs().sum()),0,n)
            opt.step()
        self.assertGreater(float(m.refiner.output.weight.abs().sum()),0)
        m.actions_after_train()

    def test_nonlinearity_shared_interest_and_permutation(self):
        torch.manual_seed(5);f=DualViewInterestRefiner()
        with torch.no_grad():f.output.weight.normal_(std=.1)
        x=torch.randn(3,4,192,requires_grad=True);w=torch.softmax(torch.randn(3,4),-1)
        y=f(x);g=torch.autograd.grad(y[0,1].sum(),x)[0]
        for start,end in [(0,64),(64,96),(96,128),(128,192)]:
            self.assertGreater(float(g[0,1,start:end].abs().sum()),0)
        self.assertEqual(float(g[0,[0,2,3]].abs().sum()),0)
        p=torch.tensor([3,1,0,2])
        self.assertTrue(torch.equal(f(x[:,p]),y[:,p]))
        self.assertTrue(torch.allclose((y*w[:,:,None]).sum(1), (f(x[:,p])*w[:,p,None]).sum(1)))
        # A nonlinear per-interest map cannot generally commute with weighted pooling.
        self.assertGreater(float(((y*w[:,:,None]).sum(1)-f((x*w[:,:,None]).sum(1))).abs().max()),1e-6)

    def test_K1_length1_full_candidates_eval_isolation(self):
        for K in [1,4]:
            m=make(K=K);m.eval();f=feed()
            f["history_items"]=f["history_items"][:,:1];f["lengths"]=torch.ones(3,dtype=torch.long)
            f["item_id"]=torch.arange(1001)[None,:].expand(3,-1)%59+1
            with torch.no_grad():
                o=m(f,True)
                ff=copy.deepcopy(f);ff["item_id"]=ff["item_id"].flip(1)
                oo=m(ff,True)
                single=m({k:v[:1] for k,v in f.items()},True)
            self.assertEqual(tuple(o["prediction"].shape),(3,1001))
            self.assertTrue(torch.equal(o["prediction"].flip(1),oo["prediction"]))
            self.assertTrue(torch.allclose(o["prediction"][:1],single["prediction"],atol=1e-7,rtol=1e-5))
            self.assertNotIn("_ss_scores",o);self.assertNotIn("_relation_ids",o)
            self.assertTrue(torch.equal(o["user_vector"],oo["user_vector"]))
            self.assertTrue(torch.isfinite(o["prediction"]).all())

    def test_original_mask_duplicates_relation_with_active_refinement(self):
        b=make(True);m=make()
        with torch.no_grad():m.refiner.output.weight.normal_(std=.01)
        f=feed();f["item_id"][2,0]=7
        torch.manual_seed(4);ob=b(f,True);b.loss(ob)
        torch.manual_seed(4);om=m(f,True);m.loss(om)
        for k in ["_relation_ids","_ss_allowed","loss_relation"]:
            self.assertTrue(torch.equal(ob[k],om[k]),k)
        self.assertTrue(om["_ss_allowed"].diagonal().all())
        self.assertFalse(om["_ss_allowed"][1,2])
        m.actions_after_train();b.actions_after_train()

    def test_same_H20_dataset_excludes_future(self):
        m=make();b=make(True)
        data=dict(user_id=[1],item_id=[36],neg_items=[np.array([55])],position=[35])
        corpus=SimpleNamespace(user_his={1:[(i+1,i) for i in range(50)]},n_items=60)
        fs=[]
        for model in [b,m]:
            d=object.__new__(model.Dataset);d.model=model;d.phase="train";d.data=data;d.corpus=corpus
            fs.append(d._get_feed_dict(0))
        self.assertTrue(np.array_equal(fs[0]["history_items"],fs[1]["history_items"]))
        self.assertEqual(fs[1]["history_items"].tolist(),list(range(16,36)))
        self.assertEqual(fs[1]["lengths"],20);self.assertNotIn(36,fs[1]["history_items"])
if __name__=="__main__":unittest.main(verbosity=2)
