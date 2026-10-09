"""Minimal CPU tests; synthetic in-memory assets, no checkpoint/data writes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import patch, mock_open
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF as C0
from models.sequential.LLMMIRecASPCFAuxNeg import LLMMIRecASPCFAuxNeg as Aux, AuxiliaryNegativeSampler

TABLE = torch.randn(33, 1536, generator=torch.Generator().manual_seed(17))
TABLE[0] = 0
BANK = np.tile(np.resize(np.arange(1,33),100), (33,1))
BANK[0] = 0
def corpus():
    frame = pd.DataFrame(dict(user_id=[1,1,2,2], item_id=[5,6,7,8], position=[1,2,1,2],
                              neg_items=[[20],[21],[22],[23]], time=[2,3,2,3]))
    return SimpleNamespace(n_items=33,n_users=4,train_clicked_set={1:{1,2,5,6},2:{3,4,7,8}},
                           user_his={1:[(1,1),(2,2)],2:[(3,1),(4,2)]},
                           data_df={p:frame.copy() for p in ("train","dev","test")})
def make(mode="random_aux", weight=.005, baseline=False):
    args = Aux.parse_model_args(argparse.ArgumentParser()).parse_args([])
    args.device=torch.device("cpu");args.model_path="unused";args.llm_emb_path="synthetic"
    args.history_max=20;args.random_seed=42;args.batch_size=2
    args.aux_mode=mode;args.lambda_aux=weight;args.aux_bank_path="synthetic"
    args.relation_sample_size=5
    torch.manual_seed(42)
    with patch("models.sequential.LLMMIRecASPCF.load_llm_table",return_value=TABLE), \
         patch("models.sequential.LLMMIRecASPCFAuxNeg.open",mock_open(),create=True), \
         patch("models.sequential.LLMMIRecASPCFAuxNeg.pickle.load",return_value=BANK):
        model=(C0 if baseline else Aux)(args,corpus())
    if not baseline:model._aux_steps_per_epoch=2
    return model
def feed():
    return dict(history_items=torch.tensor([[1,2,0],[3,4,5]]),lengths=torch.tensor([2,3]),
                item_id=torch.tensor([[6,20],[7,21]]),
                aux_neg_items=torch.tensor([[25],[26]]),
                aux_fallback=torch.tensor([0.,1.]),aux_pool_ratio=torch.tensor([.5,0.]))
def same_forward(a,b):
    torch.manual_seed(91);oa=a(feed())
    torch.manual_seed(91);ob=b(feed())
    return oa,ob
class Tests(unittest.TestCase):
    def test_zero_weight_exact(self):
        for mode in ("random_aux","shnc"):
            a,b=make(baseline=True),make(mode,0)
            self.assertEqual(list(a.state_dict()),list(b.state_dict()))
            for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[k]))
            oa,ob=same_forward(a,b)
            self.assertTrue(torch.equal(oa["prediction"],ob["prediction"]))
            self.assertTrue(torch.equal(oa["_relation_ids"],ob["_relation_ids"]))
            la,lb=a.loss(oa),b.loss(ob);self.assertTrue(torch.equal(la,lb))
            la.backward();lb.backward()
            for p,q in zip(a.parameters(),b.parameters()):
                if p.grad is not None:self.assertTrue(torch.equal(p.grad,q.grad))
            a.eval();b.eval();oa,ob=same_forward(a,b)
            self.assertTrue(torch.equal(oa["prediction"],ob["prediction"]))

    def test_aux_math_gradient_and_no_new_parameters(self):
        a,b=make(baseline=True),make()
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        b.aux_warmup_epochs=0
        pos=torch.tensor([[.2,-.3],[.5,.1]],requires_grad=True)
        neg=torch.tensor([[.4],[-.2]],requires_grad=True)
        out=dict(prediction=pos,_aux_scores=neg,_aux_fallback=torch.zeros(2),
                 _aux_pool_ratio=torch.zeros(2))
        native=a.loss(dict(prediction=pos))
        expected=native+.005*F.softplus(neg-pos[:,:1]).mean()
        actual=b.loss(out)
        self.assertTrue(torch.equal(actual,expected))
        ga=torch.autograd.grad(actual,(pos,neg),retain_graph=True)
        ge=torch.autograd.grad(expected,(pos,neg))
        for x,y in zip(ga,ge):self.assertTrue(torch.equal(x,y))

    def test_filter_fallback_and_exhaustion(self):
        bank=np.zeros((9,100),dtype=np.int64);bank[5]=np.resize([1,2,3,4,5,0],100)
        for mode in ("random_aux","shnc"):
            s=AuxiliaryNegativeSampler(9,{1:{1,2,3}},mode,42,bank)
            for i in range(30):
                n,f,_=s.sample(1,5,[4,0],[6],1,i)
                self.assertIn(n,(7,8));self.assertEqual(f,mode=="shnc")
            with self.assertRaises(ValueError):
                s.sample(1,5,[4,7,8],[6],1,0)
        bank[5]=np.resize([7,7,8],100)
        s=AuxiliaryNegativeSampler(9,{1:{1,2,3}}, "shnc",42,bank)
        n,f,ratio=s.sample(1,5,[4],[6],1,0)
        self.assertIn(n,(7,8));self.assertFalse(f);self.assertEqual(ratio,.02)
        with self.assertRaises(ValueError):AuxiliaryNegativeSampler(9,{},"shnc",42,bank.astype(float))

    def test_global_rng_and_native_dataset_stream(self):
        for mode in ("random_aux","shnc"):
            a,b=make(baseline=True),make(mode)
            np.random.seed(51);torch.manual_seed(52)
            ns=np.random.get_state();ts=torch.get_rng_state().clone()
            for i in range(8):b.aux_sampler.sample(1,5,[1,2],[20],1,i)
            self.assertTrue(np.array_equal(ns[1],np.random.get_state()[1]))
            self.assertEqual(ns[2:],np.random.get_state()[2:])
            self.assertTrue(torch.equal(ts,torch.get_rng_state()))
            da=C0.Dataset(a,corpus(),"train");db=Aux.Dataset(b,corpus(),"train")
            np.random.seed(61);da.actions_before_epoch();next_a=np.random.random()
            np.random.seed(61);db.actions_before_epoch();next_b=np.random.random()
            self.assertTrue(np.array_equal(da.data["neg_items"],db.data["neg_items"]))
            self.assertEqual(next_a,next_b)
            torch.manual_seed(62)
            aa=list(torch.utils.data.DataLoader(da,batch_size=2,shuffle=True,collate_fn=da.collate_batch))
            sa=torch.get_rng_state().clone()
            torch.manual_seed(62)
            bb=list(torch.utils.data.DataLoader(db,batch_size=2,shuffle=True,collate_fn=db.collate_batch))
            self.assertTrue(torch.equal(sa,torch.get_rng_state()))
            for x,y in zip(aa,bb):self.assertTrue(torch.equal(x["item_id"],y["item_id"]))

    def test_eval_no_aux_or_target_path(self):
        a,b=make(baseline=True),make("shnc")
        a.eval();b.eval()
        with patch.object(b.aux_sampler,"sample",side_effect=AssertionError("eval sampled")):
            d=Aux.Dataset(b,corpus(),"dev")
            self.assertNotIn("aux_neg_items",d._get_feed_dict(0))
            oa,ob=same_forward(a,b)
            self.assertTrue(torch.equal(oa["prediction"],ob["prediction"]))
            f=feed();f["item_id"]=f["item_id"][:,[1,0]]
            reordered=b(f)
            self.assertTrue(torch.equal(ob["prediction"][:,[1,0]],reordered["prediction"]))
        self.assertEqual(b._aux_total_steps,0)

    def test_original_relation_and_service_unchanged(self):
        a,b=make(baseline=True),make()
        oa,ob=same_forward(a,b)
        self.assertTrue(torch.equal(oa["prediction"],ob["prediction"]))
        self.assertTrue(torch.equal(oa["_relation_ids"],ob["_relation_ids"]))
        self.assertFalse(any(int(i) in (25,26) for i in ob["_relation_ids"]))
        a.loss(oa);b.loss(ob)
        self.assertTrue(torch.equal(oa["loss_relation"],ob["loss_relation"]))
        self.assertIs(Aux._compute_relation_loss,C0._compute_relation_loss)

    def test_finite_backward_and_train_only_warmup(self):
        for mode in ("random_aux","shnc"):
            b=make(mode);b._cur_epoch=1
            for expected in (.0005,.001):
                out=b(feed());loss=b.loss(out)
                self.assertTrue(torch.isfinite(loss));self.assertAlmostEqual(out["aux_weight"],expected)
                loss.backward()
                grads=[p.grad for p in b.parameters() if p.grad is not None]
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertTrue(any(g.abs().sum()>0 for g in grads))
                b.zero_grad()
            steps=b._aux_total_steps;b.eval();out=b(feed())
            self.assertTrue(torch.isfinite(out["prediction"]).all())
            self.assertEqual(steps,b._aux_total_steps)
            b.train();b._cur_epoch=6;b.actions_before_epoch()
            out=b(feed());b.loss(out);self.assertEqual(out["aux_weight"],.005)
if __name__=="__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
