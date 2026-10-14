"""Bounded synthetic tests for A0/A1 only; no persistent assets."""
import argparse, copy, io, sys, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tools.test_llmmirec_aspcf_sampled_softmax import TABLE, feed
from models.sequential.LLMMIRecASPCFSampledSoftmax import LLMMIRecASPCFSampledSoftmax as SS
from models.sequential.LLMMIRecASPCFLowRankQuery import LLMMIRecASPCFLowRankQuery as Model
torch.set_num_threads(2)

def make(mode="dynamic", K=4, baseline=False):
    cls = SS if baseline else Model
    args = cls.parse_model_args(argparse.ArgumentParser()).parse_args([])
    args.device = torch.device("cpu"); args.model_path = "unused"; args.buffer = 0
    args.llm_emb_path = "synthetic"; args.K = K; args.history_max = 20
    args.relation_sample_size = 5
    if not baseline: args.query_adaptation = mode
    corpus = SimpleNamespace(n_users=5, n_items=130,
                             train_clicked_set={1:{1,2,6},2:{3,4,7},3:{5,8}})
    torch.manual_seed(42)
    with patch("models.sequential.LLMMIRecASPCF.load_llm_table", return_value=TABLE):
        return cls(args, corpus)

class Tests(unittest.TestCase):
    def test_identity_shared_parameters_rng_and_loss(self):
        base = make(baseline=True)
        rng = torch.get_rng_state()
        models = [make(mode) for mode in ["static", "dynamic"]]
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for m in models:
            self.assertEqual(m.count_variables()-base.count_variables(),1544)
            for key,v in base.state_dict().items():
                self.assertTrue(torch.equal(v,m.state_dict()[key]),key)
            torch.manual_seed(9); o=base(feed(),True); lb=base.loss(o)
            torch.manual_seed(9); z=m(feed(),True); lm=m.loss(z)
            for key in ["prediction","_ss_scores","_ss_allowed","_relation_ids","loss_relation"]:
                self.assertTrue(torch.equal(o[key],z[key]),key)
            self.assertTrue(torch.equal(lb,lm))
            base.zero_grad();m.zero_grad();lb.backward();lm.backward()
            for key,p in base.named_parameters():
                self.assertTrue(torch.equal(p.grad,dict(m.named_parameters())[key].grad),key)

    def test_shape_context_dependency_and_padding(self):
        h=torch.randn(3,4,64,generator=torch.Generator().manual_seed(2))
        lengths=torch.tensor([4,2,1])
        for mode in ["static","dynamic"]:
            m=make(mode); ex=m.extractor
            with torch.no_grad():ex.low_rank.U.fill_(.03)
            q=ex.make_query(h,lengths)
            self.assertEqual(q.shape,(3,4,64))
            if mode=="static": self.assertTrue(torch.equal(q[0],q[1]))
            else: self.assertGreater(float((q[0]-q[1]).abs().max()),0)
            h2=h.clone();h2[1,2:]=900;h2[2,1:]=-800
            self.assertTrue(torch.equal(q,ex.make_query(h2,lengths)))
            v,a=ex(h,lengths)
            self.assertTrue(torch.equal(a[1,:,2:],torch.zeros_like(a[1,:,2:])))
            self.assertTrue(torch.isfinite(v).all())

    def test_two_updates_all_added_parameters_active(self):
        for mode in ["static","dynamic"]:
            m=make(mode);opt=torch.optim.Adam(m.parameters(),lr=.004)
            for step in range(2):
                opt.zero_grad();torch.manual_seed(13+step)
                o=m(feed(),True);loss=m.loss(o);loss.backward()
                self.assertTrue(torch.isfinite(loss))
                for p in m.parameters():
                    if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all())
                self.assertGreater(float(m.extractor.low_rank.U.grad.abs().sum()),0)
                if step:
                    for name,p in m.extractor.low_rank.named_parameters():
                        self.assertGreater(float(p.grad.abs().sum()),0,name)
                opt.step()
            m.actions_after_train()

    def test_K1_length1_and_eval_batch_candidate_isolation(self):
        for mode in ["static","dynamic"]:
            m=make(mode,K=1);m.eval()
            with torch.no_grad():m.extractor.low_rank.U.fill_(.03)
            f=feed();f["history_items"]=f["history_items"][:,:1]
            f["lengths"]=torch.ones(3,dtype=torch.long)
            f["item_id"]=torch.arange(1001)[None].expand(3,-1)%129+1
            with torch.no_grad():
                o=m(f,True);g=copy.deepcopy(f);g["item_id"]=g["item_id"].flip(1);z=m(g,True)
                single=m({k:v[:1] for k,v in f.items()},True)
            self.assertEqual(o["prediction"].shape,(3,1001))
            self.assertTrue(torch.equal(o["interest_weights"],torch.ones(3,1)))
            self.assertTrue(torch.equal(o["prediction"].flip(1),z["prediction"]))
            self.assertTrue(torch.allclose(single["prediction"],o["prediction"][:1],atol=1e-7,rtol=1e-5))
            self.assertNotIn("_ss_scores",o)

    def test_checkpoint_and_history_only_query(self):
        m=make();m.eval()
        buffer=io.BytesIO();torch.save(m.state_dict(),buffer);buffer.seek(0)
        n=make();n.load_state_dict(torch.load(buffer,weights_only=True));n.eval()
        f=feed()
        with torch.no_grad():
            self.assertTrue(torch.equal(m(f)["prediction"],n(f)["prediction"]))
            x=m.item_encoder(f["history_items"]);q=m.extractor.make_query(x,f["lengths"])
            g=copy.deepcopy(f);g["item_id"].fill_(129)
            q2=m.extractor.make_query(m.item_encoder(g["history_items"]),g["lengths"])
            self.assertTrue(torch.equal(q,q2))
        self.assertIs(Model.loss,SS.loss)

if __name__=="__main__":unittest.main(verbosity=2)
