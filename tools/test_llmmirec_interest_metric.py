"""Focused C/I tests: CPU synthetic fixtures; checkpoints in BytesIO only."""
import argparse, copy, io, sys, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tools.test_llmmirec_aspcf_sampled_softmax import TABLE, feed
from models.BaseModel import GeneralModel
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF as Base
from models.sequential.LLMMIRecASPCFInterestMetric import LLMMIRecASPCFInterestMetric as Model
torch.set_num_threads(2)


def make(mode="C", K=4, base=False):
    cls = Base if base else Model
    args = cls.parse_model_args(argparse.ArgumentParser()).parse_args([])
    args.device = torch.device("cpu"); args.model_path = "unused"; args.buffer = 0
    args.llm_emb_path = "synthetic"; args.relation_sample_size = 5; args.K = K
    if not base: args.metric_mode = mode
    corpus = SimpleNamespace(n_users=5, n_items=130)
    torch.manual_seed(42)
    with patch("models.sequential.LLMMIRecASPCF.load_llm_table", return_value=TABLE):
        return cls(args, corpus)


class Tests(unittest.TestCase):
    def test_zero_prediction_loss_shared_gradients_and_parameter_budget(self):
        base = make(base=True); rng = torch.get_rng_state()
        for mode in ["C", "I"]:
            m = make(mode)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(m.count_variables()-base.count_variables(),256)
            self.assertEqual(m.a.shape,(4,64)); self.assertEqual(int(torch.count_nonzero(m.a)),0)
            self.assertIs(Model.loss,Base.loss); self.assertIs(Model.Dataset,Base.Dataset)
            self.assertFalse(hasattr(m,"train_membership"))
            for k,v in base.state_dict().items(): self.assertTrue(torch.equal(v,m.state_dict()[k]),k)
            base.zero_grad();m.zero_grad()
            torch.manual_seed(91); x=base(feed(),True); lx=base.loss(x)
            torch.manual_seed(91); y=m(feed(),True); ly=m.loss(y)
            for key in ["prediction","_relation_ids","loss_relation"]: self.assertTrue(torch.equal(x[key],y[key]),key)
            self.assertTrue(torch.equal(lx,ly))
            self.assertTrue(torch.equal(ly,GeneralModel.loss(m,y)+.01*m._compute_relation_loss(y["_relation_ids"])))
            lx.backward();ly.backward()
            for name,p in base.named_parameters(): self.assertTrue(torch.equal(p.grad,dict(m.named_parameters())[name].grad),name)
            self.assertGreater(m.a.grad.abs().sum().item(),0)
            self.assertTrue(torch.isfinite(m.a.grad).all())

    def test_covariance_identity_K4_difference_and_permutation(self):
        torch.manual_seed(30)
        V=torch.randn(5,4,64,dtype=torch.double); w=torch.randn(5,4,dtype=torch.double).softmax(1)
        a=torch.randn(4,64,dtype=torch.double); u=(w[...,None]*V).sum(1)
        c=Model.correction(V,w,u,a,"C"); i=Model.correction(V,w,u,a,"I")
        abar=(w[...,None]*a).sum(1)
        rhs=(w[...,None]*(a[None]-abar[:,None])*(V-u[:,None])).sum(1)
        torch.testing.assert_close(i-c,rhs,atol=1e-12,rtol=1e-12)
        e=torch.randn(5,7,64,dtype=torch.double)
        self.assertGreater((((u+i)[:,None]*e).sum(-1)-((u+c)[:,None]*e).sum(-1)).abs().max().item(),.1)
        order=torch.tensor([2,0,3,1])
        for mode in ["C","I"]:
            torch.testing.assert_close(Model.correction(V[:,order],w[:,order],u,a[order],mode),
                                       Model.correction(V,w,u,a,mode),atol=1e-12,rtol=1e-12)

    def test_K1_exact_with_nonzero_metric_and_gradients(self):
        c=make("C",K=1);i=make("I",K=1)
        self.assertEqual(c.count_variables()-make(K=1,base=True).count_variables(),64)
        with torch.no_grad(): c.a.copy_(torch.linspace(-.2,.3,64)[None])
        i.load_state_dict(c.state_dict())
        torch.manual_seed(81);x=c(feed(),True);lx=c.loss(x)
        torch.manual_seed(81);y=i(feed(),True);ly=i.loss(y)
        self.assertTrue(torch.equal(x["interest_weights"],torch.ones(3,1)))
        self.assertTrue(torch.equal(x["prediction"],y["prediction"]))
        self.assertTrue(torch.equal(lx,ly));lx.backward();ly.backward()
        for name,p in c.named_parameters(): self.assertTrue(torch.equal(p.grad,dict(i.named_parameters())[name].grad),name)

    def test_lengths_padding_and_future_prefix(self):
        for K in [1,4]:
            for mode in ["C","I"]:
                m=make(mode,K);m.eval()
                with torch.no_grad():m.a.fill_(.1)
                for n in [1,2,3,20]:
                    f={"history_items":torch.arange(1,n+1)[None],"lengths":torch.tensor([n]),"item_id":torch.tensor([[50,60]])}
                    g=copy.deepcopy(f);g["history_items"]=torch.cat([g["history_items"],torch.zeros(1,20-n,dtype=torch.long)],1)
                    with torch.no_grad():x=m(f,True);y=m(g,True)
                    torch.testing.assert_close(x["prediction"],y["prediction"],atol=1e-7,rtol=1e-5)
                    self.assertTrue(torch.isfinite(y["prediction"]).all())
                    self.assertEqual(int(torch.count_nonzero(y["attention_maps"][...,n:])),0)
        m=make();d=object.__new__(Model.Dataset);d.model=m;d.phase="train"
        d.corpus=SimpleNamespace(user_his={1:[(j+1,j) for j in range(50)]})
        d.data=dict(user_id=[1],item_id=[36],neg_items=[torch.tensor([60]).numpy()],position=[35])
        r=d._get_feed_dict(0)
        self.assertEqual(r["history_items"].tolist(),list(range(16,36)))
        self.assertNotIn(36,r["history_items"])

    def test_eval_candidates_isolation_encoder_reuse_and_checkpoint(self):
        for mode in ["C","I"]:
            m=make(mode);m.eval()
            with torch.no_grad():m.a.copy_(torch.randn_like(m.a)*.1)
            f=feed();f["item_id"]=torch.arange(1001)[None].expand(3,-1)%129+1
            calls=[]
            hook=m.item_encoder.register_forward_hook(lambda mod,ins,out:calls.append(tuple(ins[0].shape)))
            with torch.no_grad():x=m(f,True)
            hook.remove();self.assertEqual(len(calls),2)
            self.assertEqual(x["prediction"].shape,(3,1001))
            g=copy.deepcopy(f);g["item_id"]=g["item_id"].flip(1)
            with torch.no_grad():y=m(g,True);single=m({k:v[:1] for k,v in f.items()},True)
            self.assertTrue(torch.equal(x["user_vector"],y["user_vector"]))
            self.assertTrue(torch.equal(x["prediction"].flip(1),y["prediction"]))
            torch.testing.assert_close(x["prediction"][:1],single["prediction"],atol=1e-7,rtol=1e-5)
            buffer=io.BytesIO();torch.save(m.state_dict(),buffer);buffer.seek(0);z=make(mode);z.load_state_dict(torch.load(buffer,weights_only=True));z.eval()
            with torch.no_grad():self.assertTrue(torch.equal(x["prediction"],z(f)["prediction"]))


if __name__=="__main__":unittest.main(verbosity=2)
