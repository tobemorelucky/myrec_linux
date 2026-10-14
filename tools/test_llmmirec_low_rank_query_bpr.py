"""Only BPR path and equivalence tests; no persistent data/checkpoint writes."""
import argparse, copy, io, sys, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tools.test_llmmirec_aspcf_sampled_softmax import TABLE,feed
from models.BaseModel import GeneralModel
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF as Base
from models.sequential.LLMMIRecASPCFLowRankQueryBPR import LLMMIRecASPCFLowRankQueryBPR as Model
from models.sequential.LLMMIRecASPCFSampledSoftmax import LLMMIRecASPCFSampledSoftmax as SS
torch.set_num_threads(2)

def make(mode="dynamic",base=False):
    cls=Base if base else Model
    args=cls.parse_model_args(argparse.ArgumentParser()).parse_args([])
    args.device=torch.device("cpu");args.model_path="unused";args.buffer=0
    args.llm_emb_path="synthetic";args.relation_sample_size=5
    if not base:args.query_adaptation=mode
    c=SimpleNamespace(n_users=5,n_items=130,train_clicked_set={1:{1,2,6},2:{3,4,7},3:{5,8}})
    torch.manual_seed(42)
    with patch("models.sequential.LLMMIRecASPCF.load_llm_table",return_value=TABLE):
        return cls(args,c)

class Tests(unittest.TestCase):
    def test_initial_prediction_original_loss_gradients_and_no_SS(self):
        base=make(base=True);rng=torch.get_rng_state()
        for mode in ["static","dynamic"]:
            m=make(mode)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(m.count_variables()-base.count_variables(),1544)
            self.assertFalse(hasattr(m,"train_membership"))
            self.assertFalse(hasattr(m,"sampled_softmax_temperature"))
            self.assertNotIn("sampled_softmax_temperature",vars(Model.parse_model_args(argparse.ArgumentParser()).parse_args([])))
            self.assertFalse(issubclass(Model,SS));self.assertIs(Model.loss,Base.loss)
            for key,v in base.state_dict().items():self.assertTrue(torch.equal(v,m.state_dict()[key]),key)
            torch.manual_seed(91);a=base(feed(),True);la=base.loss(a)
            torch.manual_seed(91);b=m(feed(),True);lb=m.loss(b)
            self.assertFalse(any(k.startswith("_ss_") for k in b))
            for key in ["prediction","_relation_ids","loss_relation"]:self.assertTrue(torch.equal(a[key],b[key]),key)
            self.assertTrue(torch.equal(la,lb))
            ranking=GeneralModel.loss(m,{"prediction":b["prediction"]})
            self.assertTrue(torch.equal(lb,ranking+.01*m._compute_relation_loss(b["_relation_ids"])))
            base.zero_grad();m.zero_grad();la.backward();lb.backward()
            for key,p in base.named_parameters():self.assertTrue(torch.equal(p.grad,dict(m.named_parameters())[key].grad),key)

    def test_two_updates_new_parameters_finite_and_active(self):
        for mode in ["static","dynamic"]:
            m=make(mode);opt=torch.optim.Adam(m.parameters(),lr=.004)
            for step in range(2):
                opt.zero_grad();torch.manual_seed(11+step);o=m(feed(),True);l=m.loss(o);l.backward()
                self.assertTrue(torch.isfinite(l))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
                self.assertGreater(float(m.extractor.low_rank.U.grad.abs().sum()),0)
                if step:
                    for name,p in m.extractor.low_rank.named_parameters():self.assertGreater(float(p.grad.abs().sum()),0,name)
                opt.step()

    def test_history_and_1001_eval_same_protocol_as_SS(self):
        m=make();m.eval()
        with torch.no_grad():m.extractor.low_rank.U.fill_(.03)
        f=feed();f["item_id"]=torch.arange(1001)[None].expand(3,-1)%129+1
        with torch.no_grad():
            o=m(f,True);g=copy.deepcopy(f);g["item_id"]=g["item_id"].flip(1);z=m(g,True)
            single=m({k:v[:1] for k,v in f.items()},True)
        self.assertTrue(torch.isfinite(o["prediction"]).all())
        self.assertEqual(o["prediction"].shape,(3,1001))
        self.assertTrue(torch.equal(o["prediction"],(o["user_vector"][:,None]*o["candidate_vectors"]).sum(-1)))
        self.assertTrue(torch.equal(o["prediction"].flip(1),z["prediction"]))
        self.assertTrue(torch.allclose(single["prediction"],o["prediction"][:1],atol=1e-7,rtol=1e-5))
        self.assertIs(Model.Dataset,Base.Dataset);self.assertIs(Model.Dataset,SS.Dataset)
        d=object.__new__(m.Dataset);d.model=m;d.phase="train"
        d.corpus=SimpleNamespace(user_his={1:[(i+1,i) for i in range(50)]})
        d.data=dict(user_id=[1],item_id=[36],neg_items=[torch.tensor([60]).numpy()],position=[35])
        r=d._get_feed_dict(0)
        self.assertEqual(r["history_items"].tolist(),list(range(16,36)))
        self.assertNotIn(36,r["history_items"])

    def test_checkpoint_roundtrip_and_A0_A1_parameters_match(self):
        a,b=make("static"),make("dynamic")
        for key,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[key]))
        buffer=io.BytesIO();torch.save(b.state_dict(),buffer);buffer.seek(0)
        c=make();c.load_state_dict(torch.load(buffer,weights_only=True))
        b.eval();c.eval()
        with torch.no_grad():self.assertTrue(torch.equal(b(feed())["prediction"],c(feed())["prediction"]))

if __name__=="__main__":unittest.main(verbosity=2)
