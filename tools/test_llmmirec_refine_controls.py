"""Small A/B attribution tests; no persistent data or checkpoint writes."""
import argparse, copy, sys, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from tools.test_llmmirec_dualview_refine import TABLE,feed
from models.sequential.LLMMIRecASPCFDualViewRefine import LLMMIRecASPCFDualViewRefine as Full
from models.sequential.LLMMIRecASPCFRefineControls import LLMMIRecASPCFUserRefine as A, LLMMIRecASPCFSliceRefine as B
from models.sequential.LLMMIRecASPCFSampledSoftmax import LLMMIRecASPCFSampledSoftmax as SS
torch.set_num_threads(2)
def build(cls,K=4):
    args=cls.parse_model_args(argparse.ArgumentParser()).parse_args([])
    args.device=torch.device("cpu");args.model_path="unused";args.buffer=0
    args.llm_emb_path="synthetic";args.K=K;args.history_max=20;args.relation_sample_size=5
    c=SimpleNamespace(n_users=4,n_items=60,train_clicked_set={1:{1,2,6},2:{3,4,7},3:{5,8}})
    torch.manual_seed(42)
    with patch("models.sequential.LLMMIRecASPCF.load_llm_table",return_value=TABLE):return cls(args,c)
class Tests(unittest.TestCase):
    def test_initial_identity_rng_parameters_loss_and_gradients(self):
        full=build(Full);b=build(B)
        for n,p in full.state_dict().items():self.assertTrue(torch.equal(p,b.state_dict()[n]),n)
        for cls,added in [(A,28776),(B,28864)]:
            base=build(SS);rng=torch.get_rng_state();m=build(cls)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(m.count_variables()-base.count_variables(),added)
            for n,p in base.state_dict().items():self.assertTrue(torch.equal(p,m.state_dict()[n]),n)
            torch.manual_seed(9);o=base(feed(),True);l=base.loss(o)
            torch.manual_seed(9);oo=m(feed(),True);ll=m.loss(oo)
            for k in ["prediction","_ss_scores","_ss_allowed","_relation_ids","loss_relation"]:
                self.assertTrue(torch.equal(o[k],oo[k]),k)
            self.assertTrue(torch.equal(l,ll));l.backward();ll.backward()
            for n,p in base.named_parameters():self.assertTrue(torch.equal(p.grad,dict(m.named_parameters())[n].grad),n)
            m.actions_after_train();base.actions_after_train()
    def test_A_only_user_B_only_slices_and_context(self):
        for cls in [A,B]:
            m=build(cls);m.eval()
            with patch.object(m.item_encoder,"forward",wraps=m.item_encoder.forward) as enc:
                o=m(feed(),True)
            self.assertEqual(enc.call_count,2)
            if cls is A:
                self.assertTrue(torch.equal(o["refinement_features"],o["base_user_vector"]))
                self.assertEqual(tuple(o["refinement_features"].shape),(3,64))
            else:
                v=o["interest_vectors"]
                self.assertTrue(torch.equal(o["refinement_features"][...,64:96],v[...,:32]))
                self.assertTrue(torch.equal(o["refinement_features"][...,96:128],v[...,32:]))
                ref={k:v for k,v in o.items() if k not in ("history_semantic","history_complement")}
                m._refine(ref,feed()["history_items"],feed()["lengths"]) # explicit views not read
                full=build(Full);full.eval();fo=full(feed(),True)
                self.assertTrue(torch.equal(fo["refinement_context"],o["refinement_context"]))
                self.assertTrue(torch.equal(fo["attention_maps"],o["attention_maps"]))
    def test_two_updates_all_new_parameters_effective(self):
        for cls in [A,B]:
            m=build(cls);opt=torch.optim.SGD(m.parameters(),lr=.1)
            for step in range(2):
                opt.zero_grad();o=m(feed());l=m.loss(o);l.backward()
                self.assertTrue(torch.isfinite(l))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
                self.assertGreater(float(m.refiner.output.weight.grad.abs().sum()),0)
                if step==0:self.assertEqual(float(m.refiner.input.weight.grad.abs().sum()),0)
                else:
                    for n,p in m.refiner.named_parameters():self.assertGreater(float(p.grad.abs().sum()),0,n)
                opt.step()
            m.actions_after_train()
    def test_active_refinement_eval_K1_length1_and_1001(self):
        for cls in [A,B]:
            for K in [1,4]:
                m=build(cls,K);m.eval()
                with torch.no_grad():m.refiner.output.weight.normal_(std=.01)
                f=feed();f["history_items"]=f["history_items"][:,:1];f["lengths"]=torch.ones(3,dtype=torch.long)
                f["item_id"]=torch.arange(1001)[None,:].expand(3,-1)%59+1
                with torch.no_grad():
                    o=m(f,True)
                    ff=copy.deepcopy(f);ff["item_id"]=ff["item_id"].flip(1);oo=m(ff,True)
                    single=m({k:v[:1] for k,v in f.items()},True)
                self.assertTrue(torch.equal(o["prediction"].flip(1),oo["prediction"]))
                self.assertTrue(torch.allclose(o["prediction"][:1],single["prediction"],atol=1e-7,rtol=1e-5))
                self.assertEqual(o["prediction"].shape,(3,1001))
                self.assertNotIn("_ss_scores",o);self.assertNotIn("_relation_ids",o)
                self.assertTrue(torch.equal(o["user_vector"],oo["user_vector"]))
    def test_H20_future_mask_and_relation_remain_same(self):
        for cls in [A,B]:
            m=build(cls);base=build(SS)
            d=object.__new__(m.Dataset);d.model=m;d.phase="train"
            d.corpus=SimpleNamespace(user_his={1:[(i+1,i) for i in range(50)]},n_items=60)
            d.data=dict(user_id=[1],item_id=[36],neg_items=[np.array([55])],position=[35])
            f=d._get_feed_dict(0)
            self.assertEqual(f["history_items"].tolist(),list(range(16,36)));self.assertNotIn(36,f["history_items"])
            with torch.no_grad():m.refiner.output.weight.normal_(std=.01)
            ff=feed();ff["item_id"][2,0]=7
            torch.manual_seed(4);o=base(ff,True);base.loss(o)
            torch.manual_seed(4);oo=m(ff,True);m.loss(oo)
            for k in ["_ss_allowed","_relation_ids","loss_relation"]:self.assertTrue(torch.equal(o[k],oo[k]),k)
            self.assertTrue(oo["_ss_allowed"].diagonal().all());self.assertFalse(oo["_ss_allowed"][1,2])
            m.actions_after_train();base.actions_after_train()
if __name__=="__main__":unittest.main(verbosity=2)
