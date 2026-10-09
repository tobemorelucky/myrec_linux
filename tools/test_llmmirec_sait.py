"""SAIT-specific CPU mathematical tests. Synthetic, memory-only assets; no model training run."""
import argparse
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
from unittest.mock import patch, mock_open
import numpy as np
import pandas as pd
import torch
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF as C0
from models.sequential.LLMMIRecASPCFSAIT import LLMMIRecASPCFSAIT as SAIT
from models.sequential.sait_components import SemanticInterestTransition
from tools.build_sait_transition_assets import build, semantic_prior, validate_prototypes

torch.set_num_threads(2)
rng = np.random.default_rng(13)
Q = rng.dirichlet(np.ones(32), size=33).astype(np.float32); Q[0] = 0
CENTERS = rng.normal(size=(32,512)).astype(np.float32)
PROTO = dict(soft_assignments=Q, centers=CENTERS, semantic_rank=512,
             prototype_num=32, temperature=.1)
TRAIN = pd.DataFrame(dict(user_id=[0,0,1,1,2,2], item_id=[1,2,3,4,5,6], time=[1,2,1,2,1,2]))
ASSET = build(TRAIN, PROTO)
TABLE = torch.randn(33,1536,generator=torch.Generator().manual_seed(17));TABLE[0]=0
ROOT = Path(__file__).resolve().parents[1]

def make(mode="full", baseline=False, weight=.01):
    args = SAIT.parse_model_args(argparse.ArgumentParser()).parse_args([])
    args.device=torch.device("cpu");args.model_path="unused";args.llm_emb_path="synthetic"
    args.sait_mode=mode;args.sait_asset_path=str(ROOT/"data/synthetic/sait.pkl")
    args.lambda_transition=weight;args.relation_sample_size=5
    torch.manual_seed(42)
    with patch("models.sequential.LLMMIRecASPCF.load_llm_table",return_value=TABLE), \
         patch("models.sequential.LLMMIRecASPCFSAIT.Path.open",mock_open()), \
         patch("models.sequential.LLMMIRecASPCFSAIT.pickle.load",return_value=ASSET):
        model = (C0 if baseline else SAIT)(args,SimpleNamespace(n_items=33,n_users=4))
    return model

def feed():
    return dict(user_id=torch.tensor([1,2]),history_items=torch.tensor([[1,2,0],[3,4,5]]),
                lengths=torch.tensor([2,3]), item_id=torch.tensor([[6,20],[7,21]]))

class Tests(unittest.TestCase):
    def test_disabled_exact_c0_and_initialization_rng(self):
        a,b=make(baseline=True),make("baseline")
        self.assertEqual(list(a.state_dict()),list(b.state_dict()))
        for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[k]))
        torch.manual_seed(95);oa=a(feed());ra=torch.get_rng_state()
        torch.manual_seed(95);ob=b(feed());rb=torch.get_rng_state()
        self.assertTrue(torch.equal(oa["prediction"],ob["prediction"]))
        self.assertTrue(torch.equal(ra,rb))
        la,lb=a.loss(oa),b.loss(ob);self.assertTrue(torch.equal(la,lb))
        la.backward();lb.backward()
        for pa,pb in zip(a.parameters(),b.parameters()):
            if pa.grad is not None:self.assertTrue(torch.equal(pa.grad,pb.grad))
        torch.manual_seed(42)
        with patch("models.sequential.LLMMIRecASPCF.load_llm_table",return_value=TABLE):
            args=SAIT.parse_model_args(argparse.ArgumentParser()).parse_args([])
            args.device=torch.device("cpu");args.model_path="unused";args.llm_emb_path="synthetic"
            C0(args,SimpleNamespace(n_items=33,n_users=4))
        expected=torch.get_rng_state()
        make("full")
        self.assertTrue(torch.equal(expected,torch.get_rng_state()))

    def test_shared_parameters_encoder_calls_and_relation(self):
        a,b=make(baseline=True),make()
        for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[k]))
        self.assertEqual(b.count_variables()-a.count_variables(),18056)
        with patch.object(b.item_encoder,"forward",wraps=b.item_encoder.forward) as enc:
            torch.manual_seed(71);ob=b(feed())
            self.assertEqual(enc.call_count,2) # history and original candidates only
        torch.manual_seed(71);oa=a(feed())
        self.assertTrue(torch.equal(oa["_relation_ids"],ob["_relation_ids"]))
        a.loss(oa);b.loss(ob)
        self.assertTrue(torch.equal(oa["loss_relation"],ob["loss_relation"]))
        self.assertFalse(any("embedding" in n for n,_ in b.forecaster.named_parameters()))

    def test_forward_backward_and_transition_supervision_gradients(self):
        b=make();o=b(feed(),True);loss=b.loss(o)
        self.assertTrue(torch.isfinite(loss));loss.backward()
        for p in b.parameters():
            if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all())
        for name,p in b.forecaster.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertGreater(float(p.grad.abs().sum()),0,name)
        self.assertGreater(float(b.item_encoder.semantic_branch[0].weight.grad.abs().sum()),0)
        self.assertTrue(torch.allclose(o["next_state"].sum(-1),torch.ones(2),atol=1e-6))
        b.zero_grad()
        o=b(feed(),True)
        supervised=torch.nn.functional.kl_div(o["_sait_next_state"].log(),o["_sait_target_state"],reduction="batchmean")
        supervised.backward()
        self.assertGreater(float(b.forecaster.context.weight.grad.abs().sum()),0)
        self.assertGreater(float(b.forecaster.source.grad.abs().sum()),0)

    def test_forecast_actually_changes_scoring(self):
        b=make();b.eval()
        a=b(feed(),True)
        with torch.no_grad():b.forecaster.destination.mul_(100)
        c=b(feed(),True)
        self.assertFalse(torch.equal(a["predicted_state"],c["predicted_state"]))
        self.assertFalse(torch.equal(a["prediction"],c["prediction"]))
        # Ranking gradient reaches transition factors even without transition loss.
        b=make(weight=0);o=b(feed());b.loss(o).backward()
        self.assertGreater(float(b.forecaster.destination.grad.abs().sum()),0)

    def test_candidate_permutation_and_target_train_only(self):
        b=make();b.eval();f=feed();a=b(f,True)
        perm=[1,0];g=copy.deepcopy(f);g["item_id"]=g["item_id"][:,perm]
        c=b(g,True)
        self.assertTrue(torch.equal(a["user_vector"],c["user_vector"]))
        self.assertTrue(torch.equal(a["prediction"][:,perm],c["prediction"]))
        self.assertNotIn("_sait_target_state",a)
        b.train()
        torch.manual_seed(33);a=b(f,True)
        torch.manual_seed(33);c=b(g,True)
        self.assertTrue(torch.equal(a["user_vector"],c["user_vector"]))
        self.assertTrue(torch.equal(a["_sait_target_state"],torch.tensor(Q)[f["item_id"][:,0]]))

    def test_padding_length_one_and_no_transition(self):
        b=make();b.eval()
        a=feed();a["history_items"]=torch.tensor([[1,0,0],[3,0,0]]);a["lengths"]=torch.tensor([1,1])
        short=copy.deepcopy(a);short["history_items"]=short["history_items"][:,:1]
        x,y=b(a,True),b(short,True)
        self.assertTrue(torch.allclose(x["prediction"],y["prediction"],atol=1e-8))
        self.assertTrue(torch.allclose(x["next_state"],y["next_state"],atol=1e-7))
        empty=build(TRAIN.iloc[[0,2,4]],PROTO)
        self.assertEqual(empty["meta"]["adjacent_pairs"],0)
        self.assertTrue(np.allclose(empty["train_transition"],empty["semantic_prior"][None]))
        zero=dict(PROTO,centers=np.zeros_like(CENTERS))
        self.assertTrue(np.allclose(semantic_prior(zero["centers"]),1/32))

    def test_fold_exclusion_order_and_no_cross_user_pairs(self):
        a=build(TRAIN.sample(frac=1,random_state=9),PROTO)
        for f in range(3):
            sub=TRAIN[TRAIN.user_id%3!=f]
            expected=build(sub,PROTO)["eval_transition"]
            self.assertTrue(np.allclose(a["train_transition"][f],expected,atol=1e-7))
        reverse=TRAIN.copy();reverse["time"]=-reverse["time"]
        rev=build(reverse,PROTO)
        self.assertTrue(np.allclose(rev["counts_by_fold"],a["counts_by_fold"].transpose(0,2,1)))
        self.assertEqual(a["meta"]["adjacent_pairs"],3)
        for f in range(3):
            expected=np.outer(Q[2*f+1],Q[2*f+2])
            self.assertTrue(np.allclose(a["counts_by_fold"][f],expected))
        bad=copy.deepcopy(PROTO);bad["soft_assignments"]=Q.copy();bad["soft_assignments"][0]=1
        with self.assertRaises(ValueError):validate_prototypes(bad)

    def test_modes_finite_and_markov_exact_state(self):
        for mode in ("semantic_only","behavior_only","markov"):
            b=make(mode);b.eval();o=b(feed(),True)
            self.assertTrue(torch.isfinite(o["prediction"]).all())
            if mode=="markov":
                history=feed()["history_items"];lengths=feed()["lengths"]
                expected=b.sait_q[history[torch.arange(2),lengths-1]] @ b.sait_eval_prior
                self.assertTrue(torch.allclose(o["next_state"],expected,atol=1e-7))
                self.assertEqual(sum(p.numel() for p in b.forecaster.parameters()),16512)

    def test_builder_io_train_only_and_refuses_existing_output(self):
        from tools.build_sait_transition_assets import construct
        # Existing output must fail before any PCA/train/dev/test read.
        with patch("tools.build_sait_transition_assets.Path.exists",return_value=True), \
             patch("tools.build_sait_transition_assets.pd.read_csv",side_effect=AssertionError("CSV accessed")):
            with self.assertRaises(FileExistsError):construct("beauty",ROOT/"data/synthetic/existing.pkl")
        # Exercise production I/O with strict in-memory whitelist; dev/test access fails.
        import io, pickle
        table=TABLE.numpy()
        centers=CENTERS/(np.linalg.norm(CENTERS,axis=1,keepdims=True)+1e-8)
        z=table[1:,:512];z=z/(np.linalg.norm(z,axis=1,keepdims=True)+1e-8)
        logits=z@centers.T/.1;logits-=logits.max(1,keepdims=True)
        probabilities=np.exp(logits);probabilities/=probabilities.sum(1,keepdims=True)
        proto=copy.deepcopy(PROTO);proto["soft_assignments"]=np.vstack(
            [np.zeros((1,32),np.float32),probabilities]).astype(np.float32)
        opened=[];csv_calls=[]
        class MemoryOutput(io.BytesIO):
            def close(self):pass
        output=MemoryOutput()
        def fake_open(path,mode="r",*args,**kwargs):
            opened.append((str(path),mode))
            if path.name=="llmmi_proto32_sr512.pkl" and mode=="rb":
                return io.BytesIO(pickle.dumps(proto))
            if path.name=="llm_table_pca1536.pkl" and mode=="rb":
                return io.BytesIO(pickle.dumps(table))
            if path.name=="output.pkl" and mode=="xb":return output
            raise AssertionError("Unexpected file: "+str(path))
        def fake_csv(path,*args,**kwargs):
            csv_calls.append(str(path))
            self.assertEqual(Path(path),ROOT/"data/beauty/train.csv")
            return TRAIN.copy()
        with patch("tools.build_sait_transition_assets.Path.exists",return_value=False), \
             patch("tools.build_sait_transition_assets.Path.open",fake_open), \
             patch("tools.build_sait_transition_assets.Path.mkdir"), \
             patch("tools.build_sait_transition_assets.pd.read_csv",side_effect=fake_csv), \
             patch("tools.build_sait_transition_assets.digest",return_value="synthetic"), \
             patch("builtins.print"):
            construct("beauty",ROOT/"data/synthetic/output.pkl")
        self.assertEqual(len(csv_calls),1)
        self.assertEqual(len(opened),3)
        result=pickle.loads(output.getvalue())
        self.assertEqual(result["meta"]["adjacent_pairs"],3)
        self.assertEqual(result["version"],"sait_v1")

if __name__=="__main__":
    unittest.main(verbosity=2)
