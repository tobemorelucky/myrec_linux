"""Only CICL-specific synthetic tests; no real asset writes or training."""
import argparse,copy,sys,unittest,math,itertools
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,mock_open
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF as C0
from models.sequential.LLMMIRecASPCFCICL import LLMMIRecASPCFCICL as CICL
from models.sequential.cicl_components import CoalitionTeacher,corrected_weights
torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parents[1]
TABLE=torch.randn(33,1536,generator=torch.Generator().manual_seed(17));TABLE[0]=0
rng=np.random.default_rng(21);Q=rng.dirichlet(np.ones(32),size=33).astype(np.float32);Q[0]=0
ASSET=dict(soft_assignments=Q,semantic_rank=512)
def make(c0=False,weight=.01,interests=4):
    args=CICL.parse_model_args(argparse.ArgumentParser()).parse_args([])
    args.device=torch.device("cpu");args.model_path="unused";args.buffer=0
    args.llm_emb_path="synthetic";args.cicl_proto_path=str(ROOT/"data/synthetic/proto.pkl")
    args.lambda_contribution=weight;args.relation_sample_size=5;args.K=interests
    torch.manual_seed(42)
    with patch("models.sequential.LLMMIRecASPCF.load_llm_table",return_value=TABLE),patch("models.sequential.LLMMIRecASPCFCICL.Path.open",mock_open()),patch("models.sequential.LLMMIRecASPCFCICL.pickle.load",return_value=ASSET):
        return (C0 if c0 else CICL)(args,SimpleNamespace(n_items=33,n_users=4))
def feed():
    return dict(user_id=torch.tensor([1,2]),history_items=torch.tensor([[1,2,0],[3,4,5]]),lengths=torch.tensor([2,3]),item_id=torch.tensor([[6,20],[7,21]]))
class Tests(unittest.TestCase):
    def test_zero_prediction_shared_gradients_rng_relation(self):
        a,b=make(True),make()
        for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[k]),k)
        make(True);ra=torch.get_rng_state();make();self.assertTrue(torch.equal(ra,torch.get_rng_state()))
        torch.manual_seed(91);oa=a(feed(),True)
        torch.manual_seed(91);ob=b(feed(),True)
        self.assertTrue(torch.equal(oa["prediction"],ob["prediction"]))
        self.assertTrue(torch.equal(oa["user_vector"],ob["user_vector"]))
        self.assertTrue(torch.equal(oa["interest_weights"],ob["interest_weights"]))
        a.loss(oa).backward();b.loss(ob).backward()
        self.assertTrue(torch.equal(oa["loss_relation"],ob["loss_relation"]))
        for name,p in a.named_parameters():
            if p.grad is not None:self.assertTrue(torch.equal(p.grad,dict(b.named_parameters())[name].grad),name)
        self.assertGreater(float(b.contribution_predictor.output.weight.grad.abs().sum()),0)
        self.assertEqual(b.count_variables()-a.count_variables(),20736)

    def test_zero_aux_exact_loss_and_shared_encoder_calls(self):
        a,b=make(True),make(weight=0)
        torch.manual_seed(62);oa=a(feed())
        with patch.object(b.item_encoder,"forward",wraps=b.item_encoder.forward) as enc:
            torch.manual_seed(62);ob=b(feed())
            self.assertEqual(enc.call_count,2)
        self.assertTrue(torch.equal(a.loss(oa),b.loss(ob)))

    def test_shapley_reference_efficiency_symmetry_and_extremes(self):
        teacher=CoalitionTeacher().double()
        scores=torch.tensor([[[3.,0.],[1.,0.],[-2.,0.],[.5,0.]],[[1000.,0.],[-1000.,0.],[0.,0.],[0.,0.]]],dtype=torch.float64,requires_grad=True)
        phi,q,u=teacher(scores)
        self.assertFalse(phi.requires_grad);self.assertFalse(q.requires_grad)
        self.assertTrue(torch.equal(u[:,0],torch.zeros(2,dtype=torch.float64)))
        self.assertTrue(torch.allclose(phi.sum(1),u[:,-1],atol=1e-7,rtol=1e-7))
        ref=torch.zeros_like(phi)
        for k in range(4):
            others=[j for j in range(4) if j!=k]
            for n in range(4):
                for s in itertools.combinations(others,n):
                    idx=sum(1<<j for j in s);f=math.factorial(n)*math.factorial(3-n)/24
                    ref[:,k]+=f*(u[:,idx|(1<<k)]-u[:,idx])
        self.assertTrue(torch.allclose(phi,ref,atol=1e-7))
        equal=torch.ones(2,4,2);equal[:,:,0]=2
        ph,qq,_=teacher(equal.double())
        self.assertTrue(torch.allclose(ph,ph[:,0:1].expand(-1,4),atol=1e-7))
        self.assertTrue(torch.allclose(qq,torch.full_like(qq,.25),atol=1e-7))
        for t in [phi,q,u]:self.assertTrue(torch.isfinite(t).all())

    def test_permutation_equivariance_predictor_and_teacher(self):
        b=make();b.eval()
        with torch.no_grad():b.contribution_predictor.output.weight.normal_(0,.02)
        o=b(feed(),True);perm=torch.tensor([2,0,3,1])
        p=b.contribution_predictor
        original=p(o["interest_vectors"],o["attention_maps"],o["history_vectors"],feed()["lengths"],b.cicl_q[feed()["history_items"]])
        reordered=p(o["interest_vectors"][:,perm],o["attention_maps"][:,perm],o["history_vectors"],feed()["lengths"],b.cicl_q[feed()["history_items"]])
        self.assertTrue(torch.allclose(original[:,perm],reordered,atol=1e-7))
        score=torch.randn(2,4,2);phi,q,_=b.coalition_teacher(score);ph2,q2,_=b.coalition_teacher(score[:,perm])
        self.assertTrue(torch.allclose(phi[:,perm],ph2,atol=1e-7))
        self.assertTrue(torch.allclose(q[:,perm],q2,atol=1e-7))
        ww=corrected_weights(o["base_interest_weights"],original)
        wr=corrected_weights(o["base_interest_weights"][:,perm],reordered)
        self.assertTrue(torch.allclose(ww[:,perm],wr,atol=1e-7))

    def test_teacher_eval_isolation_and_target_independent_user(self):
        b=make();b.eval()
        with patch.object(b.coalition_teacher,"forward",side_effect=AssertionError("Eval teacher")):
            a=b(feed(),True);f=copy.deepcopy(feed());f["item_id"]=f["item_id"][:,[1,0]];c=b(f,True)
        self.assertTrue(torch.equal(a["user_vector"],c["user_vector"]))
        self.assertTrue(torch.equal(a["prediction"][:,[1,0]],c["prediction"]))
        self.assertNotIn("_cicl_teacher",a)
        b.train();torch.manual_seed(15);a=b(feed(),True)
        torch.manual_seed(15);c=b(f,True)
        self.assertTrue(torch.equal(a["user_vector"],c["user_vector"]))
        self.assertFalse(torch.equal(a["_cicl_teacher"],c["_cicl_teacher"]))

    def test_aux_only_updates_predictor_not_teacher_or_backbone(self):
        b=make();o=b(feed())
        loss=torch.nn.functional.kl_div(o["_cicl_aux_weights"].log(),o["_cicl_teacher"],reduction="batchmean");loss.backward()
        for name,p in b.named_parameters():
            if not name.startswith("contribution_predictor"):self.assertIsNone(p.grad,name)
        self.assertGreater(float(b.contribution_predictor.output.weight.grad.abs().sum()),0)
        self.assertFalse(o["_cicl_teacher"].requires_grad)

    def test_finite_nonzero_correction_padding_length_one(self):
        b=make()
        with torch.no_grad():b.contribution_predictor.output.weight.normal_(0,.02)
        o=b(feed(),True);loss=b.loss(o);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for p in b.parameters():
            if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all())
        self.assertTrue((o["interest_weights"]>0).all())
        self.assertTrue(torch.allclose(o["interest_weights"].sum(1),torch.ones(2),atol=1e-6))
        self.assertGreater(float(b.contribution_predictor.hidden.weight.grad.abs().sum()),0)
        b.eval();f=feed();f["history_items"]=torch.tensor([[1,0,0],[3,0,0]]);f["lengths"]=torch.ones(2,dtype=torch.long)
        a=b(f,True);f["history_items"]=f["history_items"][:,:1];c=b(f,True)
        self.assertTrue(torch.allclose(a["prediction"],c["prediction"],atol=1e-8))

    def test_k2_coalition_math_and_permutation(self):
        t=CoalitionTeacher(interests=2).double()
        scores=torch.tensor([[[3.,.5],[-1.,0.]],[[0.,0.],[0.,0.]]],dtype=torch.float64)
        phi,q,u=t(scores)
        self.assertEqual(tuple(u.shape),(2,4))
        self.assertTrue(torch.equal(u[:,0],torch.zeros(2,dtype=torch.float64)))
        margins=scores[:,:,0]-scores[:,:,1]
        self.assertTrue(torch.equal(u[:,1],torch.sigmoid(margins[:,0])-.5))
        self.assertTrue(torch.equal(u[:,2],torch.sigmoid(margins[:,1])-.5))
        self.assertTrue(torch.equal(u[:,3],torch.sigmoid(margins.mean(1))-.5))
        ref=torch.stack([.5*u[:,1]+.5*(u[:,3]-u[:,2]),.5*u[:,2]+.5*(u[:,3]-u[:,1])],1)
        self.assertTrue(torch.allclose(phi,ref,atol=1e-14))
        self.assertTrue(torch.allclose(phi.sum(1),u[:,3],atol=1e-14))
        ph2,q2,_=t(scores[:,[1,0]])
        self.assertTrue(torch.allclose(phi[:,[1,0]],ph2,atol=1e-14))
        self.assertTrue(torch.allclose(q[:,[1,0]],q2,atol=1e-14))

    def test_k4_legacy_teacher_bit_exact(self):
        # Reference is the previous K4 formula and float32 coefficient construction.
        masks=torch.tensor([[bool(s & (1<<k)) for k in range(4)] for s in range(16)],dtype=torch.float32)
        coalition=masks/masks.sum(-1,keepdim=True).clamp_min(1)
        coeff=torch.zeros(16,4)
        for k in range(4):
            for s in range(16):
                if s & (1<<k):continue
                n=int(masks[s].sum())
                factor=math.factorial(n)*math.factorial(3-n)/math.factorial(4)
                coeff[s|(1<<k),k]+=factor;coeff[s,k]-=factor
        t=CoalitionTeacher(interests=4)
        self.assertTrue(torch.equal(t.coalition,coalition))
        self.assertTrue(torch.equal(t.coeff,coeff))
        for dtype in [torch.float32,torch.float64]:
            scores=torch.randn(8,4,2,generator=torch.Generator().manual_seed(52),dtype=dtype)
            u=torch.sigmoid((scores[:,:,0]-scores[:,:,1])@coalition.to(dtype).T)-.5
            phi=u@coeff.to(dtype);q=torch.softmax(phi/.1,-1)
            actual=t.to(dtype)(scores)
            for a,b in zip(actual,(phi,q,u)):self.assertTrue(torch.equal(a,b))

    def test_nokl_same_capacity_weights_and_serving_path(self):
        a,b=make(weight=.01),make(weight=0)
        self.assertEqual(a.count_variables(),b.count_variables())
        for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[k]),k)
        with torch.no_grad():
            a.contribution_predictor.output.weight.normal_(0,.02)
            b.load_state_dict(a.state_dict())
        torch.manual_seed(83);oa=a(feed(),True)
        with patch.object(b.coalition_teacher,"forward",side_effect=AssertionError("noKL teacher")):
            torch.manual_seed(83);ob=b(feed(),True)
        self.assertTrue(torch.equal(oa["prediction"],ob["prediction"]))
        self.assertTrue(torch.equal(oa["interest_weights"],ob["interest_weights"]))
        self.assertNotIn("_cicl_teacher",ob)
        loss=b.loss(ob);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(b.contribution_predictor.output.weight.grad.abs().sum()),0)
        self.assertGreater(float(b.contribution_predictor.hidden.weight.grad.abs().sum()),0)
        for m in [a,b]:
            m.eval()
            with patch.object(m.coalition_teacher,"forward",side_effect=AssertionError("eval teacher")):
                old=m(feed(),True);f=feed();f["item_id"]=f["item_id"][:,[1,0]];new=m(f,True)
            self.assertTrue(torch.equal(old["user_vector"],new["user_vector"]))

    def test_k2_model_zero_path_features_and_finite_loss(self):
        a,b=make(True,interests=2),make(interests=2)
        torch.manual_seed(63);oa=a(feed(),True)
        torch.manual_seed(63);ob=b(feed(),True)
        self.assertTrue(torch.equal(oa["prediction"],ob["prediction"]))
        self.assertEqual(tuple(ob["_cicl_teacher"].shape),(2,2))
        self.assertEqual(b.contribution_predictor.hidden.in_features,322)
        self.assertEqual(sum(p.numel() for p in b.contribution_predictor.parameters()),20736)
        loss=b.loss(ob);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in b.parameters() if p.grad is not None))
        b.eval()
        with torch.no_grad():b.contribution_predictor.output.weight.normal_(0,.02)
        o=b(feed(),True)
        p=b.contribution_predictor
        orig=p(o["interest_vectors"],o["attention_maps"],o["history_vectors"],feed()["lengths"],b.cicl_q[feed()["history_items"]])
        perm=p(o["interest_vectors"][:,[1,0]],o["attention_maps"][:,[1,0]],o["history_vectors"],feed()["lengths"],b.cicl_q[feed()["history_items"]])
        self.assertTrue(torch.allclose(orig[:,[1,0]],perm,atol=1e-7))

if __name__=="__main__":unittest.main(verbosity=2)
