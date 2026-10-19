"""Focused tests for one RelativeSemantic adaptation; no Dev/Test or full epochs."""
import os,sys,argparse,copy,io,json,time,hashlib,pickle,signal,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
if "--gpu" in sys.argv:os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[sys.argv.index("--gpu")+1]
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models.sequential.LLMMIRecASPCFRefineControls import LLMMIRecASPCFUserRefine as Base
from models.sequential.LLMMIRecASPCFRelativeSemantic import LLMMIRecASPCFRelativeSemantic as Model,RelativeSemanticAttention,history_semantic_kernel
from models.sequential.llmmi_components import QueryMultiInterestExtractor
from models.sequential.LLMMIRecASPCFSampledSoftmax import masked_softmax_loss
from tools.test_llmmirec_dualview_refine import TABLE,feed
torch.set_num_threads(2)
def build(cls=Model,K=4,enabled=1):
    a=cls.parse_model_args(argparse.ArgumentParser()).parse_args([])
    a.device=torch.device("cpu");a.model_path="unused";a.buffer=0;a.K=K;a.history_max=20
    a.llm_emb_path="synthetic";a.relation_sample_size=5
    if cls is Model:a.relative_semantic_enabled=enabled
    c=SimpleNamespace(n_users=4,n_items=60,train_clicked_set={1:{1,2,6},2:{3,4,7},3:{5,8}})
    torch.manual_seed(42)
    with patch("models.sequential.LLMMIRecASPCF.load_llm_table",return_value=TABLE):
        return cls(a,c)
class Tests(unittest.TestCase):
    def test_01_exact_identity_initialization_loss_shared_gradients_and_rng(self):
        baseline=build(Base);rng=torch.get_rng_state().clone()
        for enabled in [0,1]:
            m=build(enabled=enabled);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(m.count_variables()-baseline.count_variables(),1)
            self.assertEqual(float(m.extractor.alpha),0)
            for n,p in baseline.state_dict().items():self.assertTrue(torch.equal(p,m.state_dict()[n]),n)
            for training in [False,True]:
                baseline.train(training);m.train(training)
                torch.manual_seed(9);a=baseline(feed(),True)
                torch.manual_seed(9);b=m(feed(),True)
                for k in ["prediction","attention_maps","interest_vectors","interest_weights","candidate_vectors","user_vector"]:
                    self.assertTrue(torch.equal(a[k],b[k]),k)
                if training:
                    for k in ["_ss_scores","_ss_allowed","_relation_ids"]:self.assertTrue(torch.equal(a[k],b[k]),k)
                    baseline.zero_grad();m.zero_grad()
                    la=baseline.loss(a);lb=m.loss(b);self.assertTrue(torch.equal(la,lb))
                    self.assertTrue(torch.equal(a["loss_relation"],b["loss_relation"]))
                    la.backward();lb.backward()
                    for n,p in baseline.named_parameters():self.assertTrue(torch.equal(p.grad,dict(m.named_parameters())[n].grad),n)
                    baseline.actions_after_train();m.actions_after_train()
    def test_02_K_padding_short_zero_semantic_and_empty(self):
        for K in [1,2,4]:
            original=QueryMultiInterestExtractor(K,64,64)
            ex=RelativeSemanticAttention(original)
            h=torch.randn(5,20,64);z=torch.randn(5,20,512);lens=torch.tensor([0,1,2,3,20])
            valid=torch.arange(20)[None]<lens[:,None]
            with torch.no_grad():ex.alpha.fill_(1.2)
            a=ex(h,lens,z,return_details=True)
            hh=h.masked_fill(~valid[...,None],999);zz=z.masked_fill(~valid[...,None],-999)
            b=ex(hh,lens,zz,return_details=True)
            self.assertTrue(torch.equal(a[0],b[0]));self.assertEqual(a[0].shape,(5,K,64))
            self.assertTrue(torch.isfinite(a[0]).all());self.assertEqual(float(a[1][0].abs().sum()),0)
            self.assertEqual(float(a[2]["semantic_propagation"][1].abs().sum()),0)
            zero=ex(h,lens,torch.zeros_like(z),return_details=True)
            base=original(h,lens)
            self.assertTrue(torch.equal(zero[0],base[0]))
            if K==1:
                self.assertEqual(float(a[2]["semantic_propagation"].abs().sum()),0)
                self.assertTrue(torch.equal(a[0],base[0]))
    def test_03_formula_centering_bounds_identical_semantics_and_permutation(self):
        ex=RelativeSemanticAttention(QueryMultiInterestExtractor(4,64,64)).double()
        h=torch.randn(2,5,64,dtype=torch.double);z=torch.randn(2,5,512,dtype=torch.double);lens=torch.tensor([5,3])
        with torch.no_grad():ex.alpha.fill_(.7)
        v,a,d=ex(h,lens,z,return_details=True)
        exact=torch.einsum("bkj,btj->bkt",d["relative_attention"],d["semantic_kernel"])
        torch.testing.assert_close(exact,d["semantic_propagation"],atol=1e-15,rtol=1e-12)
        self.assertLess(float(d["relative_attention"].sum(1).abs().max()),1e-14)
        self.assertLessEqual(float(d["semantic_propagation"].abs().max()),2+1e-12)
        same=ex(h,lens,torch.ones_like(z),return_details=True)[2]
        torch.testing.assert_close(same["semantic_propagation"],-same["relative_attention"],atol=1e-14,rtol=1e-10)
        perm=torch.tensor([2,0,3,1]);other=copy.deepcopy(ex)
        with torch.no_grad():other.query.copy_(ex.query[perm])
        vv,aa=other(h,lens,z)
        torch.testing.assert_close(vv,v[:,perm],atol=1e-14,rtol=1e-10)
        torch.testing.assert_close(aa,a[:,perm],atol=1e-14,rtol=1e-10)
    def test_04_alpha_task_derivative_finite_difference_and_fixed_semantic(self):
        ex=RelativeSemanticAttention(QueryMultiInterestExtractor(2,64,64)).double()
        h=torch.randn(2,4,64,dtype=torch.double,requires_grad=True)
        z=torch.randn(2,4,512,dtype=torch.double,requires_grad=True);lens=torch.tensor([4,3])
        v,a=ex(h,lens,z);loss=v[:,0].square().sum()
        grad=torch.autograd.grad(loss,[ex.alpha,h,z],allow_unused=True)
        self.assertIsNone(grad[2]);self.assertGreater(abs(float(grad[0])),0)
        scores=[]
        for x in [-1e-4,1e-4]:
            with torch.no_grad():ex.alpha.fill_(x);scores.append(float(ex(h,lens,z)[0][:,0].square().sum()))
        numerical=(scores[1]-scores[0])/2e-4
        self.assertAlmostEqual(float(grad[0]),numerical,places=7)
        self.assertTrue(torch.isfinite(grad[1]).all())
    def test_05_candidate_user_isolation_mask_loss_and_encoder_reuse(self):
        m=build().eval()
        with torch.no_grad():m.extractor.alpha.fill_(.8)
        f=feed()
        with patch.object(m.item_encoder,"forward",wraps=m.item_encoder.forward) as encoder:a=m(f,True)
        self.assertEqual(encoder.call_count,2)
        altered=copy.deepcopy(f);altered["item_id"]=altered["item_id"].flip(1);b=m(altered,True)
        self.assertTrue(torch.equal(a["user_vector"],b["user_vector"]))
        self.assertTrue(torch.equal(a["prediction"].flip(1),b["prediction"]))
        single=m({k:v[:1] for k,v in f.items()},True)
        torch.testing.assert_close(a["prediction"][:1],single["prediction"],atol=1e-8,rtol=1e-5)
        self.assertNotIn("_ss_scores",a);self.assertNotIn("_relation_ids",a)
        f["item_id"]=torch.arange(1001)[None].expand(3,-1)%59+1
        self.assertEqual(m(f)["prediction"].shape,(3,1001))
        base=build(Base).train();m=build().train();f=feed();f["item_id"][2,0]=7
        torch.manual_seed(4);a=base(f,True)
        torch.manual_seed(4);b=m(f,True)
        self.assertTrue(torch.equal(a["_ss_allowed"],b["_ss_allowed"]))
        self.assertTrue(b["_ss_allowed"].diagonal().all());self.assertFalse(b["_ss_allowed"][1,2])
        self.assertIs(Model.loss,Base.loss);self.assertIs(Model._ranking_loss,Base._ranking_loss)
        self.assertIs(Model._compute_relation_loss,Base._compute_relation_loss)
        # No negative batch slots: exactly graph-connected zero ranking loss.
        scores=torch.randn(2,2,requires_grad=True);loss,valid=masked_softmax_loss(scores,torch.eye(2,dtype=torch.bool),1)
        self.assertEqual(float(loss),0);loss.backward();self.assertTrue(torch.isfinite(scores.grad).all())
    def test_06_Train_ordinal_prefix_and_checkpoint_roundtrip(self):
        m=build();d=object.__new__(m.Dataset);d.model=m;d.phase="train"
        d.corpus=SimpleNamespace(user_his={1:[(i+1,i) for i in range(50)]},n_items=60)
        d.data=dict(user_id=[1],item_id=[36],neg_items=[np.array([55])],position=[35])
        f=d._get_feed_dict(0);self.assertEqual(f["history_items"].tolist(),list(range(16,36)))
        self.assertNotIn(36,f["history_items"])
        with torch.no_grad():m.extractor.alpha.fill_(.4)
        a=m.eval()(feed())["prediction"];mem=io.BytesIO();torch.save(m.state_dict(),mem);mem.seek(0)
        n=build();n.load_state_dict(torch.load(mem,weights_only=True))
        self.assertTrue(torch.equal(a,n.eval()(feed())["prediction"]))
        self.assertNotIn("item_encoder.llm_table",m.state_dict())
        with patch.object(Path,"read_text",return_value=json.dumps({**m.checkpoint_metadata(),"enabled":False})):
            with self.assertRaises(ValueError):m.load_model("unused")
    def test_07_constant_offdiagonal_identity(self):
        # Unit vectors sharing sqrt(c) and distinct orthogonal residuals:
        # actual normalized off-diagonal kernel equals c.
        for K in [1,2,4]:
            ex=RelativeSemanticAttention(QueryMultiInterestExtractor(K,64,64)).double()
            h=torch.arange(2*5*64,dtype=torch.double).reshape(2,5,64)/1000
            lengths=torch.tensor([5,3])
            for c in [0.,.4,1.]:
                z=torch.zeros(2,5,512,dtype=torch.double)
                z[:,:,0]=c**.5
                for j in range(5):z[:,j,j+1]=(1-c)**.5
                details=ex(h,lengths,z,return_details=True)[2]
                torch.testing.assert_close(details["semantic_propagation"],-c*details["relative_attention"],atol=1e-14,rtol=1e-10)

def train_fixture(c,count=1024):
    df=c.data_df["train"];ix=np.flatnonzero(df.position.to_numpy()>0);rng=np.random.default_rng(42)
    ix=rng.choice(ix,count,replace=False);hs=[];lens=[];items=[];users=[]
    for i in ix:
        row=df.iloc[int(i)];u=int(row.user_id);pos=int(row.position)
        prefix=c.user_his[u][max(0,pos-20):pos]
        assert prefix and all(t<=row.time for _,t in prefix)
        h=[int(j) for j,_ in prefix];negative=int(rng.integers(1,c.n_items))
        while negative in c.train_clicked_set[u]:negative=int(rng.integers(1,c.n_items))
        hs.append(h+[0]*(20-len(h)));lens.append(len(h));items.append([int(row.item_id),negative]);users.append(u)
    b={k:torch.tensor(v) for k,v in dict(history_items=hs,lengths=lens,item_id=items,user_id=users).items()}
    digest=hashlib.sha256()
    for k,v in sorted(b.items()):digest.update(k.encode());digest.update(v.numpy().tobytes())
    return b,dict(samples=count,users=len(set(users)),length_counts={str(j):lens.count(j) for j in sorted(set(lens))},batch_sha256=digest.hexdigest())
def real(dataset,output):
    signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError("170s budget")));signal.alarm(170)
    started=time.perf_counter()
    summary=ROOT/"new_log/aspcf_refine_controls"/dataset/"user_refine_k4_h20_seed42_20261013T020946Z_attribution/summary.json"
    s=json.loads(summary.read_text());cp=ROOT/s["checkpoint"]
    sha=lambda:hashlib.sha256(cp.read_bytes()).hexdigest()
    assert sha()==s["checkpoint_sha256"]
    corpus=pickle.load(open(ROOT/"data"/dataset/"SeqReader.pkl","rb"));cpu,sampling=train_fixture(corpus)
    b={k:v.cuda() for k,v in cpu.items()}
    def make(cls,enabled=1):
        parser=cls.parse_model_args(argparse.ArgumentParser());a,_=parser.parse_known_args(s["common_command"])
        a.device=torch.device("cuda:0");a.model_path=str(ROOT/"new_model/relative_semantic_unused_test.pt");a.buffer=0
        if cls is Model:
            a.relative_semantic_enabled=enabled;a.relative_semantic_base_checkpoint=str(cp)
        torch.manual_seed(42);m=cls(a,corpus).cuda()
        if cls is Base:m.load_state_dict(torch.load(cp,map_location="cpu",weights_only=True))
        return m
    base=make(Base);off=make(Model,0);on=make(Model,1)
    for m in [off,on]:
        for n,p in base.state_dict().items():assert torch.equal(p,m.state_dict()[n]),n
    def task(m,bb,seed=55):
        m.train();torch.manual_seed(seed);o=m(bb,True)
        loss,valid=m._ranking_loss(o["_ss_scores"],o["_ss_allowed"])
        return o,loss
    o0,l0=task(base,b);oo,lo=task(off,b);oi,li=task(on,b)
    for key in ["prediction","_ss_scores","_ss_allowed","_relation_ids","attention_maps"]:
        assert torch.equal(o0[key],oo[key]) and torch.equal(o0[key],oi[key]),key
    assert torch.equal(l0,lo) and torch.equal(l0,li)
    initial=dict(forward_loss_exact=True,shared_parameter_exact=True,
                 baseline_parameters=base.count_variables(),new_parameters=on.count_variables(),
                 baseline_sha256=sha(),checkpoint=str(cp.relative_to(ROOT)),
                 historical_dev=s["dev_full"])
    del o0,oo,oi,l0,lo,li,base
    gradients=[]
    for j in range(3):
        bb={k:v[j*256:(j+1)*256] for k,v in b.items()}
        out,rank=task(on,bb,seed=550+j)
        g=torch.autograd.grad(rank,[on.extractor.alpha,on.extractor.query,on.extractor.Wk.weight],retain_graph=False)
        assert all(torch.isfinite(x).all() for x in g)
        gradients.append(dict(batch=j,rows=256,ranking=float(rank),alpha_gradient=float(g[0]),query_task_gradient=float(g[1].norm()),
                       key_task_gradient=float(g[2].norm()),relative_attention_rms=float(out["relative_attention"].square().mean().sqrt()),
                       propagation_rms=float(out["semantic_propagation"].square().mean().sqrt()),
                       propagation_max=float(out["semantic_propagation"].abs().max())))
    assert any(x["alpha_gradient"]!=0 for x in gradients),"alpha has no observed task gradient; stop before any continuation"
    timing={}
    for name,m in [("C0_off",off),("C1_on",on)]:
        optimizer=torch.optim.Adam(m.customize_parameters(),lr=.004 if dataset=="beauty" else .001,weight_decay=1e-6)
        losses=[]
        def step(seed):
            optimizer.zero_grad();m.train();torch.manual_seed(seed);o=m(b);loss=m.loss(o);loss.backward();optimizer.step();losses.append(loss.detach())
        step(42000);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
        for j in range(3):step(42001+j)
        torch.cuda.synchronize()
        assert torch.isfinite(torch.stack(losses)).all()
        assert all(torch.isfinite(p).all() for p in m.parameters())
        timing[name]=dict(step_seconds=(time.perf_counter()-t)/3,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                          peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,alpha_after_four_steps=float(m.extractor.alpha))
        m.actions_after_train()
    result=dict(dataset=dataset,physical_gpu=1,GPU=torch.cuda.get_device_name(0),sampling=sampling,
                initial=initial,pretrained_three_batch_task_gradients=gradients,timing=timing,
                temporary_steps_per_control=4,parameters_saved=False,Dev_evaluated=False,Test_evaluated=False,
                optimizer="both fresh Adam, original bias-exempt weight decay; not exact historical optimizer resume",
                seconds=time.perf_counter()-started)
    assert sha()==s["checkpoint_sha256"]
    p=Path(output);assert p.resolve().is_relative_to(ROOT) and not p.exists()
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=="__main__":
    if "--real" in sys.argv:
        p=argparse.ArgumentParser();p.add_argument("--real",choices=["beauty","ml-1m"]);p.add_argument("--gpu",default="1");p.add_argument("--output",required=True)
        a=p.parse_args();assert a.gpu=="1";real(a.real,a.output)
    else:unittest.main(verbosity=2)
