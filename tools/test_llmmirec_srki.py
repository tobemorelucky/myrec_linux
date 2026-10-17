"""Bounded SRKI tests. No main.py training, Dev/Test dataset, cache or checkpoint writes."""
import argparse, copy, io, json, math, os, pickle, signal, sys, time, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
# Physical GPU selection is done before importing torch.
if "--gpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.BaseModel import GeneralModel
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF as Base
from models.sequential.LLMMIRecASPCFSRKI import (
    LLMMIRecASPCFSRKI as Model, residual_support, SRKIExtractor, ContentAggregator)
torch.set_num_threads(2)

def make(K=2, source="llm", base=False, corpus=None, dataset=None, device="cpu"):
    cls = Base if base else Model
    args = cls.parse_model_args(argparse.ArgumentParser()).parse_args([])
    args.K = K; args.device = torch.device(device); args.buffer = 0; args.dropout = 0.0
    args.model_path = str(ROOT/"new_model"/"srki_unused_test_path.pt")
    args.llm_emb_path = "synthetic" if dataset is None else str(ROOT/"data"/dataset/"handled/llm_table_pca1536.pkl")
    args.relation_sample_size = 128
    if not base: args.srki_kernel_source = source
    torch.manual_seed(42)
    if corpus is None:
        gen = torch.Generator().manual_seed(19)
        table = torch.randn(80, 1536, generator=gen); table[0] = 0
        with patch("models.sequential.LLMMIRecASPCF.load_llm_table", return_value=table):
            model = cls(args, SimpleNamespace(n_users=10, n_items=80))
    else:
        args.dropout = 0.1
        model = cls(args, corpus)
    return model.to(device)

def feed():
    return dict(history_items=torch.tensor([[1,2,3,4],[4,2,0,0],[5,0,0,0]]),
                lengths=torch.tensor([4,2,1]), item_id=torch.tensor([[9,10],[11,12],[13,14]]))

class Tests(unittest.TestCase):
    def test_01_cholesky_psd_reconstruction_and_energy(self):
        torch.manual_seed(1)
        z=torch.randn(3,7,9,dtype=torch.double); mask=torch.arange(7)[None]<torch.tensor([7,4,1])[:,None]
        for K in [1,2,4]:
            x=residual_support(z,mask,K,True); F=x["factors"]
            torch.testing.assert_close(x["kernel"],F.transpose(1,2)@F+x["residuals"][:,-1],atol=2e-14,rtol=2e-14)
            self.assertGreater(torch.linalg.eigvalsh(x["residuals"]).min().item(),-1e-12)
            self.assertLess((x["residual_energy"][:,1:]-x["residual_energy"][:,:-1]).max().item(),1e-12)
            self.assertTrue(torch.equal(x["pivots"][:,0],torch.tensor([6,3,0])))
            torch.testing.assert_close(x["base_attention"].sum(-1),x["interest_mask"].double())
    def test_02_duplicate_negative_and_near_zero(self):
        z=torch.ones(2,3,5,dtype=torch.double);mask=torch.tensor([[1,1,1],[1,0,0]],dtype=torch.bool)
        x=residual_support(z,mask,4,True);self.assertTrue(torch.equal(x["interest_mask"].sum(1),torch.ones(2,dtype=torch.long)))
        negative=torch.tensor([[[1.,0.],[-1.,0.]]],dtype=torch.double)
        out=residual_support(negative,torch.ones(1,2,dtype=torch.bool),4,True)
        self.assertEqual(out["interest_mask"].sum().item(),1)
        # Known limitation, not silently 'fixed': negative factor support is discarded.
        self.assertEqual(out["base_attention"][0,:,0].sum().item(),0)
        for z0 in [torch.zeros_like(z),torch.ones_like(z)*1e-12]:
            x=residual_support(z0,mask,4,True);self.assertTrue(x["fallback_used"].all())
            torch.testing.assert_close(x["base_attention"][:,0].sum(-1),torch.ones(2,dtype=torch.double))
    def test_03_padding_lengths_and_K(self):
        for K in [1,2,4]:
            torch.manual_seed(2); ex=SRKIExtractor(K).double()
            h=torch.randn(4,20,64,dtype=torch.double);z=torch.randn(4,20,12,dtype=torch.double)
            valid=torch.arange(20)[None]<torch.tensor([1,2,3,20])[:,None]
            a=ex(h,z,valid,True);hh=h.clone();zz=z.clone();hh[~valid]=1e4;zz[~valid]=-1e4
            b=ex(hh,zz,valid,True)
            torch.testing.assert_close(a["interest_vectors"],b["interest_vectors"],atol=0,rtol=0)
            self.assertEqual(a["interest_vectors"].shape,(4,K,64))
            self.assertTrue((a["interest_mask"].sum(-1)<=torch.tensor([1,2,3,20])).all())
            empty=ex(h[:,:0],z[:,:0],valid[:,:0],True)
            self.assertEqual(empty["interest_vectors"].abs().sum().item(),0)
            self.assertFalse(empty["interest_mask"].any())
    def test_04_interest_weighting_and_K1_head(self):
        ex=SRKIExtractor(2).double()
        with torch.no_grad():ex.importance.weight.zero_()
        h=torch.randn(2,5,64,dtype=torch.double);z=torch.randn(2,5,6,dtype=torch.double);valid=torch.ones(2,5,dtype=torch.bool)
        a=ex(h,z,valid,True)
        torch.testing.assert_close(a["attention_maps"],a["base_attention"])
        loss=(a["interest_vectors"]*torch.randn_like(a["interest_vectors"])).sum();loss.backward()
        self.assertGreater(ex.importance.weight.grad.norm().item(),0)
        head=ContentAggregator(64,1).double();v=torch.randn(2,1,64,dtype=torch.double)
        u,w=head(v,h[:,-1],torch.ones(2,1,dtype=torch.bool))
        torch.testing.assert_close(u,v[:,0],atol=0,rtol=0);self.assertTrue(torch.equal(w,torch.ones(2,1,dtype=torch.double)))
        self.assertEqual(sum(p.numel() for p in head.parameters()),0)
    def test_05_head_gradient_formula_and_slot_permutation(self):
        torch.manual_seed(3);head=ContentAggregator().double()
        v=torch.randn(2,3,64,dtype=torch.double,requires_grad=True);last=torch.randn(2,64,dtype=torch.double)
        mask=torch.ones(2,3,dtype=torch.bool);g=torch.randn(2,64,dtype=torch.double)
        u,w=head(v,last,mask);grad=torch.autograd.grad((u*g).sum(),v)[0]
        q=head.Wq(last)/8
        expected=w[...,None]*g[:,None]+w[...,None]*((v-u[:,None])*g[:,None]).sum(-1,keepdim=True)*q[:,None]
        torch.testing.assert_close(grad,expected,atol=1e-12,rtol=1e-12)
        order=torch.tensor([2,0,1]);up,_=head(v[:,order],last,mask[:,order]);torch.testing.assert_close(u,up)
        same=v[:,:1].expand(-1,3,-1).detach().clone().requires_grad_()
        uu,ww=head(same,last,mask);gg=torch.autograd.grad((uu*g).sum(),same)[0]
        torch.testing.assert_close(gg,ww[...,None]*g[:,None],atol=1e-12,rtol=1e-12)
    def test_06_sources_initialization_and_permutation(self):
        sources=[make(source=s) for s in ["llm","aspcf","permuted_llm"]]
        for m in sources[1:]:
            for key,value in sources[0].state_dict().items():self.assertTrue(torch.equal(value,m.state_dict()[key]))
        m=sources[-1];p=m.srki_semantic_permutation
        self.assertEqual(p[0].item(),0);self.assertTrue(torch.equal(p[1:].sort().values,torch.arange(1,80)))
        b=feed();_,z,_=m.encode_history(b["history_items"],b["lengths"])
        torch.testing.assert_close(z,m.item_encoder.llm_table[p[b["history_items"]]][...,:512])
        for m in sources:
            calls=[]
            hook=m.item_encoder.register_forward_hook(lambda *args:calls.append(1))
            out=m.eval()(b,True);hook.remove()
            self.assertEqual(len(calls),2)
            self.assertTrue(torch.isfinite(out["prediction"]).all())
    def test_07_targets_batch_and_candidates_isolated(self):
        m=make().eval();b=feed();x=m(b,True);other=copy.deepcopy(b);other["item_id"]=other["item_id"].flip(1)
        y=m(other,True);torch.testing.assert_close(x["interest_vectors"],y["interest_vectors"],atol=0,rtol=0)
        torch.testing.assert_close(x["prediction"].flip(1),y["prediction"],atol=0,rtol=0)
        single=m({k:v[:1] for k,v in b.items()},True)
        torch.testing.assert_close(single["prediction"],x["prediction"][:1],atol=1e-10,rtol=1e-5)
        many=copy.deepcopy(b);many["item_id"]=b["item_id"][:,:1].expand(-1,1001)
        out=m(many);self.assertEqual(out["prediction"].shape,(3,1001))
        torch.testing.assert_close(out["prediction"],out["prediction"][:,:1].expand(-1,1001))
    def test_08_original_loss_and_shared_encoder(self):
        m=make();base=make(base=True);base.item_encoder.load_state_dict(m.item_encoder.state_dict())
        b=feed();torch.testing.assert_close(m.item_encoder(b["history_items"]),base.item_encoder(b["history_items"]),atol=0,rtol=0)
        out=m.train()(b);r=GeneralModel.loss(m,out);rel=base._compute_relation_loss(out["_relation_ids"])
        self.assertIs(Model.loss,Base.loss);self.assertIs(Model._compute_relation_loss,Base._compute_relation_loss)
        torch.testing.assert_close(m.loss(out),r+.01*rel,atol=0,rtol=0)
        m.loss(out).backward();self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
    def test_09_state_checkpoint_and_metadata_guard(self):
        m=make();b=feed();x=m.eval()(b)["prediction"];mem=io.BytesIO();torch.save(m.state_dict(),mem);mem.seek(0)
        n=make();n.load_state_dict(torch.load(mem));torch.testing.assert_close(n.eval()(b)["prediction"],x,atol=0,rtol=0)
        self.assertNotIn("item_encoder.llm_table",m.state_dict())
        self.assertNotIn("srki_semantic_permutation",m.state_dict())
        wrong=make(K=4)
        with patch.object(Path,"read_text",return_value=json.dumps(m.checkpoint_metadata())):
            with self.assertRaises(ValueError):wrong.load_model("unused")
    def test_10_kernel_does_not_backpropagate(self):
        h=torch.randn(2,4,64,requires_grad=True);z=torch.randn(2,4,8,requires_grad=True)
        ex=SRKIExtractor(2);o=ex(h,z,torch.ones(2,4,dtype=torch.bool),True);o["interest_vectors"].square().sum().backward()
        self.assertIsNone(z.grad);self.assertGreater(h.grad.norm().item(),0)

def fixture(dataset,count):
    corpus=pickle.load(open(ROOT/"data"/dataset/"SeqReader.pkl","rb"))
    df=corpus.data_df["train"];eligible=np.flatnonzero(df.position.to_numpy()>0)
    rng=np.random.default_rng(42);ix=rng.choice(eligible,count,replace=False)
    hs=[];lens=[];items=[]
    for i in ix:
        row=df.iloc[int(i)];u=int(row.user_id);pos=int(row.position)
        prefix=corpus.user_his[u][max(0,pos-20):pos]
        assert prefix and all(t<=row.time for _,t in prefix)
        h=[int(j) for j,_ in prefix];assert len(h)<=20
        negative=int(rng.integers(1,corpus.n_items))
        while negative in corpus.train_clicked_set[u]:negative=int(rng.integers(1,corpus.n_items))
        hs.append(h+[0]*(20-len(h)));lens.append(len(h));items.append([int(row.item_id),negative])
    b={k:torch.tensor(v) for k,v in dict(history_items=hs,lengths=lens,item_id=items).items()}
    return corpus,b,dict(samples=count,users=int(df.iloc[ix].user_id.nunique()),length_counts={str(j):lens.count(j) for j in sorted(set(lens))},
                         protocol="cached ordinal Train prefix strictly before position; timestamps may tie; one original train-only-filtered negative")

def probe(model,b):
    model.eval();o=model(b,True);rank=GeneralModel.loss(model,o)
    params={"importance":model.extractor.importance.weight,"Wv":model.extractor.Wv.weight,
            "semantic_branch":model.item_encoder.semantic_branch[0].weight}
    params["Wv_bias"]=model.extractor.Wv.bias
    if model.aggregator.Wq is not None:
        params["head_Wq"]=model.aggregator.Wq.weight
        params["head_bias"]=model.aggregator.Wq.bias
    gradients=torch.autograd.grad(rank,list(params.values()),retain_graph=True,allow_unused=True)
    data={}
    for (name,p),g in zip(params.items(),gradients):
        task=0. if g is None else g.norm().item();decay=0. if "bias" in name else 1e-6*p.norm().item()
        data[name]=dict(task_gradient=task,decay_norm=decay,decay_to_task=decay/max(task,1e-30))
    v=o["interest_vectors"];vm=v.mean(1,keepdim=True)
    return dict(bpr=rank.item(),margin=(o["prediction"][:,0]-o["prediction"][:,1]).mean().item(),gradients=data,
                active_slot_counts={str(k):int((o["interest_mask"].sum(1)==k).sum()) for k in range(model.K+1)},
                fallback_users=int(o["fallback_used"].sum()),uncovered_history_fraction=float(((o["base_attention"].sum(1)==0)&(b["history_items"]>0)).sum()/(b["history_items"]>0).sum()),
                slot_relative_deviation=(v-vm).norm().item()/max(v.norm().item(),1e-30),
                weight_entropy=(-(o["interest_weights"]*o["interest_weights"].clamp_min(1e-30).log()).sum(1)).mean().item())

def real(dataset,gpu):
    signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError("170 second budget")));signal.alarm(170)
    start=time.perf_counter();assert gpu=="1","GPU0 disallowed in this bounded test"
    assert torch.cuda.is_available()
    corpus,cpu,sampling=fixture(dataset,1024);device="cuda:0";b={k:v.to(device) for k,v in cpu.items()}
    model=make(corpus=corpus,dataset=dataset,device=device);initial=probe(model,b)
    opt=torch.optim.Adam(model.customize_parameters(),lr=.004 if dataset=="beauty" else .001,weight_decay=1e-6)
    counts=[]
    for step in range(3):
        model.train();opt.zero_grad();out=model(b);loss=model.loss(out)
        assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        counts.append(sum(p.grad is not None and p.grad.norm().item()>0 for p in model.parameters()));opt.step()
    after=probe(model,b)
    timing={}
    # Timing baseline and SRKI, same batch/source encoder/optimizer definition. No full epoch.
    baseline=make(base=True,corpus=corpus,dataset=dataset,device=device)
    baseline.item_encoder.load_state_dict(model.item_encoder.state_dict())
    for name,m,optimizer in [("SRKI",model,opt),("C0",baseline,torch.optim.Adam(baseline.customize_parameters(),lr=.004 if dataset=="beauty" else .001,weight_decay=1e-6))]:
        m.train()
        def one():
            optimizer.zero_grad();o=m(b);ll=m.loss(o);ll.backward();optimizer.step()
        for _ in range(3):one()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
        for _ in range(5):one()
        torch.cuda.synchronize();elapsed=(time.perf_counter()-t)/5
        timing[name]=dict(step_seconds=elapsed,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                          peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
    # Model isolation on real Train histories, not Dev/Test.
    model.eval()
    with torch.no_grad():
        original=model(b,True);swapped={**b,"item_id":b["item_id"].flip(1)};changed=model(swapped,True)
        assert torch.equal(original["interest_vectors"],changed["interest_vectors"])
        torch.testing.assert_close(original["prediction"].flip(1),changed["prediction"],atol=0,rtol=0)
    result=dict(dataset=dataset,physical_gpu=gpu,gpu_name=torch.cuda.get_device_name(0),K=2,
                model_version=model.VERSION,parameters=model.count_variables(),C0_parameters=baseline.count_variables(),
                train_only=True,DevTest_executed=False,sampling=sampling,initial=initial,after_three_updates=after,
                after_eleven_updates=probe(model,b),
                finite_backward=True,nonzero_parameter_gradient_counts=counts,timing=timing,
                training_batches_per_epoch=math.ceil((len(corpus.data_df["train"])-int((corpus.data_df["train"].position==0).sum()))/1024),
                seconds=time.perf_counter()-start,caveat="fixed Train batch; no evidence of ranking improvement; allocated peak includes both resident models")
    signal.alarm(0);return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--real",choices=["beauty","ml-1m"]);p.add_argument("--gpu",default="1")
    args=p.parse_args()
    if args.real:print("RESULT_JSON "+json.dumps(real(args.real,args.gpu),allow_nan=False))
    else:
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        print("RESULT_JSON "+json.dumps(dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),successful=result.wasSuccessful())))
        sys.exit(0 if result.wasSuccessful() else 1)
