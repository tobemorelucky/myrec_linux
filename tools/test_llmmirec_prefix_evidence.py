"""Focused prefix-evidence tests. Synthetic CPU; optional bounded real Train GPU1.
No main.py execution, optimizer in unit tests only fixtures, no Dev/Test ranking.
"""
import argparse,copy,io,json,math,pickle,signal,sys,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.BaseModel import GeneralModel
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF as Base
from models.sequential.LLMMIRecASPCFPrefixEvidence import (
    LLMMIRecASPCFPrefixEvidence as Model,PrefixEvidenceTracker as Tracker)

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)

def make(variant="A2",K=4,base=False):
    cls=Base if base else Model
    args=cls.parse_model_args(argparse.ArgumentParser()).parse_args([])
    args.K=K;args.device=torch.device("cpu");args.model_path="unused";args.buffer=0
    args.llm_emb_path="synthetic";args.relation_sample_size=5
    if not base:args.prefix_variant=variant
    g=torch.Generator().manual_seed(1024)
    table=torch.randn(80,1536,generator=g);table[0]=0
    torch.manual_seed(42)
    with patch("models.sequential.LLMMIRecASPCF.load_llm_table",return_value=table):
        return cls(args,SimpleNamespace(n_users=8,n_items=80))

def feed():
    return dict(history_items=torch.tensor([[1,2,3,4],[4,2,0,0],[5,0,0,0]]),
                lengths=torch.tensor([4,2,1]),item_id=torch.tensor([[9,10],[11,12],[13,14]]))

class Tests(unittest.TestCase):
    def test_evidence_original_formula_covariance_and_neutral_cases(self):
        t=Tracker(4).double()
        p=torch.tensor([[[.8,.1,.05,.05],[.1,.7,.1,.1],[.4,.2,.3,.1]]],dtype=torch.double,requires_grad=True)
        r=torch.tensor([[.9,.1,.3]],dtype=torch.double);valid=torch.ones_like(r,dtype=torch.bool)
        d=t.evidence(p,r,valid)
        mu=r.mean(-1,keepdim=True);mass=p.sum(1)
        plus=(mu+(p*r[...,None]).sum(1))/(1+mass)
        minus=(mu+((1-p)*r[...,None]).sum(1))/(1+(1-p).sum(1))
        torch.testing.assert_close(d,(plus+1e-8).log()-(minus+1e-8).log(),atol=1e-12,rtol=1e-12)
        self.assertGreater(d.std().item(),.1)
        d.square().sum().backward();self.assertGreater(p.grad.abs().sum().item(),0)
        for pp,rr,vv in [(torch.full_like(p,.25),r,valid),(p,torch.ones_like(r)*.3,valid),
                         (p[:,:1],r[:,:1],valid[:,:1]),(p,r,torch.zeros_like(valid))]:
            self.assertLess(t.evidence(pp,rr,vv).abs().max().item(),1e-12)

    def test_reference_recurrence_and_K1(self):
        torch.manual_seed(7);t=Tracker(4).double()
        e=torch.randn(2,5,64,dtype=torch.double);z=torch.randn(2,5,9,dtype=torch.double)
        lens=torch.tensor([5,3]);out=t(e,lens,z)
        p=out["assignments"];kernel,pairs=t.semantic_kernel(z,torch.arange(5)[None]<lens[:,None])
        for b in range(2):
            prev=e.new_full((4,),.25)
            for j in range(int(lens[b])):
                if j:
                    r=kernel[b,j,:j];m=pairs[b,j,:j].double()
                    mu=(r*m).sum()/m.sum().clamp_min(1)
                    past=p[b,:j]
                    mp=(mu+(past*r[:,None]).sum(0))/(1+(past*m[:,None]).sum(0))
                    mm=(mu+((1-past)*r[:,None]).sum(0))/(1+((1-past)*m[:,None]).sum(0))
                    d=(mp+1e-8).log()-(mm+1e-8).log() if j>1 else torch.zeros_like(prev)
                else:d=torch.zeros_like(prev)
                content=t.We(e[b,j])@t.query.T/math.sqrt(64)
                expected=(content+d+.5*((.9*prev+.1/4).log())).softmax(-1)
                torch.testing.assert_close(expected,p[b,j],atol=1e-12,rtol=1e-12);prev=p[b,j]
        k=Tracker(1).double()
        a=k(e,lens,z,True);b=k(e,lens,z,False)
        self.assertTrue(torch.equal(a["interests"],b["interests"]))
        torch.testing.assert_close(a["interests"][:,0],torch.stack([k.Wv(e[i,:int(lens[i])]).mean(0) for i in range(2)]))
        self.assertEqual(int(torch.count_nonzero(a["evidence"])),0)

    def test_prefix_causality_padding_zero_semantics_and_empty(self):
        torch.manual_seed(11);t=Tracker(4).double()
        e=torch.randn(3,20,64,dtype=torch.double);z=torch.randn(3,20,512,dtype=torch.double)
        lens=torch.tensor([20,3,1]);a=t(e,lens,z)
        for n in [1,2,3]:
            short=t(e[:1,:n],torch.tensor([n]),z[:1,:n])
            torch.testing.assert_close(short["assignments"],a["assignments"][:1,:n],atol=1e-12,rtol=1e-12)
        changed_e=e.clone();changed_z=z.clone();changed_e[1,3:]=1e4;changed_z[1,3:]=-1e4
        b=t(changed_e,lens,changed_z)
        self.assertTrue(torch.equal(a["interests"][1],b["interests"][1]))
        for zz in [torch.zeros_like(z),torch.ones_like(z)*1e-12]:
            out=t(e,lens,zz);self.assertTrue(torch.isfinite(out["interests"]).all())
            self.assertEqual(int(torch.count_nonzero(out["evidence"])),0)
        empty=t(e[:,:0],torch.zeros(3,dtype=torch.long),z[:,:0])
        self.assertEqual(int(torch.count_nonzero(empty["interests"])),0)
        self.assertTrue(torch.equal(empty["final_state"],e.new_full((3,4),.25)))

    def test_slot_permutation_and_gradients(self):
        torch.manual_seed(8);a=Tracker(4).double();b=copy.deepcopy(a);order=torch.tensor([2,0,3,1])
        with torch.no_grad():b.query.copy_(a.query[order])
        e=torch.randn(2,6,64,dtype=torch.double,requires_grad=True);z=torch.randn(2,6,7,dtype=torch.double);length=torch.tensor([6,4])
        ea=e.detach().clone().requires_grad_();x=a(e,length,z);y=b(ea,length,z)
        torch.testing.assert_close(x["interests"][:,order],y["interests"],atol=1e-12,rtol=1e-12)
        coeff=torch.randn_like(x["interests"]);(x["interests"]*coeff).sum().backward();(y["interests"]*coeff[:,order]).sum().backward()
        torch.testing.assert_close(e.grad,ea.grad,atol=1e-11,rtol=1e-11)
        torch.testing.assert_close(a.query.grad[order],b.query.grad,atol=1e-11,rtol=1e-11)

    def test_shared_initialization_rng_loss_relation_and_checkpoint(self):
        base=make(base=True);rng=torch.get_rng_state();a=make("A1");b=make("A2")
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(a.count_variables(),b.count_variables())
        self.assertEqual(a.count_variables()-base.count_variables(),-5504)
        self.assertFalse(hasattr(a,"position_emb"));self.assertFalse(hasattr(a.extractor,"Wq"))
        for name,p in a.state_dict().items():self.assertTrue(torch.equal(p,b.state_dict()[name]))
        for prefix in ["item_encoder.","aggregator.","extractor.Wv."]:
            for name,p in base.state_dict().items():
                if name.startswith(prefix):self.assertTrue(torch.equal(p,a.state_dict()[name]),name)
        self.assertIs(Model.loss,Base.loss);self.assertIs(Model.Dataset,Base.Dataset)
        f=feed();torch.manual_seed(16);x=base(f,True);torch.manual_seed(16);y=b(f,True)
        self.assertTrue(torch.equal(x["_relation_ids"],y["_relation_ids"]))
        self.assertTrue(torch.equal(base._compute_relation_loss(x["_relation_ids"]),b._compute_relation_loss(y["_relation_ids"])))
        loss=b.loss(y)
        self.assertTrue(torch.equal(loss,GeneralModel.loss(b,y)+.01*b._compute_relation_loss(y["_relation_ids"])))
        loss.backward()
        for p in b.parameters():self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(b.extractor.query.grad.norm().item(),0)
        buf=io.BytesIO();torch.save(b.state_dict(),buf);buf.seek(0);c=make();c.load_state_dict(torch.load(buf,weights_only=True))
        b.eval();c.eval()
        with torch.no_grad():self.assertTrue(torch.equal(b(f)["prediction"],c(f)["prediction"]))

    def test_eval_isolation_lengths_and_actual_updates(self):
        for K in [1,4]:
            m=make(K=K);m.eval()
            for n in [0,1,2,3,20]:
                f=dict(history_items=torch.arange(1,21)[None],lengths=torch.tensor([n]),item_id=(torch.arange(1001)%79+1)[None])
                f["history_items"][:,n:]=0
                with torch.no_grad():a=m(f,True);g=copy.deepcopy(f);g["item_id"]=g["item_id"].flip(1);b=m(g,True)
                self.assertTrue(torch.equal(a["user_vector"],b["user_vector"]))
                self.assertTrue(torch.equal(a["prediction"].flip(1),b["prediction"]))
                self.assertTrue(torch.isfinite(a["prediction"]).all())
                if K==1:self.assertTrue(torch.equal(a["interest_weights"],torch.ones(1,1)))
        m=make();opt=torch.optim.Adam(m.parameters(),lr=.004,weight_decay=1e-6)
        q=m.extractor.query.detach().clone()
        for _ in range(2):
            opt.zero_grad();out=m(feed());m.loss(out).backward();opt.step()
        self.assertGreater((m.extractor.query-q).norm().item(),0)


    def test_batch_independence_and_no_semantic_evidence_model_equivalence(self):
        a=make("A1");b=make("A2");a.eval();b.eval()
        f=feed()
        with torch.no_grad():
            x=b(f,True)
            one=b({k:v[:1] for k,v in f.items()},True)
            g=copy.deepcopy(f);g["history_items"][1:]=7;other=b(g,True)
            torch.testing.assert_close(x["prediction"][:1],one["prediction"],atol=1e-7,rtol=1e-5)
            self.assertTrue(torch.equal(x["prediction"][:1],other["prediction"][:1]))
            a.item_encoder.llm_table[:,:512].zero_();b.item_encoder.llm_table[:,:512].zero_()
            self.assertTrue(torch.equal(a(f)["prediction"],b(f)["prediction"]))


def real():
    import os,subprocess
    signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError("real check >175s")))
    signal.alarm(175)
    gpu=subprocess.check_output(["nvidia-smi","--id=1","--query-gpu=uuid","--format=csv,noheader"],text=True).strip()
    apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"],text=True)
    if any(r.startswith(gpu+",") for r in apps.splitlines()):raise RuntimeError("GPU1 occupied")
    assert os.environ.get("CUDA_VISIBLE_DEVICES")=="1"
    result={};start=time.monotonic()
    for ds in ["beauty","ml-1m"]:
        corpus=pickle.load(open(ROOT/"data"/ds/"SeqReader.pkl","rb"));df=corpus.data_df["train"]
        eligible=np.flatnonzero(df.position.to_numpy()>0);rng=np.random.default_rng(20261016)
        indices=rng.choice(eligible,1024,replace=False);hs=[];lengths=[];cs=[]
        for idx in indices:
            r=df.iloc[int(idx)];u=int(r.user_id);pos=int(r.position)
            history=[int(v[0]) for v in corpus.user_his[u][max(0,pos-20):pos]]
            assert all(v[1]<=r.time for v in corpus.user_his[u][max(0,pos-20):pos])
            neg=int(rng.integers(1,corpus.n_items))
            while neg in corpus.train_clicked_set[u]:neg=int(rng.integers(1,corpus.n_items))
            hs.append(history+[0]*(20-len(history)));lengths.append(len(history));cs.append([int(r.item_id),neg])
        batch={k:torch.tensor(v,device="cuda") for k,v in dict(history_items=hs,lengths=lengths,item_id=cs).items()}
        result[ds]={}
        shared=None
        for variant in ["A1","A2"]:
            args=Model.parse_model_args(argparse.ArgumentParser()).parse_args([])
            args.device=torch.device("cpu");args.buffer=0;args.model_path="unused";args.prefix_variant=variant
            args.llm_emb_path=str(ROOT/"data"/ds/"handled/llm_table_pca1536.pkl")
            torch.manual_seed(42);m=Model(args,corpus)
            if shared is None:shared={k:v.clone() for k,v in m.state_dict().items()}
            else:assert all(torch.equal(v,shared[k]) for k,v in m.state_dict().items())
            m.cuda();torch.cuda.reset_peak_memory_stats();m.train()
            opt=torch.optim.Adam(m.parameters(),lr=.004 if ds=="beauty" else .001,weight_decay=1e-6)
            q=m.extractor.query.detach().clone();samples=[]
            torch.manual_seed(200)
            for step in range(3):
                t=time.monotonic();opt.zero_grad();out=m(batch,True)
                rank=GeneralModel.loss(m,out);loss=m.loss(out);loss.backward()
                grads={name:float(p.grad.norm()) for name,p in m.extractor.named_parameters() if p.grad is not None}
                assert grads and all(math.isfinite(v) for v in grads.values())
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
                p=out["prefix_assignments"].detach();d=out["semantic_evidence"].detach();mask=batch["history_items"]>0
                samples.append(dict(step=step,loss=float(loss.detach()),ranking=float(rank.detach()),gradients=grads,
                    p_std=float(p[mask].std()),max_evidence=float(d.abs().max()),
                    mean_slot_entropy=float(-(p[mask]*p[mask].clamp_min(1e-8).log()).sum(-1).mean())))
                opt.step();torch.cuda.synchronize();samples[-1]["step_seconds"]=time.monotonic()-t
            # Bounded steady-state timing: no per-batch item()/gradient scans.
            torch.cuda.synchronize();bench=time.monotonic()
            for _ in range(12):
                opt.zero_grad();bo=m(batch);bl=m.loss(bo);bl.backward();opt.step()
            torch.cuda.synchronize();steady=(time.monotonic()-bench)/12
            opt.zero_grad();probe=m(batch,True);ranking=GeneralModel.loss(m,probe)
            task=torch.autograd.grad(ranking,m.extractor.query)[0]
            task_norm=float(task.norm());decay_norm=1e-6*float(m.extractor.query.norm())
            m.eval()
            with torch.no_grad():final=m(batch,True)
            result[ds][variant]=dict(samples=samples,parameters=m.count_variables(),
                query_update_norm=float((m.extractor.query-q).norm()),
                max_A2_vs_A1_query_initialization_difference=0,
                max_final_evidence=float(final["semantic_evidence"].abs().max()),
                mean_interest_mass=final["interest_mass"].mean(0).cpu().tolist(),
                peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                train_batches=math.ceil(len(eligible)/1024),test_read=False,
                measured_optimizer_steps=15,steady_train_step_seconds=steady,
                steady_train_only_200_epoch_seconds=steady*math.ceil(len(eligible)/1024)*200,
                final_task_query_gradient_norm=task_norm,final_decay_query_norm=decay_norm,
                decay_to_task_query_ratio=decay_norm/max(task_norm,1e-30))
            del m,opt,out,final;torch.cuda.empty_cache()
    print("REAL_RESULT "+json.dumps(dict(results=result,elapsed_seconds=time.monotonic()-start,test_read=False)))
    signal.alarm(0)

if __name__=="__main__":
    if "--real" in sys.argv:real()
    else:unittest.main(verbosity=2)
