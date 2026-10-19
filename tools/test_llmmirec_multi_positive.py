"""Bounded synthetic + Train-only checks. No Reader/Runner/main, Dev/Test files or checkpoints."""
import argparse, copy, hashlib, itertools, json, os, signal, sys, time, types, unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF as Base
from models.sequential.LLMMIRecASPCFMultiPositive import LLMMIRecASPCFMultiPositive as Model, auxiliary_bpr, exclusive_assignment
RESULT={}

def args(K=2,mode="exclusive",weight=.01,device="cpu",ds="beauty"):
    a=Model.parse_model_args(argparse.ArgumentParser()).parse_args([])
    a.device=torch.device(device);a.buffer=0;a.model_path="unused";a.K=K;a.history_max=20
    a.mp_mode=mode;a.lambda_aux=weight;a.llm_emb_path=str(ROOT/"data"/ds/"handled/llm_table_pca1536.pkl")
    return a

def small_corpus():
    seq={1:[(1,0),(2,1),(2,1),(3,2),(4,3)],2:[(5,0),(6,1)]}
    rows=[dict(user_id=u,item_id=i,time=t,position=j,neg_items=[100]) for u,ls in seq.items() for j,(i,t) in enumerate(ls) if not (u==1 and j==4)]
    train=pd.DataFrame(rows)
    return types.SimpleNamespace(n_users=3,n_items=12102,user_his=seq,data_df={"train":train},
                                train_clicked_set={1:{1,2,3},2:{5,6}})

def model(c=None,K=2,mode="exclusive",weight=.01,cls=Model):
    torch.manual_seed(42);return cls(args(K,mode,weight),c or small_corpus())

def synthetic_batch(K=2):
    return dict(history_items=torch.tensor([[1,2,3],[4,0,0],[0,0,0],[6,7,0]]),lengths=torch.tensor([3,1,0,2]),
                item_id=torch.tensor([[8,100],[9,101],[10,102],[11,103]]),user_id=torch.tensor([1,1,2,2]),
                mp_extra_item=torch.tensor([12,13,14,15]),mp_extra_valid=torch.ones(4,dtype=torch.long),
                mp_assignment_key=torch.arange(4))

class Checks(unittest.TestCase):
    def test_01_base_equivalence_all_gradients(self):
        c=small_corpus();b=synthetic_batch();base=model(c,cls=Base);new=model(c,weight=0)
        self.assertEqual(base.count_variables(),new.count_variables());self.assertEqual(base.state_dict().keys(),new.state_dict().keys())
        self.assertTrue(all(torch.equal(v,new.state_dict()[k]) for k,v in base.state_dict().items()))
        values=[]
        for m in [base,new]:
            m.train();torch.manual_seed(99);o=m(b);loss=m.loss(o);loss.backward()
            values.append((o,loss.detach(),{n:None if p.grad is None else p.grad.clone() for n,p in m.named_parameters()},torch.get_rng_state()))
        x,y=values;self.assertTrue(torch.equal(x[0]["prediction"],y[0]["prediction"]));self.assertTrue(torch.equal(x[0]["_relation_ids"],y[0]["_relation_ids"]))
        self.assertTrue(torch.equal(x[1],y[1]));self.assertTrue(torch.equal(x[3],y[3]))
        for n,g in x[2].items():self.assertTrue((g is None and y[2][n] is None) or (g is not None and torch.equal(g,y[2][n])),n)
        RESULT["zero_aux_equivalence"]=dict(score=True,original_BPR_relation_loss=True,all_shared_gradients=True,RNG=True,new_parameters=0)

    def test_02_exact_matching_and_ties(self):
        for K in [2,4]:
            torch.manual_seed(18);d=torch.randn(8,K,2,requires_grad=True);a,t=exclusive_assignment(d,torch.arange(8));chosen=d.detach().gather(1,a[:,None]).sum((-1,-2))
            brute=torch.stack([d.detach()[:,i,0]+d.detach()[:,j,1] for i in range(K) for j in range(K) if i!=j],-1).max(-1).values
            self.assertTrue(torch.equal(chosen,brute));self.assertTrue((a[:,0]!=a[:,1]).all());self.assertFalse(a.requires_grad)
        a,t=exclusive_assignment(torch.zeros(16,2,2),torch.arange(16));self.assertEqual(int((a[:,0]==0).sum()),8);self.assertEqual(int((a[:,0]==1).sum()),8)
        RESULT["tied_synthetic_assignment"]=dict(rows=16,first_target_head0=8,first_target_head1=8)

    def test_03_equal_vectors_different_supervision(self):
        v=torch.zeros(1,2,2,requires_grad=True);positive=torch.eye(2)[None];neg=torch.zeros(1,2)
        loss,_=auxiliary_bpr(v,v.mean(1),positive,neg,torch.tensor([True]),torch.tensor([0]),"exclusive")
        g=torch.autograd.grad(loss,v)[0][0];cos=F.cosine_similarity(g[0],g[1],dim=0);rel=(g[0]-g[1]).norm()/g.norm(dim=-1).mean()
        self.assertAlmostEqual(float(cos),0.,places=6);self.assertGreater(float(rel),1.)
        RESULT["equal_V_gradient"]=dict(loss=float(loss),head_norms=g.norm(dim=-1).tolist(),cosine=float(cos),relative_difference=float(rel))
        # Equal Query rows: verify distinct parameter gradients too, not merely leaf V.
        m=model();m.extractor.query.data[1].copy_(m.extractor.query.data[0]);torch.manual_seed(19)
        h=torch.randn(2,5,64);V,_=m.extractor(h,torch.tensor([5,5]));self.assertTrue(torch.equal(V[:,0],V[:,1]));e=torch.randn(2,2,64);n=torch.randn(2,64)
        loss,_=auxiliary_bpr(V,V.mean(1),e,n,torch.ones(2,dtype=torch.bool),torch.arange(2),"exclusive");g=torch.autograd.grad(loss,m.extractor.query)[0]
        self.assertGreater(float((g[0]-g[1]).norm()),0);RESULT["equal_Query_gradient"]=dict(cosine=float(F.cosine_similarity(g[0],g[1],dim=0)),norm=float(g.norm()),relative_difference=float((g[0]-g[1]).norm()/g.norm(dim=-1).mean()))

    def test_04_train_dataset_and_boundary(self):
        c=small_corpus();m=model(c);d=m.Dataset(m,c,"train");ref=Base.Dataset(m,c,"train")
        self.assertEqual(len(d),len(ref));self.assertEqual(d.mp_extra.tolist(),[0,3,0,0]);self.assertEqual(d.mp_valid.tolist(),[0,1,0,0])
        for i in range(len(d)):
            a,b=d[i],ref[i]
            for k in b:self.assertTrue(np.array_equal(a[k],b[k]),k)
            self.assertNotIn(4,[int(a["mp_extra_item"])]) # held-out-like suffix absent from Train labels
        # A later Train label change cannot alter the earlier history slice.
        c2=copy.deepcopy(c);c2.data_df["train"].loc[(c2.data_df["train"].user_id==1)&(c2.data_df["train"].position==3),"item_id"]=9
        d2=m.Dataset(m,c2,"train");self.assertTrue(np.array_equal(d[1]["history_items"],d2[1]["history_items"]));self.assertEqual(int(d2[1]["mp_extra_item"]),9)

    def test_05_conflict_duplicate_empty_and_K(self):
        b=synthetic_batch();b["mp_extra_item"]=torch.tensor([100,9,14,15]);b["mp_extra_valid"][3]=0
        for K in [1,2,4]:
            m=model(K=K);m.train();o=m(b);loss=m.loss(o);self.assertTrue(torch.isfinite(loss));loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
            if K>1:self.assertFalse(o["_mp_valid"].any());self.assertEqual(float(o["_mp_aux"]),0)
            else:self.assertNotIn("_mp_aux",o)

    def test_06_evaluation_is_original_scoring(self):
        b=synthetic_batch();b["item_id"]=torch.arange(1,1002)[None].expand(4,-1).clone();b["item_id"][:,50]=b["item_id"][:,51]
        base=model(cls=Base);m=model();base.eval();m.eval()
        with torch.no_grad():
            x=base(b)["prediction"];y=m(b)["prediction"];self.assertTrue(torch.equal(x,y));self.assertEqual(tuple(y.shape),(4,1001));self.assertTrue(torch.equal(y[:,50],y[:,51]))
            b2={k:v[:1] for k,v in b.items()};b2["mp_extra_item"][:]=999999
            self.assertTrue(torch.allclose(y[:1],m(b2)["prediction"],atol=2e-7,rtol=1e-6))
        # Synthetic evaluation phase: no real Dev/Test data files or evaluation metrics.
        c=small_corpus();c.data_df["dev"]=c.data_df["train"].iloc[[2]].copy();d=m.Dataset(m,c,"dev");self.assertFalse(any(k.startswith("mp_") for k in d[0]))

    def test_07_relation_and_encoding_reuse(self):
        b=synthetic_batch();base=model(cls=Base);m=model();calls=[]
        handle=m.item_encoder.register_forward_pre_hook(lambda mod,inputs:calls.append(tuple(inputs[0].shape)))
        base.train();m.train();torch.manual_seed(51);x=base(b,True);torch.manual_seed(51);y=m(b,True);handle.remove()
        self.assertTrue(torch.equal(x["prediction"],y["prediction"]));self.assertTrue(torch.equal(x["_relation_ids"],y["_relation_ids"]))
        self.assertEqual(calls,[(4,3),(4,2),(4,)])
        RESULT["encoder_calls"]=calls

    def test_08_extra_control_and_serialization(self):
        b=synthetic_batch();m=model(mode="extra");m.train();o=m(b,True)
        e=m.item_encoder(b["mp_extra_item"]);u=o["user_vector"];pos=torch.stack([o["candidate_vectors"][:,0],e],1);delta=pos-o["candidate_vectors"][:,1,None]
        expected=F.softplus(-torch.einsum("bd,bmd->bm",u,delta)).mean(-1)[o["_mp_valid"]].mean();self.assertTrue(torch.equal(expected,o["_mp_aux"]))
        import io
        f=io.BytesIO();torch.save(m.state_dict(),f);f.seek(0);n=model(mode="extra");n.load_state_dict(torch.load(f,weights_only=True));m.eval();n.eval()
        with torch.no_grad():self.assertTrue(torch.equal(m(b)["prediction"],n(b)["prediction"]))


    def test_09_active_K4_and_extra_controls(self):
        b=synthetic_batch()
        for mode in ["extra","exclusive"]:
            m=model(K=4,mode=mode);m.train();o=m(b);self.assertEqual(int(o["_mp_valid"].sum()),3)
            loss=m.loss(o);loss.backward();self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
            if mode=="exclusive":self.assertTrue((o["_mp_assignment"][:,0]!=o["_mp_assignment"][:,1]).all())


def real_fixture(ds,device):
    df=pd.read_csv(ROOT/"data"/ds/"train.csv",sep="\t",usecols=["user_id","item_id","time"])
    df=df.sort_values(["time","user_id"],kind="mergesort");df["position"]=df.groupby("user_id",sort=False).cumcount()
    history={int(u):list(zip(g.item_id.astype(int),g.time)) for u,g in df.groupby("user_id",sort=False)}
    c=types.SimpleNamespace(n_users=int(df.user_id.max())+1,n_items=12102 if ds=="beauty" else 3707,
                            data_df={"train":df},user_his=history,train_clicked_set={u:{i for i,t in ls} for u,ls in history.items()})
    a=args(device=device,ds=ds);torch.manual_seed(42);m=Model(a,c).to(device);d=m.Dataset(m,c,"train")
    # Reuse actual original negative sampling, including its NumPy random stream.
    np.random.seed(42);d.actions_before_epoch();pick=np.random.default_rng(20261017).choice(len(d),1024,replace=False)
    b=d.collate_batch([d[int(i)] for i in pick]);b={k:v.to(device) if torch.is_tensor(v) else v for k,v in b.items()};return m,b,c


def real_checks(device):
    rows=[]
    for ds in ["beauty","ml-1m"]:
        m,b,c=real_fixture(ds,device);m.train();o=m(b,True);loss=m.loss(o)
        qg=torch.autograd.grad(o["_mp_aux"],m.extractor.query,retain_graph=True)[0]
        vg=torch.autograd.grad(o["_mp_aux"],o["interest_vectors"],retain_graph=True)[0]
        active=o["_mp_valid"];assignment=o["_mp_assignment"];ties=o["_mp_tied"]
        cos=F.cosine_similarity(vg[:,0],vg[:,1],dim=-1,eps=1e-30)[active]
        relative=(vg[:,0]-vg[:,1]).norm(dim=-1)/(vg.norm(dim=-1).mean(-1).clamp_min(1e-30))
        loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        row=dict(dataset=ds,device=str(device),batch=1024,parameters=m.count_variables(),added_parameters=0,active=int(active.sum()),active_ratio=float(active.float().mean()),aux_loss=float(o["_mp_aux"].detach()),total_loss=float(loss.detach()),tie_fraction=float(ties[active].float().mean()),target0_head0=int(((assignment[:,0]==0)&active).sum()),target0_head1=int(((assignment[:,0]==1)&active).sum()),V_aux_gradient_cosine_mean=float(cos.mean()),V_aux_gradient_relative_difference=float(relative[active].mean()),query_aux_gradient_norm=float(qg.norm()),query_aux_gradient_cosine=float(F.cosine_similarity(qg[0],qg[1],dim=0,eps=1e-30)),query_aux_gradient_relative_difference=float((qg[0]-qg[1]).norm()/qg.norm(dim=-1).mean().clamp_min(1e-30)))
        # Fixed short synthetic-update budget, no epochs/checkpoints/Dev/Test.
        def benchmark(mode):
            torch.manual_seed(42);mm=Base(args(device=device,ds=ds),c).to(device) if mode=="base" else Model(args(device=device,ds=ds),c).to(device)
            lr=.004 if ds=="beauty" else .001;opt=torch.optim.Adam(mm.customize_parameters(),lr=lr,weight_decay=1e-6)
            def step():
                opt.zero_grad(set_to_none=True);ll=mm.loss(mm(b));assert torch.isfinite(ll);ll.backward();opt.step()
            for _ in range(2):step()
            if device.type=="cuda":torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            started=time.perf_counter()
            for _ in range(5):step()
            if device.type=="cuda":torch.cuda.synchronize()
            v=dict(step_ms=(time.perf_counter()-started)*200)
            if device.type=="cuda":v.update(peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
            del opt,mm
            if device.type=="cuda":torch.cuda.empty_cache()
            return v
        del o,loss,qg,vg,m
        if device.type=="cuda":torch.cuda.empty_cache()
        row["short_steps_base"]=benchmark("base");row["short_steps_exclusive"]=benchmark("exclusive");rows.append(row)
    return rows

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--gpu",default=None);a=p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]="" if a.gpu is None else a.gpu
    signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError("170s bounded tests")));signal.alarm(170);torch.set_num_threads(2);started=time.time()
    if a.gpu is None:
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks);r=unittest.TextTestRunner(verbosity=2).run(suite)
        print(json.dumps(dict(synthetic=RESULT,seconds=time.time()-started)));sys.exit(0 if r.wasSuccessful() else 1)
    device=torch.device("cuda:0");print(json.dumps(dict(real=real_checks(device),seconds=time.time()-started)))
