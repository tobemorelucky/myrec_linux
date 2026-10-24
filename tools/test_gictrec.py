"""Bounded structural checks only: synthetic math and two real Train samples."""
import sys,time,json,argparse,signal,copy,importlib.util,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from torch.nn import functional as F
from models.sequential.gict_components import *
from models.sequential.GICTRec import GICTRec
from models.sequential.LLMMIRecASPCFAdaptiveSIL import LLMMIRecASPCFAdaptiveSIL,segment_ids
from build_gict_bank import build_bank
from gict_data import train_corpus,args_from_config,project_output,sha
ROOT=Path(__file__).resolve().parents[1]


def synthetic(device):
    torch.manual_seed(7)
    # Chunked search equals exhaustive SMALL reference; all self rows excluded.
    A=5;U=9;K=3;D=64
    users=torch.arange(1,U+1,device=device)
    probs=torch.rand(17,A,device=device).softmax(-1);probs[0]=0
    payload=dict(metadata=dict(schema="gict_train_bank_v1"),anchors=torch.rand(A,8,device=device),
        item_probabilities=probs,user_ids=users,user_profiles=torch.rand(U,A,device=device).softmax(-1),
        interest_profiles=torch.rand(U,K,A,device=device).softmax(-1),
        interest_vectors=torch.rand(U,K,D,device=device),interest_support=torch.ones(U,K,device=device),
        user_vectors=torch.rand(U,D,device=device))
    bank=SemanticInterestIndex(payload,query_block=2,user_block=3).to(device)
    hist=torch.tensor([[1,2,0],[3,4,5],[0,0,0]],device=device);lens=torch.tensor([2,3,0],device=device);uid=torch.tensor([1,2,3],device=device)
    ids,valid=bank.retrieve(hist,lens,uid,8)
    assert not (bank.user_ids[ids][valid]==uid[:,None].expand_as(ids)[valid]).any()
    assert not valid[2].any()
    full=bank.prefix_profile(hist,lens).sqrt()@bank.user_profiles.sqrt().T
    full.masked_fill_(uid[:,None]==users[None],-torch.inf)
    assert torch.allclose(full[:2].topk(8,-1).values,full.gather(1,ids)[:2])
    empty=dict(payload)
    for name in ["user_ids","user_profiles","interest_profiles","interest_vectors","interest_support","user_vectors"]:
        empty[name]=payload[name][:0]
    no_bank=SemanticInterestIndex(empty,query_block=2,user_block=3).to(device)
    empty_ids,empty_mask=no_bank.retrieve(hist,lens,uid,8)
    assert not empty_mask.any() and torch.isfinite(no_bank.interest_vectors[empty_ids]).all()
    for k in [1,2,3,4,7]:
        current=torch.randn(3,k,D,device=device,requires_grad=True)
        profiles=torch.rand(3,k,A,device=device).softmax(-1).requires_grad_()
        support=torch.ones(3,k,device=device);support[2]=0
        nv=torch.randn(3,8,k,D,device=device,requires_grad=True)
        npf=torch.rand(3,8,k,A,device=device).softmax(-1)
        ns=torch.ones(3,8,k,device=device);nu=torch.randn(3,8,D,device=device)
        mask=torch.ones(3,8,dtype=torch.bool,device=device);mask[2]=False
        for variant in ["G1","G2","G3"]:
            block=ReliableInterestTransfer(D).to(device)
            out,_=block(current,profiles,support,nv,npf,ns,nu,mask,variant)
            assert torch.equal(out,current),"Zero output projection is not exact"
            out.square().sum().backward(retain_graph=True)
            assert torch.isfinite(block.output.weight.grad).all() and block.output.weight.grad.abs().sum()>0
            with torch.no_grad():block.output.weight.normal_(0,.01)
            changed,details=block(current,profiles,support,nv,npf,ns,nu,mask,variant)
            assert torch.equal(changed[2],current[2])
            changed.square().sum().backward(retain_graph=True)
            assert nv.grad is None,"Neighbor teacher gradient leaked"
            if variant in ("G2","G3"):
                assert profiles.grad is not None and profiles.grad.abs().sum()>0
            assert torch.isfinite(changed).all()
    # Pad content cannot alter interest semantics; current history prefix only.
    att=torch.rand(3,2,3,device=device)
    p,s=interest_profiles(probs,hist,att,lens)
    h2=hist.clone();h2[0,2]=16
    p2,s2=interest_profiles(probs,h2,att,lens)
    assert torch.equal(p,p2) and torch.equal(s,s2)
    return dict(K=[1,2,3,4,7],self_exclusion=True,chunk_reference=True,no_neighbor=True,
                zero_init=True,padding=True,teacher_detach=True,finite_gradients=True)


def real_check(ds,config,checkpoint,report_dir,device="cpu"):
    started=time.perf_counter();a=args_from_config(config);a.device=device
    c=train_corpus(a)
    assert set(c.data_df)=={"train"}
    bank=build_bank(a,c,checkpoint,max_users=128,anchor_count=32)
    a.gict_variant="G3";a.gict_neighbors=8;a.gict_temperature=.1
    state=torch.load(checkpoint,map_location="cpu",weights_only=True)
    m=GICTRec(a,c);m.load_c1(state);m.attach_bank(bank,allow_sample=True);m=m.to(device).eval()
    reference=LLMMIRecASPCFAdaptiveSIL(a,c).to(device).eval();reference.load_state_dict(state,strict=True)
    # Also check the ACTUAL trained snapshot wrapper, not only the public base class.
    entry=Path(config).parents[2]/"objective_model.py"
    spec=importlib.util.spec_from_file_location("gict_frozen_reference_"+ds.replace("-",""),entry)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    old_args=copy.copy(a);old_args.objective="SS";old_args.sil_variant="C1";old_args.sil_use_behavior=1
    archived=module.ObjectiveSIL(old_args,c).to(device).eval();archived.load_state_dict(state,strict=True)
    us=bank["user_ids"][:8].tolist()
    history=torch.zeros(8,20,dtype=torch.long,device=device);lengths=torch.tensor([0,1,2,3,4,7,10,20],device=device)
    for i,u in enumerate(us):
        seq=[x[0] for x in c.user_his[u]][:int(lengths[i])]
        lengths[i]=len(seq)
        if seq:history[i,:len(seq)]=torch.tensor(seq,device=device)
    feed=dict(user_id=torch.tensor(us,device=device),history_items=history,lengths=lengths,
              item_id=torch.tensor([[50+i,100+i,150+i] for i in range(8)],device=device))
    with torch.no_grad():
        old=reference(feed,True);frozen=archived(feed,True);new=m(feed,True)
    assert torch.equal(old["prediction"],frozen["prediction"]),"Public C1 != actual frozen C1 snapshot"
    assert torch.equal(old["prediction"],new["prediction"]),"Initial G3 not C1-equivalent"
    assert torch.equal(old["interest_vectors"],new["interest_vectors"])
    empty=dict(feed);empty["history_items"]=history[:,:0];empty["lengths"]=lengths.new_zeros(8)
    with torch.no_grad():empty_out=m(empty,True)
    assert torch.isfinite(empty_out["prediction"]).all()
    assert torch.count_nonzero(empty_out["interest_vectors"])==0
    m.gict_variant="G0"
    with torch.no_grad():off=m(feed,True)
    assert torch.equal(off["prediction"],old["prediction"])
    m.gict_variant="G3"
    # Zero branch preserves TRAIN loss/common gradients and RNG, including dropout/relation sampling.
    m.train();reference.train();m.zero_grad();reference.zero_grad()
    rng=torch.get_rng_state()
    cuda_rng=torch.cuda.get_rng_state(device) if str(device).startswith("cuda") else None
    train_new=m(feed,True);loss_new=m.loss(train_new);loss_new.backward()
    torch.set_rng_state(rng)
    if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng,device)
    train_old=reference(feed,True);loss_old=reference.loss(train_old);loss_old.backward()
    assert torch.equal(loss_new,loss_old),"Zero branch changed original train loss"
    reference_parameters=dict(reference.named_parameters())
    for name,param in m.named_parameters():
        if name.startswith("transfer."):continue
        baseline=reference_parameters[name]
        if baseline.grad is not None:
            assert param.grad is not None and torch.equal(param.grad,baseline.grad),name
    m.eval();reference.eval()
    # Activate output only for mechanism/gradient/target-independence checks.
    with torch.no_grad():m.transfer.output.weight.normal_(0,.01)
    other=dict(feed);other["item_id"]=torch.tensor([[5,6,7]]*8,device=device)
    with torch.no_grad():o1=m(feed,True);o2=m(other,True)
    assert torch.equal(o1["interest_vectors"],o2["interest_vectors"])
    assert torch.equal(o1["gict_neighbor_users"],o2["gict_neighbor_users"])
    assert not (o1["gict_neighbor_users"]==feed["user_id"][:,None])[o1["gict_neighbor_valid"]].any()
    padded=dict(feed);padded["history_items"]=history.clone()
    valid=torch.arange(20,device=device)[None]<lengths[:,None]
    padded["history_items"][~valid]=9
    # Valid padding must be item0 in the backbone; test extended zero padding equivalence.
    shorter=dict(feed);shorter["history_items"]=history[:,:max(int(lengths.max()),1)]
    with torch.no_grad():short=m(shorter,True)
    assert torch.allclose(short["interest_vectors"],o1["interest_vectors"],atol=2e-6,rtol=2e-6)
    # Local bank causality/cross-segment gradients are exactly zero.
    x=torch.randn(1,8,64,device=device,requires_grad=True)
    ids,valids=segment_ids(x.new_zeros(1,8),torch.tensor([8],device=device),"fixed",0.,fixed_len=4)
    local=m.sil_block.local_bank(x,ids,valids)
    grad=torch.autograd.grad(local[0,2].square().sum(),x)[0]
    assert torch.count_nonzero(grad[0,3:])==0
    # Loss unchanged; finite original Softmax + relation and trainable transfer gradients.
    m.train();m.cur_epoch=1;m.actions_before_epoch();m.zero_grad()
    out=m(feed,True);loss=m.loss(out);loss.backward()
    assert out["_ss_allowed"].sum(-1).gt(1).any(), "No valid Softmax contrast"
    assert out["loss_ranking"]>0, "Degenerate ranking check"
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert m.transfer.output.weight.grad is not None and m.transfer.output.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in m.semantic_index.parameters())
    assert not any(t.requires_grad for t in m.semantic_index.buffers())
    # Neighbor encoding cannot occur online: index contains tensors only, not a teacher module.
    assert all(not isinstance(x,LLMMIRecASPCFAdaptiveSIL) for x in m.semantic_index.modules())
    m.eval()
    with torch.inference_mode():
        for _ in range(2):m(feed)
        if str(device).startswith("cuda"):torch.cuda.synchronize()
        t=time.perf_counter()
        for _ in range(5):m(feed)
        if str(device).startswith("cuda"):torch.cuda.synchronize()
        ms=(time.perf_counter()-t)*1000/5
    path=report_dir/(ds+"_sample_bank.pt")
    with path.open("xb") as f:torch.save(bank,f)
    result=dict(dataset=ds,K=a.K,users_cached=128,cache_bytes=m.semantic_index.bytes(),
        full_bank_bytes_estimate=m.semantic_index.item_probabilities.numel()*4
          +len(c.user_his)*(a.K*(32+64+1)+32+64)*4+len(c.user_his)*8+32*1536*4,
        cache_build_seconds=bank["metadata"]["build_seconds"],step_8_rows_ms=ms,
        seconds=time.perf_counter()-started,Test_read_or_evaluated=False,train_only_csv=True,
        actual_snapshot_exact=True,branch_off_exact=True,initial_G3_exact=True,
        target_independent=True,self_future_row_excluded=True,local_causality=True,
        finite_loss=float(loss.detach().cpu()),teacher_detached=True,online_teacher_encodes=0,
        bank_metadata=bank["metadata"])
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--output-dir",required=True);p.add_argument("--gpu",type=int)
    p.add_argument("--synthetic-only",action="store_true");x=p.parse_args()
    signal.alarm(170);torch.set_num_threads(4);out=project_output(x.output_dir);out.mkdir()
    start=time.perf_counter();result=dict(synthetic_cpu=synthetic("cpu"),real=[],Test_evaluations=0,formal_training=False)
    if not x.synthetic_only:
        pairs=[
          ("beauty","research/elector_sil_formal_20261023_v1/runs/beauty_k2_s42_SS-C1"),
          ("ml-1m","research/elector_sil_k3_20261023_v1/runs/ml-1m_k3_s42_SS-C1")]
        for ds,folder in pairs:
            folder=ROOT/folder;result["real"].append(real_check(ds,folder/"config.json",folder/"best.pt",out))
    if x.gpu is not None:
        # Respect all-user process inventory AND the project's shared GPU lease.
        path=ROOT/"research/elector_sil_objective_full_20261023_v2/scheduler.py"
        spec=importlib.util.spec_from_file_location("gict_gpu_lease",path);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
        inventory=r.observe()
        if x.gpu not in inventory or not r.idle(inventory[x.gpu]):result["GPU"]="skipped_busy"
        else:
            fd=r.acquire(ROOT/("research/gpu_resource_scheduler_20261022_v1/gpu%d.lock"%x.gpu))
            if fd is None:result["GPU"]="skipped_locked"
            else:
                try:
                    if not r.idle(r.observe()[x.gpu]):result["GPU"]="skipped_became_busy"
                    else:
                        device="cuda:"+str(x.gpu);torch.cuda.set_device(x.gpu);torch.cuda.reset_peak_memory_stats(x.gpu)
                        result["synthetic_GPU"]=synthetic(device)
                        gpu_dir=out/"gpu";gpu_dir.mkdir()
                        result["real_GPU"]=[]
                        if not x.synthetic_only:
                            for ds,folder in pairs:
                                folder=ROOT/folder
                                result["real_GPU"].append(real_check(ds,folder/"config.json",folder/"best.pt",gpu_dir,device))
                        result["GPU_peak_MiB"]=torch.cuda.max_memory_allocated(x.gpu)/2**20
                finally:os.close(fd)
    result.update(passed=True,seconds=time.perf_counter()-start)
    with (out/"checks.json").open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
