"""Explicit-output, Train-only anchor/teacher-cache builder; no training/evaluation."""
import sys,time,json,argparse,signal
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from models.sequential.LLMMIRecASPCFAdaptiveSIL import LLMMIRecASPCFAdaptiveSIL, array_sha
from models.sequential.gict_components import semantic_anchors,interest_profiles
from gict_data import train_corpus,args_from_config,sha,project_output
import numpy as np


@torch.inference_mode()
def build_bank(args,corpus,checkpoint,max_users=0,anchor_count=32,batch_size=64):
    start=time.perf_counter()
    if max_users<0 or batch_size<1 or args.history_max<1:raise ValueError("Invalid cache limits")
    teacher=LLMMIRecASPCFAdaptiveSIL(args,corpus).eval()
    teacher.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True),strict=True)
    teacher.requires_grad_(False)
    device=torch.device(args.device);teacher=teacher.to(device)
    train=corpus.data_df["train"]
    anchors,table=semantic_anchors(teacher.item_encoder.llm_table,
        torch.as_tensor(train.item_id.to_numpy(np.int64),device=device),anchors=anchor_count,seed=0)
    users=sorted(corpus.user_his)
    scope="full_train"
    if max_users and len(users)>max_users:
        users=users[:max_users];scope="sample_check_only"
    rows={k:[] for k in ["user_profiles","interest_profiles","interest_vectors","interest_support","user_vectors"]}
    for offset in range(0,len(users),batch_size):
        us=users[offset:offset+batch_size]
        histories=[corpus.user_his[u][-args.history_max:] for u in us]
        lengths=torch.tensor([len(x) for x in histories],device=device)
        history=torch.zeros((len(us),max(int(lengths.max()),1)),dtype=torch.long,device=device)
        for r,seq in enumerate(histories):
            history[r,:len(seq)]=torch.tensor([x[0] for x in seq],device=device)
        feed=dict(user_id=torch.tensor(us,device=device),history_items=history,lengths=lengths,
                  item_id=torch.zeros((len(us),1),dtype=torch.long,device=device))
        out=teacher(feed,True)
        profile,support=interest_profiles(table,history,out["attention_maps"],lengths)
        coarse=table[history].sum(1)/lengths[:,None].clamp_min(1)
        for key,val in dict(user_profiles=coarse,interest_profiles=profile,interest_vectors=out["interest_vectors"],
                            interest_support=support,user_vectors=out["user_vector"]).items():
            rows[key].append(val.detach().cpu())
    import hashlib
    metadata=dict(schema="gict_train_bank_v1",scope=scope,dataset=args.dataset,K=args.K,history_max=args.history_max,
        anchors=len(anchors),neighbors=8,teacher_checkpoint=str(Path(checkpoint).resolve()),
        teacher_checkpoint_sha256=sha(checkpoint),embedding_sha256=sha(args.llm_emb_path),
        train_frame_sha256=array_sha(train[["user_id","item_id","position"]].to_numpy(np.int64)),
        train_csv_sha256=sha(Path(args.path)/args.dataset/"train.csv"),
        users=len(users),teacher_frozen=True,source="train.csv only; final Train prefixes; target-free dummy candidate0",
        self_exclusion="all cache rows with query user_id are excluded, including own later Train events",
        sources={str(Path(p).resolve()):sha(p) for p in [__file__,"models/sequential/LLMMIRecASPCFAdaptiveSIL.py",
                    "models/sequential/gict_components.py","tools/gict_data.py"]})
    result=dict(metadata=metadata,anchors=anchors.cpu(),item_probabilities=table.cpu(),
                user_ids=torch.tensor(users,dtype=torch.long),**{k:torch.cat(v) for k,v in rows.items()})
    metadata["tensor_bytes"]=sum(v.numel()*v.element_size() for v in result.values() if torch.is_tensor(v))
    metadata["build_seconds"]=time.perf_counter()-start
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--checkpoint",required=True)
    p.add_argument("--output",required=True);p.add_argument("--max-users",type=int,default=0)
    p.add_argument("--anchors",type=int,default=32);p.add_argument("--timeout-seconds",type=int,default=165)
    x=p.parse_args();signal.alarm(min(x.timeout_seconds,170));torch.set_num_threads(4)
    path=project_output(x.output);a=args_from_config(x.config);c=train_corpus(a)
    result=build_bank(a,c,x.checkpoint,x.max_users,x.anchors)
    # Exclusive create; never overwrite a bank or checkpoint.
    with path.open("xb") as f:torch.save(result,f)
    print(json.dumps(result["metadata"],indent=2))
if __name__=="__main__":main()
