"""Fixed Train-only, user-balanced, R/P and real/permuted semantic screening."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import argparse, hashlib, json, pickle, signal, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
torch.set_num_threads(2)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def metrics(s,tie):
    p=s[:,0,None]
    rank=1+(s[:,1:]>p).sum(1)+((s[:,1:]==p)&(tie[:,1:]>tie[:,0,None])).sum(1)
    return dict(HR5=(rank<=5).astype(float),HR10=(rank<=10).astype(float),
        N5=np.where(rank<=5,1/np.log2(rank+1),0),N10=np.where(rank<=10,1/np.log2(rank+1),0),
        pairwise_win=((p>s[:,1:])+.5*(p==s[:,1:])).mean(1),margin=(p-s[:,1:]).mean(1))

def mean(m,mask=None):
    return {k:float(np.mean(v if mask is None else v[mask])) for k,v in m.items()}

def corr(a,b):
    a=rankdata(np.asarray(a).ravel());b=rankdata(np.asarray(b).ravel())
    return float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None

def scores(z,hist,cand,p):
    result=[np.empty((len(hist),33),np.float32) for _ in range(3)]
    for lo in range(0,len(hist),64):
        hh=hist[lo:lo+64];cc=cand[lo:lo+64]
        h=z[torch.from_numpy(p[hh])];c=z[torch.from_numpy(p[cc])]
        lengths=torch.from_numpy((hh>0).sum(1));valid=torch.from_numpy(hh>0)
        mu=torch.nn.functional.normalize(h.sum(1)/lengths[:,None],dim=1,eps=1e-12)
        last=h[torch.arange(len(h)),lengths-1]
        pair=torch.bmm(h,c.transpose(1,2)).masked_fill(~valid[:,:,None],-torch.inf)
        values=[torch.bmm(mu[:,None],c.transpose(1,2))[:,0],torch.bmm(last[:,None],c.transpose(1,2))[:,0],pair.max(1).values]
        for dst,v in zip(result,values):dst[lo:lo+len(h)]=v.numpy()
    return result

def main(ds):
    start=time.monotonic()
    def stop(*args):raise TimeoutError('170-second dataset limit')
    signal.signal(signal.SIGALRM,stop);signal.alarm(170)
    root=Path(__file__).resolve().parents[1]
    train=root/f'data/{ds}/train.csv';emb=root/f'data/{ds}/handled/llm_table_pca1536.pkl'
    df=pd.read_csv(train,sep='\t',usecols=['user_id','item_id','time'])
    df['source_row']=np.arange(len(df));df=df.sort_values(['user_id','time','source_row'],kind='mergesort')
    raw=np.asarray(pickle.load(open(emb,'rb')),dtype=np.float32)
    assert raw.ndim==2 and raw.shape[1]==1536 and df.item_id.max()<len(raw)
    assert np.isfinite(raw).all() and np.linalg.norm(raw[0])==0 and (df.item_id>0).all()
    assert (np.linalg.norm(raw[df.item_id.unique(),:512],axis=1)>1e-12).all()
    z=torch.nn.functional.normalize(torch.from_numpy(raw[:,:512].copy()),dim=1,eps=1e-12);z[0]=0
    counts=np.bincount(df.item_id.to_numpy(),minlength=len(raw));catalog=np.flatnonzero(counts>0)
    popbin=np.zeros(len(raw),int);popbin[catalog]=np.floor(np.log2(counts[catalog])).astype(int)
    pools={int(k):catalog[popbin[catalog]==k] for k in np.unique(popbin[catalog])}
    items=df.item_id.to_numpy();times=df.time.to_numpy();users=df.user_id.to_numpy();source=df.source_row.to_numpy()
    starts=np.r_[0,np.flatnonzero(users[1:]!=users[:-1])+1];ends=np.r_[starts[1:],len(df)]
    rng=np.random.default_rng(42);eligible=np.flatnonzero(ends-starts>=2)
    samples=[];badtime=0;repeated=0;inspected=0
    for gi in rng.permutation(eligible):
        inspected+=1;s=int(starts[gi]);e=int(ends[gi])
        for off in rng.permutation(np.arange(1,e-s)):
            j=s+int(off);h=items[max(s,j-20):j]
            if times[j-1]>=times[j]:badtime+=1;continue
            if int(items[j]) in h:repeated+=1;continue
            samples.append((int(users[j]),int(source[j]),h.copy(),int(items[j])));break
        if len(samples)==1024:break
    n=len(samples);assert n>0
    hist=np.zeros((n,20),np.int64);targets=np.array([x[3] for x in samples]);lengths=np.array([len(x[2]) for x in samples])
    for j,x in enumerate(samples):hist[j,:len(x[2])]=x[2]
    rr=np.random.default_rng(np.random.SeedSequence([42,1]));pr=np.random.default_rng(np.random.SeedSequence([42,2]))
    cand={k:np.empty((n,33),np.int64) for k in ['R','P']};exact=[];distance_values=[]
    for j,(u,row,h,t) in enumerate(samples):
        excluded=np.unique(np.r_[0,h,t]);legal=catalog[~np.isin(catalog,excluded)];assert len(legal)>=32
        rn=rr.choice(legal,32,replace=False);selected=[];k=int(popbin[t])
        for distance in range(max(pools)+1):
            parts=[pools[b] for b in sorted(pools) if abs(b-k)==distance]
            pool=np.concatenate(parts) if parts else np.empty(0,np.int64)
            pool=pool[~np.isin(pool,excluded)];take=min(32-len(selected),len(pool))
            if take:selected.extend(pr.choice(pool,take,replace=False).tolist())
            if len(selected)==32:break
        pn=np.array(selected);assert len(pn)==32
        for name,neg in [('R',rn),('P',pn)]:
            assert len(np.unique(neg))==32 and not np.isin(neg,excluded).any();cand[name][j]=np.r_[t,neg]
        exact.append(float(np.mean(popbin[pn]==k)));distance_values.extend(np.abs(popbin[pn]-k).tolist())
    permutations={'real':np.arange(len(raw),dtype=np.int64)};ph={}
    for seed in [20261015,20261016,20261017]:
        p=np.r_[0,np.random.default_rng(seed).permutation(np.arange(1,len(raw)))];assert len(np.unique(p))==len(p)
        permutations[str(seed)]=p;ph[str(seed)]=hashlib.sha256(p.tobytes()).hexdigest()
    tie=np.random.default_rng(np.random.SeedSequence([42,3])).random((n,33))
    bs=np.random.default_rng(20261015).integers(0,n,size=(1000,n))
    overall={};peruser={};real_scores={}
    for protocol,cc in cand.items():
        overall[protocol]={'popularity_only':mean(metrics(np.log1p(counts[cc]),tie))};peruser[protocol]={}
        for view,p in permutations.items():
            sc=scores(z,hist,cc,p)
            if view=='real':real_scores[protocol]=sc
            for si,a in enumerate(sc,1):
                key=f'S{si}';m=metrics(a,tie);peruser[protocol][view+key]=m
                overall[protocol].setdefault(key,{})[view]=dict(**mean(m),score_popularity_spearman=corr(a,counts[cc]),win_target_popularity_spearman=corr(m['pairwise_win'],counts[targets]))
        for key in ['S1','S2','S3']:
            real=peruser[protocol]['real'+key]
            pm={k:np.mean([peruser[protocol][v+key][k] for v in ph],axis=0) for k in real}
            delta={k:real[k]-pm[k] for k in real}
            overall[protocol][key]['paired_real_minus_permuted_mean']=mean(delta)
            overall[protocol][key]['paired_user_bootstrap_95CI']={k:np.quantile(delta[k][bs].mean(1),[.025,.975]).tolist() for k in ['N5','pairwise_win']}
    sim=real_scores['P'][0][:,0];margin=peruser['P']['realS1']['margin'];tp=counts[targets]
    specs=dict(history_length=[('1-5',lengths<=5),('6-10',(lengths>=6)&(lengths<=10)),('11-20',lengths>=11)],
        target_train_count=[('1-5',tp<=5),('6-20',(tp>=6)&(tp<=20)),('21-100',(tp>=21)&(tp<=100)),('101+',tp>=101)],
        history_target_cosine=[('[-1,0)',sim<0),('[0,.2)',(sim>=0)&(sim<.2)),('[.2,.4)',(sim>=.2)&(sim<.4)),('[.4,1]',sim>=.4)],
        P_S1_semantic_margin=[('<-.1',margin<-.1),('[-.1,0)',(margin>=-.1)&(margin<0)),('[0,.1)',(margin>=0)&(margin<.1)),('>=.1',margin>=.1)])
    groups={}
    for name,bins in specs.items():
        groups[name]=[]
        for label,mask in bins:
            count=int(mask.sum());row=dict(bin=label,n=count,sufficient=count>=64)
            if count>=64:
                row['results']={}
                for protocol in ['R','P']:
                    row['results'][protocol]={}
                    for key in ['S1','S2','S3']:
                        real=peruser[protocol]['real'+key]
                        pm={k:np.mean([peruser[protocol][v+key][k] for v in ph],axis=0) for k in real}
                        row['results'][protocol][key]=dict(real=mean(real,mask),permuted_mean=mean(pm,mask),delta={k:float((real[k]-pm[k])[mask].mean()) for k in real})
            groups[name].append(row)
    manifest=np.array([[x[0],x[1],x[3],len(x[2])] for x in samples],np.int64)
    result=dict(dataset=ds,n=n,users=len(set(x[0] for x in samples)),eligible_users=len(eligible),inspected_users=inspected,
        rejected_candidate_prefixes=dict(timestamp_not_strict=badtime,target_repeated_in_H20=repeated),train_rows=len(df),catalog_train_items=len(catalog),embedding_shape=list(raw.shape),
        space='Raw PCA1536 first512, item L2; S1 normalized mean of unit items, S2 last, S3 max cosine',
        history_lengths=dict(min=int(lengths.min()),max=int(lengths.max()),mean=float(lengths.mean()),median=float(np.median(lengths))),
        candidates='Train-observed catalog; exclude current H20/positive/padding/duplicate negatives only; no future-positive filtering',
        popularity_matching=dict(rule='floor(log2(train count)); nearest log bins fallback, uniform equal-distance pools',exact_negative_fraction=float(np.mean(exact)),all32_exact_sample_fraction=float(np.mean(np.array(exact)==1)),fallback_samples=int(np.sum(np.array(exact)<1)),mean_bin_distance=float(np.mean(distance_values))),
        seeds=dict(sample=42,R=[42,1],P=[42,2],tie=[42,3],permutation=list(ph),bootstrap=20261015),
        hashes=dict(train_csv=sha(train),llm_pkl=sha(emb),sample_manifest=hashlib.sha256(manifest.tobytes()).hexdigest(),histories=hashlib.sha256(hist.tobytes()).hexdigest(),candidates={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in cand.items()},permutations=ph),
        overall=overall,groups=groups,
        caveats='Train diagnosis not held-out; target-derived semantic bins descriptive/tautological; 3 fixed permutations not trained-run significance; bootstrap conditional on samples/candidates/permutations; no model/dev/test')
    result['seconds']=time.monotonic()-start;signal.alarm(0);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['beauty','ml-1m'],required=True)
    print(json.dumps(main(p.parse_args().dataset)))
