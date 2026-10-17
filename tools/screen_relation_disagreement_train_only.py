"""Finite Train-only relationship-disagreement screening. No recommender/Dev/Test.
Past reference item-user incidence; user-held-out low-dimensional statistical proxy.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse,hashlib,json,math,pickle,signal,time,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

ROOT=Path(__file__).resolve().parents[1]
def midcdf(reference,x):
    r=np.sort(reference);return (np.searchsorted(r,x,'left')+np.searchsorted(r,x,'right'))/(2*len(r))
def stats(x):
    x=np.asarray(x,dtype=float).ravel()
    return dict(n=len(x),mean=float(x.mean()),std=float(x.std()),p10=float(np.quantile(x,.1)),median=float(np.median(x)),p90=float(np.quantile(x,.9))) if len(x) else dict(n=0)
def maskedmean(x,m,axis):return (x*m).sum(axis)/np.maximum(m.sum(axis),1)
def cov(x,a,valid):
    # x [N,L], a [N,2,L], both share history mask.
    xm=maskedmean(x,valid,1);am=maskedmean(a,valid[:,None,:],2)
    return maskedmean((x-xm[:,None])[:,None,:]*(a-am[:,:,None]),valid[:,None,:],2)
def interval(a):return [float(v) for v in np.quantile(a,[.025,.975])]

def main(ds):
    start=time.monotonic();signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError('170s per dataset; stop')));signal.alarm(170)
    train=ROOT/'data'/ds/'train.csv';emb=ROOT/'data'/ds/'handled/llm_table_pca1536.pkl'
    df=pd.read_csv(train,sep='\t',usecols=['user_id','item_id','time']);df['row']=np.arange(len(df))
    cutoff=np.quantile(df.time.to_numpy(),.75,method='nearest');ref=df[df.time<cutoff]
    raw=np.asarray(pickle.load(open(emb,'rb')),dtype=np.float32);N=len(raw)
    assert raw.ndim==2 and raw.shape[1]==1536 and np.isfinite(raw).all() and df.item_id.max()<N
    z=raw[:,:512].copy();zn=np.linalg.norm(z,axis=1);z/=np.maximum(zn[:,None],1e-12);z[0]=0
    usermax=int(df.user_id.max())+1
    M=csr_matrix((np.ones(len(ref),np.float32),(ref.item_id.to_numpy(),ref.user_id.to_numpy())),shape=(N,usermax));M.sum_duplicates();M.data[:]=1;M.sort_indices()
    degrees=np.asarray(M.sum(1)).ravel();pop=np.bincount(ref.item_id.to_numpy(),minlength=N)
    catalog=np.flatnonzero((degrees>0)&(zn>1e-12));bins=np.floor(np.log2(np.maximum(pop,1))).astype(int)
    def bcos(a,b):
        shape=a.shape;a=a.ravel();b=b.ravel();code=np.minimum(a,b).astype(np.int64)*N+np.maximum(a,b)
        uni,inv=np.unique(code,return_inverse=True);uu=uni//N;vv=uni%N;values=np.empty(len(uni),np.float32)
        for s in range(0,len(uni),2048):
            aa=uu[s:s+2048];bb=vv[s:s+2048]
            dots=np.asarray(M[aa].multiply(M[bb]).sum(1)).ravel()
            values[s:s+2048]=dots/np.sqrt(np.maximum(degrees[aa]*degrees[bb],1))
        return values[inv].reshape(shape)
    # Calibration is past-reference-only, without query targets or proxy labels.
    rng=np.random.default_rng(20261016);ca=rng.choice(catalog,4096);cb=rng.choice(catalog,4096)
    while np.any(ca==cb):
        mask=ca==cb;cb[mask]=rng.choice(catalog,int(mask.sum()))
    cs=np.einsum('ij,ij->i',z[ca],z[cb]);cc=bcos(ca,cb)
    df=df.sort_values(['user_id','time','row'],kind='mergesort');items=df.item_id.to_numpy();times=df.time.to_numpy();users=df.user_id.to_numpy();rows=df.row.to_numpy()
    starts=np.r_[0,np.flatnonzero(users[1:]!=users[:-1])+1];ends=np.r_[starts[1:],len(df)]
    rr=np.random.default_rng(42);samples=[];rejected=dict(time=0,target_unseen=0,history_support=0,repeated_target=0);inspected=0
    for gi in rr.permutation(np.flatnonzero(ends-starts>=3)):
        inspected+=1;s=int(starts[gi]);e=int(ends[gi]);eligible=np.flatnonzero(times[s:e]>=cutoff)
        eligible=eligible[eligible>=2]
        for off in rr.permutation(eligible):
            j=s+int(off);h=items[max(s,j-20):j]
            if times[j-1]>=times[j]:rejected['time']+=1;continue
            target=int(items[j])
            if degrees[target]==0 or zn[target]<=1e-12:rejected['target_unseen']+=1;continue
            if target in h:rejected['repeated_target']+=1;continue
            if np.sum((degrees[h]>0)&(zn[h]>1e-12))<2:rejected['history_support']+=1;continue
            samples.append((int(users[j]),int(rows[j]),h.copy(),target,float(times[j])));break
        if len(samples)==1024:break
    n=len(samples);assert n>=256,'too few legal common-support users; stop'
    H=np.zeros((n,20),np.int64);lens=np.array([len(x[2]) for x in samples]);target=np.array([x[3] for x in samples]);uid=np.array([x[0] for x in samples])
    assert len(np.unique(uid))==n and min(x[4] for x in samples)>=cutoff and ref.time.max()<cutoff
    for i,x in enumerate(samples):H[i,:len(x[2])]=x[2]
    valid=(H>0)&(degrees[H]>0)&(zn[H]>1e-12);neg=[];fallback=0;nr=np.random.default_rng(20261017)
    for i,y in enumerate(target):
        legal=catalog[~np.isin(catalog,np.r_[0,H[i],y])];distance=np.abs(bins[legal]-bins[y]);best=distance.min();fallback+=int(best>0)
        neg.append(int(nr.choice(legal[distance==best])))
    C=np.stack([target,np.array(neg)],1)
    ZH=z[H];ZC=z[C];S=np.einsum('nld,nmd->nlm',ZH,ZH,optimize=True).clip(-1,1)
    aa=np.broadcast_to(H[:,:,None],(n,20,20));bb=np.broadcast_to(H[:,None,:],(n,20,20));B=bcos(aa,bb)
    US=midcdf(cs,S);UB=midcdf(cc,B);mask=valid[:,:,None]&valid[:,None,:]&~np.eye(20,dtype=bool)[None]
    # Zero support/diagonal never interpreted as observed relation.
    degs=maskedmean(US,mask,2);degb=maskedmean(UB,mask,2);dis=maskedmean(np.abs(US-UB),mask,2);common=(degs+degb)/2
    acs=np.einsum('ncd,nld->ncl',ZC,ZH,optimize=True).clip(-1,1)
    acb=bcos(np.broadcast_to(C[:,:,None],(n,2,20)),np.broadcast_to(H[:,None,:],(n,2,20)))
    A=midcdf(cs,acs);Q=midcdf(cc,acb)
    def viewfeat(a):
        mean=maskedmean(a,valid[:,None,:],2);maximum=np.where(valid[:,None,:],a,-np.inf).max(2)
        last=(np.where(valid,np.arange(20),-1)).max(1);tail=a[np.arange(n)[:,None],np.arange(2)[None,:],last[:,None]]
        return mean,tail,maximum
    sm,sl,sx=viewfeat(A);bm,bl,bx=viewfeat(Q)
    base=np.stack([sm,sl,sx,bm,bl,bx,np.log1p(pop[C]),sm*bm,sl*bl,sx*bx,sm**2,bm**2,cov(degs,A,valid),cov(degb,Q,valid)],-1)
    contrast=A-Q;fc=cov(common,contrast,valid);fd=cov(dis,contrast,valid)
    shuffled=dis.copy();nullrng=np.random.default_rng(20261018)
    for i in range(n):shuffled[i,valid[i]]=nullrng.permutation(dis[i,valid[i]])
    fn=cov(shuffled,contrast,valid)
    matrices={'B':base,'C':np.concatenate([base,fc[:,:,None]],-1),
              'D':np.concatenate([base,fc[:,:,None],fd[:,:,None]],-1),'N':np.concatenate([base,fc[:,:,None],fn[:,:,None]],-1)}
    folds=list(KFold(5,shuffle=True,random_state=42).split(uid));prediction={};losses={};correct={};models={}
    for name,features in matrices.items():
        X=features[:,0]-features[:,1];assert np.isfinite(X).all();pred=np.empty(n);converged=True
        for tr,te in folds:
            assert not set(uid[tr])&set(uid[te]);scale=np.sqrt(np.mean(X[tr]**2,axis=0));scale[scale<1e-12]=1
            mirrored=np.r_[X[tr]/scale,-X[tr]/scale];labels=np.r_[np.ones(len(tr)),np.zeros(len(tr))]
            model=LogisticRegression(C=1.,fit_intercept=False,solver='lbfgs',max_iter=200,tol=1e-6,random_state=42)
            model.fit(mirrored,labels);pred[te]=model.decision_function(X[te]/scale);converged &= bool(model.n_iter_[0]<200)
        loss=np.logaddexp(0,-pred);hit=(pred>0).astype(float)+.5*(pred==0);prediction[name]=pred;losses[name]=loss;correct[name]=hit
        models[name]=dict(features=X.shape[1],OOF_pair_logloss=float(loss.mean()),OOF_pair_accuracy=float(hit.mean()),converged=converged)
    bootstrap=np.random.default_rng(20261019).integers(0,n,size=(1000,n));comparisons={}
    for before,after in [('B','C'),('C','D'),('N','D')]:
        gain=losses[before]-losses[after];hgain=correct[after]-correct[before]
        comparisons[before+'-'+after]=dict(logloss_reduction=float(gain.mean()),relative_logloss_reduction=float(gain.mean()/losses[before].mean()),
            logloss_reduction_user_bootstrap_95CI=interval(gain[bootstrap].mean(1)),pair_accuracy_gain=float(hgain.mean()),accuracy_gain_95CI=interval(hgain[bootstrap].mean(1)))
    # Descriptive noise reference: one fixed behavior-ID permutation within past popularity bins.
    perm=np.arange(N);rngp=np.random.default_rng(20261020)
    for k in np.unique(bins[catalog]):
        pool=catalog[bins[catalog]==k];perm[pool]=rngp.permutation(pool)
    BP=bcos(perm[aa],perm[bb]);UP=midcdf(cc,BP)
    tri=mask&np.triu(np.ones((20,20),dtype=bool),1)[None]
    def relationstats(bs,us):
        return dict(raw_spearman=float(spearmanr(S[tri],bs[tri]).statistic),CDF_absolute_gap=stats(np.abs(US-us)[tri]),
            opposite_quartile_fraction=float(np.mean(((US[tri]>=.75)&(us[tri]<=.25))|((US[tri]<=.25)&(us[tri]>=.75)))))
    result=dict(dataset=ds,seconds=time.monotonic()-start,train_only=True,DevTest_accessed=False,recommender_trained=False,
        protocol='Fixed preregistration; 75% past time reference; future Train paired target vs pop-matched negative; 5 user-fold OOF proxy',
        reference=dict(cutoff=float(cutoff),max_reference_time=float(ref.time.max()),rows=len(ref),fraction=len(ref)/len(df),items=len(catalog),
            sparse_shape=list(M.shape),sparse_nnz=int(M.nnz),behavior='binary past item-user incidence cosine, not complement or neural embedding',
            calibration_pairs=4096,behavior_cosine_zero_fraction=float(np.mean(cc==0)),semantic_calibration=stats(cs),behavior_calibration=stats(cc)),
        sample=dict(n=n,users=n,inspected=inspected,rejections=rejected,history_length=stats(lens),common_support_history=stats(valid.sum(1)),
            common_support_fraction=float(np.sum(valid)/np.sum(lens)),pop_bin_exact_fraction=float(np.mean(bins[C[:,0]]==bins[C[:,1]])),fallback_samples=fallback,
            manifest_SHA256=hashlib.sha256(np.array([[x[0],x[1],x[3]] for x in samples],dtype=np.int64).tobytes()).hexdigest(),
            histories_SHA256=hashlib.sha256(H.tobytes()).hexdigest(),candidates_SHA256=hashlib.sha256(C.tobytes()).hexdigest()),
        relations=dict(history_pairs=int(tri.sum()),real=relationstats(B,UB),pop_bin_permuted_behavior=relationstats(BP,UP),
            history_disagreement_degree=stats(dis[valid]),mean_candidate_disagreement_feature=stats(fd)),models=models,comparisons=comparisons,
        caveats='Conditional gain in fixed proxy feature class, not new information given full relations, not causal/ranking gains; CI conditional on frozen OOF fits')
    signal.alarm(0);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['beauty','ml-1m'],required=True);print('RESULT_JSON '+json.dumps(main(p.parse_args().dataset),allow_nan=False))