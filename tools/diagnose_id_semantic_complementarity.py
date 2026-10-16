"""Frozen Dev-only error complementarity; no training/fusion/test/output files."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse, hashlib, json, pickle, re, signal, time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata
from models.sequential.LLMMIRec import LLMMIRec as Legacy
from models.sequential.LLMMIRecASPCFSampledSoftmax import LLMMIRecASPCFSampledSoftmax as Strong
from models.sequential.llmmi_utils import load_llm_table
from helpers.BaseRunner import BaseRunner
torch.set_num_threads(2)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rank(s):
    order=(-s).argsort(axis=1)
    return np.argwhere(order==0)[:,1]+1
def ndcg(r,k=5):return (r<=k)/np.log2(r+1)
def ranksum(r):
    return dict(mean=float(r.mean()),median=float(np.median(r)),p25=float(np.quantile(r,.25)),p75=float(np.quantile(r,.75)),p90=float(np.quantile(r,.9)),
        bins={label:int(mask.sum()) for label,mask in [('1-5',r<=5),('6-10',(r>5)&(r<=10)),('11-20',(r>10)&(r<=20)),('21-100',(r>20)&(r<=100)),('101-1001',r>100)]})
def subset(r,s,mask):
    n=int(mask.sum())
    if not n:return dict(n=0)
    return dict(n=n,ID_rank=ranksum(r[mask]),S1_rank=ranksum(s[mask]),S1_minus_ID_N5=float((ndcg(s)-ndcg(r))[mask].mean()))
def compare(pred,sem,lengths,counts,target):
    r=rank(pred);s=rank(sem);a=r<=5;b=s<=5;n=len(r)
    groups={}
    for label,mask in [('A',a&b),('B',a&~b),('C',~a&b),('D',~a&~b)]:
        groups[label]=dict(**subset(r,s,mask),fraction=float(mask.mean()))
    miss=~a;improve=miss&(s<r);stillmiss=improve&(s>5)
    agreement=[];top_equal=[];jac=[]
    for lo in range(0,n,64):
        pp=pred[lo:lo+64];ss=sem[lo:lo+64]
        p=rankdata(pp,axis=1);q=rankdata(ss,axis=1)
        p-=p.mean(1,keepdims=True);q-=q.mean(1,keepdims=True)
        agreement.extend(((p*q).sum(1)/np.sqrt((p*p).sum(1)*(q*q).sum(1))).tolist())
        pt=(-pp).argsort(1)[:,:5];st=(-ss).argsort(1)[:,:5]
        top_equal.extend(np.all(np.sort(pt,axis=1)==np.sort(st,axis=1),axis=1).tolist())
        jac.extend([len(set(x)&set(y))/len(set(x)|set(y)) for x,y in zip(pt,st)])
    strata={}
    specs=dict(history_length=[('1-5',lengths<=5),('6-10',(lengths>5)&(lengths<=10)),('11-20',lengths>10)],
        train_target_count=[('1-5',counts[target]<=5),('6-20',(counts[target]>5)&(counts[target]<=20)),('21-100',(counts[target]>20)&(counts[target]<=100)),('101+',counts[target]>100)])
    for name,bins in specs.items():
        strata[name]=[]
        for label,mask in bins:
            row=dict(bin=label,n=int(mask.sum()),sufficient=int(mask.sum())>=64)
            if row['sufficient']:
                row.update(ID_HR5=float(a[mask].mean()),S1_HR5=float(b[mask].mean()),semantic_only_hits=int((~a&b&mask).sum()),ID_only_hits=int((a&~b&mask).sum()),
                    ID_N5=float(ndcg(r)[mask].mean()),S1_N5=float(ndcg(s)[mask].mean()))
            strata[name].append(row)
    agree=np.array(agreement)
    return dict(ID_metrics=BaseRunner.evaluate_method(pred,[5,10,20],['HR','NDCG']),S1_metrics=BaseRunner.evaluate_method(sem,[5,10,20],['HR','NDCG']),
        ID_rank=ranksum(r),S1_rank=ranksum(s),groups=groups,C_fraction_of_ID_misses=float((~a&b).sum()/miss.sum()),B_fraction_of_ID_hits=float((a&~b).sum()/a.sum()),
        ID_misses=dict(**subset(r,s,miss),S1_improves_rank=int(improve.sum()),improved_but_still_miss=int(stillmiss.sum()),rescued_top5=int((~a&b).sum()),
            S1_worsens_rank=int((miss&(s>r)).sum()),rank_ties=int((miss&(s==r)).sum()),mean_rank_gain=float((r-s)[miss].mean()),median_rank_gain=float(np.median((r-s)[miss]))),
        D_class_rank_improved=int(((~a&~b)&(s<r)).sum()),agreement=dict(candidate_score_spearman_mean=float(agree.mean()),median=float(np.median(agree)),fraction_above_95=float((agree>.95).mean()),
            positive_rank_equal_fraction=float((r==s).mean()),top5_slot_set_equal_fraction=float(np.mean(top_equal)),top5_slot_jaccard_mean=float(np.mean(jac))),
        S1_minus_ID_N5=float((ndcg(s)-ndcg(r)).mean()),strata=strata,
        label_oracle_diagnostic_only=dict(HR5=float((a|b).mean()),N5=float(np.maximum(ndcg(r),ndcg(s)).mean()),warning='Uses true target rank; not deployable, not an achieved model, not selection'),
        positive_ranks=dict(ID=r.tolist(),S1=s.tolist()))

def main(ds):
    start=time.monotonic()
    def timeout(*a):raise TimeoutError('170-second dataset limit; no retry')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(170)
    root=Path(__file__).resolve().parents[1]
    cpfile=root/f'data/{ds}/SeqReader.pkl'
    corpus=pickle.load(open(cpfile,'rb'))
    d=root/f'new_log/aspcf_softmax_llm_controls/{ds}/softmax_k4_id_seed42_20261011T085651Z_llm_controls'
    ref=json.loads((d/'summary.json').read_text());assert ref['status']=='complete'
    cmd=ref['common_command'];args,_=Strong.parse_model_args(argparse.ArgumentParser(allow_abbrev=False)).parse_known_args(cmd)
    args.device=torch.device('cpu');args.model_path=str(root/ref['checkpoint']);args.buffer=0
    assert args.item_encoder=='id' and args.K==4 and args.history_max==20 and args.test_all==0
    strong=Strong(args,corpus);strong.load_state_dict(torch.load(args.model_path,map_location='cpu',weights_only=True),strict=True);strong.eval()
    assert sha(args.model_path)==ref['checkpoint_sha256']
    legacy_cp=root/f'new_model/llmmirec_phase0/{ds}/id/LLMMIRec_id_seed42.pt'
    legacy_log=root/f'new_log/llmmirec_phase0/{ds}/id/LLMMIRec_id_seed42.log';lt=legacy_log.read_text()
    la=Legacy.parse_model_args(argparse.ArgumentParser()).parse_args(['--item_encoder','id','--K','4','--history_max','20','--emb_size','64','--attn_size','64','--dropout','.1','--num_neg','1'])
    la.device=torch.device('cpu');la.model_path=str(legacy_cp);la.buffer=0
    legacy=Legacy(la,corpus);legacy.load_state_dict(torch.load(legacy_cp,map_location='cpu',weights_only=True),strict=True);legacy.eval()
    assert Legacy.Dataset._get_feed_dict is Strong.Dataset._get_feed_dict
    data=Strong.Dataset(strong,corpus,'dev');n=len(data)
    assert n==len(np.unique(data.data['user_id']))
    train=corpus.data_df['train'];agg=train.groupby('user_id').position.agg(['min','max','nunique'])
    for u,p in zip(data.data['user_id'],data.data['position']):
        x=agg.loc[u];assert x['min']==0 and x['max']==p-1 and x['nunique']==p
    z=load_llm_table(str(root/f'data/{ds}/handled/llm_table_pca1536.pkl'),corpus.n_items)[:,:512]
    z=torch.nn.functional.normalize(z,dim=1,eps=1e-12);z[0]=0
    sem=np.empty((n,1001),np.float32);pred={k:np.empty_like(sem) for k in ['BPR_legacy_ID','Softmax_strong_ID']}
    models={'BPR_legacy_ID':legacy,'Softmax_strong_ID':strong}
    paths={'BPR_legacy_ID':legacy_cp,'Softmax_strong_ID':Path(args.model_path)};before={k:sha(p) for k,p in paths.items()}
    hh=hashlib.sha256();cc=hashlib.sha256();lengths=[];targets=[];duplicate_slots=0
    devtime=dict(zip(corpus.data_df['dev'].user_id,corpus.data_df['dev'].time))
    with torch.inference_mode():
        for lo in range(0,n,64):
            records=[data[i] for i in range(lo,min(lo+64,n))];feed=data.collate_batch(records)
            h=feed['history_items'];c=feed['item_id'];lens=feed['lengths'];assert c.shape[1]==1001 and int(lens.max())<=20
            for record in records:
                assert (record['history_times']<=devtime[record['user_id']]).all()
            fullh=np.pad(h.numpy(),((0,0),(0,20-h.shape[1])))
            hh.update(fullh.tobytes());cc.update(c.numpy().tobytes());lengths.extend(lens.tolist());targets.extend(c[:,0].tolist())
            duplicate_slots+=sum(1000-len(np.unique(row[1:])) for row in c.numpy())
            for name,model in models.items():
                y=model(feed)['prediction'];assert bool(torch.isfinite(y).all());pred[name][lo:lo+len(records)]=y.numpy()
            mu=torch.nn.functional.normalize(z[h].sum(1)/lens[:,None],dim=1,eps=1e-12)
            sem[lo:lo+len(records)]=torch.bmm(mu[:,None],z[c].transpose(1,2))[:,0].numpy()
    assert all(sha(p)==before[k] for k,p in paths.items())
    counts=np.bincount(train.item_id.to_numpy(),minlength=corpus.n_items)
    results={k:compare(v,sem,np.array(lengths),counts,np.array(targets)) for k,v in pred.items()}
    for name,old in [('BPR_legacy_ID',re.findall(r'Best Iter\(dev\)=.*',lt)[-1]),('Softmax_strong_ID',ref['best_dev'])]:
        expected=float(re.search(r'NDCG@5:([\d.]+)',old).group(1));assert abs(results[name]['ID_metrics']['NDCG@5']-expected)<.00011,(name,expected,results[name]['ID_metrics'])
    meta=dict(dataset=ds,n=n,users=n,identity_rule='Fixed historical phase0 ID BPR seed42; fixed 20261011 pureID Softmax control seed42; never choose by current results',
        BPR_config='LLMMIRec phase0 ID, learnable Query, batch256/lr.001/K4/H20/dropout.1; historical class, not frozen ASPCF single-factor ablation',
        Softmax_config='LLMMIRecASPCFSampledSoftmax ID, batch1024/lr.004 Beauty .001 ML/K4/H20/dropout.1; strong reference, not BPR',
        checkpoints={k:dict(path=str(paths[k]),sha256=before[k],unchanged=True,parameters=models[k].count_variables()) for k in models},
        source_best=dict(BPR_legacy_ID=re.findall(r'Best Iter\(dev\)=.*',lt)[-1],Softmax_strong_ID=ref['best_dev']),
        history_hash=hh.hexdigest(),candidate_hash=cc.hexdigest(),dev_users_hash=hashlib.sha256(np.array(data.data['user_id'],dtype=np.int64).tobytes()).hexdigest(),
        retained_duplicate_negative_slots=duplicate_slots,history_lengths=dict(mean=float(np.mean(lengths)),min=min(lengths),max=max(lengths)),
        space='raw PCA1536 first512 item L2 then normalized mean; only S1',protocol='Original SequentialModel.Dataset Dev, full1001 slots, duplicate negatives retained, original BaseRunner np.argsort; strict history positions are Train',
        predictions_cached=False,ASPCF_third_column='No aligned raw prediction cache found in related directories; not inferred from aggregate metrics',results=results)
    meta['seconds']=time.monotonic()-start;signal.alarm(0);return meta

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['beauty','ml-1m'],required=True)
    print(json.dumps(main(p.parse_args().dataset)))
