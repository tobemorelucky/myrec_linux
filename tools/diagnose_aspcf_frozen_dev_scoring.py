"""One frozen Dev pass per historical checkpoint. No Train/Test execution or model changes."""
import os,sys,json,time,hashlib,pickle,argparse,signal,csv,itertools,math
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='1'
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from helpers.BaseRunner import BaseRunner
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF as Model
from utils import utils
SPEC={('beauty',2):('llmmirec_aspcf_K2','e123e0f9f3b5c7825ab3cd6e08982de0b5637880f2e34a4e159c43568aff641d',.1332),('beauty',4):('llmmirec_aspcf_phase2','463ecf71b88756f4323428ee77c827335a7eae482857c52e90d6a0043c5c1100',.1300),('ml-1m',2):('llmmirec_aspcf_K2','1285b09e5d98526a50f51a2dda3a22edddb21613a78c25392ab6c49f611b9d45',.2298),('ml-1m',4):('llmmirec_aspcf_phase2','2ffa35f1a5f5979e43c4f264b5a883cc7e3e177097f946a498a93126f06e331b',.2271)}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(a):
 a=np.asarray(a,dtype=np.float64).reshape(-1);return dict(mean=float(a.mean()),median=float(np.median(a)),p10=float(np.quantile(a,.1)),p90=float(np.quantile(a,.9)),max=float(a.max()))
def row_metrics(pred):
 order=(-pred).argsort(axis=1);rank=np.argwhere(order==0)[:,1]+1;ndcg=(rank<=5)/np.log2(rank+1)
 return order,rank,ndcg

def run(ds,K,outdir):
 signal.alarm(170);started=time.time();torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
 stem,expected,ndcg_ref=SPEC[(ds,K)];source=ROOT/'new_model'/stem/ds/'seed42/LLMMIRecASPCF_seed42.pt'
 assert source.is_file() and sha(source)==expected
 parser=BaseRunner.parse_runner_args(Model.parse_model_args(argparse.ArgumentParser()))
 logfile=ROOT/'new_log'/stem/ds/'seed42/LLMMIRecASPCF_seed42.log';old={}
 for line in logfile.read_text().splitlines():
  cols=line.split('|')
  if len(cols)>=2:old[cols[0].strip()]=cols[1].strip()
 tokens=[]
 for a in parser._actions:
  if a.dest in old and a.dest!='help':tokens.extend([a.option_strings[0],old[a.dest]])
 args=parser.parse_args(tokens);args.device=torch.device('cuda:0');args.buffer=0;args.llm_emb_path=str(ROOT/'data'/ds/'handled/llm_table_pca1536.pkl')
 assert args.K==K and args.history_max==20 and args.num_neg==1 and args.lambda_relation==.01 and args.test_all==0
 with (ROOT/'data'/ds/'SeqReader.pkl').open('rb') as f:corpus=pickle.load(f)
 corpus.data_df={'dev':corpus.data_df['dev']};assert corpus.data_df['dev'].user_id.is_unique
 model=Model(args,corpus).to(args.device);model.load_state_dict(torch.load(source,map_location=args.device,weights_only=True),strict=True);model.eval()
 data=model.Dataset(model,corpus,'dev');assert all(len(n)==1000 for n in data.data['neg_items'])
 loader=DataLoader(data,batch_size=args.eval_batch_size,shuffle=False,num_workers=0,collate_fn=data.collate_batch)
 prediction=[[],[],[]];lengths=[];users=[];weights=[];vector_diff=[];score_diff=[];winner_all=np.zeros(K,np.int64);winner_unique=np.zeros(K,np.int64);winner_top5=np.zeros(K,np.int64);ties=0;slots=0;identity_max=0.;hashes={k:hashlib.sha256() for k in ['users','history','candidates']}
 torch.cuda.reset_peak_memory_stats()
 with torch.inference_mode():
  for batch in loader:
   for key,keys in [('users',['user_id']),('history',['history_items','lengths']),('candidates',['item_id'])]:
    for name in keys:
     x=batch[name].numpy();hashes[key].update(name.encode());hashes[key].update(str(x.shape).encode());hashes[key].update(x.tobytes())
   lengths.append(batch['lengths'].numpy());users.append(batch['user_id'].numpy());b=utils.batch_to_gpu(batch,args.device)
   o=model(b,True);v=o['interest_vectors'];w=o['interest_weights'];e=o['candidate_vectors']
   # Direct sums per head: do not modify norms, temperature, item IDs or slots.
   z=(v[:,:,None,:]*e[:,None,:,:]).sum(-1)
   s0=o['prediction'];s1,heads=z.max(1);strongest=w.argmax(1);s2=z[torch.arange(len(w),device=w.device),strongest]
   check=(z*w[:,:,None]).sum(1);identity_max=max(identity_max,float((check-s0).abs().max()))
   assert torch.allclose(check,s0,atol=2e-5,rtol=2e-5)
   prediction[0].append(s0.cpu().numpy());prediction[1].append(s1.cpu().numpy());prediction[2].append(s2.cpu().numpy());weights.append(w.cpu().numpy())
   pairv=[];pairs=[]
   centered=z-z.mean(-1,keepdim=True)
   for i,j in itertools.combinations(range(K),2):
    pairv.append((v[:,i]-v[:,j]).norm(dim=-1)/(v[:,[i,j]].norm(dim=-1).mean(-1).clamp_min(1e-30)))
    pairs.append((centered[:,i]-centered[:,j]).square().mean(-1).sqrt()/centered[:,[i,j]].square().mean(-1).sqrt().mean(-1).clamp_min(1e-30))
   vector_diff.append(torch.stack(pairv,1).mean(1).cpu().numpy());score_diff.append(torch.stack(pairs,1).mean(1).cpu().numpy())
   top=z.topk(2,dim=1).values;unique=top[:,0]!=top[:,1]
   winner_all+=torch.bincount(heads.flatten(),minlength=K).cpu().numpy();winner_unique+=torch.bincount(heads[unique],minlength=K).cpu().numpy();ties+=int((~unique).sum());slots+=heads.numel()
   # Match Runner's exact NumPy argsort, not a separate GPU topk tie convention.
   oi=(-s1.cpu().numpy()).argsort(axis=1)[:,:5];ha=heads.cpu().numpy();winner_top5+=np.bincount(np.take_along_axis(ha,oi,1).flatten(),minlength=K)
 del z,e,o,v,model,loader
 preds=[np.concatenate(x) for x in prediction];L=np.concatenate(lengths);U=np.concatenate(users);W=np.concatenate(weights)
 orders=[];ranks=[];ndcgs=[];metrics=[]
 for pred in preds:
  metrics.append({k:float(v) for k,v in BaseRunner.evaluate_method(pred,[5,10,20],['HR','NDCG']).items()});o,r,n=row_metrics(pred);orders.append(o);ranks.append(r);ndcgs.append(n)
 assert abs(metrics[0]['NDCG@5']-ndcg_ref)<.000051,('S0 historical metric mismatch',ds,K,metrics[0])
 comparisons={}
 for i in [1,2]:
  delta=preds[i]-preds[0];difference=ndcgs[i]-ndcgs[0];center=preds[0]-preds[0].mean(1,keepdims=True)
  comparisons['S'+str(i)]=dict(ndcg5_minus_S0=metrics[i]['NDCG@5']-metrics[0]['NDCG@5'],score_absolute_difference=stats(np.mean(abs(delta),axis=1)),score_RMS_correction_ratio=stats(np.sqrt(np.mean(delta**2,axis=1))/np.maximum(np.sqrt(np.mean(preds[0]**2,axis=1)),1e-30)),centered_score_correction_ratio=stats(np.sqrt(np.mean((delta-delta.mean(1,keepdims=True))**2,axis=1))/np.maximum(np.sqrt(np.mean(center**2,axis=1)),1e-30)),any_slot_order_change=float(np.mean(np.any(orders[i]!=orders[0],axis=1))),top5_slot_set_change=float(np.mean(np.any(np.sort(orders[i][:,:5],axis=1)!=np.sort(orders[0][:,:5],axis=1),axis=1))),positive_rank_change=float(np.mean(ranks[i]!=ranks[0])),ndcg5_win_loss_tie=[int((difference>0).sum()),int((difference<0).sum()),int((difference==0).sum())])
 groups=[]
 for lo,hi in [(1,5),(6,10),(11,20)]:
  take=(L>=lo)&(L<=hi)
  if not take.any():continue
  groups.append(dict(history=f'{lo}-{hi}',users=int(take.sum()),scores={f'S{i}':{k:float(v) for k,v in BaseRunner.evaluate_method(pred[take],[5,10,20],['HR','NDCG']).items()} for i,pred in enumerate(preds)},ndcg5_win_loss_tie={f'S{i}':[int(((ndcgs[i]-ndcgs[0])[take]>0).sum()),int(((ndcgs[i]-ndcgs[0])[take]<0).sum()),int(((ndcgs[i]-ndcgs[0])[take]==0).sum())] for i in [1,2]}))
 result=dict(dataset=ds,K=K,seed=42,loss='historical original single-negative BPR + relation .01; no loss execution here',phase='dev',users=len(U),candidates=1001,duplicate_slots_preserved=True,checkpoint=str(source),checkpoint_sha256=expected,config={k:str(v) if k=='device' else v for k,v in vars(args).items()},metrics={f'S{i}':m for i,m in enumerate(metrics)},comparisons=comparisons,history_groups=groups,attention_not_modified=True,interest_pair_relative_norm_difference=stats(np.concatenate(vector_diff)),interest_pair_centered_candidate_score_RMS_difference=stats(np.concatenate(score_diff)),head_usage=dict(S0_weight_mean=W.mean(0).tolist(),S2_user_argmax_counts=np.bincount(W.argmax(1),minlength=K).tolist(),S1_all_slot_argmax_counts=winner_all.tolist(),S1_unique_max_slot_counts=winner_unique.tolist(),S1_top5_argmax_counts=winner_top5.tolist(),S1_exact_head_tie_slot_fraction=ties/slots,tie_rule='torch.max first head for exact ties; unique-only counts also given'),input_sha256={k:x.hexdigest() for k,x in hashes.items()},weighted_sum_numerical_identity_max_abs=identity_max,rank_change_definition='slot ordering using original NumPy argsort; exact score ties may affect raw ordering. Positive proportional scaling alone preserves ranking except finite precision/ties.',seconds=time.time()-started,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,checkpoint_unchanged=sha(source)==expected,train_executed=False,test_executed=False)
 assert result['checkpoint_unchanged']
 csvpath=outdir/(ds+'_K'+str(K)+'_dev_users.csv');assert not csvpath.exists()
 with csvpath.open('x',newline='') as f:
  writer=csv.writer(f);writer.writerow(['user_id','history_length','S0_rank','S1_rank','S2_rank','S0_NDCG5','S1_NDCG5','S2_NDCG5'])
  writer.writerows(zip(U,L,*ranks,*ndcgs))
 path=outdir/(ds+'_K'+str(K)+'.json');assert not path.exists();path.write_text(json.dumps(result,indent=2));print(json.dumps(dict(dataset=ds,K=K,metrics=result['metrics'],comparisons=comparisons,vector_difference=result['interest_pair_relative_norm_difference'],score_difference=result['interest_pair_centered_candidate_score_RMS_difference'],head_usage=result['head_usage'],seconds=result['seconds'])),flush=True)
 del preds,prediction,orders,data,corpus;torch.cuda.empty_cache();signal.alarm(0)
 return result
if __name__=='__main__':
 signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError('170s per checkpoint bound')))
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);assert out.resolve().is_relative_to(ROOT) and not out.exists();out.mkdir(parents=True)
 files=['models/sequential/LLMMIRecASPCF.py','models/sequential/llmmi_components.py','models/BaseModel.py','helpers/BaseRunner.py','tools/diagnose_aspcf_frozen_dev_scoring.py'];sources={f:sha(ROOT/f) for f in files}
 results=[run(ds,K,out) for ds in ['beauty','ml-1m'] for K in [2,4]]
 for ds in ['beauty','ml-1m']:
  x=[r for r in results if r['dataset']==ds];assert x[0]['input_sha256']==x[1]['input_sha256'],'K2/K4 input mismatch'
 assert all(sha(ROOT/f)==h for f,h in sources.items())
 (out/'complete.json').write_text(json.dumps(dict(results=results,source_sha256=sources,source_unchanged=True,train_executed=False,test_executed=False),indent=2))
