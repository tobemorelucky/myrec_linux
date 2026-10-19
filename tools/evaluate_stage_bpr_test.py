"""One authorized Test acceptance of ten fixed best-Dev checkpoints; no training/Dev inference."""
import os,sys,json,pickle,argparse,hashlib,time,re,csv,logging,signal,gc,subprocess
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
import helpers.BaseRunner as runner_module
from helpers.BaseRunner import BaseRunner
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF as Base
from models.sequential.LLMMIRecASPCFReadoutBPR import LLMMIRecASPCFReadoutBPR as Joint
from tools.diagnose_interest_deletion_dev import configure
from tools.run_frozen_interest_readout import state_hash
from utils import utils
FILES=['tools/evaluate_stage_bpr_test.py','models/sequential/LLMMIRecASPCF.py',
 'models/sequential/LLMMIRecASPCFReadoutBPR.py','models/sequential/LLMMIRecASPCFRefineBPR.py',
 'tools/run_frozen_interest_readout.py','tools/diagnose_interest_deletion_dev.py',
 'tools/diagnose_aspcf_frozen_dev_scoring.py','models/sequential/LLMMIRecASPCFDualViewRefine.py',
 'models/sequential/LLMMIRecASPCFRefineControls.py','models/sequential/llmmi_components.py',
 'models/sequential/llmmi_utils.py','models/BaseModel.py','helpers/BaseRunner.py','helpers/SeqReader.py','helpers/BaseReader.py','main.py']
TAG='20261019_aspcf_A_P0P1_seed42'
OUT=ROOT/'new_log/stage_test_acceptance'/TAG

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def dump(path,obj):
 p=Path(path);assert p.resolve().is_relative_to(ROOT) and not p.exists()
 with p.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
def describe(a):
 a=np.asarray(a,dtype=float)
 return dict(min=float(a.min()),median=float(np.median(a)),mean=float(a.mean()),max=float(a.max()),p10=float(np.quantile(a,.1)),p90=float(np.quantile(a,.9)))
def job(ds,name):
 if name in ['P0','P1','MLP_A']:
  if name=='MLP_A':root=ROOT/'new_log/refine_bpr_round1/20261018_dualview_bpr_k4_seed42_round1'/ds/'B1'
  else:root=ROOT/'new_log/p0p1_joint_bpr/20261019_p0p1_joint_k4_seed42'/ds/name
  c=json.loads((root/'config.json').read_text());s=json.loads((root/'summary.json').read_text())
  assert s['success'] and s['best_checkpoint_loaded_verified']
  if name=='MLP_A':cls,a,path,expected,dev=configure(ds,4,'A')
  else:
   cls=Joint;a=argparse.Namespace(**c['effective_args']);path=Path(a.model_path);expected=s['checkpoint_sha256'];dev=s['dev']
  best=s['checkpoint_saved_epoch'];provenance=dict(config=str(root/'config.json'),summary=str(root/'summary.json'),source_sha256=c['source_sha256'])
  assert all(sha(ROOT/f)==h for f,h in c['source_sha256'].items())
 else:
  K=int(name[-1]);cls,a,path,expected,dev=configure(ds,K,'base')
  stem='llmmirec_aspcf_K2' if K==2 else 'llmmirec_aspcf_phase2'
  log=ROOT/'new_log'/stem/ds/'seed42/LLMMIRecASPCF_seed42.log'
  bests=re.findall(r'Best Iter\(dev\)=\s*(\d+)',log.read_text());assert len(bests)>=1
  best=int(bests[-1]);provenance=dict(historical_log=str(log),log_sha256=sha(log),
   verified_dev_record=str(ROOT/'new_log/aspcf_frozen_dev_scoring/20261018_s0_s1_s2_seed42'/(ds+'_K'+str(K)+'.json')),
   asset_provenance='Current asset SHA matched archived BPR run manifests and same SeqReader cache used for verified historical Dev; original August training did not archive an independent full asset manifest.')
 assert path.is_file() and sha(path)==expected
 assert a.K in [2,4] and a.history_max==20 and a.num_neg==1 and a.lambda_relation==.01 and a.test_all==0
 original={k:str(v) if k=='device' else v for k,v in vars(a).items()}
 a.device=torch.device('cuda:0');a.model_path=str(path);a.llm_emb_path=str(ROOT/'data'/ds/'handled/llm_table_pca1536.pkl')
 a.train=0;a.load=1;a.dev_only=0;a.test_epoch=-1
 return dict(dataset=ds,name=name,K=a.K,seed=42,best_epoch=best,checkpoint=str(path),checkpoint_sha256=expected,
  historical_dev=dev,provenance=provenance,training_config=original,evaluation_config={k:str(v) if k=='device' else v for k,v in vars(a).items()}),cls,a

def protocol(corpus,ds,model):
 report={};users={};rows={}
 for phase in ['dev','test']:
  d=Base.Dataset(model,corpus,phase);U=[];L=[];extras=[];targetdups=[];within=[];equal_times=0
  hashes={k:hashlib.sha256() for k in ['users','history','candidates']}
  counts=corpus.data_df['train'].groupby('user_id').size().to_dict()
  for i in range(len(d)):
   x=d._get_feed_dict(i);u=int(x['user_id']);pos=int(d.data['position'][i]);target=int(x['item_id'][0]);tm=d.data['time'][i]
   seq=corpus.user_his[u]
   assert seq[pos][0]==target and seq[pos][1]==tm
   history=seq[max(0,pos-20):pos]
   assert len(history)==x['lengths'] and all(t<=tm for _,t in history)
   assert np.array_equal(x['history_items'],np.array([it for it,_ in history]))
   candidates=x['item_id'];assert len(candidates)==1001
   assert np.all((candidates>0)&(candidates<corpus.n_items))
   neg=candidates[1:];extras.append(1000-len(np.unique(neg)));targetdups.append(int((neg==target).sum()))
   U.append(u);L.append(len(history));within.append(pos-counts[u])
   equal_times+=bool(history and history[-1][1]==tm)
   for name,ks in [('users',['user_id']),('history',['history_items','lengths']),('candidates',['item_id'])]:
    for k in ks:
     v=np.asarray(x[k]);hashes[name].update(k.encode());hashes[name].update(str(v.shape).encode());hashes[name].update(v.tobytes())
  U=np.asarray(U);L=np.asarray(L);users[phase]=np.unique(U);rows[phase]=U
  report[phase]=dict(rows=len(U),users=len(users[phase]),multiple_rows_per_user=len(U)-len(users[phase]),candidates_per_row=1001,
   duplicate_negative_extra_slots=sum(extras),duplicate_negative_extra_slot_fraction=float(sum(extras)/(len(U)*1000)),
   rows_with_duplicate_negatives=int(np.count_nonzero(extras)),rows_with_duplicate_negatives_fraction=float(np.mean(np.asarray(extras)>0)),
   positive_repeated_in_negative_slots=int(sum(targetdups)),history_lengths=describe(L),history_at_H20_fraction=float(np.mean(L==20)),
   history_groups={label:int(mask.sum()) for label,mask in [('1-3',L<=3),('4-9',(L>=4)&(L<=9)),('10-19',(L>=10)&(L<20)),('20',L==20)]},
   position_minus_train_count_values={str(v):int(n) for v,n in zip(*np.unique(within,return_counts=True))},
   last_history_equal_target_timestamp_rows=int(equal_times),prefix_verified=True,input_row_hash={k:h.hexdigest() for k,h in hashes.items()})
 overlap=np.intersect1d(users['dev'],users['test']);report['user_overlap']=dict(count=len(overlap),dev_fraction=len(overlap)/len(users['dev']),test_fraction=len(overlap)/len(users['test']))
 report['history_note']='Original SeqReader prefix [:position], capped H20. Test can include already observed Dev interaction; no current/future target input. Equal timestamps follow cached stable order, not strict real-time precedence.'
 return report

def bootstrap(delta,user):
 uid,idx=np.unique(user,return_inverse=True);s=np.bincount(idx,weights=delta);n=np.bincount(idx)
 pairs,counts=np.unique(np.column_stack([s,n]),axis=0,return_counts=True)
 gen=np.random.default_rng(20261019);draw=gen.multinomial(len(uid),counts/counts.sum(),size=2000)
 means=(draw@pairs[:,0])/(draw@pairs[:,1]);ci=np.quantile(means,[.025,.975])
 return dict(mean_delta=float(np.mean(delta)),cluster_bootstrap95=ci.tolist(),users=len(uid),rows=len(delta),replicates=2000,seed=20261019,
  note='Paired user clusters; all rows of a selected user retained. Fixed models and fixed candidate sample; not training-seed uncertainty or Dev-selection adjustment.')

def main():
 assert not OUT.exists() and OUT.resolve().is_relative_to(ROOT)
 OUT.mkdir(parents=True,exist_ok=False);logging.basicConfig(filename=OUT/'evaluation.log',level=logging.INFO,format='%(asctime)s %(message)s')
 logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
 start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('600-second bounded Test audit')));signal.alarm(600)
 gpu=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).split(',')
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 assert gpu[0].strip() not in apps and float(gpu[1])>7000,'GPU1 busy; do not interfere'
 source={f:sha(ROOT/f) for f in FILES};all_jobs=[];missing=[];protocols={};assets={};results=[];comparisons={}
 constructors={}
 for ds in ['beauty','ml-1m']:
  configs=[ROOT/'new_log'/fam/rid/ds/arm/'config.json' for fam,rid,arm in [
   ('refine_bpr_round1','20261018_dualview_bpr_k4_seed42_round1','B1'),('p0p1_joint_bpr','20261019_p0p1_joint_k4_seed42','P0')]]
  archived=[json.loads(p.read_text())['asset_sha256'] for p in configs];assert archived[0]==archived[1]
  assert all(sha(ROOT/f)==h for f,h in archived[0].items())
  assets[ds]=dict(verified_training_assets=archived[0],test_csv_sha256=sha(ROOT/'data'/ds/'test.csv'))
  for name in ['ASPCF_K2','ASPCF_K4','MLP_A','P0','P1']:
   try:j,cls,a=job(ds,name)
   except (FileNotFoundError,AssertionError,KeyError) as e:
    missing.append(dict(dataset=ds,name=name,error=repr(e)));continue
   all_jobs.append(j);constructors[(ds,name)]=(cls,a)
 dump(OUT/'manifest.json',dict(jobs=all_jobs,missing=missing,asset_sha256=assets,source_sha256=source,
  source_head='d5d1a1633f74b11f463eaa2e3937c92ddb316d70',train_executed=False,dev_inference=False,test_authorized=True))
 for ds in ['beauty','ml-1m']:
  with (ROOT/'data'/ds/'SeqReader.pkl').open('rb') as f:c=pickle.load(f)
  # Cached CSV targets/candidates must match immutable raw files; no regeneration.
  import pandas as pd,ast
  for phase in ['train','dev','test']:
   raw=pd.read_csv(ROOT/'data'/ds/(phase+'.csv'),sep='\t');cache=c.data_df[phase]
   assert len(raw)==len(cache)
   for k in ['user_id','item_id','time']:assert np.array_equal(raw[k].to_numpy(),cache[k].to_numpy())
   if phase!='train':
    for ra,ca in zip(raw.neg_items,cache.neg_items):assert np.array_equal(np.asarray(ast.literal_eval(ra) if isinstance(ra,str) else ra),np.asarray(ca))
  predictions={};userids=None;expected_hash=None
  for j in [j for j in all_jobs if j['dataset']==ds]:
   name=j['name'];cls,a=constructors[(ds,name)];utils.init_seed(42);t=time.monotonic()
   model=cls(a,c).to(a.device);state=torch.load(j['checkpoint'],map_location=a.device,weights_only=True)
   llmkey='backbone.item_encoder.llm_table' if j['name'] in ['P0','P1'] else 'item_encoder.llm_table'
   assert llmkey not in state,'Unexpected persistent LLM table: inspect before evaluation'
   # Original LLM table is persistent=False: asset SHA, not checkpoint tensor, establishes identity.
   parameter_count=sum(p.numel() for p in model.parameters())
   model.load_state_dict(state,strict=True);del state;model.eval();model.requires_grad_(False)
   before=state_hash(model.state_dict())
   if ds not in protocols:
    protocols[ds]=protocol(c,ds,model);dump(OUT/(ds+'_protocol.json'),protocols[ds])
   data=cls.Dataset(model,c,'test');data.prepare();runner=BaseRunner(a)
   assert a.train==0 and a.load==1 and a.dev_only==0 and a.test_epoch==-1
   # Intercept only yielded batches; original BaseRunner.predict is unchanged.
   original=runner_module.DataLoader;hashes={k:hashlib.sha256() for k in ['users','history','candidates']};U=[]
   class RecordedLoader(original):
    def __iter__(self):
     assert self.dataset.phase=='test'
     for b in super().__iter__():
      assert b['item_id'].shape[1]==1001;U.append(b['user_id'].numpy().copy())
      for name2,keys in [('users',['user_id']),('history',['history_items','lengths']),('candidates',['item_id'])]:
       for k in keys:
        x=b[k].numpy();hashes[name2].update(k.encode());hashes[name2].update(str(x.shape).encode());hashes[name2].update(x.tobytes())
      yield b
   runner_module.DataLoader=RecordedLoader;torch.cuda.reset_peak_memory_stats()
   try:pred=runner.predict(data)
   finally:runner_module.DataLoader=original
   assert pred.shape==(len(data),1001) and np.isfinite(pred).all()
   assert before==state_hash(model.state_dict()) and sha(j['checkpoint'])==j['checkpoint_sha256']
   actual_hash={k:h.hexdigest() for k,h in hashes.items()};users=np.concatenate(U)
   if expected_hash is None:expected_hash=actual_hash;userids=users
   else:assert actual_hash==expected_hash and np.array_equal(userids,users)
   metrics={k:float(v) for k,v in BaseRunner.evaluate_method(pred,[5,10,20],['HR','NDCG']).items()}
   order=(-pred).argsort(axis=1);rank=np.argwhere(order==0)[:,1]+1;ndcg=(rank<=5)/np.log2(rank+1)
   predictions[name]=ndcg
   # Store raw scores once for paired analyses, not another forward.
   p=OUT/(ds+'_'+name+'_test_predictions.npz');assert not p.exists()
   np.savez(p,prediction=pred,user_id=users,positive_rank=rank,NDCG5=ndcg)
   result=dict(**j,test=metrics,test_minus_dev_NDCG5=metrics['NDCG@5']-j['historical_dev']['NDCG@5'],rows=len(data),users=len(np.unique(users)),
    parameters=parameter_count,trainable_parameters_during_test=0,external_llm_asset_sha_verified=True,seconds=time.monotonic()-t,input_sha256=actual_hash,predictions=str(p),
    checkpoint_unchanged=True,model_state_unchanged=True,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
    scoring='Original frozen model.forward via original BaseRunner.predict; original evaluate_method; 1001 slots untouched')
   results.append(result);dump(OUT/(ds+'_'+name+'_test.json'),result);print('TEST_RESULT '+json.dumps(dict(dataset=ds,model=name,test=metrics,gap=result['test_minus_dev_NDCG5'])),flush=True)
   del model,data,runner,pred,order;gc.collect();torch.cuda.empty_cache()
  comparisons[ds]={}
  for left,right in [('P0','P1'),('P0','MLP_A'),('P0','ASPCF_K4')]+([('P0','ASPCF_K2')] if ds=='beauty' else []):
   if left in predictions and right in predictions:comparisons[ds][left+'-'+right]=bootstrap(predictions[left]-predictions[right],userids)
  dump(OUT/(ds+'_paired_bootstrap.json'),comparisons[ds]);del c,predictions;gc.collect()
 assert all(sha(ROOT/f)==h for f,h in source.items())
 report=dict(results=results,missing=missing,protocol=protocols,comparisons=comparisons,source_sha256=source,asset_sha256=assets,
  total_seconds=time.monotonic()-start,source_unchanged=True,train_executed=False,dev_inference=False,test_executed=True)
 dump(OUT/'results.json',report);dump(OUT/'complete.json',dict(status='complete',evaluated=len(results),missing=missing,train=False,dev_inference=False,test=True,seconds=report['total_seconds']))
 with (OUT/'six_metrics.csv').open('x',newline='') as f:
  writer=csv.writer(f);writer.writerow(['dataset','model','K','best_epoch','phase','HR@5','NDCG@5','HR@10','NDCG@10','HR@20','NDCG@20'])
  for r in results:
   for phase,key in [('Dev archived','historical_dev'),('Test','test')]:writer.writerow([r['dataset'],r['name'],r['K'],r['best_epoch'],phase]+[r[key][k] for k in ['HR@5','NDCG@5','HR@10','NDCG@10','HR@20','NDCG@20']])
 print('TEST_ACCEPTANCE_COMPLETE '+str(OUT/'results.json'),flush=True)
if __name__=='__main__':main()
