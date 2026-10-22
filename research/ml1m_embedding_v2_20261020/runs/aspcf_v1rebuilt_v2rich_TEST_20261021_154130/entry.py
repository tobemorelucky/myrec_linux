"""One-time authorized Test of fixed best-Dev V1-Rebuilt/V2; no training."""
import os,sys,pathlib,json,pickle,hashlib,time,datetime,importlib.util,gc
ROOT=pathlib.Path('/home/yyx/hzgcode/pom2.0');os.chdir(ROOT)
os.environ['CUDA_VISIBLE_DEVICES']='0'
sys.path.insert(0,str(ROOT))
import numpy as np,torch
torch.set_num_threads(4)
W=ROOT/'research/ml1m_embedding_v2_20261020';OUT=pathlib.Path(sys.argv[1])
spec=importlib.util.spec_from_file_location('existing_v2_entry',W/'ops/run_v2_rich_aspcf_softmax_20261021.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
from helpers.BaseRunner import BaseRunner
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def save(p,data):
 with p.open('x') as f:json.dump(data,f,indent=2)
results={}
for label,name in [('V1_Rebuilt','p920_aspcf_ss_v1_rebuilt_k4_seed42_20261020_214225'),('V2_Rich','p920_aspcf_ss_v2_rich_k4_seed42_20261021_145239')]:
 run=W/'runs'/name;complete=json.loads((run/'COMPLETE.json').read_text());prov=json.loads((run/'provenance.json').read_text())
 assert complete['exit_code']==0 and sha(run/'best.pt')==complete['checkpoint_sha256']
 assert all(sha(ROOT/p)==h for p,h in prov['sources'].items())
 c=json.loads((run/'config.json').read_text())
 c.update(gpu='0',train=0,load=1,dev_only=0,test_epoch=-1)
 c['log_file']=str(OUT/(label+'_eval.log'))
 expected=prov.get('embedding_sha256',prov.get('embedding_pkl_sha256'));assert sha(c['llm_emb_path'])==expected
 save(OUT/(label+'_config.json'),c)
 started=time.monotonic()
 model,corpus,args,Model=entry.build(c)
 model.load_model();model.eval()
 with open(c['llm_emb_path'],'rb') as f:table=np.asarray(pickle.load(f),dtype=np.float32)
 assert torch.equal(model.item_encoder.llm_table.detach().cpu(),torch.from_numpy(table))
 assert not model.test_all
 ds=Model.Dataset(model,corpus,'test');ds.prepare()
 lengths=[];duplicates=0;users=[]
 for i in range(len(ds)):
  feed=ds[i];ids=np.asarray(feed['item_id'])
  assert ids.shape==(1001,) and ids[0]>0
  duplicates+=1000-len(set(map(int,ids[1:])))
  lengths.append(int(feed['lengths']));users.append(int(feed['user_id']))
 assert max(lengths)<=20
 runner=BaseRunner(args);torch.cuda.reset_peak_memory_stats()
 predictions=runner.predict(ds)
 assert predictions.shape==(len(ds),1001) and np.isfinite(predictions).all()
 metrics=runner.evaluate_method(predictions,[5,10,20],['HR','NDCG'])
 assert sha(run/'best.pt')==complete['checkpoint_sha256'] and sha(c['llm_emb_path'])==expected
 result={'model':c['model_name'],'embedding_version':label,'phase':'Test','seed':42,'K':4,
  'best_epoch':complete.get('best_epoch',47),'checkpoint':str(run/'best.pt'),'checkpoint_sha256':complete['checkpoint_sha256'],
  'embedding_path':c['llm_emb_path'],'embedding_sha256':expected,'Test':metrics,'parameter_count':model.count_variables(),
  'test_samples':len(ds),'test_users':len(set(users)),'candidate_slots':1001,
  'negative_duplicate_slots':duplicates,'negative_duplicate_slot_fraction':duplicates/(1000*len(ds)),
  'history_length_min':min(lengths),'history_length_median':float(np.median(lengths)),'history_length_max':max(lengths),
  'sources_sha256':prov['sources'],'test_csv_sha256':sha(ROOT/'data/ml-1m/test.csv'),
  'SeqReader_sha256':sha(ROOT/'data/ml-1m/SeqReader.pkl'),'seconds':time.monotonic()-started,
  'time':datetime.datetime.now().astimezone().isoformat(),'optimizer_updates':0,
  'checkpoint_selection':'fixed historical best-Dev; no Test selection','predictions_evaluated_once':True}
 save(OUT/(label+'_TEST.json'),result);results[label]=result
 del model,ds,runner,predictions;gc.collect();torch.cuda.empty_cache()
save(OUT/'COMPLETE.json',{'passed':True,'models':results,'training_started':False})
print(json.dumps({'passed':True,'out':str(OUT),'models':results}))
