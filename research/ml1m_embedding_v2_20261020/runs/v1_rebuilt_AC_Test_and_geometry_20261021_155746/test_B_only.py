"""B Test acceptance; minimal reuse of A/C Test entry; reuse original entry.build and Runner."""
import pathlib,json,sys,os,importlib.util,pickle,time,hashlib,gc
ROOT=pathlib.Path('/home/yyx/hzgcode/pom2.0');os.chdir(ROOT);sys.path.insert(0,str(ROOT));os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['V1_ENTRY_PHASE']='TEST'
import numpy as np,torch
torch.set_num_threads(2)
W=ROOT/'research/ml1m_embedding_v2_20261020';OUT=pathlib.Path(sys.argv[1]);label=sys.argv[2]
spec=importlib.util.spec_from_file_location('existing_control',W/'ops/v1_controls_entry_20261021.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
from helpers.BaseRunner import BaseRunner
old=W/'runs/v1_controls_gpu0_20261021_100939'/label
done=json.loads((old/'COMPLETE.json').read_text());assert done['status']=='COMPLETE'
assert done['exit_code']==0 or label=='A' and done['adopted_existing_training']
assert done['best_epoch']==157 and done['end_epoch']==177
assert entry.sha(old/'best.pt')==done['checkpoint_sha256']
job=json.loads((old/'job_gpu1_unlimited.json').read_text());job['output_dir']=str(OUT)
job['config'].update(gpu='0',train=0,load=1,dev_only=0,test_epoch=-1,log_file=str(OUT/'eval.log'))
entry.save(OUT/'EVAL_CONFIG.json',job);entry.bind(job)
start=time.monotonic();model,corpus,args,cls=entry.build(job);model.load_model();model.eval()
# Recheck frozen feature identity after loading best-Dev weights.
with (entry.ASSET/'base_pca.pkl').open('rb') as f:expected=np.asarray(pickle.load(f),dtype=np.float32)
assert torch.equal(model.llm_table.detach().cpu(),torch.from_numpy(expected))
ds=cls.Dataset(model,corpus,'test');ds.prepare()
assert not model.test_all
slots=[len(ds[i]['item_id']) for i in range(len(ds))];assert set(slots)=={1001}
runner=BaseRunner(args);pred=runner.predict(ds);assert pred.shape==(len(ds),1001) and np.isfinite(pred).all()
metrics=runner.evaluate_method(pred,[5,10,20],['HR','NDCG'])
assert entry.sha(old/'best.pt')==done['checkpoint_sha256']
result={'passed':True,'label':label,'model':job['name'],'loss':job['loss'],'K':job['config']['K'],'seed':42,'phase':'Test','Test':metrics,
 'train_complete':True,'Dev_best_selected':True,'Test_accepted':True,'best_epoch':done['best_epoch'],'end_epoch':done['end_epoch'],
 'training_seconds':done['training_seconds'],'os_exit_code':done['exit_code'],'training_completion_evidence':'exit0; END/BestDev/checkpoint SHA and saved Dev',
 'checkpoint':str(old/'best.pt'),'checkpoint_sha256':done['checkpoint_sha256'],'embedding_sha256':entry.EXPECTED,
 'test_samples':len(ds),'candidate_slots':1001,'duplicates_preserved':True,'parameter_count':model.count_variables(),
 'test_csv_sha256':entry.sha(ROOT/'data/ml-1m/test.csv'),'seconds':time.monotonic()-start,
 'optimizer_updates':0,'original_entry_sha256':entry.sha(W/'ops/v1_controls_entry_20261021.py'),'runner_sha256':entry.sha(ROOT/'helpers/BaseRunner.py')}
entry.save(OUT/'TEST_COMPLETE.json',result);print(json.dumps(result))
