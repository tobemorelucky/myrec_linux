"""Single global-scale experiment; reuse original successful launcher/model/Runner."""
import pathlib,os,sys,json,argparse,importlib.util,subprocess,datetime,time,pickle,csv,traceback
ROOT=pathlib.Path('/home/yyx/hzgcode/pom2.0');W=ROOT/'research/ml1m_embedding_v2_20261020'
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
spec=importlib.util.spec_from_file_location('original_v2_launcher',W/'ops/run_v2_rich_aspcf_softmax_20261021.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def append(p,x):
 with p.open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n')
def measure(model,ids,out,stage,table=None):
 import torch,numpy as np
 cpu=torch.random.get_rng_state();gpu=torch.cuda.get_rng_state();enc=model.item_encoder;was=model.training
 old=enc.llm_table.detach().clone() if table is not None else None
 captured={}
 def hook(name):
  def take(module,inputs,value):captured[name]=value.detach()
  return take
 handles=[enc.complement_tail.register_forward_hook(hook('tail')),enc.gate.register_forward_hook(hook('gate_logits'))]
 try:
  with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.no_grad():
   model.eval()
   if table is not None:enc.llm_table.copy_(torch.from_numpy(table).to(enc.llm_table.device))
   d=enc(ids,return_components=True)
   def stat(x):
    x=x.double().flatten();assert torch.isfinite(x).all(),stage
    return dict(mean=float(x.mean()),std=float(x.std(unbiased=False)),P5=float(torch.quantile(x,.05)),P50=float(torch.quantile(x,.5)),P95=float(torch.quantile(x,.95)))
   result=dict(stage=stage,items=len(ids),semantic_norm=stat(d['semantic'].norm(dim=-1)),complement_norm=stat(d['complement'].norm(dim=-1)),tail_norm=stat(captured['tail'].norm(dim=-1)),gate_logit_norm=stat(captured['gate_logits'].norm(dim=-1)),embedding_norm=stat(d['emb'].norm(dim=-1)),alpha_sem=stat(d['alpha_sem']),alpha_comp=stat(d['alpha_comp']),gate_extreme_fraction=float(((d['alpha_sem']<.01)|(d['alpha_sem']>.99)).float().mean()),all_finite=True)
 finally:
  with torch.no_grad():
   if old is not None:enc.llm_table.copy_(old)
  for h in handles:h.remove()
  model.train(was)
 assert torch.equal(cpu,torch.random.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state()),'Probe changed RNG'
 result['training_rng_preserved']=True;append(out/'ACTIVATIONS.jsonl',result)

def install_train_probe(out,qa):
 import torch,numpy as np
 from helpers.BaseRunner import BaseRunner
 original=BaseRunner.fit
 def fit(self,dataset,epoch=-1):
  model=dataset.model
  if not hasattr(self,'_scale_probe_ids'):
   catalog=np.unique(np.asarray(dataset.data['item_id']).reshape(-1));catalog=catalog[catalog>0]
   chosen=np.random.RandomState(20261021).choice(catalog,min(512,len(catalog)),replace=False)
   self._scale_probe_ids=torch.tensor(chosen,device=model.device,dtype=torch.long)
   write(out/'TRAIN_PROBE_IDS.json',dict(item_ids=chosen.tolist(),source='fixed random subset of Train positive catalog; no Dev/Test labels'))
   for key in ['V1_Rebuilt','V2_Rich']:
    with open(qa['inputs'][key]['path'],'rb') as f:table=pickle.load(f)
    measure(model,self._scale_probe_ids,out,'initial_same_parameters_'+key,table)
   measure(model,self._scale_probe_ids,out,'initial_V2_Scaled')
   original_loss=model.loss
   def checked_loss(value):
    loss=original_loss(value)
    if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite original training loss')
    return loss
   model.loss=checked_loss
  value=original(self,dataset,epoch)
  if epoch in [1,10,50]:measure(model,self._scale_probe_ids,out,'epoch_'+str(epoch))
  return value
 BaseRunner.fit=fit

def final_test(c,out,qa):
 import numpy as np,torch
 from helpers.BaseRunner import BaseRunner
 train=json.loads((out/'TRAIN_COMPLETE.json').read_text());assert train['exit_code']==0
 assert base.sha(out/'best.pt')==train['checkpoint_sha256']
 model,corpus,args,Model=base.build(c);model.load_model();model.eval()
 ids=json.loads((out/'TRAIN_PROBE_IDS.json').read_text())['item_ids']
 measure(model,torch.tensor(ids,device=model.device),out,'best_V2_Scaled')
 ds=Model.Dataset(model,corpus,'test');ds.prepare();assert not model.test_all
 assert all(len(ds[i]['item_id'])==1001 for i in range(len(ds)))
 pred=BaseRunner(args).predict(ds);assert pred.shape==(len(ds),1001) and np.isfinite(pred).all()
 metrics=BaseRunner.evaluate_method(pred,[5,10,20],['HR','NDCG'])
 assert base.sha(out/'best.pt')==train['checkpoint_sha256'] and base.sha(c['llm_emb_path'])==qa['output_sha256']
 write(out/'TEST_COMPLETE.json',dict(Test=metrics,best_epoch=train['best_epoch'],checkpoint_sha256=train['checkpoint_sha256'],embedding_sha256=qa['output_sha256'],test_samples=len(ds),candidate_slots=1001,duplicates_preserved=True,optimizer_updates=0,exploratory=True,test_csv_sha256=base.sha(ROOT/'data/ml-1m/test.csv')))

def summaries(out,qa,result):
 oldtests=W/'runs/aspcf_v1rebuilt_v2rich_TEST_20261021_154130'
 original=W/'runs/p920_aspcf_ss_v2_rich_k4_seed42_20261021_145239'
 refs=json.loads((W/'runs/v1_controls_split_20261021_103559/gpu0/RESULTS.json').read_text())['results']
 ref=next(x for x in refs if x['label']=='ASPCF_reference')
 v2=json.loads((original/'COMPLETE.json').read_text())
 records=[]
 for name,dv,tr,t,scale,emb in [
  ('V1_Rebuilt',ref['Dev'],ref,json.loads((oldtests/'V1_Rebuilt_TEST.json').read_text())['Test'],1,qa['inputs']['V1_Rebuilt']['sha256']),
  ('V2_Rich',v2['Dev'],v2,json.loads((oldtests/'V2_Rich_TEST.json').read_text())['Test'],1,qa['inputs']['V2_Rich']['sha256']),
  ('V2_Scaled',result['Dev'],result,result['Test'],qa['scale'],qa['output_sha256'])]:
  records.append(dict(version=name,Dev=dv,Test=t,best_epoch=tr.get('best_epoch',47),end_epoch=tr.get('end_epoch',67),training_seconds=tr.get('training_seconds',tr.get('elapsed_seconds')),scale=scale,embedding_sha256=emb))
 write(out/'COMPARISON.json',records)
 fields=['version','phase','HR@5','NDCG@5','HR@10','NDCG@10','HR@20','NDCG@20','best_epoch','end_epoch','training_seconds','scale','embedding_sha256']
 with (out/'COMPARISON.csv').open('x') as f:
  cw=csv.DictWriter(f,fieldnames=fields);cw.writeheader()
  for r in records:
   for phase in ['Dev','Test']:cw.writerow({**{k:r[k] for k in ['version','best_epoch','end_epoch','training_seconds','scale','embedding_sha256']},'phase':phase,**r[phase]})
 lines=['# ML-1M V2全局尺度校准：探索性结果','',f"scale={qa['scale']:.17g}；仅改变输入Embedding，全程相同ASPCF Softmax配置。",'','|版本|阶段|HR@5|NDCG@5|HR@10|NDCG@10|HR@20|NDCG@20|best/end|训练秒|','|---|---|---:|---:|---:|---:|---:|---:|---|---:|']
 for r in records:
  for phase in ['Dev','Test']:
   values='|'.join(format(r[phase][m],'.9f') for m in fields[2:8]);lines.append(f"|{r['version']}|{phase}|{values}|{r['best_epoch']}/{r['end_epoch']}|{r['training_seconds']:.3f}|")
 lines+=['','## 与原V2及V1-Rebuilt的差值']
 for r in records[:2]:
  lines.append(f"- Scaled−{r['version']}：Dev N5 {result['Dev']['NDCG@5']-r['Dev']['NDCG@5']:+.9f}，Test N5 {result['Test']['NDCG@5']-r['Test']['NDCG@5']:+.9f}。")
 lines+=['','初始同参数三表及训练epoch1/10/50、best的激活/门控见ACTIVATIONS.jsonl；均固定Train电影，无额外目标或优化步骤。','Test已被历史观察，本结果仅探索性。全局缩放不改变语义方向；若未改善，应降低单一范数放大解释的优先级，但不能排除分支尺度比例、谱与语义关系变化。','完整配置config.json、输入SCALE_QA.json、provenance.json、checkpoint SHA及终态COMPLETE.json均保留。']
 (out/'RESULTS.md').write_text('\n'.join(lines))

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--mode',required=True,choices=['supervise','smoke','train','dev','test']);a=p.parse_args()
 out=pathlib.Path(a.run_dir).resolve();assert out.is_relative_to(W) and out.is_dir()
 c=json.loads((out/'config.json').read_text());qa=json.loads((out/'SCALE_QA.json').read_text());base.EXPECTED=qa['output_sha256'];base.__file__=__file__
 os.environ['CUDA_VISIBLE_DEVICES']=str(c['gpu'])
 if a.mode=='supervise':
  original_save=base.save
  def staged(p,x):original_save(p.with_name({'COMPLETE.json':'TRAIN_COMPLETE.json','exit.json':'TRAIN_EXIT.json'}.get(p.name,p.name)),x)
  base.save=staged
  try:
   base.supervise(c,out)
   if not (out/'TRAIN_COMPLETE.json').exists():raise RuntimeError('Training not complete; no Test')
   result=json.loads((out/'TRAIN_COMPLETE.json').read_text());result.update(train_complete=True,Dev_best_selected=True,Test_accepted=False,scale=qa['scale'],exploratory=True)
   env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
   with (out/'test.out').open('xb') as f:rc=subprocess.run([base.PY,'-u','-B',__file__,'--run-dir',str(out),'--mode','test'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300).returncode
   if rc:raise RuntimeError('Train/Dev completed; Test acceptance failed; no retry')
   result.update(Test=json.loads((out/'TEST_COMPLETE.json').read_text())['Test'],Test_accepted=True)
   summaries(out,qa,result);write(out/'COMPLETE.json',result);write(out/'exit.json',dict(exit_code=0,train_complete=True,Test_accepted=True))
  except BaseException as e:
   if not (out/'FAILED.json').exists():write(out/'FAILED.json',dict(error=repr(e),train_complete=(out/'TRAIN_COMPLETE.json').exists(),Test_accepted=(out/'TEST_COMPLETE.json').exists(),time=datetime.datetime.now().astimezone().isoformat()))
   if not (out/'exit.json').exists():write(out/'exit.json',dict(exit_code=1,error=repr(e)))
   raise
 elif a.mode=='test':base.bind(c,out,'test');final_test(c,out,qa)
 else:
  if a.mode=='train':install_train_probe(out,qa)
  base.main()
if __name__=='__main__':main()
