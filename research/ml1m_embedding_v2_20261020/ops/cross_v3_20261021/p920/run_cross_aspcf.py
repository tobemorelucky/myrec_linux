"""Crossed assets: reuse original ASPCF SS launcher/model/protocol; no original file writes."""
import os,sys,pathlib,json,argparse,importlib.util,inspect,hashlib,pickle,subprocess,datetime,re,time,traceback
ROOT=pathlib.Path('/home/yyx/hzgcode/pom2.0');W=ROOT/'research/ml1m_embedding_v2_20261020'
OPS=W/'ops/cross_v3_20261021/p920';RUNS=W/'runs/cross_v3_20261021/p920'
SELF=OPS/'run_cross_aspcf.py';PY='/home/yyx/anaconda3/envs/hzg_py10/bin/python'
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
spec=importlib.util.spec_from_file_location('original_scaled',W/'ops/run_v2_scaled_aspcf_20261021.py')
scaled=importlib.util.module_from_spec(spec);spec.loader.exec_module(scaled);base=scaled.base
ORIGINAL_GENERIC=pathlib.Path(base.__file__);ORIGINAL_SCALED=pathlib.Path(scaled.__file__)
# Exact, auditable one-line generalization: GPU may equal reference GPU; scientific input must differ.
original=inspect.getsource(base.supervise)
needle="assert set(changed)=={'gpu','llm_emb_path','log_file','model_path'},changed"
replacement="assert set(changed)<={'gpu','llm_emb_path','log_file','model_path'} and 'llm_emb_path' in changed,changed"
assert original.count(needle)==1
exec(compile(original.replace(needle,replacement),str(SELF),'exec'),base.__dict__)
base.__file__=str(SELF)
VARIANTS={'A1':'A1_oldlora_rich512/rich_pca.pkl','B0':'B0_v2lora_base/base_pca.pkl','V3':'V3/rich_pca.pkl'}
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(p,x):
 with pathlib.Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def inside(p,parent):
 p=pathlib.Path(p);assert p.is_absolute() and p.resolve()==p and p.is_relative_to(parent),str(p)
 return p
def now():return datetime.datetime.now().astimezone().isoformat()
def invariant_files():
 prov=json.loads((base.REF/'provenance.json').read_text())
 sources={p:sha(ROOT/p) for p in prov['sources']}
 assert sources==prov['sources'],'Baseline sources changed'
 data={p:sha(ROOT/'data/ml-1m'/p) for p in prov['data_sha256']}
 assert data==prov['data_sha256'],'Baseline Train/Dev/corpus changed'
 data['test.csv']=sha(ROOT/'data/ml-1m/test.csv')
 return dict(sources=sources,data_sha256=data,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),original_generic_sha256=sha(ORIGINAL_GENERIC),original_scaled_sha256=sha(ORIGINAL_SCALED),wrapper_sha256=sha(SELF))
def prepare(a):
 import numpy as np
 assert a.variant in VARIANTS and a.gpu in ['0','1']
 p=inside(RUNS/'incoming'/VARIANTS[a.variant],RUNS);assert p.is_file(),'Expected crossed embedding has not arrived'
 assert re.fullmatch('[0-9a-f]{64}',a.embedding_sha256 or '')
 assert sha(p)==a.embedding_sha256,'Delivered matrix SHA mismatch'
 assert re.fullmatch('[A-Za-z0-9_-]+',a.run_name or '')
 out=inside(RUNS/a.run_name,RUNS);assert not out.exists(),'Refuse existing run directory'
 with p.open('rb') as f:table=pickle.load(f)
 assert isinstance(table,np.ndarray) and table.shape==(3707,1536) and table.dtype==np.float32
 assert np.isfinite(table).all() and not table[0].any() and (np.linalg.norm(table[1:],axis=1)>0).all()
 assert len(np.unique(table[1:],axis=0))==3706,'Duplicate nonpadding embeddings'
 prov=invariant_files()
 delivery=json.loads(pathlib.Path(a.delivery_manifest).read_text()) if a.delivery_manifest else {}
 assert delivery.get('variant')==a.variant,'Delivery manifest must declare crossed identity'
 assert delivery.get('embedding_sha256')==a.embedding_sha256,'Delivery manifest SHA mismatch'
 assert delivery.get('canonical_item_ids')=='1..3706','Delivery must attest canonical item alignment'
 expected={'A1':('old','rich512'),'B0':('v2','base_original')}
 if a.variant in expected:
  assert (delivery.get('lora_family'),delivery.get('text_recipe'))==expected[a.variant],'Wrong LoRA/text crossed identity'
 else:assert delivery.get('text_recipe') and delivery.get('lora_family')=='v3','Missing V3 identity'
 c=json.loads((base.REF/'config.json').read_text())
 c.update(gpu=a.gpu,llm_emb_path=str(p),log_file=str(out/'train.log'),model_path=str(out/'best.pt'))
 assert c['K']==4 and c['random_seed']==42 and c['history_max']==20 and c['lr']==.001 and c['batch_size']==1024 and c['lambda_relation']==.01
 out.mkdir()
 save(out/'config.json',c)
 save(out/'ASSET_QA.json',dict(variant=a.variant,path=str(p),sha256=a.embedding_sha256,shape=list(table.shape),dtype=str(table.dtype),finite=True,padding_zero=True,nonzero_rows=3706,no_duplicate_rows=True,delivery_manifest=delivery,identity_note='LoRA/text/canonical alignment attested by producer manifest; matrix SHA and shape independently verified',global_scaling=1.0))
 save(out/'PREPARED.json',dict(time=now(),variant=a.variant,train_complete=False,Dev_best_selected=False,Test_accepted=False,from_scratch=True,reference=str(base.REF),provenance=prov,in_memory_adaptation=dict(before=needle,after=replacement),launcher_reuse='Original scaled launcher imports original generic V2 base; wrapper reuses base.build/bind/smoke/dev/supervise, and independent fixed-best Test. No scaling/probes/model copies.'))
 print(json.dumps(dict(prepared=str(out),command=[PY,'-u','-B',str(SELF),'--run-dir',str(out),'--mode','supervise'])))
 return out
def verify(out):
 out=inside(out,RUNS);c=json.loads((out/'config.json').read_text());qa=json.loads((out/'ASSET_QA.json').read_text());prepared=json.loads((out/'PREPARED.json').read_text())
 assert sha(c['llm_emb_path'])==qa['sha256']
 current=invariant_files();assert current==prepared['provenance'],'Sources/data/launchers changed since preparation'
 base.EXPECTED=qa['sha256'];os.environ['CUDA_VISIBLE_DEVICES']=str(c['gpu'])
 return c,qa,prepared
def final_test(c,out,qa,prepared):
 import numpy as np,torch
 from helpers.BaseRunner import BaseRunner
 train=json.loads((out/'TRAIN_COMPLETE.json').read_text())
 assert train['exit_code']==0 and train['checkpoint_sha256']==sha(out/'best.pt')
 save(out/'TEST_STARTED.json',dict(time=now(),checkpoint_sha256=train['checkpoint_sha256'],single_attempt=True))
 base.bind(c,out,'test');model,corpus,args,Model=base.build(c);model.load_model();model.eval()
 assert model.optimizer is None
 assert all(torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point())
 ds=Model.Dataset(model,corpus,'test');ds.prepare();assert not model.test_all
 assert all(len(ds[i]['item_id'])==1001 for i in range(len(ds)))
 pred=BaseRunner(args).predict(ds)
 assert pred.shape==(len(ds),1001) and np.isfinite(pred).all()
 metrics=BaseRunner.evaluate_method(pred,[5,10,20],['HR','NDCG'])
 assert sha(out/'best.pt')==train['checkpoint_sha256'] and sha(c['llm_emb_path'])==qa['sha256']
 assert invariant_files()==prepared['provenance']
 save(out/'TEST_COMPLETE.json',dict(Test=metrics,time=now(),variant=qa['variant'],checkpoint_sha256=train['checkpoint_sha256'],embedding_sha256=qa['sha256'],best_epoch=train['best_epoch'],test_samples=len(ds),candidate_slots=1001,duplicates_preserved=True,optimizer_updates=0,exploratory=True,provenance=prepared['provenance']))
def supervise(c,out,qa,prepared):
 oldsave=base.save
 def staged(p,x):oldsave(p.with_name({'COMPLETE.json':'TRAIN_COMPLETE.json','exit.json':'TRAIN_EXIT.json'}.get(p.name,p.name)),x)
 base.save=staged
 try:
  base.supervise(c,out)
  assert (out/'TRAIN_COMPLETE.json').is_file(),'No reliable successful best-Dev training; no Test'
  result=json.loads((out/'TRAIN_COMPLETE.json').read_text());assert result['exit_code']==0
  losses=re.findall(r'Epoch\s+\d+\s+loss=\s*(\S+)',(out/'train.log').read_text())
  assert losses and all(__import__('math').isfinite(float(x)) for x in losses),'Nonfinite training loss; no Test'
  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
  with (out/'test.out').open('xb') as f:
   rc=subprocess.run([PY,'-u','-B',str(SELF),'--run-dir',str(out),'--mode','test'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300).returncode
  assert rc==0,'Train/Dev complete; Test failed; no automatic retry'
  test=json.loads((out/'TEST_COMPLETE.json').read_text())
  result.update(variant=qa['variant'],Test=test['Test'],train_complete=True,Dev_best_selected=True,Test_accepted=True,exploratory=True,all_epoch_loss_finite=True,complete_epochs=len(losses))
  save(out/'COMPLETE.json',result);save(out/'exit.json',dict(exit_code=0,time=now(),train_complete=True,Dev_best_selected=True,Test_accepted=True))
  lines=['# Crossed-asset ASPCF SS exploratory result','',f"Variant {qa['variant']}; seed42, K4, H20, lr .001, batch1024, relation .01. Original V1 configuration except embedding/output/GPU.",'',f"Best/end epochs {result['best_epoch']}/{result['end_epoch']}; from scratch; Dev chooses best, then one fixed-checkpoint Test.",'', '|phase|HR@5|NDCG@5|HR@10|NDCG@10|HR@20|NDCG@20|','|---|---:|---:|---:|---:|---:|---:|']
  for phase in ['Dev','Test']:lines.append('|'+phase+'|'+'|'.join(f'{result[phase][m]:.12g}' for m in ['HR@5','NDCG@5','HR@10','NDCG@10','HR@20','NDCG@20'])+'|')
  lines+=['',f"Checkpoint SHA256: {result['checkpoint_sha256']}",f"Embedding SHA256: {qa['sha256']}",'','Historical Test observations informed this research; this is exploratory, not an independent confirmatory test.']
  with (out/'RESULTS.md').open('x') as f:f.write('\n'.join(lines))
 except BaseException as e:
  if not (out/'FAILED.json').exists():save(out/'FAILED.json',dict(error=repr(e),traceback=traceback.format_exc(),time=now(),train_complete=(out/'TRAIN_COMPLETE.json').exists(),Test_accepted=(out/'TEST_COMPLETE.json').exists()))
  if not (out/'exit.json').exists():save(out/'exit.json',dict(exit_code=1,error=repr(e),time=now()))
  raise
def main():
 p=argparse.ArgumentParser()
 p.add_argument('--mode',required=True,choices=['prepare','validate','supervise','smoke','train','dev','test'])
 p.add_argument('--variant',choices=list(VARIANTS));p.add_argument('--gpu',choices=['0','1'])
 p.add_argument('--run-name');p.add_argument('--run-dir');p.add_argument('--embedding-sha256');p.add_argument('--delivery-manifest')
 a=p.parse_args()
 if a.mode=='prepare':prepare(a);return
 c,qa,prepared=verify(pathlib.Path(a.run_dir));out=pathlib.Path(a.run_dir)
 if a.mode=='validate':print(json.dumps(dict(valid=True,variant=qa['variant'],gpu=c['gpu'],config=c,qa=qa,prepared=prepared)));return
 if a.mode=='supervise':supervise(c,out,qa,prepared)
 elif a.mode=='test':final_test(c,out,qa,prepared)
 else:base.main()
if __name__=='__main__':main()
