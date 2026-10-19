"""Four authorized joint P0/P1 BPR runs. Scheduling only; reuse Probe unchanged."""
import os,sys,json,time,pickle,hashlib,argparse,logging,traceback,subprocess,signal,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from tools.run_refine_bpr_round1 import sha,bh,write_new,arguments,rng,restore,rh,fixture
SOURCES=['tools/run_p0p1_joint_bpr.py','tools/run_frozen_interest_readout.py',
 'models/sequential/LLMMIRecASPCFReadoutBPR.py','tools/run_refine_bpr_round1.py',
 'models/sequential/LLMMIRecASPCF.py','models/sequential/llmmi_components.py',
 'models/sequential/llmmi_utils.py','models/BaseModel.py','helpers/BaseRunner.py',
 'helpers/SeqReader.py','helpers/BaseReader.py','main.py']
FAMILY='p0p1_joint_bpr'
def location(rid):
 import re
 assert re.fullmatch('[A-Za-z0-9_.-]+',rid)
 return ROOT/'new_log'/FAMILY/rid

def prepare(rid):
 root=location(rid);assert not root.exists()
 source={f:sha(ROOT/f) for f in SOURCES};jobs=[]
 for ds in ['beauty','ml-1m']:
  assets={str((ROOT/'data'/ds/f).relative_to(ROOT)):sha(ROOT/'data'/ds/f)
          for f in ['train.csv','dev.csv','SeqReader.pkl','item2id.json','handled/llm_table_pca1536.pkl']}
  for mode in ['P0','P1']:
   log=root/ds/mode;weight=ROOT/'new_model'/FAMILY/rid/ds/mode/'best.pt'
   assert not log.exists() and not weight.parent.exists()
   a,hist=arguments(ds,'user',log/'train.log',weight)
   delattr(a,'refine_mode');a.readout_mode=mode;a.backbone_mode='joint'
   assert (a.epoch,a.early_stop,a.dropout,a.K,a.history_max)==(200,10,.1,4,20)
   jobs.append(dict(dataset=ds,arm=mode,run_id=rid,effective_args=vars(a),historical_config=hist,
    source_sha256=source,asset_sha256=assets,test_allowed=False,initialization='from_scratch',
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'run',str(log/'config.json')]))
 for job in jobs:
  p=Path(job['effective_args']['log_file']).parent;p.mkdir(parents=True,exist_ok=False)
  Path(job['effective_args']['model_path']).parent.mkdir(parents=True,exist_ok=False)
  write_new(p/'config.json',job)
 # Preserve exact source versions alongside configs; never overwrite old sources.
 for f in SOURCES:
  target=root/'source_archive'/f;target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as out:out.write((ROOT/f).read_bytes())
 write_new(root/'manifest.json',dict(run_id=rid,jobs=jobs,test_allowed=False))
 print(json.dumps(dict(root=str(root),jobs=[j['effective_args']['log_file'] for j in jobs])))

def mechanism(model,batch,label):
 import torch
 from tools.run_frozen_interest_readout import stats
 saved=rng();was=model.training;checked=model.backbone._first_batch_checked
 try:
  model.eval()
  with torch.no_grad():
   out=model.backbone(batch,True);new=model.from_output(out)
   v,a,w,u=out['interest_vectors'],out['attention_maps'],out['interest_weights'],out['user_vector']
   # All unordered pairs, not a selected favorable pair.
   pairs=torch.triu_indices(model.K,model.K,offset=1,device=v.device)
   rel=(v[:,pairs[0]]-v[:,pairs[1]]).norm(dim=-1)/v.norm(dim=-1).mean(-1,keepdim=True).clamp_min(1e-12)
   al1=(a[:,pairs[0]]-a[:,pairs[1]]).abs().sum(-1)
   result=dict(label=label,rows=len(v),head_relative_difference=stats(rel),attention_head_L1=stats(al1),
    residual_to_user_norm=stats(new['delta'].norm(dim=-1)/u.norm(dim=-1).clamp_min(1e-12)),
    margin=stats(new['prediction'][:,0]-new['prediction'][:,1]))
   alt=dict(out);alt['interest_vectors']=u[:,None,:].expand_as(v)
   changed=model.from_output(alt)
   result['replace_independent_V_by_u_score_MAE']=float((changed['prediction']-new['prediction']).abs().mean())
   result['replacement_note']='Frozen Train probe; off-distribution sensitivity, not causal or retrained attribution.'
   return result
 finally:
  model.train(was);model.backbone._first_batch_checked=checked;restore(saved)

def run(path):
 import numpy as np,torch
 import helpers.BaseRunner as rm
 from helpers.BaseRunner import BaseRunner
 from models.BaseModel import GeneralModel
 from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF
 from models.sequential.LLMMIRecASPCFReadoutBPR import LLMMIRecASPCFReadoutBPR as Model
 from tools.run_frozen_interest_readout import Probe,state_hash
 from utils import utils
 conf=json.loads(Path(path).read_text());a=argparse.Namespace(**conf['effective_args']);log=Path(a.log_file).parent
 assert not Path(a.log_file).exists() and not Path(a.model_path).exists()
 assert all(sha(ROOT/f)==h for f,h in conf['source_sha256'].items())
 assert (a.K,a.num_neg,a.epoch,a.test_epoch,a.dev_only,a.load,a.backbone_mode)==(4,1,200,-1,1,0,'joint')
 assert Model.forward is Probe.forward and Model.input is Probe.input and Model.from_output is Probe.from_output
 logging.basicConfig(filename=a.log_file,level=logging.INFO,format='%(asctime)s %(message)s')
 logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
 start=time.monotonic();state=dict(success=False,test_executed=False,epochs=[],probes=[])
 original_loader=rm.DataLoader;handle=None
 def timeout(*_):raise TimeoutError('Four-hour per-run wall budget exceeded; configuration unchanged')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(4*3600)
 try:
  a.device=torch.device('cuda:0');utils.init_seed(42)
  with (ROOT/'data'/a.dataset/'SeqReader.pkl').open('rb') as f:corpus=pickle.load(f)
  corpus.data_df={p:corpus.data_df[p] for p in ['train','dev']}
  model=Model(a,corpus).to(a.device)
  assert all(p.requires_grad for p in model.parameters()) and model.backbone_mode=='joint'
  assert sum(p.numel() for p in model.readout.parameters())==27808
  state.update(parameters=model.count_variables(),added_parameters=27808,
   initial_shared_sha256=state_hash(model.backbone.state_dict()),initial_readout_sha256=state_hash(model.readout.state_dict()),
   initial_state_sha256=state_hash(model.state_dict()),initial_rng=rh())
  saved=rng();utils.init_seed(42);base=LLMMIRecASPCF(a,corpus).to(a.device)
  assert state_hash(base.state_dict())==state['initial_shared_sha256'];del base;restore(saved)
  assert rh()==state['initial_rng'];state['base_initialization_verified']=True
  data={p:Model.Dataset(model,corpus,p) for p in ['train','dev']}
  for d in data.values():d.prepare()
  state.update(train_rows=len(data['train']),dev_rows=len(data['dev']))
  # One epoch0 Dev pass, identical existing backbone/Probe math and original evaluator.
  saved=rng();checked=model.backbone._first_batch_checked;preds=[];inputs=hashlib.sha256()
  model.eval()
  with torch.inference_mode():
   for batch in original_loader(data['dev'],batch_size=a.eval_batch_size,shuffle=False,num_workers=a.num_workers,
                               collate_fn=data['dev'].collate_batch,pin_memory=a.pin_memory):
    for key in ['user_id','item_id','history_items','lengths']:inputs.update(batch[key].numpy().tobytes())
    assert batch['item_id'].shape[1]==1001
    b=utils.batch_to_gpu(batch,a.device);out=model.backbone(b,True);new=model.from_output(out)
    assert torch.count_nonzero(new['delta'])==0 and torch.equal(new['prediction'],out['prediction'])
    preds.append(new['prediction'].cpu().numpy())
  initial_pred=np.concatenate(preds);del preds
  state['epoch0']=dict(dev={k:float(v) for k,v in BaseRunner.evaluate_method(initial_pred,[5,10,20],['NDCG','HR']).items()},
                       predictions_sha256=bh(initial_pred.tobytes()),inputs_sha256=inputs.hexdigest(),zero_residual_identity=True)
  del initial_pred;model.train();model.backbone._first_batch_checked=checked;restore(saved)
  batch=fixture(corpus,a.device);state['probe_fixture_sha256']=bh(b''.join(t.cpu().numpy().tobytes() for t in batch.values()))
  state['probes'].append(mechanism(model,batch,'initial'))
  print('INITIALIZATION '+json.dumps({k:v for k,v in state.items() if k not in ['epochs','probes']}),flush=True)
  write_new(log/'initialization.json',{k:v for k,v in state.items() if k!='epochs'})
  active=False;relations=[]
  class RecordedLoader(original_loader):
   def __iter__(self):
    if getattr(self.dataset,'phase',None)!='train':yield from super().__iter__();return
    h=hashlib.sha256();rows=steps=0
    for b in super().__iter__():
     h.update(b['user_id'].numpy().tobytes());h.update(b['item_id'].numpy().tobytes())
     rows+=len(b['user_id']);steps+=1;yield b
    state['epochs'][-1].update(train_order_sha256=h.hexdigest(),rows=rows,steps=steps)
  rm.DataLoader=RecordedLoader
  def hook(m,inputs,out):
   if active:
    assert m.training and m.backbone.training and m.backbone.dropout.training
    assert inputs[0]['item_id'].shape[1]==2 and '_relation_ids' in out
    relations.append(out['_relation_ids'].detach())
  handle=model.register_forward_hook(hook)
  class DevOnlyRunner(BaseRunner):
   def _build_optimizer(self,m):
    opt=super()._build_optimizer(m);items=[p for g in opt.param_groups for p in g['params']]
    assert len(items)==len({id(p) for p in items})
    assert {id(p) for p in items}=={id(p) for p in m.parameters() if p.requires_grad}
    state['optimizer']=[{k:v for k,v in g.items() if k!='params'}|{'numel':sum(p.numel() for p in g['params'])} for g in opt.param_groups]
    return opt
   def fit(self,dataset,epoch=-1):
    nonlocal active,relations
    assert dataset.phase=='train';rec=dict(epoch=epoch,rng_start=rh());state['epochs'].append(rec)
    active=True;relations=[];t=time.monotonic()
    try:loss=super().fit(dataset,epoch)
    finally:active=False
    rec.update(loss=float(loss),fit_seconds=time.monotonic()-t,rng_after_fit=rh(),
     negatives_sha256=bh(np.ascontiguousarray(dataset.data['neg_items']).tobytes()),
     relation_sha256=bh(torch.cat(relations).cpu().numpy().tobytes()))
    assert np.isfinite(loss)
    if epoch==1:
     state['first_epoch_gradient_norms']={name:float(torch.sqrt(sum((p.grad.detach()**2).sum() for p in module.parameters())))
      for name,module in [('ItemEncoder',model.backbone.item_encoder),('Query',model.backbone.extractor),
       ('Aggregator',model.backbone.aggregator),('Position',model.backbone.position_emb),('readout',model.readout)]}
     assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    return loss
   def predict(self,dataset):
    assert dataset.phase=='dev','Test is prohibited'
    result=super().predict(dataset);assert result.shape==(len(dataset),1001)
    return result
   def evaluate(self,dataset,topks,metrics):
    assert dataset.phase=='dev';t=time.monotonic()
    result=super().evaluate(dataset,[5,10,20],metrics)
    assert all(np.isfinite(v) for v in result.values())
    if not getattr(self,'final_validation',False):
     rec=state['epochs'][-1];rec.update(dev={k:float(v) for k,v in result.items()},dev_seconds=time.monotonic()-t,rng_after_dev=rh())
     write_new(log/('epoch_%03d.json'%rec['epoch']),rec)
     print('DEV_FULL '+json.dumps(dict(epoch=rec['epoch'],**rec['dev'])),flush=True)
    return result
  runner=DevOnlyRunner(a);assert runner.main_metric=='NDCG@5'
  torch.cuda.reset_peak_memory_stats();t=time.monotonic();runner.train(data)
  state['training_dev_seconds']=time.monotonic()-t
  vals=[e['dev']['NDCG@5'] for e in state['epochs']];last=max(i+1 for i,v in enumerate(vals) if v==max(vals))
  state.update(best_reported_epoch=vals.index(max(vals))+1,checkpoint_saved_epoch=last,end_epoch=len(vals),checkpoint_sha256=sha(a.model_path))
  loaded=torch.load(a.model_path,map_location=a.device,weights_only=True)
  assert all(torch.equal(v,loaded[k]) for k,v in model.state_dict().items());del loaded
  runner.final_validation=True;state['dev']={k:float(v) for k,v in runner.evaluate(data['dev'],[5,10,20],runner.metrics).items()}
  assert state['dev']==state['epochs'][last-1]['dev'];state['best_checkpoint_loaded_verified']=True
  state['probes'].append(mechanism(model,batch,'best_'+str(last)))
  assert sha(a.model_path)==state['checkpoint_sha256'];state['success']=True
 except BaseException as e:
  state.update(error=repr(e),traceback=traceback.format_exc());raise
 finally:
  signal.alarm(0);rm.DataLoader=original_loader
  if handle is not None:handle.remove()
  state.update(total_seconds=time.monotonic()-start,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
   peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,source_unchanged={f:sha(ROOT/f)==h for f,h in conf['source_sha256'].items()})
  write_new(log/'summary.json',state)

def orchestrate(rid):
 root=location(rid);manifest=json.loads((root/'manifest.json').read_text());results=[]
 for job in manifest['jobs']:
  while True:
   gpu=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
   apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
   if gpu[0].strip() not in apps and float(gpu[1])>=7000:break
   print('GPU1 busy; safely waiting without touching other processes',flush=True);time.sleep(60)
  log=Path(job['effective_args']['log_file']).parent
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
  with (log/'train.out').open('xb') as output:
   child=subprocess.Popen(job['command'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT)
   write_new(log/'process.json',dict(pid=child.pid,started_unix=time.time(),gpu=1));code=child.wait()
  rec=dict(dataset=job['dataset'],arm=job['arm'],exit_code=code,summary=str(log/'summary.json'))
  results.append(rec);write_new(root/(job['dataset']+'_'+job['arm']+'_finished.json'),rec)
  if code:write_new(root/'blocked.json',dict(results=results));raise RuntimeError(rec)
  s=json.loads((log/'summary.json').read_text());assert s['success'] and all(s['source_unchanged'].values())
  if job['arm']=='P1':
   ds=job['dataset'];x,y=[json.loads((root/ds/arm/'summary.json').read_text()) for arm in ['P0','P1']]
   fields=['initial_shared_sha256','initial_readout_sha256','initial_state_sha256','initial_rng','probe_fixture_sha256','epoch0','optimizer']
   checks={k:x[k]==y[k] for k in fields}
   for f in ['rng_start','rng_after_fit','rng_after_dev','negatives_sha256','train_order_sha256','relation_sha256','steps','rows']:
    checks[f]=all(e[f]==v[f] for e,v in zip(x['epochs'],y['epochs']))
   write_new(root/(ds+'_fairness.json'),dict(checks=checks,common_epochs=min(x['end_epoch'],y['end_epoch'])))
   assert all(checks.values()),'Pairing failed: do not start another dataset'
  print('RUN_COMPLETE '+json.dumps({**rec,'dev':s['dev']}),flush=True)
 summaries=[dict(dataset=j['dataset'],arm=j['arm'],**json.loads(Path(r['summary']).read_text())) for j,r in zip(manifest['jobs'],results)]
 write_new(root/'results.json',dict(runs=summaries,test_executed=False))
 with (root/'dev_curves.csv').open('x',newline='') as f:
  writer=csv.writer(f);writer.writerow(['dataset','arm','epoch','loss','HR@5','NDCG@5','HR@10','NDCG@10','HR@20','NDCG@20','fit_seconds','dev_seconds'])
  for s in summaries:
   for e in s['epochs']:writer.writerow([s['dataset'],s['arm'],e['epoch'],e['loss']]+[e['dev'][k] for k in ['HR@5','NDCG@5','HR@10','NDCG@10','HR@20','NDCG@20']]+[e['fit_seconds'],e['dev_seconds']])
 write_new(root/'complete.json',dict(status='complete',results=results,test_executed=False))
 print('ALL_FOUR_COMPLETE '+str(root/'results.json'),flush=True)
if __name__=='__main__':
 action=sys.argv[1]
 if action=='prepare':prepare(sys.argv[2])
 elif action=='run':run(sys.argv[2])
 elif action=='orchestrate':orchestrate(sys.argv[2])
 else:raise ValueError(action)
