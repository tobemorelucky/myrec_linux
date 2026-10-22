"""A1 historical BPR runs: original models, main, Reader and Runner; isolated paths."""
import os,sys,json,pickle,pathlib,argparse,importlib,hashlib,runpy,datetime,time,subprocess,signal,re,traceback,math
R=pathlib.Path('/home/yyx/hzgcode/pom2.0');W=R/'research/ml1m_embedding_v2_20261020'
O=W/'ops/a1_history_bpr_20261021_night';U=W/'runs/a1_history_bpr_20261021_night'
PY='/home/yyx/anaconda3/envs/hzg_py10/bin/python'
A=W/'runs/cross_v3_20261021/p920/incoming/A1_oldlora_rich512/rich_pca.pkl'
EXPECTED='03cf9649844da2c172487a0f552a49a4d331d8262135e3b70e48d6e772ad7905'
def now():return datetime.datetime.now().astimezone().isoformat()
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=pathlib.Path(p);assert p.resolve()==p and p.is_relative_to(W)
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def flags(c):return [v for k,x in c.items() for v in ['--'+k,str(x)]]
def bind(c,out,mode):
 assert pathlib.Path(c['llm_emb_path'])==A and sha(A)==EXPECTED
 sys.path.insert(0,str(R));os.chdir(R)
 m=importlib.import_module('models.sequential.'+c['model_name']);old=m.load_llm_table
 def checked(path,expected_rows):
  import torch,numpy as np
  assert pathlib.Path(path).resolve()==A and sha(path)==EXPECTED
  v=old(path,expected_rows)
  with A.open('rb') as f:a=pickle.load(f)
  assert a.shape==(3707,1536) and a.dtype==np.float32 and np.isfinite(a).all() and not a[0].any()
  assert torch.equal(v.cpu(),torch.from_numpy(a))
  save(out/('EMBEDDING_'+mode+'.json'),dict(path=str(A),sha256=EXPECTED,model_buffer_verified=True))
  return v
 m.load_llm_table=checked
def build(c):
 import torch,main
 from helpers.SeqReader import SeqReader
 from helpers.BaseRunner import BaseRunner
 from utils import utils
 M=getattr(importlib.import_module('models.sequential.'+c['model_name']),c['model_name'])
 p=main.parse_global_args(argparse.ArgumentParser());p=SeqReader.parse_data_args(p);p=BaseRunner.parse_runner_args(p);p=M.parse_model_args(p)
 a,extra=p.parse_known_args(flags(c));assert extra in ([],['--model_name',c['model_name']]),extra
 a.device=torch.device('cuda');utils.init_seed(a.random_seed)
 with (R/'data/ml-1m/SeqReader.pkl').open('rb') as f:corpus=pickle.load(f)
 model=M(a,corpus).to(a.device)
 return model,corpus,a,M
def smoke(c,out,job):
 import numpy as np,torch
 from utils import utils
 model,corpus,a,M=build(c);model.train();ds=M.Dataset(model,corpus,'train')
 ix=np.random.RandomState(42).choice(len(ds),a.batch_size,replace=False)
 ds.data={k:np.asarray(v)[ix] for k,v in ds.data.items()};ds.actions_before_epoch()
 feed=utils.batch_to_gpu(ds.collate_batch([ds[i] for i in range(len(ix))]),model.device)
 torch.cuda.reset_peak_memory_stats();state=hashlib.sha256()
 for n,p in model.named_parameters():state.update(n.encode());state.update(p.detach().cpu().numpy().tobytes())
 if job.get('teacher'):
  assert sha(job['teacher'])==job['teacher_sha256']
  with open(job['teacher'],'rb') as f:t=pickle.load(f)
  if hasattr(model,'t_semantic_assign'):torch.testing.assert_close(model.t_semantic_assign.cpu(),torch.from_numpy(t['soft_assignments']),rtol=0,atol=0)
  if hasattr(model,'t_fine_assign'):torch.testing.assert_close(model.t_fine_assign.cpu(),torch.from_numpy(t['fine_assignments']),rtol=0,atol=0)
 z=model(feed);loss=model.loss(z);assert torch.isfinite(loss);loss.backward()
 grads={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
 assert grads and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 original_loss=type(model).loss
 save(out/'SMOKE_COMPLETE.json',dict(passed=True,loss=float(loss.detach()),original_loss_function=original_loss.__module__+'.'+original_loss.__qualname__,forward_keys=list(z),query_grad_norm=grads.get('extractor.query'),parameter_count=model.count_variables(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),initial_parameter_sha256=state.hexdigest(),optimizer_steps=0,Train_only=True,embedding_sha256=EXPECTED,teacher_sha256=job.get('teacher_sha256')))
def evaluate(c,out,phase,job):
 import numpy as np,torch
 from helpers.BaseRunner import BaseRunner
 model,corpus,a,M=build(c);model.load_model();model.eval()
 assert all(torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point())
 ds=M.Dataset(model,corpus,phase);ds.prepare();assert not model.test_all
 assert all(len(ds[i]['item_id'])==1001 for i in range(len(ds)))
 pred=BaseRunner(a).predict(ds);assert pred.shape==(len(ds),1001) and np.isfinite(pred).all()
 metrics=BaseRunner.evaluate_method(pred,[5,10,20],['HR','NDCG'])
 save(out/(phase.upper()+'_COMPLETE.json'),dict(metrics=metrics,samples=len(ds),candidate_slots=1001,duplicates_preserved=True,checkpoint_sha256=sha(c['model_path']),embedding_sha256=EXPECTED,teacher_sha256=job.get('teacher_sha256'),optimizer_updates=0,exploratory=True,time=now()))
def supervise(job,out):
 c=job['config'];env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=str(c['gpu']),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
 cmd=[PY,'-u','-B',str(O/'entry.py'),'--job',str(out/'job.json')]
 for phase,timeout in [('smoke',180),('train',14400),('dev',300),('test',300)]:
  if phase=='test':save(out/'TEST_STARTED.json',dict(time=now(),one_attempt=True,checkpoint_sha256=sha(out/'best.pt')))
  with (out/(phase+'.out')).open('xb') as f:
   child=subprocess.Popen(cmd+['--mode',phase],cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   save(out/(phase.upper()+'_PID.json'),dict(pid=child.pid,time=now(),gpu=c['gpu']))
   start=time.monotonic()
   try:rc=child.wait(timeout=timeout)
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    raise TimeoutError(phase+' timeout; owned process stopped, no retry')
  assert rc==0,(phase,rc)
  if phase=='train':
   log=(out/'train.log').read_text();assert ' END:' in log and 'Test After Training:' not in log
   losses=re.findall(r'Epoch\s+(\d+)\s+loss=\s*(\S+)',log);assert losses and all(math.isfinite(float(x[1])) for x in losses)
   best=int(re.findall(r'Best Iter\(dev\)=\s*(\d+)',log)[-1])
   save(out/'TRAIN_COMPLETE.json',dict(exit_code=0,best_epoch=best,end_epoch=int(losses[-1][0]),training_seconds=time.monotonic()-start,checkpoint_sha256=sha(out/'best.pt'),train_complete=True,Dev_best_selected=True,Test_accepted=False,time=now()))
   save(out/'DEV_CURVE.json',re.findall(r'Epoch\s+\d+\s+loss=.*',log))
 result=json.loads((out/'TRAIN_COMPLETE.json').read_text())
 result.update(Dev=json.loads((out/'DEV_COMPLETE.json').read_text())['metrics'],Test=json.loads((out/'TEST_COMPLETE.json').read_text())['metrics'],Test_accepted=True,model=c['model_name'],mechanism=job['mechanism'],Loss=job['loss'],embedding_sha256=EXPECTED,teacher_sha256=job.get('teacher_sha256'),parameter_count=json.loads((out/'SMOKE_COMPLETE.json').read_text())['parameter_count'],exploratory=True)
 save(out/'COMPLETE.json',result)
def main():
 p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--mode',required=True,choices=['supervise','smoke','train','dev','test']);a=p.parse_args()
 path=pathlib.Path(a.job);assert path.resolve()==path and path.is_relative_to(U)
 job=json.loads(path.read_text());out=path.parent;c=job['config']
 assert sha(A)==EXPECTED and all(sha(R/n)==h for n,h in job['source_sha256'].items())
 if job.get('teacher'):assert sha(job['teacher'])==job['teacher_sha256']
 os.environ['CUDA_VISIBLE_DEVICES']=str(c['gpu']);os.environ['PYTHONDONTWRITEBYTECODE']='1'
 if a.mode=='supervise':
  try:supervise(job,out)
  except BaseException as e:
   save(out/'FAILED.json',dict(error=repr(e),traceback=traceback.format_exc(),time=now(),train_complete=(out/'TRAIN_COMPLETE.json').exists(),Test_accepted=(out/'TEST_COMPLETE.json').exists(),no_retry=True));raise
  return
 bind(c,out,a.mode)
 if a.mode=='smoke':smoke(c,out,job)
 elif a.mode in ['dev','test']:evaluate(c,out,a.mode,job)
 else:
  assert c['train']==1 and c['load']==0 and c['dev_only']==1 and c['test_epoch']==-1
  assert json.loads((out/'SMOKE_COMPLETE.json').read_text())['passed']
  sys.argv=['main.py']+flags(c);runpy.run_path(str(R/'main.py'),run_name='__main__')
if __name__=='__main__':main()
