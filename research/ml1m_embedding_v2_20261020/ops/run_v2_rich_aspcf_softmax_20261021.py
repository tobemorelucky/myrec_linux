"""V2 embedding-only ASPCF SS run; original model/main/Reader/Runner unchanged."""
import pathlib,json,sys,os,argparse,hashlib,importlib,pickle,re,runpy,time,subprocess,datetime,signal
ROOT=pathlib.Path('/home/yyx/hzgcode/pom2.0')
WORK=ROOT/'research/ml1m_embedding_v2_20261020'
REF=WORK/'runs/p920_aspcf_ss_v1_rebuilt_k4_seed42_20261020_214225'
EXPECTED='e9e7f7d7fd2cdbf23c79102842826275e785314413f08e7e08ab28c557eee3eb'
PY='/home/yyx/anaconda3/envs/hzg_py10/bin/python'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2)
def flags(c):return [v for k,x in c.items() for v in ['--'+k,str(x)]]
def bind(c,out,mode):
 sys.path.insert(0,str(ROOT))
 p=pathlib.Path(c['llm_emb_path'])
 assert p.resolve()==p and sha(p)==EXPECTED
 m=importlib.import_module('models.sequential.LLMMIRecASPCF');original=m.load_llm_table
 def checked(path,expected_rows):
  assert pathlib.Path(path).resolve()==p and sha(p)==EXPECTED
  value=original(path,expected_rows)
  import numpy as np,torch
  with p.open('rb') as f:expected=pickle.load(f)
  assert expected.shape==(3707,1536) and expected.dtype==np.float32 and not expected[0].any() and np.isfinite(expected).all()
  assert torch.equal(value.cpu(),torch.from_numpy(expected))
  save(out/('EMBEDDING_'+mode.upper()+'.json'),dict(path=str(p),sha256=EXPECTED,shape=list(value.shape),dtype=str(value.dtype),model_buffer_verified=True))
  return value
 m.load_llm_table=checked
def build(c):
 import main
 from helpers.SeqReader import SeqReader
 from helpers.BaseRunner import BaseRunner
 from models.sequential.LLMMIRecASPCFSampledSoftmax import LLMMIRecASPCFSampledSoftmax as Model
 import torch
 p=argparse.ArgumentParser();p=main.parse_global_args(p);p=SeqReader.parse_data_args(p);p=BaseRunner.parse_runner_args(p);p=Model.parse_model_args(p)
 a,extra=p.parse_known_args(flags(c));assert extra in ([],['--model_name',c['model_name']])
 a.device=torch.device('cuda')
 from utils import utils
 utils.init_seed(a.random_seed)
 with (ROOT/'data/ml-1m/SeqReader.pkl').open('rb') as f:corpus=pickle.load(f)
 model=Model(a,corpus).to(a.device)
 return model,corpus,a,Model
def smoke(c,out):
 import numpy as np,torch
 from utils import utils
 model,corpus,a,Model=build(c);ds=Model.Dataset(model,corpus,'train')
 indices=np.random.RandomState(42).choice(len(ds),a.batch_size,replace=False)
 ds.data={k:np.asarray(v)[indices] for k,v in ds.data.items()};ds.actions_before_epoch()
 feed=utils.batch_to_gpu(ds.collate_batch([ds[i] for i in range(a.batch_size)]),model.device)
 model.train();torch.cuda.reset_peak_memory_stats()
 result=model(feed);loss=model.loss(result);assert torch.isfinite(loss)
 loss.backward();grads=[p.grad for p in model.parameters() if p.grad is not None]
 assert grads and all(torch.isfinite(g).all() for g in grads)
 save(out/'SMOKE_COMPLETE.json',dict(passed=True,loss=float(loss.detach()),train_batch=a.batch_size,history_shape=list(feed['history_items'].shape),parameter_count=model.count_variables(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),optimizer_steps=0,Train_only=True))
def dev(c,out):
 from helpers.BaseRunner import BaseRunner
 model,corpus,a,Model=build(c);model.load_model();model.eval()
 ds=Model.Dataset(model,corpus,'dev');ds.prepare()
 metrics=BaseRunner(a).evaluate(ds,[5,10,20],['HR','NDCG'])
 save(out/'BEST_DEV_FULL.json',dict(Dev=metrics,checkpoint_sha256=sha(out/'best.pt'),parameter_count=model.count_variables(),embedding_sha256=EXPECTED,test_called=False))
def supervise(c,out):
 # All training hyperparameters copied from successful V1 config, not defaults.
 ref=json.loads((REF/'config.json').read_text())
 changed={k:[ref.get(k),c.get(k)] for k in set(ref)|set(c) if ref.get(k)!=c.get(k)}
 assert set(changed)=={'gpu','llm_emb_path','log_file','model_path'},changed
 prov=json.loads((REF/'provenance.json').read_text())
 assert all(sha(ROOT/p)==h for p,h in prov['sources'].items()),'Baseline source changed'
 assert all(sha(ROOT/'data/ml-1m'/p)==h for p,h in prov['data_sha256'].items()),'Data changed'
 assert c['dev_only']==1 and c['test_epoch']==-1 and c['load']==0 and c['train']==1
 assert "phases = ['train', 'dev'] if args.dev_only" in (ROOT/'main.py').read_text()
 assert sha(c['llm_emb_path'])==EXPECTED
 save(out/'CONFIG_DIFF_VS_V1.json',changed)
 save(out/'provenance.json',dict(reference=str(REF),sources=prov['sources'],data_sha256=prov['data_sha256'],embedding_sha256=EXPECTED,from_scratch=True,test_forbidden=True,launcher_sha256=sha(__file__)))
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUDA_VISIBLE_DEVICES=str(c['gpu']))
 uuid=subprocess.check_output(['nvidia-smi','--id='+str(c['gpu']),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 assert not any(line.split(',')[0].strip()==uuid for line in procs.splitlines()),'Selected GPU busy'
 base=[PY,'-u','-B',str(pathlib.Path(__file__).resolve()),'--run-dir',str(out)]
 with (out/'smoke.out').open('xb') as f:
  done=subprocess.run(base+['--mode','smoke'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
 if done.returncode:raise RuntimeError('Train-only smoke failed; no full training')
 start=time.monotonic();began=datetime.datetime.now().astimezone().isoformat()
 with (out/'train.out').open('xb') as log:
  child=subprocess.Popen(base+['--mode','train'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  save(out/'child.json',dict(pid=child.pid,start=began,gpu=c['gpu'],command=base+['--mode','train']))
  try:rc=child.wait(timeout=14400)
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=60)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   rc=124
 result=dict(exit_code=rc,start=began,end=datetime.datetime.now().astimezone().isoformat(),training_seconds=time.monotonic()-start,test_called=False)
 if rc:save(out/'FAILED.json',result);save(out/'exit.json',result);return
 text=(out/'train.log').read_text()
 assert 'Test After Training:' not in text and 'Best Iter(dev)=' in text
 result['best_epoch']=int(re.findall(r'Best Iter\(dev\)=\s*(\d+)',text)[-1])
 result['end_epoch']=int(re.findall(r'Epoch\s+(\d+)\s+loss=',text)[-1])
 result['checkpoint_sha256']=sha(out/'best.pt')
 save(out/'DEV_CURVE.json',re.findall(r'Epoch\s+\d+\s+loss=.*',text))
 with (out/'dev.out').open('xb') as f:
  done=subprocess.run(base+['--mode','dev'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
 if done.returncode:raise RuntimeError('Training ended but best Dev evaluation failed')
 result.update(json.loads((out/'BEST_DEV_FULL.json').read_text()))
 result['NDCG5_delta_vs_V1_Rebuilt']=result['Dev']['NDCG@5']-.2721348685543765
 result['NDCG5_relative_vs_V1_Rebuilt']=result['NDCG5_delta_vs_V1_Rebuilt']/.2721348685543765
 save(out/'COMPLETE.json',result);save(out/'exit.json',result)
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--mode',choices=['supervise','smoke','train','dev'],required=True)
 a=p.parse_args();out=pathlib.Path(a.run_dir).resolve()
 assert out.is_relative_to(WORK) and out.is_dir()
 os.chdir(ROOT);c=json.loads((out/'config.json').read_text());os.environ['CUDA_VISIBLE_DEVICES']=str(c['gpu'])
 if a.mode=='supervise':
  try:supervise(c,out)
  except BaseException as e:
   if not (out/'FAILED.json').exists():save(out/'FAILED.json',dict(error=repr(e),test_called=False))
   raise
  return
 bind(c,out,a.mode)
 if a.mode=='smoke':smoke(c,out)
 elif a.mode=='dev':dev(c,out)
 else:
  save(out/'TRAIN_PID.json',dict(pid=os.getpid(),gpu=c['gpu'],time=datetime.datetime.now().astimezone().isoformat()))
  sys.argv=['main.py']+flags(c);runpy.run_path(str(ROOT/'main.py'),run_name='__main__')
if __name__=='__main__':main()
