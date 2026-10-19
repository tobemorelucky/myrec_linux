"""Bounded original-ASPCF O0/O1 Query-only coupled-L2 attribution; Train only."""
import os,sys,time,json,pickle,hashlib,copy,random,argparse,signal,subprocess,traceback,logging,math
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF as Model
from models.BaseModel import GeneralModel
from helpers.BaseRunner import BaseRunner
from utils import utils
POINTS={0,1,10,50,100,200}
PARAMS=['extractor.query','extractor.Wq.weight','extractor.Wq.bias','extractor.Wk.weight','extractor.Wk.bias']
SOURCES=['models/sequential/LLMMIRecASPCF.py','models/sequential/llmmi_components.py','models/BaseModel.py','helpers/BaseRunner.py','helpers/SeqReader.py','utils/utils.py','tools/diagnose_aspcf_query_decay_paired200.py']
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def bytesha(b):return hashlib.sha256(b).hexdigest()
def dump_new(path,x):
 p=Path(path);assert p.resolve().is_relative_to(ROOT) and not p.exists()
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def rng():
 return (torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone(),np.random.get_state(),random.getstate())
def restore(r):
 torch.set_rng_state(r[0]);torch.cuda.set_rng_state(r[1]);np.random.set_state(r[2]);random.setstate(r[3])
def rnghash(r):
 return dict(cpu=bytesha(r[0].numpy().tobytes()),cuda=bytesha(r[1].cpu().numpy().tobytes()),numpy=bytesha(pickle.dumps(r[2])),python=bytesha(pickle.dumps(r[3])))
def digest(m):
 d=hashlib.sha256()
 for n,t in m.state_dict().items():d.update(n.encode());d.update(t.detach().cpu().contiguous().numpy().tobytes())
 return d.hexdigest()
def number(t):return float(t.detach())
def norm(t):return number(t.double().norm())
def cosine(a,b):
 a=a.detach().double().reshape(-1);b=b.detach().double().reshape(-1);den=a.norm()*b.norm()
 return number(a.dot(b)/den) if den>1e-300 else None
def config(ds,out):
 hist=ROOT/'new_log/llmmirec_aspcf_K2'/ds/'seed42/LLMMIRecASPCF_seed42.log'
 p=argparse.ArgumentParser();p.add_argument('--random_seed',type=int,default=42)
 p.add_argument('--dataset');p.add_argument('--gpu',default='1')
 p=BaseRunner.parse_runner_args(p);p=Model.parse_model_args(p);old={}
 known={a.dest for a in p._actions}
 for l in hist.read_text().splitlines():
  if '|' in l:
   s=l.split('|');k=s[0].strip();v=s[1].strip()
   if k in known and '...' not in v:old[k]=v
 cli=[s for k,v in old.items() for s in ['--'+k,v]]
 a=p.parse_args(cli);a.device=torch.device('cuda');a.gpu='1';a.llm_emb_path=str(ROOT/'data'/ds/'handled/llm_table_pca1536.pkl')
 a.model_path=str(out/'NO_CHECKPOINT_IS_SAVED')
 assert (a.K,a.history_max,a.batch_size,a.random_seed,a.num_neg,a.lambda_relation,a.l2,a.optimizer)==(2,20,1024,42,1,.01,1e-6,'Adam')
 assert a.lr==(.004 if ds=='beauty' else .001) and a.dropout==.1 and a.grad_clip==0 and a.num_workers==5
 return a,dict(historical_log=str(hist),historical_log_sha256=sha(hist),historical_logged_arguments=old,
  effective_arguments={k:str(v) if isinstance(v,torch.device) else v for k,v in vars(a).items()})
def optimizer_groups(opt,m):
 names={id(p):n for n,p in m.named_parameters()}
 return [{**{k:v for k,v in g.items() if k!='params'},'parameter_names':[names[id(p)] for p in g['params']],'parameter_count':sum(p.numel() for p in g['params'])} for g in opt.param_groups]
def validate_optimizers(ms,opts,l2):
 maps=[]
 for m,o in zip(ms,opts):
  ids=[id(p) for g in o.param_groups for p in g['params']]
  assert len(ids)==len(set(ids)) and set(ids)=={id(p) for p in m.parameters() if p.requires_grad}
  mp={n:{k:v for k,v in g.items() if k!='params'} for n,p in m.named_parameters() for g in o.param_groups if any(p is z for z in g['params'])}
  maps.append(mp)
 for n in maps[0]:
  a,b=maps[0][n],maps[1][n]
  assert all(a[k]==b[k] for k in a if k!='weight_decay')
  assert (a['weight_decay'],b['weight_decay'])==((l2,0) if n=='extractor.query' else (a['weight_decay'],a['weight_decay']))
 return True
def fixture(corpus,device):
 df=corpus.data_df['train'];good=df[df.position>0];groups=good.groupby('user_id',sort=False).indices
 local=np.random.default_rng(42);users=local.choice(np.array(list(groups)),64,replace=False);hs=[];ls=[];ids=[];us=[]
 for u in users:
  row=good.iloc[int(local.choice(groups[u]))];pos=int(row.position)
  seq=corpus.user_his[int(u)][max(0,pos-20):pos];assert seq and all(t<=row.time for _,t in seq)
  h=[int(j) for j,_ in seq];neg=int(local.integers(1,corpus.n_items))
  while neg in corpus.train_clicked_set[int(u)]:neg=int(local.integers(1,corpus.n_items))
  hs.append(h+[0]*(20-len(h)));ls.append(len(h));ids.append([int(row.item_id),neg]);us.append(int(u))
 return {k:torch.tensor(v,device=device) for k,v in dict(history_items=hs,lengths=ls,item_id=ids,user_id=us).items()}
def probe(m,b,l2):
 was=m.training;checked=m._first_batch_checked
 try:
  m.eval();out=m(b,True);loss=GeneralModel.loss(m,out)
  named=dict(m.named_parameters());gs=torch.autograd.grad(loss,[named[n] for n in PARAMS])
  v=out['interest_vectors'];att=out['attention_maps'];margin=out['prediction'][:,0]-out['prediction'][:,1]
  q=named['extractor.query'];Q=m.extractor.Wq(q)
  record=dict(query_norm=norm(q),pure_BPR_loss=number(loss),query_task_gradient_norm=norm(gs[0]),
   original_lambda_query_norm=l2*norm(q),task_to_original_L2_ratio=norm(gs[0])/max(l2*norm(q),1e-300),
   attention_head_L1=number((att[:,0]-att[:,1]).abs().sum(-1).mean()),
   relative_interest_head_difference=number(((v[:,0]-v[:,1]).norm(dim=-1)/v.norm(dim=-1).mean(-1).clamp_min(1e-12)).mean()),
   margin_mean=number(margin.mean()),margin_median=number(margin.median()),
   margin_min=number(margin.min()),margin_max=number(margin.max()),
   Wq_task_gradient_norm=math.sqrt(norm(gs[1])**2+norm(gs[2])**2),
   Wk_task_gradient_norm=math.sqrt(norm(gs[3])**2+norm(gs[4])**2))
  tensors=dict(query=q.detach().clone(),effective_query=Q.detach(),attention=att.detach(),interest=v.detach(),prediction=out['prediction'].detach(),margin=margin.detach())
  assert all(torch.isfinite(g).all() for g in gs)
  return record,tensors
 finally:m.train(was);m._first_batch_checked=checked
def pair_probe(ms,b,step,l2,updates):
 before=rng();parts=[]
 for m in ms:
  rec,ts=probe(m,b,l2);parts.append((rec,ts))
 assert rnghash(before)==rnghash(rng())
 a,bv=parts[0][1],parts[1][1]
 ds=[]
 for n in ['query','effective_query']:
  x,y=a[n],bv[n];ds.append((n,dict(relative_difference=norm(y-x)/max(norm(x),1e-300),cosine=cosine(x,y),
   unit_direction_distance=norm(torch.nn.functional.normalize(x,dim=-1)-torch.nn.functional.normalize(y,dim=-1)))))
 s0=s1=sd=0.
 with torch.no_grad():
  for p,z in zip(ms[0].parameters(),ms[1].parameters()):
   s0+=number(p.double().square().sum());s1+=number(z.double().square().sum());sd+=number((p.double()-z.double()).square().sum())
 differences=dict(ds)
 differences.update(all_parameter_L2_difference=math.sqrt(sd),all_parameter_relative_difference=math.sqrt(sd/max(s0,1e-300)),
  prediction_MAE=number((a['prediction']-bv['prediction']).abs().mean()),
  prediction_relative_L2=norm(a['prediction']-bv['prediction'])/max(norm(a['prediction']),1e-300),
  attention_between_arm_L1=number((a['attention']-bv['attention']).abs().sum(-1).mean()),
  interest_between_arm_relative_L2=norm(a['interest']-bv['interest'])/max(norm(a['interest']),1e-300),
  margin_mean_O1_minus_O0=number((bv['margin']-a['margin']).mean()))
 return dict(completed_steps=step,O0=parts[0][0],O1=parts[1][0],differences=differences,
  actual_training_update=updates,probe_RNG_unchanged=True,
  note='Probe is dropout-off fixed Train; update diagnostics refer to the just-completed stochastic training step, not the probe gradient.')
def adam_query_prediction(q,g,opt,wd):
 group=next(x for x in opt.param_groups if any(p is q for p in x['params']))
 assert not group['amsgrad'] and not group['maximize'] and not group.get('decoupled_weight_decay',False)
 st=opt.state[q];n=int(st['step'].item())+1 if 'step' in st else 1
 m=st.get('exp_avg',torch.zeros_like(q));v=st.get('exp_avg_sq',torch.zeros_like(q));b1,b2=group['betas']
 grad=g+wd*q.detach()
 mm=m*b1+grad*(1-b1);vv=v*b2+grad.square()*(1-b2)
 delta=-group['lr']/(1-b1**n)*mm/(vv.sqrt()/math.sqrt(1-b2**n)+group['eps'])
 return (q.detach()+delta)-q.detach()
def update_arm(m,opt,batch,record,l2):
 named=dict(m.named_parameters());selected=[named[n] for n in PARAMS]
 opt.zero_grad();out=m(batch);rank=GeneralModel.loss(m,out);loss=m.loss(out)
 assert torch.isfinite(loss),'Non-finite total loss'
 before=None;task=None;cf=None;pred=None;decay=None
 if record:
  task=torch.autograd.grad(rank,selected,retain_graph=True)
  before=[p.detach().clone() for p in selected]
  wd=next(g['weight_decay'] for g in opt.param_groups if any(p is selected[0] for p in g['params']))
  pred=adam_query_prediction(selected[0],task[0].detach(),opt,wd)
  cf=adam_query_prediction(selected[0],task[0].detach(),opt,0)
  decay=l2*before[0]
 loss.backward()
 if record:
  assert all(torch.isfinite(p.grad).all() for p in selected)
  assert torch.allclose(selected[0].grad,task[0],atol=1e-12,rtol=1e-5),'relation unexpectedly changes direct Query gradient'
 opt.step()
 rec=None
 if record:
  delta=[p.detach()-old for p,old in zip(selected,before)];dq=delta[0]
  error=norm(dq-pred)/max(norm(dq),1e-300)
  assert error<1e-3,'Adam predicted step mismatch'
  rec=dict(main_BPR_loss=number(rank),relation_loss=number(out['loss_relation']),total_loss=number(loss),
   query_pre_norm=norm(before[0]),query_post_norm=norm(selected[0]),
   query_task_gradient_norm=norm(task[0]),original_L2_gradient_norm=norm(decay),effective_query_weight_decay=wd,
   query_task_to_original_L2_ratio=norm(task[0])/max(norm(decay),1e-300),
   query_update_norm=norm(dq),query_relative_update=norm(dq)/max(norm(before[0]),1e-300),
   update_cos_negative_task=cosine(dq,-task[0]),update_cos_negative_decay=cosine(dq,-decay),
   task_cos_decay=cosine(task[0],decay),
   same_history_no_current_decay_counterfactual_update_norm=norm(cf),
   actual_vs_no_current_decay_update_relative_difference=norm(dq-cf)/max(norm(dq),1e-300),
   actual_vs_no_current_decay_update_cosine=cosine(dq,cf),Adam_formula_relative_error=error,
   Wq_task_gradient_norm=math.sqrt(norm(task[1])**2+norm(task[2])**2),
   Wq_update_norm=math.sqrt(norm(delta[1])**2+norm(delta[2])**2),
   Wk_task_gradient_norm=math.sqrt(norm(task[3])**2+norm(task[4])**2),
   Wk_update_norm=math.sqrt(norm(delta[3])**2+norm(delta[4])**2),
   counterfactual_scope='Only current Query decay removed while retaining actual previous Adam moments. Not a full no-decay training history.')
 return rec,out
def run_dataset(ds,out):
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Per-dataset 170-second bound')));signal.alarm(170)
 begin=time.perf_counter();args,c=config(ds,out);utils.init_seed(args.random_seed)
 with (ROOT/'data'/ds/'SeqReader.pkl').open('rb') as f:corpus=pickle.load(f)
 m0=Model(args,corpus).to(args.device);before=rng();m1=copy.deepcopy(m0);assert rnghash(before)==rnghash(rng())
 ms=[m0,m1];initial=[digest(m) for m in ms];assert initial[0]==initial[1]
 runner=BaseRunner(args);opts=[runner._build_optimizer(m) for m in ms]
 q=m1.extractor.query
 for g in opts[1].param_groups:g['params']=[p for p in g['params'] if p is not q]
 opts[1].add_param_group(dict(params=[q],weight_decay=0.))
 assert validate_optimizers(ms,opts,args.l2)
 for m in ms:m.train()
 batch_probe=fixture(corpus,args.device)
 result=dict(dataset=ds,success=False,config=c,initial_state_sha256=initial[0],same_initial_state=True,
   parameters=m0.count_variables(),optimizer_defaults=opts[0].defaults,
   optimizer_groups={arm:optimizer_groups(o,m) for arm,o,m in zip(['O0','O1'],opts,ms)},
   no_duplicate_missing_or_idle_parameters=True,checkpoints_written=False,dev_test_executed=False,
   records=[pair_probe(ms,batch_probe,0,args.l2,None)],epochs=[],all_steps_RNG_equal=True,all_relation_samples_equal=True)
 dataset=Model.Dataset(m0,corpus,'train');assert dataset.phase=='train'
 result['train_rows']=len(dataset);result['asset_sha256']={str((ROOT/'data'/ds/f).relative_to(ROOT)):sha(ROOT/'data'/ds/f) for f in ['train.csv','SeqReader.pkl','handled/llm_table_pca1536.pkl']};step=0;order=hashlib.sha256()
 torch.cuda.reset_peak_memory_stats()
 for epoch in [1,2]:
  if step>=200:break
  before_epoch=time.perf_counter();dataset.actions_before_epoch()
  negative_sha=bytesha(np.ascontiguousarray(dataset.data['neg_items']).tobytes())
  loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,
   collate_fn=dataset.collate_batch,pin_memory=args.pin_memory)
  rows=0
  for b in loader:
   if step>=200:break
   step+=1
   order.update(b['user_id'].numpy().tobytes());order.update(b['item_id'].numpy().tobytes())
   rows+=int(b['item_id'].shape[0])
   batch=utils.batch_to_gpu(b,args.device)
   for m in ms:m._cur_epoch=epoch;m.cur_epoch=epoch;m.align_warmup=args.align_warmup;m.align_ramp=args.align_ramp
   pre=rng();records=[];outs=[]
   for arm,(m,o) in enumerate(zip(ms,opts)):
    restore(pre);t=time.perf_counter()
    rec,output=update_arm(m,o,batch,step in POINTS,args.l2);records.append(rec);outs.append(output)
    post=rng()
    if arm==0:after0=post
    else:assert rnghash(after0)==rnghash(post),'Dropout/relation RNG diverged'
   assert torch.equal(outs[0]['_relation_ids'],outs[1]['_relation_ids']),'Relation samples differ'
   if step==1:assert torch.equal(outs[0]['prediction'],outs[1]['prediction']),'First stochastic predictions differ'
   del outs,output
   if step in POINTS:
    torch.cuda.synchronize()
    result['records'].append(pair_probe(ms,batch_probe,step,args.l2,dict(O0=records[0],O1=records[1])))
    print('POINT '+json.dumps(dict(dataset=ds,steps=step,O0_margin=result['records'][-1]['O0']['margin_mean'],O1_margin=result['records'][-1]['O1']['margin_mean'])),flush=True)
  result['epochs'].append(dict(epoch=epoch,steps_completed=step,rows_consumed=rows,full_epoch_negative_sha256=negative_sha,
      consumed_order_and_candidates_sha256=order.hexdigest(),seconds=time.perf_counter()-before_epoch))
 assert step==200
 torch.cuda.synchronize();signal.alarm(0)
 result.update(success=True,completed_steps=step,seconds=time.perf_counter()-begin,
   peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
   sample_order_and_negatives_same_by_construction=True,consumed_order_and_candidates_sha256=order.hexdigest(),
   final_state_sha256={arm:digest(m) for arm,m in zip(['O0','O1'],ms)},
   protocol_note='Original Dataset/actions_before_epoch/DataLoader shuffle; the same GPU batch is fed to both arms. All CPU/CUDA/NumPy/Python states are restored before O1 and advanced once per paired step. No Dev loader RNG consumption between Train epochs.',
   original_training_difference='Short diagnostic omits Dev and its DataLoader RNG draws; two arms are exactly paired, but after an epoch boundary not claimed bitwise identical to a historical full Train+Dev run.')
 return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=(ROOT/a.output).resolve()
 assert out.is_relative_to(ROOT) and not out.exists(),'New project-local output required'
 gpu=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 assert float(gpu[1])>7000 and gpu[0].strip() not in apps,'GPU1 occupied; do not interfere'
 out.mkdir(parents=True,exist_ok=False);logging.basicConfig(level=logging.WARNING)
 state=dict(status='running',gpu=1,source_sha256={f:sha(ROOT/f) for f in SOURCES},datasets={})
 start=time.perf_counter()
 try:
  for ds in ['beauty','ml-1m']:
   state['datasets'][ds]=run_dataset(ds,out)
   dump_new(out/(ds+'.json'),state['datasets'][ds])
  state['status']='complete'
 except BaseException as exc:
  state['error']=repr(exc);state['traceback']=traceback.format_exc();state['status']='failed';raise
 finally:
  state['seconds']=time.perf_counter()-start
  state['source_unchanged']={f:sha(ROOT/f)==v for f,v in state['source_sha256'].items()}
  dump_new(out/'results.json',state)
 print('COMPLETE '+json.dumps(dict(output=str(out/'results.json'),seconds=state['seconds'],source_unchanged=state['source_unchanged'])),flush=True)
if __name__=='__main__':main()
