"""Only six authorized 10-epoch fresh-Adam continuations. No main.py or Test entry."""
import os,sys,json,time,hashlib,pickle,random,argparse,copy,subprocess,traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from helpers.BaseRunner import BaseRunner
from models.BaseModel import GeneralModel
from models.sequential.LLMMIRecASPCFMultiPositive import LLMMIRecASPCFMultiPositive as Model, auxiliary_bpr,exclusive_assignment
from utils import utils
EXPECTED={'beauty':('e123e0f9f3b5c7825ab3cd6e08982de0b5637880f2e34a4e159c43568aff641d',.1332),'ml-1m':('1285b09e5d98526a50f51a2dda3a22edddb21613a78c25392ab6c49f611b9d45',.2298)}
MODES={'C0':'off','Cextra':'extra','Cexclusive':'exclusive'}
def write(p,x):
 p=Path(p);assert p.resolve().is_relative_to(ROOT) and not p.exists();p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seed():random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
def rng():return dict(cpu=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),cuda=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest(),numpy=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),python=hashlib.sha256(pickle.dumps(random.getstate())).hexdigest())
def stat(x):
 x=x.detach().double().flatten();return dict(mean=float(x.mean()),median=float(x.median()),p10=float(torch.quantile(x,.1)),p90=float(torch.quantile(x,.9)),max=float(x.max()))
def make(ds,arm,rid):
 seed();parser=BaseRunner.parse_runner_args(Model.parse_model_args(argparse.ArgumentParser()))
 source=ROOT/'new_model/llmmirec_aspcf_K2'/ds/'seed42/LLMMIRecASPCF_seed42.pt'
 log=ROOT/'new_log/llmmirec_aspcf_K2'/ds/'seed42/LLMMIRecASPCF_seed42.log'
 old={}
 for line in log.read_text().splitlines():
  parts=line.split('|')
  if len(parts)>=2 and parts[0].strip():old[parts[0].strip()]=parts[1].strip()
 tokens=[]
 for action in parser._actions:
  if action.dest in old and action.dest!='help':tokens.extend([action.option_strings[0],old[action.dest]])
 a=parser.parse_args(tokens);a.device=torch.device('cuda:0');a.buffer=0;a.mp_mode=MODES[arm];a.lambda_aux=.1;a.epoch=10;a.early_stop=0;a.test_epoch=-1;a.topk='5,10,20';a.metric='NDCG,HR'
 a.llm_emb_path=str(ROOT/'data'/ds/'handled/llm_table_pca1536.pkl')
 folder=ROOT/'new_log/multi_positive_matched10'/rid/ds/arm
 weights=ROOT/'new_model/multi_positive_matched10'/rid/ds/arm
 a.model_path=str(weights/'best.pt')
 assert a.K==2 and a.history_max==20 and a.batch_size==1024 and a.num_neg==1 and a.dropout==.1 and a.lambda_relation==.01 and a.l2==1e-6 and a.optimizer=='Adam'
 assert a.lr==(.004 if ds=='beauty' else .001) and a.grad_clip==0
 assert sha(source)==EXPECTED[ds][0]
 # Reuse trusted cache; construct Train/Dev Dataset objects only. No Test object or evaluation.
 with (ROOT/'data'/ds/'SeqReader.pkl').open('rb') as f:c=pickle.load(f)
 c.data_df={k:c.data_df[k] for k in ['train','dev']}
 m=Model(a,c).to(a.device);m.load_state_dict(torch.load(source,map_location=a.device,weights_only=True),strict=True)
 train=m.Dataset(m,c,'train');dev=m.Dataset(m,c,'dev');runner=BaseRunner(a)
 assert len(dev.data['neg_items'][0])==1000 and m.test_all==0
 return a,m,train,dev,runner,folder,weights,source

def query_update(model,gradient,runner,wd):
 q=model.extractor.query;p=torch.nn.Parameter(q.detach().clone())
 op=torch.optim.Adam([p],lr=runner.learning_rate,weight_decay=wd)
 if model.optimizer is not None and q in model.optimizer.state:op.state[p]=copy.deepcopy(model.optimizer.state[q])
 p.grad=gradient.detach().clone();op.step();return p.detach()-q.detach()

def probe(m,b,runner,epoch,baseline):
 before=rng();was=m.training;checked=m._first_batch_checked
 with torch.random.fork_rng(devices=[0]):
  m.eval();o=m(b,True);q=m.extractor.query;v=o['interest_vectors'];att=o['attention_maps']
  positive=torch.stack([o['candidate_vectors'][:,0],m.item_encoder(b['mp_extra_item'])],1);negative=o['candidate_vectors'][:,1]
  valid=b['mp_extra_valid'].bool() & (b['mp_extra_item']!=b['item_id'][:,1]) & (b['lengths']>0)
  ex,de=auxiliary_bpr(v,o['user_vector'],positive,negative,valid,b['mp_assignment_key'],'exclusive')
  extra,_=auxiliary_bpr(v,o['user_vector'],positive,negative,valid,b['mp_assignment_key'],'extra')
  main=GeneralModel.loss(m,o)
  gm=torch.autograd.grad(main,q,retain_graph=True)[0];ge=torch.autograd.grad(extra,q,retain_graph=True)[0];gx=torch.autograd.grad(ex,q,retain_graph=True)[0]
  task=gm+(.1*ge if m.mp_mode=='extra' else .1*gx if m.mp_mode=='exclusive' else 0)
  upd=query_update(m,task,runner,runner.l2);du=query_update(m,torch.zeros_like(task),runner,runner.l2);tu=query_update(m,task,runner,0)
  gap=((v[:,0].double()-v[:,1].double())*(positive[:,0].double()-positive[:,1].double())).sum(-1)
  scale=torch.einsum('bkd,bmd->bkm',v,positive).abs().mean((1,2)).clamp_min(1e-12)
  assign=de['assignment'][:,0].detach();base=assign.clone() if baseline is None else baseline
  record=dict(epoch=epoch,rows=len(valid),active=int(valid.sum()),mode='fixed Train dropout-off gradient probe; optimizer state cloned, model not updated',attention_head_L1=stat((att[:,0]-att[:,1]).abs().sum(-1)),V_relative_difference=stat((v[:,0]-v[:,1]).norm(dim=-1)/v.norm(dim=-1).mean(-1).clamp_min(1e-30)),delta_abs=stat(gap[valid].abs()),delta_relative_score=stat((gap.abs()/scale)[valid]),relative_delta_lt_1e6=float((gap.abs()/scale<1e-6)[valid].float().mean()),relative_delta_lt_1e4=float((gap.abs()/scale<1e-4)[valid].float().mean()),exact_tie_fraction=float(de['tied'][valid].float().mean()),head0=int(((assign==0)&valid).sum()),head1=int(((assign==1)&valid).sum()),assignment_change_from_epoch0=float((assign!=base)[valid].float().mean()),main_loss=float(main.detach()),extra_aux_loss=float(extra.detach()),exclusive_aux_loss=float(ex.detach()),query_norm=float(q.norm()),query_main_gradient=float(gm.norm()),query_extra_gradient=float(ge.norm()),query_exclusive_gradient=float(gx.norm()),query_total_task_norm=float(task.norm()),query_decay_norm=float((runner.l2*q).norm()),task_over_decay=float(task.norm()/(runner.l2*q).norm().clamp_min(1e-30)),cloned_optimizer_update_norm=float(upd.norm()),cloned_decay_and_momentum_update_norm=float(du.norm()),cloned_actual_vs_no_task_relative_difference=float((upd-du).norm()/upd.norm().clamp_min(1e-30)),cloned_actual_vs_no_decay_relative_difference=float((upd-tu).norm()/upd.norm().clamp_min(1e-30)))
  assert all(torch.isfinite(g).all() for g in [gm,ge,gx,upd]);m.train(was);m._first_batch_checked=checked
 assert before==rng(),'Probe altered training RNG'
 return record,base

def execute(ds,arm,rid,stage):
 torch.set_num_threads(2);a,m,train,dev,runner,folder,weights,source=make(ds,arm,rid);started=time.time()
 state=dict(dataset=ds,arm=arm,stage=stage,success=False,test_executed=False,checkpoint_source=str(source),source_sha256=sha(source),parameters=m.count_variables(),config={k:str(v) if k=='device' else v for k,v in vars(a).items()},epochs=[],probes=[])
 # Cache a fixed Train batch and original negative realization, then reset all random streams.
 np.random.seed(42);train.actions_before_epoch();pick=np.random.default_rng(20261017).choice(len(train),1024,replace=False)
 fixed=utils.batch_to_gpu(train.collate_batch([train[int(i)] for i in pick]),m.device)
 seed();initial=runner.evaluate(dev,[5,10,20],['NDCG','HR']);assert abs(initial['NDCG@5']-EXPECTED[ds][1])<.000051
 state['epoch0']=initial;state['shared_initial_sha256']=hashlib.sha256(b''.join(n.encode()+v.detach().cpu().numpy().tobytes() for n,v in m.state_dict().items())).hexdigest()
 if stage=='prepare':write(folder/'epoch0.json',state);print(json.dumps(dict(dataset=ds,arm=arm,epoch0=initial)),flush=True);return
 assert stage=='train' and not (folder/'summary.json').exists() and not weights.exists();weights.mkdir(parents=True)
 record,baseline=probe(m,fixed,runner,0,None);state['probes'].append(record)
 write(folder/'probe_00.json',record)
 seed();m.optimizer=runner._build_optimizer(m);best=initial['NDCG@5'];state['best_epoch']=0;state['best_dev']=initial
 m.save_model(str(weights/'best.pt'));torch.cuda.reset_peak_memory_stats()
 try:
  for epoch in range(1,11):
   t=time.time();entry=dict(epoch=epoch,rng_start=rng());train.actions_before_epoch();entry['negative_sha256']=hashlib.sha256(np.ascontiguousarray(train.data['neg_items']).tobytes()).hexdigest()
   dl=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers,collate_fn=train.collate_batch,pin_memory=bool(a.pin_memory));order=hashlib.sha256();m.train();m.cur_epoch=epoch;losses=[];auxes=[];updates=0
   for step,bb in enumerate(dl):
    order.update(bb['user_id'].numpy().tobytes());order.update(bb['item_id'][:,0].numpy().tobytes())
    bb=utils.batch_to_gpu(bb,m.device);m.optimizer.zero_grad();out=m(bb);loss=m.loss(out);assert torch.isfinite(loss);loss.backward()
    selected=(epoch in (1,10) and step==0)
    if selected:
     q=m.extractor.query;before=q.detach().clone();g=q.grad.detach().clone();entry['first_query_task_norm']=float(g.norm());entry['first_query_decay_norm']=float((a.l2*q).norm())
    m.optimizer.step()
    if selected:entry['first_actual_query_update_norm']=float((q.detach()-before).norm())
    losses.append(float(loss.detach()));auxes.append(float(out['_mp_aux'].detach()) if '_mp_aux' in out else 0.);updates+=1
   entry.update(train_seconds=time.time()-t,loss=float(np.mean(losses)),aux_loss=float(np.mean(auxes)),steps=updates,order_sha256=order.hexdigest(),rng_after_fit=rng())
   t=time.time();entry['dev']=runner.evaluate(dev,[5,10,20],['NDCG','HR']);entry['dev_seconds']=time.time()-t
   if entry['dev']['NDCG@5']>=best:
    best=entry['dev']['NDCG@5'];state['best_epoch']=epoch;state['best_dev']=entry['dev'];m.save_model(str(weights/'best.pt'))
   state['epochs'].append(entry);write(folder/('epoch_%02d.json'%epoch),entry)
   if epoch in (1,10):record,_=probe(m,fixed,runner,epoch,baseline);state['probes'].append(record);write(folder/('probe_%02d.json'%epoch),record)
   print(json.dumps(dict(dataset=ds,arm=arm,epoch=epoch,dev=entry['dev'],train_seconds=entry['train_seconds'])),flush=True)
  m.save_model(str(weights/'final_epoch10.pt'));state['final_dev']=state['epochs'][-1]['dev'];state['checkpoint_sha256']=sha(weights/'best.pt');state['final_checkpoint_sha256']=sha(weights/'final_epoch10.pt');state['success']=True
 finally:
  state['seconds']=time.time()-started;state['peak_allocated_MiB']=torch.cuda.max_memory_allocated()/2**20;state['peak_reserved_MiB']=torch.cuda.max_memory_reserved()/2**20
  assert sha(source)==state['source_sha256'];write(folder/'summary.json',state)

def wait_gpu():
 while True:
  row=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[1].split(',')
  if float(row[1])<=60 and float(row[2])>=4000:return
  print('Waiting for GPU1 without interfering with other processes',flush=True);time.sleep(60)
def orchestrate(rid):
 folder=ROOT/'new_log/multi_positive_matched10'/rid;folder.mkdir(parents=True,exist_ok=False)
 def call(ds,arm,stage):
  wait_gpu();p=folder/(ds+'_'+arm+'_'+stage+'.out')
  with p.open('xb') as f:child=subprocess.Popen([sys.executable,'-B',__file__,'run',ds,arm,rid,stage],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
  assert child.wait()==0,(ds,arm,stage)
 for ds in EXPECTED:
  for arm in MODES:call(ds,arm,'prepare')
  gates=[json.loads((folder/ds/arm/'epoch0.json').read_text()) for arm in MODES]
  assert all(x['epoch0']==gates[0]['epoch0'] and x['shared_initial_sha256']==gates[0]['shared_initial_sha256'] for x in gates)
 write(folder/'all_epoch0_passed.json',dict(success=True,test_executed=False))
 for ds in EXPECTED:
  for arm in MODES:call(ds,arm,'train')
  runs=[json.loads((folder/ds/arm/'summary.json').read_text()) for arm in MODES]
  for other in runs[1:]:
   for x,y in zip(runs[0]['epochs'],other['epochs']):
    assert all(x[k]==y[k] for k in ['rng_start','rng_after_fit','negative_sha256','order_sha256','steps']),('Matching failure',ds,x['epoch'])
  comparison=[dict(epoch=i,exclusive_minus_extra=runs[2]['epochs'][i-1]['dev']['NDCG@5']-runs[1]['epochs'][i-1]['dev']['NDCG@5'],exclusive_minus_C0=runs[2]['epochs'][i-1]['dev']['NDCG@5']-runs[0]['epochs'][i-1]['dev']['NDCG@5']) for i in range(1,11)]
  write(folder/(ds+'_comparison.json'),dict(matched_checks=True,epochs=comparison,summaries=runs))
 write(folder/'complete.json',dict(success=True,datasets=list(EXPECTED),epochs=10,test_executed=False))
if __name__=='__main__':
 if sys.argv[1]=='orchestrate':orchestrate(sys.argv[2])
 elif sys.argv[1]=='run':execute(*sys.argv[2:])
 else:raise ValueError(sys.argv[1])
