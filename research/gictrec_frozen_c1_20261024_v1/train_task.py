"""Exactly three frozen-C1 branch-only runs, epoch0 eligible, one fixed-best Test."""
import argparse,traceback
from common import *
class Runner(old.BaseRunner):
 def __init__(self,a,p):super().__init__(a);self.path=p;self.updates=0
 def evaluate(self,dataset,topks,metrics_):
  assert dataset.phase=='dev'
  got=score_views(dataset);protocol_lock('dev',got)
  self.last=dict(epoch=int(dataset.model.cur_epoch),raw=got['metrics']['raw'],locked_prior=got['metrics']['locked_prior'],metrics=got['metrics']['locked_prior'],selection='locked prior Dev NDCG@5',input_hash=got['input_hash'])
  with (self.path/'dev_metrics.jsonl').open('a') as f:f.write(json.dumps(self.last)+'\n')
  return self.last['metrics']
 def fit(self,dataset,epoch=-1):
  m=dataset.model
  if m.optimizer is None:m.optimizer=torch.optim.Adam([m.transfer.output.weight],lr=.001,weight_decay=self.l2)
  dataset.actions_before_epoch();m.cur_epoch=epoch;m.actions_before_epoch();m.train()
  assert not m.dropout.training and not m.item_encoder.training and not m.sil_block.training
  assert [n for n,p in m.named_parameters() if p.requires_grad]==['transfer.output.weight']
  loader=DataLoader(dataset,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers,collate_fn=dataset.collate_batch,pin_memory=self.pin_memory)
  losses=[];first=None;last=None;relations=[]
  for batch in loader:
   feed=utils.batch_to_gpu(batch,m.device);m.optimizer.zero_grad(set_to_none=True)
   out=m(feed,True);loss=m.loss(out)
   if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
   assert not out['c1_interest_vectors'].requires_grad and not out['candidate_vectors'].requires_grad
   loss.backward();grad=m.transfer.output.weight.grad
   assert grad is not None and torch.isfinite(grad).all()
   m.optimizer.step();self.updates+=1
   if first is None:
    first=batch_diagnostics(m,out,float(grad.norm()))
    assert all(p.grad is None for n,p in m.named_parameters() if not n.startswith('transfer.'))
   last=batch_diagnostics(m,out,float(grad.norm()));losses.append(float(loss.detach()))
   relations.append(float(out['loss_relation']))
  assert last['projection_norm']>0 and last['residual_ratio_mean']>0
  assert common_hash(m)==self.baseline_hash,'Frozen C1 parameters changed'
  self.last_update=dict(epoch=epoch,optimizer_steps=self.updates,first_batch=first,last_batch=last,mean_relation_loss=float(np.mean(relations)))
  with (self.path/'transfer_updates.jsonl').open('a') as f:f.write(json.dumps(self.last_update)+'\n')
  return float(np.mean(losses))
def run(variant):
 verify();assert read(OUT/'DIAGNOSTICS.json')['passed']
 p=OUT/'runs'/variant;assert p.is_dir() and not (p/'train.log').exists()
 logging.basicConfig(filename=p/'train.log',filemode='x',level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',force=True)
 logging.getLogger().addHandler(logging.StreamHandler(sys.stdout));start=time.perf_counter()
 m,a,c,spec=model(variant,'cuda:0');a.model_path=str(p/'best.pt');m.model_path=a.model_path
 baseline=common_hash(m);data=datasets(m,a,c);r=Runner(a,p);r.baseline_hash=baseline
 cfg=dict(dataset='ml-1m',K=3,seed=42,variant=variant,mode='all C1 frozen, deterministic eval features; only transfer.output.weight trainable',
  args=vars(a),head=read(OUT/'PLAN.json')['head'],branch_lr=.001,branch_weight_decay=a.l2,parameters=4096,
  initialization=spec,bank_sha256=sha(spec['bank']),frozen_parameter_sha256=baseline,source_sha256=read(OUT/'SOURCE_LOCK.json'),
  lambda_pop=.30,raw_prior_lambda=m.sil_raw_pop_lambda,selection='locked-prior Dev NDCG@5; epoch0 eligible',
  optimizer='fresh Adam only4096 transfer parameters; constant branch lr .001; original relation loss is a frozen constant')
 write(p/'config.json',cfg);m.cur_epoch=0;anchor=r.evaluate(data['dev'],[5,10,20],['HR','NDCG'])
 expected=read(Path(spec['config']).parent/'best_metadata.json')
 for scheme in ['raw','locked_prior']:
  assert all(abs(r.last[scheme][k]-expected[scheme][k])<1e-10 for k in KEYS)
 write(p/'EPOCH0.json',r.last)
 criteria=[anchor['NDCG@5']];best=copy.deepcopy(r.last);post=None;epoch=0;times=[]
 torch.save(m.state_dict(),p/'best.pt');write(p/'best_metadata.json',best)
 write(p/'READY.json',dict(pid=os.getpid(),GPU=os.environ['CUDA_VISIBLE_DEVICES'],variant=variant,parameters=4096,frozen_C1_hash=baseline,epoch0=anchor,cache_bytes=m.semantic_index.bytes()))
 logging.info('TRAIN_START frozen C1 %s branch_lr=.001, epoch0 N5 %.9f',variant,criteria[0])
 for epoch in range(1,a.epoch+1):
  tick=time.perf_counter();loss=r.fit(data['train'],epoch);training=time.perf_counter()-tick
  metric=r.evaluate(data['dev'],[5,10,20],['HR','NDCG']);criteria.append(metric['NDCG@5'])
  if post is None or metric['NDCG@5']>=post['metrics']['NDCG@5']:
   post=copy.deepcopy(r.last);post['transfer_update']=copy.deepcopy(r.last_update)
   torch.save(m.state_dict(),p/'best_nonzero_epoch.pt');write(p/'best_nonzero_metadata.json',post,False)
  improved=metric['NDCG@5']>=best['metrics']['NDCG@5']
  if improved:
   best=copy.deepcopy(r.last);torch.save(m.state_dict(),p/'best.pt');write(p/'best_metadata.json',best,False)
  times.append(dict(epoch=epoch,train_seconds=training,total_seconds=time.perf_counter()-tick))
  logging.info('Epoch %d loss=%.6f train=%.2fs Dev=%s projection_norm=%.6f residual_ratio=%.6f %s',epoch,loss,training,metric,r.last_update['last_batch']['projection_norm'],r.last_update['last_batch']['residual_ratio_mean'],'*' if improved else '')
  if a.early_stop>0 and r.eval_termination(criteria):logging.info('Early stop at %d',epoch);break
 m.actions_after_train();assert common_hash(m)==baseline
 torch.save(transfer_state(m),p/'last_transfer.pt');write(p/'LAST_TRANSFER_UPDATE.json',r.last_update)
 seconds=time.perf_counter()-start
 m.load_state_dict(torch.load(p/'best.pt',map_location=m.device,weights_only=True),strict=True);checkpoint=sha(p/'best.pt')
 tr=dict(train_complete=True,Dev_best_selected=True,Test_accepted=False,best=best,best_epoch=best['epoch'],best_nonzero_epoch=post,
  epochs=epoch,training_seconds=seconds,epoch_times=times,checkpoint_sha256=checkpoint,config=cfg,frozen_C1_unchanged=common_hash(m)==baseline,
  optimizer_steps=r.updates,final_update=r.last_update)
 write(p/'train_result.json',tr)
 del data;gc.collect();heldout(c,a,'test');td=m.Dataset(m,c,'test');td.prepare()
 write(p/'TEST_STARTED.json',dict(checkpoint_sha256=checkpoint,best_epoch=best['epoch'],passes=1,lambda_pop=.30))
 tick=time.perf_counter();got=score_views(td,p/'test_user_ranks.csv');test_seconds=time.perf_counter()-tick
 assert checkpoint==sha(p/'best.pt');protocol_lock('test',got)
 with (p/'test_metrics.csv').open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['scheme',*KEYS]);writer.writeheader()
  for s,v in got['metrics'].items():writer.writerow(dict(scheme=s,**v))
 result=dict(tr);result.update(Test=got['metrics'],Test_passes=1,Test_seconds=test_seconds,Test_accepted=True,Test_input_hash=got['input_hash'],exploratory=True)
 verify();write(p/'result.json',result);write(p/'TEST_COMPLETE.json',dict(one_Test_pass=True,result_sha256=sha(p/'result.json'),checkpoint_sha256=checkpoint))
 logging.info('COMPLETE fixed best once Test; no automatic next-round search')
def main():
 q=argparse.ArgumentParser();q.add_argument('--variant',choices=['G1','G2','G3'],required=True);x=q.parse_args()
 try:run(x.variant)
 except BaseException as e:
  write(OUT/'runs'/x.variant/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),no_retry=True));raise
if __name__=='__main__':main()
