"""One-shot eight-arm report, called on job exit; no training, prediction or polling."""
import pathlib,json,csv,os,datetime
R=pathlib.Path('/home/yyx/hzgcode/pom2.0');W=R/'research/ml1m_embedding_v2_20261020';C=W/'runs/cross_v3_20261021/p920';U=W/'runs/a1_history_bpr_20261021_night';OUT=W/'runs/recovery_v3_hsdir_20261022'
KEYS=['HR@5','NDCG@5','HR@10','NDCG@10','HR@20','NDCG@20']
def read(p):return json.loads(p.read_text()) if p.exists() else {}
def main():
 OUT.mkdir(exist_ok=True);records=[]
 specs=[('A1_ASPCF_Softmax',C/'aspcf_A1_k4_seed42_20261021_cross'),('V3_ASPCF_Softmax',C/'aspcf_V3_k4_seed42_20261022_recovery')]+[(k,U/k) for k in ['E1','E2','E3','E4','E5','E6']]
 for name,d in specs:
  j=read(d/'job.json');c=j.get('config',read(d/'config.json'));z=read(d/'COMPLETE.json');tr=read(d/'TRAIN_COMPLETE.json');fail=read(d/'FAILED.json')
  status='Train_Dev_Test_complete' if z.get('Test_accepted') else 'failed' if fail else 'Train_Dev_complete_Test_pending' if tr else 'running' if (d/'TRAIN_PID.json').exists() or (d/'child.json').exists() else 'queued'
  r=dict(experiment=name,model=c.get('model_name','LLMMIRecASPCFSampledSoftmax'),mechanism=j.get('mechanism','ASPCF'),Loss=j.get('loss','filtered Batch Softmax+relation'),K=c.get('K',4),lr=c.get('lr',.001),batch=c.get('batch_size',1024),embedding_sha256=z.get('embedding_sha256',j.get('embedding_sha256',read(d/'ASSET_QA.json').get('sha256',''))),teacher_sha256=j.get('teacher_sha256',''),best_epoch=z.get('best_epoch',tr.get('best_epoch','')),end_epoch=z.get('end_epoch',tr.get('end_epoch','')),training_seconds=z.get('training_seconds',tr.get('training_seconds','')),status=status,log_path=str(d/'train.log'),checkpoint_path=str(d/'best.pt'),checkpoint_sha256=z.get('checkpoint_sha256',tr.get('checkpoint_sha256','')))
  for phase in ['Dev','Test']:
   vals=z.get(phase,{})
   if not vals:vals=read(d/(phase.upper()+'_COMPLETE.json')).get('metrics',{})
   if phase=='Dev' and not vals:vals=read(d/'BEST_DEV_FULL.json').get('Dev',{})
   for k in KEYS:r[phase+'_'+k]=vals.get(k,'')
  records.append(r)
 stamp=datetime.datetime.now().astimezone().isoformat();fields=list(records[0])
 p=OUT/'EIGHT_RESULTS.csv';t=p.with_suffix('.'+str(os.getpid())+'.tmp')
 with t.open('w') as f:q=csv.DictWriter(f,fieldnames=fields);q.writeheader();q.writerows(records)
 os.replace(t,p)
 text=['# Eight exploratory recommendation experiments','',stamp,'','All Test results use best-Dev once. No new evaluation is performed by this reporter. Different losses are performance references, not pure structural attribution.','','|experiment|status|loss|best/end|seconds|Dev N5|Test N5|','|---|---|---|---|---:|---:|---:|']
 for r in records:text.append('|'+r['experiment']+'|'+r['status']+'|'+r['Loss']+'|'+str(r['best_epoch'])+'/'+str(r['end_epoch'])+'|'+str(r['training_seconds'])+'|'+str(r['Dev_NDCG@5'])+'|'+str(r['Test_NDCG@5'])+'|')
 for phase in ['Dev','Test']:
  text+=['','## '+phase,'','|experiment|'+'|'.join(KEYS)+'|','|---|'+'|'.join(['---:']*6)+'|']
  for r in records:text.append('|'+r['experiment']+'|'+'|'.join(str(r[phase+'_'+k]) for k in KEYS)+'|')
 lookup={r['experiment']:r for r in records}
 text+=['','## Predeclared comparisons']
 for a,b in [('V3_ASPCF_Softmax','A1_ASPCF_Softmax'),('E2','E1'),('E3','E2'),('E4','E2'),('E6','E5')]:
  x,y=lookup[a],lookup[b];s=[]
  for phase in ['Dev','Test']:
   v=x[phase+'_NDCG@5'];w=y[phase+'_NDCG@5']
   s.append(phase+' '+(format(v-w,'+.12g') if isinstance(v,(int,float)) and isinstance(w,(int,float)) else 'pending'))
  text.append('- '+a+' minus '+b+': '+', '.join(s))
 text+=['','Test-observed adaptations are exploratory; no seed variation or paired uncertainty claim. CAISD/TASID comparisons retain their matching configurations; HSDIR uses lr .002/b2048 and must compare E6 against E5.']
 p=OUT/'EIGHT_RESULTS.md';t=p.with_suffix('.'+str(os.getpid())+'.tmp');t.write_text('\n'.join(text));os.replace(t,p)
 p=OUT/'EIGHT_RESULTS.json';t=p.with_suffix('.'+str(os.getpid())+'.tmp');t.write_text(json.dumps(dict(time=stamp,results=records),indent=2));os.replace(t,p)
 print(OUT/'EIGHT_RESULTS.md')
if __name__=='__main__':main()
