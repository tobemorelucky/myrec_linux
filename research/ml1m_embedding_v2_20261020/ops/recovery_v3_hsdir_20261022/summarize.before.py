"""Manual, read-only six-variant collection; writes only new timestamped summaries."""
import pathlib,json,csv,datetime
R=pathlib.Path('/home/yyx/hzgcode/pom2.0');W=R/'research/ml1m_embedding_v2_20261020';O=W/'runs/cross_v3_20261021/p920'
meta={
 'V1_Rebuilt':('old Base-trained LoRA','Base','raw SCFT / initial tau3, final tau missing',1000,8),
 'V2_Rich':('V2 D400','Rich512','L2 coupled tau; final3.041642',400,4),
 'V2_Scaled':('V2 D400','Rich512 x global .100491','same V2 SCFT, post-PCA global scale',400,4),
 'A1':('old Base-trained LoRA','Rich512, OOD risk','old raw SCFT',1000,8),
 'B0':('V2 D400','Base original','V2 L2 coupled tau',400,4),
 'V3':('new from base, conditional only','Rich512','L2 soft-target Ts=.2 Tt=.2, fixed and target differentiable',800,4)}
records=json.loads((W/'runs/p920_aspcf_ss_v2_scaled_k4_seed42_20261021_164007/COMPARISON.json').read_text())
for v in ['A1','B0','V3']:
 run=O/('aspcf_'+v+'_k4_seed42_20261021_cross');c=run/'COMPLETE.json';failed=run/'FAILED.json'
 if c.exists():
  r=json.loads(c.read_text());r.update(version=v,status='Test accepted',run_dir=str(run));records.append(r)
 else:records.append(dict(version=v,status='failed' if failed.exists() else 'running' if (run/'child.json').exists() else 'not_started_or_not_authorized',run_dir=str(run)))
for r in records:
 name=r['version'];lo,prompt,loss,steps,acc=meta[name];r.update(lora_origin=lo,prompt=prompt,scft_loss=loss,scft_steps=steps,scft_accumulation=acc,nominal_movie_exposures=steps*acc*16,exposure_note='nominal full microbatches; true tail batches slightly fewer',recommendation_loss='same ASPCF filtered Softmax+relation .01',K=4,seed=42,exploratory=True)
 if name.startswith('V'):r.setdefault('status','Test accepted' if 'Test' in r else 'not_completed')
 if name in ['A1','B0','V3']:
  aq=pathlib.Path(r['run_dir'])/'ASSET_QA.json'
  if aq.exists():r['asset_qa']=json.loads(aq.read_text())
summary=O/'summaries';summary.mkdir(exist_ok=True);tag=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
with (summary/(tag+'.json')).open('x') as f:json.dump(records,f,indent=2)
fields=['version','status','lora_origin','prompt','scft_loss','scft_steps','scft_accumulation','nominal_movie_exposures','phase','HR@5','NDCG@5','HR@10','NDCG@10','HR@20','NDCG@20','best_epoch','end_epoch','training_seconds','embedding_sha256','checkpoint_sha256']
with (summary/(tag+'.csv')).open('x') as f:
 cw=csv.DictWriter(f,fieldnames=fields);cw.writeheader()
 for r in records:
  for phase in ['Dev','Test']:
   row={k:r.get(k) for k in fields};row['phase']=phase;row.update(r.get(phase,{}));cw.writerow(row)
lines=['# ML-1M six-variant exploratory acceptance snapshot','', '|version|state|best/end|Dev N5|Test N5|training seconds|','|---|---|---|---:|---:|---:|']
for r in records:lines.append('|'+r['version']+'|'+r['status']+'|'+str(r.get('best_epoch','pending'))+'/'+str(r.get('end_epoch','pending'))+'|'+str(r.get('Dev',{}).get('NDCG@5','pending'))+'|'+str(r.get('Test',{}).get('NDCG@5','pending'))+'|'+str(r.get('training_seconds','pending'))+'|')
lines+=['','CSV/JSON contain all six metrics per phase and source/LoRA identities. Pending/failed rows are not Test results.','All Test comparisons exploratory; no Test-driven parameter/epoch/seed selection. Old vs V2 LoRA varies several training definitions, not just loss; V3 also increases max steps, so no pure loss causality claim.']
(summary/(tag+'.md')).write_text('\n'.join(lines));print(str(summary/(tag+'.md')))
