"""CPU asset verification, original teacher builders with isolated path-only AST redirection."""
import os,pathlib,json,pickle,hashlib,ast,sys,re,datetime,subprocess
R=pathlib.Path('/home/yyx/hzgcode/pom2.0');W=R/'research/ml1m_embedding_v2_20261020'
O=W/'ops/a1_history_bpr_20261021_night';U=W/'runs/a1_history_bpr_20261021_night'
A=W/'runs/cross_v3_20261021/p920/incoming/A1_oldlora_rich512/rich_pca.pkl'
H='03cf9649844da2c172487a0f552a49a4d331d8262135e3b70e48d6e772ad7905'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 if p.exists():
  assert json.loads(p.read_text())==x,'Refuse changed existing CPU artifact';return
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def redirected(file,paths,argv):
 tree=ast.parse(file.read_text());changed=[]
 class T(ast.NodeTransformer):
  def visit_Assign(self,node):
   if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in paths:
    n=node.targets[0].id;changed.append(n);node.value=ast.Constant(str(paths[n]))
   return self.generic_visit(node)
 tree=T().visit(tree);ast.fix_missing_locations(tree);assert sorted(changed)==sorted(paths),changed
 old=sys.argv;sys.argv=[str(file)]+argv
 try:exec(compile(tree,str(file),'exec'),dict(__name__='__main__',__file__=str(file)))
 finally:sys.argv=old
 return dict(source=str(file),sha256=sha(file),path_only_assignments=paths)
def main():
 import numpy as np
 assert U.resolve()==U and not (U/'PREPARED.json').exists();U.mkdir(exist_ok=True);assets=U/'teachers';assets.mkdir(exist_ok=True)
 assert A.resolve()==A and sha(A)==H
 with A.open('rb') as f:table=pickle.load(f)
 m=json.loads((A.parent/'READY.json').read_text());assert m['variant']=='A1' and m['embedding_sha256']==H
 assert sha(R/'data/ml-1m/item2id.json')==m['producer_manifest']['mapping_sha256']
 assert table.shape==(3707,1536) and table.dtype==np.float32 and np.isfinite(table).all() and not table[0].any() and (np.linalg.norm(table[1:],axis=1)>0).all()
 save(U/'ASSET_QA.json',dict(path=str(A),sha256=H,mapping_sha256=m['producer_manifest']['mapping_sha256'],shape=list(table.shape),dtype=str(table.dtype),padding_zero=True,valid_items=3706))
 fine=assets/'a1_proto32_sr512.pkl';hier=assets/'a1_hier_proto32_8_sr512.pkl'
 f=redirected(R/'tools/build_llmmi_semantic_prototypes.py',dict(llm_path=A,out_path=fine),['--dataset','ml-1m','--semantic_rank','512','--prototype_num','32','--temperature','.1','--seed','42'])
 h=redirected(R/'tools/build_llmmi_hierarchical_teacher.py',dict(proto_path=fine,out_path=hier),['--dataset','ml-1m','--semantic_rank','512','--fine_num','32','--coarse_num','8','--seed','42'])
 for p in [fine,hier]:
  with p.open('rb') as x:d=pickle.load(x)
  vals=d.get('soft_assignments',d.get('fine_assignments'));assert vals.shape==(3707,32) and np.isfinite(vals).all() and not vals[0].any()
  np.testing.assert_allclose(vals[1:].sum(1),1,atol=1e-5)
  save(p.with_suffix('.json'),dict(embedding_path=str(A),embedding_sha256=H,teacher_sha256=sha(p),mapping_sha256=m['producer_manifest']['mapping_sha256'],source_scope='item semantic metadata only, all canonical catalog items; no Dev/Test or Train interaction labels used',builder_fine={**f,'path_only_assignments':{k:str(v) for k,v in f['path_only_assignments'].items()}},builder_hier={**h,'path_only_assignments':{k:str(v) for k,v in h['path_only_assignments'].items()}}))
 sources={n:sha(R/n) for n in ['main.py','helpers/BaseRunner.py','helpers/SeqReader.py','helpers/BaseReader.py','models/BaseModel.py','models/sequential/LLMMIRecASPCF.py','models/sequential/LLMMIRecCAISD.py','models/sequential/LLMMIRecHSDIR.py','models/sequential/llmmi_components.py','models/sequential/llmmi_utils.py']}
 ref=W/'runs/p920_aspcf_ss_v1_rebuilt_k4_seed42_20261020_214225/config.json';base=json.loads(ref.read_text());base.pop('sampled_softmax_temperature')
 base.update(model_name='LLMMIRecASPCF',gpu='1',llm_emb_path=str(A),metric='NDCG,HR')
 caisd=dict(base,model_name='LLMMIRecCAISD',semantic_teacher_path=str(fine),semantic_distill_mode='uniform',semantic_teacher_mode='responsibility',semantic_responsibility_alpha=.5,semantic_relation_mode='none',lambda_interest_semantic=.005,tasid_mode='none',tasid_student_mode='llm_only',lambda_tasid=.01,tasid_temp=.1)
 hs=dict(base,model_name='LLMMIRecHSDIR',gpu='0',lr=.002,batch_size=2048,lambda_hsr=0.,hsr_teacher_mode='hierarchical',hsr_student_temp=1.,hsr_loss_mode='absolute',hsr_margin=.1,hsr_pair_margin=.1,hsr_confidence_mode='semantic',hsr_route_source='raw',teacher_path=str(hier),aggregation_mode='base',support_beta=1.)
 specs=[('E1',base,'ASPCF baseline','BPR+relation',None),('E3',dict(caisd,tasid_mode='target',tasid_student_mode='asymmetric'),'CAISD uniform + TASID asymmetric','BPR+relation+profile+TASID',fine),('E4',dict(caisd,tasid_mode='target',tasid_student_mode='llm_only'),'CAISD uniform + TASID llm_only','BPR+relation+profile+TASID',fine),('E2',caisd,'CAISD uniform responsibility','BPR+relation+profile',fine),('E5',hs,'HSDIR family control lambda_hsr0','BPR+relation',hier),('E6',dict(hs,lambda_hsr=.01),'HSDIR hierarchical absolute lambda_hsr.01','BPR+relation+HSR',hier)]
 for label,c,mech,loss,t in specs:
  d=U/label;d.mkdir();c=dict(c,log_file=str(d/'train.log'),model_path=str(d/'best.pt'))
  job=dict(config=c,mechanism=mech,loss=loss,embedding_sha256=H,source_sha256=sources,from_scratch=True,exploratory=True,max_train_seconds=14400)
  if t:job.update(teacher=str(t),teacher_sha256=sha(t))
  save(d/'job.json',job)
 assert {k:(caisd[k],base.get(k)) for k in caisd if k in base and caisd[k]!=base[k]}=={'model_name':('LLMMIRecCAISD','LLMMIRecASPCF')}
 hist=[]
 paths=[
 'new_log/llmmirec_caisd_phase2/ml-1m/caisd_seed42.nohup.out','new_log/llmmirec_caisd_phase2/ml-1m/tasid_asym_l001_t01_seed42.nohup.out','new_log/llmmirec_caisd_phase2/ml-1m/tasid_llmonly_l001_t01_seed42.nohup.out',
 'new_log/llmmirec_caisd_phase2/beauty/caisd_seed42.nohup.out','new_log/llmmirec_caisd_phase2/beauty/tasid_asym_l001_t01_seed42.nohup.out','new_log/llmmirec_caisd_phase2/beauty/tasid_llmonly_l001_t01_seed42.nohup.out',
 'new_log/llmmirec_hsdir_phase1/ml-1m/hsdir_hierarchical_lh0/LLMMIRecHSDIR_hsdir_hierarchical_lh0_seed42.log','new_log/llmmirec_hsdir_phase1/ml-1m/hsdir_hierarchical_lh0.01/LLMMIRecHSDIR_hsdir_hierarchical_lh0.01_seed42.log',
 'new_log/llmmirec_hsdir_phase1/beauty/hsdir_hierarchical_lh0.01/LLMMIRecHSDIR_hsdir_hierarchical_lh0.01_seed42.log']
 for n in paths:
  p=R/n
  if not p.exists():hist.append(dict(path=n,missing=True));continue
  lines=p.read_text(errors='replace').splitlines();hist.append(dict(path=n,sha256=sha(p),configuration_header=lines[:130],final_lines=[s for s in lines if 'Best Iter(dev)' in s or 'Test After Training:' in s][-2:]))
 save(U/'HISTORICAL_EVIDENCE.json',hist)
 save(U/'PREPARED.json',dict(time=datetime.datetime.now().astimezone().isoformat(),six_jobs=True,E7='not scheduled: historical confidence uses different profile/teacher settings; avoid adding unmatched arm',gpu0='reserved for V3 receiver, then E5,E6',gpu1='E1,E3,E4,E2',git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),sources=sources,teacher_sha256={fine.name:sha(fine),hier.name:sha(hier)},reference_config=str(ref)))
 print(json.dumps(dict(prepared=str(U),teacher_sha256={fine.name:sha(fine),hier.name:sha(hier)})))
if __name__=='__main__':main()
