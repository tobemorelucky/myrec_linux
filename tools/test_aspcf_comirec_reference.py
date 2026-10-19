"""Bounded reference tests + optional Train-only GPU steps; no real Dev/Test or checkpoints."""
import argparse,os,sys,copy,io,json,time,signal,types,hashlib,unittest
from pathlib import Path
boot=argparse.ArgumentParser(add_help=False);boot.add_argument('--gpu');bootargs,_=boot.parse_known_args();os.environ['CUDA_VISIBLE_DEVICES']='' if bootargs.gpu is None else bootargs.gpu
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF as Base
from models.sequential.LLMMIRecASPCFComiRecReference import LLMMIRecASPCFComiRecReference as Ref,ComiRecDR,squash,independent_max
from models.BaseModel import GeneralModel
RESULT={}

def args(K=2,extractor='query',scoring='max',ds='beauty',device='cpu'):
 a=Ref.parse_model_args(argparse.ArgumentParser()).parse_args([]);a.device=torch.device(device);a.buffer=0;a.model_path='unused';a.K=K;a.history_max=20;a.random_seed=42;a.reference_extractor=extractor;a.reference_scoring=scoring;a.llm_emb_path=str(ROOT/'data'/ds/'handled/llm_table_pca1536.pkl');return a

def corpus():
 return types.SimpleNamespace(n_users=4,n_items=12102,user_his={1:[(1,0),(2,1),(3,2),(4,3)],2:[(5,0),(6,1)]},data_df={'train':pd.DataFrame(dict(user_id=[1,1,1,2],item_id=[1,2,3,5],time=[0,1,2,0],position=[0,1,2,0],neg_items=[[100]]*4))},train_clicked_set={1:{1,2,3},2:{5}})
def make(K=2,extractor='query',scoring='max',base=False):
 torch.manual_seed(42);return (Base if base else Ref)(args(K,extractor,scoring),corpus())
def batch():
 return dict(history_items=torch.tensor([[1,2,3],[4,0,0],[0,0,0],[6,7,0]]),lengths=torch.tensor([3,1,0,2]),item_id=torch.tensor([[8,100],[9,101],[10,102],[11,103]]),user_id=torch.tensor([1,1,2,2]))
def official_reference(h,w,lengths,K,rounds):
 B,L,D=h.shape;flat=w.reshape(w.shape[0],K*D,D)
 votes=(flat[:L][None]*h[:,:,None,:]).sum(-1).reshape(B,L,K,D).permute(0,2,1,3)
 stop=votes.detach();b=h.new_zeros(B,K,L);mask=(torch.arange(L)[None,:]<lengths[:,None]).to(h.dtype)
 for i in range(rounds):
  c=b.softmax(1)*mask[:,None,:];used=votes if i==rounds-1 else stop
  s=torch.matmul(c[:,:,None,:],used).squeeze(2);norm=s.square().sum(-1,keepdim=True);v=s*norm/(1+norm)/torch.sqrt(norm+1e-9)
  if i<rounds-1:b=b+(stop*v[:,:,None,:]).sum(-1)
 return v,c

class Checks(unittest.TestCase):
 def test_01_shared_initialization_rng_and_parameters(self):
  for K in [1,2,4]:
   base=make(K,base=True);state=torch.get_rng_state()
   for name,ext,head in [('A1','query','max'),('D0','dr','history'),('D1','dr','max')]:
    m=make(K,ext,head);self.assertTrue(torch.equal(state,torch.get_rng_state()))
    for k,v in base.state_dict().items():
     if k.startswith('item_encoder.') or k.startswith('position_emb.') or (ext=='query' and k.startswith('extractor.')) or (head=='history' and k.startswith('aggregator.')):self.assertTrue(torch.equal(v,m.state_dict()[k]),k)
    if head=='max':self.assertFalse(hasattr(m,'aggregator'));self.assertFalse(any(n.startswith('aggregator.') for n,_ in m.named_parameters()))
    RESULT.setdefault('parameters',{}).setdefault(str(K),{})[name]=m.count_variables()
   RESULT['parameters'][str(K)]['A0']=base.count_variables()
 def test_02_official_dr_forward_and_gradient(self):
  torch.manual_seed(19);h=torch.randn(3,5,4,requires_grad=True);lengths=torch.tensor([5,2,0]);dr=ComiRecDR(2,4,5,3,42)
  v,a=dr(h,lengths);hh=h.detach().clone().requires_grad_();ww=dr.vote_weight.detach().clone().requires_grad_();vref,aref=official_reference(hh,ww,lengths,2,3)
  self.assertTrue(torch.allclose(v,vref,atol=2e-7,rtol=1e-6));self.assertTrue(torch.allclose(a,aref,atol=2e-7,rtol=1e-6))
  gh,gw=torch.autograd.grad(v.sum(),[h,dr.vote_weight]);rh,rw=torch.autograd.grad(vref.sum(),[hh,ww]);self.assertTrue(torch.allclose(gh,rh,atol=2e-7,rtol=2e-6));self.assertTrue(torch.allclose(gw,rw,atol=2e-7,rtol=2e-6));self.assertGreater(float(gh.norm()),0)
  RESULT['official_equivalence']=dict(v_max_error=float((v-vref).abs().max()),route_max_error=float((a-aref).abs().max()),history_gradient_norm=float(gh.norm()),weight_gradient_norm=float(gw.norm()))
 def test_03_route_axis_agreement_and_boundaries(self):
  for K in [1,2,4]:
   dr=ComiRecDR(K,4,20,3,42);h=torch.randn(4,3,4,requires_grad=True);lengths=torch.tensor([0,1,2,3]);v,a=dr(h,lengths)
   valid=torch.arange(3)[None,:]<lengths[:,None];self.assertTrue(torch.allclose(a.sum(1),valid.float(),atol=1e-6));self.assertTrue((v.norm(dim=-1)<1).all());self.assertTrue((v[0]==0).all())
   changed=h.detach().clone();changed[~valid]=1000;vv,_=dr(changed,lengths);self.assertTrue(torch.allclose(v,vv))
   v.sum().backward();self.assertTrue(torch.isfinite(h.grad).all());self.assertTrue(torch.isfinite(dr.vote_weight.grad).all())
   empty,_=dr(h[:,:0],torch.zeros(4,dtype=torch.long));self.assertEqual(tuple(empty.shape),(4,K,4));self.assertTrue((empty==0).all())
   if K==1:
    votes=torch.einsum('bld,lkod->bklo',h,dr.vote_weight[:3]);expected=squash((votes*valid[:,None,:,None]).sum(2));self.assertTrue(torch.allclose(v,expected))
   else:self.assertGreater(float((a[3]-1/K).abs().max()),1e-5)
 def test_04_query_path_matches_original(self):
  b=batch();base=make(base=True);m=make();base.eval();m.eval()
  x=base(b,True);y=m(b,True)
  for key in ['interest_vectors','attention_maps','history_vectors','candidate_vectors']:self.assertTrue(torch.equal(x[key],y[key]),key)
  self.assertTrue(torch.equal(y['prediction'],torch.bmm(y['interest_vectors'],y['candidate_vectors'].transpose(1,2)).max(1).values))
 def test_05_independent_positive_negative_max(self):
  v=torch.tensor([[[1.,0.],[0.,1.]]],requires_grad=True);e=torch.tensor([[[2.,0.],[0.,3.]]],requires_grad=True);scores,z,heads=independent_max(v,e)
  self.assertEqual(heads.tolist(),[[0,1]]);loss=GeneralModel.loss(None,{'prediction':scores});self.assertTrue(torch.allclose(loss,F.softplus(torch.tensor(1.))))
  loss.backward();self.assertTrue((v.grad.norm(dim=-1)>0).all());RESULT['independent_candidate_heads']=heads.tolist()
 def test_06_relation_and_original_bpr(self):
  b=batch()
  for ext in ['query','dr']:
   m=make(extractor=ext);base=make(base=True);m.train();base.train();torch.manual_seed(83);x=base(b);torch.manual_seed(83);y=m(b)
   self.assertTrue(torch.equal(x['_relation_ids'],y['_relation_ids']));rank=GeneralModel.loss(m,y);analytic=F.softplus(-(y['prediction'][:,0]-y['prediction'][:,1])).mean();self.assertTrue(torch.allclose(rank,analytic,atol=1e-7,rtol=1e-6))
   rel=m._compute_relation_loss(y['_relation_ids']);total=m.loss(y);self.assertTrue(torch.allclose(total,rank+.01*rel,atol=1e-7));total.backward();self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
 def test_07_history_candidate_isolation_and_1001_slots(self):
  b=batch();b['item_id']=torch.arange(1,1002)[None].repeat(4,1);b['item_id'][:,50]=b['item_id'][:,51]
  for ext in ['query','dr']:
   m=make(extractor=ext);m.eval()
   with torch.no_grad():
    x=m(b,True);bb={k:v.clone() for k,v in b.items()};bb['item_id'][:,0]=99;y=m(bb,True)
    self.assertTrue(torch.equal(x['interest_vectors'],y['interest_vectors']));self.assertTrue(torch.equal(x['prediction'][:,1:],y['prediction'][:,1:]));self.assertTrue(torch.equal(x['prediction'][:,50],x['prediction'][:,51]));self.assertEqual(tuple(x['prediction'].shape),(4,1001))
    single=m({k:v[:1] for k,v in b.items()})['prediction'];self.assertTrue(torch.allclose(single,x['prediction'][:1],atol=2e-7,rtol=2e-6))
 def test_08_dataset_reuse_and_empty_history(self):
  for ext in ['query','dr']:
   m=make(extractor=ext);d=m.Dataset(m,corpus(),'train');ref=Base.Dataset(m,corpus(),'train');self.assertEqual(len(d),len(ref))
   for i in range(len(d)):
    for k in ref[i]:self.assertTrue(np.array_equal(d[i][k],ref[i][k]))
   for K in [1,2,4]:
    for scoring in (['history','max'] if ext=='dr' else ['max']):
     mm=make(K,ext,scoring);b=batch();b['history_items']=b['history_items'][:,:0];b['lengths']=torch.zeros(4,dtype=torch.long);o=mm(b,True);self.assertTrue((o['prediction']==0).all());self.assertFalse(o['interest_mask'].any());self.assertEqual(tuple(o['interest_vectors'].shape),(4,K,64))
 def test_09_reproducibility_serialization_and_no_idle_aggregator(self):
  b=batch()
  for ext,sc in [('query','max'),('dr','max'),('dr','history')]:
   m=make(extractor=ext,scoring=sc);n=make(extractor=ext,scoring=sc);self.assertTrue(all(torch.equal(v,n.state_dict()[k]) for k,v in m.state_dict().items()));f=io.BytesIO();torch.save(m.state_dict(),f);f.seek(0);n.load_state_dict(torch.load(f,weights_only=True));m.eval();n.eval();self.assertTrue(torch.equal(m(b)['prediction'],n(b)['prediction']))
   opt=torch.optim.Adam(m.customize_parameters());ids={id(p) for group in opt.param_groups for p in group['params']};self.assertEqual(ids,{id(p) for p in m.parameters() if p.requires_grad})
   if sc=='max':self.assertFalse(any(n.startswith('aggregator.') for n,_ in m.named_parameters()))
 def test_10_short_history_repeated_items_task_gradients(self):
  for K in [1,2,4]:
   m=make(K,'dr','max');b=batch();b['lengths']=torch.tensor([2,1,3,3]);b['history_items']=(torch.arange(3)[None,:]<b['lengths'][:,None]).long();m.eval();o=m(b,True);rank=GeneralModel.loss(m,o)
   gradients=torch.autograd.grad(rank,[m.extractor.vote_weight,o['history_vectors'],m.item_encoder.semantic_branch[0].weight],allow_unused=False)
   self.assertTrue(all(torch.isfinite(g).all() and g.norm()>0 for g in gradients))
   RESULT.setdefault('synthetic_task_gradient',{})[str(K)]=[float(g.norm()) for g in gradients]



 def test_11_nonempty_K_variants_and_K1_reduction(self):
  b=batch()
  for K in [1,2,4]:
   values={}
   for name,ext,head in [('A1','query','max'),('D0','dr','history'),('D1','dr','max')]:
    m=make(K,ext,head);m.eval();out=m(b,True);self.assertEqual(tuple(out['interest_vectors'].shape),(4,K,64));self.assertEqual(tuple(out['prediction'].shape),(4,2));self.assertTrue(torch.isfinite(m.loss(out)));values[name]=out['prediction'].detach()
   if K==1:
    base=make(K,base=True);base.eval();self.assertTrue(torch.allclose(base(b)['prediction'],values['A1'],atol=1e-7,rtol=1e-6));self.assertTrue(torch.allclose(values['D0'],values['D1'],atol=1e-7,rtol=1e-6))


def train_corpus(ds):
 df=pd.read_csv(ROOT/'data'/ds/'train.csv',sep='\t',usecols=['user_id','item_id','time']).sort_values(['time','user_id'],kind='mergesort');df['position']=df.groupby('user_id',sort=False).cumcount()
 his={int(u):list(zip(g.item_id.astype(int),g.time)) for u,g in df.groupby('user_id',sort=False)}
 return types.SimpleNamespace(n_users=int(df.user_id.max())+1,n_items=12102 if ds=='beauty' else 3707,data_df={'train':df},user_his=his,train_clicked_set={u:{i for i,t in h} for u,h in his.items()})

def real_short(device):
 rows=[]
 for ds in ['beauty','ml-1m']:
  c=train_corpus(ds)
  for name,ext,head in [('A0','query','history'),('A1','query','max'),('D0','dr','history'),('D1','dr','max')]:
   a=args(2,ext,head,ds,str(device));torch.manual_seed(42);m=(Base if name=='A0' else Ref)(a,c).to(device);d=m.Dataset(m,c,'train');np.random.seed(42);d.actions_before_epoch();pick=np.random.default_rng(20261018).choice(len(d),1024,replace=False);b=d.collate_batch([d[int(i)] for i in pick]);b={k:v.to(device) if torch.is_tensor(v) else v for k,v in b.items()};m.train();torch.manual_seed(130);captured=[];hook=m.extractor.register_forward_hook(lambda mod,inputs,output:captured.append(output[0].detach()));o=m(b,True);hook.remove();rank=GeneralModel.loss(m,o);total=m.loss(o)
   route_parameter=m.extractor.query if ext=='query' else m.extractor.vote_weight
   params=[route_parameter,o['history_vectors'],m.item_encoder.semantic_branch[0].weight]
   gradients=torch.autograd.grad(rank,params,retain_graph=True);total.backward()
   assert torch.isfinite(total) and all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
   pre=captured[0];v=o['interest_vectors'];mask=(torch.arange(b['history_items'].shape[1],device=device)[None,:]<b['lengths'][:,None]);attention=o['attention_maps'];coupling=attention.clamp_min(1e-30);entropy=-(coupling*coupling.log()).sum(1) if ext=='dr' else -(coupling*coupling.log()).sum(-1)
   row=dict(dataset=ds,variant=name,K=2,parameters=m.count_variables(),ranking_loss=float(rank.detach()),relation_loss=float(o['loss_relation']),total_loss=float(total.detach()),route_parameter_task_gradient=float(gradients[0].norm()),route_parameter_decay_norm=float((1e-6*route_parameter).norm()),route_task_to_decay_ratio=float(gradients[0].norm()/(1e-6*route_parameter).norm().clamp_min(1e-30)),history_task_gradient=float(gradients[1].norm()),semantic_first_layer_task_gradient=float(gradients[2].norm()),pre_dropout_V_norm_mean=float(pre.norm(dim=-1).mean()),pre_dropout_V_norm_max=float(pre.norm(dim=-1).max()),candidate_norm_mean=float(o['candidate_vectors'].norm(dim=-1).mean()),score_RMS=float(o['prediction'].square().mean().sqrt()),positive_negative_margin_mean=float((o['prediction'][:,0]-o['prediction'][:,1]).mean()),V_relative_head_difference=float(((pre[:,0]-pre[:,1]).norm(dim=-1)/pre.norm(dim=-1).mean(-1).clamp_min(1e-30)).mean()),coupling_entropy_mean=float(entropy[mask].mean()) if ext=='dr' else float(entropy.mean()),train_rows=len(d))
   if ext=='dr':row['route_max_probability_mean']=float(attention.max(1).values[mask].mean())
   if head=='max':row['positive_negative_same_selected_head_fraction']=float((o['selected_interest'][:,0]==o['selected_interest'][:,1]).float().mean())
   initial_route=route_parameter.detach().clone();del o,rank,total,gradients,captured;opt=torch.optim.Adam(m.customize_parameters(),lr=.004 if ds=='beauty' else .001,weight_decay=1e-6)
   def step():
    opt.zero_grad();out=m(b);loss=m.loss(out);assert torch.isfinite(loss);loss.backward();opt.step()
   for _ in range(2):step()
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
   for _ in range(5):step()
   torch.cuda.synchronize();row['step_ms']=(time.perf_counter()-started)*200;row['peak_allocated_MiB']=torch.cuda.max_memory_allocated()/2**20;row['peak_reserved_MiB']=torch.cuda.max_memory_reserved()/2**20;row['query_or_vote_parameter_update_from_initial']=float((route_parameter.detach()-initial_route).norm())
   rows.append(row);print(json.dumps(row),flush=True);del m,opt,d,b,route_parameter;torch.cuda.empty_cache()
 return rows
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--gpu');p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(2);signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError('170s bounded test')));signal.alarm(170);started=time.time();dest=Path(a.output);assert dest.resolve().is_relative_to(ROOT) and not dest.exists()
 if a.gpu is None:
  r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks));result=dict(success=r.wasSuccessful(),tests=r.testsRun,results=RESULT)
 else:result=dict(success=True,real=real_short(torch.device('cuda:0')))
 result.update(seconds=time.time()-started,full_training=False,dev_executed=False,test_executed=False);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2));print(json.dumps(result));sys.exit(0 if result['success'] else 1)
