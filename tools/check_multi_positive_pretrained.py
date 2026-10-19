"""Bounded pretrained Train-only gate; no persistent parameter updates or evaluation."""
import os,sys,json,time,hashlib,signal,argparse
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.test_llmmirec_multi_positive import real_fixture
from models.BaseModel import GeneralModel
from models.sequential.LLMMIRecASPCFMultiPositive import exclusive_assignment
signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError("170s bound")))
signal.alarm(170);torch.set_num_threads(2)

def stats(x):
 x=x.detach().double().flatten();return dict(mean=x.mean().item(),median=x.median().item(),p10=torch.quantile(x,.1).item(),p90=torch.quantile(x,.9).item(),max=x.max().item())
def adam_update(q,g,lr,wd):
 p=torch.nn.Parameter(q.detach().clone());op=torch.optim.Adam([p],lr=lr,weight_decay=wd);p.grad=g.detach().clone();op.step();return p.detach()-q.detach()
def main():
 rows=[]
 for ds,sha in [('beauty','e123e0f9f3b5c7825ab3cd6e08982de0b5637880f2e34a4e159c43568aff641d'),('ml-1m','1285b09e5d98526a50f51a2dda3a22edddb21613a78c25392ab6c49f611b9d45')]:
  path=ROOT/'new_model/llmmirec_aspcf_K2'/ds/'seed42/LLMMIRecASPCF_seed42.pt'
  digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==sha
  m,b,c=real_fixture(ds,torch.device('cuda:0'));m.load_state_dict(torch.load(path,map_location=m.device,weights_only=True),strict=True)
  m.lambda_aux=.1;m.eval();result=dict(dataset=ds,checkpoint=str(path),sha256=digest,lambda_aux=.1,rows=1024,blocks=[])
  q=m.extractor.query;result['query_parameter_norm']=float(q.norm())
  # Use eval (no dropout) to measure matching reliability in the serving representation.
  with torch.no_grad():
   o=m(b,True);v=o['interest_vectors'];a=o['attention_maps'];pos=torch.stack([o['candidate_vectors'][:,0],m.item_encoder(b['mp_extra_item'])],1);neg=o['candidate_vectors'][:,1]
   valid=b['mp_extra_valid'].bool() & (b['mp_extra_item']!=b['item_id'][:,1]);active=valid & (b['lengths']>1)
   margins=torch.einsum('bkd,bmd->bkm',v,pos-neg[:,None]);assign,tie=exclusive_assignment(margins,b['mp_assignment_key'])
   gap32=(margins[:,0,0]+margins[:,1,1])-(margins[:,1,0]+margins[:,0,1])
   gap64=((v[:,0].double()-v[:,1].double())*(pos[:,0].double()-pos[:,1].double())).sum(-1)
   scale=torch.einsum('bkd,bmd->bkm',v,pos).abs().mean((1,2)).clamp_min(1e-12)
   # Sign describes target0->head0 for the unique optimum.
   ref=gap64>0;flips=[]
   local=torch.Generator(device=v.device).manual_seed(42017)
   for _ in range(4):
    vp=torch.nextafter(v,torch.where(torch.rand(v.shape,device=v.device,generator=local)>.5,torch.full_like(v,float('inf')),torch.full_like(v,float('-inf'))))
    pp=torch.nextafter(pos,torch.where(torch.rand(pos.shape,device=v.device,generator=local)>.5,torch.full_like(pos,float('inf')),torch.full_like(pos,float('-inf'))))
    gp=((vp[:,0].double()-vp[:,1].double())*(pp[:,0].double()-pp[:,1].double())).sum(-1)
    flips.append(float(((gp>0)!=ref)[active].float().mean()))
   result.update(active=int(valid.sum()),length_gt1=int(active.sum()),delta_abs=stats(gap64[valid].abs()),delta_relative_score=stats((gap64.abs()/scale)[valid]),delta_relative_lt_1e6=float((gap64.abs()/scale<1e-6)[valid].float().mean()),delta_relative_lt_1e4=float((gap64.abs()/scale<1e-4)[valid].float().mean()),exact_matcher_tie_fraction=float(tie[valid].float().mean()),matcher_vs_double_flip_fraction=float(((assign[:,0]==0)!=ref)[active].float().mean()),one_ulp_flip_fractions_length_gt1=flips,attention_L1=stats((a[:,0]-a[:,1]).abs().sum(-1)),interest_relative_difference=stats((v[:,0]-v[:,1]).norm(dim=-1)/v.norm(dim=-1).mean(-1).clamp_min(1e-30)),target0_head0=int(((assign[:,0]==0)&valid).sum()),target0_head1=int(((assign[:,0]==1)&valid).sum()),score_abs_mean=float(scale[valid].mean()))
  # Three fixed Train minibatches: same forward for original BPR and auxiliary.
  for block in range(3):
   bb={k:(x[block*256:(block+1)*256] if torch.is_tensor(x) else x) for k,x in b.items()};m.train();torch.manual_seed(910+block);torch.cuda.manual_seed_all(910+block)
   out=m(bb,True);mainloss=GeneralModel.loss(m,out);aux=out['_mp_aux'];rel=m._compute_relation_loss(out['_relation_ids'])
   gm=torch.autograd.grad(mainloss,q,retain_graph=True)[0];ga=torch.autograd.grad(aux,q,retain_graph=True)[0];gr=torch.autograd.grad(rel,q,retain_graph=True,allow_unused=True)[0];assert gr is None
   total=gm+.1*ga;decay=1e-6*q.detach();lr=.004 if ds=='beauty' else .001
   update=adam_update(q,total,lr,1e-6);taskonly=adam_update(q,total,lr,0);decayonly=adam_update(q,torch.zeros_like(q),lr,1e-6)
   result['blocks'].append(dict(block=block,main_loss=float(mainloss.detach()),aux_loss=float(aux.detach()),relation_loss=float(rel.detach()),weighted_aux_loss_over_main=float(.1*aux.detach()/mainloss.detach()),main_query_gradient=float(gm.norm()),aux_query_gradient=float(ga.norm()),weighted_aux_gradient_over_main=float(.1*ga.norm()/gm.norm().clamp_min(1e-30)),task_gradient_over_decay=float(total.norm()/decay.norm().clamp_min(1e-30)),query_decay_norm=float(decay.norm()),query_total_task_norm=float(total.norm()),fresh_Adam_update_norm=float(update.norm()),task_only_Adam_update_norm=float(taskonly.norm()),decay_only_Adam_update_norm=float(decayonly.norm()),actual_vs_decay_update_relative_difference=float((update-decayonly).norm()/update.norm().clamp_min(1e-30)),actual_vs_task_update_cosine=float(F.cosine_similarity(update.flatten(),taskonly.flatten(),dim=0,eps=1e-30))))
   assert all(torch.isfinite(g).all() for g in [gm,ga,total,update]);del out
  assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
  rows.append(result);print(json.dumps(result),flush=True)
  del m,b,c,q;torch.cuda.empty_cache()
 return rows
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();dest=Path(args.output);assert dest.resolve().is_relative_to(ROOT) and not dest.exists();started=time.time();rows=main();dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(dict(datasets=rows,seconds=time.time()-started,test_executed=False,dev_executed=False),indent=2))
