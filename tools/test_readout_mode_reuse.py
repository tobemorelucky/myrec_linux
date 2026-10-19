"""Necessary reuse/mode tests on Train only; no epoch, Dev/Test or checkpoint load."""
import os,sys,time,pickle,json,copy,importlib.util,signal,subprocess
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from tools import run_frozen_interest_readout as F
from tools.run_refine_bpr_round1 import fixture,rng,restore,rh
from models.sequential.LLMMIRecASPCFReadoutBPR import LLMMIRecASPCFReadoutBPR as Joint
from helpers.BaseRunner import BaseRunner
from models.BaseModel import GeneralModel
from utils import utils
ARCHIVE=ROOT/'new_log/readout_reuse_preparation/20261019_modes/source_archive/run_frozen_interest_readout_v1.py'

def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);opt=p.parse_args()
 output=(ROOT/opt.output).resolve();assert output.is_relative_to(ROOT) and not output.exists()
 assert F.sha(ARCHIVE)=='a69d18a29d8e1c8c50fcb2fe3b6f62721e3caa0f3bcd854439dc79f2bc6b64b5'
 gpu=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).split(',')
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 assert gpu[0].strip() not in apps and float(gpu[1])>7000
 torch.set_num_threads(2);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('170-second short-test bound')));signal.alarm(170)
 spec=importlib.util.spec_from_file_location('archived_frozen_readout',ARCHIVE)
 legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
 assert Joint.input is F.Probe.input and Joint.from_output is F.Probe.from_output and Joint.forward is F.Probe.forward
 start=time.monotonic();result=dict(datasets={},full_training_started=False,dev_test_evaluated=False)
 for ds in ['beauty','ml-1m']:
  a,unused_ck,unused_sha,unused_ref,hist=F.get_config(ds)
  a.model_path=str(output.parent/(ds+'_UNUSED.pt'));assert not Path(a.model_path).exists()
  # Metadata is reused; no torch.load of ANY historical checkpoint occurs.
  with (ROOT/'data'/ds/'SeqReader.pkl').open('rb') as f:c=pickle.load(f)
  c.data_df={'train':c.data_df['train']}
  # Reuse the existing frozen unit_tests verbatim, on a fresh backbone.
  utils.init_seed(42);fbase=F.Base(a,c).to(a.device)
  with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
   torch.manual_seed(42);fp0=F.Probe(fbase,'P0').to(a.device)
   fp1=F.Probe(fbase,'P1').to(a.device);fp1.readout.load_state_dict(fp0.readout.state_dict())
  ctx=dict(a=a,base=fbase,models=dict(P0=fp0,P1=fp1),data={'train':F.Base.Dataset(fbase,c,'train')},
   config=dict(readout_initial_sha256=F.state_hash(fp0.readout.state_dict()),backbone_state_sha256=F.state_hash(fbase.state_dict())))
  frozen_tests=F.unit_tests(ctx)
  b=fixture(c,a.device,n=16)
  # Verify legacy/default frozen math even with a nonzero output projection.
  with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
   old=legacy.Probe(fbase,'P0').to(a.device)
  old.readout.load_state_dict(fp0.readout.state_dict())
  with torch.no_grad():old.readout[-1].weight.fill_(.001);fp0.readout.load_state_dict(old.readout.state_dict())
  old.train();fp0.train();left=old(b);right=fp0(b)
  assert not fbase.training and left.keys()==right.keys()
  assert all(torch.equal(left[k],right[k]) for k in left)
  assert torch.equal(old.loss(left),fp0.loss(right))
  del old,fp0,fp1,fbase,ctx,left,right
  # Fresh original baseline and both thin joint wrappers: no loading/freeze.
  utils.init_seed(42);base=F.Base(a,c).to(a.device);base_rng=rh();models=[];initial=[]
  for mode in ['P0','P1']:
   a.readout_mode=mode;a.backbone_mode='joint';utils.init_seed(42)
   m=Joint(a,c).to(a.device);assert rh()==base_rng
   assert m.training and m.backbone.training and m.backbone.dropout.training
   assert all(p.requires_grad for p in m.parameters())
   assert all(torch.equal(v,m.backbone.state_dict()[k]) for k,v in base.state_dict().items())
   initial.append(F.state_hash(m.state_dict()));models.append(m)
  assert initial[0]==initial[1]
  assert F.state_hash(models[0].readout.state_dict())=='96e9ce9d173238512ce1864ac6ea50639e1cdac352d78dc3802add2c66e8c5fd'
  # One common legal Train tensor + identical RNG for original dropout/relation.
  saved=rng();outs=[];gradients=[];losses=[];end_rng=[]
  for m in [base]+models:
   restore(saved);m.train();m.zero_grad(set_to_none=True)
   out=m(b);rank=GeneralModel.loss(m,out);total=m.loss(out);total.backward()
   assert '_relation_ids' in out and 'loss_relation' in out
   assert torch.allclose(total,rank+.01*out['loss_relation'],atol=1e-8,rtol=1e-6)
   assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
   outs.append(out);losses.append([float(rank.detach()),float(out['loss_relation']),float(total.detach())]);end_rng.append(rh())
   gradients.append({n:p.grad.detach().clone() if p.grad is not None else None for n,p in (m.backbone if isinstance(m,Joint) else m).named_parameters()})
  assert losses[0]==losses[1]==losses[2] and end_rng[0]==end_rng[1]==end_rng[2]
  assert torch.equal(outs[0]['prediction'],outs[1]['prediction']) and torch.equal(outs[0]['prediction'],outs[2]['prediction'])
  assert torch.equal(outs[0]['_relation_ids'],outs[1]['_relation_ids']) and torch.equal(outs[0]['_relation_ids'],outs[2]['_relation_ids'])
  for g in gradients[1:]:
   for n,v in gradients[0].items():
    assert (v is None)==(g[n] is None)
    if v is not None:assert torch.equal(v,g[n]),n
  # Epoch0-style EVAL prediction equality on the SAME fixed Train prefix only.
  with torch.no_grad():
   for m in [base]+models:m.eval()
   baseline_eval=base(b)['prediction']
   assert all(torch.equal(m(b)['prediction'],baseline_eval) for m in models)
  updates=[]
  for m in models:
   optimizer=BaseRunner(a)._build_optimizer(m)
   active=[p for g in optimizer.param_groups for p in g['params']]
   assert len(active)==len({id(p) for p in active})
   assert {id(p) for p in active}=={id(p) for p in m.parameters() if p.requires_grad}
   step_records=[];restore(saved)
   for step in [1,2]:
    # Restore only ONCE per arm, then let dropout/relation RNG advance naturally.
    m.train();optimizer.zero_grad(set_to_none=True)
    assert m.backbone.training and m.backbone.dropout.training
    out=m(b);total=m.loss(out);total.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    modules={name:float(torch.sqrt(sum((p.grad.detach()**2).sum() for p in mod.parameters())))
      for name,mod in [('ItemEncoder',m.backbone.item_encoder),('QueryExtractor',m.backbone.extractor),
                       ('Aggregator',m.backbone.aggregator),('Position',m.backbone.position_emb),('readout',m.readout)]}
    assert all(v>0 for v in modules.values())
    step_records.append(dict(step=step,loss=float(total.detach()),module_gradient_norms=modules,
      inner_gradient=float(m.readout[0].weight.grad.norm()),output_gradient=float(m.readout[-1].weight.grad.norm()),rng_after=rh()))
    optimizer.step()
   assert step_records[0]['inner_gradient']==0 and step_records[1]['inner_gradient']>0
   updates.append(step_records)
  assert all(updates[0][i]['rng_after']==updates[1][i]['rng_after'] for i in [0,1])
  result['datasets'][ds]=dict(reused_frozen_tests=frozen_tests,legacy_frozen_nonzero_scores_loss_exact=True,
   shared_initial_state_exact=True,initial_state_sha256=initial[0],constructor_rng_matches_original=True,
   initial_prediction_BPR_relation_shared_gradients_exact=True,epoch0_style_eval_Train_only_exact=True,
   losses_BPR_relation_total=losses,all_optimizer_parameters_covered=True,two_step_updates=updates,
   readout_parameters=sum(p.numel() for p in models[0].readout.parameters()),total_parameters=models[0].count_variables(),
   reused_input_and_forward_function_objects=True,historical_config=dict(path=hist['path'],sha256=hist['sha256'],lr=a.lr,l2=a.l2,batch=a.batch_size,optimizer=a.optimizer))
  print('PASS '+json.dumps(dict(dataset=ds,total_parameters=result['datasets'][ds]['total_parameters'],
    readout_parameters=27808,legacy_exact=True,losses=losses,updates=updates)),flush=True)
  del models,base,c,outs,gradients,b;torch.cuda.empty_cache()
 result.update(success=True,seconds=time.monotonic()-start,source_sha256={p:F.sha(ROOT/p) for p in [
  'tools/run_frozen_interest_readout.py','models/sequential/LLMMIRecASPCFReadoutBPR.py','tools/test_readout_mode_reuse.py']})
 F.dump(output,result);signal.alarm(0);print('ALL_PASS '+str(output),flush=True)
if __name__=='__main__':main()
