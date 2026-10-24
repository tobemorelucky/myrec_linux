"""One safe final-evaluation repair of the existing completed Toys BPR, never training."""
import argparse,traceback,importlib.util,logging,types
from common import *
def main():
 verify();plan=read(OUT/'PLAN.json');old=Path(plan['Toys']['old_root']);repair=Path(plan['Toys']['repair_root'])
 for pid in [plan['Toys']['BPR_pid'],plan['Toys']['old_dispatcher_pid']]:
  if Path('/proc/%d/stat'%pid).exists():
   state=Path('/proc/%d/stat'%pid).read_text().split(') ',1)[1].split()[0]
   assert state=='Z','Original BPR/evaluation/dispatcher must exit first'
 p=OUT/'toys_bpr_acceptance';original=old/'arms/BPR-B0'
 assert not (original/'TEST_COMPLETE.json').exists(),'Already accepted; do not repeat Test'
 assert not (p/'TEST_STARTED.json').exists(),'No automatic repeated repair evaluation'
 tr=read(original/'train_result.json');assert tr['train_complete'] and tr['Dev_best_selected']
 failure=read(original/'FAILURE.json')
 assert 'numpy' in failure['traceback'].lower() and ('cuda' in failure['traceback'].lower() or 'gpu' in failure['traceback'].lower())
 assert "score_views" in failure['traceback'],'Only the known final evaluator failure is authorized'
 assert read(repair/'EVAL_FIX_CHECK_FINAL.json')['passed']
 assert sha(repair/'common.py')==read(repair/'REPAIR_FINAL.json')['new_common_sha256']
 checkpoint=original/'best.pt';assert sha(checkpoint)==tr['checkpoint_sha256']
 assert (old/'POP_PRIOR_LOCK.json').exists(),'Do not recalibrate the BPR final prior'
 spec=importlib.util.spec_from_file_location('toys_fixed_eval_common',repair/'common.py')
 fixed=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixed);fixed.verify()
 # Read the original final BPR prior and Train metadata; output stays in new directory.
 fixed.OUT=old
 c=fixed.read_corpus('test')
 a=types.SimpleNamespace(**read(original/'config.json')['args']);a.device=torch.device('cuda:0');a.model_path=str(checkpoint)
 utils.init_seed(42);m=fixed.ASPCF(a,c).to(a.device)
 m.load_state_dict(torch.load(checkpoint,map_location=a.device,weights_only=True),strict=True)
 d=m.Dataset(m,c,'test');d.prepare()
 lock=read(old/'POP_PRIOR_LOCK.json')
 write(p/'TEST_STARTED.json',dict(checkpoint_sha256=sha(checkpoint),best_epoch=tr['best']['epoch'],original_failed_attempt=read(original/'TEST_STARTED.json') if (original/'TEST_STARTED.json').exists() else None,
  repair='Only known GPU-to-CPU conversion; no training/selection/prior changes',prior_sha256=sha(old/'POP_PRIOR_LOCK.json')))
 start=time.perf_counter();got=fixed.score_views(d,p/'test_user_ranks.csv')
 assert sha(checkpoint)==tr['checkpoint_sha256']
 write(p/'TEST_PROTOCOL.json',dict(input_hash=got['input_hash'],rows=got['rows'],candidates=1001,duplicate_slots_preserved=True))
 with (p/'test_metrics.csv').open('x',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=['scheme',*KEYS]);writer.writeheader()
  for s,v in got['metrics'].items():writer.writerow(dict(scheme=s,**v))
 result=dict(tr);result.update(Test=got['metrics'],Test_accepted=True,Test_passes=1,Test_seconds=time.perf_counter()-start,
  original_failed_evaluation=True,original_train_result=str(original/'train_result.json'),prior_scale=lock,Test_input_hash=got['input_hash'])
 write(p/'result.json',result);write(p/'TEST_COMPLETE.json',dict(one_Test_pass=True,result_sha256=sha(p/'result.json'),checkpoint_sha256=sha(checkpoint)))
if __name__=='__main__':
 try:main()
 except BaseException as error:
  write(OUT/'toys_bpr_acceptance/FAILURE.json',dict(error=repr(error),traceback=traceback.format_exc(),no_retry=True));raise
