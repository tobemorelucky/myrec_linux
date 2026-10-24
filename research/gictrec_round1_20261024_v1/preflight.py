"""Bounded full-bank diagnostics and real optimizer update checks; Dev only."""
import signal,importlib.util
from common import *
from models.sequential.gict_components import interest_profiles
def summary(x):
 x=x.detach().float().flatten().cpu()
 return dict(min=float(x.min()),p10=float(x.quantile(.1)),median=float(x.median()),p90=float(x.quantile(.9)),max=float(x.max()),mean=float(x.mean()))
def main():
 signal.alarm(170);torch.set_num_threads(4);start=time.perf_counter();verify()
 spec=importlib.util.spec_from_file_location('resource',ROOT/'research/elector_sil_objective_full_20261023_v2/scheduler.py')
 r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
 inv=r.observe();free=[g for g,v in inv.items() if r.idle(v)]
 assert free,'No idle GPU: preflight should wait, never preempt'
 gpu=free[0];fd=r.acquire(ROOT/('research/gpu_resource_scheduler_20261022_v1/gpu%d.lock'%gpu))
 assert fd is not None
 try:
  assert r.idle(r.observe()[gpu]);torch.cuda.set_device(gpu);torch.cuda.reset_peak_memory_stats(gpu)
  device='cuda:'+str(gpu);result={}
  for ds in ['ml-1m','beauty']:
   checks={};initial_hash=None;zero_output=None
   for variant in ['G0','G1','G2','G3']:
    m,a,c=make_model(ds,variant,device);m.eval()
    h=common_hash(m)
    if initial_hash is None:initial_hash=h
    assert h==initial_hash
    users=sorted(u for u,v in c.user_his.items() if len(v)>=2)
    rng=np.random.default_rng(42);users=np.asarray(users)[np.linspace(0,len(users)-1,128,dtype=int)]
    history=np.zeros((128,20),np.int64);lengths=[];pos=[]
    for row,u in enumerate(users):
     seq=c.user_his[int(u)];n=min(1+row%20,len(seq)-1)
     history[row,:n]=[x[0] for x in seq[:n]];lengths.append(n);pos.append(seq[n][0])
    item=np.stack([pos,rng.choice(c.data_df['train'].item_id.unique(),128)],1)
    feed={k:torch.as_tensor(v,device=device) for k,v in dict(user_id=users,history_items=history,lengths=lengths,item_id=item).items()}
    with torch.no_grad():
     initial=m(feed,True)
     if zero_output is None:zero_output=initial['prediction'].clone()
     assert torch.equal(initial['prediction'],zero_output)
    m.cur_epoch=1;m.actions_before_epoch();m.train()
    optimizer=torch.optim.Adam(m.customize_parameters(),lr=a.lr,weight_decay=a.l2)
    torch.cuda.synchronize();tick=time.perf_counter();gradient=0.
    for step in range(3):
     optimizer.zero_grad();got=m(feed,True);loss=m.loss(got);assert torch.isfinite(loss)
     loss.backward()
     if variant!='G0':
      gradient=float(m.transfer.output.weight.grad.norm());assert gradient>0
     optimizer.step()
    torch.cuda.synchronize();seconds=(time.perf_counter()-tick)/3
    m.eval()
    with torch.no_grad():updated=m(feed,True)
    check=dict(train_step_seconds_128=seconds,rows_per_second_128=128/seconds,parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
     projection_gradient_norm=gradient,projection_weight_norm=float(m.transfer.output.weight.norm()),finite_loss=float(loss.detach()),common_initial_sha256=h)
    if variant!='G0':
     assert m.transfer.output.weight.norm()>0 and updated['transfer_delta'].abs().sum()>0
     valid=updated['gict_neighbor_valid'];neighbor=updated['gict_neighbor_users']
     assert not (neighbor==feed['user_id'][:,None])[valid].any()
     altered=dict(feed);altered['item_id']=feed['item_id'].flip(0)
     with torch.no_grad():changed=m(altered,True)
     assert torch.equal(updated['interest_vectors'],changed['interest_vectors'])
     assert torch.equal(updated['gict_neighbor_users'],changed['gict_neighbor_users'])
     check.update(nonzero_parameter_update=True,nonzero_transfer=True,target_independent=True,
      measured_neighbor_coverage=float(valid.any(1).float().mean()),mean_valid_neighbors=float(valid.float().sum(1).mean()))
     if variant!='G1':
      check['matching_JS']=summary(updated['match_distance'])
      check['current_support']=summary(interest_profiles(m.semantic_index.item_probabilities,feed['history_items'],updated['attention_maps'],feed['lengths'])[1])
      check['neighbor_support']=summary(m.semantic_index.interest_support)
      if variant=='G3':check['reliability']=summary(updated['reliability'])
    # Reference Dev once per dataset, no Test read. Exact archived six metrics.
    if variant=='G0':
     m.load_c1(torch.load(read(OUT/'PLAN.json')['datasets'][ds]['checkpoint'],map_location='cpu',weights_only=True))
     heldout(c,a,'dev');d=m.Dataset(m,c,'dev');d.prepare();anchor=score_views(d)
     expected=read(Path(read(OUT/'PLAN.json')['datasets'][ds]['config']).parent/'best_metadata.json')
     for scheme in ('raw','locked_prior'):
      assert all(abs(anchor['metrics'][scheme][key]-expected[scheme][key])<1e-10 for key in KEYS),(ds,scheme,anchor['metrics'][scheme])
     protocol_lock(ds,'dev',anchor)
     anchor_path=OUT/(ds+'_EPOCH0_ANCHOR.json')
     if anchor_path.exists():assert read(anchor_path)==anchor
     else:write(anchor_path,anchor)
     check['full_Dev_anchor_exact']=True;del d
    checks[variant]=check
    del optimizer,got,updated,initial,m,c;torch.cuda.empty_cache()
   result[ds]=checks
  result.update(passed=True,seconds=time.perf_counter()-start,physical_GPU=gpu,GPU_peak_MiB=torch.cuda.max_memory_allocated(gpu)/2**20,
    full_training=False,Test_evaluations=0,diagnostic_query_rows_per_dataset=128)
  write(OUT/'PREFLIGHT.json',result);print(json.dumps(result,indent=2))
 finally:os.close(fd)
if __name__=='__main__':main()
