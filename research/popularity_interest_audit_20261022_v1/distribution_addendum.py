"""Metadata-only addendum after coefficient locks/Test: no model inference."""
from pathlib import Path
import sys,json,pickle,signal,time,hashlib
import numpy as np
ROOT=Path("/home/yyx/hzgcode/pom2.0");sys.path.insert(0,str(ROOT))
from models.sequential.LLMMIRecASPCFSampledSoftmax import build_train_membership
OUT=ROOT/"research/popularity_interest_audit_20261022_v1"
signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError("170s metadata limit")))
signal.alarm(170);start=time.perf_counter();results={}
for ds in ["ml-1m","beauty"]:
 assert (OUT/ds/"TEST_COMPLETE.json").exists()
 f=np.load(OUT/ds/"frequency_distribution.npz");q=f["expected"];raw=f["raw"];target=f["target"]
 uniform=np.ones(len(raw));uniform[0]=0;uniform/=uniform.sum()
 with (ROOT/"data"/ds/"SeqReader.pkl").open("rb") as stream:c=pickle.load(stream)
 packed=build_train_membership(c).numpy().view(np.uint64)
 result=dict(catalog_items=len(raw)-1,raw_zero_items=int((raw[1:]==0).sum()),
             target_zero_items=int((target[1:]==0).sum()),by_phase={})
 for phase in ["dev","test"]:
  neg=f[phase+"_negative"].astype(float);neg/=neg.sum()
  good=(q>0)&(neg>0)&(raw>0);lr=np.log(q[good]/neg[good]);x=np.log1p(raw[good])
  corr=float(np.corrcoef(x,lr)[0,1]);coeff=np.polyfit(x,lr,1)
  frame=c.data_df[phase];uids=frame.user_id.to_numpy(np.int64);targets=frame.item_id.to_numpy(np.int64)
  known=0;self_item=0;slots=0
  for off in range(0,len(frame),1024):
   ar=np.stack(frame.neg_items.to_numpy()[off:off+1024]).astype(np.int64)
   uu=uids[off:off+1024]
   clicked=(np.right_shift(packed[uu[:,None],ar//64],(ar%64).astype(np.uint64))&np.uint64(1)).astype(bool)
   known+=int(clicked.sum());self_item+=int((ar==targets[off:off+1024,None]).sum());slots+=ar.size
  result["by_phase"][phase]=dict(TV_negative_to_catalog_uniform=float(abs(neg-uniform).sum()/2),
    common_positive_support_items=int(good.sum()),correlation_on_common_support=corr,linear_shape_R2=corr*corr,
    log_density_ratio_on_log_raw_frequency_slope=float(coeff[0]),
    negative_mass_on_zero_target_count=float(neg[target==0].sum()),
    negative_slots_Train_clicked=known,negative_Train_clicked_fraction=known/slots,
    negative_slots_equal_positive=self_item,negative_equal_positive_fraction=self_item/slots)
 results[ds]=result
write=dict(results=results,scope="metadata only; no predictive use of Dev/Test negative frequencies; locked scores unchanged",
 seconds=time.perf_counter()-start)
with (OUT/"distribution_addendum.json").open("x") as stream:json.dump(write,stream,indent=2,allow_nan=False)
print(json.dumps(write))
signal.alarm(0)

