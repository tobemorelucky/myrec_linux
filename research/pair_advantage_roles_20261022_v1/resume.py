"""Resume after a report-only NameError: reuse completed Beauty JSON, run ML exactly once."""
import json,time,signal
from pathlib import Path
import audit
audit.csvwrite=audit.reuse.csvwrite
OUT=Path(__file__).resolve().parent
def main():
 started=time.perf_counter();pre=json.loads((OUT/"PREREG.json").read_text())
 assert audit.sha(OUT/"audit.py")==pre["runner_sha256"]
 assert audit.sha(OUT/"role_objective.py")==pre["module_sha256"]
 beauty=json.loads((OUT/"beauty.json").read_text())
 assert beauty["Test_evaluations"]==0
 assert not (OUT/"beauty_dev_metrics.csv").exists() and not (OUT/"ml-1m.json").exists()
 audit.write(OUT/"REPAIR_report_only.json",dict(reason="missing csvwrite alias after Beauty JSON had been written; no numerical/model code change",
  original_runner_sha256=pre["runner_sha256"],resume_sha256=audit.sha(__file__),
  reused_beauty_result_sha256=audit.sha(OUT/"beauty.json"),reused_beauty_total_seconds=beauty["total_seconds"],
  no_Beauty_retraining_or_Dev_repeat=True))
 rows=[]
 for stage,schemes in (("initial",{"existing":beauty["initial_Dev"]}),("after16",beauty["final_Dev"])):
  for name,scores in schemes.items():
   for scoremode,mm in scores.items():rows.append(dict(stage=stage,scheme=name,scoring=scoremode,**mm))
 audit.csvwrite(OUT/"beauty_dev_metrics.csv",rows)
 signal.alarm(100);audit.torch.set_num_threads(4)
 ml=audit.dataset_run("ml-1m",pre)
 audit.write(OUT/"COMPLETE.json",dict(combined_dataset_body_seconds=beauty["total_seconds"]+ml["total_seconds"],
  resume_body_seconds=time.perf_counter()-started,math_preparation_seconds_upper_bound=10,
  config=audit.CONFIG,datasets=["beauty","ml-1m"],Test_evaluations=0,full_training_started=False,
  module_sha256=audit.sha(OUT/"role_objective.py"),runner_sha256=audit.sha(OUT/"audit.py"),resume_sha256=audit.sha(__file__),
  Beauty_reused_not_repeated=True))
 print("ALL_COMPLETE",flush=True)
if __name__=="__main__":main()

