"""Explicit full Train-only graph build, no model training or evaluation."""
import sys,json,time,signal,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch,numpy as np
from gict_data import args_from_config,train_corpus,project_output,sha
from sgit_graph import build_graphs
from models.sequential.llmmi_utils import load_llm_table
from models.sequential.LLMMIRecASPCFAdaptiveSIL import array_sha
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--timeout-seconds',type=int,default=170)
 x=p.parse_args();signal.alarm(min(x.timeout_seconds,170));torch.set_num_threads(4)
 output=project_output(x.output);a=args_from_config(x.config);c=train_corpus(a)
 table=load_llm_table(a.llm_emb_path,c.n_items)
 tensors,metadata=build_graphs(c.user_his,table,R=8,folds=3)
 frame=c.data_df['train'];metadata.update(schema='sgit_crossfit_item_graph_v1',scope='full_crossfit_train',dataset=a.dataset,
  train_csv_sha256=sha(Path(a.path)/a.dataset/'train.csv'),embedding_sha256=sha(a.llm_emb_path),
  train_frame_sha256=array_sha(frame[['user_id','item_id','position']].to_numpy(np.int64)),Train_nodes=int(frame.item_id.nunique()),
  Train_rows=len(frame),users=len(c.user_his),semantic_dim=table.shape[1],
  leakage_policy='3 user folds; graph/count/support/semantic candidate catalog for every query exclude its entire user fold; same selection for Train/Dev/Test. Other users Train only, not globally time-sliced.',
  sources={str(Path(p).resolve()):sha(p) for p in [__file__,'tools/sgit_graph.py','tools/gict_data.py']})
 with output.open('xb') as f:torch.save(dict(metadata=metadata,**tensors),f)
 with Path(str(output)+'.json').open('x') as f:json.dump(metadata,f,indent=2)
 print(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
