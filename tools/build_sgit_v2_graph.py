"""Explicit Train-only bridge cache build from an immutable validated V1 cache."""
import sys,argparse,signal,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch,numpy as np
from gict_data import args_from_config,train_corpus,project_output,sha
from models.sequential.LLMMIRecASPCFAdaptiveSIL import array_sha
from sgit_v2_graph import build_bridge
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--base-graph',required=True);p.add_argument('--output',required=True)
    x=p.parse_args();signal.alarm(170);torch.set_num_threads(4);tick=time.perf_counter()
    target=project_output(x.output);a=args_from_config(x.config);c=train_corpus(a)
    base=torch.load(x.base_graph,map_location='cpu',weights_only=True);meta=base['metadata']
    assert meta['scope']=='full_crossfit_train' and meta['schema']=='sgit_crossfit_item_graph_v1'
    assert meta['folds']==3 and meta['R']==8
    assert meta['train_frame_sha256']==array_sha(c.data_df['train'][['user_id','item_id','position']].to_numpy(np.int64))
    assert meta['embedding_sha256']==sha(a.llm_emb_path)
    graph=build_bridge(base);graph['metadata'].update(base_graph_sha256=sha(x.base_graph),builder_sha256=sha(ROOT/'tools/sgit_v2_graph.py'))
    with target.open('xb') as f:torch.save(graph,f)
    value=dict(graph=graph['metadata'],graph_sha256=sha(target),seconds=time.perf_counter()-tick,Dev=False,Test=False)
    with Path(str(target)+'.json').open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps(value,indent=2))
if __name__=='__main__':main()
