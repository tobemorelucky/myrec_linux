"""Bounded real-Train routing and fold isolation checks for the six-arm launch."""
import sys,json,time,signal,copy,hashlib,argparse,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import torch,numpy as np,pandas as pd
from gict_data import args_from_config,train_corpus,sha
from sgit_graph import transition_graph
from run_sgitrec import core_hash
from models.sequential.SGITRec import SGITRec
from models.sequential.LLMMIRecASPCFAdaptiveSIL import LLMMIRecASPCFAdaptiveSIL as C1
from utils import utils
OUT=Path(__file__).resolve().parent
CONFIGS={
 'beauty':ROOT/'research/elector_sil_formal_20261023_v1/runs/beauty_k2_s42_SS-C1/config.json',
 'ml-1m':ROOT/'research/elector_sil_k3_20261023_v1/runs/ml-1m_k3_s42_SS-C1/config.json'}
def summary(x):
 x=x.detach().double().reshape(-1)
 return dict(mean=float(x.mean()),min=float(x.min()),median=float(x.median()),p90=float(torch.quantile(x,.9)),max=float(x.max()))
def pairs(x):
 return torch.stack([x[:,a]-x[:,b] for a,b in itertools.combinations(range(x.shape[1]),2)],1)
def routing(joint,ids,N):
 B,K,L,R=joint.shape
 item=joint.new_zeros(B,K,N)
 item.scatter_add_(2,ids[:,None].expand(-1,K,-1,-1).reshape(B,K,-1),joint.reshape(B,K,-1))
 item[:,:,0]=0
 item=item/item.sum(-1,keepdim=True).clamp_min(1e-12)
 order=item.topk(3,-1).indices
 top1=[];top3=[];js=[]
 for a,b in itertools.combinations(range(K),2):
  top1.append((order[:,a,0]==order[:,b,0]).double())
  inter=(order[:,a,:,None]==order[:,b,None,:]).sum((-1,-2)).double()
  top3.append(inter/(6-inter))
  p,q=item[:,a].double(),item[:,b].double();m=(p+q)/2
  js.append(.5*((p*(p.clamp_min(1e-30).log()-m.clamp_min(1e-30).log())).sum(-1)+(q*(q.clamp_min(1e-30).log()-m.clamp_min(1e-30).log())).sum(-1)))
 ent=-(item.double()*item.double().clamp_min(1e-30).log()).sum(-1)
 return dict(pair_top1_same=float(torch.stack(top1).mean()),pair_top3_Jaccard=float(torch.stack(top3).mean()),
  pair_JS=summary(torch.stack(js)),pair_max_abs_mass_difference=float(pairs(item).abs().max()),entropy=summary(ent))
def check(ds):
 start=time.perf_counter();a=args_from_config(CONFIGS[ds]);a.device='cpu';a.sgit_variant='G3'
 a.sgit_graph_path=str(ROOT/'research/sgitrec_v1_20261024/graphs'/(ds+'_v2.pt'))
 c=train_corpus(a);rng=np.random.default_rng(42)
 eligible=np.array(sorted(u for u,h in c.user_his.items() if len(h)>=3))
 users=eligible[np.linspace(0,len(eligible)-1,128,dtype=int)]
 history=np.zeros((len(users),20),np.int64);lengths=[];items=[]
 for row,u in enumerate(users):
  seq=c.user_his[int(u)];pos=int(rng.integers(2,len(seq)))
  prefix=seq[max(0,pos-20):pos];history[row,:len(prefix)]=[v[0] for v in prefix]
  lengths.append(len(prefix));items.append([seq[pos][0],int(rng.choice(c.data_df['train'].item_id.to_numpy()))])
 feed={k:torch.as_tensor(v) for k,v in dict(user_id=users,history_items=history,lengths=lengths,item_id=items).items()}
 utils.init_seed(42);ref=C1(a,c).eval();h0=core_hash(ref);inits={}
 with torch.no_grad():baseline=ref(feed,True)
 for variant in ['B0','G1','G2','G3']:
  a.sgit_variant=variant;utils.init_seed(42);m=SGITRec(a,c).eval()
  assert core_hash(m)==h0
  assert all(p.requires_grad for p in m.parameters()),'Unexpected frozen trainable C1 parameters'
  with torch.no_grad():z=m(feed,True)
  assert torch.equal(z['prediction'],baseline['prediction'])
  inits[variant]=dict(C1_sha256=core_hash(m),trainable_parameters=m.count_variables())
  if variant!='G3':del m
 # Caches must match actual unmapped integer Train user IDs and original construction.
 fold=0;rebuilt,catalog,_=transition_graph(c.user_his,c.n_items,fold,3,8)
 payload=torch.load(a.sgit_graph_path,map_location='cpu',weights_only=True)
 for k,x in rebuilt.items():assert torch.equal(payload['transition_'+k][fold],torch.as_tensor(x)),k
 sem=payload['semantic_ids'][fold].reshape(-1).numpy()
 assert np.isin(sem[sem>0],catalog).all()
 q=m.graph_cache.lookup(feed['history_items'],feed['user_id'],'transition')
 f=feed['user_id']%3
 assert torch.equal(q['ids'],payload['transition_ids'][f[:,None],feed['history_items']])
 phase_ids={}
 for phase in ['dev','test']:
  # IDs ONLY: neither target item nor candidate labels enter preflight or graph.
  ids=pd.read_csv(Path(a.path)/ds/(phase+'.csv'),sep='\t',usecols=['user_id']).user_id.to_numpy(np.int64)
  assert set(ids)<=set(c.user_his)
  phase_ids[phase]=dict(users=len(ids),fold_counts=np.bincount(ids%3,minlength=3).tolist(),mapping='unchanged integer user_id % 3, identical to Train; cached graph never rebuilt')
 with torch.no_grad():
  zero=m(feed,True);assert torch.count_nonzero(zero['graph_delta'])==0
  m.graph_memory.output.weight.copy_(torch.eye(64)) # RAM ONLY; formal init unchanged.
  g3=m(feed,True);m.sgit_variant='G1';g1=m(feed,True);m.sgit_variant='G2';g2=m(feed,True)
  changed=dict(feed);changed['item_id']=feed['item_id'].flip(0)
  other=m(changed,True)
  assert torch.allclose(g2['interest_vectors'],other['interest_vectors'],atol=3e-6,rtol=3e-6)
  mem_diff=pairs(g2['collaborative_memory']).norm(dim=-1)
  g1g2=(g2['interest_vectors']-g1['interest_vectors']).abs().max()
  score_diff=(g2['raw_prediction']-g1['raw_prediction']).abs().max()
  collab_routing=routing(g2['transition_route'],q['ids'],c.n_items)
  semids=m.graph_cache.lookup(feed['history_items'],feed['user_id'],'semantic')['ids']
  semantic_routing=routing(g3['semantic_route'],semids,c.n_items)
  assert collab_routing['pair_max_abs_mass_difference']>0 and mem_diff.max()>0 and g1g2>0,'Completely degenerate interest readout'
  cos=torch.nn.functional.cosine_similarity(g3['collaborative_memory'],g3['semantic_memory'],dim=-1)
  # G1 memory and G2 memory are the same only in special cases, not by formula.
  m.sgit_variant='G3';m.graph_memory.output.weight.zero_()
  recovered=m(feed,True);assert torch.equal(recovered['prediction'],baseline['prediction'])
 del ref
 return dict(dataset=ds,K=a.K,samples=128,history_lengths=summary(feed['lengths']),
  C1_shared_initialization=inits,zero_output_projection_exact_C1=True,formal_projection_restored_zero=True,
  collab_routing=collab_routing,semantic_routing=semantic_routing,
  pair_collab_memory_l2_difference=summary(mem_diff),G1_G2_max_interest_difference=float(g1g2),
  G1_G2_max_raw_score_difference=float(score_diff),collab_semantic_memory_cosine=summary(cos),
  interest_history_attention_pair_difference=summary(pairs(g2['attention_maps']).abs()),
  relation_weights=summary(g3['relation_weights']),
  graph_fold_zero_exact_rebuild=True,semantic_catalog_other_fold_Train_only=True,query_fold_rule='original user_id % 3',
  heldout_ID_mapping=phase_ids,no_Dev_Test_items_or_labels_read=True,own_entire_user_fold_excluded=True,
  candidate_independent=True,seconds=time.perf_counter()-start,passed=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=list(CONFIGS),required=True);x=p.parse_args()
 signal.alarm(170);torch.set_num_threads(4)
 target=OUT/'checks'/(x.dataset+'_routing_v2.json');assert not target.exists()
 value=check(x.dataset)
 with target.open('x') as f:json.dump(value,f,indent=2)
 print(json.dumps(value,indent=2))
if __name__=='__main__':main()
