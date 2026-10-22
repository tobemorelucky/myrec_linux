"""Single bounded CPU geometry audit; no recommendation labels or asset writes."""
import os,time,pathlib,json,pickle,hashlib
import numpy as np
R=pathlib.Path('/home/yyx/hzgcode/pom2.0');W=R/'research/ml1m_embedding_v2_20261020'
OUT=pathlib.Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def summary(v):
 v=np.asarray(v);q=np.quantile(v,[0,.01,.05,.5,.95,.99,1])
 return dict(zip(['min','p01','p05','median','p95','p99','max'],map(float,q)),mean=float(np.mean(v)),std=float(np.std(v)))
def spectrum(x,detail=False):
 x=x.astype(np.float64);x=x-x.mean(0,keepdims=True)
 vals=np.linalg.eigvalsh(x.T@x/(len(x)-1))[::-1];minimum=float(vals.min());vals=np.maximum(vals,0)
 p=vals/vals.sum();s=np.sqrt(vals*(len(x)-1));ps=s/s.sum()
 entropy=lambda a:float(np.exp(-np.sum(a[a>0]*np.log(a[a>0]))))
 d={'effective_rank_singular_entropy':entropy(ps),'effective_rank_variance_entropy':entropy(p),'participation_rank':float(1/np.sum(p*p)),
 'variance_share_top1':float(p[:1].sum()),'variance_share_top5':float(p[:5].sum()),'variance_share_top10':float(p[:10].sum()),
 'variance_share_top50':float(p[:50].sum()),'variance_share_top100':float(p[:100].sum()),
 'rank90':int(np.searchsorted(np.cumsum(p),.90)+1),'rank95':int(np.searchsorted(np.cumsum(p),.95)+1),
 'rank99':int(np.searchsorted(np.cumsum(p),.99)+1),'smallest_cov_eigenvalue_before_clip':minimum}
 if detail:d['singular_values']=s.tolist()
 return d
start=time.monotonic()
mapping=json.loads((R/'data/ml-1m/item2id.json').read_text());assert sorted(mapping.values())==list(range(1,3707))
movies={}
for line in (OUT/'movies.dat').read_text(encoding='latin-1').splitlines():
 mid,title,genres=line.split('::');movies[mid]={'movie_id':int(mid),'title':title,'genres':genres.split('|')}
assert set(mapping)<=set(movies)
meta=[None]*3706
for mid,i in mapping.items():meta[i-1]=dict(movies[mid],item_id=i)
genres=sorted(set(g for m in meta for g in m['genres']))
gb=np.array([[g in m['genres'] for g in genres] for m in meta],dtype=np.int16)
rng=np.random.RandomState(42);left=rng.randint(0,3706,100000);right=rng.randint(0,3706,100000);right[right==left]=(right[right==left]+1)%3706
inter=(gb[left]*gb[right]).sum(-1);un=gb[left].sum(-1)+gb[right].sum(-1)-inter
random_genre={'share_any_genre':float((inter>0).mean()),'mean_jaccard':float(np.mean(inter/np.maximum(un,1)))}
results={};neighbors={};pair_values={}
for label,rel in [('V1_Rebuilt','imported_assets/v1_rebuilt_20261020/base_pca.pkl'),('V2_Rich','imported_assets/v2_rich512_l2_d400_20261021/rich_pca.pkl')]:
 p=W/rel
 with p.open('rb') as f:a=pickle.load(f)
 assert a.shape==(3707,1536) and a.dtype==np.float32 and not np.any(a[0]) and np.isfinite(a).all()
 x=a[1:];norm=np.linalg.norm(x.astype(np.float64),axis=1);assert norm.min()>1e-4
 z=(x/norm[:,None]).astype(np.float32);pairs=[]
 for i in range(0,len(left),4096):
  sl=slice(i,i+4096);pairs.extend(np.einsum('ij,ij->i',z[left[sl]],z[right[sl]]).tolist())
 pairs=np.asarray(pairs);pair_values[label]=pairs
 spec=spectrum(x,True);direction=spectrum(z)
 sim=z@z.T;np.fill_diagonal(sim,-np.inf)
 top=np.argpartition(sim,-10,axis=1)[:,-10:];scores=np.take_along_axis(sim,top,axis=1)
 order=np.argsort(-scores,axis=1);top=np.take_along_axis(top,order,axis=1);scores=np.take_along_axis(scores,order,axis=1);neighbors[label]=top
 shared=(gb[:,None,:]*gb[top]).sum(-1);union=gb.sum(-1)[:,None]+gb[top].sum(-1)-shared
 d={'path':str(p),'sha256':sha(p),'shape':list(a.shape),'dtype':str(a.dtype),'valid_rows':3706,'padding_zero':True,'all_finite':True,
 'norms':summary(norm),'norms_first512':summary(np.linalg.norm(x[:,:512].astype(np.float64),axis=1)),
 'norms_tail1024':summary(np.linalg.norm(x[:,512:].astype(np.float64),axis=1)),
 'exact_duplicate_rows':len(x)-len({hashlib.sha256(v.tobytes()).hexdigest() for v in x}),
 'pair_cosine':summary(pairs),'pair_fraction_gt_08':float((pairs>.8).mean()),'pair_fraction_gt_095':float((pairs>.95).mean()),
 'centered_spectrum':spec,'unit_direction_spectrum':direction,'mean_unit_direction_norm':float(np.linalg.norm(z.mean(0))),
 'top1_cosine':summary(scores[:,0]),'top10_cosine':summary(scores.ravel()),
 'knn10_genre_share_any':float((shared>0).mean()),'knn10_genre_jaccard':float(np.mean(shared/np.maximum(union,1))),
 'fixed_cases':[]}
 for mid in ['1','260','318','593','924']:
  if mid not in mapping:continue
  i=mapping[mid]-1
  d['fixed_cases'].append({'query':meta[i],'neighbors':[dict(meta[int(k)],cosine=float(sim[i,k])) for k in top[i,:5]]})
 results[label]=d
 print(json.dumps({'finished':label,'elapsed_seconds':time.monotonic()-start}),flush=True)
overlap=np.asarray([len(set(a)&set(b))/10 for a,b in zip(neighbors['V1_Rebuilt'],neighbors['V2_Rich'])])
result={'cpu_only':True,'seed':42,'pair_sample_size':100000,'fixed_cases_original_MovieID':[1,260,318,593,924],
 'movies_sha256':sha(OUT/'movies.dat'),'mapping_sha256':sha(R/'data/ml-1m/item2id.json'),
 'random_pair_genre_baseline':random_genre,'results':results,
 'cross_version_neighbor_overlap_fraction':summary(overlap),'pair_cosine_Pearson':float(np.corrcoef(pair_values['V1_Rebuilt'],pair_values['V2_Rich'])[0,1]),
 'coordinate_comparison':'No per-axis subtraction or cross-PCA dot product; only within-space rotation-invariant geometry',
 'elapsed_seconds':time.monotonic()-start}
with (OUT/'GEOMETRY_COMPLETE.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({'passed':True,'seconds':result['elapsed_seconds'],'baseline':random_genre,
 'summary':{k:{a:v[a] for a in ['norms','pair_cosine','top1_cosine','knn10_genre_share_any','knn10_genre_jaccard','unit_direction_spectrum']} for k,v in results.items()},
 'raw_spectrum':{k:{x:y for x,y in v['centered_spectrum'].items() if x!='singular_values'} for k,v in results.items()},
 'overlap':result['cross_version_neighbor_overlap_fraction'],'pair_Pearson':result['pair_cosine_Pearson']}))
