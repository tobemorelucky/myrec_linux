"""Exactly four authorized from-scratch BPR/Dev runs; original Runner mathematics unchanged."""
import os,sys,time,json,hashlib,pickle,random,runpy,subprocess,re,traceback,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SOURCES=['models/sequential/LLMMIRecASPCFComiRecReference.py','models/sequential/LLMMIRecASPCF.py','models/sequential/llmmi_components.py','models/BaseModel.py','helpers/BaseRunner.py','helpers/SeqReader.py','helpers/BaseReader.py','main.py','tools/run_comirec_reference_round1.py']
def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):d.update(b)
    return d.hexdigest()
def write_new(path,value):
    p=Path(path);assert p.resolve().is_relative_to(ROOT) and not p.exists(),str(p)
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def runtime(args):
    assert args.pop(0)=='--'
    def arg(k):return args[args.index(k)+1]
    assert arg('--gpu')=='1' and arg('--dev_only')=='1' and arg('--test_epoch')=='-1'
    assert arg('--load')=='0' and arg('--train')=='1' and arg('--regenerate')=='0'
    assert arg('--reference_scoring')=='max' and arg('--model_name')=='LLMMIRecASPCFComiRecReference'
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    import numpy as np
    import torch
    import helpers.BaseRunner as runner_module
    from helpers.BaseRunner import BaseRunner
    from models.BaseModel import GeneralModel
    from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF
    from utils import utils
    log=Path(arg('--log_file'));config=json.loads((log.parent/'config.json').read_text())
    state=dict(success=False,test_executed=False,epochs=[],probes=[],actual_update_probes=[])
    started=time.monotonic();active=False;fixture=None;steps=0;epoch_active=0;probe_steps=set()
    train_started=None
    def rng():
        return dict(cpu=sha_bytes(torch.get_rng_state().numpy().tobytes()),
                    cuda=sha_bytes(torch.cuda.get_rng_state().cpu().numpy().tobytes()),
                    numpy=sha_bytes(pickle.dumps(np.random.get_state())),python=sha_bytes(pickle.dumps(random.getstate())))
    def digest(model,prefix):
        d=hashlib.sha256()
        for name,t in model.state_dict().items():
            if name.startswith(prefix):
                d.update(name.encode());d.update(t.detach().cpu().contiguous().numpy().tobytes())
        return d.hexdigest()
    def probe(model,corpus,epoch):
        nonlocal fixture
        before=rng();was=model.training;npstate=np.random.get_state();pystate=random.getstate()
        try:
            with torch.random.fork_rng(devices=[0]):
                if fixture is None:
                    local=np.random.default_rng(42);df=corpus.data_df['train']
                    indices=local.choice(np.flatnonzero(df.position.to_numpy()>0),256,replace=False)
                    hs=[];lens=[];users=[];items=[]
                    for idx in indices:
                        row=df.iloc[int(idx)];u=int(row.user_id);pos=int(row.position)
                        seq=corpus.user_his[u][max(0,pos-20):pos]
                        assert len(seq)>0 and all(t<=row.time for _,t in seq)
                        h=[int(j) for j,_ in seq];neg=int(local.integers(1,corpus.n_items))
                        while neg in corpus.train_clicked_set[u]:neg=int(local.integers(1,corpus.n_items))
                        hs.append(h+[0]*(20-len(h)));lens.append(len(h));users.append(u);items.append([int(row.item_id),neg])
                    fixture={k:torch.tensor(v,device=model.device) for k,v in dict(history_items=hs,lengths=lens,user_id=users,item_id=items).items()}
                    state['fixed_train_fixture_sha256']=sha_bytes(b''.join(x.detach().cpu().numpy().tobytes() for x in fixture.values()))
                model.eval();out=model(fixture,True)
                rank=GeneralModel.loss(model,out)
                p=model.extractor.query if model.reference_extractor=='query' else model.extractor.vote_weight
                grad=torch.autograd.grad(rank,p)[0]
                assert torch.isfinite(grad).all() and torch.isfinite(rank)
                v=out['interest_vectors'];z=out['interest_candidate_scores'];pred=out['prediction'];head=out['selected_interest']
                norm=v.norm(dim=-1);diff=(v[:,0]-v[:,1]).norm(dim=-1)/norm.mean(-1).clamp_min(1e-12)
                q=lambda x:{'mean':float(x.mean()),'median':float(x.median()),'max':float(x.max()),'min':float(x.min())}
                record=dict(epoch=epoch,rows=256,dropout=False,mode='fixed Train pure single-negative BPR, no update',
                    ranking_loss=float(rank),interest_norm=q(norm),relative_head_difference=q(diff),
                    positive_scores=q(pred[:,0]),negative_scores=q(pred[:,1]),margin=q(pred[:,0]-pred[:,1]),
                    head_score_absolute_difference=q((z[:,0]-z[:,1]).abs()),
                    positive_head_counts=torch.bincount(head[:,0],minlength=2).tolist(),
                    negative_head_counts=torch.bincount(head[:,1],minlength=2).tolist(),
                    positive_negative_different_heads=float((head[:,0]!=head[:,1]).float().mean()),
                    routing_parameter_norm=float(p.norm()),routing_task_gradient_norm=float(grad.norm()),
                    routing_decay_gradient_norm=float((p*config['historical_config']['l2_float']).norm()),
                    task_to_decay_ratio=float(grad.norm()/(p.norm()*config['historical_config']['l2_float']).clamp_min(1e-30)))
                valid=torch.arange(20,device=model.device)[None]<fixture['lengths'][:,None]
                a=out['attention_maps']
                if model.reference_extractor=='dr':
                    r=a.permute(0,2,1)[valid]
                    record['routing_entropy_mean']=float(-(r*r.clamp_min(1e-30).log()).sum(-1).mean())
                    record['routing_max_probability_mean']=float(r.max(-1).values.mean())
                else:record['attention_head_L1_mean']=float((a[:,0]-a[:,1]).abs().sum(-1).mean())
                state['probes'].append(record)
                print('FIXED_TRAIN_PROBE '+json.dumps(record),flush=True)
        finally:
            model.train(was);np.random.set_state(npstate);random.setstate(pystate)
        assert before==rng(),'Probe altered training RNG'
        state['probes'][-1]['rng_unchanged']=True
    # Record actual CPU sample order without adding RNG draws or GPU synchronizations.
    OriginalLoader=runner_module.DataLoader
    class RecordedLoader(OriginalLoader):
        def __iter__(self):
            if getattr(self.dataset,'phase',None)!='train' or not active:
                yield from super().__iter__();return
            order=hashlib.sha256();count=0
            for b in super().__iter__():
                order.update(b['user_id'].numpy().tobytes());order.update(b['item_id'][:,0].contiguous().numpy().tobytes())
                count+=int(b['item_id'].shape[0]);yield b
            state['epochs'][-1]['train_order_sha256']=order.hexdigest()
            state['epochs'][-1]['rows_seen']=count
    runner_module.DataLoader=RecordedLoader
    old_fit=BaseRunner.fit;old_eval=BaseRunner.evaluate;old_train=BaseRunner.train;old_build=BaseRunner._build_optimizer
    def build(self,model):
        opt=old_build(self,model)
        p=model.extractor.query if model.reference_extractor=='query' else model.extractor.vote_weight
        pending={}
        def pre(opt,unused_args,unused_kwargs):
            nonlocal steps
            steps+=1
            if epoch_active in (1,50,100,200) and epoch_active not in probe_steps:
                probe_steps.add(epoch_active)
                assert p.grad is not None and torch.isfinite(p.grad).all()
                group=next(g for g in opt.param_groups if any(x is p for x in g['params']))
                pending.update(epoch=epoch_active,step=steps,before=p.detach().clone(),
                    total_loss_gradient_norm=float(p.grad.norm()),
                    decay_gradient_norm=float(p.detach().norm()*group['weight_decay']))
        def post(opt,unused_args,unused_kwargs):
            if pending:
                before=pending.pop('before');delta=p.detach()-before
                pending.update(update_norm=float(delta.norm()),relative_update_norm=float(delta.norm()/before.norm().clamp_min(1e-30)),
                    note='Observed Adam update uses total BPR+relation gradient plus coupled weight decay; not a pure task attribution')
                state['actual_update_probes'].append(dict(pending));pending.clear()
        opt.register_step_pre_hook(pre);opt.register_step_post_hook(post)
        return opt
    def fit(self,dataset,epoch=-1):
        nonlocal epoch_active
        epoch_active=epoch;t=time.monotonic()
        entry=dict(epoch=epoch,rng_start=rng());state['epochs'].append(entry)
        loss=old_fit(self,dataset,epoch)
        entry.update(fit_seconds=time.monotonic()-t,loss=float(loss),rng_after_fit=rng(),
            train_negative_sha256=sha_bytes(np.ascontiguousarray(dataset.data['neg_items']).tobytes()),optimizer_steps_total=steps)
        assert np.isfinite(loss),'Non-finite training loss'
        if epoch in (1,50,100,200):probe(dataset.model,dataset.corpus,epoch)
        return loss
    def evaluate(self,dataset,topks,metrics):
        assert dataset.phase=='dev','Test is forbidden'
        t=time.monotonic();result=old_eval(self,dataset,[5,10,20],self.metrics)
        assert all(np.isfinite(v) for v in result.values()),'Non-finite Dev metrics'
        entry=state['epochs'][-1];entry['dev']={k:float(v) for k,v in result.items()}
        entry['dev_seconds']=time.monotonic()-t
        entry['peak_allocated_MiB']=torch.cuda.max_memory_allocated()/2**20
        write_new(log.parent/('epoch_%03d.json'%entry['epoch']),entry)
        print('DEV_FULL '+json.dumps(dict(epoch=entry['epoch'],**entry['dev'])),flush=True)
        if time.monotonic()-started>4*3600:raise TimeoutError('Authorized per-run four-hour resource budget reached')
        return result
    def train(self,data):
        nonlocal active,train_started
        assert set(data)=={'train','dev'} and self.test_epoch==-1
        assert self.epoch==200 and self.early_stop==10 and self.batch_size==1024
        model=data['train'].model;assert model.optimizer is None
        assert model.num_neg==1 and model.K==2 and model.lambda_relation==.01
        state.update(parameters=model.count_variables(),initial_item_encoder_sha256=digest(model,'item_encoder.'),
                     initial_position_sha256=digest(model,'position_emb.'),initial_rng=rng(),
                     train_rows=len(data['train']),dev_rows=len(data['dev']),initialization='from scratch; no checkpoint loaded')
        # Compare real run initial shared tensors against a newly seeded original ASPCF.
        npstate=np.random.get_state();pystate=random.getstate()
        with torch.random.fork_rng(devices=[0]):
            utils.init_seed(42)
            a=types.SimpleNamespace(**config['effective_args']);a.device=model.device
            base=LLMMIRecASPCF(a,data['train'].corpus)
            assert digest(base,'item_encoder.')==state['initial_item_encoder_sha256']
            assert digest(base,'position_emb.')==state['initial_position_sha256']
            if model.reference_extractor=='query':assert digest(base,'extractor.')==digest(model,'extractor.')
            del base
        np.random.set_state(npstate);random.setstate(pystate)
        assert rng()==state['initial_rng']
        state['original_aspcf_shared_initialization_verified']=True
        probe(model,data['train'].corpus,0)
        write_new(log.parent/'initialization.json',{k:v for k,v in state.items() if k not in ('epochs','probes','actual_update_probes')})
        train_started=time.monotonic();active=True
        try:old_train(self,data)
        finally:active=False
        state['training_dev_wall_seconds']=time.monotonic()-train_started
        values=[e['dev'][self.main_metric] for e in state['epochs']];best=max(values)
        first=values.index(best)+1;last=max(i+1 for i,v in enumerate(values) if v==best)
        state.update(best_reported_epoch=first,checkpoint_saved_epoch=last,end_epoch=len(values),
                     dev=state['epochs'][last-1]['dev'],checkpoint_sha256=sha(model.model_path))
        loaded=torch.load(model.model_path,map_location=model.device,weights_only=True)
        assert all(torch.equal(t,loaded[n]) for n,t in model.state_dict().items())
        state['best_checkpoint_loaded_verified']=True
        probe(model,data['train'].corpus,'best_epoch_'+str(last))
        state['best_probe_checkpoint_unchanged']=sha(model.model_path)==state['checkpoint_sha256']
    BaseRunner._build_optimizer=build;BaseRunner.fit=fit;BaseRunner.evaluate=evaluate;BaseRunner.train=train
    torch.cuda.reset_peak_memory_stats()
    sys.argv=['main.py']+args
    try:
        runpy.run_path(str(ROOT/'main.py'),run_name='__main__')
        state['success']=True
    except BaseException as exc:
        state['error']=repr(exc);state['traceback']=traceback.format_exc();raise
    finally:
        state['total_seconds']=time.monotonic()-started
        state['peak_allocated_MiB']=torch.cuda.max_memory_allocated()/2**20
        state['peak_reserved_MiB']=torch.cuda.max_memory_reserved()/2**20
        state['source_unchanged']={f:sha(ROOT/f)==v for f,v in config['source_sha256'].items()}
        write_new(log.parent/'summary.json',state)
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def plan(rid):
    assert re.fullmatch('[A-Za-z0-9_.-]+',rid)
    from importlib import import_module
    import argparse
    from helpers.SeqReader import SeqReader
    from helpers.BaseRunner import BaseRunner
    from models.sequential.LLMMIRecASPCFComiRecReference import LLMMIRecASPCFComiRecReference as Ref
    main=runpy.run_path(str(ROOT/'main.py'),run_name='prepare_only')
    parser=main['parse_global_args'](argparse.ArgumentParser())
    parser=SeqReader.parse_data_args(parser);parser=BaseRunner.parse_runner_args(parser);parser=Ref.parse_model_args(parser)
    source={f:sha(ROOT/f) for f in SOURCES}
    jobs=[]
    for ds in ['beauty','ml-1m']:
        hist=ROOT/'new_log/llmmirec_aspcf_K2'/ds/'seed42/LLMMIRecASPCF_seed42.log'
        old={}
        for line in hist.read_text().splitlines():
            if '|' in line:
                parts=line.split('|');k=parts[0].strip();v=parts[1].strip()
                if k in {a.dest for a in parser._actions}:old[k]=v
        assert old['K']=='2' and old['lr']==('0.004' if ds=='beauty' else '0.001')
        old['l2_float']=float(old['l2'])
        assets={str((ROOT/'data'/ds/x).relative_to(ROOT)):sha(ROOT/'data'/ds/x) for x in ['train.csv','dev.csv','SeqReader.pkl','item2id.json','handled/llm_table_pca1536.pkl']}
        for arm,ext in [('A1','query'),('D1','dr')]:
            log=ROOT/'new_log/comirec_reference_bpr_round1'/rid/ds/arm
            model=ROOT/'new_model/comirec_reference_bpr_round1'/rid/ds/arm
            values={k:v for k,v in old.items() if k!='l2_float' and '...' not in v}
            values.update(gpu='1',dataset=ds,K='2',reference_extractor=ext,reference_scoring='max',dr_iterations='3',
                history_max='20',num_neg='1',lambda_relation='0.01',dropout='0.1',epoch='200',early_stop='10',
                batch_size='1024',eval_batch_size='256',random_seed='42',optimizer='Adam',l2='1e-06',grad_clip='0',
                pin_memory='1',num_workers='5',topk='5,10,20',metric='NDCG,HR',train='1',load='0',
                regenerate='0',dev_only='1',test_epoch='-1',test_all='0',
                llm_emb_path=str(ROOT/'data'/ds/'handled/llm_table_pca1536.pkl'),log_file=str(log/'train.log'),model_path=str(model/'best.pt'))
            cli=[x for k,v in values.items() for x in ['--'+k,str(v)]]
            effective=vars(parser.parse_args(cli))
            assert effective['lr']==(.004 if ds=='beauty' else .001) and effective['l2']==1e-6
            command=[sys.executable,'-B',str(Path(__file__).resolve()),'main','--','--model_name','LLMMIRecASPCFComiRecReference']+cli
            conf=dict(dataset=ds,arm=arm,run_id=rid,historical_log=str(hist),historical_log_sha256=sha(hist),
                historical_config=old,effective_args=effective,command=command,source_sha256=source,asset_sha256=assets,
                test_allowed=False,initialization='seed42 from scratch, no checkpoint load; original shared initialization')
            jobs.append((log,model,conf))
    for log,model,conf in jobs:
        assert not log.exists() and not model.exists()
        assert log.resolve().is_relative_to(ROOT) and model.resolve().is_relative_to(ROOT)
    for log,model,conf in jobs:
        log.mkdir(parents=True,exist_ok=False);model.mkdir(parents=True,exist_ok=False)
        write_new(log/'config.json',conf)
    out=ROOT/'new_log/comirec_reference_bpr_round1'/rid
    write_new(out/'manifest.json',dict(run_id=rid,jobs=[c for _,_,c in jobs],test_allowed=False))
    print(json.dumps(dict(run_id=rid,jobs=[str(l) for l,_,_ in jobs]),indent=2))
def orchestrate(rid):
    root=ROOT/'new_log/comirec_reference_bpr_round1'/rid
    manifest=json.loads((root/'manifest.json').read_text());results=[]
    for job in manifest['jobs']:
        while True:
            row=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
            # No other task is stopped. Wait if it is busy; sequential own runs only.
            if float(row[0])>=7000 and float(row[1])<=60:break
            print('GPU1 unavailable, waiting safely',flush=True);time.sleep(60)
        log=Path(job['effective_args']['log_file']).parent
        assert not (log/'train.log').exists() and not (log/'train.out').exists() and not Path(job['effective_args']['model_path']).exists()
        assert all(sha(ROOT/f)==v for f,v in job['source_sha256'].items()),'Source changed after preparation'
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
        with (log/'train.out').open('xb') as f:
            child=subprocess.Popen(job['command'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
            write_new(log/'process.json',dict(pid=child.pid,started_unix=time.time(),gpu=1))
            code=child.wait()
        result=dict(dataset=job['dataset'],arm=job['arm'],exit_code=code,summary=str(log/'summary.json'))
        results.append(result);write_new(root/(job['dataset']+'_'+job['arm']+'_finished.json'),result)
        if code!=0:
            write_new(root/'blocked.json',dict(status='stopped_after_engineering_failure',results=results,test_executed=False))
            raise RuntimeError(result)
        s=json.loads((log/'summary.json').read_text());assert s['success'] and all(s['source_unchanged'].values())
        print('RUN_COMPLETE '+json.dumps({**result,'dev':s['dev'],'end_epoch':s['end_epoch']}),flush=True)
    for ds in ['beauty','ml-1m']:
        a,d=[json.loads((root/ds/arm/'summary.json').read_text()) for arm in ['A1','D1']]
        n=min(len(a['epochs']),len(d['epochs']))
        checks=dict(shared_item_initialization=a['initial_item_encoder_sha256']==d['initial_item_encoder_sha256'],
                    shared_positions=a['initial_position_sha256']==d['initial_position_sha256'],
                    fixed_train_probe=a['fixed_train_fixture_sha256']==d['fixed_train_fixture_sha256'],
                    initial_rng=a['initial_rng']==d['initial_rng'],
                    common_epoch_negatives=all(x['train_negative_sha256']==y['train_negative_sha256'] for x,y in zip(a['epochs'],d['epochs'])),
                    common_epoch_sample_order=all(x['train_order_sha256']==y['train_order_sha256'] for x,y in zip(a['epochs'],d['epochs'])),
                    common_epoch_rng=all(x['rng_start']==y['rng_start'] and x['rng_after_fit']==y['rng_after_fit'] for x,y in zip(a['epochs'],d['epochs'])))
        write_new(root/(ds+'_fairness.json'),dict(common_epochs=n,checks=checks,note='Architecture/dropout differences reported if RNG diverges; shared initialization mandatory'))
        assert checks['shared_item_initialization'] and checks['shared_positions'] and checks['fixed_train_probe']
    write_new(root/'complete.json',dict(status='complete',results=results,test_executed=False,finished_unix=time.time()))
if __name__=='__main__':
    action=sys.argv[1]
    if action=='main':runtime(sys.argv[2:])
    elif action=='prepare':plan(sys.argv[2])
    elif action=='orchestrate':orchestrate(sys.argv[2])
    else:raise ValueError(action)
