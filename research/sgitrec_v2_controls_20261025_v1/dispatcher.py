"""Ten authorized controlled tasks, shared physical GPU pool, no preemption, no retries."""
import os,sys,json,time,subprocess,traceback,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from gict_data import sha
spec=importlib.util.spec_from_file_location('v2_resource',ROOT/'research/elector_sil_objective_full_20261023_v2/scheduler.py')
resource=importlib.util.module_from_spec(spec);spec.loader.exec_module(resource)
RESOURCE=ROOT/'research/gpu_resource_scheduler_20261022_v1'
PYTHON='/home/yyx/anaconda3/envs/hzg_py10/bin/python'
def write(p,x):
    p=Path(p);assert p.resolve().is_relative_to(ROOT)
    with p.open('x') as f:json.dump(x,f,indent=2)
def verify():
    p=json.loads((OUT/'PLAN.json').read_text())
    assert p['authorized_tasks']==10 and len(p['tasks'])==10 and p['no_extra_tasks']
    for key in ['source_sha256','protected_sha256']:
        for path,h in p[key].items():
            assert Path(path).resolve().is_relative_to(ROOT) and sha(path)==h,path
    for ds,d in p['datasets'].items():
        for cfg in d['configs'].values():assert sha(cfg['path'])==cfg['sha256']
        for g in d['graphs'].values():assert sha(g['path'])==g['sha256']
    for name in ['math','ml-1m_real','beauty_real','ml-1m_seed','beauty_seed','ml-1m_gpu','beauty_gpu','runtime_smoke','final_checks']:
        assert json.loads((OUT/'checks'/(name+'.json')).read_text())['passed']
    return p
def accepted(run):
    r=json.loads((run/'result.json').read_text());m=json.loads((run/'TEST_COMPLETE.json').read_text())
    assert r['train_complete'] and r['Dev_best_selected'] and r['Test_accepted'] and r['Test_passes']==1
    assert m['one_Test_pass'] and sha(run/'result.json')==m['result_sha256']
    assert sha(run/'best.pt')==r['checkpoint_sha256']==m['checkpoint_sha256']
    return dict(best_epoch=r['best']['epoch'],checkpoint_sha256=r['checkpoint_sha256'])
def state(running,pending,complete,failures):
    value=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        state='FAILED' if failures else ('RUNNING' if running else ('WAITING_GPU' if pending else 'COMPLETED')),
        running={name:{k:v for k,v in job.items() if k not in ['process','gpu_fd','task_fd']} for name,job in running.items()},
        pending=[x['name'] for x in pending],completed=complete,failures=failures,poll_seconds=20,
        policy='GPU1 Beauty-first/GPU0 ML-first with work stealing; both GPUs interchangeable; one task per GPU; fail stops pending')
    tmp=OUT/'queue/STATE.tmp'
    with tmp.open('w') as f:json.dump(value,f,indent=2)
    os.replace(tmp,OUT/'queue/STATE.json')
    print(json.dumps(value),flush=True)
def main():
    pool=None;running={};pending=[];complete=[];failures=[]
    try:
        pool=resource.acquire(RESOURCE/'sgitrec_v2_controls_20261025_v1.lock')
        if pool is None:raise RuntimeError('Duplicate dispatcher')
        assert not (OUT/'queue/STARTED.json').exists(),'No restart or duplicate submission'
        plan=verify();pending=plan['tasks'].copy()
        for task in pending:assert not (OUT/'runs'/task['name']).exists()
        write(OUT/'queue/STARTED.json',dict(pid=os.getpid(),authorized=10,tasks=pending))
        while True:
            for name,job in list(running.items()):
                code=job['process'].poll()
                if code is None:continue
                try:
                    if code!=0:raise RuntimeError('exit='+str(code))
                    got=accepted(Path(job['run']));complete.append(name)
                    write(OUT/'queue'/(name+'_ACCEPTED.json'),dict(pid=job['pid'],**got))
                except BaseException as e:
                    error=dict(task=name,error=repr(e));failures.append(error)
                    write(OUT/'queue'/(name+'_FAILURE.json'),dict(**error,traceback=traceback.format_exc(),no_retry=True))
                finally:
                    os.close(job['gpu_fd']);os.close(job['task_fd']);del running[name]
            if not failures:
                inv=resource.observe()
                for gpu in [0,1]:
                    if not pending:break
                    if gpu in [x['GPU'] for x in running.values()]:continue
                    if gpu not in inv or not resource.idle(inv[gpu]):continue
                    lease=resource.acquire(RESOURCE/('gpu%d.lock'%gpu))
                    if lease is None:continue
                    if not resource.idle(resource.observe()[gpu]):os.close(lease);continue
                    preferred='beauty' if gpu==1 else 'ml-1m'
                    task=next((t for t in pending if t['dataset']==preferred),pending[0]);name=task['name'];run=OUT/'runs'/name
                    tf=resource.acquire(RESOURCE/('sgitrec_v2_controls_'+name+'.lock'))
                    if tf is None:os.close(lease);raise RuntimeError('Duplicate task '+name)
                    try:
                        plan=verify();assert not run.exists();ds=plan['datasets'][task['dataset']]
                        write(OUT/'queue'/(name+'_CLAIM.json'),dict(dispatcher=os.getpid(),task=task,GPU=gpu))
                        cmd=['nohup',PYTHON,'-u','-B',str(ROOT/'tools/run_sgitrec_v2_controls.py'),
                            '--config',ds['configs'][str(task['seed'])]['path'],'--graph',ds['graphs'][task['graph_key']]['path'],
                            '--variant',task['variant'],'--seed',str(task['seed']),'--output-dir',str(run),'--gpu',str(gpu),
                            '--gpu-lease-fd',str(lease),'--manifest',str(OUT/'PLAN.json')]
                        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
                        stdout=OUT/'queue'/(name+'.stdout.log')
                        with stdout.open('x') as log:
                            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,
                                stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,pass_fds=(pool,tf,lease))
                        running[name]=dict(pid=child.pid,process=child,GPU=gpu,gpu_fd=lease,task_fd=tf,
                            run=str(run),log=str(run/'train.log'))
                        pending.remove(task)
                        write(OUT/'queue'/(name+'_PID.json'),dict(pid=child.pid,GPU=gpu,command=cmd,stdout=str(stdout)))
                    except BaseException:
                        if not any(j['gpu_fd']==lease for j in running.values()):os.close(lease);os.close(tf)
                        raise
            state(running,pending,complete,failures)
            if not running and (not pending or failures):break
            time.sleep(20)
        write(OUT/'queue'/('FAILED.json' if failures else 'QUEUE_COMPLETE.json'),
            dict(completed=complete,pending=pending,failures=failures,no_more_tasks=True))
    except BaseException as e:
        write(OUT/'queue/DISPATCH_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),
            running={n:j['pid'] for n,j in running.items()},no_kill=True,no_retry=True))
        raise
    finally:
        if pool is not None:os.close(pool)
if __name__=='__main__':main()
