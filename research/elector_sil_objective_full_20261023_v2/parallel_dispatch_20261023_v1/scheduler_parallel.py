"""Adopt the live BPR child; dispatch only untouched SS arms on idle locked GPUs."""
import argparse, fcntl, hashlib, importlib.util, json, os, signal, subprocess, sys, time, traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=HERE.parent
ROOT=Path('/home/yyx/hzgcode/pom2.0').resolve()
assert HERE.is_relative_to(ROOT) and OUT.resolve().is_relative_to(ROOT)
spec=importlib.util.spec_from_file_location('original_dispatch',OUT/'scheduler.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
DS='ml-1m'; OLD=2403913; ADOPT=2414357
ARMS=['SS-C0','SS-M2']
RESOURCE=original.RESOURCE
def dump(path,value):
    path=Path(path)
    assert path.parent.resolve().is_relative_to(ROOT)
    with path.open('x') as f:json.dump(value,f,indent=2)
def read(path):return json.loads(Path(path).read_text())
def proc(pid):
    try:
        stat=Path('/proc/%d/stat'%pid).read_text()
        tail=stat[stat.rfind(')')+2:].split()
        return dict(pid=pid,state=tail[0],ppid=int(tail[1]),start=tail[19],
                    cmd=Path('/proc/%d/cmdline'%pid).read_bytes().replace(b'\0',b' ').decode().strip())
    except (FileNotFoundError,ProcessLookupError):return None
def live(pid,identity):
    p=proc(pid)
    return bool(p and p['start']==identity and p['state'] not in ('Z','X'))
def check_acceptance(folder):
    value=read(folder/'result.json');done=read(folder/'TEST_COMPLETE.json')
    assert value['train_complete'] and value['Dev_best_selected'] and value['Test_accepted']
    assert value['Test_passes']==1 and done['one_Test_pass']
    assert done['result_sha256']==original.sha(folder/'result.json')
    assert done['checkpoint_sha256']==value['checkpoint_sha256']==original.sha(folder/'best.pt')
    assert not (folder/'FAILURE.json').exists()
    return dict(result_sha256=done['result_sha256'],checkpoint_sha256=done['checkpoint_sha256'])
def preconditions():
    original.verify()
    state=read(OUT/DS/'STATE.json')
    assert (state['pid'],state['train_pid'],state['current_arm'],state['GPU'])==(OLD,ADOPT,'BPR-C1',0),state
    parent=proc(OLD);child=proc(ADOPT)
    assert parent and '--dataset ml-1m' in parent['cmd'] and str(OUT/'scheduler.py') in parent['cmd']
    assert child and child['ppid']==OLD and str(OUT/'train_task.py') in child['cmd'] and '--arm BPR-C1' in child['cmd']
    assert os.getsid(ADOPT)==ADOPT and os.getpgid(ADOPT)==ADOPT
    assert read(OUT/DS/'arms/BPR-C1/PID.json')['pid']==ADOPT
    assert not (OUT/DS/'SCHEDULER_FAILURE.json').exists()
    for arm in ARMS:
        folder=OUT/DS/'arms'/arm
        assert folder.resolve().is_relative_to(ROOT) and not any(folder.iterdir()),arm
    resource=original.observe()
    assert resource[0]['pids']==[ADOPT] and original.idle(resource[1]),resource
    # Existing BPR child inherited both old job and GPU0 locks.
    inherited={os.stat(p).st_ino for p in Path('/proc/%d/fd'%ADOPT).iterdir() if p.exists()}
    assert os.stat(RESOURCE/'gpu0.lock').st_ino in inherited
    assert os.stat(RESOURCE/'elector_objective_full_20261023_v2_ml-1m.lock').st_ino in inherited
    return state,parent,child,resource
def try_gpu():
    inventory=original.observe()
    for gpu in (1,0):
        if gpu not in inventory or not original.idle(inventory[gpu]):continue
        fd=original.acquire(RESOURCE/('gpu%d.lock'%gpu))
        if fd is None:continue
        time.sleep(1)
        fresh=original.observe()
        if gpu in fresh and original.idle(fresh[gpu]):return gpu,fd
        os.close(fd)
    return None,None
def main():
    queue_fd=original.acquire(RESOURCE/'elector_objective_full_20261023_v2_ml1m_parallel_v1.lock')
    assert queue_fd is not None,'Duplicate replacement dispatcher'
    running={};failed=[];adopted=None;adopt_identity=None;last=None
    stopped=False;handed=False
    def state(name,**extra):
        nonlocal last
        payload=dict(state=name,pid=os.getpid(),time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                     adopted_BPR_pid=ADOPT,adopted_status=adopted,
                     active_tasks={arm:dict(pid=entry['child'].pid,GPU=entry['gpu']) for arm,entry in running.items()},
                     pending=ARMS[index:],failures=failed,**extra)
        signature=json.dumps({k:v for k,v in payload.items() if k!='time_utc'},sort_keys=True)
        if signature!=last:
            temp=HERE/'STATE.tmp'
            with temp.open('w') as f:json.dump(payload,f,indent=2)
            os.replace(temp,HERE/'STATE.json')
            print(json.dumps(payload),flush=True);last=signature
    def inspect_adopted():
        nonlocal adopted
        if adopted is not None:return
        if live(ADOPT,adopt_identity):return
        folder=OUT/DS/'arms/BPR-C1'
        try:
            evidence=check_acceptance(folder)
            adopted='COMPLETED'
            dump(folder/'exit_status_adopted.json',dict(pid=ADOPT,actual_exit_code=None,
                 acceptance_verified=True,reason='Parent scheduler deliberately replaced; result hashes verified',**evidence))
        except BaseException as exc:
            adopted='FAILED';failed.append('BPR-C1: '+repr(exc))
            dump(HERE/'ADOPTED_FAILURE.json',dict(pid=ADOPT,error=repr(exc),following_tasks_halted=True))
    index=0
    try:
        old_state,parent,adopt_proc,inventory=preconditions()
        adopt_identity=adopt_proc['start']
        dump(HERE/'HANDOFF_PLAN.json',dict(old_scheduler=parent,adopted_train=adopt_proc,
             old_state=old_state,new_scheduler_pid=os.getpid(),pending=ARMS,GPU_preference=[1,0],
             script_sha256=original.sha(Path(__file__)),inventory=inventory,
             model_and_config_unchanged=True,failure_policy='Halt unstarted tasks; never kill ongoing training'))
        # Stop only the non-GPU queue parent briefly to eliminate boundary races.
        os.kill(OLD,signal.SIGSTOP);stopped=True
        for _ in range(40):
            p=proc(OLD)
            if p and p['state'] in ('T','t'):break
            time.sleep(.05)
        assert proc(OLD)['state'] in ('T','t')
        assert live(ADOPT,adopt_identity) and proc(ADOPT)['ppid']==OLD
        for arm in ARMS:assert not any((OUT/DS/'arms'/arm).iterdir())
        # The child has its own session and inherited locks. Never signal its process group.
        os.kill(OLD,signal.SIGTERM);os.kill(OLD,signal.SIGCONT);stopped=False;handed=True
        for _ in range(60):
            if not live(OLD,parent['start']):break
            time.sleep(.05)
        assert not live(OLD,parent['start'])
        assert live(ADOPT,adopt_identity),'Adopted training must remain alive'
        dump(HERE/'HANDOFF_COMPLETE.json',dict(old_scheduler_pid=OLD,old_scheduler_stopped=True,
             adopted_train_pid=ADOPT,adopted_train_untouched=True,new_scheduler_pid=os.getpid()))
        state('HANDOFF_COMPLETE')
        while True:
            inspect_adopted()
            for arm,entry in list(running.items()):
                code=entry['child'].poll()
                if code is None:continue
                folder=OUT/DS/'arms'/arm
                dump(folder/'exit_status.json',dict(code=code,pid=entry['child'].pid,dispatcher_pid=os.getpid()))
                try:
                    assert code==0,'exit='+str(code)
                    check_acceptance(folder)
                except BaseException as exc:failed.append(arm+': '+repr(exc))
                os.close(entry['gpu_fd']);os.close(entry['arm_fd'])
                del running[arm]
            if failed:
                state('FOLLOWING_TASKS_HALTED')
                if not running and adopted is not None:break
            elif index<len(ARMS):
                original.verify()
                gpu,gpu_fd=try_gpu()
                if gpu is None:
                    state('RUNNING' if running else 'WAITING_GPU')
                else:
                    inspect_adopted()
                    if failed:os.close(gpu_fd);continue
                    arm=ARMS[index];folder=OUT/DS/'arms'/arm
                    assert not any(folder.iterdir()),'Existing output; refuse overwrite '+arm
                    arm_fd=original.acquire(RESOURCE/('elector_objective_full_20261023_v2_ml1m_'+arm+'.lock'))
                    assert arm_fd is not None,'Arm already claimed '+arm
                    dump(HERE/(arm+'_CLAIM.json'),dict(pid=os.getpid(),arm=arm,GPU=gpu))
                    env=os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',
                               OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                               CUDA_CACHE_PATH=str(OUT/DS/'cuda_cache'),
                               TORCHINDUCTOR_CACHE_DIR=str(OUT/DS/'torch_cache'))
                    with (folder/'stdout.log').open('x') as log:
                        child=subprocess.Popen(['nohup',original.PYTHON,'-u','-B',str(OUT/'train_task.py'),
                                '--dataset',DS,'--arm',arm],cwd=ROOT,stdin=subprocess.DEVNULL,
                                stdout=log,stderr=subprocess.STDOUT,env=env,close_fds=True,
                                pass_fds=(queue_fd,gpu_fd,arm_fd),start_new_session=True)
                    dump(folder/'PID.json',dict(pid=child.pid))
                    running[arm]=dict(child=child,gpu=gpu,gpu_fd=gpu_fd,arm_fd=arm_fd)
                    index+=1
                    state('RUNNING')
            elif not running and adopted is not None:
                state('COMPLETED');break
            elif not running:
                state('WAITING_ADOPTED_TRAINING')
            else:
                state('RUNNING')
            time.sleep(10)
        if failed:raise RuntimeError('; '.join(failed))
    except BaseException as exc:
        if stopped:os.kill(OLD,signal.SIGCONT)
        dump(HERE/'SCHEDULER_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),
             following_tasks_halted=True,handoff_completed=handed,active_train_pids=[entry['child'].pid for entry in running.values()]))
        raise
    finally:
        for entry in running.values():
            os.close(entry['gpu_fd']);os.close(entry['arm_fd'])
        os.close(queue_fd)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if args.check:
        state,parent,child,inventory=preconditions()
        compile(Path(__file__).read_text(),str(__file__),'exec')
        # Resource selection: idle fallback, lock contention, and post-lock recheck.
        real_observe,real_acquire,real_sleep,real_close=original.observe,original.acquire,time.sleep,os.close
        base={0:dict(pids=[ADOPT],free=15000,util=25),1:dict(pids=[],free=16000,util=0)}
        observations=[]
        def exercise(sequence,locked=False):
            calls=iter(sequence);closed=[]
            original.observe=lambda:next(calls)
            original.acquire=lambda _:None if locked else 987654
            time.sleep=lambda _:None
            os.close=lambda fd:closed.append(fd)
            return try_gpu(),closed
        try:
            assert exercise([base,base])[0]==(1,987654)
            assert exercise([base],locked=True)[0]==(None,None)
            busy={0:base[0],1:dict(pids=[123],free=15000,util=1)}
            assert exercise([base,busy])==((None,None),[987654])
            fallback={0:dict(pids=[],free=16000,util=0),1:busy[1]}
            assert exercise([fallback,fallback])[0]==(0,987654)
        finally:
            original.observe,original.acquire,time.sleep,os.close=real_observe,real_acquire,real_sleep,real_close
        print(json.dumps(dict(passed=True,resource_tests=4,parent=parent,adopted=child,inventory=inventory,
             unstarted=ARMS,no_model_change=True,original_sources_verified=True)))
    else:main()