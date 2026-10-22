import pathlib,importlib.util,inspect,sys,subprocess
R=pathlib.Path('/home/yyx/hzgcode/pom2.0');W=R/'research/ml1m_embedding_v2_20261020'
s=importlib.util.spec_from_file_location('existing_a1_scheduler',W/'ops/a1_history_bpr_20261021_night/gpu_scheduler.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
source=inspect.getsource(m.main)
a="q=U/('queue_gpu'+str(gpu));q.mkdir(exist_ok=False)"
b="q=U/'queue_gpu1_recovery_20261022';q.mkdir(exist_ok=False)"
c="ids=['E5','E6'] if gpu==0 else ['E1','E3','E4','E2']"
assert source.count(a)==source.count(c)==1
source=source.replace(a,b).replace(c,"ids=['E5','E6']")
exec(compile(source,__file__,'exec'),m.__dict__)
assert sys.argv[1:]==['--gpu','1']
try:m.main()
finally:
 p=W/'ops/recovery_v3_hsdir_20261022/consolidate.py'
 if p.exists():subprocess.run([sys.executable,'-B',str(p)],cwd=R,timeout=60)
