"""Final immutable assets, Dev-only entry and exact historical A/B seed0 verification."""
import sys,signal,json,ast,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from gict_data import sha
import citer_dev_protocol as protocol
signal.alarm(170);tick=time.perf_counter();protocol.install_guard()
before=json.load(open(OUT/'BEFORE.json'))
for p,h in before['protected_sha256'].items():assert sha(p)==h,p
old=json.load(open(ROOT/'research/sgitrec_v3_predictions_20261025_v1/PLAN.json'))
prior=json.load(open(ROOT/'research/sgitrec_v2_controls_20261025_v1/PLAN.json'))
for ds in ['beauty','ml-1m']:
 check=json.load(open(OUT/'checks'/(ds+'_real.json')))
 for seed in ['0','42']:assert check['initial_C1_hash'][seed]==prior['datasets'][ds]['initial_sha256'][seed]
 diag=json.load(open(OUT/'analysis'/(ds+'_DEV_diagnosis.json')))
 assert diag['sample_complementarity'] and diag['Test'] is False
 for label,ref in diag['checkpoints'].items():assert sha(ref['path'])==ref['sha256']
 assert sha(diag['arrays'])==diag['arrays_sha256']
 assert diag['graph_sha256']==old['datasets'][ds]['graphs']['predictions']['sha256']
# New runner has no Test reader/evaluator path at all; the runtime audit guard is additional.
source=(ROOT/'tools/run_citerrec_dev.py').read_text();tree=ast.parse(source)
assert 'protocol.heldout' not in source and 'test_user_ranks.csv' in source
calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in ['heldout','protocol_lock'] for n in calls)
assert "protocol.load_dev(c,a)" in source and "protocol.install_guard()" in source
assert "data={s:m.Dataset(m,c,s) for s in ('train','dev')}" in source
assert "'DEV_COMPLETE.json'" in source and "'TEST_STARTED.json'" not in source
# Existing seed0 A/B full runs, if any, would block duplicate launch rather than silently rerun.
existing=[]
for pattern in ['*/runs/*s0*TargetInterest*','*/runs/*s0*Direct-Fixed*']:
 existing += [str(p) for p in (ROOT/'research').glob(pattern) if (p/'result.json').exists()]
assert not existing,existing
for name in ['math','ml-1m_real','beauty_real','runtime_smoke']:assert json.load(open(OUT/'checks'/(name+'.json')))['passed']
source_files=['models/sequential/CITERRec.py','models/sequential/citer_components.py','tools/citer_dev_protocol.py','tools/diagnose_citer_dev.py','tools/run_citerrec_dev.py','tools/test_citerrec.py',
 'new_bash/run_citerrec_dev.sh','research/citerrec_v1_20261026_v1/dispatcher.py','research/citerrec_v1_20261026_v1/runtime_smoke.py','research/citerrec_v1_20261026_v1/final_checks.py']
for p in source_files:
 if p.endswith('.py'):ast.parse((ROOT/p).read_text())
result=dict(passed=True,protected_SHA_unchanged=len(before['protected_sha256']),fresh_initial_seed0_seed42_exact=True,
 exact_legacy_AB_reused_for_replications=True,existing_duplicate_seed0_AB=existing,dev_only_entry_has_no_Test_reader=True,
 Dev_guard_regression_passed=True,complementarity_gate_passed_both_samples=True,
 historical_single_branch_note='Reliability disabled gives exact A/B; enabled gates define new controlled experts and are not claimed identical to historical A/B.',
 engineering_fix='GPU fixed-expert reduction aligned with old B candidate chunk order; resolved max3.7e-9 numerical difference before any formal training.',
 risks=['Interest gate ranking gradients initially tiny; record every epoch, no automatic learning-rate or loss searches.','Sample strata are exploratory and contain independently trained backbone confounding.'],
 source_sha256={str(ROOT/p):sha(ROOT/p) for p in source_files},seconds=time.perf_counter()-tick)
with (OUT/'checks/final_checks.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
