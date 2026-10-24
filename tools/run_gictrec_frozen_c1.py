"""Explicit entry for the authorized ML A1 K3 frozen-C1 three-arm experiment."""
import runpy,sys
from pathlib import Path
path=Path(__file__).resolve().parents[1]/'research/gictrec_frozen_c1_20261024_v1/train_task.py'
sys.path.insert(0,str(path.parent))
runpy.run_path(str(path),run_name='__main__')
