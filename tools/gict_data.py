"""Train-only corpus construction shared by index builder and structural checks."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(2**20), b""): h.update(block)
    return h.hexdigest()


def project_output(path):
    root=Path(__file__).resolve().parents[1]
    p=Path(path).resolve()
    if not p.is_relative_to(root):
        raise ValueError("Output must stay inside the project")
    if p.exists():
        raise FileExistsError(p)
    if not p.parent.is_dir():
        raise ValueError("Create/choose an existing project output directory")
    return p


def train_corpus(args):
    # Deliberately never unpickle SeqReader: it includes held-out frames/histories.
    path=Path(args.path)/args.dataset/"train.csv"
    frame=pd.read_csv(path, sep="\t", usecols=["user_id","item_id","time"])
    ordered=frame.sort_values(["time","user_id"],kind="mergesort")
    positions=ordered.groupby("user_id",sort=False).cumcount()
    frame["position"]=positions.reindex(frame.index)
    asset=json.loads(Path(args.sil_asset_path).read_text())
    c=SimpleNamespace(n_users=int(frame.user_id.max())+1,n_items=int(asset["n_items"]),
        data_df={"train":frame},train_clicked_set={},user_his={})
    for u,g in frame.groupby("user_id",sort=True):
        g=g.sort_values("position",kind="mergesort")
        c.train_clicked_set[int(u)]=set(map(int,g.item_id))
        c.user_his[int(u)]=list(zip(map(int,g.item_id),map(int,g.time)))
    return c


def args_from_config(config):
    x=json.loads(Path(config).read_text())
    a=SimpleNamespace(**x["args"])
    a.device="cpu";a.sil_mode="fixed";a.sil_use_local=1;a.sil_use_recent=1
    a.buffer=0;a.test_all=0
    return a
