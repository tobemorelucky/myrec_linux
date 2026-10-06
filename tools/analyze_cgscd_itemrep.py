# -*- coding: UTF-8 -*-
"""
Diagnose the item representation produced by CGSCD vs ASPCF.

Motivation: Chapter 3's first round showed a "+@5 / -@20" pattern — CGSCD
improves top-5 ranking but degrades recall at larger K. This tool checks the
most likely mechanical causes:

  1. gate collapse      (alpha_sem ~ 0 or ~ 1 for nearly all items)
  2. embedding scale    (norm distribution / outliers)
  3. representation
     collapse           (mean pairwise cosine, effective rank)
  4. branch imbalance   (norm of the shared vs complement half)

The training log truncates arg values to 20 chars, so this tool takes the model
config explicitly. Defaults match new_bash/run_llmmirec_cgscd_phase3.sh.

Usage:
  python tools/analyze_cgscd_itemrep.py \
      --model_name LLMMIRecCGSCD --dataset beauty \
      --checkpoint new_model/llmmirec_cgscd/beauty/crosscov_rel1/seed42/LLMMIRecCGSCD_seed42.pt
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sequential.LLMMIRecCGSCD import LLMMIRecCGSCD   # noqa: E402
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF   # noqa: E402
from torch.utils.data import DataLoader           # noqa: E402
import pickle                                     # noqa: E402

MODELS = {"LLMMIRecCGSCD": LLMMIRecCGSCD, "LLMMIRecASPCF": LLMMIRecASPCF}


def pct(x, q):
    return float(np.percentile(x, q))


def summarize(name, a):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    if a.size == 0:
        return {}
    return {
        f"{name}_mean": float(a.mean()), f"{name}_std": float(a.std()),
        f"{name}_min": float(a.min()), f"{name}_p05": pct(a, 5),
        f"{name}_p50": pct(a, 50), f"{name}_p95": pct(a, 95),
        f"{name}_max": float(a.max()),
    }


def eff_rank(X):
    """Entropy-based effective rank of a [N, D] matrix (after centering)."""
    X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    p = s ** 2
    tot = p.sum()
    if tot <= 0:
        return float("nan")
    p = p / tot
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True, choices=list(MODELS))
    ap.add_argument("--dataset", type=str, default="beauty")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--phase", type=str, default="test", choices=["train", "dev", "test"])
    ap.add_argument("--max_batches", type=int, default=30)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output_dir", type=str, default="./diagnostics_cgscd")
    # model config (defaults match run_llmmirec_cgscd_phase3.sh)
    ap.add_argument("--emb_size", type=int, default=64)
    ap.add_argument("--attn_size", type=int, default=64)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--history_max", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--semantic_rank", type=int, default=512)
    ap.add_argument("--semantic_dim", type=int, default=32)
    ap.add_argument("--complement_dim", type=int, default=32)
    ap.add_argument("--cgscd_basis_path", type=str, default="")
    ap.add_argument("--cgscd_cf_path", type=str, default="")
    ap.add_argument("--cgscd_shared_dim", type=int, default=32)
    ap.add_argument("--cgscd_compl_dim", type=int, default=32)
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    a = args
    a.llm_emb_path = f"./data/{a.dataset}/handled/llm_table_pca1536.pkl"
    if not a.cgscd_basis_path:
        # prefer crosscov; caller should override when analysing a CCA run
        cand = [f"./data/{a.dataset}/handled/cgscd_basis_crosscov_r32.pkl",
                f"./data/{a.dataset}/handled/cgscd_basis_cca_r32.pkl"]
        a.cgscd_basis_path = next((c for c in cand if os.path.exists(c)), "")
    if not a.cgscd_cf_path:
        a.cgscd_cf_path = f"./data/{a.dataset}/handled/itm_emb_pomrec.pkl"

    # fields required by BaseModel / GeneralModel / SequentialModel but not model-specific
    a.device = torch.device(a.device)
    a.model_path = a.checkpoint
    a.buffer = 1          # eval iteration reads buffer_dict -> ds.prepare() required
    a.num_neg = 1
    a.test_all = 0
    a.dropout = 0.1
    for k in ("lambda_relation", "relation_sample_size",
              "relation_teacher_temp", "relation_student_temp"):
        setattr(a, k, 0.0)
    # ASPCF-only fields fall back to their parse_model_args defaults via getattr
    a.aspcf_gate_mode = "basic"
    a.semantic_hidden = 128
    a.tail_hidden = 64
    a.complement_hidden = 64
    a.gate_hidden = 64
    a.adapter_hidden = 256
    a.adapter_activation = "gelu"
    a.adapter_use_ln = 0
    a.gamma_init = 0.1
    a.gamma_trainable = 0
    a.item_encoder = "cgscd" if a.model_name == "LLMMIRecCGSCD" else "aspcf"

    corpus = pickle.load(open(f"./data/{a.dataset}/SeqReader.pkl", "rb"))
    model = MODELS[a.model_name](a, corpus)
    state = torch.load(a.checkpoint, map_location="cpu")
    if isinstance(state, dict):
        for w in ("state_dict", "model", "net"):
            if w in state and isinstance(state[w], dict):
                state = state[w]
                break
    missing, unexpected = model.load_state_dict(state, strict=False)
    model = model.to(a.device)
    model.device = a.device
    model.eval()

    print(f"[diag] {args.model_name} on {args.dataset}")
    print(f"[diag] loaded ckpt: {args.checkpoint}")
    if missing:
        print(f"[diag] MISSING keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"[diag] unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    n_loaded = len(state)
    if n_loaded < 10:
        raise RuntimeError(f"checkpoint only had {n_loaded} entries — wrong file?")

    ds = model.Dataset(model, corpus, a.phase)
    ds.prepare()                       # mandatory: buffer=1 reads buffer_dict
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                        num_workers=0, pin_memory=False, collate_fn=ds.collate_batch)

    acc = {"alpha_sem": [], "emb_norm": [], "sem_norm": [], "comp_norm": [],
           "hist_emb": [], "cos_sem_comp": []}
    n = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model(batch, return_intermediate=True)
            hist = batch["history_items"] > 0
            alpha = out["history_alpha_sem"][hist]
            sem = out["history_semantic"][hist]
            comp = out["history_complement"][hist]
            emb = out["history_vectors"][hist]
            acc["alpha_sem"].append(alpha.cpu().numpy())
            acc["sem_norm"].append(sem.norm(dim=-1).cpu().numpy())
            acc["comp_norm"].append(comp.norm(dim=-1).cpu().numpy())
            acc["emb_norm"].append(emb.norm(dim=-1).cpu().numpy())
            cs = torch.nn.functional.cosine_similarity(sem, comp, dim=-1)
            acc["cos_sem_comp"].append(cs.cpu().numpy())
            if len(acc["hist_emb"]) < 4000:
                acc["hist_emb"].append(emb.cpu().numpy().reshape(-1, emb.size(-1)))
            n += 1
            if args.max_batches and n >= args.max_batches:
                break

    stats = {}
    stats.update(summarize("alpha_sem", np.concatenate(acc["alpha_sem"])))
    stats.update(summarize("emb_norm", np.concatenate(acc["emb_norm"])))
    stats.update(summarize("sem_norm", np.concatenate(acc["sem_norm"])))
    stats.update(summarize("comp_norm", np.concatenate(acc["comp_norm"])))
    stats.update(summarize("cos_sem_comp", np.concatenate(acc["cos_sem_comp"])))
    stats["alpha_frac_below_0.05"] = float((np.concatenate(acc["alpha_sem"]) < 0.05).mean())
    stats["alpha_frac_above_0.95"] = float((np.concatenate(acc["alpha_sem"]) > 0.95).mean())
    stats["branch_norm_ratio"] = stats["comp_norm_mean"] / max(stats["sem_norm_mean"], 1e-12)

    H = np.concatenate(acc["hist_emb"])[:4000]
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
    sub = Hn[: min(1200, len(Hn))]
    C = sub @ sub.T
    off = C[~np.eye(len(sub), dtype=bool)]
    stats["mean_pairwise_cos"] = float(off.mean())
    stats["eff_rank"] = eff_rank(H)
    stats["n_emb"] = int(len(H))

    os.makedirs(args.output_dir, exist_ok=True)
    # Checkpoints from different runs share both the basename (LLMMIRecCGSCD_seed42.pt)
    # AND the immediate parent (seed42), so the tag must include two directory levels
    # (<experiment_label>/seed<N>) or one run's diagnostics silently overwrite another's.
    d = os.path.abspath(os.path.dirname(args.checkpoint))
    two_levels = "_".join(os.path.basename(os.path.dirname(d)).split(os.sep)) + \
                 "_" + os.path.basename(d)
    tag = f"{args.model_name}_{args.dataset}_{two_levels}"
    out_path = os.path.join(args.output_dir, f"{tag}.json")
    json.dump(stats, open(out_path, "w"), indent=2)

    print(json.dumps(stats, indent=2))
    print(f"[diag] saved: {out_path}")


if __name__ == "__main__":
    main()
