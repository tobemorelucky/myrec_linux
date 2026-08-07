#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
LLMMIRec training throughput benchmark.

Measures end-to-end training throughput, data-wait time, GPU forward/backward
time, and GPU memory across different batch_size x num_workers combinations
for the ML-1M item_encoder=id configuration.

Two modes:
  quick (default) — two-phase: sweep batch_size first, then num_workers
  full           — exhaustive grid over all batch_size x num_workers

Usage (via bash wrapper):
  bash new_bash/run_llmmirec_throughput_benchmark.sh           # quick
  bash new_bash/run_llmmirec_throughput_benchmark.sh 1 quick   # quick, physical GPU 1
  bash new_bash/run_llmmirec_throughput_benchmark.sh 0 full    # full, physical GPU 0

Or directly:
  python tools/benchmark_llmmirec_throughput.py --gpu 1 --mode quick
"""

import os
import sys
import gc
import time
import pickle
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from utils import utils
from helpers.SeqReader import SeqReader                             # noqa: E402
from models.sequential.LLMMIRec import LLMMIRec                     # noqa: E402

# ---------------------------------------------------------------------------
# Dataset-specific constants (mirror the formal training commands EXACTLY)
#
# All three datasets share identical LLMMIRec item_encoder=id parameters.
# The only difference is the dataset name (affects data path and corpus).
# ---------------------------------------------------------------------------

VALID_DATASETS = ["ml-1m", "beauty", "toys"]


def get_fixed_args(dataset):
    """Return the fixed arg dict for a given dataset.

    Parameters are identical across ml-1m / beauty / toys for
    LLMMIRec item_encoder=id mode, verified against the phase0 bash scripts.
    """
    if dataset not in VALID_DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Valid: {VALID_DATASETS}"
        )
    return dict(
        model_name="LLMMIRec",
        dataset=dataset,
        path="./data/",
        random_seed=42,
        emb_size=64,
        attn_size=64,
        K=4,
        history_max=20,
        item_encoder="id",
        llm_emb_path="",
        adapter_hidden=256,
        adapter_activation="gelu",
        adapter_use_ln=0,
        gamma_init=0.1,
        gamma_trainable=0,
        dropout=0.1,
        lr=0.001,
        l2=1e-6,
        num_neg=1,
        pin_memory=1,
        # Non-training args needed by model init
        model_path="",
        buffer=1,
        test_all=0,
        epoch=200,
        early_stop=10,
        topk="5,10,20,50",
        metric="NDCG,HR",
    )

# ---------------------------------------------------------------------------
# TSV columns (order-stable)
# ---------------------------------------------------------------------------
COLUMNS = [
    "batch_size",
    "num_workers",
    "warmup_batches",
    "measured_batches",
    "total_seconds",
    "avg_batch_ms",
    "samples_per_second",
    "avg_data_wait_ms",
    "avg_forward_ms",
    "avg_backward_step_ms",
    "peak_allocated_mb",
    "peak_reserved_mb",
    "test_stage",
    "status",
]


# ===========================================================================
#  Helpers
# ===========================================================================

def build_arg_namespace(batch_size, num_workers, device, physical_gpu, dataset):
    """Build an argparse.Namespace with all attributes the model expects."""
    ns = argparse.Namespace(**get_fixed_args(dataset))
    ns.batch_size = batch_size
    ns.num_workers = num_workers
    ns.device = device
    ns.gpu = physical_gpu
    return ns


def build_dataloader(dataset, batch_size, num_workers, pin_memory):
    """Create a fresh DataLoader with drop_last=True."""
    dataset.actions_before_epoch()  # re-sample negatives (mirrors real training)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=dataset.collate_batch,
        pin_memory=bool(pin_memory),
        drop_last=True,
    )


def cleanup_config(model, dl, dl_iter, batch, loss, out_dict, optimizer):
    """Thoroughly clean up after each configuration."""
    for obj in [batch, loss, out_dict, dl_iter]:
        if obj is not None:
            del obj
    if dl is not None:
        del dl
    if optimizer is not None:
        del optimizer
    if model is not None:
        del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# ===========================================================================
#  Single-config benchmark
# ===========================================================================

def run_config(args, batch_size, num_workers, device,
               warmup_batches=20, measured_batches=200, test_stage=""):
    """Benchmark a single (batch_size, num_workers) configuration.

    Returns a dict with all measured metrics.  Only swallows genuine CUDA OOM;
    other RuntimeErrors propagate.
    """
    result = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "warmup_batches": warmup_batches,
        "measured_batches": measured_batches,
        "test_stage": test_stage,
        "status": "OK",
    }

    model = None
    dl = None
    dl_iter = None
    optimizer = None
    batch = None
    loss = None
    out_dict = None

    try:
        # ----------------------------------------------------------------
        # 1. Re-init seed (same init across configs)
        # ----------------------------------------------------------------
        utils.init_seed(args.random_seed)

        # ----------------------------------------------------------------
        # 2. Load pre-built corpus (NOT timed)
        # ----------------------------------------------------------------
        corpus_path = os.path.join(args.path, args.dataset, "SeqReader.pkl")
        with open(corpus_path, "rb") as f:
            corpus = pickle.load(f)

        # ----------------------------------------------------------------
        # 3. Build model & optimizer (NOT timed)
        # ----------------------------------------------------------------
        model = LLMMIRec(args, corpus).to(device)
        optimizer = torch.optim.Adam(
            model.customize_parameters(), lr=args.lr, weight_decay=args.l2
        )
        model.optimizer = optimizer
        model.train()

        # ----------------------------------------------------------------
        # 4. Build dataset & DataLoader (NOT timed)
        # ----------------------------------------------------------------
        dataset = LLMMIRec.Dataset(model, corpus, "train")
        dataset.prepare()
        dl = build_dataloader(
            dataset, batch_size, num_workers, args.pin_memory
        )
        dl_iter = iter(dl)

        # ----------------------------------------------------------------
        # 5. Warmup
        # ----------------------------------------------------------------
        fwd_ev_start = torch.cuda.Event(enable_timing=True)
        fwd_ev_end = torch.cuda.Event(enable_timing=True)
        bwd_ev_start = torch.cuda.Event(enable_timing=True)
        bwd_ev_end = torch.cuda.Event(enable_timing=True)

        for _ in range(warmup_batches):
            try:
                batch = next(dl_iter)
            except StopIteration:
                dl = build_dataloader(
                    dataset, batch_size, num_workers, args.pin_memory
                )
                dl_iter = iter(dl)
                batch = next(dl_iter)

            batch = utils.batch_to_gpu(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out_dict = model(batch)
            loss = model.loss(out_dict)
            loss.backward()
            optimizer.step()

        # ----------------------------------------------------------------
        # 6. Sync + reset peak memory BEFORE measurement
        # ----------------------------------------------------------------
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

        # ----------------------------------------------------------------
        # 7. Measurement
        # ----------------------------------------------------------------
        data_wait_times = []       # seconds (CPU wall clock)
        forward_times = []         # ms (CUDA event)
        backward_step_times = []   # ms (CUDA event)
        total_samples = 0

        measured_count = 0
        total_start = time.perf_counter()
        torch.cuda.synchronize()   # final barrier before timed region

        while measured_count < measured_batches:
            # ---- data loading wall-clock time ----
            t0 = time.perf_counter()
            try:
                batch = next(dl_iter)
            except StopIteration:
                dl = build_dataloader(
                    dataset, batch_size, num_workers, args.pin_memory
                )
                dl_iter = iter(dl)
                batch = next(dl_iter)
            t1 = time.perf_counter()
            data_wait_times.append(t1 - t0)

            batch = utils.batch_to_gpu(batch, device)
            actual_bs = batch["batch_size"]
            total_samples += actual_bs

            optimizer.zero_grad(set_to_none=True)

            # ---- GPU forward timing ----
            fwd_ev_start.record()
            out_dict = model(batch)
            fwd_ev_end.record()

            # ---- GPU backward + step timing ----
            bwd_ev_start.record()
            loss = model.loss(out_dict)
            loss.backward()
            optimizer.step()
            bwd_ev_end.record()

            # One sync per measured batch to read CUDA event times
            torch.cuda.synchronize()
            forward_times.append(fwd_ev_start.elapsed_time(fwd_ev_end))
            backward_step_times.append(
                bwd_ev_start.elapsed_time(bwd_ev_end)
            )
            measured_count += 1

        torch.cuda.synchronize()
        total_end = time.perf_counter()

        # ----------------------------------------------------------------
        # 8. Compute statistics
        # ----------------------------------------------------------------
        total_seconds = total_end - total_start
        avg_batch_ms = (total_seconds / measured_count) * 1000.0
        samples_per_second = total_samples / total_seconds

        avg_data_wait_ms = float(np.mean(data_wait_times)) * 1000.0
        avg_forward_ms = float(np.mean(forward_times))
        avg_backward_step_ms = float(np.mean(backward_step_times))

        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
        peak_rsvd = torch.cuda.max_memory_reserved(device) / (1024.0 ** 2)

        result.update({
            "measured_batches": measured_count,
            "total_seconds": round(total_seconds, 3),
            "avg_batch_ms": round(avg_batch_ms, 2),
            "samples_per_second": round(samples_per_second, 1),
            "avg_data_wait_ms": round(avg_data_wait_ms, 2),
            "avg_forward_ms": round(avg_forward_ms, 2),
            "avg_backward_step_ms": round(avg_backward_step_ms, 2),
            "peak_allocated_mb": round(peak_alloc, 1),
            "peak_reserved_mb": round(peak_rsvd, 1),
        })

    except torch.cuda.OutOfMemoryError:
        result["status"] = "OOM"
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) or "out of memory" in str(e):
            result["status"] = "OOM"
        else:
            raise

    finally:
        cleanup_config(model, dl, dl_iter, batch, loss, out_dict, optimizer)

    return result


# ===========================================================================
#  Quick mode helpers
# ===========================================================================

def _print_config_result(r):
    """Print a single config result to console."""
    print(f"  status             = {r['status']}")
    if r["status"] == "OK":
        print(f"  total_seconds      = {r['total_seconds']}")
        print(f"  avg_batch_ms       = {r['avg_batch_ms']}")
        print(f"  samples_per_second = {r['samples_per_second']}")
        print(f"  avg_data_wait_ms   = {r['avg_data_wait_ms']}")
        print(f"  avg_forward_ms     = {r['avg_forward_ms']}")
        print(f"  avg_backward_step_ms = {r['avg_backward_step_ms']}")
        print(f"  peak_allocated_mb  = {r['peak_allocated_mb']}")
        print(f"  peak_reserved_mb   = {r['peak_reserved_mb']}")


def _best_by_sps(results):
    """Return the result with highest samples_per_second among OK results."""
    ok = [r for r in results if r["status"] == "OK"]
    if not ok:
        return None
    return max(ok, key=lambda r: r["samples_per_second"])


def _oom_configs(results):
    """Return list of (batch_size, num_workers) for OOM results."""
    return [(r["batch_size"], r["num_workers"]) for r in results
            if r["status"] == "OOM"]


def run_quick(args, device):
    """Two-phase quick benchmark.

    Phase 1: sweep batch_size (num_workers=2) → pick best_batch_size.
    Phase 2: sweep num_workers (batch_size=best_batch_size).
    """
    dataset = getattr(args, "dataset", "ml-1m")
    batch_sizes = [256, 512, 1024, 2048, 4096]
    num_workers_list = [0, 2, 4, 5]

    results = []

    # ==================================================================
    # Phase 1 — batch size sweep
    # ==================================================================
    print("\n" + "=" * 60)
    print(f"QUICK MODE — Phase 1: batch size sweep  (num_workers=2, dataset={dataset})")
    print("=" * 60)

    for bs in batch_sizes:
        print()
        print("-" * 60)
        print(f"  batch_size={bs}, num_workers=2")
        print("-" * 60)

        ns = build_arg_namespace(bs, 2, device, args.gpu, dataset)
        r = run_config(ns, bs, 2, device,
                       warmup_batches=10, measured_batches=100,
                       test_stage="batch_sweep")
        results.append(r)
        _print_config_result(r)

    # ---- pick best batch_size ----
    phase1_ok = [r for r in results if r["status"] == "OK"]
    if not phase1_ok:
        print("\n[FATAL] All batch_size configs OOM in Phase 1 — aborting.")
        sys.exit(1)

    best_phase1 = max(phase1_ok, key=lambda r: r["samples_per_second"])
    best_bs = best_phase1["batch_size"]
    print(f"\n>>> Phase 1 best: batch_size={best_bs}  "
          f"(samples_per_second={best_phase1['samples_per_second']})")

    # ==================================================================
    # Phase 2 — num_workers sweep
    # ==================================================================
    print("\n" + "=" * 60)
    print(f"QUICK MODE — Phase 2: num_workers sweep  (batch_size={best_bs})")
    print("=" * 60)

    for nw in num_workers_list:
        # Skip (best_bs, 2) — already measured in Phase 1
        if nw == 2:
            print(f"\n  (batch_size={best_bs}, num_workers=2) — reusing Phase 1 result")
            continue

        print()
        print("-" * 60)
        print(f"  batch_size={best_bs}, num_workers={nw}")
        print("-" * 60)

        ns = build_arg_namespace(best_bs, nw, device, args.gpu, dataset)
        r = run_config(ns, best_bs, nw, device,
                       warmup_batches=10, measured_batches=100,
                       test_stage="worker_sweep")
        results.append(r)
        _print_config_result(r)

    # ==================================================================
    # Console summary
    # ==================================================================
    print("\n" + "=" * 60)
    print("QUICK MODE — Summary")
    print("=" * 60)

    # 1. Phase 1 winner
    print(f"\n1. Best batch_size from Phase 1: {best_bs}")

    # 2. Overall best
    overall_best = _best_by_sps(results)
    if overall_best:
        print(f"\n2. Best overall config: "
              f"batch_size={overall_best['batch_size']}, "
              f"num_workers={overall_best['num_workers']}  "
              f"(samples_per_second={overall_best['samples_per_second']})")

        # 3. Speedup vs baseline (bs=256, nw=5)
        baseline = None
        for r in results:
            if (r["batch_size"] == 256 and r["num_workers"] == 5
                    and r["status"] == "OK"):
                baseline = r
                break

        if baseline is not None:
            speedup = (overall_best["samples_per_second"]
                       / baseline["samples_per_second"])
            print(f"\n3. Speedup vs baseline (bs=256, nw=5): {speedup:.2f}x")
        else:
            print("\n3. Baseline (bs=256, nw=5) not available — "
                  "speedup not computed.")
    else:
        print("\n2. No successful configs.")

    # 4. OOM configs
    ooms = _oom_configs(results)
    if ooms:
        print(f"\n4. OOM configs:")
        for bs, nw in ooms:
            print(f"     batch_size={bs}, num_workers={nw}")
    else:
        print(f"\n4. No OOM configs.")

    # 5. Output file
    print(f"\n5. Results: {args.output}")

    return results


def run_targeted(args, device):
    """Targeted batch_size sweep with fixed num_workers=5.

    Sweeps batch_size, compares each to baseline (bs=256, nw=5),
    and reports the fastest config.
    """
    dataset = getattr(args, "dataset", "ml-1m")
    batch_sizes = [256, 512, 1024, 2048, 4096]
    FIXED_NW = 5

    results = []

    print("\n" + "=" * 60)
    print(f"TARGETED MODE — batch_size sweep  "
          f"(num_workers={FIXED_NW}, dataset={dataset})")
    print("=" * 60)

    for bs in batch_sizes:
        print()
        print("-" * 60)
        print(f"  batch_size={bs}, num_workers={FIXED_NW}")
        print("-" * 60)

        ns = build_arg_namespace(bs, FIXED_NW, device, args.gpu, dataset)
        r = run_config(ns, bs, FIXED_NW, device,
                       warmup_batches=10, measured_batches=100,
                       test_stage="targeted")
        results.append(r)
        _print_config_result(r)

    # ==================================================================
    # Console summary
    # ==================================================================
    print("\n" + "=" * 60)
    print("TARGETED MODE — Summary")
    print("=" * 60)

    # Baseline
    baseline = None
    for r in results:
        if r["batch_size"] == 256 and r["status"] == "OK":
            baseline = r
            break

    # Per-config table
    print(f"\n  {'batch_size':>10}  {'samples/s':>12}  {'speedup':>8}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*8}")
    for r in results:
        if r["status"] == "OK":
            sps = r["samples_per_second"]
            if baseline is not None and baseline["samples_per_second"] > 0:
                su = sps / baseline["samples_per_second"]
                su_str = f"{su:.2f}x"
            else:
                su_str = "N/A"
            print(f"  {r['batch_size']:>10}  {sps:>12.1f}  {su_str:>8}")
        else:
            print(f"  {r['batch_size']:>10}  {'OOM':>12}  {'--':>8}")

    # Best config
    best = _best_by_sps(results)
    if best:
        print(f"\n>>> Best config: batch_size={best['batch_size']}, "
              f"num_workers={best['num_workers']}  "
              f"(samples_per_second={best['samples_per_second']})")
        if (baseline is not None and baseline["samples_per_second"] > 0
                and best["batch_size"] != 256):
            speedup = best["samples_per_second"] / baseline["samples_per_second"]
            print(f"    Speedup vs baseline (bs=256, nw=5): {speedup:.2f}x")
    else:
        print("\n>>> No successful configs.")

    # OOM configs
    ooms = _oom_configs(results)
    if ooms:
        print(f"\nOOM configs:")
        for bs, nw in ooms:
            print(f"  batch_size={bs}, num_workers={nw}")
    else:
        print(f"\nNo OOM configs.")

    print(f"\nResults: {args.output}")

    return results


def run_full(args, device):
    """Full grid benchmark over all batch_size x num_workers combinations."""
    dataset = getattr(args, "dataset", "ml-1m")
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    num_workers_list = [
        int(x.strip()) for x in args.num_workers_list.split(",")
    ]

    results = []

    for bs in batch_sizes:
        for nw in num_workers_list:
            print()
            print("-" * 60)
            print(f"  batch_size={bs}, num_workers={nw}")
            print("-" * 60)

            ns = build_arg_namespace(bs, nw, device, args.gpu, dataset)
            r = run_config(
                ns, bs, nw, device,
                warmup_batches=args.warmup_batches,
                measured_batches=args.measured_batches,
                test_stage="full",
            )
            results.append(r)
            _print_config_result(r)

    return results


# ===========================================================================
#  Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LLMMIRec Throughput Benchmark"
    )
    parser.add_argument(
        "--gpu", type=str, default="0",
        help="Physical GPU ID (sets CUDA_VISIBLE_DEVICES; PyTorch sees cuda:0)"
    )
    parser.add_argument(
        "--mode", type=str, default="quick",
        choices=["quick", "full", "targeted"],
        help="Benchmark mode: quick (two-phase), full (exhaustive grid), "
             "targeted (batch_size sweep at fixed num_workers=5)"
    )
    parser.add_argument(
        "--dataset", type=str, default="ml-1m",
        choices=VALID_DATASETS,
        help="Dataset: ml-1m, beauty, or toys"
    )
    # Full-mode overrides
    parser.add_argument(
        "--batch_sizes", type=str, default="256,512,1024,2048,4096",
    )
    parser.add_argument(
        "--num_workers_list", type=str, default="0,2,4,5",
    )
    parser.add_argument(
        "--warmup_batches", type=int, default=20,
    )
    parser.add_argument(
        "--measured_batches", type=int, default=200,
    )
    parser.add_argument(
        "--output", type=str, default="",
        help="Override output TSV path (auto-generated if empty)"
    )
    bench_args = parser.parse_args()

    # ---- GPU setup ----
    # --gpu is the PHYSICAL GPU ID.  We isolate that GPU via
    # CUDA_VISIBLE_DEVICES so PyTorch always sees it as logical cuda:0.
    os.environ["CUDA_VISIBLE_DEVICES"] = bench_args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Output path ----
    dataset = bench_args.dataset
    if bench_args.output:
        output_path = bench_args.output
    elif bench_args.mode == "quick":
        output_path = f"new_log/llmmirec_throughput/{dataset}_id_seed42_quick.tsv"
    elif bench_args.mode == "full":
        output_path = f"new_log/llmmirec_throughput/{dataset}_id_seed42_full.tsv"
    else:  # targeted
        output_path = f"new_log/llmmirec_throughput/{dataset}_id_seed42_targeted.tsv"
    bench_args.output = output_path  # stash for summary
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ---- Header ----
    print("=" * 60)
    print("LLMMIRec Throughput Benchmark")
    print(f"  Mode                 = {bench_args.mode}")
    print(f"  Dataset              = {dataset}")
    print(f"  Physical GPU         = {bench_args.gpu}")
    print(f"  CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    print(f"  PyTorch device       = {device}")
    if torch.cuda.is_available():
        print(f"  CUDA device name     = {torch.cuda.get_device_name(0)}")
        print(f"  CUDA device count    = {torch.cuda.device_count()}")
        cap = torch.cuda.get_device_capability(0)
        print(f"  CUDA capability      = {cap[0]}.{cap[1]}")
    print(f"  Output               = {output_path}")
    print("=" * 60)

    # ---- Run ----
    if bench_args.mode == "quick":
        results = run_quick(bench_args, device)
    elif bench_args.mode == "full":
        results = run_full(bench_args, device)
    else:
        results = run_targeted(bench_args, device)

    # ---- Write TSV ----
    with open(output_path, "w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for r in results:
            row = [str(r.get(c, "")) for c in COLUMNS]
            f.write("\t".join(row) + "\n")

    print()
    print("=" * 60)
    print(f"Results written to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
