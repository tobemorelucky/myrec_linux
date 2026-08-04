# -*- coding: UTF-8 -*-
"""
plot_target_interest_summary.py
===============================
Clean 1×2 figure for Target-interest Consistency (Section 2.4).

Left  (a): Dumbbell plot — target_gap vs neg_gap per dataset, bootstrap CI.
Right (b): 100% stacked horizontal bars — k* (interest usage) distribution.

Input : analysis_figures/dumps/{beauty,ml-1m,toys}_target_interest_analysis_dump.npz
Output: analysis_figures/figures/target_interest_summary.pdf / .png / .svg
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FONT_SIZE = 6
DATASETS = ["beauty", "ml-1m", "toys"]
DS_LABELS = {"beauty": "Beauty", "ml-1m": "ML-1M", "toys": "Toys"}


def bootstrap_ci(values, n_boot=2000, ci=95.0, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    n = len(values)
    if n < 2:
        m = float(np.mean(values)) if n == 1 else 0.0
        return m, m, m
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = np.mean(sample)
    lo_pct = (100.0 - ci) / 2.0
    hi_pct = 100.0 - lo_pct
    lo = float(np.percentile(means, lo_pct))
    hi = float(np.percentile(means, hi_pct))
    m = float(np.mean(values))
    return m, lo, hi


def set_journal_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(direction="in", width=0.5, labelsize=FONT_SIZE, pad=1.2)
    ax.set_facecolor("white")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dumps_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "analysis_figures", "dumps"))
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.random_seed)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    })

    # ---- Load & compute stats ----
    stats = {}
    for ds in DATASETS:
        p = os.path.join(args.dumps_dir, f"{ds}_target_interest_analysis_dump.npz")
        if not os.path.exists(p):
            print(f"[ERROR] {p} not found")
            sys.exit(1)
        d = np.load(p, allow_pickle=True)

        tg = d["target_gap"].astype(np.float64)
        ng = d["neg_gap"].astype(np.float64)
        ks = d["k_star"]
        K = d["target_affinity"].shape[1]

        t_mean, t_lo, t_hi = bootstrap_ci(tg, rng=rng)
        n_mean, n_lo, n_hi = bootstrap_ci(ng, rng=rng)

        k_dist = {k: int((ks == k).sum()) for k in range(K)}
        k_frac = {k: v / len(ks) for k, v in k_dist.items()}

        stats[ds] = {
            "t_mean": t_mean, "t_lo": t_lo, "t_hi": t_hi,
            "n_mean": n_mean, "n_lo": n_lo, "n_hi": n_hi,
            "K": K, "k_dist": k_dist, "k_frac": k_frac,
        }

        print(f"=== {ds} ===")
        print(f"  target_gap: mean={t_mean:.4f}  95% CI=[{t_lo:.4f}, {t_hi:.4f}]")
        print(f"  neg_gap:    mean={n_mean:.4f}  95% CI=[{n_lo:.4f}, {n_hi:.4f}]")
        for k in range(K):
            print(f"  k*={k}:  {k_dist[k]:6d}  ({k_frac[k]:.1%})")
        print()

    # ---- Figure ----
    FIG_W = 12.5 / 2.54
    FIG_H = 4.8 / 2.54
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(wspace=0.48, left=0.12, right=0.97, top=0.90, bottom=0.16)

    n_ds = len(DATASETS)
    y_positions = np.arange(n_ds)[::-1]  # top-to-bottom

    # ===================================================================
    #  (a) Dumbbell plot — target_gap vs neg_gap
    # ===================================================================
    for i, ds in enumerate(DATASETS):
        s = stats[ds]
        y = y_positions[i]

        # Connecting line
        ax_a.plot([s["t_mean"], s["n_mean"]], [y, y],
                  color="black", linewidth=0.8, zorder=2)

        # Target gap: hollow circle
        ax_a.errorbar(s["t_mean"], y,
                      xerr=[[s["t_mean"] - s["t_lo"]], [s["t_hi"] - s["t_mean"]]],
                      fmt="o", markersize=5, markerfacecolor="white",
                      markeredgecolor="black", markeredgewidth=0.6,
                      color="black", linewidth=0.5, capsize=2.5,
                      label="Target gap" if i == 0 else "", zorder=3)

        # Negative gap: filled square
        ax_a.errorbar(s["n_mean"], y,
                      xerr=[[s["n_mean"] - s["n_lo"]], [s["n_hi"] - s["n_mean"]]],
                      fmt="s", markersize=5, markerfacecolor="black",
                      markeredgecolor="black", markeredgewidth=0.6,
                      color="black", linewidth=0.5, capsize=2.5,
                      label="Negative gap" if i == 0 else "", zorder=3)

    ax_a.set_yticks(y_positions)
    ax_a.set_yticklabels([DS_LABELS[ds] for ds in DATASETS], fontsize=FONT_SIZE)
    ax_a.set_xlabel("Distance gap", fontsize=FONT_SIZE, labelpad=2)
    ax_a.set_title("(a) Target-interest margin", fontsize=FONT_SIZE, pad=3,
                   fontweight="normal")
    set_journal_style(ax_a)

    # Remove left spine ticks since we have named y labels
    ax_a.spines["left"].set_visible(False)
    ax_a.tick_params(left=False)

    ax_a.legend(fontsize=FONT_SIZE, loc="lower right",
                frameon=False, handlelength=1.4, handletextpad=0.4,
                labelspacing=0.2, borderpad=0.2)

    # ===================================================================
    #  (b) 100% stacked horizontal bars — k* distribution
    # ===================================================================
    K = stats[DATASETS[0]]["K"]
    hatches_k = {0: "", 1: "////", 2: "\\\\"}

    bar_h = 0.45
    for i, ds in enumerate(DATASETS):
        s = stats[ds]
        y = y_positions[i]
        left = 0.0
        for k in range(K):
            frac = s["k_frac"].get(k, 0.0)
            if frac > 0.001:
                ax_b.barh(y, frac, bar_h, left=left,
                          color="white", edgecolor="black", linewidth=0.4,
                          hatch=hatches_k.get(k, ""),
                          label=f"I{k+1}" if i == 0 else "",
                          zorder=3)
                # Annotate percentage if wide enough
                if frac > 0.08:
                    ax_b.text(left + frac / 2, y, f"{frac:.0%}",
                              ha="center", va="center", fontsize=FONT_SIZE - 1,
                              color="black")
            left += frac

    ax_b.set_yticks(y_positions)
    ax_b.set_yticklabels([DS_LABELS[ds] for ds in DATASETS], fontsize=FONT_SIZE)
    ax_b.set_xlabel("Proportion", fontsize=FONT_SIZE, labelpad=2)
    ax_b.set_xlim(0, 1)
    ax_b.set_title("(b) Target-interest usage", fontsize=FONT_SIZE, pad=3,
                   fontweight="normal")
    set_journal_style(ax_b)
    ax_b.spines["left"].set_visible(False)
    ax_b.tick_params(left=False)

    ax_b.legend(fontsize=FONT_SIZE, loc="lower right",
                frameon=False, handlelength=1.4, handletextpad=0.4,
                labelspacing=0.2, borderpad=0.2)

    # ---- Save ----
    base = os.path.join(PROJECT_ROOT, "analysis_figures", "figures",
                        "target_interest_summary")
    os.makedirs(os.path.dirname(base), exist_ok=True)

    for ext, dpi in [(".pdf", None), (".png", 600), (".svg", None)]:
        out = base + ext
        try:
            kw = dict(bbox_inches="tight", pad_inches=0.05)
            if dpi:
                kw["dpi"] = dpi
            fig.savefig(out, **kw)
            print(f"Saved {out}")
        except Exception as e:
            print(f"[INFO] {ext} skipped: {e}")

    plt.close(fig)


if __name__ == "__main__":
    main()
