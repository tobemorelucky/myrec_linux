# -*- coding: UTF-8 -*-
"""
plot_target_interest_analysis.py
=================================
Read target_interest_analysis_dump.npz for all 3 datasets and produce a
1×2 black-and-white journal figure.

Left  (a): Beauty target-interest affinity heatmap  (sorted by k*)
Right (b): target_gap / neg_gap comparison for 3 datasets with bootstrap CI

Output: analysis_figures/figures/target_interest_analysis.pdf / .png / .svg
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


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def set_journal_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(direction="in", width=0.5, labelsize=FONT_SIZE, pad=1.2)
    ax.set_facecolor("white")


def bootstrap_ci(values, n_boot=2000, ci=95.0, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    n = len(values)
    if n < 2:
        m = float(np.mean(values)) if n == 1 else 0.0
        return m, m
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = np.mean(sample)
    lo = (100.0 - ci) / 2.0
    hi = 100.0 - lo
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))


def select_affinity_samples(target_affinity, k_star, n_select=100):
    """
    Select n_select samples with clearest target-interest response
    (largest gap between max and second-max affinity), sorted by k_star.
    """
    N, K = target_affinity.shape
    if N <= n_select:
        idx = np.arange(N)
    else:
        # Affinity margin: max - second_max
        sorted_aff = np.sort(target_affinity, axis=1)[:, ::-1]
        margin = sorted_aff[:, 0] - sorted_aff[:, 1]
        idx = np.argsort(margin)[::-1][:n_select]

    # Sort selected by k_star, then within each k* by max affinity descending
    ks = k_star[idx]
    max_aff = target_affinity[idx].max(axis=1)
    order = np.lexsort((-max_aff, ks))
    return idx[order]


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot target-interest analysis")
    parser.add_argument("--dumps_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "analysis_figures", "dumps"))
    parser.add_argument("--output_pdf", type=str, default="")
    parser.add_argument("--output_png", type=str, default="")
    parser.add_argument("--n_heatmap", type=int, default=100)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.random_seed)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    })

    # ---- Load ----
    all_data = {}
    for ds in DATASETS:
        p = os.path.join(args.dumps_dir, f"{ds}_target_interest_analysis_dump.npz")
        if not os.path.exists(p):
            print(f"[WARN] {p} not found, skipping {ds}")
            continue
        all_data[ds] = np.load(p, allow_pickle=True)

    # ---- Stats & bootstrap ----
    stats = {}
    for ds in DATASETS:
        d = all_data[ds]
        tg = d["target_gap"].astype(np.float64)
        t_lo, t_hi = bootstrap_ci(tg, rng=rng)
        t_mean = float(tg.mean())

        has_neg = "neg_gap" in d.files
        if has_neg:
            ng = d["neg_gap"].astype(np.float64)
            n_lo, n_hi = bootstrap_ci(ng, rng=rng)
            n_mean = float(ng.mean())
        else:
            n_lo, n_hi, n_mean = None, None, None

        ks = d["k_star"]
        K = d["target_affinity"].shape[1]
        k_dist = {k: int((ks == k).sum()) for k in range(K)}

        stats[ds] = {
            "target_mean": t_mean, "target_lo": t_lo, "target_hi": t_hi,
            "neg_mean": n_mean, "neg_lo": n_lo, "neg_hi": n_hi,
            "has_neg": has_neg, "k_dist": k_dist, "N": len(tg),
        }

        print(f"=== {ds} ===")
        print(f"  N={len(tg)}, K={K}, k* dist={k_dist}")
        print(f"  target_gap: mean={t_mean:.4f}  CI=[{t_lo:.4f}, {t_hi:.4f}]")
        if has_neg:
            print(f"  neg_gap:    mean={n_mean:.4f}  CI=[{n_lo:.4f}, {n_hi:.4f}]")
        else:
            print(f"  neg_gap:    unavailable")
        print()

    # ---- Figure ----
    FIG_W = 13.2 / 2.54
    FIG_H = 5.7 / 2.54
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(wspace=0.52, left=0.09, right=0.98, top=0.88, bottom=0.20)

    # ===================================================================
    #  (a) Beauty target-interest affinity heatmap
    # ===================================================================
    bd = all_data["beauty"]
    aff = bd["target_affinity"]
    ks = bd["k_star"]
    K = aff.shape[1]

    idx = select_affinity_samples(aff, ks, n_select=args.n_heatmap)
    aff_sel = aff[idx]
    ks_sel = ks[idx]

    im = ax_a.imshow(
        aff_sel, aspect="auto", cmap="Greys",
        origin="upper", vmin=0.0, vmax=1.0,
        interpolation="nearest",
        extent=[0.5, K + 0.5, len(idx), 0],
    )

    ax_a.set_xlabel("Interest ID", fontsize=FONT_SIZE, labelpad=2)
    ax_a.set_ylabel("Test samples", fontsize=FONT_SIZE, labelpad=2)
    ax_a.set_title("(a) Target-interest affinity", fontsize=FONT_SIZE, pad=3, fontweight="normal")
    set_journal_style(ax_a)
    ax_a.set_xticks(range(1, K + 1))
    ax_a.set_xticklabels([f"I{i}" for i in range(1, K + 1)], fontsize=FONT_SIZE)
    ax_a.set_yticks([])

    cbar = fig.colorbar(im, ax=ax_a, fraction=0.035, pad=0.03, ticks=[0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=FONT_SIZE, width=0.5, pad=1)
    cbar.outline.set_linewidth(0.5)
    cbar.set_label("Affinity", fontsize=FONT_SIZE, labelpad=1)

    # ===================================================================
    #  (b) Target / Negative gap (3 datasets)
    # ===================================================================
    n_ds = len(DATASETS)
    x = np.arange(n_ds)
    bar_w = 0.28

    # Target gap
    t_means = [stats[ds]["target_mean"] for ds in DATASETS]
    t_err_lo = [stats[ds]["target_mean"] - stats[ds]["target_lo"] for ds in DATASETS]
    t_err_hi = [stats[ds]["target_hi"] - stats[ds]["target_mean"] for ds in DATASETS]

    ax_b.bar(x - bar_w * 0.48, t_means, bar_w * 0.88,
             yerr=[t_err_lo, t_err_hi],
             color="white", edgecolor="black", linewidth=0.4, hatch="///",
             error_kw=dict(linewidth=0.4, capsize=2.2),
             label="Target gap", zorder=3)

    # Negative gap
    all_have_neg = all(stats[ds]["has_neg"] for ds in DATASETS)
    if all_have_neg:
        n_means = [stats[ds]["neg_mean"] for ds in DATASETS]
        n_err_lo = [stats[ds]["neg_mean"] - stats[ds]["neg_lo"] for ds in DATASETS]
        n_err_hi = [stats[ds]["neg_hi"] - stats[ds]["neg_mean"] for ds in DATASETS]

        ax_b.bar(x + bar_w * 0.48, n_means, bar_w * 0.88,
                 yerr=[n_err_lo, n_err_hi],
                 color="white", edgecolor="black", linewidth=0.4, hatch="\\\\",
                 error_kw=dict(linewidth=0.4, capsize=2.2),
                 label="Negative gap", zorder=3)

    # Annotate mean values above bars
    for i, ds in enumerate(DATASETS):
        tm = stats[ds]["target_mean"]
        t_hi = stats[ds]["target_hi"]
        ax_b.annotate(f"{tm:.3f}", xy=(x[i] - bar_w * 0.48, t_hi + 0.01),
                      ha="center", va="bottom", fontsize=FONT_SIZE - 1, color="black")
        if all_have_neg:
            nm = stats[ds]["neg_mean"]
            n_hi = stats[ds]["neg_hi"]
            ax_b.annotate(f"{nm:.3f}", xy=(x[i] + bar_w * 0.48, n_hi + 0.01),
                          ha="center", va="bottom", fontsize=FONT_SIZE - 1, color="black")

    # y-axis auto-scale
    y_top = 0.0
    for ds in DATASETS:
        y_top = max(y_top, stats[ds]["target_hi"])
        if stats[ds]["has_neg"]:
            y_top = max(y_top, stats[ds]["neg_hi"])
    ax_b.set_ylim(None, y_top + 0.08)

    ax_b.set_xticks(x)
    ax_b.set_xticklabels([DS_LABELS[ds] for ds in DATASETS], fontsize=FONT_SIZE)
    ax_b.set_ylabel("Distance gap", fontsize=FONT_SIZE, labelpad=2)
    set_journal_style(ax_b)
    ax_b.set_title("(b) Target-interest margin", fontsize=FONT_SIZE, pad=3, fontweight="normal")

    # Legend at figure bottom center
    fig.legend(fontsize=FONT_SIZE, loc="lower center",
               bbox_to_anchor=(0.72, 0.01), ncol=2,
               frameon=False,
               handlelength=1.4, handletextpad=0.4,
               labelspacing=0.2, columnspacing=0.8)

    # ---- Save ----
    base = os.path.join(PROJECT_ROOT, "analysis_figures", "figures",
                        "target_interest_analysis")
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
