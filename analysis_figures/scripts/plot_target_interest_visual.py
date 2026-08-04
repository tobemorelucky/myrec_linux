# -*- coding: UTF-8 -*-
"""
plot_target_interest_visual.py
===============================
Advanced target-interest consistency visualization.

Left  (a): Target-interest affinity map  — stratified by k*, row-normalized,
           with group separators.
Right (b): Target-interest specificity  — violin/box plot per dataset.

Input : analysis_figures/dumps/{beauty,ml-1m,toys}_target_interest_analysis_dump.npz
Output: analysis_figures/figures/target_interest_visual.pdf / .png / .svg
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FONT_SIZE = 6
DATASETS = ["beauty", "ml-1m", "toys"]
DS_LABELS = {"beauty": "Beauty", "ml-1m": "ML-1M", "toys": "Toys"}
HATCHES_DS = {"beauty": "", "ml-1m": "//", "toys": "\\\\"}


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


def stratified_sample_by_kstar(target_affinity, k_star, total_n=120, rng=None):
    """
    Stratified sampling: ~equal samples per k* group, sorted by specificity
    (max - second_max) descending within each group.
    Returns indices in display order (grouped by k*).
    """
    if rng is None:
        rng = np.random.RandomState(42)
    N, K = target_affinity.shape
    sorted_aff = np.sort(target_affinity, axis=1)[:, ::-1]
    specificity = sorted_aff[:, 0] - sorted_aff[:, 1]

    unique_ks = np.unique(k_star)
    per_group = max(1, total_n // len(unique_ks))

    selected = []
    group_boundaries = []  # cumulative counts between groups
    cum = 0
    for k in unique_ks:
        mask = k_star == k
        idx_k = np.where(mask)[0]
        # Sort by specificity descending
        idx_sorted = idx_k[np.argsort(specificity[idx_k])[::-1]]
        take = min(per_group, len(idx_sorted))
        selected.append(idx_sorted[:take])
        cum += take
        group_boundaries.append(cum)

    idx = np.concatenate(selected).astype(np.int64)
    # Remove last boundary (it's the total count)
    boundaries = group_boundaries[:-1]
    return idx, boundaries


def minmax_norm_rows(X):
    """Min-max normalize each row to [0,1]."""
    X = X.astype(np.float64)
    xmin = X.min(axis=1, keepdims=True)
    xmax = X.max(axis=1, keepdims=True)
    denom = xmax - xmin
    denom[denom < 1e-12] = 1.0
    return (X - xmin) / denom


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Target-interest visual figure")
    parser.add_argument("--dumps_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "analysis_figures", "dumps"))
    parser.add_argument("--heatmap_dataset", type=str, default="beauty",
                        choices=["beauty", "ml-1m", "toys"])
    parser.add_argument("--output_pdf", type=str, default="")
    parser.add_argument("--output_png", type=str, default="")
    parser.add_argument("--n_heatmap", type=int, default=120)
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

    # ---- Compute specificity per dataset ----
    spec_stats = {}
    for ds in DATASETS:
        d = all_data[ds]
        aff = d["target_affinity"].astype(np.float64)
        ks = d["k_star"]
        K = aff.shape[1]

        # specificity = max - second_max
        sorted_aff = np.sort(aff, axis=1)[:, ::-1]
        spec = sorted_aff[:, 0] - sorted_aff[:, 1]

        q25, q50, q75 = np.percentile(spec, [25, 50, 75])
        spec_stats[ds] = {
            "specificity": spec,
            "mean": float(np.mean(spec)),
            "median": float(q50),
            "std": float(np.std(spec)),
            "q25": float(q25),
            "q75": float(q75),
            "K": K,
            "k_dist": {k: int((ks == k).sum()) for k in range(K)},
        }
        print(f"=== {ds} ===")
        print(f"  K={K}, k* dist={spec_stats[ds]['k_dist']}")
        print(f"  specificity mean={spec_stats[ds]['mean']:.4f}  median={spec_stats[ds]['median']:.4f}")
        print(f"  specificity std={spec_stats[ds]['std']:.4f}  Q25={q25:.4f}  Q75={q75:.4f}")
        print()

    # ---- Figure ----
    FIG_W = 12.5 / 2.54
    FIG_H = 4.6 / 2.54
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(wspace=0.48, left=0.10, right=0.97, top=0.88, bottom=0.18)

    # ===================================================================
    #  (a) Target-interest affinity map
    # ===================================================================
    hds = args.heatmap_dataset
    hd = all_data[hds]
    aff = hd["target_affinity"].astype(np.float64)
    ks = hd["k_star"]
    K = aff.shape[1]

    idx, boundaries = stratified_sample_by_kstar(aff, ks, total_n=args.n_heatmap, rng=rng)
    aff_sel = aff[idx]
    K_selected = len(idx)

    # Row-normalize for visualization
    aff_disp = minmax_norm_rows(aff_sel)
    print(f"[Heatmap] dataset={hds}, {K_selected} samples, "
          f"K={K}, group boundaries at {boundaries}")
    print(f"[Heatmap] Values are row-normalized (min-max) for visualization only.")

    im = ax_a.imshow(
        aff_disp, aspect="auto", cmap="Greys",
        origin="upper", vmin=0.0, vmax=1.0,
        interpolation="nearest",
        extent=[0.5, K + 0.5, K_selected, 0],
    )

    # Group separators: thin horizontal lines at boundaries
    for b in boundaries:
        ax_a.axhline(y=b, color="black", linewidth=0.3, linestyle="-")

    ax_a.set_xlabel("Interest ID", fontsize=FONT_SIZE, labelpad=2)
    ax_a.set_ylabel("Test samples", fontsize=FONT_SIZE, labelpad=2)
    ax_a.set_title("(a) Target-interest affinity map", fontsize=FONT_SIZE, pad=3,
                   fontweight="normal")
    set_journal_style(ax_a)
    ax_a.set_xticks(range(1, K + 1))
    ax_a.set_xticklabels([f"I{i}" for i in range(1, K + 1)], fontsize=FONT_SIZE)
    ax_a.set_yticks([])

    cbar = fig.colorbar(im, ax=ax_a, fraction=0.035, pad=0.03, ticks=[0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=FONT_SIZE, width=0.5, pad=1)
    cbar.outline.set_linewidth(0.5)
    cbar.set_label("Affinity", fontsize=FONT_SIZE, labelpad=1)

    # ===================================================================
    #  (b) Target-interest specificity  (violin + box)
    # ===================================================================
    spec_data = [spec_stats[ds]["specificity"] for ds in DATASETS]
    positions = np.arange(len(DATASETS))

    vp = ax_b.violinplot(
        spec_data, positions=positions,
        showmeans=False, showmedians=False, showextrema=False,
        widths=0.55,
    )

    for j, body in enumerate(vp["bodies"]):
        body.set_facecolor("white")
        body.set_edgecolor("black")
        body.set_linewidth(0.5)
        body.set_alpha(1.0)
        # Use hatch per dataset
        body.set_hatch(HATCHES_DS[DATASETS[j]])

    # Overlay boxplot elements: median line, IQR box
    bp = ax_b.boxplot(
        spec_data, positions=positions,
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=0.8),
        whiskerprops=dict(color="black", linewidth=0.5),
        capprops=dict(color="black", linewidth=0.5),
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=0.6),
    )

    ax_b.set_xticks(positions)
    ax_b.set_xticklabels([DS_LABELS[ds] for ds in DATASETS], fontsize=FONT_SIZE)
    ax_b.set_ylabel("Specificity", fontsize=FONT_SIZE, labelpad=2)
    set_journal_style(ax_b)
    ax_b.set_title("(b) Target-interest specificity", fontsize=FONT_SIZE, pad=3,
                   fontweight="normal")

    # ---- Save ----
    base = os.path.join(PROJECT_ROOT, "analysis_figures", "figures",
                        "target_interest_visual")
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
