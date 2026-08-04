# -*- coding: UTF-8 -*-
"""
plot_gate_analysis.py
=====================
Read gate_analysis_dump.npz and produce a 1×2 black-and-white journal figure.

Panel (a): Gate heatmap  —  grayscale, sorted by max-gate position, top-100
                            samples with highest gate variance.
Panel (b): Target-related gate comparison  —  same vs different category.

Output: analysis_figures/figures/{dataset}_gate_analysis.pdf / .png
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


def select_samples(gates, history_items, n_select=100):
    """
    Select n_select samples with highest gate variance.
    Returns indices and the sorted ordering (by max-gate position).
    """
    N, L = gates.shape
    # Gate variance per sample (only over valid positions)
    valid = (history_items > 0).astype(np.float32)
    gate_var = np.array([
        np.var(gates[i][valid[i] > 0]) if valid[i].sum() > 0 else 0
        for i in range(N)
    ])
    idx = np.argsort(gate_var)[::-1][:n_select]

    # Sort selected by position of max gate
    gate_sub = gates[idx]
    max_pos = np.argmax(gate_sub * valid[idx], axis=1)
    sort_order = np.argsort(max_pos)
    return idx[sort_order]


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot gate analysis figure")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to gate_analysis_dump.npz")
    parser.add_argument("--dataset", type=str, default="beauty")
    parser.add_argument("--output_pdf", type=str, default="")
    parser.add_argument("--output_png", type=str, default="")
    parser.add_argument("--n_heatmap", type=int, default=100,
                        help="Number of samples in heatmap")
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    # ---- Font ----
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    })

    # ---- Load ----
    data = np.load(args.input, allow_pickle=True)
    history_items = data["history_items"].astype(np.int64)
    target_items = data["target_items"].astype(np.int64)
    gates = data["gates"].astype(np.float64)
    history_categories = data["history_categories"].astype(np.int64)
    target_categories = data["target_categories"].astype(np.int64)

    N, L = gates.shape
    valid_mask = (history_items > 0)
    print(f"Loaded {N} test samples, history_len={L}")
    print(f"Gate stats: mean={gates[valid_mask].mean():.4f}, "
          f"std={gates[valid_mask].std():.4f}, "
          f"min={gates[valid_mask].min():.4f}, max={gates[valid_mask].max():.4f}")

    has_cat = not (target_categories[0] == -1 and history_categories.max() <= 0)
    print(f"Category info available: {has_cat}")

    # ---- Figure: 1×2 horizontal ----
    FIG_W = 13.0 / 2.54
    FIG_H = 5.8 / 2.54
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(wspace=0.45, left=0.10, right=0.97, top=0.88, bottom=0.15)

    # ===================================================================
    #  (a) Gate heatmap
    # ===================================================================
    idx = select_samples(gates, history_items, n_select=args.n_heatmap)
    gates_sel = gates[idx]          # (K, L)
    hist_sel = history_items[idx]   # (K, L)
    valid_sel = valid_mask[idx]
    K = len(idx)

    # Mask padding for display
    gates_disp = np.where(valid_sel, gates_sel, np.nan)

    im = ax_a.imshow(
        gates_disp, aspect="auto", cmap="Greys",
        origin="upper", vmin=0.0, vmax=1.0,
        interpolation="nearest",
    )

    ax_a.set_xlabel("History position", fontsize=FONT_SIZE, labelpad=2)
    ax_a.set_ylabel("Test samples", fontsize=FONT_SIZE, labelpad=2)
    ax_a.set_title("(a) Gate heatmap", fontsize=FONT_SIZE, pad=3, fontweight="normal")
    set_journal_style(ax_a)

    # Colorbar (compact)
    cbar = fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.03,
                        ticks=[0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=FONT_SIZE - 1, width=0.5, pad=1)
    cbar.outline.set_linewidth(0.5)
    cbar.set_label("Gate", fontsize=FONT_SIZE - 1, labelpad=1)

    # ===================================================================
    #  (b) Target-related gate comparison
    # ===================================================================
    if has_cat:
        # ---- Same vs Different target category ----
        same_vals = []
        diff_vals = []
        for i in range(N):
            tgt_cat = target_categories[i]
            h_cats = history_categories[i]
            valid = valid_mask[i]
            if tgt_cat < 0 or not valid.any():
                continue
            g = gates[i][valid]
            c = h_cats[valid]
            same_mask = (c == tgt_cat)
            diff_mask = (c != tgt_cat) & (c >= 0)
            if same_mask.any():
                same_vals.append(g[same_mask].mean())
            if diff_mask.any():
                diff_vals.append(g[diff_mask].mean())

        same_mean = float(np.mean(same_vals)) if same_vals else 0.0
        same_std = float(np.std(same_vals)) if same_vals else 0.0
        diff_mean = float(np.mean(diff_vals)) if diff_vals else 0.0
        diff_std = float(np.std(diff_vals)) if diff_vals else 0.0

        print(f"Same-cat gate: mean={same_mean:.4f} std={same_std:.4f}  (n={len(same_vals)})")
        print(f"Diff-cat gate: mean={diff_mean:.4f} std={diff_std:.4f}  (n={len(diff_vals)})")

        x_pos = [0, 1]
        means = [same_mean, diff_mean]
        stds = [same_std, diff_std]
        labels = ["Same category", "Different category"]
        hatches = ["", "//"]

        bar_w = 0.50
        for xp, m, s, h, lb in zip(x_pos, means, stds, hatches, labels):
            ax_b.bar(xp, m, bar_w, yerr=s,
                     color="white", edgecolor="black", linewidth=0.5,
                     hatch=h,
                     error_kw=dict(linewidth=0.5, capsize=2.5),
                     label=lb, zorder=3)

        ax_b.set_xticks(x_pos)
        ax_b.set_xticklabels(labels, fontsize=FONT_SIZE)
        ax_b.set_ylabel("Average gate", fontsize=FONT_SIZE, labelpad=2)
        ax_b.set_title("(b) Target-related gate", fontsize=FONT_SIZE, pad=3,
                       fontweight="normal")

    else:
        # ---- Fallback: top vs bottom gate similarity ----
        # Not needed since categories are available for all 3 datasets
        ax_b.text(0.5, 0.5, "Categories unavailable", transform=ax_b.transAxes,
                  ha="center", va="center", fontsize=FONT_SIZE)
        ax_b.set_title("(b) Target-related gate", fontsize=FONT_SIZE, pad=3,
                       fontweight="normal")

    set_journal_style(ax_b)
    ax_b.tick_params(pad=1.2)

    # ---- Save ----
    base_pdf = args.output_pdf or os.path.join(
        PROJECT_ROOT, "analysis_figures", "figures",
        f"{args.dataset}_gate_analysis.pdf"
    )
    base_png = args.output_png or os.path.join(
        PROJECT_ROOT, "analysis_figures", "figures",
        f"{args.dataset}_gate_analysis.png"
    )

    for p in [base_pdf, base_png]:
        os.makedirs(os.path.dirname(p), exist_ok=True)

    fig.savefig(base_pdf, dpi=600, bbox_inches="tight", pad_inches=0.04)
    print(f"Saved {base_pdf}")

    fig.savefig(base_png, dpi=600, bbox_inches="tight", pad_inches=0.04)
    print(f"Saved {base_png}")

    plt.close(fig)


if __name__ == "__main__":
    main()
