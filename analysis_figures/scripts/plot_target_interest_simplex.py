# -*- coding: UTF-8 -*-
"""
plot_target_interest_simplex.py
===============================
Ternary simplex plot of target-item response across 3 interest components.

For each dataset, softmax(target_affinity / temp) → (p1,p2,p3) barycentric,
mapped to 2-D triangle coordinates, displayed as grayscale hexbin density.

Requires K=3 (3 interest components).

Left→Right: Beauty, ML-1M, Toys.

Output: analysis_figures/figures/target_interest_simplex.pdf / .png / .svg
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FONT_SIZE = 6
DATASETS = ["beauty", "ml-1m", "toys"]
DS_LABELS = {"beauty": "Beauty", "ml-1m": "ML-1M", "toys": "Toys"}

# Triangle geometry
H = np.sqrt(3) / 2          # ≈ 0.866
TRI_VERTICES = np.array([
    [0.0, 0.0],              # I1
    [1.0, 0.0],              # I2
    [0.5, H],                # I3
])
# Vertex label offsets (in axes fraction)
LABEL_OFFSETS = [
    (-0.04, -0.06),          # I1: below-left
    (1.04, -0.06),           # I2: below-right
    (0.50, H + 0.04),        # I3: above
]
LABEL_HALIGNS = ["right", "left", "center"]
LABEL_VALIGNS = ["top", "top", "bottom"]


def softmax(x, temp=0.1):
    """Row-wise softmax with temperature."""
    x = x.astype(np.float64) / temp
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def to_barycentric(p):
    """p: (N, 3) with p1+p2+p3=1 → (x, y) in triangle coords."""
    x = p[:, 1] + 0.5 * p[:, 2]            # p2 + 0.5*p3
    y = H * p[:, 2]                         # h * p3
    return x, y


def inside_triangle(x, y):
    """Boolean mask: point is inside the unit triangle."""
    ok = (y >= 0) & (y <= 2 * H * x) & (y <= 2 * H * (1 - x))
    return ok


def draw_triangle(ax):
    """Draw triangle outline and vertex labels."""
    tri = Polygon(TRI_VERTICES, fill=False, edgecolor="black",
                  linewidth=0.6, zorder=5)
    ax.add_patch(tri)

    labels = ["I1", "I2", "I3"]
    for i, (x0, y0) in enumerate(TRI_VERTICES):
        ax.annotate(
            labels[i],
            xy=(x0, y0),
            xytext=LABEL_OFFSETS[i][:2],
            textcoords="data" if i < 2 else "data",
            fontsize=FONT_SIZE, fontweight="normal",
            ha=LABEL_HALIGNS[i], va=LABEL_VALIGNS[i],
        )


def set_journal_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(direction="in", width=0.5, labelsize=FONT_SIZE, pad=1.2)
    ax.set_facecolor("white")
    ax.set_aspect("equal")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Target-interest simplex plot")
    parser.add_argument("--dumps_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "analysis_figures", "dumps"))
    parser.add_argument("--temp", type=float, default=0.1,
                        help="Softmax temperature (default 0.1)")
    parser.add_argument("--gridsize", type=int, default=22,
                        help="Hexbin grid size")
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    })

    # ---- Load & compute ----
    all_probs = {}
    for ds in DATASETS:
        p = os.path.join(args.dumps_dir, f"{ds}_target_interest_analysis_dump.npz")
        if not os.path.exists(p):
            print(f"[ERROR] {p} not found")
            sys.exit(1)
        d = np.load(p, allow_pickle=True)
        aff = d["target_affinity"].astype(np.float64)
        K = aff.shape[1]
        if K != 3:
            print(f"[ERROR] K={K} for {ds}, simplex requires K=3")
            sys.exit(1)

        probs = softmax(aff, temp=args.temp)          # (N, 3)
        all_probs[ds] = probs
        ks = d["k_star"]
        k_dist = {int(k): int((ks == k).sum()) for k in range(K)}

        # Specificity
        ps = np.sort(probs, axis=1)[:, ::-1]
        spec = ps[:, 0] - ps[:, 1]

        print(f"=== {ds} (temp={args.temp}) ===")
        print(f"  k* distribution: {k_dist}")
        print(f"  p (softmax) mean: [{probs[:,0].mean():.4f}, "
              f"{probs[:,1].mean():.4f}, {probs[:,2].mean():.4f}]")
        print(f"  specificity mean={spec.mean():.4f}  median={float(np.median(spec)):.4f}  "
              f"std={spec.std():.4f}")
        frac = {k: v / len(ks) for k, v in k_dist.items()}
        print(f"  interest fraction: {frac}")
        print()

    # ---- Figure: 1×3 ----
    FIG_W = 14.0 / 2.54
    FIG_H = 5.5 / 2.54
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(wspace=0.40, left=0.06, right=0.99, top=0.93, bottom=0.10)

    for i, ds in enumerate(DATASETS):
        ax = axes[i]
        probs = all_probs[ds]
        x, y = to_barycentric(probs)

        # Filter to triangle interior
        inside = inside_triangle(x, y)
        x_in = x[inside]
        y_in = y[inside]

        # Hexbin density
        hb = ax.hexbin(x_in, y_in, gridsize=args.gridsize,
                       cmap="Greys", mincnt=1,
                       linewidths=0.1, edgecolors="none")

        draw_triangle(ax)

        ax.set_xlim(-0.10, 1.10)
        ax.set_ylim(-0.10, H + 0.08)
        ax.set_xticks([])
        ax.set_yticks([])
        set_journal_style(ax)
        ax.set_title(f"({chr(97 + i)}) {DS_LABELS[ds]}",
                     fontsize=FONT_SIZE, pad=2, fontweight="normal")

    # ---- Save ----
    base = os.path.join(PROJECT_ROOT, "analysis_figures", "figures",
                        "target_interest_simplex")
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
