# -*- coding: UTF-8 -*-
"""
plot_semantic_residual_analysis.py
==================================
Semantic Residual Injection Analysis.

Based on already-exported embedding_alignment_dump.npz files, compute:
  1. e_sem = e_cf + gamma * s_aligned   (fused item embedding used in the model)
  2. Item-item cosine similarity matrices for e_cf, s_raw, s_aligned, e_sem.
  3. Spearman rank correlation between the upper-triangles of the CF similarity
     matrix and each semantic similarity matrix:
        Corr(s_raw, e_cf)       →  "Raw Sem"
        Corr(s_aligned, e_cf)   →  "Aligned Sem"
        Corr(e_sem,   e_cf)     →  "Residual Emb"
  4. Residual contribution ratio per item:
        rho_i = ||gamma * s_aligned_i||_2  /  ||e_cf_i||_2

Figure (1×2, black & white journal style, PNG output):
  (a) Structural consistency with collaborative space
      — grouped bars, x = {Raw Sem, Aligned Sem, Residual Emb}
      — 3 datasets differentiated by hatch patterns
  (b) Residual contribution ratio
      — boxplot per dataset

Output: analysis_figures/figures/semantic_residual_analysis.png
"""

import os
import sys
import argparse
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
DATASETS = ["beauty", "ml-1m", "toys"]
GAMMAS = {"beauty": 0.1, "ml-1m": 0.08, "toys": 0.05}
# Pretty names for display
DS_LABELS = {"beauty": "Beauty", "ml-1m": "ML-1M", "toys": "Toys"}

# Hatch styles for the three datasets (black & white)
HATCHES = {"beauty": "", "ml-1m": "//", "toys": "\\\\"}
# Edge distinction for boxplots
BOX_EDGES = {"beauty": "-", "ml-1m": "--", "toys": "-."}

SEMANTIC_LABELS = ["Raw Sem", "Aligned Sem", "Residual Emb"]

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def upper_tri_flat(S: np.ndarray) -> np.ndarray:
    """Return the upper-triangle (excluding diagonal) of a square matrix, flattened."""
    n = S.shape[0]
    iu = np.triu_indices(n, k=1)
    return S[iu]


def compute_all(data: dict, gamma: float):
    """
    data: dict with keys e_cf, s_raw, s_aligned  (each N×D numpy array)
    Returns dict with all statistics.
    """
    e_cf = data["e_cf"].astype(np.float64)
    s_raw = data["s_raw"].astype(np.float64)
    s_aligned = data["s_aligned"].astype(np.float64)

    # Fused embedding
    e_sem = e_cf + gamma * s_aligned

    # ---- Cosine similarity matrices ----
    S_cf = cosine_similarity(e_cf)        # (N, N)
    S_sem = cosine_similarity(e_sem)
    S_aligned = cosine_similarity(s_aligned)
    S_raw = cosine_similarity(s_raw)

    # ---- Upper triangles ----
    u_cf = upper_tri_flat(S_cf)
    u_raw = upper_tri_flat(S_raw)
    u_aligned = upper_tri_flat(S_aligned)
    u_sem = upper_tri_flat(S_sem)

    # ---- Spearman correlations with CF ----
    r_raw, _ = spearmanr(u_raw, u_cf)
    r_aligned, _ = spearmanr(u_aligned, u_cf)
    r_sem, _ = spearmanr(u_sem, u_cf)

    # ---- Residual ratio per item ----
    e_cf_norm = np.linalg.norm(e_cf, axis=1)           # (N,)
    sem_inj_norm = gamma * np.linalg.norm(s_aligned, axis=1)  # (N,)
    rho = sem_inj_norm / (e_cf_norm + 1e-12)           # (N,)

    return {
        "Corr_raw": float(r_raw),
        "Corr_aligned": float(r_aligned),
        "Corr_e_sem": float(r_sem),
        "rho": rho,
    }


# ---------------------------------------------------------------------------
#  Plotting helpers
# ---------------------------------------------------------------------------
def set_journal_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(direction="in", width=0.8, labelsize=9)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Semantic Residual Injection Analysis")
    parser.add_argument("--dumps_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "analysis_figures", "dumps"),
                        help="Directory containing embedding_alignment_dump.npz files")
    parser.add_argument("--output", type=str, default="",
                        help="Output PNG path")
    args = parser.parse_args()

    # ---- Font ----
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    })

    # ---- Load & compute for all datasets ----
    all_results = {}
    for ds in DATASETS:
        npz_path = os.path.join(args.dumps_dir, f"{ds}_embedding_alignment_dump.npz")
        if not os.path.exists(npz_path):
            print(f"[WARN] {npz_path} not found, skipping {ds}")
            continue
        data = np.load(npz_path, allow_pickle=True)
        res = compute_all(data, GAMMAS[ds])
        all_results[ds] = res

        rho_mean = float(np.mean(res["rho"]))
        rho_std = float(np.std(res["rho"]))
        print(f"=== {ds} (gamma={GAMMAS[ds]}) ===")
        print(f"  Corr(s_raw,   e_cf) = {res['Corr_raw']:.4f}")
        print(f"  Corr(s_aligned, e_cf) = {res['Corr_aligned']:.4f}")
        print(f"  Corr(e_sem,   e_cf) = {res['Corr_e_sem']:.4f}")
        print(f"  rho  mean = {rho_mean:.4f}, std = {rho_std:.4f}")
        print()

    # ---- Figure ----
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 5))
    fig.subplots_adjust(wspace=0.35)

    # =====================
    #  (a) Structural consistency with collaborative space
    # =====================
    n_sem = len(SEMANTIC_LABELS)  # 3
    n_ds = len(DATASETS)          # 3

    x = np.arange(n_sem)          # 0, 1, 2
    bar_width = 0.25
    offsets = np.linspace(-bar_width, bar_width, n_ds)  # equally spaced offsets

    for j, ds in enumerate(DATASETS):
        vals = [
            all_results[ds]["Corr_raw"],
            all_results[ds]["Corr_aligned"],
            all_results[ds]["Corr_e_sem"],
        ]
        bars = ax_a.bar(
            x + offsets[j], vals, bar_width * 0.95,
            color="white",
            edgecolor="black",
            linewidth=0.8,
            hatch=HATCHES[ds],
            label=DS_LABELS[ds],
            zorder=3,
        )

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(SEMANTIC_LABELS, fontsize=10)
    ax_a.set_ylabel("Spearman Correlation", fontsize=10)
    set_journal_style(ax_a)
    ax_a.set_title("(a) Structural Consistency\nwith Collaborative Space", fontsize=11, pad=10)
    ax_a.legend(fontsize=8, loc="lower left", frameon=True, framealpha=0.9,
                edgecolor="gray", fancybox=False)
    # Slightly extend y-limit for legend clearance
    ylim_a = ax_a.get_ylim()
    ax_a.set_ylim(ylim_a[0], ylim_a[1] + 0.02)

    # =====================
    #  (b) Residual contribution ratio
    # =====================
    box_data = []
    box_labels = []
    for ds in DATASETS:
        box_data.append(all_results[ds]["rho"])
        box_labels.append(DS_LABELS[ds])

    bp = ax_b.boxplot(
        box_data, tick_labels=box_labels,
        patch_artist=True,
        widths=0.5,
        showfliers=True,
        flierprops=dict(marker="o", markersize=3, markerfacecolor="white",
                        markeredgecolor="black", markeredgewidth=0.5),
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="black", linewidth=0.8),
        capprops=dict(color="black", linewidth=0.8),
        boxprops=dict(edgecolor="black", linewidth=0.8),
    )

    # Apply hatch per box
    for j, (box, ds) in enumerate(zip(bp["boxes"], DATASETS)):
        box.set_facecolor("white")
        box.set_hatch(HATCHES[ds])

    ax_b.set_ylabel(r"Residual Ratio  $\rho_i$", fontsize=10)
    set_journal_style(ax_b)
    ax_b.set_title("(b) Residual Contribution Ratio", fontsize=11, pad=10)

    # ---- Save ----
    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(
            PROJECT_ROOT, "analysis_figures", "figures",
            "semantic_residual_analysis.png"
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
