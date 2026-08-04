# -*- coding: UTF-8 -*-
"""
plot_embedding_alignment.py
===========================
Read the npz dump from dump_embedding_alignment.py and produce a 1×2
black-and-white journal-style figure.

Left  – Before Alignment :  e_cf (black filled)  +  s_raw   (white filled, black edge)
Right – After  Alignment :  e_cf (black filled)  +  s_aligned (white filled, black edge)

Dimensionality reduction:  UMAP → TSNE → PCA  (automatic fallback).
Different categories → different marker shapes.
Output: PDF under analysis_figures/figures/
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("pdf")  # no display needed
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# -------------------------------
#  Dimensionality reduction
# -------------------------------
def reduce_2d(X, random_seed: int = 42):
    """
    Try UMAP → TSNE → PCA.  Returns (method_name, coords_2d).
    """
    # ---- UMAP ----
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=random_seed,
                            n_neighbors=30, min_dist=0.1, metric="cosine",
                            low_memory=True)
        coords = reducer.fit_transform(X)
        return "UMAP", coords
    except Exception as e:
        print(f"[WARN] UMAP failed ({e}), falling back to TSNE")

    # ---- TSNE ----
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=random_seed,
                    perplexity=min(30, X.shape[0] - 1),
                    metric="cosine", init="pca", learning_rate="auto")
        coords = tsne.fit_transform(X)
        return "t-SNE", coords
    except Exception as e:
        print(f"[WARN] TSNE failed ({e}), falling back to PCA")

    # ---- PCA ----
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=random_seed)
    coords = pca.fit_transform(X)
    return "PCA", coords


# -------------------------------
#  Plotting
# -------------------------------
MARKERS = ["o", "s", "^", "D", "v", "p", "*", "h", "X", "P"]


def set_journal_style(ax):
    """Black-and-white journal style: hide top/right spines, tick inward."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(direction="in", width=0.8, labelsize=9)


def plot_one_panel(ax, cf_2d, sem_2d, labels, unique_labels, label_names, title):
    """
    Plot one panel:
      - cf_2d : black filled markers
      - sem_2d: white filled, black edge markers
    Different labels → different marker shapes.
    """
    n_cats = len(unique_labels)

    # Plot CF points (black filled)
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        ax.scatter(
            cf_2d[mask, 0], cf_2d[mask, 1],
            marker=MARKERS[i % len(MARKERS)],
            s=12, c="black", edgecolors="none", alpha=0.8,
            zorder=3, label=f"{label_names[i]} (CF)"
        )

    # Plot semantic points (white filled, black edge)
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        ax.scatter(
            sem_2d[mask, 0], sem_2d[mask, 1],
            marker=MARKERS[i % len(MARKERS)],
            s=18, c="white", edgecolors="black", linewidths=0.6, alpha=0.85,
            zorder=4, label=f"{label_names[i]} (Sem)"
        )

    set_journal_style(ax)
    ax.set_title(title, fontsize=12, fontweight="normal", pad=8)
    ax.set_xlabel("Dim 1", fontsize=10)
    ax.set_ylabel("Dim 2", fontsize=10)

    # Legend: only keep semantic entries (fewer items)
    handles, _labels = ax.get_legend_handles_labels()
    n_each = len(handles) // 2
    # keep only the semantic half
    if n_each > 0:
        sem_handles = handles[n_each:]
        sem_labels = _labels[n_each:]
        ax.legend(sem_handles, sem_labels, fontsize=7,
                  loc="lower left", frameon=True, framealpha=0.9,
                  edgecolor="gray", fancybox=False)


def main():
    parser = argparse.ArgumentParser(description="Plot embedding alignment figure")
    parser.add_argument("--npz", type=str, required=True,
                        help="Path to the .npz dump file")
    parser.add_argument("--output", type=str, default="",
                        help="Output PDF path (default: auto under figures/)")
    parser.add_argument("--method", type=str, default="umap",
                        choices=["umap", "tsne", "pca"],
                        help="Force a specific reduction method")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="",
                        help="Dataset name for default output filename")
    args = parser.parse_args()

    # ---- Load npz ----
    data = np.load(args.npz, allow_pickle=True)
    item_ids = data["item_ids"]
    labels = data["labels"]
    label_names = list(data["label_names"])
    e_cf = data["e_cf"]            # (N, emb_size)
    s_raw = data["s_raw"]          # (N, d_llm)
    s_aligned = data["s_aligned"]  # (N, emb_size)

    unique_labels = np.unique(labels)
    n_items = len(item_ids)
    print(f"Loaded {n_items} items, {len(unique_labels)} categories")
    print(f"  e_cf shape:      {e_cf.shape}")
    print(f"  s_raw shape:     {s_raw.shape}")
    print(f"  s_aligned shape: {s_aligned.shape}")
    print(f"  Labels: {label_names}")

    # ---- Determine output path ----
    if args.output:
        out_path = args.output
    else:
        ds = args.dataset if args.dataset else "unknown"
        out_path = os.path.join(
            PROJECT_ROOT, "analysis_figures", "figures",
            f"{ds}_embedding_alignment.pdf"
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ---- Font setup ----
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    })

    # ---- Dimensionality reduction ----
    # Before: e_cf + s_raw  (they may have different dims; concat after projecting s_raw?)
    # Actually we need to reduce both sets into a shared 2D space.
    # Approach: fit UMAP on the concatenation of both point sets, then split.
    #
    # For "Before": e_cf (emb_size=64) and s_raw (d_llm=1536) have DIFFERENT dims.
    #   → We reduce each separately? No, we can't fit UMAP on concat of different dims.
    #
    # Approach for "Before": reduce e_cf independently, reduce s_raw independently.
    #   Then plot both in their own 2D spaces — BUT that means the two point clouds
    #   are NOT in the same coordinate frame. That's misleading.
    #
    # Better approach: project s_raw into e_cf space via the adapter first?
    #   But that defeats the purpose — we want to see the BEFORE alignment.
    #
    # Best approach: use CCA-like or just run UMAP on e_cf and then transform s_raw?
    #   UMAP doesn't support transform for out-of-sample data without fitting first.
    #   If we fit UMAP on X=concat[e_cf], the new points can't be added.
    #
    # Practical solution:
    #   For "Before": reduce e_cf and s_raw SEPARATELY → two independent 2D plots.
    #     This shows the STRUCTURE of each space, but not their relative positions.
    #   Actually this is the standard way alignment visualizations are done.
    #
    # WAIT — the correct approach for DIMENSION-MATCHED data:
    #   Fit UMAP on concat[e_cf, s_aligned] to get shared space.
    #   For "Before", e_cf and s_raw have different dimensionalities, so they CAN'T
    #   share a UMAP space. Instead, reduce each independently OR project s_raw.
    #
    # Let me use this approach:
    #   For BOTH panels: first project everything to a common dimension via PCA(emb_size),
    #   then run UMAP on the concatenation. But s_raw is 1536-dim and e_cf is 64-dim.
    #
    # SIMPLEST CORRECT APPROACH:
    #   Panel "Before": reduce e_cf→2D, reduce s_raw→2D, plot both (different spaces!)
    #   Panel "After": fit UMAP on concat[e_cf, s_aligned] → split (same space!)
    #
    # Actually for the "Before" panel, since s_raw is in LLM space and e_cf is in CF space,
    # they fundamentally live in different spaces. Showing them in independent 2D reductions
    # is the honest thing to do — it shows the topological structure of each.
    #
    # But that's confusing for readers. Let me think again...

    # REVISED APPROACH:
    # For "Before":
    #   Project s_raw to emb_size via a random-but-fixed projection (or PCA)
    #   Then fit UMAP on concat[e_cf, projected_s_raw] → split
    #   This puts them in a shared coordinate frame, which is what the reader expects.
    #
    # Actually, a better approach that respects the data:
    #   For "Before": UMAP on concat[e_cf, s_raw_pca] where s_raw_pca = PCA(s_raw, 64)
    #   For "After":  UMAP on concat[e_cf, s_aligned] (already same dim)
    #
    # This is honest: we project s_raw to 64-d to match e_cf dim for the shared space.

    if args.method == "umap":
        method = None  # auto-fallback
    elif args.method == "tsne":
        method = "tsne"
    else:
        method = "pca"

    # ---- Panel 1: Before Alignment ----
    print("\n--- Before Alignment ---")
    # Project s_raw to emb_size via PCA so we can concat with e_cf
    from sklearn.decomposition import PCA
    d_emb = e_cf.shape[1]
    pca_proj = PCA(n_components=d_emb, random_state=args.random_seed)
    s_raw_proj = pca_proj.fit_transform(s_raw)
    X_before = np.concatenate([e_cf, s_raw_proj], axis=0)
    n = e_cf.shape[0]

    if method is None:
        method_before, coords_before = reduce_2d(X_before, args.random_seed)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        tsne = TSNE(2, random_state=args.random_seed, perplexity=min(30, 2 * n - 1),
                    metric="cosine", init="pca", learning_rate="auto")
        coords_before = tsne.fit_transform(X_before)
        method_before = "t-SNE"
    else:
        from sklearn.decomposition import PCA
        pca = PCA(2, random_state=args.random_seed)
        coords_before = pca.fit_transform(X_before)
        method_before = "PCA"

    cf_before_2d = coords_before[:n]
    sem_before_2d = coords_before[n:]

    # ---- Panel 2: After Alignment ----
    print("--- After Alignment ---")
    X_after = np.concatenate([e_cf, s_aligned], axis=0)

    if method is None:
        method_after, coords_after = reduce_2d(X_after, args.random_seed)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        tsne = TSNE(2, random_state=args.random_seed, perplexity=min(30, 2 * n - 1),
                    metric="cosine", init="pca", learning_rate="auto")
        coords_after = tsne.fit_transform(X_after)
        method_after = "t-SNE"
    else:
        from sklearn.decomposition import PCA
        pca = PCA(2, random_state=args.random_seed)
        coords_after = pca.fit_transform(X_after)
        method_after = "PCA"

    cf_after_2d = coords_after[:n]
    sem_after_2d = coords_after[n:]

    print(f"Before: {method_before}  |  After: {method_after}")

    # ---- Create figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.subplots_adjust(wspace=0.3)

    plot_one_panel(ax1, cf_before_2d, sem_before_2d, labels, unique_labels, label_names,
                   title="(a) Before Alignment")
    plot_one_panel(ax2, cf_after_2d, sem_after_2d, labels, unique_labels, label_names,
                   title="(b) After Alignment")

    # Add reduction method note
    fig.text(0.5, 0.01,
             f"Dimensionality reduction: Left={method_before}, Right={method_after}",
             ha="center", fontsize=7, style="italic", color="gray")

    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    main()
