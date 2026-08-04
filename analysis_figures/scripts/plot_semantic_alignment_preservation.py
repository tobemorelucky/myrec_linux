# -*- coding: UTF-8 -*-
"""
plot_semantic_alignment_preservation_cn.py
==========================================
Semantic Alignment Preservation Analysis (Chinese labels version).

Based on already-exported embedding_alignment_dump.npz files.

Metrics
-------
1. Same-item alignment gap:
     pos  = cos(sem_i,   cf_i)       # same item
     neg  = cos(sem_i,   cf_j)       # different item (shuffled, j≠i)
     gap  = pos - neg
   Computed for s_raw (PCA→64d first) and s_aligned.

2. Collaborative preservation score:
     preserve_i = cos(e_sem_i,  e_cf_i)

3. Semantic injection score:
     inject_i   = cos(e_sem_i,  s_aligned_i) - cos(e_cf_i, s_aligned_i)

Figure (1×2, black & white):
  (a) 对齐差异
  (b) 协同保持与语义注入

Output:
    analysis_figures/figures/semantic_alignment_preservation_1x2_cn.pdf
    analysis_figures/figures/semantic_alignment_preservation_1x2_cn.png
    analysis_figures/figures/semantic_alignment_preservation_1x2_cn.svg
"""

import os
import argparse
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
DATASETS = ["beauty", "ml-1m", "toys"]
GAMMAS = {"beauty": 0.1, "ml-1m": 0.08, "toys": 0.05}
DS_LABELS = {"beauty": "Beauty", "ml-1m": "ML-1M", "toys": "Toys&Games"}

FONT_SIZE = 6  # pt

# ---------------------------------------------------------------------------
#  Fonts
# ---------------------------------------------------------------------------
FONT_DIR = os.path.join(PROJECT_ROOT, "Fonts")
SIMSUN_PATH = os.path.join(FONT_DIR, "simsun.ttc")
TIMES_PATH = os.path.join(FONT_DIR, "times.ttf")

if not os.path.exists(SIMSUN_PATH):
    raise FileNotFoundError(f"Chinese font not found: {SIMSUN_PATH}")

if not os.path.exists(TIMES_PATH):
    raise FileNotFoundError(f"English font not found: {TIMES_PATH}")

FONT_CN = FontProperties(fname=SIMSUN_PATH, size=FONT_SIZE)   # 宋体
FONT_EN = FontProperties(fname=TIMES_PATH, size=FONT_SIZE)    # Times New Roman

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def shuffled_neg_indices(n: int, rng: np.random.RandomState) -> np.ndarray:
    """Return a permutation of [0..n-1] with no element staying in place."""
    perm = rng.permutation(n)
    self_mask = perm == np.arange(n)
    if self_mask.any():
        idx = np.where(self_mask)[0]
        shifted = np.roll(perm[idx], 1)
        perm[idx] = shifted
        if n == 1:
            return perm
        still = perm == np.arange(n)
        if still.any():
            for i in np.where(still)[0]:
                j = (i + 1) % n
                perm[i], perm[j] = perm[j], perm[i]
        assert not (perm == np.arange(n)).any(), "Failed to generate derangement"
    return perm


def cos_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity. a, b both (N, D). Returns (N,)."""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return (a_n * b_n).sum(axis=1)


def compute_all(data: dict, gamma: float, rng: np.random.RandomState):
    """
    Returns a dict of results keyed by metric name.
    """
    e_cf = data["e_cf"].astype(np.float64)
    s_raw = data["s_raw"].astype(np.float64)
    s_aligned = data["s_aligned"].astype(np.float64)
    N, d_cf = e_cf.shape
    d_llm = s_raw.shape[1]

    # Fused embedding
    e_sem = e_cf + gamma * s_aligned

    # Shuffled indices (derangement)
    neg_idx = shuffled_neg_indices(N, rng)

    # ---------------------------------------------------------------
    # 1. Same-item alignment gap for s_aligned (dim matched)
    # ---------------------------------------------------------------
    pos_aligned = cos_sim(s_aligned, e_cf)
    neg_aligned = cos_sim(s_aligned, e_cf[neg_idx])
    gap_aligned = pos_aligned - neg_aligned

    # ---------------------------------------------------------------
    # 2. Same-item alignment gap for s_raw (PCA→d_cf first)
    # ---------------------------------------------------------------
    raw_dim_matched = (d_llm == d_cf)
    if raw_dim_matched:
        s_raw_proj = s_raw
        pca_note = "s_raw dim == e_cf dim, no PCA needed"
    else:
        pca = PCA(n_components=d_cf, random_state=rng.randint(0, 2**31))
        s_raw_proj = pca.fit_transform(s_raw)
        pca_note = f"s_raw PCA {d_llm}→{d_cf}, explained_var={float(pca.explained_variance_ratio_.sum()):.4f}"

    pos_raw = cos_sim(s_raw_proj, e_cf)
    neg_raw = cos_sim(s_raw_proj, e_cf[neg_idx])
    gap_raw = pos_raw - neg_raw

    # ---------------------------------------------------------------
    # 3. Collaborative preservation
    # ---------------------------------------------------------------
    preserve = cos_sim(e_sem, e_cf)

    # ---------------------------------------------------------------
    # 4. Semantic injection
    # ---------------------------------------------------------------
    cos_e_sem_s = cos_sim(e_sem, s_aligned)
    cos_e_cf_s = cos_sim(e_cf, s_aligned)
    inject = cos_e_sem_s - cos_e_cf_s

    return {
        "gap_raw": gap_raw,
        "gap_aligned": gap_aligned,
        "preserve": preserve,
        "inject": inject,
        "pca_note": pca_note,
    }


def set_journal_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(direction="in", width=0.5, labelsize=FONT_SIZE, pad=1.5)
    ax.set_facecolor("white")


def set_tick_font(ax, x_axis=True, y_axis=True):
    """Set tick labels to Times New Roman."""
    if x_axis:
        for label in ax.get_xticklabels():
            label.set_fontproperties(FONT_EN)
    if y_axis:
        for label in ax.get_yticklabels():
            label.set_fontproperties(FONT_EN)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Semantic Alignment Preservation Analysis (Chinese labels)")
    parser.add_argument(
        "--dumps_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "analysis_figures", "dumps")
    )
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    # ---- Font rc ----
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    })
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "path"   # 避免 SVG 文本重叠/错位

    rng = np.random.RandomState(args.random_seed)

    print(f"[Font] Chinese: {SIMSUN_PATH}")
    print(f"[Font] English: {TIMES_PATH}")

    # ---- Compute ----
    all_res = {}
    for ds in DATASETS:
        npz_path = os.path.join(args.dumps_dir, f"{ds}_embedding_alignment_dump.npz")
        if not os.path.exists(npz_path):
            print(f"[WARN] {npz_path} not found, skipping {ds}")
            continue

        data = np.load(npz_path, allow_pickle=True)
        res = compute_all(data, GAMMAS[ds], rng)
        all_res[ds] = res

        print(f"=== {ds} (gamma={GAMMAS[ds]}) ===")
        print(f"  PCA note: {res['pca_note']}")
        print(f"  gap_raw     mean={float(np.mean(res['gap_raw'])):.4f}, std={float(np.std(res['gap_raw'])):.4f}")
        print(f"  gap_aligned mean={float(np.mean(res['gap_aligned'])):.4f}, std={float(np.std(res['gap_aligned'])):.4f}")
        print(f"  preserve    mean={float(np.mean(res['preserve'])):.4f}, std={float(np.std(res['preserve'])):.4f}")
        print(f"  inject      mean={float(np.mean(res['inject'])):.4f}, std={float(np.std(res['inject'])):.4f}")
        print()

    # ---- Figure: 1×2 horizontal ----
    FIG_W = 13.0 / 2.54
    FIG_H = 5.7 / 2.54
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(
        wspace=0.42,
        left=0.10,
        right=0.96,
        top=0.70,
        bottom=0.18
    )

    n_ds = len(DATASETS)

    # Common bar params
    bar_w = 0.32
    err_kw = dict(linewidth=0.5, capsize=2.0)
    bar_kw = dict(color="white", edgecolor="black", linewidth=0.5, zorder=3)
    leg_kw = dict(
        fontsize=FONT_SIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=2,
        borderaxespad=0,
        frameon=False,
        labelspacing=0.15,
        handlelength=1.35,
        handletextpad=0.35,
        columnspacing=0.8,
        prop=FONT_CN
    )

    # =====================
    #  (a) 对齐差异
    # =====================
    x = np.arange(n_ds)
    hatch_pairs = {"gap_raw": "..", "gap_aligned": "//"}
    label_pairs = {"gap_raw": "原始语义", "gap_aligned": "对齐语义"}

    for idx, key in enumerate(["gap_raw", "gap_aligned"]):
        means = [float(np.mean(all_res[ds][key])) for ds in DATASETS]
        stds = [float(np.std(all_res[ds][key])) for ds in DATASETS]
        offset = (idx - 0.5) * bar_w
        ax_a.bar(
            x + offset, means, bar_w,
            yerr=stds, error_kw=err_kw,
            hatch=hatch_pairs[key], label=label_pairs[key],
            **bar_kw,
        )

    ax_a.set_xticks(x)
    ax_a.set_xticklabels([DS_LABELS[ds] for ds in DATASETS], fontsize=FONT_SIZE)
    ax_a.set_ylabel("对齐差异", fontsize=FONT_SIZE, labelpad=2)
    ax_a.set_title("（a）对齐差异", fontsize=FONT_SIZE, y=1.26, pad=0, fontweight="normal")

    ax_a.yaxis.label.set_fontproperties(FONT_CN)
    ax_a.title.set_fontproperties(FONT_CN)

    set_journal_style(ax_a)
    set_tick_font(ax_a, x_axis=True, y_axis=True)

    legend_a = ax_a.legend(**leg_kw)
    for text in legend_a.get_texts():
        text.set_fontproperties(FONT_CN)

    ax_a.tick_params(pad=1.2)

    # Reference line
    yl_a, yh_a = ax_a.get_ylim()
    if yl_a < 0 < yh_a:
        ax_a.axhline(y=0, color="black", linewidth=0.3, linestyle=":", zorder=0)

    # =====================
    #  (b) 协同保持与语义注入
    # =====================
    pairs_b = {"preserve": "协同保持", "inject": "语义注入"}
    hatches_b = {"preserve": "", "inject": "//"}

    for idx, key in enumerate(["preserve", "inject"]):
        means = [float(np.mean(all_res[ds][key])) for ds in DATASETS]
        stds = [float(np.std(all_res[ds][key])) for ds in DATASETS]
        offset = (idx - 0.5) * bar_w
        ax_b.bar(
            x + offset, means, bar_w,
            yerr=stds, error_kw=err_kw,
            hatch=hatches_b[key], label=pairs_b[key],
            **bar_kw,
        )

    ax_b.set_xticks(x)
    ax_b.set_xticklabels([DS_LABELS[ds] for ds in DATASETS], fontsize=FONT_SIZE)
    ax_b.set_ylabel("余弦相似度得分", fontsize=FONT_SIZE, labelpad=2)
    ax_b.set_title("（b）协同保持与语义注入", fontsize=FONT_SIZE, y=1.26, pad=0, fontweight="normal")

    ax_b.yaxis.label.set_fontproperties(FONT_CN)
    ax_b.title.set_fontproperties(FONT_CN)

    set_journal_style(ax_b)
    set_tick_font(ax_b, x_axis=True, y_axis=True)

    legend_b = ax_b.legend(**leg_kw)
    for text in legend_b.get_texts():
        text.set_fontproperties(FONT_CN)

    ax_b.tick_params(pad=1.2)

    # ---- Save ----
    if args.output:
        base = args.output
    else:
        base = os.path.join(
            PROJECT_ROOT, "analysis_figures", "figures",
            "semantic_alignment_preservation_1x2_cn"
        )

    os.makedirs(os.path.dirname(base), exist_ok=True)

    fig.savefig(base + ".pdf", dpi=600, bbox_inches="tight", pad_inches=0.04)
    print(f"Saved {base}.pdf")

    fig.savefig(base + ".png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    print(f"Saved {base}.png")

    try:
        fig.savefig(base + ".svg", dpi=600, bbox_inches="tight", pad_inches=0.04)
        print(f"Saved {base}.svg")
    except Exception as e:
        print(f"[INFO] SVG output skipped ({e})")

    plt.close(fig)


if __name__ == "__main__":
    main()