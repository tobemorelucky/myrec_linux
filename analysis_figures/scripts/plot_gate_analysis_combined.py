# -*- coding: UTF-8 -*-
"""
plot_gate_analysis_combined_cn.py
=================================
Combined 1×2 figure for Target-aware Historical Behavior Reweighting.

Left  (a): Beauty gate heatmap
           (representative, 100 high-variance samples,
            min 10 valid history positions).

Right (b): Same-cat vs Diff-cat average gate for all 3 datasets
           with bootstrap 95% CI.

Input:
    analysis_figures/dumps/{beauty,ml-1m,toys}_gate_analysis_dump.npz

Output:
    analysis_figures/figures/gate_analysis_combined_cn.pdf
    analysis_figures/figures/gate_analysis_combined_cn.png
    analysis_figures/figures/gate_analysis_combined_cn.svg
    analysis_figures/figures/gate_analysis_combined_cn.tif
    analysis_figures/figures/gate_analysis_combined_cn.jpg

Recommended submission format:
    TIFF: 1200 dpi + LZW lossless compression
"""

import os
import sys
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

FONT_SIZE = 6

DATASETS = [
    "beauty",
    "ml-1m",
    "toys",
]

DS_LABELS = {
    "beauty": "Beauty",
    "ml-1m": "ML-1M",
    "toys": "Toys&Games",
}


# ---------------------------------------------------------------------------
# Font settings
# ---------------------------------------------------------------------------

FONT_DIR = os.path.join(
    PROJECT_ROOT,
    "Fonts"
)

SIMSUN_PATH = os.path.join(
    FONT_DIR,
    "simsun.ttc"
)

TIMES_PATH = os.path.join(
    FONT_DIR,
    "times.ttf"
)


if not os.path.exists(SIMSUN_PATH):
    raise FileNotFoundError(
        f"Chinese font not found: {SIMSUN_PATH}"
    )

if not os.path.exists(TIMES_PATH):
    raise FileNotFoundError(
        f"English font not found: {TIMES_PATH}"
    )


FONT_CN = FontProperties(
    fname=SIMSUN_PATH,
    size=FONT_SIZE
)

FONT_EN = FontProperties(
    fname=TIMES_PATH,
    size=FONT_SIZE
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_journal_style(ax):
    """
    Set journal-style axes.
    """

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    ax.tick_params(
        direction="in",
        width=0.5,
        labelsize=FONT_SIZE,
        pad=1.2
    )

    ax.set_facecolor("white")


def set_tick_font(
    ax,
    x_axis=True,
    y_axis=True
):
    """
    Set tick labels to Times New Roman.
    """

    if x_axis:
        for label in ax.get_xticklabels():
            label.set_fontproperties(FONT_EN)

    if y_axis:
        for label in ax.get_yticklabels():
            label.set_fontproperties(FONT_EN)


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    ci: float = 95.0,
    rng: np.random.RandomState = None
):
    """
    Bootstrap CI for the mean.

    Returns:
        (lower, upper)
    """

    if rng is None:
        rng = np.random.RandomState(42)

    n = len(values)

    if n < 2:
        m = (
            float(np.mean(values))
            if n == 1
            else 0.0
        )

        return m, m

    means = np.empty(
        n_boot
    )

    for i in range(n_boot):

        sample = rng.choice(
            values,
            size=n,
            replace=True
        )

        means[i] = np.mean(
            sample
        )

    lo = (
        100.0 - ci
    ) / 2.0

    hi = (
        100.0 - lo
    )

    return (
        float(
            np.percentile(
                means,
                lo
            )
        ),
        float(
            np.percentile(
                means,
                hi
            )
        )
    )


def select_heatmap_samples(
    gates,
    history_items,
    n_select=100,
    min_history=10
):
    """
    Select representative samples.

    First filter by minimum valid history length,
    then select samples with highest gate variance.
    """

    N, L = history_items.shape

    valid = (
        history_items > 0
    )

    n_valid = valid.sum(
        axis=1
    )

    # ------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------

    eligible = np.where(
        n_valid >= min_history
    )[0]

    if len(eligible) < n_select:

        print(
            f"[WARN] Only {len(eligible)} samples with "
            f">={min_history} valid history positions "
            f"(need {n_select})"
        )

        n_select = min(
            n_select,
            len(eligible)
        )

    if n_select == 0:

        return np.array(
            [],
            dtype=np.int64
        )

    # ------------------------------------------------------------
    # Gate variance over valid positions only
    # ------------------------------------------------------------

    gate_var = np.zeros(
        N,
        dtype=np.float64
    )

    for i in eligible:

        g = gates[i][
            valid[i]
        ]

        gate_var[i] = (
            np.var(g)
            if len(g) > 0
            else 0.0
        )

    top_idx = np.argsort(
        gate_var[eligible]
    )[::-1][:n_select]

    selected = eligible[
        top_idx
    ]

    # ------------------------------------------------------------
    # Sort by position of maximum gate
    # ------------------------------------------------------------

    gate_sub = gates[
        selected
    ]

    max_pos = np.zeros(
        len(selected),
        dtype=np.int64
    )

    for i, si in enumerate(
        selected
    ):

        v = valid[
            si
        ]

        g = gate_sub[
            i
        ].copy()

        g[~v] = -1.0

        max_pos[i] = np.argmax(
            g
        )

    sort_order = np.argsort(
        max_pos
    )

    return selected[
        sort_order
    ]


def compute_category_gates(
    gates,
    history_items,
    history_categories,
    target_categories
):
    """
    Compute per-sample mean gate for:

        1. same-category history items
        2. different-category history items

    Returns:
        same_means, diff_means
    """

    N = len(
        target_categories
    )

    same_means = []
    diff_means = []

    for i in range(N):

        tgt_cat = target_categories[
            i
        ]

        if tgt_cat < 0:
            continue

        valid = (
            history_items[i] > 0
        )

        h_cats = history_categories[
            i
        ][valid]

        g = gates[
            i
        ][valid]

        same_mask = (
            (h_cats == tgt_cat)
            &
            (h_cats >= 0)
        )

        diff_mask = (
            (h_cats != tgt_cat)
            &
            (h_cats >= 0)
        )

        if same_mask.any():

            same_means.append(
                float(
                    g[
                        same_mask
                    ].mean()
                )
            )

        if diff_mask.any():

            diff_means.append(
                float(
                    g[
                        diff_mask
                    ].mean()
                )
            )

    return (
        np.array(
            same_means
        ),
        np.array(
            diff_means
        )
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Combined gate analysis figure (Chinese labels)"
    )

    parser.add_argument(
        "--dumps_dir",
        type=str,
        default=os.path.join(
            PROJECT_ROOT,
            "analysis_figures",
            "dumps"
        )
    )

    parser.add_argument(
        "--output_pdf",
        type=str,
        default=""
    )

    parser.add_argument(
        "--output_png",
        type=str,
        default=""
    )

    parser.add_argument(
        "--output_svg",
        type=str,
        default=""
    )

    parser.add_argument(
        "--output_tif",
        type=str,
        default=""
    )

    parser.add_argument(
        "--output_jpg",
        type=str,
        default=""
    )

    parser.add_argument(
        "--n_heatmap",
        type=int,
        default=100
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42
    )

    args = parser.parse_args()


    rng = np.random.RandomState(
        args.random_seed
    )


    # -----------------------------------------------------------------------
    # Global matplotlib rc
    # -----------------------------------------------------------------------

    plt.rcParams.update({

        "font.family": "serif",

        "font.serif": [
            "Times New Roman",
            "DejaVu Serif"
        ],

        "mathtext.fontset": "stix",

        "axes.unicode_minus": False,

    })

    plt.rcParams[
        "hatch.linewidth"
    ] = 0.6

    plt.rcParams[
        "pdf.fonttype"
    ] = 42

    plt.rcParams[
        "ps.fonttype"
    ] = 42


    # SVG uses paths to avoid possible font/layout mismatch
    plt.rcParams[
        "svg.fonttype"
    ] = "path"


    print(
        f"[Font] Chinese: {SIMSUN_PATH}"
    )

    print(
        f"[Font] English: {TIMES_PATH}"
    )


    # -----------------------------------------------------------------------
    # Load all three datasets
    # -----------------------------------------------------------------------

    all_data = {}

    for ds in DATASETS:

        p = os.path.join(
            args.dumps_dir,
            f"{ds}_gate_analysis_dump.npz"
        )

        if not os.path.exists(
            p
        ):

            print(
                f"[WARN] {p} not found, skipping {ds}"
            )

            continue

        all_data[
            ds
        ] = np.load(
            p,
            allow_pickle=True
        )


    if "beauty" not in all_data:

        raise FileNotFoundError(
            "beauty_gate_analysis_dump.npz not found."
        )


    for ds in DATASETS:

        if ds not in all_data:

            raise FileNotFoundError(
                f"{ds}_gate_analysis_dump.npz not found."
            )


    # =======================================================================
    # Pre-compute category gate statistics
    # =======================================================================

    stats = {}

    for ds in DATASETS:

        d = all_data[
            ds
        ]

        same_arr, diff_arr = compute_category_gates(

            d[
                "gates"
            ],

            d[
                "history_items"
            ],

            d[
                "history_categories"
            ],

            d[
                "target_categories"
            ],

        )


        same_mean = float(
            np.mean(
                same_arr
            )
        )

        diff_mean = float(
            np.mean(
                diff_arr
            )
        )

        delta = (
            same_mean
            -
            diff_mean
        )


        same_lo, same_hi = bootstrap_ci(
            same_arr,
            rng=rng
        )

        diff_lo, diff_hi = bootstrap_ci(
            diff_arr,
            rng=rng
        )


        stats[
            ds
        ] = {

            "same_mean":
                same_mean,

            "same_lo":
                same_lo,

            "same_hi":
                same_hi,

            "same_n":
                len(
                    same_arr
                ),

            "diff_mean":
                diff_mean,

            "diff_lo":
                diff_lo,

            "diff_hi":
                diff_hi,

            "diff_n":
                len(
                    diff_arr
                ),

            "delta":
                delta,

        }


        print(
            f"=== {ds} ==="
        )

        print(
            f"  Same-cat: "
            f"mean={same_mean:.4f}  "
            f"CI=[{same_lo:.4f}, {same_hi:.4f}]  "
            f"n={len(same_arr)}"
        )

        print(
            f"  Diff-cat: "
            f"mean={diff_mean:.4f}  "
            f"CI=[{diff_lo:.4f}, {diff_hi:.4f}]  "
            f"n={len(diff_arr)}"
        )

        print(
            f"  Delta:    "
            f"{delta:+.4f}"
        )

        print()


    # =======================================================================
    # Figure: 1 × 2 horizontal
    # =======================================================================

    FIG_W = (
        13.2 / 2.54
    )

    FIG_H = (
        5.7 / 2.54
    )


    fig, (
        ax_a,
        ax_b
    ) = plt.subplots(

        1,

        2,

        figsize=(
            FIG_W,
            FIG_H
        )

    )


    fig.subplots_adjust(

        wspace=0.52,

        left=0.09,

        right=0.98,

        top=0.88,

        bottom=0.19

    )


    # =======================================================================
    # (a) Beauty gate heatmap
    # =======================================================================

    bd = all_data[
        "beauty"
    ]


    idx = select_heatmap_samples(

        bd[
            "gates"
        ],

        bd[
            "history_items"
        ],

        n_select=args.n_heatmap,

        min_history=10

    )


    K = len(
        idx
    )


    print(
        f"Heatmap: {K} beauty samples selected"
    )


    gates_sel = bd[
        "gates"
    ][idx]


    hist_sel = bd[
        "history_items"
    ][idx]


    valid_sel = (
        hist_sel > 0
    )


    L = gates_sel.shape[
        1
    ]


    gates_disp = np.where(

        valid_sel,

        gates_sel,

        np.nan

    )


    im = ax_a.imshow(

        gates_disp,

        aspect="auto",

        cmap="Greys",

        origin="upper",

        vmin=0.0,

        vmax=1.0,

        interpolation="nearest",

        extent=[
            0.5,
            L + 0.5,
            K,
            0
        ],

    )


    ax_a.set_xlabel(

        "历史行为位置",

        fontsize=FONT_SIZE,

        labelpad=2

    )


    ax_a.set_ylabel(

        "测试样本",

        fontsize=FONT_SIZE,

        labelpad=2

    )


    ax_a.set_title(

        "（a）门控权重热力图",

        fontsize=FONT_SIZE,

        pad=3,

        fontweight="normal"

    )


    ax_a.xaxis.label.set_fontproperties(
        FONT_CN
    )

    ax_a.yaxis.label.set_fontproperties(
        FONT_CN
    )

    ax_a.title.set_fontproperties(
        FONT_CN
    )


    set_journal_style(
        ax_a
    )


    ax_a.set_xticks(
        [
            1,
            5,
            10,
            15,
            20
        ]
    )


    ax_a.set_xlim(
        0.5,
        L + 0.5
    )


    ax_a.set_yticks(
        []
    )


    # x-axis numbers use Times New Roman
    set_tick_font(

        ax_a,

        x_axis=True,

        y_axis=False

    )


    cbar = fig.colorbar(

        im,

        ax=ax_a,

        fraction=0.035,

        pad=0.03,

        ticks=[
            0,
            0.5,
            1.0
        ]

    )


    cbar.ax.tick_params(

        labelsize=FONT_SIZE,

        width=0.5,

        pad=1

    )


    cbar.outline.set_linewidth(
        0.5
    )


    cbar.set_label(

        "门控权重",

        fontsize=FONT_SIZE,

        labelpad=1

    )


    cbar.ax.yaxis.label.set_fontproperties(
        FONT_CN
    )


    # Colorbar numbers use Times New Roman
    for label in cbar.ax.get_yticklabels():

        label.set_fontproperties(
            FONT_EN
        )


    # =======================================================================
    # (b) Same-category vs different-category
    # =======================================================================

    n_ds = len(
        DATASETS
    )


    x = np.arange(
        n_ds
    )


    bar_w = 0.30


    hatches_pair = {

        "same": "///",

        "diff": "\\\\"

    }


    labels_pair = {

        "same":
            "目标同类别",

        "diff":
            "目标异类别"

    }


    bar_containers = {}


    for idx_key, key in enumerate(
        [
            "same",
            "diff"
        ]
    ):

        means = []

        yerr_lo = []

        yerr_hi = []


        for ds in DATASETS:

            s = stats[
                ds
            ]

            m = s[
                f"{key}_mean"
            ]

            lo = s[
                f"{key}_lo"
            ]

            hi = s[
                f"{key}_hi"
            ]


            means.append(
                m
            )

            yerr_lo.append(
                m - lo
            )

            yerr_hi.append(
                hi - m
            )


        offset = (
            idx_key - 0.5
        ) * bar_w


        bar_containers[
            key
        ] = ax_b.bar(

            x + offset,

            means,

            bar_w,

            yerr=[
                yerr_lo,
                yerr_hi
            ],

            color="white",

            edgecolor="black",

            linewidth=0.4,

            hatch=hatches_pair[
                key
            ],

            error_kw=dict(

                linewidth=0.4,

                capsize=2.2

            ),

            label=labels_pair[
                key
            ],

            zorder=3,

        )


    # -----------------------------------------------------------------------
    # Y-axis auto-scale
    # -----------------------------------------------------------------------

    all_tops = []


    for ds_name in DATASETS:

        s = stats[
            ds_name
        ]

        all_tops.append(

            max(

                s[
                    "same_mean"
                ],

                s[
                    "diff_mean"
                ]

            )

        )


    y_top = (
        max(
            all_tops
        )
        +
        0.08
    )


    ax_b.set_ylim(
        0,
        y_top
    )


    ax_b.set_xticks(
        x
    )


    ax_b.set_xticklabels(

        [
            DS_LABELS[
                ds
            ]
            for ds in DATASETS
        ],

        fontsize=FONT_SIZE

    )


    ax_b.set_ylabel(

        "平均门控权重",

        fontsize=FONT_SIZE,

        labelpad=2

    )


    ax_b.set_title(

        "（b）目标相关门控权重对比",

        fontsize=FONT_SIZE,

        pad=3,

        fontweight="normal"

    )


    ax_b.yaxis.label.set_fontproperties(
        FONT_CN
    )


    ax_b.title.set_fontproperties(
        FONT_CN
    )


    set_journal_style(
        ax_b
    )


    # Dataset names and y-axis values use Times New Roman
    set_tick_font(

        ax_b,

        x_axis=True,

        y_axis=True

    )


    # =======================================================================
    # Legend
    # =======================================================================

    leg_handles = [

        Patch(

            facecolor="white",

            edgecolor="black",

            linewidth=0.5,

            hatch="////",

            label=labels_pair[
                "same"
            ]

        ),

        Patch(

            facecolor="white",

            edgecolor="black",

            linewidth=0.5,

            hatch="\\\\\\\\",

            label=labels_pair[
                "diff"
            ]

        ),

    ]


    legend = fig.legend(

        handles=leg_handles,

        fontsize=FONT_SIZE,

        loc="lower center",

        bbox_to_anchor=(
            0.80,
            -0.005
        ),

        ncol=2,

        frameon=False,

        handlelength=2.2,

        handleheight=1.1,

        handletextpad=0.45,

        labelspacing=0.2,

        columnspacing=0.9,

        prop=FONT_CN,

    )


    # =======================================================================
    # Save
    # =======================================================================

    output_dir = os.path.join(

        PROJECT_ROOT,

        "analysis_figures",

        "figures"

    )


    os.makedirs(
        output_dir,
        exist_ok=True
    )


    # -----------------------------------------------------------------------
    # Output paths
    # -----------------------------------------------------------------------

    if args.output_pdf:

        out_pdf = args.output_pdf

    else:

        out_pdf = os.path.join(

            output_dir,

            "gate_analysis_combined_cn.pdf"

        )


    if args.output_png:

        out_png = args.output_png

    else:

        out_png = os.path.join(

            output_dir,

            "gate_analysis_combined_cn.png"

        )


    if args.output_svg:

        out_svg = args.output_svg

    else:

        out_svg = os.path.join(

            output_dir,

            "gate_analysis_combined_cn.svg"

        )


    if args.output_tif:

        out_tif = args.output_tif

    else:

        out_tif = os.path.join(

            output_dir,

            "gate_analysis_combined_cn.tif"

        )


    if args.output_jpg:

        out_jpg = args.output_jpg

    else:

        out_jpg = os.path.join(

            output_dir,

            "gate_analysis_combined_cn.jpg"

        )


    # -----------------------------------------------------------------------
    # Save PDF
    # Vector format
    # -----------------------------------------------------------------------

    try:

        fig.savefig(

            out_pdf,

            bbox_inches="tight",

            pad_inches=0.05,

            facecolor="white"

        )

        print(
            f"Saved PDF: {out_pdf}"
        )

    except Exception as e:

        print(
            f"[INFO] PDF output skipped: {e}"
        )


    # -----------------------------------------------------------------------
    # Save SVG
    # Vector format
    # -----------------------------------------------------------------------

    try:

        fig.savefig(

            out_svg,

            bbox_inches="tight",

            pad_inches=0.05,

            facecolor="white"

        )

        print(
            f"Saved SVG: {out_svg}"
        )

    except Exception as e:

        print(
            f"[INFO] SVG output skipped: {e}"
        )


    # -----------------------------------------------------------------------
    # Save PNG
    # 1200 dpi lossless raster image
    # -----------------------------------------------------------------------

    try:

        fig.savefig(

            out_png,

            dpi=1200,

            bbox_inches="tight",

            pad_inches=0.05,

            facecolor="white"

        )

        print(
            f"Saved PNG (1200 dpi): {out_png}"
        )

    except Exception as e:

        print(
            f"[INFO] PNG output skipped: {e}"
        )


    # -----------------------------------------------------------------------
    # Save TIFF
    #
    # Recommended submission format:
    #   1200 dpi
    #   LZW lossless compression
    #
    # TIFF is preferred over JPEG for figures containing:
    #   - text
    #   - thin lines
    #   - axes
    #   - formulas
    #   - heatmaps
    # -----------------------------------------------------------------------

    try:

        fig.savefig(

            out_tif,

            dpi=1200,

            bbox_inches="tight",

            pad_inches=0.05,

            facecolor="white",

            pil_kwargs={

                "compression":
                    "tiff_lzw"

            }

        )

        print(
            f"Saved TIFF (1200 dpi, LZW): {out_tif}"
        )

    except Exception as e:

        print(
            f"[INFO] TIFF output skipped: {e}"
        )


    # -----------------------------------------------------------------------
    # Save JPEG
    #
    # JPEG is only provided as an additional format.
    # TIFF should be preferred for journal submission.
    # -----------------------------------------------------------------------

    try:

        fig.savefig(

            out_jpg,

            dpi=1200,

            bbox_inches="tight",

            pad_inches=0.05,

            facecolor="white",

            pil_kwargs={

                "quality":
                    100,

                "subsampling":
                    0

            }

        )

        print(
            f"Saved JPG (1200 dpi, quality=100): {out_jpg}"
        )

    except Exception as e:

        print(
            f"[INFO] JPG output skipped: {e}"
        )


    plt.close(
        fig
    )


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    main()