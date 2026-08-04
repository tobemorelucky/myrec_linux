# -*- coding: UTF-8 -*-
"""
plot_target_interest_margin_single_cn.py
========================================
Single-panel dumbbell plot: target_gap vs neg_gap across 3 datasets,
with bootstrap 95% CI error bars and ratio annotations.

Input:
    analysis_figures/dumps/{beauty,ml-1m,toys}_target_interest_analysis_dump.npz

Output:
    analysis_figures/figures/target_interest_margin_single_cn.pdf
    analysis_figures/figures/target_interest_margin_single_cn.png
    analysis_figures/figures/target_interest_margin_single_cn.svg
"""

import os
import sys
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties


# =========================================================
# Project path
# =========================================================
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


# =========================================================
# Configuration
# =========================================================
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


# =========================================================
# Fonts
# =========================================================
FONT_DIR = os.path.join(PROJECT_ROOT, "Fonts")

SIMSUN_PATH = os.path.join(
    FONT_DIR,
    "simsun.ttc",
)

TIMES_PATH = os.path.join(
    FONT_DIR,
    "times.ttf",
)

if not os.path.exists(SIMSUN_PATH):
    raise FileNotFoundError(
        f"未找到宋体字体文件：{SIMSUN_PATH}"
    )

if not os.path.exists(TIMES_PATH):
    raise FileNotFoundError(
        f"未找到 Times New Roman 字体文件：{TIMES_PATH}"
    )

# 中文：宋体六号
FONT_CN = FontProperties(
    fname=SIMSUN_PATH,
    size=FONT_SIZE,
)

# 英文、数字、数据集名称：Times New Roman 六号
FONT_EN = FontProperties(
    fname=TIMES_PATH,
    size=FONT_SIZE,
)


# =========================================================
# Bootstrap confidence interval
# =========================================================
def bootstrap_ci(
    values,
    n_boot=2000,
    ci=95.0,
    rng=None,
):
    """
    使用 Bootstrap 计算均值及其置信区间。

    Returns:
        mean, lower_bound, upper_bound
    """
    if rng is None:
        rng = np.random.RandomState(42)

    n = len(values)

    if n < 2:
        mean_value = (
            float(np.mean(values))
            if n == 1
            else 0.0
        )
        return mean_value, mean_value, mean_value

    means = np.empty(
        n_boot,
        dtype=np.float64,
    )

    for i in range(n_boot):
        sample = rng.choice(
            values,
            size=n,
            replace=True,
        )
        means[i] = np.mean(sample)

    lower_percentile = (
        100.0 - ci
    ) / 2.0

    upper_percentile = (
        100.0 - lower_percentile
    )

    lower_bound = float(
        np.percentile(
            means,
            lower_percentile,
        )
    )

    upper_bound = float(
        np.percentile(
            means,
            upper_percentile,
        )
    )

    mean_value = float(
        np.mean(values)
    )

    return (
        mean_value,
        lower_bound,
        upper_bound,
    )


# =========================================================
# Font helpers
# =========================================================
def set_tick_font(
    ax,
    x_axis=True,
    y_axis=True,
):
    """
    将坐标刻度统一设置为 Times New Roman。
    """
    if x_axis:
        for label in ax.get_xticklabels():
            label.set_fontproperties(FONT_EN)

    if y_axis:
        for label in ax.get_yticklabels():
            label.set_fontproperties(FONT_EN)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Target-interest margin analysis "
            "with Chinese labels"
        )
    )

    parser.add_argument(
        "--dumps_dir",
        type=str,
        default=os.path.join(
            PROJECT_ROOT,
            "analysis_figures",
            "dumps",
        ),
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    rng = np.random.RandomState(
        args.random_seed
    )

    # -----------------------------------------------------
    # Matplotlib global settings
    # -----------------------------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    })

    # PDF 中保留 TrueType 字体
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # 将 SVG 文字转换成矢量轮廓，
    # 避免浏览器或 Word 替换字体后发生文字重叠
    plt.rcParams["svg.fonttype"] = "path"

    print(f"[Font] Chinese: {SIMSUN_PATH}")
    print(f"[Font] English: {TIMES_PATH}")

    # =====================================================
    # Load and compute
    # =====================================================
    stats = {}

    for dataset in DATASETS:
        dump_path = os.path.join(
            args.dumps_dir,
            f"{dataset}_target_interest_analysis_dump.npz",
        )

        if not os.path.exists(dump_path):
            print(
                f"[ERROR] {dump_path} not found"
            )
            sys.exit(1)

        data = np.load(
            dump_path,
            allow_pickle=True,
        )

        target_gap = data[
            "target_gap"
        ].astype(np.float64)

        negative_gap = data[
            "neg_gap"
        ].astype(np.float64)

        k_star = data["k_star"]

        interest_num = data[
            "target_affinity"
        ].shape[1]

        (
            target_mean,
            target_lower,
            target_upper,
        ) = bootstrap_ci(
            target_gap,
            rng=rng,
        )

        (
            negative_mean,
            negative_lower,
            negative_upper,
        ) = bootstrap_ci(
            negative_gap,
            rng=rng,
        )

        ratio = (
            negative_mean / target_mean
            if target_mean > 0
            else float("inf")
        )

        k_distribution = {
            k: int(
                (k_star == k).sum()
            )
            for k in range(interest_num)
        }

        k_fraction = {
            k: value / len(k_star)
            for k, value
            in k_distribution.items()
        }

        stats[dataset] = {
            "t_mean": target_mean,
            "t_lo": target_lower,
            "t_hi": target_upper,
            "n_mean": negative_mean,
            "n_lo": negative_lower,
            "n_hi": negative_upper,
            "ratio": ratio,
            "K": interest_num,
            "k_dist": k_distribution,
            "k_frac": k_fraction,
        }

        # 终端输出，不会进入图片
        print(f"=== {dataset} ===")

        print(
            "  target_gap: "
            f"mean={target_mean:.4f}  "
            f"95% CI=[{target_lower:.4f}, "
            f"{target_upper:.4f}]"
        )

        print(
            "  neg_gap:    "
            f"mean={negative_mean:.4f}  "
            f"95% CI=[{negative_lower:.4f}, "
            f"{negative_upper:.4f}]"
        )

        print(
            "  ratio:      "
            f"neg/target = {ratio:.1f}x"
        )

        for k in range(interest_num):
            print(
                f"  k*={k}:  "
                f"{k_distribution[k]:6d}  "
                f"({k_fraction[k]:.1%})"
            )

        print()

    # =====================================================
    # Figure: single-panel dumbbell plot
    # =====================================================
    FIG_W = 9.8 / 2.54
    FIG_H = 5.8 / 2.54

    fig, ax = plt.subplots(
        figsize=(FIG_W, FIG_H)
    )

    fig.subplots_adjust(
        left=0.25,
        right=0.96,
        top=0.78,
        bottom=0.24,
    )

    dataset_num = len(DATASETS)

    y_positions = np.arange(
        dataset_num
    )[::-1]

    for index, dataset in enumerate(DATASETS):
        result = stats[dataset]
        y_position = y_positions[index]

        # -------------------------------------------------
        # 两个指标之间的水平连接线
        # -------------------------------------------------
        ax.plot(
            [
                result["t_mean"],
                result["n_mean"],
            ],
            [
                y_position,
                y_position,
            ],
            color="black",
            linewidth=0.8,
            zorder=2,
        )

        # -------------------------------------------------
        # 正样本兴趣间隔：空心圆
        # -------------------------------------------------
        ax.errorbar(
            result["t_mean"],
            y_position,
            xerr=[
                [
                    result["t_mean"]
                    - result["t_lo"]
                ],
                [
                    result["t_hi"]
                    - result["t_mean"]
                ],
            ],
            fmt="o",
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.6,
            color="black",
            linewidth=0.5,
            capsize=2.5,
            label=(
                "正样本兴趣间隔"
                if index == 0
                else ""
            ),
            zorder=3,
        )

        # -------------------------------------------------
        # 负样本分离间隔：实心方块
        # -------------------------------------------------
        ax.errorbar(
            result["n_mean"],
            y_position,
            xerr=[
                [
                    result["n_mean"]
                    - result["n_lo"]
                ],
                [
                    result["n_hi"]
                    - result["n_mean"]
                ],
            ],
            fmt="s",
            markersize=5.5,
            markerfacecolor="black",
            markeredgecolor="black",
            markeredgewidth=0.6,
            color="black",
            linewidth=0.5,
            capsize=2.5,
            label=(
                "负样本分离间隔"
                if index == 0
                else ""
            ),
            zorder=3,
        )

        # -------------------------------------------------
        # 比例标注
        # 使用 Times New Roman 显示 × 和数字
        # -------------------------------------------------
        ax.annotate(
            f"×{result['ratio']:.1f}",
            xy=(
                result["n_hi"],
                y_position,
            ),
            xytext=(3, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontproperties=FONT_EN,
            color="black",
        )

    # =====================================================
    # Axis labels
    # =====================================================
    ax.set_yticks(
        y_positions
    )

    ax.set_yticklabels(
        [
            DS_LABELS[dataset]
            for dataset in DATASETS
        ],
        fontsize=FONT_SIZE,
    )

    # 数据集名称使用 Times New Roman
    for label in ax.get_yticklabels():
        label.set_fontproperties(FONT_EN)

    ax.set_ylim(
        -0.55,
        2.35,
    )

    # 横轴中文：宋体六号
    ax.set_xlabel(
        "距离间隔",
        fontproperties=FONT_CN,
        labelpad=6,
    )

    # =====================================================
    # Journal style
    # =====================================================
    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )

    ax.spines["left"].set_linewidth(
        0.5
    )

    ax.spines["bottom"].set_linewidth(
        0.5
    )

    ax.tick_params(
        direction="in",
        width=0.5,
        labelsize=FONT_SIZE,
        pad=3,
    )

    ax.set_facecolor(
        "white"
    )

    # 横轴数值和纵轴数据集名称使用 Times New Roman
    set_tick_font(
        ax,
        x_axis=True,
        y_axis=True,
    )

    # =====================================================
    # Figure-level legend
    # =====================================================
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=5.5,
            label="正样本兴趣间隔",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markerfacecolor="black",
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=5.5,
            label="负样本分离间隔",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.62, 0.98),
        ncol=2,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=1.0,
        labelspacing=0.1,
        borderpad=0.1,

        # 中文图例使用宋体六号
        prop=FONT_CN,
    )

    # =====================================================
    # Save
    # =====================================================
    output_base = os.path.join(
        PROJECT_ROOT,
        "analysis_figures",
        "figures",
        "target_interest_margin_single_cn",
    )

    os.makedirs(
        os.path.dirname(output_base),
        exist_ok=True,
    )

    output_settings = [
        (".pdf", None),
        (".png", 600),
        (".svg", None),
    ]

    for extension, dpi in output_settings:
        output_path = (
            output_base + extension
        )

        try:
            save_options = {
                "bbox_inches": "tight",
                "pad_inches": 0.04,
            }

            if dpi is not None:
                save_options["dpi"] = dpi

            fig.savefig(
                output_path,
                **save_options,
            )

            print(
                f"Saved {output_path}"
            )

        except Exception as error:
            print(
                f"[INFO] {extension} skipped: "
                f"{error}"
            )

    plt.close(fig)


if __name__ == "__main__":
    main()