# -*- coding: UTF-8 -*-
"""
plot_semantic_alignment_preservation.py

不修改原有指标计算方式，仅调整图4的绘制方式：
1. 图(a)明确为“同物品对齐间隔”；
2. 图例改为“原始语义间隔 / 对齐语义间隔”；
3. 图(b)保留“协同保持得分 / 语义注入得分”；
4. 去掉物品级标准差误差棒，改为柱顶标注均值；
5. 仍在终端输出 mean/std，便于核对数据。
"""

import os
import argparse
import numpy as np
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

DATASETS = ["beauty", "ml-1m", "toys"]

GAMMAS = {
    "beauty": 0.10,
    "ml-1m": 0.08,
    "toys": 0.05,
}

DS_LABELS = {
    "beauty": "Beauty",
    "ml-1m": "ML-1M",
    "toys": "Toys&Games",
}

FONT_SIZE = 6

FONT_DIR = os.path.join(PROJECT_ROOT, "Fonts")
SIMSUN_PATH = os.path.join(FONT_DIR, "simsun.ttc")
TIMES_PATH = os.path.join(FONT_DIR, "times.ttf")

if not os.path.exists(SIMSUN_PATH):
    raise FileNotFoundError(f"Chinese font not found: {SIMSUN_PATH}")

if not os.path.exists(TIMES_PATH):
    raise FileNotFoundError(f"English font not found: {TIMES_PATH}")

FONT_CN = FontProperties(fname=SIMSUN_PATH, size=FONT_SIZE)
FONT_EN = FontProperties(fname=TIMES_PATH, size=FONT_SIZE)
FONT_VALUE = FontProperties(fname=TIMES_PATH, size=5.5)


def shuffled_neg_indices(n, rng):
    """生成不与自身匹配的随机负样本索引。"""
    perm = rng.permutation(n)
    self_mask = perm == np.arange(n)

    if self_mask.any():
        idx = np.where(self_mask)[0]
        perm[idx] = np.roll(perm[idx], 1)

        if n == 1:
            return perm

        still = perm == np.arange(n)

        if still.any():
            for i in np.where(still)[0]:
                j = (i + 1) % n
                perm[i], perm[j] = perm[j], perm[i]

        assert not (perm == np.arange(n)).any(), \
            "Failed to generate derangement"

    return perm


def cos_sim(a, b):
    """逐行余弦相似度。"""
    a_n = a / (
        np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    )
    b_n = b / (
        np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    )
    return (a_n * b_n).sum(axis=1)


def compute_all(data, gamma, rng):
    """保持原脚本指标计算逻辑不变。"""

    e_cf = data["e_cf"].astype(np.float64)
    s_raw = data["s_raw"].astype(np.float64)
    s_aligned = data["s_aligned"].astype(np.float64)

    n, d_cf = e_cf.shape
    d_llm = s_raw.shape[1]

    # 残差融合
    e_sem = e_cf + gamma * s_aligned

    # 随机异物品索引
    neg_idx = shuffled_neg_indices(n, rng)

    # 1. 对齐语义的同物品对齐间隔
    pos_aligned = cos_sim(s_aligned, e_cf)
    neg_aligned = cos_sim(s_aligned, e_cf[neg_idx])
    gap_aligned = pos_aligned - neg_aligned

    # 2. 原始语义的同物品对齐间隔
    if d_llm == d_cf:
        s_raw_proj = s_raw
        pca_note = "s_raw dim == e_cf dim, no PCA needed"
    else:
        pca = PCA(
            n_components=d_cf,
            random_state=rng.randint(0, 2**31)
        )
        s_raw_proj = pca.fit_transform(s_raw)

        explained_var = float(
            pca.explained_variance_ratio_.sum()
        )

        pca_note = (
            f"s_raw PCA {d_llm}→{d_cf}, "
            f"explained_var={explained_var:.4f}"
        )

    pos_raw = cos_sim(s_raw_proj, e_cf)
    neg_raw = cos_sim(s_raw_proj, e_cf[neg_idx])
    gap_raw = pos_raw - neg_raw

    # 3. 协同保持得分
    preserve = cos_sim(e_sem, e_cf)

    # 4. 语义注入得分
    inject = (
        cos_sim(e_sem, s_aligned)
        - cos_sim(e_cf, s_aligned)
    )

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

    ax.tick_params(
        direction="in",
        width=0.5,
        labelsize=FONT_SIZE,
        pad=1.5
    )

    ax.set_facecolor("white")


def set_tick_font(ax):
    for label in ax.get_xticklabels():
        label.set_fontproperties(FONT_EN)

    for label in ax.get_yticklabels():
        label.set_fontproperties(FONT_EN)


def add_value_labels(ax, bars, values):
    """柱顶标注均值。"""
    y_min, y_max = ax.get_ylim()
    offset = max((y_max - y_min) * 0.025, 0.002)

    for bar, value in zip(bars, values):
        if value >= 0:
            y = value + offset
            va = "bottom"
        else:
            y = value - offset
            va = "top"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{value:.3f}",
            ha="center",
            va=va,
            fontproperties=FONT_VALUE,
            fontsize=5.5,
            clip_on=False
        )


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Alignment Preservation Analysis"
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
        "--output",
        type=str,
        default=""
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "DejaVu Serif"
        ],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    })

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "path"

    rng = np.random.RandomState(args.random_seed)

    print(f"[Font] Chinese: {SIMSUN_PATH}")
    print(f"[Font] English: {TIMES_PATH}")

    all_res = {}

    for ds in DATASETS:
        npz_path = os.path.join(
            args.dumps_dir,
            f"{ds}_embedding_alignment_dump.npz"
        )

        if not os.path.exists(npz_path):
            print(
                f"[WARN] {npz_path} not found, "
                f"skipping {ds}"
            )
            continue

        data = np.load(
            npz_path,
            allow_pickle=True
        )

        res = compute_all(
            data,
            GAMMAS[ds],
            rng
        )

        all_res[ds] = res

        print(
            f"\n=== {ds} "
            f"(gamma={GAMMAS[ds]}) ==="
        )

        print(
            f"  PCA note: {res['pca_note']}"
        )

        for key in [
            "gap_raw",
            "gap_aligned",
            "preserve",
            "inject"
        ]:
            mean = float(
                np.mean(res[key])
            )

            std = float(
                np.std(res[key])
            )

            print(
                f"  {key:<11} "
                f"mean={mean:.4f}, "
                f"std={std:.4f}"
            )

    for ds in DATASETS:
        if ds not in all_res:
            raise FileNotFoundError(
                f"Missing result for dataset: {ds}"
            )

    # 13 cm × 5.7 cm
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(
            13.0 / 2.54,
            5.7 / 2.54
        )
    )

    fig.subplots_adjust(
        wspace=0.42,
        left=0.10,
        right=0.96,
        top=0.70,
        bottom=0.18
    )

    x = np.arange(
        len(DATASETS)
    )

    bar_w = 0.32

    bar_kw = {
        "color": "white",
        "edgecolor": "black",
        "linewidth": 0.5,
        "zorder": 3,
    }

    leg_kw = {
        "fontsize": FONT_SIZE,
        "loc": "upper center",
        "bbox_to_anchor": (0.5, 1.13),
        "ncol": 2,
        "borderaxespad": 0,
        "frameon": False,
        "handlelength": 1.35,
        "handletextpad": 0.35,
        "columnspacing": 0.8,
        "prop": FONT_CN,
    }
    # ============================================================
    # (a) 同物品对齐间隔
    # ============================================================

    means_raw = [
        float(np.mean(all_res[ds]["gap_raw"]))
        for ds in DATASETS
    ]

    means_aligned = [
        float(np.mean(all_res[ds]["gap_aligned"]))
        for ds in DATASETS
    ]

    # 左：原始语义间隔
    # 使用纯黑色，提高非常小的柱体的可见性
    bars_raw = ax_a.bar(
        x - bar_w / 2,
        means_raw,
        bar_w,
        color="black",
        edgecolor="black",
        linewidth=0.6,
        label="原始语义间隔",
        zorder=3
    )

    # 右：对齐语义间隔
    # 使用纯白色 + 黑色边框
    bars_aligned = ax_a.bar(
        x + bar_w / 2,
        means_aligned,
        bar_w,
        color="white",
        edgecolor="black",
        linewidth=0.8,
        label="对齐语义间隔",
        zorder=3
    )

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(
        [DS_LABELS[ds] for ds in DATASETS],
        fontsize=FONT_SIZE
    )

    ax_a.set_ylabel(
        "对齐间隔",
        fontsize=FONT_SIZE,
        labelpad=2
    )

    ax_a.set_title(
        "（a）同物品对齐间隔",
        fontsize=FONT_SIZE,
        y=1.28,
        pad=0,
        fontweight="normal"
    )

    ax_a.yaxis.label.set_fontproperties(FONT_CN)
    ax_a.title.set_fontproperties(FONT_CN)

    set_journal_style(ax_a)
    set_tick_font(ax_a)

    # 从0开始，更符合“间隔”柱状图的直观含义
    all_a_values = means_raw + means_aligned
    max_a = max(all_a_values)

    if min(all_a_values) >= 0:
        ax_a.set_ylim(0, max_a * 1.15)

    # y=0基准线
    ax_a.axhline(
        y=0,
        color="black",
        linewidth=0.4,
        zorder=1
    )

    # 图(a)不再添加柱顶数值。
    # 原始语义间隔较小，添加数字容易遮挡柱体和坐标轴。

    legend_a = ax_a.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        frameon=False,
        borderaxespad=0,
        handlelength=1.25,
        handleheight=0.8,
        handletextpad=0.35,
        columnspacing=0.8,
        prop=FONT_CN
    )

    for text in legend_a.get_texts():
        text.set_fontproperties(FONT_CN)

    # ============================================================
    # (b) 协同保持与语义注入
    # ============================================================

    keys_b = [
        "preserve",
        "inject"
    ]

    labels_b = {
        "preserve": "协同保持得分",
        "inject": "语义注入得分",
    }

    hatches_b = {
        "preserve": "",
        "inject": "//",
    }

    bars_b = []
    values_b = []

    for idx, key in enumerate(keys_b):
        means = [
            float(
                np.mean(
                    all_res[ds][key]
                )
            )
            for ds in DATASETS
        ]

        offset = (
            idx - 0.5
        ) * bar_w

        bars = ax_b.bar(
            x + offset,
            means,
            bar_w,
            hatch=hatches_b[key],
            label=labels_b[key],
            **bar_kw
        )

        bars_b.append(bars)
        values_b.append(means)

    ax_b.set_xticks(x)

    ax_b.set_xticklabels(
        [
            DS_LABELS[ds]
            for ds in DATASETS
        ],
        fontsize=FONT_SIZE
    )

    ax_b.set_ylabel(
        "指标得分",
        fontsize=FONT_SIZE,
        labelpad=2
    )

    ax_b.set_title(
        "（b）协同保持与语义注入",
        fontsize=FONT_SIZE,
        y=1.26,
        pad=0,
        fontweight="normal"
    )

    ax_b.yaxis.label.set_fontproperties(
        FONT_CN
    )

    ax_b.title.set_fontproperties(
        FONT_CN
    )

    set_journal_style(ax_b)
    set_tick_font(ax_b)

    y_min, y_max = ax_b.get_ylim()
    span = y_max - y_min

    ax_b.set_ylim(
        y_min,
        y_max + span * 0.15
    )

    for bars, values in zip(
        bars_b,
        values_b
    ):
        add_value_labels(
            ax_b,
            bars,
            values
        )

    legend_b = ax_b.legend(
        **leg_kw
    )

    for text in legend_b.get_texts():
        text.set_fontproperties(
            FONT_CN
        )

    # ============================================================
    # 保存
    # ============================================================

    if args.output:
        base = args.output
    else:
        base = os.path.join(
            PROJECT_ROOT,
            "analysis_figures",
            "figures",
            "semantic_alignment_preservation_1x2_cn"
        )

    os.makedirs(
        os.path.dirname(base),
        exist_ok=True
    )

    save_common = {
        "bbox_inches": "tight",
        "pad_inches": 0.04,
        "facecolor": "white",
    }

    fig.savefig(
        base + ".pdf",
        **save_common
    )

    fig.savefig(
        base + ".svg",
        **save_common
    )

    fig.savefig(
        base + ".png",
        dpi=1200,
        **save_common
    )

    fig.savefig(
        base + ".tif",
        dpi=1200,
        pil_kwargs={
            "compression": "tiff_lzw"
        },
        **save_common
    )

    fig.savefig(
        base + ".jpg",
        dpi=1200,
        pil_kwargs={
            "quality": 100,
            "subsampling": 0
        },
        **save_common
    )

    print(
        f"\nSaved figures to: "
        f"{base}.*"
    )

    plt.close(fig)


if __name__ == "__main__":
    main()