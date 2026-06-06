import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 示例数据
# ----------------------------
datasets = ["ML-1M", "Beauty", "Movies & TV"]

left_titles = ["MINER", "MINER", "MINER"]
right_titles = ["TiMiRec", "TiMiRec", "TiMiRec"]

# 每行一个数据集，列分别是 [Recall_baseline, Recall_ours, NDCG_baseline, NDCG_ours]
miner_data = {
    "ML-1M": [0.387, 0.411, 0.221, 0.240],
    "Beauty": [0.172, 0.183, 0.101, 0.110],
    "Movies & TV": [0.337, 0.358, 0.201, 0.219],
}

timirec_data = {
    "ML-1M": [0.428, 0.445, 0.252, 0.267],
    "Beauty": [0.201, 0.199, 0.112, 0.119],
    "Movies & TV": [0.342, 0.364, 0.205, 0.223],
}

# ----------------------------
# 画图参数
# ----------------------------
fig, axes = plt.subplots(3, 2, figsize=(8, 9))
bar_width = 0.24
x = np.array([0, 1])   # Recall@10, NDCG@10

# 配色
c1 = "#e6a57a"   # 左图 baseline
c2 = "#5b8fc2"   # ours
c3 = "#edcc63"   # 右图 baseline

def add_gain_text(ax, x_center, y_top, base, ours):
    gain = (ours - base) / base * 100
    arrow = "▲" if gain >= 0 else "▼"
    color = "#4f7f35" if gain >= 0 else "#cc2f2f"
    ax.text(x_center, y_top + 0.01, f"{arrow}{gain:.2f}%", ha="center",
            va="bottom", fontsize=11, color=color, fontweight="bold")

for r, ds in enumerate(datasets):
    # -------- 左列：PoM-MINER --------
    ax = axes[r, 0]
    rb, ro, nb, no = miner_data[ds]

    ax.bar(x - bar_width/2, [rb, nb], width=bar_width, color=c1, label="MINER")
    ax.bar(x + bar_width/2, [ro, no], width=bar_width, color=c2, label="PoM-MINER")

    ax.set_title(ds, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(["Recall@10", "NDCG@10"], fontsize=11)
    ax.grid(axis="both", linestyle="-", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, frameon=True)

    ymin = 0.0 if ds != "ML-1M" else 0.1
    ymax = max(rb, ro, nb, no) + 0.09
    ax.set_ylim(ymin, ymax)

    add_gain_text(ax, x[0], max(rb, ro), rb, ro)
    add_gain_text(ax, x[1], max(nb, no), nb, no)

    # -------- 右列：PoM-TiMiRec --------
    ax = axes[r, 1]
    rb, ro, nb, no = timirec_data[ds]

    ax.bar(x - bar_width/2, [rb, nb], width=bar_width, color=c3, label="TiMiRec")
    ax.bar(x + bar_width/2, [ro, no], width=bar_width, color=c2, label="PoM-TiMiRec")

    ax.set_title(ds, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(["Recall@10", "NDCG@10"], fontsize=11)
    ax.grid(axis="both", linestyle="-", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, frameon=True)

    ymin = 0.0 if ds != "ML-1M" else 0.1
    ymax = max(rb, ro, nb, no) + 0.09
    ax.set_ylim(ymin, ymax)

    add_gain_text(ax, x[0], max(rb, ro), rb, ro)
    add_gain_text(ax, x[1], max(nb, no), nb, no)

plt.tight_layout()
plt.savefig("bar_comparison.pdf", bbox_inches="tight")
plt.show()