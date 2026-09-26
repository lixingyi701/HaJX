"""Create two report figures from existing Q1 cross-scale result tables."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "src/outputs_q1/tables"
FIGURES = ROOT / "src/outputs_q1/figures"
SCALES = ("1M_heldout", "60M", "1B")
LABELS = ("100万参数\n1M留出", "6000万参数\n60M标定", "10亿参数\n1B检验")


def configure() -> None:
    plt.rcParams.update({
        "font.family": "Hiragino Sans GB",
        "font.size": 9,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
    })
    FIGURES.mkdir(parents=True, exist_ok=True)


def ranking_figure() -> None:
    linear = pd.read_csv(TABLES / "T6_cross_scale_validation.csv").set_index("scale")
    tree = pd.read_csv(TABLES / "T6_gbdt_cross_scale.csv").set_index("test_scale")
    y_linear = np.array([linear.loc[s, "spearman_target"] for s in SCALES])
    y_tree = np.array([tree.loc[s, "spearman_target"] for s in SCALES])
    assert np.allclose(y_linear, (0.5225766765850309, 0.46692177843900207, 0.677014652014652))
    assert np.allclose(y_tree, (0.9415341039139391, 0.9054403181506065, 0.7608058608058608))

    fig, ax = plt.subplots(figsize=(6.5, 3.3), layout="constrained")
    x = np.arange(3)
    ax.plot(x, y_linear, "o-", color="#315f83", linewidth=2.1, markersize=5.5, label="线性模型")
    ax.plot(x, y_tree, "s--", color="#bb7140", linewidth=2.1, markersize=5.5, label="树模型 GBDT")
    for y, dy in ((y_linear, -0.055), (y_tree, 0.035)):
        for xi, yi in zip(x, y):
            ax.text(xi, yi + dy, f"{yi:.3f}", ha="center", va="center", fontsize=8.5)
    ax.set_xticks(x, LABELS)
    ax.set_xlim(-0.22, 2.22)
    ax.set_ylim(0.35, 1.02)
    ax.set_yticks((0.4, 0.6, 0.8, 1.0))
    ax.set_ylabel("预测与实测配比优劣顺序的一致性\nSpearman 相关系数（越高越一致）")
    ax.grid(axis="y", color="#d8e0e5", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, ncol=2)
    fig.savefig(FIGURES / "F6_mixture_order_agreement.pdf", bbox_inches="tight")
    plt.close(fig)


def loss_figure() -> None:
    data = pd.read_csv(TABLES / "T6_scale_correction.csv")
    names = {
        "frozen": "冻结1M",
        "power": "截距+缩放修正(幂律)",
    }
    rows = {}
    for scale in ("60M", "1B"):
        group = data[data.scale == scale].set_index("method")
        rows[scale] = {
            "actual": group.loc[names["frozen"], "mean_obs_loss"],
            "frozen": group.loc[names["frozen"], "mean_pred_loss"],
            "corrected": group.loc[names["power"], "mean_pred_loss"],
        }
    assert np.isclose(rows["60M"]["actual"], 3.8229111776185722)
    assert np.isclose(rows["1B"]["corrected"], 2.9963960969125862)

    fig, ax = plt.subplots(figsize=(6.5, 3.4), layout="constrained")
    x = np.arange(2)
    width = 0.23
    for i, (key, label, color, hatch) in enumerate((
        ("actual", "实际交叉熵", "#374550", ""),
        ("frozen", "直接沿用百万参数模型", "#bd7650", "//"),
        ("corrected", "加入尺度修正", "#4b8b83", ".."),
    )):
        vals = [rows[s][key] for s in ("60M", "1B")]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=label, color=color,
                      hatch=hatch, edgecolor="#26313a", linewidth=0.45)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.06, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x, ("6000万参数\n用于标定", "10亿参数\n独立检验"))
    ax.set_ylim(0, 6.0)
    ax.set_ylabel("13个验证域的平均交叉熵 Loss\n（越低越好）")
    ax.grid(axis="y", color="#d8e0e5", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=3,
              frameon=False, fontsize=8)
    fig.savefig(FIGURES / "F6_scale_loss_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    configure()
    ranking_figure()
    loss_figure()
    print("Saved F6_mixture_order_agreement.pdf and F6_scale_loss_comparison.pdf")
