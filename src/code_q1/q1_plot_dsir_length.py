"""Plot the existing A1 DSIR length diagnostic without refitting the quality model."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/outputs_q1/tables/T2_dsir_length_diagnostic.csv"
OUTPUT = ROOT / "src/outputs_q1/figures/F2_dsir_length_correction.pdf"
COLUMNS = ("rho_raw_vs_wc", "rho_perword_vs_wc", "rho_residual_vs_wc")
LABELS = ("Raw DSIR", "Per-word DSIR", "Length residual")


def main() -> None:
    data = pd.read_csv(SOURCE)
    assert len(data) == 21 and set(data.domain) == {
        "arxiv", "book", "c4", "commoncrawl", "github", "stackexchange", "wikipedia"
    }
    values = data.loc[:, COLUMNS].abs().to_numpy()
    medians = np.median(values, axis=0)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(6.5, 3.15), layout="constrained")
    for row in values:
        ax.plot(range(3), row, color="#b7c0c8", linewidth=0.6, alpha=0.65, zorder=1)
    colors = ("#43566b", "#6386a2", "#2c817a")
    for stage in range(3):
        ax.scatter(np.full(len(values), stage), values[:, stage], s=15,
                   color=colors[stage], edgecolors="white", linewidths=0.35, zorder=2)
        ax.plot([stage - 0.19, stage + 0.19], [medians[stage]] * 2,
                color="#151b20", linewidth=2.7, zorder=3)
        ax.text(stage, 1.045, f"median {medians[stage]:.3f}", ha="center", va="bottom",
                color="#151b20", fontsize=8.5)
    ax.annotate("book: 0.503", xy=(2, values[(data.domain == "book").to_numpy(), 2].max()),
                xytext=(1.43, 0.48), arrowprops={"arrowstyle": "-", "color": "#58646c", "lw": 0.8},
                fontsize=8, color="#3c484f")
    ax.set_xticks(range(3), LABELS)
    ax.set_xlim(-0.38, 2.38)
    ax.set_ylim(0, 1.13)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("|Spearman correlation with word count|")
    ax.grid(axis="y", color="#dbe1e5", linewidth=0.6)
    ax.set_axisbelow(True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT}; medians: {medians.round(4).tolist()}")


if __name__ == "__main__":
    main()
