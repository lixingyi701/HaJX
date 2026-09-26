# -*- coding: utf-8 -*-
"""配比主模型 clr+Huber 在 1M 留出集（A6/A7）上的拟合可视化：逐域预测 vs 实测对数交叉熵。"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q1_00_common import CACHE, RT, FIGS, clr, predict_lin, r2_score, setup_cjk_matplotlib


def main() -> None:
    d = np.load(f"{CACHE}/step5_mixture.npz", allow_pickle=True)
    B = d["B_clr"]                      # (13, 18): [intercept] + 17 clr 系数
    DOM13 = [str(x) for x in d["DOM13"]]
    DOM17 = [str(x) for x in d["DOM17"]]

    M = pd.read_csv(f"{RT}/test_mixture_1m.csv")
    L = pd.read_csv(f"{RT}/test_pile_loss_1m.csv")
    df = M.merge(L, on="index", how="inner")
    pcols = [c for c in M.columns if c != "index"]
    lcols = [c for c in L.columns if c != "index"]
    P = df[pcols].to_numpy(dtype=float)
    Y = df[lcols].to_numpy(dtype=float)

    Z = clr(P)
    Yh_ln = np.stack([predict_lin(B[v], Z) for v in range(B.shape[0])], axis=1)   # (n,13)
    Y_ln = np.log(Y)

    plt = setup_cjk_matplotlib()
    fig, axes = plt.subplots(4, 4, figsize=(11.5, 8.8), layout="constrained")
    axes = axes.ravel()
    for v, ax in enumerate(axes):
        if v >= len(DOM13):
            ax.axis("off")
            continue
        x, y = Y_ln[:, v], Yh_ln[:, v]
        ax.scatter(x, y, s=9, alpha=0.5, color="#315f83", linewidths=0, zorder=2)
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "--", color="#bb7140", linewidth=1.1, zorder=3)
        ax.set_title(f"{DOM13[v]}　R²={r2_score(x, y):.2f}", fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("clr+Huber 配比主模型在 1M 留出集（A6/A7，256 组）上的逐域拟合", fontsize=10.5)
    fig.supxlabel("实测 ln Loss", fontsize=9)
    fig.supylabel("预测 ln Loss", fontsize=9)
    fig.savefig(f"{FIGS}/F5_mixture_fit.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    r2_ln = [r2_score(Y_ln[:, v], Yh_ln[:, v]) for v in range(len(DOM13))]
    Yh = np.exp(Yh_ln)
    r2_raw = [r2_score(Y[:, v], Yh[:, v]) for v in range(len(DOM13))]
    print("逐域 R²(ln):", [round(x, 3) for x in r2_ln])
    print("逐域 R²(原始 Loss):", [round(x, 3) for x in r2_raw])
    print(f"R²(ln) 均值={np.mean(r2_ln):.4f}；R²(原始) 均值={np.mean(r2_raw):.4f}")
    print("Saved F5_mixture_fit.png")


if __name__ == "__main__":
    main()
