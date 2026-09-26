# -*- coding: utf-8 -*-
"""
q4_06_frontier_qr.py — 问题四·QR 前沿交叉验证（独立路线对照，纯增量，不动主链）

定位：与主链 M1b/M1c（先拟合个体、再对前沿状态做双顺序 Shapley）口径独立的交叉验证。
      分位数回归（Koenker & Bassett 1978）直接在 C2×C4 面板上拟合经验能力前沿：
          ln S = a + b_N·ln N + b_T·(t − 2023)，τ = 0.9
      不替代主链，不进 P4 接口，不改动 q4_00–q4_05 / q4_99 的任何产物。

口径：
  - 面板与主链同源（q4_00_common 的 load_c1 × load_c4_lm × join_c1_c4，经 q4_04_decomp.build_panel），
    base/chat 分开，时间轴 = C2 Epoch 发布日；数据截止口径同主链（面板含发布晚于 T0=2025.25 的模型，
    结论为旧锚点面板上的条件结果，主链重跑后须复核）。
  - QR-A：全样本 τ=0.9 分位数回归（主链 M3_q90 的 lnS–lnN 口径版本）。
  - QR-B：先按发布日累计取各时点刷新历史最高分的累计前沿样本，再做 τ=0.9 分位数回归。
  - 贡献推算：在主窗口（及两个对照窗口）前沿状态之间，
      规模贡献 = b_N·ΔlnN_前沿，技术贡献 = b_T·Δt（对数得分尺度），技术占比 = 技术/(规模+技术)；
      得分尺度增量按 规模→技术 顺序复合 exp 近似。
  - 占比是该回归式所解释变化内部的份额，不是观测总增量的精确二分（同 §3.2 既定口径）。
  - 不确定性：按 C4 模型簇 bootstrap 400 次（≥200 的规格），报 5%–95% 经验区间；
      这是经验前沿分位数回归的抽样不确定性，与 C6 桥接 bootstrap 区间是两个独立通道，不得混称。

依赖：不引入新包；分位数回归复用 q4_00_common.quantile_reg（IRLS，与主链 M3 同一实现）。
产物：tables/T9_frontier_qr.csv；figures/F15_frontier_qr.png（不覆盖现有 P4/T5–T8/F9–F14）。
运行：.venv/bin/python src/code_q4/q4_06_frontier_qr.py
"""
import numpy as np
import pandas as pd
from q4_00_common import (TAB, FIG, IFACE, AVG, quantile_reg, setup_cjk_matplotlib)
from q4_04_decomp import build_panel

RNG = np.random.default_rng(20260926)
T_REF = 2023.0
TAU = 0.9
TOPK = 3
WINDOWS = [(2023.5, 2025.0), (2024.0, 2025.0), (2024.5, 2025.25)]   # 与 q4_04 一致；主窗口为第一个
N_BOOT = 400


def frontier_ln_state(d, T):
    """截至时点 T（发布日 < T）累计前沿 TOPK 个模型的均值状态（lnN 为对数平均 = 几何均值口径）"""
    q = d[d.t < T].nlargest(TOPK, AVG)
    return dict(S=float(q[AVG].mean()), lnN=float(q.lnN.mean()), t=float(q.t.mean()),
                models=";".join(q.Model))


def cumulative_frontier(d):
    """按发布日排序后，累计刷新历史最高 Average 的样本（经验前沿点）"""
    dd = d.sort_values("t")
    best = -np.inf
    keep = []
    for i, r in dd.iterrows():
        if r[AVG] > best:
            keep.append(i)
            best = r[AVG]
    return d.loc[keep].copy()


def fit_qr90(d):
    """ln S = a + b_N·ln N + b_T·(t−2023)，τ=0.9；返回 (a, b_N, b_T)"""
    X = np.c_[np.ones(len(d)), d.lnN.values, d.t.values - T_REF]
    return quantile_reg(X, np.log(d[AVG].values), TAU)


def decompose(b, s0, s1):
    """由系数推算规模/技术贡献；返回对数尺度与得分尺度（规模→技术顺序复合）结果"""
    sc_log = float(b[1] * (s1["lnN"] - s0["lnN"]))
    te_log = float(b[2] * (s1["t"] - s0["t"]))
    tot = sc_log + te_log
    share = te_log / tot if abs(tot) > 1e-9 else np.nan
    S0 = s0["S"]
    dS_scale = S0 * (np.exp(sc_log) - 1)
    dS_tech = S0 * np.exp(sc_log) * (np.exp(te_log) - 1)
    return sc_log, te_log, share, dS_scale, dS_tech, S0 * (np.exp(tot) - 1)


def run_one(P, grp):
    """对一个组（base/chat）跑 QR-A 与 QR-B，返回 (rows, boots, fit_objects)"""
    d = P[P.group == grp].reset_index(drop=True)
    d["lnN"] = np.log(d.N)
    fr = cumulative_frontier(d)
    fits = {"QR-A 全样本": (d, fit_qr90(d)), "QR-B 累计前沿样本": (fr, fit_qr90(fr))}
    # 簇 bootstrap：对 QR-A 按 c4_model 簇重抽样；QR-B 在前样本上按 c4_model 簇重抽样
    boots = {}
    for name, (dd, _) in fits.items():
        clusters = dd.c4_model.unique()
        bs = []
        for _ in range(N_BOOT):
            cs = RNG.choice(clusters, len(clusters), replace=True)
            db = pd.concat([dd[dd.c4_model == c] for c in cs], ignore_index=True)
            try:
                bs.append(fit_qr90(db))
            except Exception:
                continue
        boots[name] = np.array(bs)
    # 分窗口分解 + 区间
    rows = []
    for name, (dd, b) in fits.items():
        bs = boots[name]
        for win in WINDOWS:
            s0, s1 = frontier_ln_state(d, win[0]), frontier_ln_state(d, win[1])
            sc, te, share, dS_sc, dS_te, dS_model = decompose(b, s0, s1)
            bshares, bbN, bbT = [], [], []
            for bb_ in bs:
                try:
                    bshares.append(decompose(bb_, s0, s1)[2])
                    bbN.append(bb_[1]); bbT.append(bb_[2])
                except Exception:
                    continue
            bshares = np.array(bshares); bshares = bshares[np.isfinite(bshares)]
            rows.append(dict(
                group=grp, method=name, window=f"{win[0]:.2f}→{win[1]:.2f}",
                n_fit=len(dd), n_cluster_fit=dd.c4_model.nunique(), n_boot_ok=len(bs),
                a=float(b[0]), b_N=float(b[1]), b_T=float(b[2]),
                b_N_p05=float(np.percentile(bbN, 5)) if len(bbN) else np.nan,
                b_N_p95=float(np.percentile(bbN, 95)) if len(bbN) else np.nan,
                b_T_p05=float(np.percentile(bbT, 5)) if len(bbT) else np.nan,
                b_T_p95=float(np.percentile(bbT, 95)) if len(bbT) else np.nan,
                S_start=s0["S"], S_end=s1["S"], dS_obs=s1["S"] - s0["S"],
                dlnN_frontier=s1["lnN"] - s0["lnN"], dt=s1["t"] - s0["t"],
                scale_contrib_log=sc, tech_contrib_log=te,
                scale_contrib_score=dS_sc, tech_contrib_score=dS_te, dS_model_score=dS_model,
                tech_share=share,
                tech_share_p05=float(np.nanpercentile(bshares, 5)) if len(bshares) else np.nan,
                tech_share_p95=float(np.nanpercentile(bshares, 95)) if len(bshares) else np.nan,
                frontier_start=s0["models"], frontier_end=s1["models"]))
    return rows, fits, d, fr


if __name__ == "__main__":
    P, _e = build_panel()
    all_rows, fit_store, panels = [], {}, {}
    for grp in ("base", "chat"):
        rows, fits, d, fr = run_one(P, grp)
        all_rows += rows
        fit_store[grp] = fits
        panels[grp] = (d, fr)
        for name, (dd, b) in fits.items():
            print(f"[{grp}] {name}: n={len(dd)}, a={b[0]:.3f}, b_N={b[1]:.3f}, b_T={b[2]:.3f}")

    T9 = pd.DataFrame(all_rows)

    # ---------------- 与主链 M1b/M1c/M2/M3 对照（读既有 P4 接口，只读不改） ----------------
    ref_cols = {"M1b": np.nan, "M1c": np.nan, "M2": np.nan, "M3_q90": np.nan}
    try:
        dec = pd.read_csv(f"{IFACE}/P4_decomposition.csv")
        main = dec[(dec.window == "2023.50→2025.00") & (dec.group == "base")].set_index("method")
        for m in ref_cols:
            if m in main.index:
                ref_cols[m] = float(main.loc[m, "tech_share"])
    except Exception as ex:
        print("读取 P4_decomposition.csv 对照失败（对照列置空）:", ex)
    for m, v in ref_cols.items():
        T9[f"ref_{m}_tech_share"] = np.where(T9.group == "base", v, np.nan)

    mainw = T9[T9.window == "2023.50→2025.00"]
    print(mainw[["group", "method", "b_N", "b_T", "tech_share", "tech_share_p05", "tech_share_p95",
                 "ref_M1b_tech_share", "ref_M1c_tech_share"]].round(4).to_string(index=False))

    T9.to_csv(f"{TAB}/T9_frontier_qr.csv", index=False, encoding="utf-8-sig")

    # ---------------- 作图：前沿散点 + QR90 拟合线（分 base/chat） ----------------
    plt = setup_cjk_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, grp in zip(axes, ("base", "chat")):
        d, fr = panels[grp]
        sc_ = ax.scatter(d.lnN / np.log(10), d[AVG], c=d.t, cmap="viridis", s=22, alpha=.65,
                         label=f"{grp} 全部（n={len(d)}）")
        ax.scatter(fr.lnN / np.log(10), fr[AVG], facecolors="none", edgecolors="#d7301f", s=60,
                   lw=1.2, label=f"累计前沿样本（n={len(fr)}）")
        xg = np.linspace(d.lnN.min() / np.log(10) - 0.2, d.lnN.max() / np.log(10) + 0.2, 100)
        for yr, ls in ((2023.5, "--"), (2025.0, "-")):
            for name, (_, b), col in (("QR-A", fit_store[grp]["QR-A 全样本"], "k"),
                                      ("QR-B", fit_store[grp]["QR-B 累计前沿样本"], "#d7301f")):
                ax.plot(xg, np.exp(b[0] + b[1] * xg * np.log(10) + b[2] * (yr - T_REF)),
                        color=col, ls=ls, lw=1.3,
                        label=f"{name} τ={TAU} @t={yr}" if yr == 2025.0 else None)
        ax.set_xlabel("log10 参数量 N")
        ax.set_title(f"{grp}：QR90 经验前沿  (b_N, b_T) "
                     f"QR-A=({fit_store[grp]['QR-A 全样本'][1][1]:.3f}, {fit_store[grp]['QR-A 全样本'][1][2]:.3f})")
        ax.legend(fontsize=7, loc="lower right")
        plt.colorbar(sc_, ax=ax, label="发布时间")
    axes[0].set_ylabel("Average")
    fig.suptitle("图15  分位数回归前沿交叉验证（QR90：ln S = a + b_N·ln N + b_T·(t−2023)）")
    fig.tight_layout(); fig.savefig(f"{FIG}/F15_frontier_qr.png"); plt.close(fig)
    print("done: tables/T9_frontier_qr.csv, figures/F15_frontier_qr.png")
