# -*- coding: utf-8 -*-
"""
q3_02_transition.py — 问题三·Step5/7：结构性转移的定义、识别与敏感性

【定义】记 x*(C) = (N*,D*,Q*) 为预算 C 下的最优解，称 C_crit 处发生结构性转移，若：
  口径 A（KKT 活跃约束集切换，主定义）：
      𝒜(C) = { Q≥Q0 活跃 | 无约束活跃(内点) | Q≤1 活跃 } 在 C_crit 两侧不同。
      解析识别：Φ(C) = [(κ_N A N^-α+κ_D B D^-β+k0)(κN+Δg)] / [βB D^-β g'(Q)] 在 Q=Q0 处穿越 1。
      （Φ = 提质边际收益 / 提质挤占数据的边际损失；Φ<1 ⇒ 不提质是 KKT 点）
  口径 B（份额弹性符号翻转）：e_Q(C) = d ln s_Q / d ln C 由正变负（s_Q 取极大）。
  口径 C（配置比斜率拐点）：固定 Q 时 D*/N* ∝ C^{(α-β)/(α+β)}（常斜率 0.0968）；
      偏差 δ(C) = d ln(D/N)/d ln C − (α−β)/(α+β) 取极值处即转移。
【识别方法】细扫描 + 二分法（精度 1e-3 dex）；三口径 C_crit 相差 < 0.5 dex 视为一致。
【敏感性】L_ctx 五档、Q0 相图、η ±50%、问题二 bootstrap (E,A,B,κ_N,κ_D,k0)。
产出：tables/T2_*.csv, interface/P3_structural_transition.json, P3_sensitivity_Lctx.csv, F5-F7
"""
import json
import numpy as np
import pandas as pd
from q3_00_model import (PAR, ETA, LCTX_CRIT, Q0_MAIN, Q0_WEB, G_LIST, LCTX_C7, TAB, FIG,
                         IFACE, ROOT, loss, solve, c_crit, phi_ratio, N_star_closed,
                         setup_cjk_matplotlib)

Q0S = {"主口径p*加权": Q0_MAIN, "副口径网页pile_cc": Q0_WEB}
SLOPE0 = (PAR["al"] - PAR["be"]) / (PAR["al"] + PAR["be"])


def bisect_phi(g, Lc, Q0, lo=15.0, hi=26.0, it=50):
    """口径 A 的解析识别：Φ(C; Q=Q0) = 1 的根（Φ 随 C 单调增）"""
    if Q0 >= 1 - 1e-10:
        return np.nan
    if not (phi_ratio(10 ** lo, g, Lc, Q0, Q0) < 1
            and phi_ratio(10 ** hi, g, Lc, Q0, Q0) >= 1):
        return np.nan
    for _ in range(it):
        m = (lo + hi) / 2
        if phi_ratio(10 ** m, g, Lc, Q0, Q0) > 1: hi = m
        else: lo = m
    return (lo + hi) / 2


def local_criteria(S, g, Lc, q0name):
    s = S[(S.g == g) & (S.L_ctx == Lc) & (S.Q0口径 == q0name)].sort_values("log10C")
    x = s.log10C.values * np.log(10)
    sQ = s.s_Q.values
    iB = int(np.argmax(sQ))                                # s_Q 极大 = 弹性翻转
    lnDN = np.log(s.DN.values)
    slope = np.gradient(lnDN, x)
    dev = slope - SLOPE0
    iC = int(np.argmax(np.abs(dev)))
    return s.log10C.values[iB], sQ[iB], s.log10C.values[iC], dev[iC]


if __name__ == "__main__":
    S = pd.read_pickle(f"{TAB}/T1_scan.pkl")

    # ============ 1. 三口径 C_crit ============
    rows = []
    for q0name, Q0 in Q0S.items():
        for g in G_LIST:
            for Lc in LCTX_C7:
                leave, leave_status = c_crit(g, Lc, Q0, which="leave", return_status=True)
                full, full_status = c_crit(g, Lc, Q0, which="full", return_status=True)
                cA_leave, cA_full = float(leave[0]), float(full[0])
                cA_phi = bisect_phi(g, Lc, Q0)
                cB, sQmax, cC, devC = local_criteria(S, g, Lc, q0name)
                rows.append(dict(Q0口径=q0name, Q0=Q0, g=g, L_ctx=Lc,
                                 A_leave_Q0=cA_leave, A_reach_Q1=cA_full,
                                 A_leave_status=str(leave_status[0]), A_full_status=str(full_status[0]),
                                 A_width_dex=cA_full - cA_leave, A_phi_analytic=cA_phi,
                                 B_sQ_peak=cB, sQ_peak=sQmax,
                                 C_DN_kink=cC, DN_slope_dev=devC))
    T = pd.DataFrame(rows)
    T["spread_ABC_dex"] = T[["A_leave_Q0", "B_sQ_peak", "C_DN_kink"]].max(1) - \
        T[["A_leave_Q0", "B_sQ_peak", "C_DN_kink"]].min(1)
    T.to_csv(f"{TAB}/T2_ccrit_three_criteria.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(T[T.L_ctx == 2048].round(3).to_string(index=False))
    print("Φ 解析根 vs 数值二分 最大偏差 %.4f dex" % (T.A_phi_analytic - T.A_leave_Q0).abs().max())
    print("三口径最大分歧 %.3f dex" % T.spread_ABC_dex.max())

    # ============ 2. KKT Φ 曲线（验证：Φ 穿 1 处即 Q* 离开 Q0） ============
    Cs = 10 ** np.linspace(17, 23, 61)
    phi = {g: [phi_ratio(c, g, 2048, Q0_WEB, Q0_WEB) for c in Cs] for g in G_LIST}
    phi1 = {g: [phi_ratio(c, g, 2048, 1.0, Q0_WEB) for c in Cs] for g in G_LIST}
    pd.DataFrame({"log10C": np.log10(Cs), **{f"Phi_Q0_{g}": v for g, v in phi.items()},
                  **{f"Phi_Q1_{g}": v for g, v in phi1.items()}}).to_csv(
        f"{TAB}/T2_phi_curve.csv", index=False, encoding="utf-8-sig")

    # ============ 3. Q0 相图 ============
    Q0grid = np.linspace(0.60, 0.995, 80)
    ph = {}
    for g in G_LIST:
        ph[g] = (c_crit(g, 2048, Q0grid, which="leave"), c_crit(g, 2048, Q0grid, which="full"))
    PH = pd.DataFrame({"Q0": Q0grid, **{f"leave_{g}": ph[g][0] for g in G_LIST},
                       **{f"full_{g}": ph[g][1] for g in G_LIST}})
    PH.to_csv(f"{TAB}/T2_phase_Q0.csv", index=False, encoding="utf-8-sig")

    # ============ 4. η ±50% ============
    er = []
    for q0name, Q0 in Q0S.items():
        for g in G_LIST:
            for eta in (1e-4, 2e-4, 3e-4):
                for Lc in (2048, 131072):
                    er.append(dict(Q0口径=q0name, g=g, eta=eta, L_ctx=Lc,
                                   C_crit_leave=float(c_crit(g, Lc, Q0, eta=eta)[0]),
                                   L_ctx_crit=6 / eta))
    ER = pd.DataFrame(er)
    ER.to_csv(f"{TAB}/T2_eta_sensitivity.csv", index=False, encoding="utf-8-sig")

    # ============ 5. bootstrap（问题二联合 bootstrap） ============
    z = np.load(f"{ROOT}/outputs_q2/interface/P2_bootstrap.npz")
    smp, cols = z["samples"], list(z["cols"])
    pb = dict(PAR)
    for c in cols:
        pb[c] = smp[:, cols.index(c)]
    bs = []
    for q0name, Q0 in Q0S.items():
        for g in G_LIST:
            cc = c_crit(g, 2048, np.full(len(smp), Q0), p=pb, which="leave")
            valid = cc[np.isfinite(cc)]
            bs.append(dict(Q0口径=q0name, g=g, point=float(c_crit(g, 2048, Q0)[0]),
                           p05=np.percentile(valid, 5) if len(valid) else np.nan,
                           p50=np.percentile(valid, 50) if len(valid) else np.nan,
                           p95=np.percentile(valid, 95) if len(valid) else np.nan,
                           n_found=len(valid), n_total=len(cc)))
            # 同时检查三档预算下 regime 是否稳定
            for lc in (19, 22, 24):
                r = solve(np.full(len(smp), 10.0 ** lc), g, 2048, np.full(len(smp), Q0), p=pb)
                bs[-1][f"regime_mode_1e{lc}"] = int(np.bincount(r["regime"]).argmax())
                bs[-1][f"regime_agree_1e{lc}"] = float((r["regime"] == np.bincount(r["regime"]).argmax()).mean())
    BS = pd.DataFrame(bs)
    BS.to_csv(f"{TAB}/T2_bootstrap_ccrit.csv", index=False, encoding="utf-8-sig")
    print(BS.round(3).to_string(index=False))

    # ============ 6. L_ctx 敏感性表（接口） ============
    OPT = pd.read_csv(f"{IFACE}/P3_optimal_config.csv")
    base = OPT[OPT.L_ctx == 2048].set_index(["Q0口径", "g", "log10C"])
    sens = OPT.copy()
    idx = list(zip(sens.Q0口径, sens.g, sens.log10C))
    sens["N_ratio_vs2048"] = sens.N_opt.values / base.loc[idx, "N_opt"].values
    sens["dL_vs2048"] = sens.L_opt.values - base.loc[idx, "L_opt"].values
    kap = 6 + ETA * sens.L_ctx
    sens["kappa"] = kap
    sens["N_ratio_analytic"] = ((6 + ETA * 2048) / kap) ** (PAR["be"] / (PAR["al"] + PAR["be"]))
    sens["attn_over_train"] = ETA * sens.L_ctx / 6
    sens = sens.merge(T[["Q0口径", "g", "L_ctx", "A_leave_Q0", "A_reach_Q1"]],
                      on=["Q0口径", "g", "L_ctx"])
    sens.to_csv(f"{IFACE}/P3_sensitivity_Lctx.csv", index=False, encoding="utf-8-sig")
    print("N* 比值 数值 vs 解析 最大相对差（仅无提质开销组）:",
          (sens.loc[sens.share_Q < 1e-9, "N_ratio_vs2048"] /
           sens.loc[sens.share_Q < 1e-9, "N_ratio_analytic"] - 1).abs().max())

    # ============ 7. 陷阱核对 ============
    traps = []
    N_t, D_t = 1.2e-3 * 1e9, 850e9
    traps.append(dict(陷阱="C=1e22 时 N*=1.2e-3B, D*=850B, Loss=3.41",
                      核对=f"该配置仅用预算 {6*N_t*D_t/1e22:.2%}；按本文标度律 L={loss(N_t, D_t, 1.0):.3f}，"
                           f"而真实最优 L={OPT[(OPT.L_ctx==2048)&(OPT.log10C==22)].L_opt.min():.3f}、N*≈5e9"))
    n_corner = int((OPT.regime == "Q0角点(不提质)").sum())
    n_inner = int(OPT.regime.str.startswith("内点").sum())
    sub = OPT[OPT.Q_opt < 1 - 1e-6]
    lcs = "/".join(f"1e{int(v)}" for v in sorted(sub.log10C.unique()))
    gs = "/".join(sorted(sub.g.unique()))
    traps.append(dict(陷阱="各档预算 Q*=1，无需比较成本函数",
                      核对=f"210 组最优解中 {n_corner} 组 Q*=Q0（不提质）、{n_inner} 组内点（部分提质），"
                           f"全部出现在 C={lcs} 的{gs}；对数型在同档已提满 ⇒ 成本函数形式决定是否提质"))
    traps.append(dict(陷阱="穷举网格 + 罚系数取 1",
                      核对="约束取等后 D 可解析反解，问题降为无约束二维搜索，无需罚函数；"
                           f"粗网格与本文求解器的 Loss 差 ≤ {pd.read_csv(f'{TAB}/T1_solver_vs_grid.csv').L_gap.max():.1e}，"
                           "但网格无法给出 C_crit 的连续识别"))
    pd.DataFrame(traps).to_csv(f"{TAB}/T2_trap_check.csv", index=False, encoding="utf-8-sig")

    # ============ 接口 JSON ============
    main = T[T.L_ctx == 2048]
    def finite_or_none(x):
        return round(float(x), 3) if np.isfinite(x) else None
    J = dict(definition=dict(
        A="KKT 活跃约束集切换：Q*=Q0 角点 → 内点 → Q*=1 角点；解析判据 Φ(C)=1",
        B="提质份额弹性 d ln s_Q / d ln C 由正转负（s_Q 极大）",
        C="配置比斜率 d ln(D*/N*)/d ln C 偏离固定质量基准 (α-β)/(α+β)=%.4f 的极值点" % SLOPE0),
        L_ctx_crit=LCTX_CRIT,
        C_crit_Lctx2048={f"{r.Q0口径}|{r.g}": dict(leave_Q0=finite_or_none(r.A_leave_Q0),
                                                     leave_status=r.A_leave_status,
                                                     reach_Q1=finite_or_none(r.A_reach_Q1),
                                                     full_status=r.A_full_status,
                                                     phi_analytic=finite_or_none(r.A_phi_analytic),
                                                     sQ_peak=round(r.B_sQ_peak, 3),
                                                     DN_kink=round(r.C_DN_kink, 3))
                         for r in main.itertuples()},
        max_spread_three_criteria_dex=round(float(T.spread_ABC_dex.max()), 3),
        bootstrap=[{k: (finite_or_none(v) if isinstance(v, (float, np.floating)) else v)
                    for k, v in row.items()}
                   for row in BS[["Q0口径", "g", "point", "p05", "p95", "n_found", "n_total"]]
                   .to_dict("records")])
    json.dump(J, open(f"{IFACE}/P3_structural_transition.json", "w"), ensure_ascii=False, indent=2,
              allow_nan=False)

    # ================= 作图 =================
    plt = setup_cjk_matplotlib()
    col = {"指数型": "#d7301f", "幂函数型": "#2b8cbe", "对数型": "#31a354"}

    # F5：三口径识别（副口径，指数型，L=2048）+ Φ 曲线
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for g in G_LIST:
        axes[0].plot(np.log10(Cs), phi[g], color=col[g], label=f"{g} Φ(Q=Q₀)")
        axes[0].plot(np.log10(Cs), phi1[g], color=col[g], ls="--", lw=1, label=f"{g} Φ(Q=1)")
    axes[0].axhline(1, color="k", lw=1); axes[0].set_yscale("log")
    axes[0].set_xlabel("log₁₀ C"); axes[0].set_ylabel("Φ = 提质边际收益 / 挤占数据损失")
    axes[0].set_title("口径A：KKT 判据 Φ 穿越 1"); axes[0].legend(fontsize=7, ncol=2)
    for g in G_LIST:
        s = S[(S.g == g) & (S.L_ctx == 2048) & (S.Q0口径 == "副口径网页pile_cc")].sort_values("log10C")
        axes[1].plot(s.log10C, s.s_Q * 100, color=col[g], label=g)
        x = s.log10C.values * np.log(10)
        axes[2].plot(s.log10C, np.gradient(np.log(s.DN.values), x), color=col[g], label=g)
    axes[1].set_xlabel("log₁₀ C"); axes[1].set_ylabel("提质开销份额 s_Q (%)")
    axes[1].set_title("口径B：s_Q 先升后降（弹性翻转）"); axes[1].legend()
    axes[2].axhline(SLOPE0, color="k", ls=":", label=f"固定质量基准 {SLOPE0:.3f}")
    axes[2].set_xlabel("log₁₀ C"); axes[2].set_ylabel("d ln(D*/N*) / d ln C")
    axes[2].set_title("口径C：配置比斜率偏离基准"); axes[2].legend(fontsize=8)
    axes[2].set_ylim(-1.5, 1.5)
    fig.suptitle("图5  结构性转移的三种识别口径（副口径 Q₀=0.889，L_ctx=2048）")
    fig.tight_layout(); fig.savefig(f"{FIG}/F5_three_criteria.png"); plt.close(fig)

    # F6：Q0 相图
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for g in G_LIST:
        ax.plot(PH.Q0, PH[f"leave_{g}"], color=col[g], lw=2, label=f"{g}：开始提质")
        ax.plot(PH.Q0, PH[f"full_{g}"], color=col[g], lw=1, ls="--", label=f"{g}：提满 Q=1")
    for q0name, Q0 in Q0S.items():
        ax.axvline(Q0, color="gray", ls=":"); ax.text(Q0, 22.6, q0name, rotation=90, fontsize=8,
                                                         ha="right", va="top", color="gray")
    for v in (19, 22, 24):
        ax.axhline(v, color="k", alpha=.12)
    ax.set_xlabel("基线质量 Q₀"); ax.set_ylabel("临界预算 log₁₀ C_crit")
    ax.set_ylim(16.5, 23)
    ax.set_title("图6  结构性转移相图：线下方=不提质，虚线上方=提满")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{FIG}/F6_phase_Q0.png"); plt.close(fig)

    # F7：L_ctx 对 C_crit 的影响 + bootstrap 区间
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for g in G_LIST:
        for q0name, ls in zip(Q0S, ("-", "--")):
            s = T[(T.g == g) & (T.Q0口径 == q0name)]
            axes[0].plot(s.L_ctx, s.A_leave_Q0, color=col[g], ls=ls, marker="o", ms=4,
                         label=f"{g}|{q0name[:3]}")
    axes[0].axvline(LCTX_CRIT, color="purple", ls="-.")
    axes[0].set_xscale("log"); axes[0].set_xlabel("L_ctx（C7 五档）")
    axes[0].set_ylabel("log₁₀ C_crit（开始提质）")
    axes[0].set_title("长上下文让提质更早划算（注意力抬高 D 的单价）"); axes[0].legend(fontsize=7)
    y = np.arange(len(BS))
    axes[1].errorbar(BS.p50, y, xerr=[BS.p50 - BS.p05, BS.p95 - BS.p50], fmt="o", capsize=3)
    axes[1].set_yticks(y); axes[1].set_yticklabels([f"{a[:3]}|{b}" for a, b in zip(BS.Q0口径, BS.g)])
    axes[1].set_xlabel("log₁₀ C_crit（bootstrap 5–95%）")
    axes[1].set_title("问题二参数不确定性下的临界预算")
    fig.suptitle("图7  临界预算的敏感性")
    fig.tight_layout(); fig.savefig(f"{FIG}/F7_ccrit_sensitivity.png"); plt.close(fig)
    print("done")
