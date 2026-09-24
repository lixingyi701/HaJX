# -*- coding: utf-8 -*-
"""
q3_01_optimize.py — 问题三·Step1-4：全组合最优解 + 求解器互验 + 主图

组合：g ∈ {指数, 幂函数, 对数} × C ∈ {1e19..1e25 每十倍} × L_ctx ∈ C7 五档 × Q0 ∈ {主 0.988, 副 0.889}
     = 3 × 7 × 5 × 2 = 210 组最优解（赛题建议的 1e19/1e22/1e24 三档全部包含）
互验：① 本求解器 vs 实现思路中的"粗网格穷举"（N 每十倍 40 点 × Q 100 点）
     ② Δg=0 区域 vs 闭式 N* = [αÃ/βB]^{1/(α+β)}(C/κ)^{β/(α+β)}
产出：tables/T1_*.csv, interface/P3_optimal_config.csv, figures/F1-F4
"""
import numpy as np
import pandas as pd
from q3_00_model import (PAR, ETA, LCTX_CRIT, Q0_MAIN, Q0_WEB, G_FUNCS, G_LIST,
                         LCTX_C7, REGIME_NAME, TAB, FIG, IFACE, loss, solve,
                         N_star_closed, setup_cjk_matplotlib)

C_MAIN = 10.0 ** np.arange(19, 26)
Q0S = {"主口径p*加权": Q0_MAIN, "副口径网页c4": Q0_WEB}


def run_grid():
    rows = []
    for q0name, Q0 in Q0S.items():
        for g in G_LIST:
            for Lc in LCTX_C7:
                r = solve(C_MAIN, g, Lc, Q0)
                for i in range(len(C_MAIN)):
                    rows.append(dict(Q0口径=q0name, Q0=Q0, g=g, L_ctx=Lc,
                                     log10C=np.log10(r["C"][i]), N_opt=r["N"][i],
                                     D_opt=r["D"][i], Q_opt=r["Q"][i], L_opt=r["L"][i],
                                     D_over_N=r["DN"][i], share_train=r["s_train"][i],
                                     share_attn=r["s_attn"][i], share_Q=r["s_Q"][i],
                                     regime=REGIME_NAME[int(r["regime"][i])]))
    return pd.DataFrame(rows)


def brute_force(C, g, Lc, Q0, nN_per_dec=40, nQ=100):
    """实现思路 Step4 的粗网格：N∈[1e6,1e13] 每十倍 40 点，Q∈[Q0,1] 100 点"""
    N = 10.0 ** np.linspace(6, 13, 7 * nN_per_dec + 1)[:, None]
    Q = np.linspace(Q0, 1, nQ)[None, :]
    gf = G_FUNCS[g][0]
    dg = np.maximum(gf(Q) - gf(Q0), 0)
    D = C / ((6 + ETA * Lc) * N + dg)
    Lm = loss(N, D, Q)
    i, j = np.unravel_index(np.argmin(Lm), Lm.shape)
    return N[i, 0], Q[0, j], Lm[i, j]


def run_scan():
    """细扫描：log10C ∈ [17, 26] 181 点，供结构性转移分析与作图"""
    Cs = 10.0 ** np.linspace(17, 26, 181)
    out = []
    for q0name, Q0 in Q0S.items():
        for g in G_LIST:
            for Lc in LCTX_C7:
                r = solve(Cs, g, Lc, Q0)
                df = pd.DataFrame({k: v for k, v in r.items()})
                df["log10C"] = np.log10(df.pop("C"))
                df["Q0口径"] = q0name; df["g"] = g; df["L_ctx"] = Lc
                out.append(df)
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    # ---------------- 主网格 ----------------
    T = run_grid()
    T.to_csv(f"{TAB}/T1_optimal_all.csv", index=False, encoding="utf-8-sig")
    T.to_csv(f"{IFACE}/P3_optimal_config.csv", index=False, encoding="utf-8-sig")
    key = T[(T.L_ctx == 2048) & T.log10C.isin([19, 22, 24])]
    key.to_csv(f"{TAB}/T1_key_budgets_Lctx2048.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(key[["Q0口径", "g", "log10C", "N_opt", "D_opt", "Q_opt", "L_opt",
               "D_over_N", "share_Q", "regime"]].to_string(index=False))

    # ---------------- 互验①：粗网格穷举 ----------------
    chk = []
    for q0name, Q0 in Q0S.items():
        for g in G_LIST:
            for lc in (19, 22, 24):
                C = 10.0 ** lc
                Nb, Qb, Lb = brute_force(C, g, 2048, Q0)
                r = solve([C], g, 2048, Q0)
                chk.append(dict(Q0口径=q0name, g=g, log10C=lc, N_grid=Nb, N_solver=r["N"][0],
                                Q_grid=Qb, Q_solver=r["Q"][0], L_grid=Lb, L_solver=r["L"][0],
                                L_gap=Lb - r["L"][0]))
    chk = pd.DataFrame(chk)
    chk.to_csv(f"{TAB}/T1_solver_vs_grid.csv", index=False, encoding="utf-8-sig")
    print("\n求解器 vs 粗网格：最大 L 差(网格-求解器) = %.2e（≥0 说明求解器不劣于网格）"
          % chk.L_gap.max(), " 最小 = %.2e" % chk.L_gap.min())

    # ---------------- 互验②：闭式解（Δg=0 区域） ----------------
    cf = []
    for Lc in LCTX_C7:
        kappa = 6 + ETA * Lc
        for lc in (19, 22, 24):
            C = 10.0 ** lc
            for Q in (Q0_MAIN, 1.0):
                # 固定 Q、无提质开销 => 闭式解严格成立
                r = solve([C], "对数型", Lc, Q0=Q)   # Q0=Q 时 Q=Q0 角点或 Q=1
                if Q == 1.0 or r["regime"][0] == 0:
                    Nc = N_star_closed(C, r["Q"][0], kappa)
                    cf.append(dict(L_ctx=Lc, log10C=lc, Q=r["Q"][0], N_numeric=r["N"][0],
                                   N_closed=Nc, rel_err=abs(r["N"][0] / Nc - 1)))
    cf = pd.DataFrame(cf)
    cf.to_csv(f"{TAB}/T1_closed_form_check.csv", index=False, encoding="utf-8-sig")
    print("闭式解互验：最大相对误差 %.2e（%d 组）" % (cf.rel_err.max(), len(cf)))

    # ---------------- 细扫描 ----------------
    S = run_scan()
    S.to_pickle(f"{TAB}/T1_scan.pkl")
    S.to_csv(f"{TAB}/T1_scan.csv", index=False, encoding="utf-8-sig")

    # ================= 作图 =================
    plt = setup_cjk_matplotlib()
    col = {"指数型": "#d7301f", "幂函数型": "#2b8cbe", "对数型": "#31a354"}

    # F1：Q*(C) —— 结构性转移最直观的证据
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, (q0name, Q0) in zip(axes, Q0S.items()):
        for g in G_LIST:
            s = S[(S.Q0口径 == q0name) & (S.g == g) & (S.L_ctx == 2048)]
            ax.plot(s.log10C, s.Q, color=col[g], lw=2, label=g)
        ax.axhline(Q0, color="gray", ls=":", lw=1)
        ax.text(17.1, Q0, f" Q₀={Q0:.3f}", va="bottom", fontsize=9, color="gray")
        for v in (19, 22, 24):
            ax.axvline(v, color="k", alpha=.15)
        ax.set_xlabel("log₁₀ C (FLOPs)"); ax.set_ylabel("最优质量 Q*")
        ax.set_title(f"{q0name}（Q₀={Q0:.3f}，L_ctx=2048）")
        ax.legend(loc="lower right")
    fig.suptitle("图1  最优质量 Q* 随预算的变化：低预算不提质、跨过临界预算后跳到 Q=1")
    fig.tight_layout(); fig.savefig(f"{FIG}/F1_Qstar_vs_C.png"); plt.close(fig)

    # F2：成本份额堆叠
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.8), sharex=True, sharey=True)
    for r_, (q0name, Q0) in enumerate(Q0S.items()):
        for c_, g in enumerate(G_LIST):
            ax = axes[r_, c_]
            s = S[(S.Q0口径 == q0name) & (S.g == g) & (S.L_ctx == 2048)]
            ax.stackplot(s.log10C, s.s_train, s.s_attn, s.s_Q,
                         colors=["#9ecae1", "#fdae6b", "#a1d99b"],
                         labels=["训练 6ND", "注意力 ηNDL", "提质 DΔg"])
            ax.set_ylim(0.8, 1.0)
            ax.set_title(f"{g} | {q0name}", fontsize=10)
            if r_ == 1: ax.set_xlabel("log₁₀ C")
            if c_ == 0: ax.set_ylabel("预算份额")
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.suptitle("图2  三项成本占预算份额（纵轴从 0.8 起，放大提质份额的先升后降）")
    fig.tight_layout(); fig.savefig(f"{FIG}/F2_cost_shares.png"); plt.close(fig)

    # F3：N*、D*、D/N
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for g in G_LIST:
        s = S[(S.Q0口径 == "副口径网页c4") & (S.g == g) & (S.L_ctx == 2048)]
        axes[0].plot(s.log10C, np.log10(s.N), color=col[g], label=g)
        axes[1].plot(s.log10C, np.log10(s.D), color=col[g], label=g)
        axes[2].plot(s.log10C, s.DN, color=col[g], label=g)
    s = S[(S.Q0口径 == "副口径网页c4") & (S.g == "对数型") & (S.L_ctx == 2048)]
    kappa = 6 + ETA * 2048
    axes[0].plot(s.log10C, np.log10(N_star_closed(10 ** s.log10C, 1.0, kappa)), "k--", lw=1,
                 label="闭式解(Q=1,Δg=0)")
    axes[0].set_ylabel("log₁₀ N*"); axes[1].set_ylabel("log₁₀ D*"); axes[2].set_ylabel("D*/N*")
    for ax in axes:
        ax.set_xlabel("log₁₀ C"); ax.legend(fontsize=8)
    axes[2].set_title("D*/N*：转移处出现凹陷（提质挤占 D）")
    fig.suptitle("图3  最优规模配置（副口径 Q₀=0.889，L_ctx=2048）")
    fig.tight_layout(); fig.savefig(f"{FIG}/F3_NDratio.png"); plt.close(fig)

    # F4：L_ctx 敏感性（C=1e22，对数型、主口径）+ 解析骨架
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    Lgrid = np.logspace(np.log10(1024), np.log10(262144), 60)
    for lc, ls in zip((19, 22, 24), (":", "-", "--")):
        C = 10.0 ** lc
        Ns = np.array([solve([C], "对数型", L, Q0_MAIN)["N"][0] for L in Lgrid])
        N0 = solve([C], "对数型", 0, Q0_MAIN)["N"][0]
        axes[0].plot(Lgrid, Ns / N0, "k" + ls, lw=1.5, label=f"数值 C=1e{lc}")
    axes[0].plot(Lgrid, (6 / (6 + ETA * Lgrid)) ** (PAR["be"] / (PAR["al"] + PAR["be"])),
                 color="#d7301f", lw=4, alpha=.35, label="解析 [6/(6+ηL)]^{β/(α+β)}")
    axes[0].axvline(LCTX_CRIT, color="purple", ls="-.", lw=1)
    axes[0].text(LCTX_CRIT * 1.08, 0.62, "L_crit=6/η\n=30000", color="purple", fontsize=9)
    for L in LCTX_C7:
        axes[0].axvline(L, color="gray", alpha=.25)
    axes[0].set_xscale("log"); axes[0].set_xlabel("L_ctx"); axes[0].set_ylabel("N*(L_ctx)/N*(L_ctx=0)")
    axes[0].set_title("长上下文 = 预算打折 6/(6+ηL)"); axes[0].legend(fontsize=8)
    M = T[(T.Q0口径 == "主口径p*加权") & (T.g == "对数型")].pivot(index="L_ctx", columns="log10C", values="L_opt")
    im = axes[1].imshow(M.values, aspect="auto", cmap="viridis_r")
    axes[1].set_xticks(range(M.shape[1])); axes[1].set_xticklabels([f"1e{int(c)}" for c in M.columns])
    axes[1].set_yticks(range(M.shape[0])); axes[1].set_yticklabels(M.index)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            axes[1].text(j, i, f"{M.values[i, j]:.3f}", ha="center", va="center", fontsize=7,
                         color="w" if M.values[i, j] > 2.4 else "k")
    axes[1].axhline(2.5, color="purple", ls="-.", lw=1)   # 8192 与 32768 之间 = 临界值所在
    axes[1].set_xlabel("预算 C"); axes[1].set_ylabel("L_ctx（C7 可行值）")
    axes[1].set_title("最优 Loss 热图（对数型，主口径；紫线=L_crit）")
    fig.colorbar(im, ax=axes[1], shrink=.8)
    fig.suptitle("图4  上下文长度敏感性")
    fig.tight_layout(); fig.savefig(f"{FIG}/F4_Lctx_sensitivity.png"); plt.close(fig)
    print("done")
