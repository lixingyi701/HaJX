# -*- coding: utf-8 -*-
"""
q2_03_economics.py — 四因素边际、领域替代/互补、单位算力下的参数与质量取舍
损失：L = E + A N^{-α} Q^{-κ_N} + B D^{-β} Q^{-κ_D}（配比乘子=1）
∂L/∂Q = -(κ_N T_N + κ_D T_D)/Q
领域关系来自问题一迁移矩阵，不读附件 A。
同样一份算力：MU = −∂L/∂x ÷ ∂C/∂x，C = (6+η L_ctx) N D + D[g(Q)-g(Q0)]_+
"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code_q1"))
from q1_00_common import ROOT as _ROOT, setup_cjk_matplotlib

ROOT = _ROOT if os.path.isdir(_ROOT) else os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = f"{ROOT}/outputs_q2"
IF1 = f"{ROOT}/outputs_q1/interface"
P = json.load(open(f"{OUT}/interface/P2_scaling_law.json", encoding="utf-8"))
assert P.get("q_shape") == "pow"
E, A, al, B, be = P["E"], P["A"], P["alpha"], P["B"], P["beta"]
kN, kD = P["kappa_N"], P["kappa_D"]
s0, s1 = P["s_p"]["s0"], P["s_p"]["s1_per_decade"]
Q0 = P["Q0_main"]
ETA, LCTX = 2e-4, 2048


def TN(N, Q):
    return A * np.power(N, -al) * np.power(Q, -kN)


def TD(D, Q):
    return B * np.power(D, -be) * np.power(Q, -kD)


def Ltil(N, D, Q):
    return E + TN(N, Q) + TD(D, Q)


def dL_dQ(N, D, Q):
    return -(kN * TN(N, Q) + kD * TD(D, Q)) / Q


def s_obs(N):
    return max(s0 + s1 * np.log10(N / 1e6), 0.0)


def N_star(C, Q, kappa=6.0):
    return ((al * A * Q**-kN) / (be * B * Q**-kD))**(1 / (al + be)) * (C / kappa)**(be / (al + be))


G = {
    "指数型": (lambda Q: 1e7 * np.exp(6.0 * Q), lambda Q: 6e7 * np.exp(6.0 * Q)),
    "幂函数型": (lambda Q: 5e9 * Q**4, lambda Q: 2e10 * Q**3),
    "对数型": (lambda Q: 2e9 * np.log1p(10 * Q), lambda Q: 2e10 / (1 + 10 * Q)),
}

if __name__ == "__main__":
    pd.set_option("display.width", 230)
    os.makedirs(f"{OUT}/tables", exist_ok=True)
    os.makedirs(f"{OUT}/figures", exist_ok=True)
    T = pd.read_csv(f"{IF1}/P1_transfer_matrix.csv", index_col=0)
    ps = pd.read_csv(f"{IF1}/P1_p_star.csv").set_index("domain")
    dom17, dom13 = T.index.tolist(), T.columns.tolist()
    Tagg = T.to_numpy() @ np.full(len(dom13), 1 / len(dom13))

    rows = []
    for Nv, Dv, Qv in [(1e9, 3e11, Q0), (1e9, 3e11, 0.9), (1e10, 1e12, Q0),
                       (1e10, 1e12, 0.9), (7e10, 2e12, Q0), (7e10, 2e12, 0.9)]:
        l = Ltil(Nv, Dv, Qv)
        dN = -al * TN(Nv, Qv) / Nv
        dD = -be * TD(Dv, Qv) / Dv
        dQ = dL_dQ(Nv, Dv, Qv)
        s = s_obs(Nv)
        dp = s * Tagg
        p_st = ps.loc[dom17, "p_star_eqweight"].to_numpy()
        i_best, i_worst = int(np.argmin(dp)), int(np.argmax(dp))
        rows.append(dict(
            N=Nv, D=Dv, Q=round(float(Qv), 3), L=l,
            MU_N=dN, MU_D=dD, MU_Q=dQ,
            eps_N=dN * Nv / l, eps_D=dD * Dv / l, eps_Q=dQ * Qv / l,
            s_obs=s, best_domain=dom17[i_best], dlnL_dp_best=float(dp[i_best]),
            worst_domain=dom17[i_worst], dlnL_dp_worst=float(dp[i_worst])))
    t31 = pd.DataFrame(rows)
    t31.to_csv(f"{OUT}/tables/T31_marginal_utility_elasticity.csv", index=False, encoding="utf-8-sig")
    print(t31.round(5).to_string(index=False))
    pd.DataFrame(dict(
        domain=dom17, dlnL_dp=Tagg,
        p_star=ps.loc[dom17, "p_star_eqweight"].to_numpy(),
        eps_p_at_1B=ps.loc[dom17, "p_star_eqweight"].to_numpy() * s_obs(1e9) * Tagg,
    )).sort_values("dlnL_dp").to_csv(
        f"{OUT}/tables/T31b_p_marginal_by_domain.csv", index=False, encoding="utf-8-sig")

    Tm = T.to_numpy()
    rows = []
    for j, v in enumerate(dom13):
        col = Tm[:, j]
        i_self = dom17.index(v) if v in dom17 else None
        helpers = [dom17[i] for i in np.argsort(col) if dom17[i] != v][:2]
        rows.append(dict(
            target=v,
            self_effect=float(col[i_self]) if i_self is not None else np.nan,
            best_cross_helper=helpers[0],
            best_cross_effect=float(col[dom17.index(helpers[0])]),
            second_helper=helpers[1]))
    pd.DataFrame(rows).to_csv(f"{OUT}/tables/T32a_self_vs_cross_transfer.csv", index=False, encoding="utf-8-sig")
    norm = np.linalg.norm(Tm, axis=1, keepdims=True)
    cos = (Tm / norm) @ (Tm / norm).T
    pairs = []
    for i in range(len(dom17)):
        for j in range(i + 1, len(dom17)):
            c = float(cos[i, j])
            pairs.append(dict(
                domain_i=dom17[i], domain_j=dom17[j], cos_sim=c,
                relation="功能替代" if c > 0.7 else ("覆盖互补" if c < -0.3 else "弱相关")))
    t32b = pd.DataFrame(pairs).sort_values("cos_sim", ascending=False)
    t32b.to_csv(f"{OUT}/tables/T32b_domain_pair_relation.csv", index=False, encoding="utf-8-sig")
    print(f"功能替代 {int((t32b.cos_sim>0.7).sum())} 对，覆盖互补 {int((t32b.cos_sim<-0.3).sum())} 对")

    kappa = 6 + ETA * LCTX
    rows = []
    for gname, (g, gd) in G.items():
        for C in [1e19, 1e22, 1e24]:
            for Qv in [0.5, float(Q0), 0.8, 0.9]:
                Nv = N_star(C, Qv, kappa)
                Dv = C / (kappa * Nv)
                dgap = max(g(Qv) - g(Q0), 0.0)
                MU_N = (al * TN(Nv, Qv) / Nv) / (kappa * Dv)
                MU_D = (be * TD(Dv, Qv) / Dv) / (kappa * Nv + dgap)
                MU_Q = (-dL_dQ(Nv, Dv, Qv)) / (Dv * gd(Qv))
                rows.append(dict(
                    g=gname, C=C, Q=round(Qv, 3), N_star=Nv, D_star=Dv,
                    MU_N_per_FLOP=MU_N, MU_D_per_FLOP=MU_D, MU_Q_per_FLOP=MU_Q,
                    R_QN=MU_Q / MU_N, R_QD=MU_Q / MU_D,
                    verdict="提质更划算" if MU_Q > MU_N else "扩参数更划算"))
    t33 = pd.DataFrame(rows)
    t33.to_csv(f"{OUT}/tables/T33_cost_adjusted_N_vs_Q.csv", index=False, encoding="utf-8-sig")
    print(t33[["g", "C", "Q", "R_QN", "verdict"]].round(3).to_string(index=False))

    Qgrid = np.linspace(0.05, 0.99, 95)
    rows = []
    for gname, (g, gd) in G.items():
        for C in np.logspace(19, 24, 6):
            def R(Qv):
                Nv = N_star(C, Qv, kappa)
                Dv = C / (kappa * Nv)
                return ((-dL_dQ(Nv, Dv, Qv)) / (Dv * gd(Qv))) / ((al * TN(Nv, Qv) / Nv) / (kappa * Dv))
            Rv = np.array([R(q) for q in Qgrid])
            fav = Qgrid[Rv > 1]
            rows.append(dict(
                g=gname, C=C,
                Q_favor_lo=float(fav.min()) if len(fav) else np.nan,
                Q_favor_hi=float(fav.max()) if len(fav) else np.nan,
                R_at_Q0=float(R(Q0))))
    t33b = pd.DataFrame(rows)
    t33b.to_csv(f"{OUT}/tables/T33b_quality_breakeven.csv", index=False, encoding="utf-8-sig")
    print(t33b.round(3).to_string(index=False))

    plt = setup_cjk_matplotlib()
    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK JP", "sans-serif"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    im = ax.imshow(cos, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(dom17))); ax.set_xticklabels(dom17, rotation=90, fontsize=7)
    ax.set_yticks(range(len(dom17))); ax.set_yticklabels(dom17, fontsize=7)
    ax.set_title("域对余弦：红=功能替代，蓝=覆盖互补")
    fig.colorbar(im, ax=ax, fraction=.046)
    ax = axes[1]
    ordr = np.argsort(Tagg)
    ax.barh(range(len(dom17)), Tagg[ordr] * s_obs(1e9),
            color=["#2b8cbe" if x < 0 else "#fd8d3c" for x in Tagg[ordr]])
    ax.set_yticks(range(len(dom17))); ax.set_yticklabels([dom17[i] for i in ordr], fontsize=7)
    ax.set_xlabel("∂lnL/∂p_i（N=1B）"); ax.set_title("增加该域对平均损失的边际")
    ax = axes[2]
    sub = t33[t33.Q.round(3) == round(Q0, 3)]
    for gname, c in zip(G, ["#a63603", "#2b8cbe", "#31a354"]):
        s = sub[sub.g == gname]
        ax.semilogx(s.C, s.R_QN, "o-", color=c, label=gname)
    ax.axhline(1, color="grey", ls="--")
    ax.set_xlabel("预算 C"); ax.set_ylabel("MU_Q / MU_N")
    ax.set_title(f"Q=Q0={Q0:.3f} 时，比值>1 表示提质更划算")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/F23_economics.png"); plt.close(fig)
    print("Q2 economics done.")
