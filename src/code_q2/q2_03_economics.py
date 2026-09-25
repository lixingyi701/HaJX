# -*- coding: utf-8 -*-
"""
q2_03_economics.py — 四因素边际、领域替代/互补、单位算力下的参数与质量取舍
损失与导数取自 q2_02 的 Law（配比乘子=1）。配比乘子只作用于随规模变化的部分 R=T_N+T_D：
    ∂L/∂p_i = s_red·R·∂φ/∂p_i（H_red，主）；H_tot 上界为 s_train·L·∂φ/∂p_i
    ∂φ/∂p_i 取问题一迁移矩阵 T（均匀配比处）13 域等权平均
同样一份算力：MU_x = −∂L/∂x ÷ ∂C/∂x，C = (6+ηL_ctx)ND + D[g(Q)−g(Q0)]_+；主式与无地板敏感性都报告。
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code_q1"))
from q1_00_common import setup_cjk_matplotlib
import q2_02_theory as th

OUT, ROOT, P = th.OUT, th.ROOT, th.P
IF1 = f"{ROOT}/outputs_q1/interface"
MAIN, ALT = th.MAIN, th.ALT
pc = P["p_channel"]
S_RED, S_TRAIN = pc["s_red_main"], pc["s_train"]
Q0 = P["Q0_main"]
ETA, LCTX = 2e-4, 2048
KAPPA = 6 + ETA * LCTX

G = {"指数型": (lambda Q: 1e7 * np.exp(6.0 * Q), lambda Q: 6e7 * np.exp(6.0 * Q)),
     "幂函数型": (lambda Q: 5e9 * Q**4, lambda Q: 2e10 * Q**3),
     "对数型": (lambda Q: 2e9 * np.log1p(10 * Q), lambda Q: 2e10 / (1 + 10 * Q))}


def R_QN(law, C, Qv, gd):
    """纯训练成本的算力最优点上，提质与扩参数的每 FLOP 边际损失下降之比。"""
    Nv = law.N_star(C, Qv, KAPPA)
    Dv = C / (KAPPA * Nv)
    mu_q = -law.dL_dQ(Nv, Dv, Qv) / (Dv * gd(Qv))
    mu_n = (law.al * law.TN(Nv, Qv) / Nv) / (KAPPA * Dv)
    return mu_q / mu_n, Nv, Dv, mu_q, mu_n


if __name__ == "__main__":
    pd.set_option("display.width", 230)
    T = pd.read_csv(f"{IF1}/P1_transfer_matrix.csv", index_col=0)
    ps = pd.read_csv(f"{IF1}/P1_p_star.csv").set_index("domain")
    dom17, dom13 = T.index.tolist(), T.columns.tolist()
    dphi = T.to_numpy() @ np.full(len(dom13), 1 / len(dom13))
    p_st = ps.loc[dom17, "p_star_eqweight"].to_numpy()
    law = MAIN

    rows = []
    for Nv, Dv, Qv in [(1e9, 3e11, Q0), (1e9, 3e11, 0.9), (1e10, 1e12, Q0),
                       (1e10, 1e12, 0.9), (7e10, 2e12, Q0), (7e10, 2e12, 0.9)]:
        l = law.loss(Nv, Dv, Qv)
        R = law.TN(Nv, Qv) + law.TD(Dv, Qv)
        dN = -law.al * law.TN(Nv, Qv) / Nv
        dD = -law.be * law.TD(Dv, Qv) / Dv
        dQ = law.dL_dQ(Nv, Dv, Qv)
        dp_main, dp_up = S_RED * R * dphi, S_TRAIN * l * dphi
        ib, iw = int(np.argmin(dphi)), int(np.argmax(dphi))
        rows.append(dict(N=Nv, D=Dv, Q=round(float(Qv), 3), L=l, MU_N=dN, MU_D=dD, MU_Q=dQ,
                         eps_N=dN * Nv / l, eps_D=dD * Dv / l, eps_Q=dQ * Qv / l,
                         floor_share_of_MU_Q=law.k0 / -dQ,
                         eps_p_sum_abs_main=float(np.abs(p_st * dp_main).sum() / l),
                         eps_p_sum_abs_upper=float(np.abs(p_st * dp_up).sum() / l),
                         best_domain=dom17[ib], dL_dp_best_main=float(dp_main[ib]),
                         worst_domain=dom17[iw], dL_dp_worst_main=float(dp_main[iw])))
    t31 = pd.DataFrame(rows)
    t31.to_csv(f"{OUT}/tables/T31_marginal_utility_elasticity.csv", index=False, encoding="utf-8-sig")
    print("[T31 四因素边际与弹性]"); print(t31.round(5).to_string(index=False))
    l1b = law.loss(1e9, 3e11, Q0)
    R1b = law.TN(1e9, Q0) + law.TD(3e11, Q0)
    pd.DataFrame(dict(domain=dom17, dphi_dp=dphi, p_star=p_st,
                      dL_dp_1B_main=S_RED * R1b * dphi, dL_dp_1B_upper=S_TRAIN * l1b * dphi)
                 ).sort_values("dphi_dp").to_csv(f"{OUT}/tables/T31b_p_marginal_by_domain.csv", index=False, encoding="utf-8-sig")

    Tm = T.to_numpy()
    rows = []
    for j, v in enumerate(dom13):
        col = Tm[:, j]
        i_self = dom17.index(v) if v in dom17 else None
        helpers = [dom17[i] for i in np.argsort(col) if dom17[i] != v][:2]
        rows.append(dict(target=v, self_effect=float(col[i_self]) if i_self is not None else np.nan,
                         best_cross_helper=helpers[0], best_cross_effect=float(col[dom17.index(helpers[0])]),
                         second_helper=helpers[1]))
    pd.DataFrame(rows).to_csv(f"{OUT}/tables/T32a_self_vs_cross_transfer.csv", index=False, encoding="utf-8-sig")
    nrm = np.linalg.norm(Tm, axis=1, keepdims=True)
    cos = (Tm / nrm) @ (Tm / nrm).T
    pairs = [dict(domain_i=dom17[i], domain_j=dom17[j], cos_sim=float(cos[i, j]),
                  relation="功能替代" if cos[i, j] > 0.7 else ("覆盖互补" if cos[i, j] < -0.3 else "弱相关"))
             for i in range(len(dom17)) for j in range(i + 1, len(dom17))]
    t32b = pd.DataFrame(pairs).sort_values("cos_sim", ascending=False)
    t32b.to_csv(f"{OUT}/tables/T32b_domain_pair_relation.csv", index=False, encoding="utf-8-sig")
    print(f"\n功能替代 {int((t32b.cos_sim > 0.7).sum())} 对，覆盖互补 {int((t32b.cos_sim < -0.3).sum())} 对，"
          f"最强替代 {t32b.iloc[0].domain_i}–{t32b.iloc[0].domain_j} ({t32b.iloc[0].cos_sim:.2f})，"
          f"最强互补 {t32b.iloc[-1].domain_i}–{t32b.iloc[-1].domain_j} ({t32b.iloc[-1].cos_sim:.2f})")

    rows = []
    for gname, (g, gd) in G.items():
        for C in [1e19, 1e22, 1e24]:
            for Qv in [0.5, float(Q0), 0.8, 0.9]:
                r, Nv, Dv, mu_q, mu_n = R_QN(law, C, Qv, gd)
                dgap = max(g(Qv) - g(Q0), 0.0)
                mu_d = (law.be * law.TD(Dv, Qv) / Dv) / (KAPPA * Nv + dgap)
                rows.append(dict(g=gname, C=C, Q=round(Qv, 3), N_star=Nv, D_star=Dv,
                                 MU_N_per_FLOP=mu_n, MU_D_per_FLOP=mu_d, MU_Q_per_FLOP=mu_q,
                                 R_QN=r, R_QD=mu_q / mu_d, R_QN_nofloor_sensitivity=R_QN(ALT, C, Qv, gd)[0],
                                 verdict="提质更划算" if r > 1 else "扩参数更划算"))
    t33 = pd.DataFrame(rows)
    t33.to_csv(f"{OUT}/tables/T33_cost_adjusted_N_vs_Q.csv", index=False, encoding="utf-8-sig")
    print("\n[T33]"); print(t33[["g", "C", "Q", "R_QN", "R_QN_nofloor_sensitivity", "verdict"]].round(3).to_string(index=False))

    Qg = np.linspace(0.05, 0.99, 95)
    rows = []
    for gname, (g, gd) in G.items():
        for C in np.logspace(19, 24, 6):
            Rv = np.array([R_QN(law, C, q, gd)[0] for q in Qg])
            fav = Qg[Rv > 1]
            rows.append(dict(g=gname, C=C, Q_favor_lo=float(fav.min()) if len(fav) else np.nan,
                             Q_favor_hi=float(fav.max()) if len(fav) else np.nan,
                             R_at_Q0=float(R_QN(law, C, Q0, gd)[0]), R_at_Q0_nofloor=float(R_QN(ALT, C, Q0, gd)[0])))
    t33b = pd.DataFrame(rows)
    t33b.to_csv(f"{OUT}/tables/T33b_quality_breakeven.csv", index=False, encoding="utf-8-sig")
    print("\n[T33b]"); print(t33b.round(3).to_string(index=False))

    plt = setup_cjk_matplotlib()
    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    im = axes[0].imshow(cos, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_xticks(range(17)); axes[0].set_xticklabels(dom17, rotation=90, fontsize=7)
    axes[0].set_yticks(range(17)); axes[0].set_yticklabels(dom17, fontsize=7)
    axes[0].set_title("域对余弦：红=功能替代，蓝=覆盖互补"); fig.colorbar(im, ax=axes[0], fraction=.046)
    o = np.argsort(dphi)
    axes[1].barh(range(17), (S_RED * R1b * dphi)[o], color=["#2b8cbe" if x < 0 else "#fd8d3c" for x in dphi[o]])
    axes[1].set_yticks(range(17)); axes[1].set_yticklabels([dom17[i] for i in o], fontsize=7)
    axes[1].set_xlabel("∂L/∂p_i（N=1B，D=300B，Q=Q0）"); axes[1].set_title("增加该域对损失的边际")
    for gname, c in zip(G, ["#a63603", "#2b8cbe", "#31a354"]):
        s = t33b[t33b.g == gname]
        axes[2].semilogx(s.C, s.R_at_Q0, "o-", color=c, label=f"{gname}（主式）")
        axes[2].semilogx(s.C, s.R_at_Q0_nofloor, ":", color=c, lw=1)
    axes[2].axhline(1, color="grey", ls="--"); axes[2].set_yscale("log")
    axes[2].set_xlabel("预算 C"); axes[2].set_ylabel("MU_Q / MU_N")
    axes[2].set_title(f"Q=Q0={Q0:.3f}：>1 表示提质更划算（虚线：无地板）"); axes[2].legend(fontsize=8); axes[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/F23_economics.png"); plt.close(fig)
    print("Q2 economics done.")
