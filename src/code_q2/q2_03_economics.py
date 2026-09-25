# -*- coding: utf-8 -*-
"""
q2_03_economics.py — 问题二·D：四因素边际效用与弹性、领域替代/互补、单位算力成本下的 N–Q 取舍
====================================================================================
输入：P2_scaling_law.json（q2_01）、问题一接口 P1_transfer_matrix / P1_p_star / P1_f_p_coefficients
     不读附件 A 原始文件。
内容：
  E1 四因素边际效用与弹性（N、D、Q 解析；p 通过问题一迁移矩阵 T 与 s(N)）
        ∂lnL/∂p_i = s(N)·Σ_v w_v T_iv（T 在均匀配比处评估，问题一口径），ε_{p_i} = p_i·∂lnL/∂p_i
  E2 领域替代/互补（题面：讨论不同领域数据之间的替代或互补关系）
        以 T 的行向量（域 i 对 13 个验证域的效应）定义：
        - 自域效应 T_vv 与最强跨域帮助 argmin_{i≠v} T_iv（迁移替代）
        - 域对功能相似度 cos(T_i·, T_j·)：>0.7 记为"功能替代"（二者作用方向一致，可互换）；
          <0 记为"覆盖互补"（各自压低不同验证域）。线性 clr 模型无交互项，超可加意义的互补不可识别，如实注明。
  E3 单位算力成本下的取舍（题面：同样多花一份算力，用在参数上还是质量上更划算）
        成本 C = 6ND + D[g(Q)−g(Q0)]_+ + ηNDL_ctx（题面三项；本文取 L_ctx=2048 的默认档，η=2e-4）
        每 FLOP 的边际损失下降：MU_N = −∂L/∂N ÷ ∂C/∂N，MU_Q = −∂L/∂Q ÷ ∂C/∂Q，MU_D 同理
        比值 R_QN = MU_Q / MU_N > 1 ⇔ 同样一份算力用于提质更划算；对三类成本函数 g 与三档预算给出
所有结论在 p=p*（m=1）处计算；p≠p* 的修正见 q2_02。
"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code_q1"))
from q1_00_common import ROOT as _ROOT, setup_cjk_matplotlib

ROOT = _ROOT if os.path.isdir(_ROOT) else os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = f"{ROOT}/outputs_q2"; IF1 = f"{ROOT}/outputs_q1/interface"
P = json.load(open(f"{OUT}/interface/P2_scaling_law.json", encoding="utf-8"))
E, A, al, B, be = P["E"], P["A"], P["alpha"], P["B"], P["beta"]
kN, kD, k0 = P["kappa_N"], P["kappa_D"], P["k0"]
s0, s1 = P["s_p"]["s0"], P["s_p"]["s1_per_decade"]
Q0 = P["Q0_main"]
ETA, LCTX = 2e-4, 2048

hN = lambda Q: 1 + kN*(1-Q); hD = lambda Q: 1 + kD*(1-Q)
def Ltil(N, D, Q): return E + A*N**-al*hN(Q) + B*D**-be*hD(Q) + k0*(1-Q)
def Phi(N, D): return kN*A*N**-al + kD*B*D**-be + k0
def sN(N): return max(s0 + s1*np.log10(N/1e6), 0.0)
def N_star(C, Q, kappa=6.0):
    return (al*A*hN(Q)/(be*B*hD(Q)))**(1/(al+be)) * (C/kappa)**(be/(al+be))

G = {"指数型": (lambda Q: 1e7*np.exp(6.0*Q), lambda Q: 6e7*np.exp(6.0*Q)),
     "幂函数型": (lambda Q: 5e9*Q**4, lambda Q: 2e10*Q**3),
     "对数型": (lambda Q: 2e9*np.log1p(10*Q), lambda Q: 2e10/(1+10*Q))}

if __name__ == "__main__":
    pd.set_option("display.width", 230)
    T = pd.read_csv(f"{IF1}/P1_transfer_matrix.csv", index_col=0)          # 17×13, ∂lnL_v/∂p_i @ uniform
    ps = pd.read_csv(f"{IF1}/P1_p_star.csv").set_index("domain")
    dom17, dom13 = T.index.tolist(), T.columns.tolist()
    w13 = np.full(len(dom13), 1/len(dom13))
    Tagg = T.to_numpy() @ w13                                                # ∂lnL̄/∂p_i（等权目标）

    # ---------------- E1 四因素边际效用与弹性 ----------------
    rows = []
    for (Nv, Dv, Qv) in [(1e9, 3e11, Q0), (1e9, 3e11, 0.9), (1e10, 1e12, Q0), (1e10, 1e12, 0.9),
                         (7e10, 2e12, Q0), (7e10, 2e12, 0.9)]:
        l = Ltil(Nv, Dv, Qv)
        dN, dD, dQ = -al*A*Nv**(-al-1)*hN(Qv), -be*B*Dv**(-be-1)*hD(Qv), -Phi(Nv, Dv)
        s = sN(Nv)
        dp = s*Tagg                                                          # ∂lnL/∂p_i
        p_st = ps.loc[dom17, "p_star_eqweight"].to_numpy()
        eps_p = p_st*dp                                                      # p_i·∂lnL/∂p_i
        i_best, i_worst = int(np.argmin(dp)), int(np.argmax(dp))
        rows.append(dict(N=Nv, D=Dv, Q=round(Qv, 3), L=l,
                         MU_N=dN, MU_D=dD, MU_Q=dQ, eps_N=dN*Nv/l, eps_D=dD*Dv/l, eps_Q=dQ*Qv/l,
                         s_N=s, best_domain=dom17[i_best], dlnL_dp_best=dp[i_best],
                         worst_domain=dom17[i_worst], dlnL_dp_worst=dp[i_worst],
                         eps_p_sum_abs=float(np.abs(eps_p).sum())))
    t31 = pd.DataFrame(rows)
    t31.to_csv(f"{OUT}/tables/T31_marginal_utility_elasticity.csv", index=False, encoding="utf-8-sig")
    print("[T31 四因素边际效用与弹性]（p=p*；p 项经 s(N) 与迁移矩阵 T）")
    print(t31.round(5).to_string(index=False))
    pd.DataFrame(dict(domain=dom17, dlnL_dp_uniform=Tagg, p_star=ps.loc[dom17, "p_star_eqweight"].to_numpy(),
                      eps_p_at_1B=ps.loc[dom17, "p_star_eqweight"].to_numpy()*sN(1e9)*Tagg)
                 ).sort_values("dlnL_dp_uniform").to_csv(f"{OUT}/tables/T31b_p_marginal_by_domain.csv", index=False, encoding="utf-8-sig")

    # ---------------- E2 领域替代 / 互补 ----------------
    Tm = T.to_numpy()
    # 自域效应 vs 最强跨域帮助
    rows = []
    for j, v in enumerate(dom13):
        col = Tm[:, j]
        i_self = dom17.index(v) if v in dom17 else None
        order = np.argsort(col)
        helpers = [dom17[i] for i in order if dom17[i] != v][:2]
        rows.append(dict(target=v, self_effect=col[i_self] if i_self is not None else np.nan,
                         best_cross_helper=helpers[0], best_cross_effect=col[dom17.index(helpers[0])],
                         second_helper=helpers[1],
                         self_share=col[i_self]/col[col < 0].sum() if i_self is not None and (col < 0).any() else np.nan))
    t32a = pd.DataFrame(rows)
    t32a.to_csv(f"{OUT}/tables/T32a_self_vs_cross_transfer.csv", index=False, encoding="utf-8-sig")
    # 域对功能相似度
    norm = np.linalg.norm(Tm, axis=1, keepdims=True)
    cos = (Tm/norm) @ (Tm/norm).T
    pairs = []
    for i in range(len(dom17)):
        for j in range(i+1, len(dom17)):
            pairs.append(dict(domain_i=dom17[i], domain_j=dom17[j], cos_sim=cos[i, j],
                              relation="功能替代(可互换)" if cos[i, j] > 0.7 else ("覆盖互补" if cos[i, j] < -0.3 else "弱相关/近似正交")))
    t32b = pd.DataFrame(pairs).sort_values("cos_sim", ascending=False)
    t32b.to_csv(f"{OUT}/tables/T32b_domain_pair_relation.csv", index=False, encoding="utf-8-sig")
    print("\n[T32a 自域效应 vs 跨域迁移]"); print(t32a.round(4).to_string(index=False))
    print(f"\n[T32b 域对关系] 功能替代对(cos>0.7) {int((t32b.cos_sim>0.7).sum())} 个，覆盖互补对(cos<-0.3) {int((t32b.cos_sim<-0.3).sum())} 个，"
          f"其余 {int(((t32b.cos_sim<=0.7)&(t32b.cos_sim>=-0.3)).sum())} 对近似正交；"
          f"最强替代：{t32b.iloc[0].domain_i}–{t32b.iloc[0].domain_j} (cos={t32b.iloc[0].cos_sim:.2f})；"
          f"最强互补：{t32b.iloc[-1].domain_i}–{t32b.iloc[-1].domain_j} (cos={t32b.iloc[-1].cos_sim:.2f})")

    # ---------------- E3 单位算力成本下的 N–Q 取舍 ----------------
    rows = []
    kappa = 6 + ETA*LCTX
    for gname, (g, gd) in G.items():
        for C in [1e19, 1e22, 1e24]:
            for Qv in [0.5, Q0, 0.8, 0.9]:
                Nv = N_star(C, Qv, kappa); Dv = C/(kappa*Nv)                 # 纯训练成本下的算力最优点
                dgap = max(g(Qv) - g(Q0), 0.0)
                MU_N = al*A*Nv**(-al-1)*hN(Qv) / (kappa*Dv)                  # 每 FLOP 的损失下降（增 N）
                MU_D = be*B*Dv**(-be-1)*hD(Qv) / (kappa*Nv + dgap)           # 增 D（同时增加质量处理与注意力开销）
                MU_Q = Phi(Nv, Dv) / (Dv*gd(Qv))                             # 提质
                MU_Q_nofloor = (Phi(Nv, Dv) - k0) / (Dv*gd(Qv))              # 去掉半合成地板 k0 的敏感性
                rows.append(dict(g=gname, C=C, Q=round(Qv, 3), N_star=Nv, D_star=Dv,
                                 MU_N_per_FLOP=MU_N, MU_D_per_FLOP=MU_D, MU_Q_per_FLOP=MU_Q,
                                 R_QN=MU_Q/MU_N, R_QD=MU_Q/MU_D, R_QN_nofloor=MU_Q_nofloor/MU_N,
                                 floor_share_of_Phi=k0/Phi(Nv, Dv),
                                 g_prime_per_token=gd(Qv), six_N_per_token=kappa*Nv,
                                 verdict="提质更划算" if MU_Q > MU_N else "扩参数更划算"))
    t33 = pd.DataFrame(rows)
    t33.to_csv(f"{OUT}/tables/T33_cost_adjusted_N_vs_Q.csv", index=False, encoding="utf-8-sig")
    print("\n[T33 单位算力：提质 vs 扩参数]（R_QN>1 ⇒ 同样一份算力用于提质更划算；L_ctx=2048）")
    print(t33[["g", "C", "Q", "N_star", "R_QN", "R_QN_nofloor", "floor_share_of_Phi", "verdict"]].round(3).to_string(index=False))
    # 提质划算区间 {Q : R_QN(Q)>1}（不同 g 的 R 随 Q 的单调方向不同，直接在网格上求区间）
    Qgrid = np.linspace(0.05, 1.0, 191)
    rows = []
    for gname, (g, gd) in G.items():
        for C in np.logspace(19, 25, 13):
            def R(Qv, floor=True):
                Nv = N_star(C, Qv, kappa); Dv = C/(kappa*Nv)
                ph = Phi(Nv, Dv) - (0.0 if floor else k0)
                return (ph/(Dv*gd(Qv))) / (al*A*Nv**(-al-1)*hN(Qv)/(kappa*Dv))
            Rv = np.array([R(q) for q in Qgrid]); fav = Qgrid[Rv > 1]
            Rn = np.array([R(q, False) for q in Qgrid]); favn = Qgrid[Rn > 1]
            rows.append(dict(g=gname, C=C,
                             Q_favor_lo=fav.min() if len(fav) else np.nan, Q_favor_hi=fav.max() if len(fav) else np.nan,
                             favor_frac=len(fav)/len(Qgrid), R_at_Q0=R(Q0),
                             Q_favor_lo_nofloor=favn.min() if len(favn) else np.nan,
                             Q_favor_hi_nofloor=favn.max() if len(favn) else np.nan, R_at_Q0_nofloor=R(Q0, False)))
    t33b = pd.DataFrame(rows)
    t33b.to_csv(f"{OUT}/tables/T33b_quality_breakeven.csv", index=False, encoding="utf-8-sig")
    print("\n[T33b 提质更划算的质量区间 [Q_lo, Q_hi]]（含地板 / 去地板；R_at_Q0 为 Q0 处 MU_Q/MU_N）")
    print(t33b.round(3).to_string(index=False))

    # ---------------- 图 F23 ----------------
    plt = setup_cjk_matplotlib()
    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK JP", "sans-serif"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    im = ax.imshow(cos, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(dom17))); ax.set_xticklabels(dom17, rotation=90, fontsize=7)
    ax.set_yticks(range(len(dom17))); ax.set_yticklabels(dom17, fontsize=7)
    ax.set_title("域对功能相似度 cos(T_i·,T_j·)：红=替代，蓝=互补"); fig.colorbar(im, ax=ax, fraction=.046)
    ax = axes[1]
    ordr = np.argsort(Tagg)
    ax.barh(range(len(dom17)), Tagg[ordr]*sN(1e9), color=["#2b8cbe" if x < 0 else "#fd8d3c" for x in Tagg[ordr]])
    ax.set_yticks(range(len(dom17))); ax.set_yticklabels([dom17[i] for i in ordr], fontsize=7)
    ax.set_xlabel("∂lnL̄/∂p_i（N=1B, 均匀配比处）"); ax.set_title("配比的边际效用：负值 = 增加该域降低平均损失"); ax.grid(alpha=.3, axis="x")
    ax = axes[2]
    for gname, c in zip(G, ["#a63603", "#2b8cbe", "#31a354"]):
        s = t33b[t33b.g == gname]
        ax.fill_between(s.C, s.Q_favor_lo.fillna(Q0), s.Q_favor_hi.fillna(Q0), color=c, alpha=.25, label=f"{gname}：提质更划算的 Q 区间")
        ax.semilogx(s.C, s.Q_favor_lo_nofloor, ":", color=c, lw=1.2)
    ax.axhline(Q0, color="grey", ls="--", lw=1, label=f"Q0={Q0:.3f}（问题一 p*）")
    ax.set_xscale("log"); ax.set_xlabel("算力预算 C (FLOPs)"); ax.set_ylabel("质量 Q"); ax.set_ylim(0, 1.05)
    ax.set_title("提质更划算的 Q 区间（虚线：去地板 k0 后的下界）", fontsize=10); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/F23_economics.png"); plt.close(fig)
    print("\nQ2-D done.")
