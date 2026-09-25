# -*- coding: utf-8 -*-
"""
q2_02_theory.py — 质量耦合标度律的弹性、替代条件与算力最优配置（配比乘子=1，即 p=p_ref）
    L = E + k0(1−Q) + T_N + T_D,  T_N = A N^{-α} h(Q;κ_N),  T_D = B D^{-β} h(Q;κ_D)
    η_X(Q) = −∂ln h(Q;κ_X)/∂ln Q ≥ 0（pow：κ；exp：κQ；lin：κQ/(1+κ(1−Q))）
弹性：ε_N = −αT_N/L，ε_D = −βT_D/L，ε_Q = −(η_N T_N + η_D T_D + k0 Q)/L
质量 Q→Q' 与参数 N→N' 等效（固定 D）：
    t = h_N(Q')/h_N(Q) + (T_D/T_N)[h_D(Q')/h_D(Q) − 1] − k0(Q'−Q)/T_N,  N' = N t^{-1/α}，t≤0 时无有限解
算力最优（6ND=C）：N* = [αA h_N / (βB h_D)]^{1/(α+β)} (C/6)^{β/(α+β)}（地板与 N 无关，不改变 N*）
    记 R*(C,Q) = L*(C,Q) − E − k0(1−Q) ∝ C^{−a}，a = αβ/(α+β)
    Q→Q' 等价于算力乘 M = {[R*(C,Q') − k0(Q'−Q)] / R*(C,Q)}^{−1/a}；方括号 ≤0 时任何算力都追不上
渐近：N,D→∞ 时 L→E+k0(1−Q)。
"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code_q1"))
from q1_00_common import ROOT as _ROOT, setup_cjk_matplotlib

ROOT = _ROOT if os.path.isdir(_ROOT) else os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = f"{ROOT}/outputs_q2"
P = json.load(open(f"{OUT}/interface/P2_scaling_law.json", encoding="utf-8"))


class Law:
    def __init__(self, kappa_N, kappa_D, k0, shape, E=P["E"], A=P["A"], alpha=P["alpha"], B=P["B"], beta=P["beta"]):
        assert shape in ("pow", "exp", "lin")
        self.kN, self.kD, self.k0, self.shape = kappa_N, kappa_D, k0, shape
        self.E, self.A, self.al, self.B, self.be = E, A, alpha, B, beta
        self.a = alpha * beta / (alpha + beta)

    def h(self, Q, k):
        if self.shape == "pow":
            return Q**-k
        if self.shape == "exp":
            return np.exp(k * (1 - Q))
        return 1 + k * (1 - Q)

    def eta(self, Q, k):
        if self.shape == "pow":
            return k + 0 * Q
        if self.shape == "exp":
            return k * Q
        return k * Q / (1 + k * (1 - Q))

    def TN(self, N, Q): return self.A * N**-self.al * self.h(Q, self.kN)
    def TD(self, D, Q): return self.B * D**-self.be * self.h(Q, self.kD)
    def loss(self, N, D, Q): return self.E + self.k0 * (1 - Q) + self.TN(N, Q) + self.TD(D, Q)

    def dL_dQ(self, N, D, Q):
        return -(self.eta(Q, self.kN) * self.TN(N, Q) + self.eta(Q, self.kD) * self.TD(D, Q)) / Q - self.k0

    def elasticities(self, N, D, Q):
        l, tN, tD = self.loss(N, D, Q), self.TN(N, Q), self.TD(D, Q)
        eN, eD = -self.al * tN / l, -self.be * tD / l
        eQ = self.dL_dQ(N, D, Q) * Q / l
        return dict(eps_N=eN, eps_D=eD, eps_Q=eQ, eQ_over_eN=eQ / eN,
                    N_eff_over_N=self.h(Q, self.kN)**(-1 / self.al),
                    D_eff_over_D=self.h(Q, self.kD)**(-1 / self.be),
                    floor_share_of_dLdQ=self.k0 / -self.dL_dQ(N, D, Q), L=l)

    def N_equiv(self, N, D, Q, Q2):
        tN = self.TN(N, Q)
        t = (self.h(Q2, self.kN) / self.h(Q, self.kN)
             + self.TD(D, Q) / tN * (self.h(Q2, self.kD) / self.h(Q, self.kD) - 1)
             - self.k0 * (Q2 - Q) / tN)
        return np.inf if t <= 0 else N * t**(-1 / self.al)

    def N_star(self, C, Q, kappa=6.0):
        return ((self.al * self.A * self.h(Q, self.kN) / (self.be * self.B * self.h(Q, self.kD)))**(1 / (self.al + self.be))
                * (C / kappa)**(self.be / (self.al + self.be)))

    def L_star(self, C, Q):
        N = self.N_star(C, Q)
        return self.loss(N, C / (6 * N), Q)

    def R_star(self, C, Q):
        return self.L_star(C, Q) - self.E - self.k0 * (1 - Q)

    def compute_multiple(self, C, Q, Q2):
        br = (self.R_star(C, Q2) - self.k0 * (Q2 - Q)) / self.R_star(C, Q)
        return np.inf if br <= 0 else float(br**(-1 / self.a))

    def C_limit_for_equiv(self, Q, Q2):
        """质量 Q→Q' 能被算力替代的最大预算：R*(C,Q') = k0(Q'−Q) 处；k0=0 时为无穷。"""
        if self.k0 <= 0:
            return np.inf
        K2 = self.R_star(1.0, Q2)
        return float((K2 / (self.k0 * (Q2 - Q)))**(1 / self.a))


MAIN = Law(P["kappa_N"], P["kappa_D"], P["k0"], P["q_shape"])
_s = P["nofloor_sensitivity"]
ALT = Law(_s["kappa_N"], _s["kappa_D"], _s["k0"], _s["q_shape"])


def golden_min(f, lo, hi, iters=200):
    g = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    for _ in range(iters):
        c, d = b - g * (b - a), a + g * (b - a)
        if f(c) < f(d):
            b = d
        else:
            a = c
    return (a + b) / 2


if __name__ == "__main__":
    os.makedirs(f"{OUT}/tables", exist_ok=True)
    os.makedirs(f"{OUT}/figures", exist_ok=True)
    pd.set_option("display.width", 230)
    law = MAIN
    print(f"主式 {P['selected_model']}（{law.shape}）：κN={law.kN:.4f} κD={law.kD:.4f} k0={law.k0:.4f}；"
          f"敏感性 {_s['model']}：κN={ALT.kN:.4f} κD={ALT.kD:.4f} k0={ALT.k0:.4f}")

    D0, rows = 300e9, []
    for tag, lw in (("main", MAIN), ("nofloor", ALT)):
        for N0 in [1e8, 1e9, 1e10, 1e11]:
            for Q0 in [0.5, 0.7, 0.9]:
                Q2 = Q0 + 0.1
                ana = lw.N_equiv(N0, D0, Q0, Q2)
                target = lw.loss(N0, D0, Q2)
                lo, hi = np.log(N0) - 1, np.log(N0) + 40
                if lw.loss(np.exp(hi), D0, Q0) > target:
                    num = np.inf
                else:
                    for _ in range(120):
                        mid = 0.5 * (lo + hi)
                        lo, hi = (mid, hi) if lw.loss(np.exp(mid), D0, Q0) > target else (lo, mid)
                    num = np.exp(0.5 * (lo + hi))
                rows.append(dict(law=tag, N=N0, D=D0, Q=Q0, Q_new=Q2, N_equiv=ana, N_equiv_numeric=num,
                                 rel_err=abs(ana - num) / num if np.isfinite(num) and np.isfinite(ana) else np.nan,
                                 multiple=ana / N0))
    t2 = pd.DataFrame(rows)
    t2.to_csv(f"{OUT}/tables/T24_equiv_substitution.csv", index=False, encoding="utf-8-sig")
    print("\n质量 +0.1 等效的参数倍数（D=300B；inf=任何参数量都无法替代）")
    print(t2.pivot_table(index=["law", "N"], columns="Q", values="multiple").round(3).to_string())
    print(f"  解析与数值最大相对误差 {np.nanmax(t2.rel_err):.2e}；无限倍的一致性："
          f"{bool(((~np.isfinite(t2.N_equiv)) == (~np.isfinite(t2.N_equiv_numeric))).all())}")

    rows = []
    for C in [1e19, 1e20, 1e21, 1e22, 1e23, 1e24]:
        for Q0 in [0.5, 0.7, 0.9, 1.0]:
            ana = law.N_star(C, Q0)
            num = np.exp(golden_min(lambda z: law.loss(np.exp(z), C / (6 * np.exp(z)), Q0), np.log(1e6), np.log(1e14)))
            rows.append(dict(C=C, Q=Q0, N_star=ana, N_star_numeric=num, rel_err=abs(ana - num) / num,
                             D_over_N=C / (6 * ana**2), N_ratio_vs_Q1=ana / law.N_star(C, 1.0), L_star=law.L_star(C, Q0),
                             floor=law.k0 * (1 - Q0), floor_share_of_reducible=law.k0 * (1 - Q0) / (law.L_star(C, Q0) - law.E)))
    t4 = pd.DataFrame(rows)
    t4.to_csv(f"{OUT}/tables/T25_compute_optimal.csv", index=False, encoding="utf-8-sig")
    print("\n算力最优 N*"); print(t4.round(4).to_string(index=False))

    el = []
    for tag, lw in (("main", MAIN), ("nofloor", ALT)):
        for N0, D0_, Q0 in [(1e9, 3e11, 0.6), (1e9, 3e11, 0.9), (1e10, 1e12, 0.6),
                            (1e10, 1e12, 0.9), (7e10, 2e12, 0.6), (7e10, 2e12, 0.9)]:
            el.append(dict(law=tag, N=N0, D=D0_, Q=Q0, **{k: float(v) for k, v in lw.elasticities(N0, D0_, Q0).items()}))
    t1 = pd.DataFrame(el)
    t1.to_csv(f"{OUT}/tables/T26_elasticities.csv", index=False, encoding="utf-8-sig")
    print("\n弹性"); print(t1.round(4).to_string(index=False))

    pd.DataFrame([dict(Q=q, L_N_inf_D300B=law.E + law.k0 * (1 - q) + law.TD(300e9, q),
                       L_ND_inf=law.E + law.k0 * (1 - q),
                       N_eff_over_N=law.h(q, law.kN)**(-1 / law.al), D_eff_over_D=law.h(q, law.kD)**(-1 / law.be))
                  for q in [0.5, P["Q0_main"], 0.7, 0.9, 1.0]]).to_csv(
        f"{OUT}/tables/T27_quality_floor.csv", index=False, encoding="utf-8-sig")

    rows = []
    for tag, lw in (("main", MAIN), ("nofloor", ALT)):
        for C in [1e19, 1e20, 1e21, 1e22, 1e23, 1e24]:
            for Q0 in [0.5, 0.7, 0.9]:
                Q2 = Q0 + 0.1
                target = lw.L_star(C, Q2)
                if lw.L_star(C * 1e30, Q0) > target:
                    exact = np.inf
                else:
                    lo, hi = np.log(C), np.log(C) + np.log(1e30)
                    for _ in range(120):
                        mid = 0.5 * (lo + hi)
                        lo, hi = (mid, hi) if lw.L_star(np.exp(mid), Q0) > target else (lo, mid)
                    exact = np.exp(0.5 * (lo + hi)) / C
                closed = lw.compute_multiple(C, Q0, Q2)
                rows.append(dict(law=tag, C=C, Q=Q0, Q_new=Q2, C_multiple_exact=exact, C_multiple_closed=closed,
                                 rel_err=abs(exact - closed) / exact if np.isfinite(exact) else np.nan,
                                 C_limit_for_equiv=lw.C_limit_for_equiv(Q0, Q2)))
    t5 = pd.DataFrame(rows)
    t5.to_csv(f"{OUT}/tables/T29_quality_compute_exchange.csv", index=False, encoding="utf-8-sig")
    print("\n质量 +0.1 等价的算力倍数（沿最优前沿；inf=任何算力都追不上）")
    print(t5.pivot_table(index=["law", "C"], columns="Q", values="C_multiple_closed").round(3).to_string())
    print("  可替代的最大预算 C_limit：", t5[t5.law == "main"].drop_duplicates("Q")[["Q", "C_limit_for_equiv"]].to_dict("records"))
    print(f"  解析与数值最大相对误差 {np.nanmax(t5.rel_err):.2e}")

    plt = setup_cjk_matplotlib()
    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    Ng = np.logspace(8, 12, 200)
    for Q0, c in [(0.5, "#a63603"), (0.7, "#fd8d3c"), (0.9, "#2b8cbe")]:
        mv = np.array([law.N_equiv(x, 300e9, Q0, Q0 + 0.1) / x for x in Ng]); mv[~np.isfinite(mv)] = np.nan
        axes[0].loglog(Ng, mv, color=c, lw=2, label=f"Q={Q0}→{Q0+0.1:.1f}")
    axes[0].set_xlabel("N"); axes[0].set_ylabel("N'/N"); axes[0].set_title("质量 +0.1 的等效参数倍数（D=300B；断点后不可替代）")
    Cs = np.logspace(19, 25, 80)
    for Q0, c in [(0.5, "#a63603"), (0.7, "#fd8d3c"), (0.9, "#2b8cbe")]:
        mv = np.array([law.compute_multiple(x, Q0, Q0 + 0.1) for x in Cs]); mv[~np.isfinite(mv)] = np.nan
        axes[1].loglog(Cs, mv, color=c, lw=2, label=f"主式 Q={Q0}→{Q0+0.1:.1f}")
        axes[1].loglog(Cs, [ALT.compute_multiple(x, Q0, Q0 + 0.1) for x in Cs], color=c, lw=1, ls=":")
    axes[1].set_xlabel("C (FLOPs)"); axes[1].set_ylabel("等价算力倍数"); axes[1].set_title("质量 +0.1 等价的算力倍数（虚线：无地板）")
    for Q0, c in [(0.5, "#a63603"), (0.7, "#fd8d3c"), (0.9, "#74a9cf"), (1.0, "#2b8cbe")]:
        axes[2].semilogx(Cs, [law.L_star(x, Q0) for x in Cs], color=c, lw=2, label=f"Q={Q0}")
        axes[2].axhline(law.E + law.k0 * (1 - Q0), color=c, lw=.8, ls="--")
    axes[2].set_xlabel("C (FLOPs)"); axes[2].set_ylabel("L*"); axes[2].set_title("算力—质量前沿（虚线：渐近地板）")
    for ax in axes:
        ax.legend(fontsize=7); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/F21_theory.png"); plt.close(fig)

    with open(f"{OUT}/interface/P2_theory.json", "w", encoding="utf-8") as f:
        json.dump(dict(
            model=P["form"], q_shape=law.shape, selected_model=P["selected_model"],
            E=law.E, A=law.A, alpha=law.al, B=law.B, beta=law.be, kappa_N=law.kN, kappa_D=law.kD, k0=law.k0,
            eta="η_X(Q)=−∂ln h/∂ln Q；pow: κ，exp: κQ，lin: κQ/(1+κ(1−Q))",
            dL_dQ="−(η_N T_N + η_D T_D)/Q − k0",
            eps_Q="Q·∂L/∂Q / L",
            N_equiv="N'=N t^{-1/α}，t=h_N(Q')/h_N(Q)+(T_D/T_N)[h_D(Q')/h_D(Q)−1]−k0(Q'−Q)/T_N；t≤0 不可替代",
            N_star="[αA h_N(Q)/(βB h_D(Q))]^{1/(α+β)}(C/6)^{β/(α+β)}（与 k0 无关）",
            compute_multiple="{[R*(C,Q')−k0(Q'−Q)]/R*(C,Q)}^{−(α+β)/(αβ)}，R*=L*−E−k0(1−Q)；方括号≤0 不可替代",
            compute_multiple_examples={f"C={C:.0e},{q}->{q+0.1:.1f}": law.compute_multiple(C, q, q + 0.1)
                                       for C in (1e19, 1e22, 1e24) for q in (0.5, 0.7, 0.9)},
            C_limit_for_equiv={f"{q}->{q+0.1:.1f}": law.C_limit_for_equiv(q, q + 0.1) for q in (0.5, 0.6, 0.7, 0.8, 0.9)},
            limit="N,D→∞ 时 L→E+k0(1−Q)",
            p_channel=P["p_channel"], Q0_main=P["Q0_main"],
            note="闭式在 p=p_ref（配比乘子=1）时成立"), f, ensure_ascii=False, indent=2)
    print("Q2 theory done.")
