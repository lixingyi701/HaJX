# -*- coding: utf-8 -*-
"""
q2_02_theory.py — 问题二·B/C：解析理论 + 数值互验（主模型 = 嵌套族中的 M4+floor，见 q2_01_fit.py）
主模型（Q=1、p=p* 时退化为经典 Chinchilla 式）：
    L(N,D,Q,p) = L̃(N,D,Q) · m(p),   m(p) = exp(s(N)[f̄(p) − f̄(p*)])
    L̃(N,D,Q)   = E + A·N^-α·h_N(Q) + B·D^-β·h_D(Q) + k0·(1−Q),   h_X(Q) = 1 + κ_X(1−Q)
记 T_N = A N^-α h_N(Q)，T_D = B D^-β h_D(Q)，R = T_N + T_D（可约损失）。
解析结果（T1–T5 均在 p=p* 处成立，此时 m(p)=1；p≠p* 时 s(N) 随 N 变化，
  ∂lnL/∂lnN 多出一项 s1·Δf̄/ln10，Δf̄=f̄(p)−f̄(p*)，1B 检验配比 P90 处约 −0.007，须数值求解）：
  T1 弹性:
        ε_N = −α T_N / L̃,   ε_D = −β T_D / L̃,
        ε_Q = −Q·(κ_N A N^-α + κ_D B D^-β + k0) / L̃
      有效规模解释：h_N(Q) 倍的 N 项 ⇔ N_eff = N·h_N(Q)^{-1/α}；D_eff = D·h_D(Q)^{-1/β}
  T2 等效替代（质量 Q→Q+δ 等效于参数 N→N'）：
        t = h_N(Q+δ)/h_N(Q) − δ(κ_D B D^-β + k0) / (A N^-α h_N(Q)),   N' = N·t^{-1/α}
      有限解当且仅当 t>0  ⇔  N < N_max(D,Q,δ) = [A h_N(Q+δ) / (δ(κ_D B D^-β + k0))]^{1/α}
      —— 参数量超过 N_max 后，δ 的质量提升无法用任何参数量替代（地板 + D 项渠道）
  T3 渐近：N→∞ 时 L̃ → E + B D^-β h_D(Q) + k0(1−Q)；N,D→∞ 时 L̃ → E + k0(1−Q)
      k0(1−Q) 是 B6/B7 半合成数据内建的"质量地板"（F=109 显著），须标注其来源
  T4 算力最优配置（6ND=C，p=p*；p≠p* 时 s(N) 使 p 与 N 的决策耦合，不再可分离）：
        N*(C,Q) = [α A h_N(Q) / (β B h_D(Q))]^{1/(α+β)} · (C/6)^{β/(α+β)}
      κ_N>κ_D ⇒ h_N/h_D 随 Q 下降而上升 ⇒ 质量越低最优 N 越大（低质数据时代应养大模型）
  T5 质量–算力汇率（沿算力最优前沿）：
        ε_C = −[αβ/(α+β)]·R/L̃，  ρ = ε_Q/ε_C = 提升 1% 质量等效的算力百分比
        ⇒ Q→Q' 等效算力倍数 ≈ exp(ρ·ln(Q'/Q))（小步近似），大步用数值解 L̃*(C',Q)=L̃*(C,Q')
本脚本：数值优化验证 T2/T4/T5 的解析解，并出全部图表。
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
E, A, al = P["E"], P["A"], P["alpha"]
B, be = P["B"], P["beta"]
kN, kD, k0 = P["kappa_N"], P["kappa_D"], P["k0"]
assert P.get("q_shape", "lin") == "lin", "本理论文件按线性乘子 h=1+κ(1-Q) 推导"


def hN(Q): return 1 + kN*(1-Q)
def hD(Q): return 1 + kD*(1-Q)
def TN(N, Q): return A*N**-al*hN(Q)
def TD(D, Q): return B*D**-be*hD(Q)
def L(N, D, Q): return E + TN(N, Q) + TD(D, Q) + k0*(1-Q)


# ---------------- T1 弹性 ----------------
def elasticities(N, D, Q):
    l = L(N, D, Q)
    eN, eD = -al*TN(N, Q)/l, -be*TD(D, Q)/l
    eQ = -Q*(kN*A*N**-al + kD*B*D**-be + k0)/l
    return dict(eps_N=eN, eps_D=eD, eps_Q=eQ, eQ_over_eN=eQ/eN,
                N_eff_ratio=hN(Q)**(-1/al), D_eff_ratio=hD(Q)**(-1/be), L=l)


# ---------------- T2 等效替代（解析） ----------------
def dN_equiv(N, Q, D, delta=0.1):
    """L(N',D,Q)=L(N,D,Q+delta) 的 N'；无解返回 inf"""
    t = hN(Q+delta)/hN(Q) - delta*(kD*B*D**-be + k0)/(A*N**-al*hN(Q))
    return np.inf if t <= 0 else N*t**(-1/al)
def N_max(D, Q, delta=0.1):
    return (A*hN(Q+delta)/(delta*(kD*B*D**-be + k0)))**(1/al)


# ---------------- T4 算力最优（解析） ----------------
def N_star(C, Q):
    return (al*A*hN(Q)/(be*B*hD(Q)))**(1/(al+be)) * (C/6)**(be/(al+be))
def L_star(C, Q):
    N = N_star(C, Q); return L(N, C/(6*N), Q)


# ---------------- T5 质量–算力汇率（解析） ----------------
def exchange_rate(C, Q):
    N = N_star(C, Q); D = C/(6*N); l = L(N, D, Q)
    R = TN(N, Q) + TD(D, Q)
    eC = -(al*be/(al+be))*R/l
    eQ = elasticities(N, D, Q)["eps_Q"]
    return dict(eps_C=eC, eps_Q=eQ, rho=eQ/eC)


# =====================================================================
if __name__ == "__main__":
    os.makedirs(f"{OUT}/tables", exist_ok=True); os.makedirs(f"{OUT}/figures", exist_ok=True)
    pd.set_option("display.width", 220)

    def golden_min(f, lo, hi, tol=1e-9, iters=300):
        g = (np.sqrt(5)-1)/2
        a_, b_ = lo, hi
        c_, d_ = b_-g*(b_-a_), a_+g*(b_-a_)
        for _ in range(iters):
            if f(c_) < f(d_): b_, d_ = d_, c_; c_ = b_-g*(b_-a_)
            else: a_, c_ = c_, d_; d_ = a_+g*(b_-a_)
            if abs(b_-a_) < tol*(abs(a_)+abs(b_)): break
        return (a_+b_)/2

    # ---- 数值互验 T2：二分解 L(N',D0,Q0)=L(N0,D0,Q0+0.1) ----
    rows = []
    D0 = 300e9
    for N0 in [1e8, 1e9, 1e10, 1e11]:
        for Q0 in [0.5, 0.7, 0.9]:
            target = L(N0, D0, Q0+0.1)
            ana = dN_equiv(N0, Q0, D0, 0.1)
            floor_inf = E + TD(D0, Q0) + k0*(1-Q0)        # N→∞ 极限
            if np.isfinite(ana) and target > floor_inf:
                lo, hi = N0, N0*1e8
                for _ in range(300):
                    mid = np.sqrt(lo*hi)
                    if L(mid, D0, Q0) > target: lo = mid
                    else: hi = mid
                num = np.sqrt(lo*hi); rel = abs(ana-num)/num
            else:
                num, rel = np.inf, 0.0
            rows.append(dict(N=N0, D=D0, Q=Q0, N_max_replaceable=N_max(D0, Q0, 0.1),
                             N_equiv_analytic=ana, N_equiv_numeric=num, rel_err=rel,
                             multiple=ana/N0 if np.isfinite(ana) else np.inf))
    t2 = pd.DataFrame(rows)
    t2.to_csv(f"{OUT}/tables/T24_equiv_substitution.csv", index=False, encoding="utf-8-sig")
    print("T2 等效替代（Q+0.1 等效的参数倍数；inf = 超过 N_max 不可替代）:")
    print(t2.round(4).to_string(index=False))

    # ---- 数值互验 T4 ----
    rows = []
    for C in [1e19, 1e20, 1e21, 1e22, 1e23, 1e24]:
        for Q0 in [0.3, 0.6, 0.9, 1.0]:
            ana = N_star(C, Q0)
            f = lambda lnN: L(np.exp(lnN), C/(6*np.exp(lnN)), Q0)
            num = np.exp(golden_min(f, np.log(1e5), np.log(1e14)))
            rows.append(dict(C=C, Q=Q0, N_star_analytic=ana, N_star_numeric=num,
                             rel_err=abs(ana-num)/num, D_over_N=C/(6*ana**2),
                             N_star_ratio_vs_Q1=ana/N_star(C, 1.0), L_star=L_star(C, Q0)))
    t4 = pd.DataFrame(rows)
    t4.to_csv(f"{OUT}/tables/T25_compute_optimal.csv", index=False, encoding="utf-8-sig")
    print("\nT4 算力最优 N*(C,Q)（N*_ratio_vs_Q1 = 相对 Q=1 的放大倍数）:")
    print(t4.round(4).to_string(index=False))

    # ---- 弹性表 ----
    el_rows = []
    for N0, D0_, Q0 in [(1e9, 300e9, 0.6), (1e9, 300e9, 0.9), (1e10, 1e12, 0.6),
                        (1e10, 1e12, 0.9), (7e10, 2e12, 0.9), (7e10, 2e12, 0.6)]:
        e = elasticities(N0, D0_, Q0)
        el_rows.append(dict(N=N0, D=D0_, Q=Q0, **{k: round(float(v), 4) for k, v in e.items()}))
    t1 = pd.DataFrame(el_rows)
    t1.to_csv(f"{OUT}/tables/T26_elasticities.csv", index=False, encoding="utf-8-sig")
    print("\nT1 弹性（eQ_over_eN = 质量 1% 相当于参数 x%）:")
    print(t1.to_string(index=False))

    # ---- 质量地板 / 渐近表 ----
    fl_rows = []
    for Q0 in [0.5, 0.7, 0.9, 0.95, 1.0]:
        fl_rows.append(dict(Q=Q0, floor_N_inf_D300B=E+TD(300e9, Q0)+k0*(1-Q0),
                            floor_N_inf_D_inf=E+k0*(1-Q0),
                            N_eff_ratio=hN(Q0)**(-1/al), D_eff_ratio=hD(Q0)**(-1/be)))
    t3 = pd.DataFrame(fl_rows)
    t3.to_csv(f"{OUT}/tables/T27_quality_floor.csv", index=False, encoding="utf-8-sig")
    print("\nT3 渐近与有效规模:")
    print(t3.round(4).to_string(index=False))

    # ---- T5 质量–算力汇率 + 数值互验（大步：解 L*(C',Q)=L*(C,Q+0.1)）----
    rows = []
    for C in [1e20, 1e22, 1e24]:
        for Q0 in [0.5, 0.7, 0.9]:
            ex = exchange_rate(C, Q0)
            target = L_star(C, Q0+0.1)
            # L*(C,Q0) 随 C 单调下降且 → E+k0(1-Q0)；若 target 低于该极限则无限算力也追不上
            if target <= E + k0*(1-Q0):
                Cmul = np.inf
            else:
                lo, hi = np.log10(C), np.log10(C)+12
                for _ in range(300):
                    mid = (lo+hi)/2
                    if L_star(10**mid, Q0) > target: lo = mid
                    else: hi = mid
                Cmul = 10**((lo+hi)/2)/C
            rows.append(dict(C=C, Q=Q0, eps_C=ex["eps_C"], eps_Q=ex["eps_Q"], rho=ex["rho"],
                             C_multiple_for_Q_plus_0p1_exact=Cmul,
                             C_multiple_small_step_approx=np.exp(ex["rho"]*np.log((Q0+0.1)/Q0))))
    t5 = pd.DataFrame(rows)
    t5.to_csv(f"{OUT}/tables/T29_quality_compute_exchange.csv", index=False, encoding="utf-8-sig")
    print("\nT5 质量–算力汇率（Q+0.1 等效的算力倍数；inf = 触及地板，算力不可替代）:")
    print(t5.round(4).to_string(index=False))

    # ---- 图 F21 ----
    plt = setup_cjk_matplotlib()
    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK JP", "sans-serif"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    Ngrid = np.logspace(7.5, 11.5, 300)
    # 图1 等损失线 N-Q（D=300B）：解 Q 使 L=Ltar，L 对 Q 线性 ⇒ 闭式
    ax = axes[0]
    for Ltar, c in [(2.4, "#2b8cbe"), (2.2, "#74a9cf"), (2.0, "#fd8d3c")]:
        base_ = E + A*Ngrid**-al + B*(300e9)**-be                 # Q=1 值
        slope_ = kN*A*Ngrid**-al + kD*B*(300e9)**-be + k0          # dL/d(1-Q)
        Qs = 1 - (Ltar - base_)/slope_
        m = (Qs > 0) & (Qs < 1)
        ax.semilogx(Ngrid[m], Qs[m], color=c, lw=2, label=f"L={Ltar}")
    ax.set_xlabel("参数量 N"); ax.set_ylabel("质量 Q")
    ax.set_title("等损失线（D=300B）：右下方向为改进")
    ax.legend(); ax.grid(alpha=.3)
    # 图2 等效替代倍数
    ax = axes[1]
    for Q0, c in [(0.5, "#2b8cbe"), (0.7, "#74a9cf"), (0.9, "#fd8d3c")]:
        mult = np.array([dN_equiv(nn, Q0, 300e9, 0.1)/nn for nn in Ngrid])
        mult[~np.isfinite(mult)] = np.nan
        ax.loglog(Ngrid, mult, color=c, lw=2, label=f"Q={Q0}→{round(Q0+0.1,1)}")
        ax.axvline(N_max(300e9, Q0, 0.1), color=c, ls=":", lw=1)
    ax.set_xlabel("参数量 N"); ax.set_ylabel("等效参数倍数 N'/N")
    ax.set_title("质量+0.1 的等效参数倍数（竖线 = N_max，其右不可替代）")
    ax.legend(); ax.grid(alpha=.3, which="both")
    # 图3 N*(C,Q)
    ax = axes[2]
    Cs = np.logspace(19, 24, 100)
    for Q0, c in [(0.3, "#a63603"), (0.6, "#fd8d3c"), (0.9, "#74a9cf"), (1.0, "#2b8cbe")]:
        ax.loglog(Cs, N_star(Cs, Q0), color=c, lw=2, label=f"Q={Q0}")
    ax.set_xlabel("算力预算 C (FLOPs)"); ax.set_ylabel("最优参数量 N*")
    ax.set_title("算力最优配置：κN>κD ⇒ 质量越低最优模型越大")
    ax.legend(); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/F21_theory.png"); plt.close(fig)

    # ---- 接口：解析式参数与函数说明（供问题三）----
    with open(f"{OUT}/interface/P2_theory.json", "w", encoding="utf-8") as f:
        json.dump(dict(
            model="L̃=E+A N^-α[1+κN(1-Q)]+B D^-β[1+κD(1-Q)]+k0(1-Q); L=L̃·m(p)",
            E=E, A=A, alpha=al, B=B, beta=be, kappa_N=kN, kappa_D=kD, k0=k0,
            N_star="N*=[αA hN(Q)/(βB hD(Q))]^{1/(α+β)}(C/6)^{β/(α+β)}",
            beta_over_sum=be/(al+be),
            dL_dQ="-(κN A N^-α + κD B D^-β + k0)   （对 Q 线性，KKT 中 ∂L/∂Q 与 N,D 有关但与 Q 无关）",
            N_max_replaceable="[A hN(Q+δ)/(δ(κD B D^-β + k0))]^{1/α}",
            quality_floor="L_min(N,D→∞)=E+k0(1-Q)（B6/B7 内建，标注为半合成假设）",
            p_channel=P["p_channel"], s_p=P["s_p"],
            Q_anchor=P["Q_anchor"],
            note="供问题三优化直接调用；以上闭式均在 p=p*（m=1）处成立，p≠p* 时 s(N) 使 p 与 N 耦合，须数值求解"),
            f, ensure_ascii=False, indent=2)
    print("\nQ2-B/C done.")
