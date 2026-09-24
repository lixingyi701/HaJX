# -*- coding: utf-8 -*-
"""
q2_02_theory.py — 问题二·B/C：解析理论 + 数值互验
基于主模型  L(N,D,Q) = E + Ã(Q)·N^-α + B·D^-β + (1-Q)k0,
其中 Ã(Q) = A + k1(1-Q)  （规模-质量交互吸收进 N 项系数）。

解析结果（推导见 问题2解析理论.tex）：
  T1 弹性:  ε_N = -α·Ã(Q)N^-α / L,  ε_D = -β·B·D^-β / L,
            ε_Q = -Q(k0 + k1·N^-α) / L
  T2 等效替代（Q+δ 等效的参数增量）:
            N' = N·[1 - δ(k0+k1N^-α)/(Ã(Q)N^-α)]^(-1/α)，
            当 δ(k0+k1N^-α) ≥ Ã(Q)N^-α 时无有限解 —— 质量地板效应
  T3 质量地板: N→∞ 时 L → E + B·D^-β + (1-Q)k0，
            (1-Q)k0 是任何参数量都无法替代的不可约损失
  T4 算力最优配置（6ND=C）:
            N*(C,Q) = [α·Ã(Q)/(β·B)]^{1/(α+β)}·(C/6)^{β/(α+β)}
            ⇒ 质量越低 Ã 越大 ⇒ 最优 N 份额越大（低质数据时代应养大模型，
              高质数据让"小模型多喂数据"成为最优）
本脚本：数值优化验证 T2/T4 的解析解，并出全部图表。
"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code_q1"))
from q1_00_common import ROOT, setup_cjk_matplotlib

OUT = f"{ROOT}/outputs_q2"
P = json.load(open(f"{OUT}/interface/P2_scaling_law.json"))
E, A, al, B, be, k0, k1 = (P["E"], P["A"], P["alpha"], P["beta"] and P["B"],
                           P["beta"], P["k0"], P["k1"])
# 上行笔误防护，逐个明确读取
E, A, al = P["E"], P["A"], P["alpha"]
B, be = P["B"], P["beta"]
k0, k1 = P["k0"], P["k1"]

def Atil(Q): return A + k1*(1-Q)
def L(N, D, Q): return E + Atil(Q)*N**-al + B*D**-be + (1-Q)*k0

# ---------------- T1 弹性 ----------------
def elasticities(N, D, Q):
    l = L(N, D, Q)
    return dict(eps_N=-al*Atil(Q)*N**-al/l,
                eps_D=-be*B*D**-be/l,
                eps_Q=-Q*(k0+k1*N**-al)/l, L=l)

# ---------------- T2 等效替代（解析） ----------------
def dN_equiv(N, Q, delta=0.1):
    """L(N',D,Q)=L(N,D,Q+delta) 的 N'；无解返回 inf"""
    num = delta*(k0 + k1*N**-al)
    den = Atil(Q)*N**-al
    t = 1 - num/den
    return np.inf if t <= 0 else N*t**(-1/al)

# ---------------- T4 算力最优（解析） ----------------
def N_star(C, Q):
    return (al*Atil(Q)/(be*B))**(1/(al+be)) * (C/6)**(be/(al+be))

# =====================================================================
if __name__ == "__main__":
    rows = []
    # ---- 数值互验 T2：黄金分割搜索 N' ----
    def golden_min(f, lo, hi, tol=1e-6, iters=200):
        g = (np.sqrt(5)-1)/2
        a_, b_ = lo, hi
        c_, d_ = b_-g*(b_-a_), a_+g*(b_-a_)
        for _ in range(iters):
            if f(c_) < f(d_): b_, d_ = d_, c_; c_ = b_-g*(b_-a_)
            else: a_, c_ = c_, d_; d_ = a_+g*(b_-a_)
            if abs(b_-a_) < tol*(abs(a_)+abs(b_)): break
        return (a_+b_)/2
    D0 = 300e9
    for N0 in [1e8, 1e9, 1e10]:
        for Q0 in [0.5, 0.7, 0.9]:
            target = L(N0, D0, Q0+0.1)
            ana = dN_equiv(N0, Q0, 0.1)
            if np.isfinite(ana):
                # 数值：解 L(N',D0,Q0)=target（单调递减，二分）
                lo, hi = N0, N0*1e6
                for _ in range(200):
                    mid = np.sqrt(lo*hi)
                    if L(mid, D0, Q0) > target: lo = mid
                    else: hi = mid
                num = np.sqrt(lo*hi)
                rel = abs(ana-num)/num
            else:
                num, rel = np.inf, 0.0
            rows.append(dict(N=N0, Q=Q0, N_equiv_analytic=ana,
                             N_equiv_numeric=num, rel_err=rel,
                             multiple=ana/N0 if np.isfinite(ana) else np.inf))
    t2 = pd.DataFrame(rows)
    t2.to_csv(f"{OUT}/tables/T24_equiv_substitution.csv", index=False)
    print("T2 等效替代（Q+0.1 等效的参数倍数）:")
    print(t2.round(4).to_string(index=False))

    # ---- 数值互验 T4 ----
    rows = []
    for C in [1e20, 1e21, 1e22, 1e23]:
        for Q0 in [0.3, 0.6, 0.9, 1.0]:
            ana = N_star(C, Q0)
            f = lambda lnN: L(np.exp(lnN), C/(6*np.exp(lnN)), Q0)
            num = np.exp(golden_min(f, np.log(1e6), np.log(1e13)))
            rows.append(dict(C=C, Q=Q0, N_star_analytic=ana, N_star_numeric=num,
                             rel_err=abs(ana-num)/num,
                             D_over_N=C/(6*ana**2)))
    t4 = pd.DataFrame(rows)
    t4.to_csv(f"{OUT}/tables/T25_compute_optimal.csv", index=False)
    print("\nT4 算力最优 N*(C,Q):")
    print(t4.round(4).to_string(index=False))

    # ---- 弹性表 ----
    el_rows = []
    for N0, D0_, Q0 in [(1e9, 300e9, 0.6), (1e9, 300e9, 0.9), (1e10, 1e12, 0.6),
                        (1e10, 1e12, 0.9), (7e10, 2e12, 0.9)]:
        e = elasticities(N0, D0_, Q0)
        el_rows.append(dict(N=N0, D=D0_, Q=Q0, **{k: round(v, 4) for k, v in e.items()}))
    t1 = pd.DataFrame(el_rows)
    t1.to_csv(f"{OUT}/tables/T26_elasticities.csv", index=False)
    print("\nT1 弹性:")
    print(t1.to_string(index=False))

    # ---- 质量地板表 ----
    fl_rows = []
    for Q0 in [0.5, 0.7, 0.9, 0.95, 1.0]:
        fl_rows.append(dict(Q=Q0, floor_N_inf_D300B=E+B*(300e9)**-be+(1-Q0)*k0,
                            floor_N_inf_D_inf=E+(1-Q0)*k0))
    t3 = pd.DataFrame(fl_rows)
    t3.to_csv(f"{OUT}/tables/T27_quality_floor.csv", index=False)
    print("\nT3 质量地板:")
    print(t3.round(4).to_string(index=False))

    # ---- 图 ----
    plt = setup_cjk_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    # 图1 等损失线 N-Q
    ax = axes[0]
    Ngrid = np.logspace(7.5, 11, 200)
    for Ltar, c in [(2.4, "#2b8cbe"), (2.2, "#74a9cf"), (2.0, "#fd8d3c")]:
        # 固定 D=300B, 求 Q(N) 使 L=Ltar
        Qs = 1 - (Ltar - E - A*Ngrid**-al - B*(300e9)**-be)/(k0 + k1*Ngrid**-al)
        m = (Qs > 0) & (Qs < 1)
        ax.semilogx(Ngrid[m], Qs[m], color=c, lw=2, label=f"L={Ltar}")
    ax.set_xlabel("参数量 N"); ax.set_ylabel("质量 Q")
    ax.set_title("等损失线（D=300B）：右下方向为改进")
    ax.legend(); ax.grid(alpha=.3)
    # 图2 等效替代倍数
    ax = axes[1]
    for Q0, c in [(0.5, "#2b8cbe"), (0.7, "#74a9cf"), (0.9, "#fd8d3c")]:
        mult = [dN_equiv(nn, Q0, 0.1)/nn for nn in Ngrid]
        mult = [m_ if np.isfinite(m_) else np.nan for m_ in mult]
        ax.loglog(Ngrid, mult, color=c, lw=2, label=f"Q={Q0}→{round(Q0+0.1,1)}")
    ax.set_xlabel("参数量 N"); ax.set_ylabel("等效参数倍数 N'/N")
    ax.set_title("质量+0.1 的等效参数倍数（发散点=质量地板）")
    ax.legend(); ax.grid(alpha=.3, which="both")
    # 图3 N*(C,Q)
    ax = axes[2]
    Cs = np.logspace(19, 24, 100)
    for Q0, c in [(0.3, "#a63603"), (0.6, "#fd8d3c"), (0.9, "#74a9cf"), (1.0, "#2b8cbe")]:
        ax.loglog(Cs, N_star(Cs, Q0), color=c, lw=2, label=f"Q={Q0}")
    ax.set_xlabel("算力预算 C (FLOPs)"); ax.set_ylabel("最优参数量 N*")
    ax.set_title("算力最优配置：质量越低最优模型越大")
    ax.legend(); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/F21_theory.png"); plt.close(fig)

    # ---- 接口：解析式参数与函数句柄说明 ----
    with open(f"{OUT}/interface/P2_theory.json", "w", encoding="utf-8") as f:
        json.dump(dict(
            model="L=E+Ã(Q)N^-α+BD^-β+(1-Q)k0, Ã(Q)=A+k1(1-Q)",
            E=E, A=A, alpha=al, B=B, beta=be, k0=k0, k1=k1,
            N_star="N*=[αÃ(Q)/(βB)]^{1/(α+β)}(C/6)^{β/(α+β)}",
            beta_over_sum=be/(al+be),
            quality_floor="L_min(N→∞,D→∞)=E+(1-Q)k0",
            note="供问题三优化直接调用"), f, ensure_ascii=False, indent=2)
    print("\nQ2-B/C done.")
