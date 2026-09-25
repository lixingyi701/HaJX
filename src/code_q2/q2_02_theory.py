# -*- coding: utf-8 -*-
"""
q2_02_theory.py — 幂次耦合标度律的弹性、替代条件与算力最优配置
主式（p 取附件 B 的隐含配比，乘子为 1）：
    L = E + A N^{-α} Q^{-κ_N} + B D^{-β} Q^{-κ_D}
    T_N = A N^{-α} Q^{-κ_N}，T_D = B D^{-β} Q^{-κ_D}，R = T_N+T_D
弹性（对 L 取对数）：
    ε_N = -α T_N / L，  ε_D = -β T_D / L，  ε_Q = -(κ_N T_N + κ_D T_D) / L
质量提升 δ 与参数的替代（固定 D）：
    t = (Q/(Q+δ))^{κ_N} + [T_D(Q)/T_N(Q)] · [(Q+δ)^{-κ_D} - Q^{-κ_D}] · Q^{κ_N} / Q^{-κ_D}
    整理后 N' = N · t^{-1/α}，t>0 才有有限解。
    局部：dlnN/dlnQ |_{L,D} = (κ_N T_N + κ_D T_D) / (α T_N)
算力最优（6ND=C）：
    N* = [α A Q^{-κ_N} / (β B Q^{-κ_D})]^{1/(α+β)} (C/6)^{β/(α+β)}
沿最优前沿，1% 质量等价的算力百分比 ρ = κ_N/α + κ_D/β（与 C、Q 无关）。
渐近：N,D→∞ 时 L→E，不可约损失不随 Q 变化。
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
assert P.get("q_shape") == "pow", "本文件只服务于幂次主式"
E, A, al = P["E"], P["A"], P["alpha"]
B, be = P["B"], P["beta"]
kN, kD = P["kappa_N"], P["kappa_D"]


def TN(N, Q):
    return A * N**-al * Q**-kN


def TD(D, Q):
    return B * D**-be * Q**-kD


def loss(N, D, Q):
    return E + TN(N, Q) + TD(D, Q)


def elasticities(N, D, Q):
    l = loss(N, D, Q)
    tN, tD = TN(N, Q), TD(D, Q)
    eN, eD = -al * tN / l, -be * tD / l
    eQ = -(kN * tN + kD * tD) / l
    return dict(eps_N=eN, eps_D=eD, eps_Q=eQ, eQ_over_eN=eQ / eN,
                N_eff_over_N=Q**(kN / al), D_eff_over_D=Q**(kD / be), L=l)


def subst_factor(N, D, Q, delta):
    """t = (N'/N)^{-α}；t<=0 表示不存在有限 N'。"""
    Q2 = Q + delta
    tN, tD = TN(N, Q), TD(D, Q)
    # A N'^{-α} Q^{-κN} = A N^{-α} Q2^{-κN} + B D^{-β}(Q2^{-κD} - Q^{-κD})
    rhs = tN * (Q / Q2)**kN + tD * ((Q2)**-kD - Q**-kD) * (Q**kN)
    return rhs / tN


def N_equiv(N, D, Q, delta=0.1):
    t = subst_factor(N, D, Q, delta)
    return np.inf if t <= 0 else N * t**(-1 / al)


def N_star(C, Q):
    return ((al * A * Q**-kN) / (be * B * Q**-kD))**(1 / (al + be)) * (C / 6)**(be / (al + be))


def L_star(C, Q):
    N = N_star(C, Q)
    return loss(N, C / (6 * N), Q)


RHO = kN / al + kD / be


def golden_min(f, lo, hi, tol=1e-9, iters=80):
    g = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
        if abs(b - a) < tol * (abs(a) + abs(b) + 1):
            break
    return (a + b) / 2


if __name__ == "__main__":
    os.makedirs(f"{OUT}/tables", exist_ok=True)
    os.makedirs(f"{OUT}/figures", exist_ok=True)
    pd.set_option("display.width", 220)
    D0 = 300e9
    rows = []
    for N0 in [1e8, 1e9, 1e10, 1e11]:
        for Q0 in [0.5, 0.7, 0.9]:
            target = loss(N0, D0, Q0 + 0.1)
            ana = N_equiv(N0, Q0, D0, 0.1) if False else N_equiv(N0, D0, Q0, 0.1)
            floor_inf = E + TD(D0, Q0)  # N→∞ 且质量仍为 Q0 时的损失，不能用来追 Q0+0.1
            # 追的是更低的损失，极限是 E+TD(D0, Q0)，若目标低于该极限则无解
            if np.isfinite(ana) and target > E + TD(D0, Q0) - 1e-9:
                num = np.exp(golden_min(lambda z: (loss(np.exp(z), D0, Q0) - target)**2,
                                         np.log(N0), np.log(N0) + 20))
                # 上式最小化平方可能停在边界；改用二分
                lo, hi = np.log(N0 * 0.5), np.log(N0) + 25
                if loss(np.exp(hi), D0, Q0) > target:
                    num = np.inf
                else:
                    for _ in range(60):
                        mid = 0.5 * (lo + hi)
                        if loss(np.exp(mid), D0, Q0) > target:
                            lo = mid
                        else:
                            hi = mid
                    num = np.exp(0.5 * (lo + hi))
                rel = abs(ana - num) / num
            else:
                num, rel = np.inf, np.nan
            rows.append(dict(N=N0, D=D0, Q=Q0, delta=0.1,
                             N_equiv=ana, N_equiv_numeric=num, rel_err=rel,
                             multiple=ana / N0 if np.isfinite(ana) else np.inf,
                             local_dlnN_dlnQ=(kN * TN(N0, Q0) + kD * TD(D0, Q0)) / (al * TN(N0, Q0))))
    t2 = pd.DataFrame(rows)
    t2.to_csv(f"{OUT}/tables/T24_equiv_substitution.csv", index=False, encoding="utf-8-sig")
    print("质量 +0.1 的等效参数倍数（inf = 数据项上的质量收益已超过整个参数项）")
    print(t2.round(4).to_string(index=False))

    rows = []
    for C in [1e19, 1e20, 1e21, 1e22, 1e23, 1e24]:
        for Q0 in [0.5, 0.7, 0.9, 1.0]:
            ana = N_star(C, Q0)
            num = np.exp(golden_min(lambda z: loss(np.exp(z), C / (6 * np.exp(z)), Q0),
                                     np.log(1e6), np.log(1e14)))
            rows.append(dict(C=C, Q=Q0, N_star=ana, N_star_numeric=num,
                             rel_err=abs(ana - num) / num, D_over_N=C / (6 * ana**2),
                             N_ratio_vs_Q1=ana / N_star(C, 1.0), L_star=L_star(C, Q0)))
    t4 = pd.DataFrame(rows)
    t4.to_csv(f"{OUT}/tables/T25_compute_optimal.csv", index=False, encoding="utf-8-sig")
    print("\n算力最优 N*")
    print(t4.round(4).to_string(index=False))

    el_rows = []
    for N0, D0_, Q0 in [(1e9, 3e11, 0.6), (1e9, 3e11, 0.9), (1e10, 1e12, 0.6),
                        (1e10, 1e12, 0.9), (7e10, 2e12, 0.6), (7e10, 2e12, 0.9)]:
        e = elasticities(N0, D0_, Q0)
        el_rows.append(dict(N=N0, D=D0_, Q=Q0, **{k: float(v) for k, v in e.items()}))
    t1 = pd.DataFrame(el_rows)
    t1.to_csv(f"{OUT}/tables/T26_elasticities.csv", index=False, encoding="utf-8-sig")
    print("\n弹性（ε_Q/ε_N：质量提高 1% 相当于参数增加的百分比）")
    print(t1.round(4).to_string(index=False))

    fl = []
    for Q0 in [0.5, 0.7, 0.9, 1.0]:
        fl.append(dict(Q=Q0, L_as_N_inf_D300B=E + TD(300e9, Q0), L_as_ND_inf=E,
                       N_eff_over_N=Q0**(kN / al), D_eff_over_D=Q0**(kD / be)))
    pd.DataFrame(fl).to_csv(f"{OUT}/tables/T27_quality_floor.csv", index=False, encoding="utf-8-sig")

    rows = []
    for C in [1e19, 1e22, 1e24]:
        for Q0 in [0.5, 0.7, 0.9]:
            target = L_star(C, Q0 + 0.1)
            lo, hi = np.log(C), np.log(C) + 8
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if L_star(np.exp(mid), Q0) > target:
                    lo = mid
                else:
                    hi = mid
            exact = np.exp(0.5 * (lo + hi)) / C
            closed = ((Q0 + 0.1) / Q0)**RHO
            rows.append(dict(C=C, Q=Q0, rho=RHO, C_multiple_exact=exact, C_multiple_closed=closed,
                             rel_err=abs(exact - closed) / exact))
    t5 = pd.DataFrame(rows)
    t5.to_csv(f"{OUT}/tables/T29_quality_compute_exchange.csv", index=False, encoding="utf-8-sig")
    print(f"\n前沿汇率 ρ=κN/α+κD/β={RHO:.4f}（质量乘子等价的算力乘子）")
    print(t5.round(4).to_string(index=False))

    plt = setup_cjk_matplotlib()
    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK JP", "sans-serif"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    Ngrid = np.logspace(8, 11.5, 200)
    ax = axes[0]
    for Q0, c in [(0.5, "#a63603"), (0.7, "#fd8d3c"), (0.9, "#2b8cbe")]:
        mult = np.array([N_equiv(nn, 300e9, Q0, 0.1) / nn for nn in Ngrid])
        mult[~np.isfinite(mult)] = np.nan
        ax.loglog(Ngrid, mult, color=c, lw=2, label=f"Q={Q0}→{Q0+0.1:.1f}")
    ax.set_xlabel("N"); ax.set_ylabel("N'/N"); ax.set_title("质量 +0.1 的等效参数倍数")
    ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
    ax = axes[1]
    Cs = np.logspace(19, 24, 80)
    for Q0, c in [(0.5, "#a63603"), (0.7, "#fd8d3c"), (0.9, "#74a9cf"), (1.0, "#2b8cbe")]:
        ax.loglog(Cs, [N_star(c_, Q0) for c_ in Cs], color=c, lw=2, label=f"Q={Q0}")
    ax.set_xlabel("C (FLOPs)"); ax.set_ylabel("N*")
    ax.set_title("κN>κD 时，质量越低最优 N 越大"); ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
    ax = axes[2]
    for Q0, c in [(0.5, "#a63603"), (0.7, "#fd8d3c"), (0.9, "#2b8cbe")]:
        ax.semilogx(Cs, [L_star(c_, Q0) for c_ in Cs], color=c, lw=2, label=f"Q={Q0}")
    ax.set_xlabel("C (FLOPs)"); ax.set_ylabel("最优损失")
    ax.set_title("算力—质量前沿"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/F21_theory.png"); plt.close(fig)

    with open(f"{OUT}/interface/P2_theory.json", "w", encoding="utf-8") as f:
        json.dump(dict(
            model=P["form"], selected_model=P["selected_model"],
            E=E, A=A, alpha=al, B=B, beta=be, kappa_N=kN, kappa_D=kD, k0=0.0, q_shape="pow",
            eps_Q="-(κ_N T_N + κ_D T_D)/L",
            N_equiv="N' = N t^{-1/α}，t = (Q/(Q+δ))^{κ_N} + (T_D/T_N) Q^{κ_N} [(Q+δ)^{-κ_D}-Q^{-κ_D}]",
            N_star="[α A Q^{-κ_N} /(β B Q^{-κ_D})]^{1/(α+β)} (C/6)^{β/(α+β)}",
            rho_quality_per_compute=RHO,
            rho_meaning="沿 6ND=C 的最优前沿，C 乘 (Q'/Q)^ρ 与把质量从 Q 提到 Q' 使损失下降相同",
            limit="N,D→∞ 时 L→E，与 Q 无关",
            p_channel=P["p_channel"], Q0_main=P["Q0_main"],
            note="闭式在配比乘子为 1 时成立，即附件 B 的隐含配比"),
            f, ensure_ascii=False, indent=2)
    print("Q2 theory done.")
