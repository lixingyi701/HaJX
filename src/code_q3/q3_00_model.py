# -*- coding: utf-8 -*-
"""
q3_00_model.py — 问题三公共模型：广义标度律 + 三项成本 + 向量化最优化求解器

优化问题（赛题原文）
    min_{N,D,Q}  L = E + A N^-α h_N(Q) + B D^-β h_D(Q) + k0(1-Q)
                 h_X(Q) = 1 + κ_X(1-Q)
    s.t.  6ND + D[g(Q)-g(Q0)]₊ + ηNDL_ctx ≤ C ,   Q0 ≤ Q ≤ 1

降维：三项成本都 ∝ D，且 L 对 D 严格递减 ⇒ 约束取等
    D(N,Q) = C / [κN + Δg(Q)] ,   κ = 6 + ηL_ctx ,  Δg = g(Q) - g(Q0)
Q<Q0 不省钱却抬高 Loss ⇒ 可行域收缩为 Q∈[Q0,1]。
于是原问题 = 在 (logN, Q) 上最小化剖面 Loss；本文件给出全向量化求解器：
    内层：对 logN 做黄金分割（L 沿 N 单峰：N 项递减、D 项随 N 递增）
    外层：对 Q 做三级网格加密（41 点 × 3 级，最终步长 < 1e-5）+ 端点保护
所有参数读自问题二接口，不重新估计。
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code_q1"))
from q1_00_common import ROOT, setup_cjk_matplotlib  # noqa: E402,F401

OUT = f"{ROOT}/outputs_q3"
TAB, FIG, IFACE = f"{OUT}/tables", f"{OUT}/figures", f"{OUT}/interface"
for _d in (TAB, FIG, IFACE):
    os.makedirs(_d, exist_ok=True)

P2 = json.load(open(f"{ROOT}/outputs_q2/interface/P2_scaling_law.json", encoding="utf-8"))
PAR = dict(E=P2["E"], A=P2["A"], al=P2["alpha"], B=P2["B"], be=P2["beta"],
           kappa_N=P2["kappa_N"], kappa_D=P2["kappa_D"], k0=P2["k0"])

ETA = 2e-4                                   # 赛题给定
LCTX_CRIT = 6 / ETA                          # 30000，解析临界值
# 主口径：Q_B = q̄(p)，Q0 = q̄(p*)。不再除以已废弃的 Pile 锚点 0.5608。
Q0_MAIN = float(P2["Q0_main"])                              # 0.6609
# 副口径：网页域 pile_cc 在同一锚点下的域质量（Q_B = q，不另做缩放）
Q0_WEB = 0.676134904846549

# ---------------- 三种质量成本函数（附录 B）：(g, g') ----------------
G_FUNCS = {
    "指数型": (lambda Q: 1e7 * np.exp(6.0 * Q),
               lambda Q: 1e7 * 6.0 * np.exp(6.0 * Q)),
    "幂函数型": (lambda Q: 5e9 * np.power(Q, 4.0),
               lambda Q: 5e9 * 4.0 * np.power(Q, 3.0)),
    "对数型": (lambda Q: 2e9 * np.log1p(10.0 * Q),
              lambda Q: 2e9 * 10.0 / (1 + 10.0 * Q)),
}
G_LIST = list(G_FUNCS)
LCTX_C7 = [2048, 4096, 8192, 32768, 131072]      # C7 max_position_embeddings 的全部取值


def hN(Q, p=PAR):
    return 1.0 + p["kappa_N"] * (1.0 - Q)


def hD(Q, p=PAR):
    return 1.0 + p["kappa_D"] * (1.0 - Q)


def loss(N, D, Q, p=PAR):
    return (p["E"] + p["A"] * hN(Q, p) * N ** -p["al"]
            + p["B"] * hD(Q, p) * D ** -p["be"] + (1.0 - Q) * p["k0"])


def N_star_closed(C, Q, kappa, p=PAR):
    """Δg=0 时的闭式解（单价 6 → κ=6+ηL_ctx）"""
    al, be = p["al"], p["be"]
    return ((al * p["A"] * hN(Q, p) / (be * p["B"] * hD(Q, p))) ** (1 / (al + be))
            * (C / kappa) ** (be / (al + be)))


# ---------------- 求解器 ----------------
_GR = (np.sqrt(5) - 1) / 2


def _profile(C, Q, dg, kappa, p, lo=1.0, hi=17.0, it=70):
    """给定 Q（及对应 Δg），对 logN 黄金分割，返回 (L_min, logN*)；全部广播"""
    C, Q, dg = np.broadcast_arrays(np.asarray(C, float), np.asarray(Q, float),
                                   np.asarray(dg, float))

    def f(x):
        N = 10.0 ** x
        D = C / (kappa * N + dg)
        return loss(N, D, Q, p)

    a = np.full(C.shape, lo); b = np.full(C.shape, hi)
    for _ in range(it):
        c = b - _GR * (b - a); d = a + _GR * (b - a)
        m = f(c) < f(d)
        b = np.where(m, d, b); a = np.where(m, a, c)
    x = (a + b) / 2
    return f(x), x


def solve(C, g="对数型", Lctx=2048, Q0=Q0_MAIN, p=PAR, eta=ETA, nQ=41, levels=3):
    """对一组预算 C（向量）求最优 (N*,D*,Q*,L*)；返回 dict of 1-D arrays"""
    C = np.atleast_1d(np.asarray(C, float))[:, None]
    Q0 = np.asarray(Q0, float)
    Q0 = np.broadcast_to(Q0[:, None] if Q0.ndim == 1 else Q0, C.shape)   # 允许逐行不同 Q0
    p = {k: (np.asarray(v)[:, None] if np.ndim(v) == 1 else v) for k, v in p.items()}
    gf = G_FUNCS[g][0]; g0 = gf(Q0)
    kappa = 6 + eta * Lctx
    lo = Q0.copy(); hi = np.full(C.shape, 1.0)
    for _ in range(levels):
        Qg = lo + (hi - lo) * np.linspace(0, 1, nQ)[None, :]
        dg = np.maximum(gf(Qg) - g0, 0.0)
        Lm, _ = _profile(C, Qg, dg, kappa, p)
        j = np.argmin(Lm, axis=1)
        step = (hi - lo) / (nQ - 1)
        Qb = Qg[np.arange(len(C)), j][:, None]
        lo = np.maximum(Qb - step, Q0); hi = np.minimum(Qb + step, 1.0)
    Qs = Qb
    dg = np.maximum(gf(Qs) - g0, 0.0)
    Ls, xs = _profile(C, Qs, dg, kappa, p)
    for Qe in (Q0.copy(), np.full(C.shape, 1.0)):   # 端点保护
        dge = np.maximum(gf(Qe) - g0, 0.0)
        Le, xe = _profile(C, Qe, dge, kappa, p)
        better = Le < Ls - 1e-13
        Qs = np.where(better, Qe, Qs); Ls = np.where(better, Le, Ls)
        xs = np.where(better, xe, xs); dg = np.where(better, dge, dg)
    N = 10.0 ** xs; D = C / (kappa * N + dg)
    r = dict(C=C.ravel(), N=N.ravel(), D=D.ravel(), Q=Qs.ravel(), L=Ls.ravel(),
             s_train=(6 * N * D / C).ravel(),
             s_attn=(eta * N * D * Lctx / C).ravel(),
             s_Q=(D * dg / C).ravel())
    r["DN"] = r["D"] / r["N"]
    r["regime"] = classify(r["Q"], Q0.ravel())
    return r


def c_crit(g, Lctx, Q0, p=PAR, eta=ETA, which="leave", lo=15.0, hi=26.0, it=40):
    """二分法求临界预算 log10 C_crit（向量化：p 的各参数可为长度 n 的数组）
    which="leave": 离开 Q0 角点（开始提质）的最小预算；
    which="full" : 进入 Q=1 角点（提满）的最小预算。"""
    n = max([np.size(v) for v in p.values()] + [np.size(Q0), 1])
    a = np.full(n, lo); b = np.full(n, hi)
    for _ in range(it):
        m = (a + b) / 2
        r = solve(10.0 ** m, g, Lctx, Q0, p, eta, nQ=21, levels=3)
        hit = r["regime"] > 0 if which == "leave" else r["regime"] == 2
        b = np.where(hit, m, b); a = np.where(hit, a, m)
    return (a + b) / 2


def classify(Q, Q0, tol=1e-4):
    """KKT 活跃约束集：0 = Q=Q0 角点（不提质）；1 = 内点；2 = Q=1 角点（提满）"""
    Q = np.asarray(Q)
    return np.where(Q <= Q0 + tol, 0, np.where(Q >= 1 - tol, 2, 1))


REGIME_NAME = {0: "Q0角点(不提质)", 1: "内点(部分提质)", 2: "Q=1角点(提满)"}


def phi_ratio(C, g, Lctx, Qeval, Q0, p=PAR, eta=ETA):
    """KKT 解析判据 Φ：在质量固定为 Qeval 时的最优 (N,D) 处，
        Φ = 提质的边际 Loss 收益 / 提质挤占 D 的边际 Loss 损失
          = (κ_N A N^-α + κ_D B D^-β + k0)(κN + Δg) / (β B D^-β g'(Qeval))
    Φ(Q0)<1 ⇔ 角点 Q*=Q0 满足 KKT（不提质）；Φ(1)>1 ⇔ Q*=1 角点。"""
    kappa = 6 + eta * Lctx
    gf, gd = G_FUNCS[g]
    dg = max(gf(Qeval) - gf(Q0), 0.0)
    _, x = _profile(np.asarray(C, float), np.asarray(Qeval, float), dg, kappa, p)
    N = 10.0 ** x; D = C / (kappa * N + dg)
    num = (p["kappa_N"] * p["A"] * N ** -p["al"]
           + p["kappa_D"] * p["B"] * D ** -p["be"] + p["k0"]) * (kappa * N + dg)
    den = p["be"] * p["B"] * D ** -p["be"] * gd(Qeval)
    return float(num / den)
