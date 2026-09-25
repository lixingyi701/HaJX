# -*- coding: utf-8 -*-
"""
q3_00_model.py — 问题三公共模型（Spec v6）

主目标读取 P2 幂次 M4+地板，配比乘子只作用在随规模变化的两项：
    T_N = A N^{-α} Q^{-κ_N},  T_D = B D^{-β} Q^{-κ_D}
    m_* = exp(s_red * φ̃(p*))
    L = E + k0(1-Q) + m_*(T_N + T_D)
约束：6ND + D[g(Q)-g(Q0)]_+ + η N D L_ctx ≤ C，Q∈[Q0,1]。
三类附录 g 在 [Q0,1] 上严格递增，故正部等于 Δg。∂L/∂D<0，预算用尽后
    D = C / (K N + Δg),  K = 6 + η L_ctx
内层对 log10 N 黄金分割，外层比较 Q0、1 与 Q 网格上的局部极小。
"""
import csv
import os
import sys
from pathlib import Path

import logging

import numpy as np

logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code_q1"))
from q1_00_common import ROOT, setup_cjk_matplotlib  # noqa: E402,F401

OUT = f"{ROOT}/outputs_q3"
TAB, FIG, IFACE = f"{OUT}/tables", f"{OUT}/figures", f"{OUT}/interface"
for _d in (TAB, FIG, IFACE):
    os.makedirs(_d, exist_ok=True)

P2_PATH = f"{ROOT}/outputs_q2/interface/P2_scaling_law.json"
P1_SUMMARY_PATH = f"{ROOT}/outputs_q1/interface/P1_summary.json"
P_STAR_PATH = f"{ROOT}/outputs_q1/interface/P1_p_star.csv"
P1_Q_PATH = f"{ROOT}/outputs_q1/interface/P1_Q_domain.csv"
C7_PATH = Path(ROOT).parent / "附件" / "C_efficiency_evolution" / "model_architecture_metadata.csv"

import json

with open(P2_PATH, encoding="utf-8") as _f:
    P2 = json.load(_f)
with open(P1_SUMMARY_PATH, encoding="utf-8") as _f:
    P1 = json.load(_f)

if P2.get("q_shape") != "pow" or P2.get("selected_model") != "M4+F" or not P2.get("floor"):
    raise ValueError("P2 主式不是幂次 M4+地板，不能按本版运行 Q3")
if P2.get("q_mapping", {}).get("main", {}).get("name") != "identity":
    raise ValueError("P2 主质量映射尚未冻结为 identity，不能运行 Q3")

PAR = dict(E=P2["E"], A=P2["A"], al=P2["alpha"], B=P2["B"], be=P2["beta"],
           kappa_N=P2["kappa_N"], kappa_D=P2["kappa_D"], k0=P2["k0"])
_pc = P2["p_channel"]
S_RED = float(_pc["s_red_main"])
PHI_TILDE = float(_pc["phi_tilde_pstar"])
M_STAR = float(np.exp(S_RED * PHI_TILDE))
if not np.isclose(M_STAR, 0.8989423, atol=5e-7):
    raise ValueError(f"m_*={M_STAR} 与 Spec 快照 0.8989423 不一致，需先修订 Spec")
H_TOT_SCALE = float(np.exp(float(_pc["s_train"]) * PHI_TILDE))
EVIDENCE = (
    f"q_shape={P2['q_shape']}; model={P2['selected_model']}; floor={P2['floor']}; "
    f"train={P2['training_data']}; holdout={P2['holdout']}; "
    f"p_channel={_pc['status']}"
)

ETA = 2e-4
LCTX_CRIT = 6 / ETA
Q0_MAIN = float(P2["Q0_main"])
if not np.isclose(Q0_MAIN, float(P1["Qbar_pstar_eqweight"]), atol=1e-12):
    raise ValueError("P1 与 P2 的主质量基线不一致")
if not np.isclose(Q0_MAIN, 0.6624116679870492, atol=1e-15):
    raise ValueError(f"Q0_main={Q0_MAIN} 与 Spec 快照不一致，需先修订 Spec")

with open(P1_Q_PATH, encoding="utf-8-sig") as _f:
    _web = [r for r in csv.DictReader(_f) if r["mixture_domain"] == "pile_cc"]
if len(_web) != 1:
    raise ValueError("P1 领域质量中必须恰有一个 pile_cc")
Q0_WEB = float(_web[0]["Q_final"])

with open(P_STAR_PATH, encoding="utf-8-sig") as _f:
    _prows = list(csv.DictReader(_f))
if not _prows or "p_star_eqweight" not in _prows[0] or "domain" not in _prows[0]:
    raise ValueError("P1_p_star.csv 缺少 domain 或 p_star_eqweight")
P_STAR_DOMAINS = [r["domain"] for r in _prows]
_w = np.array([float(r["p_star_eqweight"]) for r in _prows])
if len(P_STAR_DOMAINS) != 17 or abs(float(_w.sum()) - 1.0) > 1e-8:
    raise ValueError("主配比必须是 17 域且份额和为 1")

G_FUNCS = {
    "指数型": (lambda Q: 1e7 * np.exp(6.0 * Q),
               lambda Q: 1e7 * 6.0 * np.exp(6.0 * Q)),
    "幂函数型": (lambda Q: 5e9 * np.power(Q, 4.0),
               lambda Q: 5e9 * 4.0 * np.power(Q, 3.0)),
    "对数型": (lambda Q: 2e9 * np.log1p(10.0 * Q),
              lambda Q: 2e9 * 10.0 / (1.0 + 10.0 * Q)),
}
G_LIST = list(G_FUNCS)
_q_check = np.linspace(Q0_MAIN, 1.0, 9)
for _name, (_gf, _gd) in G_FUNCS.items():
    if np.any(np.asarray(_gd(_q_check), float) <= 0):
        raise ValueError(f"{_name} 在 [Q0,1] 上不是严格递增，不能去掉正部")

_C7_EXPECTED = [2048, 4096, 8192, 32768, 131072]
if C7_PATH.is_file():
    with C7_PATH.open(encoding="utf-8-sig") as _f:
        LCTX_C7 = sorted({int(float(r["max_position_embeddings"])) for r in csv.DictReader(_f)
                         if r.get("max_position_embeddings")})
    C7_READ = True
    if LCTX_C7 != _C7_EXPECTED:
        raise ValueError(f"C7 上下文档位与 Spec 不一致: {LCTX_C7}")
else:
    # 附件原表不在当前工作区。五档是 Spec 已核对的情景集合，本次不能重读 CSV。
    LCTX_C7 = list(_C7_EXPECTED)
    C7_READ = False

CLASS_TOL = 1e-4
TIE_ATOL = 1e-8
LOGN_LO, LOGN_HI = 1.0, 17.0
STATE_NAME = {0: "LOWER", 1: "INTERIOR", 2: "UPPER"}
_GR = (np.sqrt(5) - 1) / 2


def loss(N, D, Q, p=None, m=None, loss_scale=1.0):
    """幂次 M4+地板。m 默认 m_*；loss_scale 用于 H_tot 把整份 m=1 损失乘正常数。"""
    if p is None:
        p = PAR
    if m is None:
        m = M_STAR
    tn = p["A"] * np.power(N, -p["al"]) * np.power(Q, -p["kappa_N"])
    td = p["B"] * np.power(D, -p["be"]) * np.power(Q, -p["kappa_D"])
    return loss_scale * (p["E"] + (1.0 - Q) * p["k0"] + m * (tn + td))


def N_star_closed(C, Q, kappa, p=None):
    """固定 Q 且 Δg=0 时的闭式 N*。m_* 与地板在该条件下抵消。"""
    if p is None:
        p = PAR
    al, be = p["al"], p["be"]
    aq = p["A"] * np.power(Q, -p["kappa_N"])
    bq = p["B"] * np.power(Q, -p["kappa_D"])
    return ((al * aq) / (be * bq)) ** (1.0 / (al + be)) * (C / kappa) ** (be / (al + be))


def _as_param(p):
    return {k: (np.asarray(v)[:, None] if np.ndim(v) == 1 else v) for k, v in p.items()}


def _profile(C, Q, dg, kappa, p, lo=LOGN_LO, hi=LOGN_HI, it=70, m=None, loss_scale=1.0):
    """给定 Q 与 Δg，对 log10 N 黄金分割。返回 (L_min, logN*)。"""
    if m is None:
        m = M_STAR
    C, Q, dg = np.broadcast_arrays(np.asarray(C, float), np.asarray(Q, float),
                                   np.asarray(dg, float))

    def f(x):
        n = 10.0 ** x
        d = C / (kappa * n + dg)
        return loss(n, d, Q, p, m=m, loss_scale=loss_scale)

    a = np.full(C.shape, lo)
    b = np.full(C.shape, hi)
    for _ in range(it):
        c = b - _GR * (b - a)
        d = a + _GR * (b - a)
        take_c = f(c) < f(d)
        b = np.where(take_c, d, b)
        a = np.where(take_c, a, c)
    x = (a + b) / 2
    return f(x), x


def classify(Q, Q0, tol=CLASS_TOL):
    Q = np.asarray(Q, float)
    Q0 = np.asarray(Q0, float)
    return np.where(Q <= Q0 + tol, 0, np.where(Q >= 1.0 - tol, 2, 1))


def solve(C, g="对数型", Lctx=2048, Q0=Q0_MAIN, p=None, eta=ETA, nQ=41, levels=3,
          m=None, loss_scale=1.0):
    """对一组预算求最优配置。份额分母是实际总消耗，并另给预算使用率。"""
    if p is None:
        p = PAR
    if m is None:
        m = M_STAR
    C = np.atleast_1d(np.asarray(C, float))[:, None]
    Q0 = np.asarray(Q0, float)
    Q0 = np.broadcast_to(Q0[:, None] if Q0.ndim == 1 else Q0, C.shape)
    p = _as_param(p)
    gf = G_FUNCS[g][0]
    g0 = gf(Q0)
    kappa = 6.0 + eta * Lctx
    lo = Q0.copy()
    hi = np.full(C.shape, 1.0)
    for _ in range(levels):
        qg = lo + (hi - lo) * np.linspace(0.0, 1.0, nQ)[None, :]
        dg = np.maximum(gf(qg) - g0, 0.0)
        lm, _ = _profile(C, qg, dg, kappa, p, m=m, loss_scale=loss_scale)
        j = np.argmin(lm, axis=1)
        step = (hi - lo) / (nQ - 1)
        qb = qg[np.arange(len(C)), j][:, None]
        lo = np.maximum(qb - step, Q0)
        hi = np.minimum(qb + step, 1.0)
    qs = qb
    dg = np.maximum(gf(qs) - g0, 0.0)
    ls, xs = _profile(C, qs, dg, kappa, p, m=m, loss_scale=loss_scale)
    cand_l = [ls]
    cand_q = [qs]
    cand_x = [xs]
    for qe in (Q0.copy(), np.full(C.shape, 1.0)):
        dge = np.maximum(gf(qe) - g0, 0.0)
        le, xe = _profile(C, qe, dge, kappa, p, m=m, loss_scale=loss_scale)
        cand_l.append(le)
        cand_q.append(qe)
        cand_x.append(xe)
        better = le < ls - 1e-13
        qs = np.where(better, qe, qs)
        ls = np.where(better, le, ls)
        xs = np.where(better, xe, xs)
        dg = np.where(better, dge, dg)
    n = 10.0 ** xs
    d = C / (kappa * n + dg)
    c_train = 6.0 * n * d
    c_attn = eta * n * d * Lctx
    c_q = d * dg
    total = c_train + c_attn + c_q
    reg = classify(qs, Q0).ravel()
    tied = np.zeros(len(C), dtype=bool)
    best_cls = classify(qs, Q0)
    for le, qe in zip(cand_l, cand_q):
        tied |= ((np.abs(le - ls) <= TIE_ATOL) & (classify(qe, Q0) != best_cls)).ravel()
    state = np.array([STATE_NAME[int(v)] for v in reg], dtype=object)
    q0_flat = Q0.ravel()
    state = np.where(q0_flat >= 1.0 - 1e-10, "NO_QUALITY_RANGE", state)
    state = np.where(tied, "TIED", state)
    bound = (xs.ravel() <= LOGN_LO + 1e-4) | (xs.ravel() >= LOGN_HI - 1e-4)
    return dict(
        C=C.ravel(), N=n.ravel(), D=d.ravel(), Q=qs.ravel(), L=ls.ravel(),
        C_train=c_train.ravel(), C_attn=c_attn.ravel(), C_quality=c_q.ravel(),
        total_cost=total.ravel(), residual=C.ravel() - total.ravel(),
        budget_use=(total / C).ravel(),
        budget_rel_resid=((C - total) / C).ravel(),
        s_train=(c_train / total).ravel(), s_attn=(c_attn / total).ravel(),
        s_Q=(c_q / total).ravel(), DN=(d / n).ravel(),
        regime=reg, state=state, tied=tied, bound_hit=bound,
        logN=xs.ravel(),
    )


def c_crit(g, Lctx, Q0, p=None, eta=ETA, which="leave", lo=15.0, hi=26.0, it=40,
           return_status=False, m=None):
    """离开 Q0 或到达 Q=1 的全局临界预算。区间内无状态变化记 NOT_IDENTIFIED。"""
    if which not in ("leave", "full"):
        raise ValueError(f"未知临界点类型: {which}")
    if p is None:
        p = PAR
    n = max([np.size(v) for v in p.values()] + [np.size(Q0), 1])
    q0 = np.broadcast_to(np.atleast_1d(np.asarray(Q0, float)), (n,))
    if np.any((q0 <= 0) | (q0 > 1)):
        raise ValueError("Q0 必须在 (0,1] 内")
    quality_range = q0 < 1.0 - 1e-10

    def _reg(x):
        return solve(10.0 ** x, g, Lctx, q0, p, eta, nQ=21, levels=3, m=m)["regime"]

    left = _reg(np.full(n, lo))
    right = _reg(np.full(n, hi))
    if which == "leave":
        already = left > 0
        reached = right > 0
    else:
        already = left == 2
        reached = right == 2
    bracketed = quality_range & ~already & reached
    not_identified = quality_range & ~already & ~reached
    a = np.full(n, lo)
    b = np.full(n, hi)
    for _ in range(it):
        if not np.any(bracketed):
            break
        mid = (a + b) / 2.0
        hit = np.zeros(n, dtype=bool)
        if np.any(bracketed):
            # 只在已括根的分量上加密；其余分量保持端点，避免把未识别写成假根。
            probe = np.where(bracketed, mid, lo)
            reg = _reg(probe)
            hit = (reg > 0) if which == "leave" else (reg == 2)
            hit = hit & bracketed
        b = np.where(hit, mid, b)
        a = np.where(bracketed & ~hit, mid, a)
    values = np.where(bracketed, (a + b) / 2.0, np.nan)
    status = np.full(n, "NOT_BRACKETED", dtype=object)
    status = np.where(not_identified, "NOT_IDENTIFIED", status)
    status = np.where(bracketed, "FOUND", status)
    status = np.where(~quality_range, "NO_QUALITY_RANGE", status)
    return (values, status) if return_status else values


def phi_ratio(C, g, Lctx, Qeval, Q0, p=None, eta=ETA, m=None):
    """固定 Q=Qeval 的规模最优解上的 Φ。Φ=1 只是局部一阶必要条件。"""
    if p is None:
        p = PAR
    if m is None:
        m = M_STAR
    kappa = 6.0 + eta * Lctx
    gf, gd = G_FUNCS[g]
    dg = float(gf(Qeval) - gf(Q0))
    if dg < -1e-8:
        raise ValueError("g(Q) < g(Q0)，不能去掉正部")
    dg = max(dg, 0.0)
    _, x = _profile(np.asarray(C, float), np.asarray(Qeval, float), dg, kappa, p, m=m)
    n = 10.0 ** np.asarray(x, float)
    d = C / (kappa * n + dg)
    tn = p["A"] * n ** -p["al"] * Qeval ** -p["kappa_N"]
    td = p["B"] * d ** -p["be"] * Qeval ** -p["kappa_D"]
    s = kappa * n + dg
    num = (p["k0"] + (m / Qeval) * (p["kappa_N"] * tn + p["kappa_D"] * td)) * s
    den = m * p["be"] * td * float(gd(Qeval))
    return float(np.reshape(num / den, ()))


def profile_Fq(C, g, Lctx, Q, Q0, p=None, eta=ETA, m=None):
    """Spec 中固定 Q 的剖面导数 F_Q。"""
    if p is None:
        p = PAR
    if m is None:
        m = M_STAR
    kappa = 6.0 + eta * Lctx
    gf, gd = G_FUNCS[g]
    dg = max(float(gf(Q) - gf(Q0)), 0.0)
    _, x = _profile(np.asarray(C, float), np.asarray(Q, float), dg, kappa, p, m=m)
    n = 10.0 ** float(np.reshape(x, ()))
    d = C / (kappa * n + dg)
    tn = p["A"] * n ** -p["al"] * Q ** -p["kappa_N"]
    td = p["B"] * d ** -p["be"] * Q ** -p["kappa_D"]
    s = kappa * n + dg
    return float(-p["k0"] - (m / Q) * (p["kappa_N"] * tn + p["kappa_D"] * td)
                 + m * p["be"] * td * float(gd(Q)) / s)


def profile_loss_at_Q(C, g, Lctx, Q, Q0, p=None, eta=ETA, m=None):
    if p is None:
        p = PAR
    if m is None:
        m = M_STAR
    kappa = 6.0 + eta * Lctx
    gf = G_FUNCS[g][0]
    dg = max(float(gf(Q) - gf(Q0)), 0.0)
    lv, _ = _profile(np.asarray(C, float), np.asarray(Q, float), dg, kappa, p, m=m)
    return float(np.reshape(lv, ()))
