# -*- coding: utf-8 -*-
"""
q3_00_model.py — 问题三公共模型：广义标度律 + 三项成本 + 向量化最优化求解器

主目标：
    L = E + k0(1-Q) + [A N^{-α} Q^{-κ_N} + B D^{-β} Q^{-κ_D}] exp(s·φ̃(p*))
地板在括号外，不乘配比因子。p* 固定，s·φ̃(p*) 读问题二接口。
    s.t.  6ND + D[g(Q)-g(Q0)]₊ + η N D L_ctx ≤ C ,   Q0 ≤ Q ≤ 1

降维：三项成本都含 D，且 L 对 D 严格递减，约束取等
    D(N,Q) = C / [K N + Δg(Q)] ,   K = 6 + η L_ctx ,  Δg = g(Q) - g(Q0)
Q<Q0 不省钱却抬高 Loss，可行域取 Q∈[Q0,1]（建模假设）。
内层对 log N 黄金分割；外层对 Q 做三级网格加密，并与两端点比较。
"""
import os, sys, json, csv, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code_q1"))
from q1_00_common import ROOT, setup_cjk_matplotlib  # noqa: E402,F401

OUT = f"{ROOT}/outputs_q3"
TAB, FIG, IFACE = f"{OUT}/tables", f"{OUT}/figures", f"{OUT}/interface"
for _d in (TAB, FIG, IFACE):
    os.makedirs(_d, exist_ok=True)

P2_PATH = f"{ROOT}/outputs_q2/interface/P2_scaling_law.json"
P1_PATH = f"{ROOT}/outputs_q1/interface/P1_summary.json"
P2 = json.load(open(P2_PATH, encoding="utf-8"))
P1 = json.load(open(P1_PATH, encoding="utf-8"))
if P2.get("q_shape") != "pow" or P2.get("selected_model") != "M4+F" or not P2.get("floor"):
    raise ValueError("Q3 主式要求 P2 为幂次 M4+地板")
if P2.get("q_mapping", {}).get("main", {}).get("name") != "identity":
    raise ValueError("P2 主质量映射尚未冻结为 identity，不能运行 Q3")
_ch = P2["p_channel"]
M_STAR = float(np.exp(_ch["s_red_main"] * _ch["phi_tilde_pstar"]))
H_TOT_SCALE = float(np.exp(_ch["s_train"] * _ch["phi_tilde_pstar"]))
PAR = dict(E=P2["E"], A=P2["A"], al=P2["alpha"], B=P2["B"], be=P2["beta"],
           kappa_N=P2["kappa_N"], kappa_D=P2["kappa_D"], k0=P2["k0"],
           m=M_STAR, scale=1.0)
_nf = P2["nofloor_sensitivity"]
PAR_NOFLOOR = dict(PAR, kappa_N=_nf["kappa_N"], kappa_D=_nf["kappa_D"], k0=0.0)
PAR_M1 = dict(PAR, m=1.0)
PAR_HTOT = dict(PAR_M1, scale=H_TOT_SCALE)

ETA = 2e-4                                   # 赛题给定
LCTX_CRIT = 6 / ETA                          # 30000，解析临界值
# 主口径：Q_B = q̄(p)，Q0 = q̄(p*)；与 P1 接口逐值核对。
Q0_MAIN = float(P2["Q0_main"])
if not np.isclose(Q0_MAIN, float(P1["Qbar_pstar_eqweight"]), atol=1e-10):
    raise ValueError("P1 与 P2 的主质量基线不一致")
# 同一质量映射下的网页域对照，不把它误称为另一种 Q_B 映射。
with open(f"{ROOT}/outputs_q1/interface/P1_Q_domain.csv", encoding="utf-8-sig") as _f:
    _web = [r for r in csv.DictReader(_f) if r["mixture_domain"] == "pile_cc"]
if len(_web) != 1:
    raise ValueError("P1 领域质量中必须恰有一个 pile_cc")
Q0_WEB = float(_web[0]["Q_final"])

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
_c7_candidates = [
    Path(ROOT).parent / "附件" / "C_efficiency_evolution" / "model_architecture_metadata.csv",
    Path(ROOT).parent / "real_attachments" / "C_efficiency_evolution" / "model_architecture_metadata.csv",
    Path(ROOT).parent.parent / "real_attachments" / "C_efficiency_evolution" / "model_architecture_metadata.csv",
]
_c7_path = next((p for p in _c7_candidates if p.is_file()), _c7_candidates[0])
with _c7_path.open(encoding="utf-8-sig") as _f:
    LCTX_C7 = sorted({int(float(r["max_position_embeddings"])) for r in csv.DictReader(_f)
                     if r.get("max_position_embeddings")})
if LCTX_C7 != [2048, 4096, 8192, 32768, 131072]:
    raise ValueError(f"C7 上下文档位与 Spec 不一致: {LCTX_C7}")


def hN(Q, p=PAR):
    return np.power(Q, -p["kappa_N"])


def hD(Q, p=PAR):
    return np.power(Q, -p["kappa_D"])


def loss(N, D, Q, p=PAR):
    red = (p["A"] * hN(Q, p) * np.power(N, -p["al"])
           + p["B"] * hD(Q, p) * np.power(D, -p["be"]))
    return (p["E"] + p["k0"] * (1.0 - Q) + p["m"] * red) * p["scale"]


def N_star_closed(C, Q, kappa, p=PAR):
    """固定 Q 且 Δg=0 时的闭式 N*。m 与地板对固定 Q 的规模导数抵消。"""
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
    c_train = (6 * N * D).ravel()
    c_attn = (eta * N * D * Lctx).ravel()
    c_q = (D * dg).ravel()
    c_total = c_train + c_attn + c_q
    Cv = C.ravel()
    r = dict(C=Cv, N=N.ravel(), D=D.ravel(), Q=Qs.ravel(), L=Ls.ravel(),
             c_train=c_train, c_attn=c_attn, c_Q=c_q, c_total=c_total,
             budget_use=c_total / Cv, budget_resid=(Cv - c_total) / Cv,
             s_train=c_train / c_total, s_attn=c_attn / c_total, s_Q=c_q / c_total,
             at_bound=((np.abs(xs - 1.0) < 1e-3) | (np.abs(xs - 17.0) < 1e-3)).ravel())
    r["DN"] = r["D"] / r["N"]
    r["regime"] = classify(r["Q"], Q0.ravel())
    return r


def c_crit(g, Lctx, Q0, p=PAR, eta=ETA, which="leave", lo=15.0, hi=26.0, it=40,
           return_status=False):
    """二分法求临界预算 log10 C_crit（向量化：p 的各参数可为长度 n 的数组）
    which="leave": 离开 Q0 角点（开始提质）的最小预算；
    which="full" : 进入 Q=1 角点（提满）的最小预算。
    无提质区间或搜索端点未括根时返回 NaN；可选返回逐项状态。"""
    if which not in ("leave", "full"):
        raise ValueError(f"未知临界点类型: {which}")
    n = max([np.size(v) for v in p.values()] + [np.size(Q0), 1])
    q0 = np.broadcast_to(np.atleast_1d(np.asarray(Q0, float)), (n,))
    if np.any((q0 <= 0) | (q0 > 1)):
        raise ValueError("Q0 必须在 (0,1] 内")
    quality_range = q0 < 1 - 1e-10

    def reached(x):
        r = solve(10.0 ** x, g, Lctx, q0, p, eta, nQ=21, levels=3)
        return r["regime"] > 0 if which == "leave" else r["regime"] == 2

    a = np.full(n, lo); b = np.full(n, hi)
    bracketed = quality_range & ~reached(a) & reached(b)
    for _ in range(it):
        m = (a + b) / 2
        hit = reached(m)
        b = np.where(hit, m, b); a = np.where(hit, a, m)
    values = np.where(bracketed, (a + b) / 2, np.nan)
    status = np.where(~quality_range, "NO_QUALITY_RANGE",
                      np.where(bracketed, "FOUND", "NOT_BRACKETED"))
    return (values, status) if return_status else values


def classify(Q, Q0, tol=1e-4):
    """KKT 活跃约束集：0 = Q=Q0 角点（不提质）；1 = 内点；2 = Q=1 角点（提满）"""
    Q = np.asarray(Q)
    return np.where(Q <= Q0 + tol, 0, np.where(Q >= 1 - tol, 2, 1))


REGIME_NAME = {0: "Q0角点(不提质)", 1: "内点(部分提质)", 2: "Q=1角点(提满)"}


def profile_derivatives(C, g, Lctx, Qeval, Q0, p=PAR, eta=ETA):
    """固定 Q=Qeval 并优化规模后的一阶比。
    Φ≤1 是停在该质量的局部必要条件：分子是再提质能降低的损失，分母是为此挤占数据量带来的损失。
    幂次下质量导数带 m/Q·(κ_N T_N+κ_D T_D)，并保留地板 k0。"""
    kappa = 6 + eta * Lctx
    gf, gd = G_FUNCS[g]
    dg = max(float(gf(Qeval) - gf(Q0)), 0.0)
    _, x = _profile(np.asarray(C, float), np.asarray(Qeval, float), dg, kappa, p)
    N = 10.0 ** x
    D = C / (kappa * N + dg)
    tn = p["A"] * hN(Qeval, p) * N ** -p["al"]
    td = p["B"] * hD(Qeval, p) * D ** -p["be"]
    benefit = p["k0"] + (p["m"] / Qeval) * (p["kappa_N"] * tn + p["kappa_D"] * td)
    S = float(kappa * N + dg)
    marginal_cost = float(p["m"] * p["be"] * td * gd(Qeval) / S)
    return float(benefit * S / (p["m"] * p["be"] * td * gd(Qeval))), float(-benefit + marginal_cost)


def phi_ratio(C, g, Lctx, Qeval, Q0, p=PAR, eta=ETA):
    """Φ，见 profile_derivatives。"""
    return profile_derivatives(C, g, Lctx, Qeval, Q0, p, eta)[0]


def profile_loss(C, Q, g, Lctx, Q0, p=PAR, eta=ETA):
    """只优化 N，Q 固定。用于用有限差分核对 Φ。"""
    kappa = 6 + eta * Lctx
    gf = G_FUNCS[g][0]
    dg = max(float(gf(Q) - gf(Q0)), 0.0)
    val, _ = _profile(np.asarray(C, float), np.asarray(Q, float), dg, kappa, p)
    return float(val)


def write_manifest(output_names):
    """记录本轮输入指纹和已写出的接口文件。"""
    import subprocess
    from datetime import datetime, timezone, timedelta
    repo = Path(ROOT).parent

    def sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except Exception:
        commit = None
    now = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    inputs = {
        "P1_summary": P1_PATH,
        "P2_scaling_law": P2_PATH,
        "P2_bootstrap": f"{ROOT}/outputs_q2/interface/P2_bootstrap.npz",
        "P1_Q_domain": f"{ROOT}/outputs_q1/interface/P1_Q_domain.csv",
        "C7": str(_c7_path),
    }
    payload = dict(
        time_beijing=now, git_commit=commit,
        q_shape="pow", selected_model="M4+F", floor=True,
        Q0_main=Q0_MAIN, m=M_STAR, h_tot_scale=H_TOT_SCALE,
        m_source="exp(s_red·φ̃(p*))，只乘随规模下降的两项；地板 k0(1-Q) 在括号外",
        search_log10N=[1.0, 17.0], share_denominator="实际总消耗",
        evidence="B6 半合成幂次主式；配比通道为探索性；高预算为外推",
        inputs={k: sha(v) for k, v in inputs.items() if os.path.isfile(v)},
        outputs={name: sha(f"{IFACE}/{name}") for name in output_names
                 if os.path.isfile(f"{IFACE}/{name}")},
    )
    with open(f"{IFACE}/P3_run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
