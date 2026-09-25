# -*- coding: utf-8 -*-
"""
q2_01_fit.py — 问题二·A：广义标度律拟合（质量核心式；配比通道为探索性）
=====================================================================================
数据边界（题面：本问直接使用附件 B；附件 A 只通过问题一输出引入，不重复读取原始文件）：
    直接读取：B1 B2 B3 B4 B5 B6/B7 B8 B9 B10
    问题一接口：P1_f_p_coefficients（clr+Huber 系数）、P1_p_star、P1_Q_domain、P1_summary、
                cache/step6_scale.npz（三尺度系数漂移 a_hat/rho_hat，问题一 Step6 产出）
模型族（Q=1、p=p* 时精确退化为经典式）：
    L_core(N,D,Q) = E + A·N^-α·h_N(Q) + B·D^-β·h_D(Q) + k0·(1−Q)
    h_X(Q)：lin 1+κ(1−Q)（主）/ pow Q^-κ / exp e^{κ(1−Q)}；k0(1−Q) 为可选加性地板
    探索性 m(p) = exp( s(N)·[ f̄(p) − f̄(p*) ] )，s(N) 使用 A6–A11 重拟合漂移，不能称已验证四变量主律
嵌套关系：M0 经典 ⊂ {M1 κN=0, M2 κD=0, M3 κN=κD} ⊂ M4 ⊂ M4+floor（主）；旧 M-add = M4+floor|κD=0
步骤：
    Step-1 基线：B1 网格 α,β + LS；核对 C_FLOPs 与 6ND 的比值
    Step-2 质量项：B7（含 B6 全部 360 点）固定基线拟合；CV / 分块 CV / 12B 外推 / 嵌套 F 检验；B8 方向核查
    Step-3 配比通道：用问题一 Step6 的三尺度系数漂移 a_hat、rho_hat 标定 s(N)（不读附件 A）
    Step-4 联合 bootstrap（B1×B7）
    Step-5 验证体系：B2（族外）、B3（插值轨迹）、B4（跨族，含与 B1 的近邻关系）、B5（文献）、B9/B10（百亿以上外推）
    Step-6 Q 口径映射（主：Q_B=q̄；副：Q_B=q̄/q̄(p*)）与接口输出
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code_q1"))
from q1_00_common import ROOT as _ROOT, setup_cjk_matplotlib, r2_score, rmse, spearman, clr

ROOT = _ROOT if os.path.isdir(_ROOT) else os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ATT = os.path.join(os.path.dirname(ROOT), "附件")
if not os.path.isdir(ATT):
    raise FileNotFoundError(f"赛题附件目录不存在: {ATT}")
B = f"{ATT}/B_scaling_laws"
IF1 = f"{ROOT}/outputs_q1/interface"
CACHE1 = f"{ROOT}/outputs_q1/cache"
OUT = f"{ROOT}/outputs_q2"
RNG = np.random.default_rng(2026)
try:
    from scipy.stats import f as _fdist
except Exception:
    _fdist = None


def _plt():
    plt = setup_cjk_matplotlib()
    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK JP", "sans-serif"]
    return plt


# ---------------- Step-1 基线 ----------------
def fit_baseline(N, D, L, a_grid, b_grid):
    best = None
    for a in a_grid:
        for b in b_grid:
            X = np.column_stack([np.ones(len(L)), N**-a, D**-b])
            beta, *_ = np.linalg.lstsq(X, L, rcond=None)
            sse = ((L - X @ beta)**2).sum()
            if best is None or sse < best[0]:
                best = (sse, a, b, beta)
    return best


# ---------------- 模型族 ----------------
def hq(Q, k, shape):
    if shape == "lin":
        return 1 + k*(1-Q)
    if shape == "pow":
        return Q**(-k)
    return np.exp(k*(1-Q))


def terms(N, D, Q, base, kN, kD, shape="lin"):
    E, A, al, B_, be = base
    return A*N**-al*hq(Q, kN, shape), B_*D**-be*hq(Q, kD, shape)


def predict(N, D, Q, base, kN, kD, k0=0.0, shape="lin"):
    tN, tD = terms(N, D, Q, base, kN, kD, shape)
    return base[0] + tN + tD + k0*(1-Q)


SPECS = {
    "M0 经典(κN=κD=0)":          dict(dim=0, map=lambda th: (0.0, 0.0)),
    "M1 有效数据(κN=0)":         dict(dim=1, map=lambda th: (0.0, th[0])),
    "M2 有效参数(κD=0)":         dict(dim=1, map=lambda th: (th[0], 0.0)),
    "M3 整体乘子(κN=κD)":        dict(dim=1, map=lambda th: (th[0], th[0])),
    "M4 完全耦合(κN,κD)":        dict(dim=2, map=lambda th: (th[0], th[1])),
    "M4+floor 耦合+地板[主]":     dict(dim=2, map=lambda th: (th[0], th[1]), floor=True),
    "M4-pow 幂形状 Q^-κ":        dict(dim=2, map=lambda th: (th[0], th[1]), shape="pow"),
    "M4-pow+floor":              dict(dim=2, map=lambda th: (th[0], th[1]), shape="pow", floor=True),
    "M4-exp 指数形状 e^{κ(1-Q)}": dict(dim=2, map=lambda th: (th[0], th[1]), shape="exp"),
}
MAIN = "M4+floor 耦合+地板[主]"
LEGACY = "M-add 旧主模型(1-Q)(k0+k1N^-α)"
K_LO, K_HI = -0.5, 3.0


def _sse_given_kappa(N, D, Q, L, base, kN, kD, floor, shape):
    tN, tD = terms(N, D, Q, base, kN, kD, shape)
    r = L - base[0] - tN - tD
    if floor:
        x = 1 - Q
        k0 = float((x*r).sum()/(x*x).sum())
        r = r - k0*x
    else:
        k0 = 0.0
    return float((r*r).sum()), k0


def fit_spec(N, D, Q, L, base, spec, step=0.05, refine=2):
    dim, fmap = spec["dim"], spec["map"]
    floor, shape = spec.get("floor", False), spec.get("shape", "lin")
    if dim == 0:
        sse, k0 = _sse_given_kappa(N, D, Q, L, base, 0.0, 0.0, floor, shape)
        return dict(kN=0.0, kD=0.0, k0=k0, sse=sse)
    lo = np.full(dim, K_LO); hi = np.full(dim, K_HI); st = step
    best = None
    for lvl in range(refine + 1):
        axes = [np.arange(lo[i], hi[i] + 1e-12, st) for i in range(dim)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, dim)
        for th in grid:
            kN, kD = fmap(th)
            sse, k0 = _sse_given_kappa(N, D, Q, L, base, kN, kD, floor, shape)
            if best is None or sse < best[0]:
                best = (sse, th.copy(), kN, kD, k0)
        lo = np.maximum(best[1] - st, K_LO); hi = np.minimum(best[1] + st, K_HI); st /= 10
    return dict(kN=best[2], kD=best[3], k0=best[4], sse=best[0])


def spec_predict(N, D, Q, base, fit, spec):
    return predict(N, D, Q, base, fit["kN"], fit["kD"], fit["k0"], spec.get("shape", "lin"))


def fit_madd(N, D, Q, L, base):
    E, A, al, B_, be = base
    L0 = E + A*N**-al + B_*D**-be
    X = np.column_stack([(1-Q), (1-Q)*N**-al])
    k, *_ = np.linalg.lstsq(X, L - L0, rcond=None)
    return k
def madd_predict(N, D, Q, base, k):
    E, A, al, B_, be = base
    return E + A*N**-al + B_*D**-be + (1-Q)*(k[0] + k[1]*N**-al)


def cv_folds_random(n, k=5, rng=RNG):
    idx = rng.permutation(n)
    return [(np.setdiff1d(idx, idx[i::k]), idx[i::k]) for i in range(k)]
def cv_folds_by_group(g):
    return [(np.where(g != u)[0], np.where(g == u)[0]) for u in np.unique(g)]


# =====================================================================
if __name__ == "__main__":
    for sub in ("tables", "figures", "interface"):
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)
    pd.set_option("display.width", 220)

    # ================= Step-1 基线（B1） =================
    b1 = pd.read_csv(f"{B}/pythia_training_log_existing.csv").dropna(subset=["val_loss"])
    N1, D1, L1 = b1.N_params_B.values*1e9, b1.D_tokens_B.values*1e9, b1.val_loss.values
    ratio = b1.C_FLOPs_1e21.values*1e21/(6*N1*D1)
    sse, al, be, (E0, A0, B0) = fit_baseline(
        N1, D1, L1, np.arange(0.30, 0.381, 0.002), np.arange(0.24, 0.321, 0.002))
    base = (E0, A0, al, B0, be)
    L0_1 = E0 + A0*N1**-al + B0*D1**-be
    print(f"[基线 B1] E={E0:.4f} A={A0:.1f} α={al:.3f} B={B0:.1f} β={be:.3f} "
          f"RMSE={np.sqrt(sse/len(L1)):.6f} R²={r2_score(L1, L0_1):.6f}; C/(6ND) 中位比 {np.median(ratio):.4f}")
    pd.DataFrame([dict(n=len(L1), E=E0, A=A0, alpha=al, B=B0, beta=be, rmse=np.sqrt(sse/len(L1)),
                       r2=r2_score(L1, L0_1), C_over_6ND_median=np.median(ratio),
                       C_over_6ND_q25=np.quantile(ratio, .25), C_over_6ND_q75=np.quantile(ratio, .75),
                       hoffmann2022_consts="E=1.69,A=406.4,α=0.34,B=410.7,β=0.28",
                       rmse_vs_hoffmann=float(np.sqrt(np.mean((L1-(1.69+406.4*N1**-0.34+410.7*D1**-0.28))**2))),
                       note="B1 与 Hoffmann 2022 常数吻合到 1e-4，判定为公式生成而非真实日志")]
                 ).to_csv(f"{OUT}/tables/T20_baseline_fit.csv", index=False, encoding="utf-8-sig")

    # ================= Step-2 嵌套族（B7 ⊇ B6） =================
    nq = pd.read_csv(f"{B}/supplementary_NQ_experiment_expanded.csv")
    N, D, Q, L = (nq.N_params_B.values*1e9, nq.D_tokens_B.values*1e9,
                  nq.Q_score.values, nq.val_loss.values)
    n = len(L)
    folds_rand = cv_folds_random(n)
    folds_N = cv_folds_by_group(nq.N_params_B.values)
    big = nq.N_params_B.values >= 11.0
    rows, fits = [], {}
    for name, spec in SPECS.items():
        ft = fit_spec(N, D, Q, L, base, spec)
        fits[name] = ft
        k = spec["dim"] + (1 if spec.get("floor") else 0)
        pred = spec_predict(N, D, Q, base, ft, spec)
        def _cv(folds):
            e = []
            for tr, te in folds:
                f_ = fit_spec(N[tr], D[tr], Q[tr], L[tr], base, spec, refine=1)
                e.append(rmse(L[te], spec_predict(N[te], D[te], Q[te], base, f_, spec)))
            return float(np.mean(e))
        f_ex = fit_spec(N[~big], D[~big], Q[~big], L[~big], base, spec, refine=1)
        rows.append(dict(form=name, shape=spec.get("shape", "lin"), k=k, kN=ft["kN"], kD=ft["kD"], k0=ft["k0"],
                         in_rmse=rmse(L, pred), sse=ft["sse"],
                         aic=n*np.log(ft["sse"]/n) + 2*k, bic=n*np.log(ft["sse"]/n) + k*np.log(n),
                         cv_rmse_random=_cv(folds_rand), cv_rmse_leaveN=_cv(folds_N),
                         rmse_extrap_12B=rmse(L[big], spec_predict(N[big], D[big], Q[big], base, f_ex, spec))))
    kadd = fit_madd(N, D, Q, L, base)
    pred = madd_predict(N, D, Q, base, kadd)
    sse_add = float(((L-pred)**2).sum())
    def _cv_add(folds):
        return float(np.mean([rmse(L[te], madd_predict(N[te], D[te], Q[te], base,
                                                       fit_madd(N[tr], D[tr], Q[tr], L[tr], base)))
                              for tr, te in folds]))
    rows.append(dict(form=LEGACY, shape="lin", k=2, kN=kadd[1]/A0, kD=0.0, k0=kadd[0],
                     in_rmse=rmse(L, pred), sse=sse_add,
                     aic=n*np.log(sse_add/n) + 4, bic=n*np.log(sse_add/n) + 2*np.log(n),
                     cv_rmse_random=_cv_add(folds_rand), cv_rmse_leaveN=_cv_add(folds_N),
                     rmse_extrap_12B=rmse(L[big], madd_predict(N[big], D[big], Q[big], base,
                                                               fit_madd(N[~big], D[~big], Q[~big], L[~big], base)))))
    ftab = pd.DataFrame(rows)
    ftab.to_csv(f"{OUT}/tables/T21_form_selection.csv", index=False, encoding="utf-8-sig")
    print("\n[T21 形式裁决]（噪声 σ≈0.05 为不可约下界）")
    print(ftab.round(4).to_string(index=False))

    tab_i = ftab.set_index("form")
    def ftest(r_name, f_name):
        sr, sf = tab_i.loc[r_name, ["sse", "k"]], tab_i.loc[f_name, ["sse", "k"]]
        df1, df2 = int(sf.k - sr.k), int(n - sf.k)
        F = ((sr.sse - sf.sse)/df1) / (sf.sse/df2)
        p = float(1 - _fdist.cdf(F, df1, df2)) if _fdist is not None else np.nan
        return dict(restricted=r_name, full=f_name, F=float(F), df1=df1, df2=df2, p_value=p)
    nm = list(SPECS)
    pairs = [(nm[0], nm[1]), (nm[0], nm[2]), (nm[0], nm[3]), (nm[1], nm[4]), (nm[2], nm[4]),
             (nm[3], nm[4]), (nm[4], MAIN), (LEGACY, MAIN)]
    ntab = pd.DataFrame([ftest(a, b) for a, b in pairs])
    ntab.to_csv(f"{OUT}/tables/T21b_nested_tests.csv", index=False, encoding="utf-8-sig")
    print("\n[T21b 嵌套 F 检验]"); print(ntab.round(6).to_string(index=False))

    kN, kD, k0 = fits[MAIN]["kN"], fits[MAIN]["kD"], fits[MAIN]["k0"]
    print(f"\n[主模型] L = E + A N^-α [1+{kN:.3f}(1-Q)] + B D^-β [1+{kD:.3f}(1-Q)] + {k0:.3f}(1-Q)")

    # 无模型诊断 + 形状诊断（同前）
    cell = nq.groupby(["N_params_B", "D_tokens_B"])
    diag = []
    for (n_, d_), s in cell:
        s = s.set_index("Q_score")
        diag.append(dict(N=n_, D=d_, dL_obs=s.val_loss[0.1] - s.val_loss[1.0]))
    diag = pd.DataFrame(diag)
    Nn, Dd = diag.N.values*1e9, diag.D.values*1e9
    diag["dL_main"] = predict(Nn, Dd, 0.1, base, kN, kD, k0) - predict(Nn, Dd, 1.0, base, kN, kD, k0)
    fp = fits["M4-pow 幂形状 Q^-κ"]
    diag["dL_pow"] = (predict(Nn, Dd, 0.1, base, fp["kN"], fp["kD"], 0.0, "pow")
                      - predict(Nn, Dd, 1.0, base, fp["kN"], fp["kD"], 0.0, "pow"))
    diag.to_csv(f"{OUT}/tables/T21c_deltaL_diagnostic.csv", index=False, encoding="utf-8-sig")
    dq = pd.DataFrame(dict(Q=Q, d=L - predict(N, D, Q, base, 0, 0))).groupby("Q").d.mean()
    lin_r2 = r2_score(dq.values, np.polyval(np.polyfit(1-dq.index.values, dq.values, 1), 1-dq.index.values))
    Xq = np.column_stack([np.ones(n), 1-Q, (1-Q)*N**-al, (1-Q)*D**-be, (1-Q)**2])
    yq = L - predict(N, D, Q, base, 0, 0)
    bq, *_ = np.linalg.lstsq(Xq, yq, rcond=None)
    s2 = ((yq - Xq@bq)**2).sum()/(n - Xq.shape[1]); se_q = np.sqrt(s2*np.linalg.inv(Xq.T@Xq)[-1, -1])
    print(f"[无模型诊断] ΔL(Q0.1→1) 范围 [{diag.dL_obs.min():.3f},{diag.dL_obs.max():.3f}]，"
          f"corr(ΔL,lnN)={np.corrcoef(diag.dL_obs, np.log(diag.N))[0,1]:.3f}，corr(ΔL,lnD)={np.corrcoef(diag.dL_obs, np.log(diag.D))[0,1]:.3f}")
    print(f"[形状诊断] 分档 Δ(Q) 对 (1-Q) 线性 R²={lin_r2:.4f}；二次项系数 {bq[-1]:+.4f}（se {se_q:.4f}, t={bq[-1]/se_q:.2f}）")
    # Q=1 切片退化检验（H3）
    m1 = Q == 1.0
    r1 = L[m1] - predict(N[m1], D[m1], 1.0, base, 0, 0)
    bsm = [RNG.choice(r1, len(r1)).mean() for _ in range(4000)]
    print(f"[H3 退化检验] Q=1 切片 n={m1.sum()} 均值残差 {r1.mean():+.4f} 95%CI [{np.quantile(bsm,.025):+.4f},{np.quantile(bsm,.975):+.4f}] RMSE {np.sqrt((r1**2).mean()):.4f}")
    pd.DataFrame([dict(test="H2 分档线性R2", value=lin_r2), dict(test="H2 二次项系数", value=bq[-1]),
                  dict(test="H2 二次项se", value=se_q), dict(test="H3 Q=1均值残差", value=r1.mean()),
                  dict(test="H3 CI_lo", value=np.quantile(bsm, .025)), dict(test="H3 CI_hi", value=np.quantile(bsm, .975)),
                  dict(test="H3 RMSE", value=np.sqrt((r1**2).mean()))]
                 ).to_csv(f"{OUT}/tables/T21d_shape_degeneration_tests.csv", index=False, encoding="utf-8-sig")

    # B8 方向核查（不进入拟合）
    b8 = pd.read_csv(f"{B}/supplementary_NQ_experiment_large.csv")
    g8 = b8.groupby("Q_score").val_loss.mean()
    m8 = b8.merge(nq, on=["N_params_B", "D_tokens_B", "Q_score"], suffixes=("_8", "_7"))
    b8row = dict(n=len(b8), mean_L_at_Qmin=float(g8.iloc[0]), mean_L_at_Q1=float(g8.loc[1.0]),
                 frac_below_E=float((b8.val_loss < E0).mean()),
                 corr_with_B7_same_points=float(np.corrcoef(m8.val_loss_8, m8.val_loss_7)[0, 1]) if len(m8) > 2 else np.nan,
                 verdict="Loss 随 Q 上升且大量低于 E，与 B6/B7 语义相反，不用于拟合")
    pd.DataFrame([b8row]).to_csv(f"{OUT}/tables/T21e_B8_check.csv", index=False, encoding="utf-8-sig")
    print(f"[B8 核查] Q_min 均值 {b8row['mean_L_at_Qmin']:.3f} vs Q=1 均值 {b8row['mean_L_at_Q1']:.3f}；低于 E 的比例 {b8row['frac_below_E']:.2f}；与 B7 同点相关 {b8row['corr_with_B7_same_points']:.3f} ⇒ 排除")

    # ================= Step-3 配比通道 s(N)（仅用问题一产出） =================
    coef = pd.read_csv(f"{IF1}/P1_f_p_coefficients.csv", index_col=0)
    ps = pd.read_csv(f"{IF1}/P1_p_star.csv")
    dom17 = [c.replace("beta_clr_", "") for c in coef.columns if c.startswith("beta_clr_")]
    assert dom17 == ps.domain.tolist(), "P1 接口域顺序不一致"
    Bc = coef[["intercept"] + [f"beta_clr_{d}" for d in dom17]].to_numpy()      # (13,18)
    def fbar(P):
        Z = clr(P); return (Bc[:, 0][None, :] + Z @ Bc[:, 1:].T).mean(1)
    p_star = ps.p_star_eqweight.values
    f_star = float(fbar(p_star[None, :])[0])
    st6 = np.load(f"{CACHE1}/step6_scale.npz")
    a_hat, rho_hat = st6["a_hat"], st6["rho_hat"]                                  # (13,18)，B(N)=a_hat+rho_hat·ln(N/1e6)
    # 一致性核对：漂移模型在 1M 处的截距漂移应与问题一表 T6_coef_drift 一致
    t6 = pd.read_csv(f"{ROOT}/outputs_q1/tables/T6_coef_drift.csv")
    assert np.allclose(t6.rho_intercept.values, rho_hat[:, 0], atol=1e-6), "step6 缓存与 T6_coef_drift 不一致"
    beta1 = Bc[:, 1:]                                                              # 1M 主系数
    prow = []
    for tag, Nval in [("1M", 1e6), ("60M", 6e7), ("1B", 1e9)]:
        BN = a_hat + rho_hat*np.log(Nval/1e6)
        betaN = BN[:, 1:]
        s_v = (betaN*beta1).sum(1)/(beta1**2).sum(1)                                # 逐域最小二乘尺度
        w_v = (beta1**2).sum(1)
        s_val = float((s_v*w_v).sum()/w_v.sum())
        cosv = float(np.mean([(betaN[v]@beta1[v])/np.linalg.norm(betaN[v])/np.linalg.norm(beta1[v]) for v in range(13)]))
        Lbar = float(np.exp(BN[:, 0]).mean())                                       # clr=0（均匀配比）处 13 域均值损失
        prow.append(dict(scale=tag, N=Nval, s=s_val, s_min=float(s_v.min()), s_max=float(s_v.max()),
                         cos_beta_vs_1M=cosv, Lbar_uniform=Lbar))
    ptab = pd.DataFrame(prow)
    s1M = float(ptab.loc[0, "s"])
    ptab["pred_if_total"] = s1M
    ptab["pred_if_reducible"] = s1M*((ptab.Lbar_uniform-E0)/ptab.Lbar_uniform)/((ptab.loc[0, "Lbar_uniform"]-E0)/ptab.loc[0, "Lbar_uniform"])
    s_lin = np.polyfit(np.log10(ptab.N/1e6), ptab.s, 1)     # [s1, s0]；由漂移模型的线性性，三点严格共线
    ptab["s_fit"] = np.polyval(s_lin, np.log10(ptab.N/1e6))
    ptab.to_csv(f"{OUT}/tables/T28_p_channel.csv", index=False, encoding="utf-8-sig")
    rank = pd.read_csv(f"{IF1}/P1_rank_decay.csv")
    print("\n[T28 配比通道 s(N)]（来源：问题一 Step6 三尺度重拟合系数的漂移模型，未读附件 A）")
    print(ptab.round(4).to_string(index=False))
    print(f"  s(N) = {s_lin[1]:.3f} + ({s_lin[0]:.3f})·log10(N/1e6)，截止为 0；s(10B)={np.polyval(s_lin,4):.3f}, s(100B)={max(np.polyval(s_lin,5),0):.3f}")
    print(f"  问题一排名衰减（线性模型直接迁移）Spearman: {dict(zip(rank.scale, rank.spearman.round(3)))}")

    # ================= Step-4 联合 bootstrap =================
    boots = []
    spec_main = SPECS[MAIN]
    for _ in range(300):
        i1 = RNG.integers(0, len(L1), len(L1))
        _, a_, b_, (E_, A_, B_) = fit_baseline(N1[i1], D1[i1], L1[i1], [al], [be])
        i2 = RNG.integers(0, n, n)
        f_ = fit_spec(N[i2], D[i2], Q[i2], L[i2], (E_, A_, a_, B_, b_), spec_main, refine=1)
        boots.append([E_, A_, B_, f_["kN"], f_["kD"], f_["k0"]])
    bt = pd.DataFrame(boots, columns=["E", "A", "B", "kappa_N", "kappa_D", "k0"])
    ci = bt.quantile([.025, .5, .975]).T
    ci.columns = ["CI_lo", "median", "CI_hi"]
    ci.loc["alpha"] = [al, al, al]; ci.loc["beta"] = [be, be, be]
    ci.to_csv(f"{OUT}/tables/T22_params_ci.csv", encoding="utf-8-sig")
    print("\n[T22 参数 95% CI]"); print(ci.round(4).to_string())
    print(f"  P(κN>κD)={(bt.kappa_N > bt.kappa_D).mean():.3f}; P(κD>0)={(bt.kappa_D > 0).mean():.3f}; P(k0>0)={(bt.k0 > 0).mean():.3f}")

    # ================= Step-5 验证体系 =================
    rows = []
    def check(name, nature, use, dfN, dfD, dfL, extra=None):
        pred = predict(dfN, dfD, 1.0, base, kN, kD, k0)
        r = dict(source=name, nature=nature, use=use, n=len(dfL), corr=float(np.corrcoef(dfL, pred)[0, 1]),
                 spearman=spearman(dfL, pred), bias_mean=float((dfL-pred).mean()), rmse=rmse(dfL, pred))
        if extra: r.update(extra)
        rows.append(r)
    # B2 族外（半合成）
    cer = pd.read_csv(f"{B}/cerebras_training_log.csv").dropna(subset=["val_loss"])
    check("B2 cerebras", "半合成（按 Pythia 标度律校准+噪声）", "族外趋势检验，仅比趋势与残差符号",
          cer.N_params_B.values*1e9, cer.D_tokens_B.values*1e9, cer.val_loss.values)
    # B3 插值轨迹
    tr = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{B}/training_trajectories/*.csv"))])
    check("B3 trajectories(8×500)", "插值（由 B1 检查点插值生成）", "轨迹形状检验；与 B1 同源，不独立",
          tr.N_params_B.values*1e9, tr.D_tokens_B.values*1e9, tr.val_loss.values)
    # B4 跨族 + 与 B1 的近邻关系
    sb = pd.read_csv(f"{B}/scaling_baseline.csv")
    lb1 = np.column_stack([np.log10(N1), np.log10(D1)])
    dmin = np.array([np.sqrt(((lb1 - np.array([np.log10(nn*1e9), np.log10(dd*1e9)]))**2).sum(1)).min()
                     for nn, dd in zip(sb.N_params_B, sb.D_tokens_B)])
    sb["dist_to_B1_logND"] = dmin
    nn_tab = sb.groupby("family").agg(n=("val_loss", "size"), min_dist=("dist_to_B1_logND", "min"),
                                      median_dist=("dist_to_B1_logND", "median")).reset_index()
    nn_tab["near_duplicate_of_B1"] = nn_tab.min_dist < 0.05
    nn_tab.to_csv(f"{OUT}/tables/T23b_B4_neighbor_to_B1.csv", index=False, encoding="utf-8-sig")
    far = sb[sb.dist_to_B1_logND >= 0.05]
    check("B4 scaling_baseline(全部57)", "标注真实（12 族收敛点）", "跨族趋势检验",
          sb.N_params_B.values*1e9, sb.D_tokens_B.values*1e9, sb.val_loss.values,
          dict(n_near_duplicate_B1=int((sb.dist_to_B1_logND < 0.05).sum())))
    check("B4 去掉 B1 近邻后", "标注真实", "盲测口径（剔除与 B1 (N,D) 距离<0.05 dex 的点）",
          far.N_params_B.values*1e9, far.D_tokens_B.values*1e9, far.val_loss.values)
    # B5 文献
    pub = pd.read_csv(f"{B}/published_scaling_data.csv")
    check("B5 published(44)", "真实（文献整理）", "文献趋势检验；分词器/验证集不同，只比趋势",
          pub.N_params_B.values*1e9, pub.D_tokens_B.values*1e9, pub.val_loss.values)
    # B9/B10 百亿以上
    b9 = pd.read_csv(f"{B}/supplementary_large_models.csv")
    b10 = pd.read_csv(f"{B}/supplementary_large_baseline.csv")
    check("B10 large_baseline(估算)", "估算（由已拟合标度律估算）", "百亿以上外推对照；非观测",
          b10.N_params_B.values*1e9, b10.D_tokens_B.values*1e9, b10.val_loss.values)
    cross = pd.DataFrame(rows)
    cross.to_csv(f"{OUT}/tables/T23_cross_source.csv", index=False, encoding="utf-8-sig")
    print("\n[T23 验证体系]"); print(cross.drop(columns=["use"]).round(4).to_string(index=False))
    print("[T23b B4 与 B1 近邻]"); print(nn_tab.round(3).to_string(index=False))
    # B9 元数据 → 百亿以上外推（Q 取主口径 Q0 与 1 两档）
    b9v = b9.dropna(subset=["N_params_B", "D_tokens_B"]).copy()
    b9v = b9v[(b9v.N_params_B > 0) & (b9v.D_tokens_B > 0)]          # 剔除 D=0 等无效元数据行
    m910 = b9v.merge(b10[["family", "val_loss"]], left_on="model_name", right_on="family", how="left")
    summ = json.load(open(f"{IF1}/P1_summary.json", encoding="utf-8"))
    Q0_main = float(summ["Qbar_pstar_eqweight"])
    for q_, tag in [(1.0, "Q1"), (Q0_main, "Q0main")]:
        m910[f"L_pred_{tag}"] = predict(m910.N_params_B.values*1e9, m910.D_tokens_B.values*1e9, q_, base, kN, kD, k0)
    m910["quality_gap_Q0_vs_1"] = m910.L_pred_Q0main - m910.L_pred_Q1
    m910["reducible_at_Q1"] = m910.L_pred_Q1 - E0
    m910[["model_name", "N_params_B", "D_tokens_B", "publication_date", "accessibility", "val_loss",
          "L_pred_Q1", "L_pred_Q0main", "quality_gap_Q0_vs_1", "reducible_at_Q1"]].rename(
        columns={"val_loss": "B10_estimated_loss"}).to_csv(f"{OUT}/tables/T23c_B9_B10_extrapolation.csv", index=False, encoding="utf-8-sig")
    print(f"[B9/B10] 有有效 (N,D) 的大模型 {len(b9v)}/{len(b9)}；B10 估算值与本模型 Q=1 预测的 RMSE={rmse(m910.val_loss.dropna().values, m910.loc[m910.val_loss.notna(), 'L_pred_Q1'].values):.4f}（B10 本身即公式估算）；"
          f"Q0={Q0_main:.3f} 相对 Q=1 的质量缺口 中位 {m910.quality_gap_Q0_vs_1.median():.3f}，"
          f"占 Q=1 可约损失的 {np.median(m910.quality_gap_Q0_vs_1/m910.reducible_at_Q1)*100:.0f}%")
    # 可信度边界表
    cred = pd.DataFrame([
        dict(file="B1", nature="标注真实；实为 Hoffmann 2022 公式生成（RMSE 1.6e-4）", role="基线 E,A,α,B,β 拟合", n=len(b1)),
        dict(file="B2", nature="半合成", role="族外趋势检验", n=len(cer)),
        dict(file="B3", nature="插值（B1 派生）", role="轨迹检验，不独立", n=len(tr)),
        dict(file="B4", nature="标注真实；Pythia 族与 B1 (N,D) 近邻", role="跨族趋势检验；盲测须剔除近邻", n=len(sb)),
        dict(file="B5", nature="真实文献点；分词器不同", role="文献趋势检验", n=len(pub)),
        dict(file="B6/B7", nature="半合成（σ≈0.05 噪声）", role="κN,κD,k0 唯一来源；不得表述为观测", n=n),
        dict(file="B8", nature="半合成；Loss 随 Q 上升且低于 E", role="排除，仅作反例", n=len(b8)),
        dict(file="B9", nature="真实元数据（Epoch AI）", role="百亿以上 (N,D) 输入", n=len(b9)),
        dict(file="B10", nature="估算（公式）", role="百亿以上对照，非观测", n=len(b10)),
        dict(file="B11/B12", nature="辅助元数据", role="未进入拟合", n=np.nan)])
    cred.to_csv(f"{OUT}/tables/T23d_credibility_boundary.csv", index=False, encoding="utf-8-sig")

    # ================= Step-6 Q 口径映射与接口 =================
    qd = pd.read_csv(f"{IF1}/P1_Q_domain.csv")
    q_alt_ref = Q0_main                                             # 副口径：当前推荐配比语料 ⇔ Q_B=1
    qmap = pd.DataFrame(dict(domain=qd.mixture_domain, q_final=qd.Q_final, q_tokenw=qd.Q_tokenw,
                             Q_B_main=qd.Q_final, Q_B_alt=qd.Q_final/q_alt_ref,
                             mapping_type=qd.mapping_type, CI_lo=qd.CI_lo, CI_hi=qd.CI_hi))
    qmap.to_csv(f"{OUT}/tables/T28b_Q_mapping_domains.csv", index=False, encoding="utf-8-sig")
    print(f"\n[Q 口径] 17 域 q ∈ [{qd.Q_final.min():.3f},{qd.Q_final.max():.3f}]；q̄(p*)={Q0_main:.4f}；"
          f"主口径 Q_B(p*)={Q0_main:.4f}；副口径 Q_B(p*)=1，副口径下 arxiv={qd.Q_final.max()/q_alt_ref:.3f}")

    # 图 F22（同前，第三幅改为 s(N) 来源图）
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    for d_, c in zip(sorted(diag.D.unique()), ["#08519c", "#3182bd", "#6baed6", "#fd8d3c", "#a63603"]):
        s = diag[diag.D == d_].sort_values("N")
        ax.semilogx(s.N*1e9, s.dL_obs, "o", color=c, ms=5, label=f"观测 D={d_:.0f}B")
        ax.semilogx(s.N*1e9, s.dL_main, "-", color=c, lw=1.6); ax.semilogx(s.N*1e9, s.dL_pow, ":", color=c, lw=1.2)
    ax.set_xlabel("参数量 N"); ax.set_ylabel("ΔL = L(Q=0.1) − L(Q=1)")
    ax.set_title("质量收益随规模衰减：实线 主模型(lin+floor)，虚线 幂形状"); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3)
    ax = axes[1]
    ax.scatter(bt.kappa_N, bt.kappa_D, s=6, alpha=.4, color="#3182bd", label="bootstrap")
    ax.plot([kN], [kD], "r*", ms=12, label=f"主模型: κN={kN:.2f}, κD={kD:.2f}")
    lim = max(bt.kappa_N.max(), bt.kappa_D.max())*1.1
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="M3: κN=κD")
    ax.axvline(0, color="grey", lw=1, ls=":", label="M1: κN=0"); ax.axhline(0, color="grey", lw=1, ls="-.", label="M2/M-add: κD=0")
    ax.set_xlabel("κN"); ax.set_ylabel("κD"); ax.set_title("嵌套族参数的 bootstrap 分布"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax = axes[2]
    xs = np.log10(ptab.N/1e6)
    ax.errorbar(xs, ptab.s, yerr=[ptab.s-ptab.s_min, ptab.s_max-ptab.s], fmt="ko", capsize=4, label="s(N)（逐域范围）")
    xx = np.linspace(0, 5, 50)
    ax.plot(xx, np.maximum(np.polyval(s_lin, xx), 0), "g-", lw=2, label="s(N) 线性漂移（问题一 Step6）")
    ax.plot(xx, np.full_like(xx, s1M), "b--", label="若 p 乘总损失（恒定）")
    ax.plot(xs, ptab.pred_if_reducible, "r:", marker="s", label="若 p 乘可约损失")
    ax.set_xticks([0, 1.78, 3, 4, 5]); ax.set_xticklabels(["1M", "60M", "1B", "10B", "100B"])
    ax.set_xlabel("参数量 N"); ax.set_ylabel("配比效应强度 s"); ax.set_title("配比通道：s(N) 由问题一三尺度系数漂移标定")
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/F22_form_selection.png"); plt.close(fig)

    params = dict(
        E=E0, A=A0, alpha=al, B=B0, beta=be,
        kappa_N=kN, kappa_D=kD, k0=k0, q_shape="lin",
        form="L_core = E + A·N^-α·[1+κN(1−Q)] + B·D^-β·[1+κD(1−Q)] + k0·(1−Q); p=p*",
        family="嵌套族 M0–M4(+floor)，形状 lin/pow/exp；主模型由 CV/F 检验选出，见 T21/T21b",
        p_channel="exploratory_total_log_additive",
        p_channel_evidence="s(N) 含 A6–A11 重拟合数据；不满足留出集仅评估要求，不是已验证的四变量主律",
        s_p=dict(s0=float(s_lin[1]), s1_per_decade=float(s_lin[0]), rule="s(N)=max(s0 + s1·log10(N/1e6), 0)",
                 s_at=dict(zip(ptab.scale, ptab.s.round(4).tolist())),
                 source="问题一 Step6 三尺度重拟合系数漂移（cache/step6_scale.npz），未读附件 A"),
        fbar_pstar=f_star,
        fbar_def="13 验证域等权 mean_v[b_v + β_v·clr(p)]，系数见 outputs_q1/interface/P1_f_p_coefficients.csv",
        Q_anchor=dict(
            general="Q_B = 1 − b·(q* − q̄(p))；仿射形式在聚合可加假设下可证，q*、b 不可由数据识别",
            main="q*=1, b=1 ⇒ Q_B = q̄(p)（题面建议：完美教材 ⇔ 经典式；Q0 取自附件 A 评分；Q∈(0,1]）",
            alt=f"q*=q̄(p*)={q_alt_ref:.4f}, b=1/q* ⇒ Q_B = q̄(p)/{q_alt_ref:.4f}（当前推荐配比语料 ⇔ 经典式，敏感性口径）",
            note="κN、κD、k0 在 B 自身的 Q 网格上估计，与锚点无关"),
        Q0_main=Q0_main, Q0_alt=1.0,
        q_mapping=dict(
            main=dict(name="identity", formula="Q_B = Q_A", parameters={},
                      support=[0.0, 1.0], clip_policy="error",
                      evidence_level="题面语义锚点假设", source="doc/Q2/问题二_SPEC.md"),
            alt=dict(name="pstar_anchor", formula="Q_B = Q_A / Qbar_pstar",
                     parameters=dict(Qbar_pstar=Q0_main), support=[0.0, 1.0],
                     clip_policy="error", evidence_level="敏感性假设；部分领域超出 B 数据支持域",
                     source="doc/Q2/问题二_SPEC.md")),
        q_domain_range=[float(qd.Q_final.min()), float(qd.Q_final.max())],
        noise_sigma=float(tab_i.loc[MAIN, "in_rmse"]),
        legacy_additive_form=dict(form="L = L0 + (1-Q)(k0 + k1·N^-α)", k0=float(kadd[0]), k1=float(kadd[1]),
                                  note="旧主模型 = 新主模型在 κD=0 处的特例（κN=k1/A）"))
    with open(f"{OUT}/interface/P2_scaling_law.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    np.savez(f"{OUT}/interface/P2_bootstrap.npz", samples=bt.to_numpy(), cols=np.array(bt.columns.tolist()))
    print("\nQ2-A done.")
