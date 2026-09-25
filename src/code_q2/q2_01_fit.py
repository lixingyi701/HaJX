# -*- coding: utf-8 -*-
"""
q2_01_fit.py — 问题二：广义标度律拟合
数据边界：只读附件 B；附件 A 只通过问题一接口进入，不读原始配方文件。
主式（Q=1 且 p=p_B 时精确退化为 Chinchilla）：
    L = E + [A N^{-α} Q^{-κ_N} + B D^{-β} Q^{-κ_D}] · exp(s(N) · φ̃(p))
嵌套族（幂次形状，由检验集式的留一 N 交叉验证与 F 检验裁决）：
    M0 κ_N=κ_D=0 ⊂ {M1 κ_N=0, M2 κ_D=0, M3 κ_N=κ_D} ⊂ M4
线性乘子与加性地板只作对照，不进入主族：地板使不可约损失依赖 Q，与「Q=1 退化为经典式」冲突。
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code_q1"))
from q1_00_common import ROOT as _ROOT, setup_cjk_matplotlib, r2_score, rmse, spearman, clr

ROOT = _ROOT if os.path.isdir(_ROOT) else os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_parent = os.path.dirname(ROOT)
ATT = os.path.join(_parent, "附件")
if not os.path.isdir(os.path.join(ATT, "B_scaling_laws")):
    ATT = os.path.join(_parent, "real_attachments")
if not os.path.isdir(os.path.join(ATT, "B_scaling_laws")):
    raise FileNotFoundError(f"找不到 B_scaling_laws，已查: {_parent}/附件 与 real_attachments")
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


def predict_pow(N, D, Q, base, kN, kD):
    E, A, al, B_, be = base
    Q = np.clip(Q, 1e-3, None)
    return E + A * N**-al * Q**-kN + B_ * D**-be * Q**-kD


def predict_lin_floor(N, D, Q, base, kN, kD, k0):
    E, A, al, B_, be = base
    return E + A*N**-al*(1+kN*(1-Q)) + B_*D**-be*(1+kD*(1-Q)) + k0*(1-Q)


# 主族：幂次。spec: (dim, map th -> (kN,kD))
FAMILY = {
    "M0 经典":            (0, lambda th: (0.0, 0.0)),
    "M1 有效数据(κN=0)":  (1, lambda th: (np.zeros_like(th, dtype=float), np.asarray(th, float))),
    "M2 有效参数(κD=0)":  (1, lambda th: (np.asarray(th, float), np.zeros_like(th, dtype=float))),
    "M3 整体乘子(κN=κD)": (1, lambda th: (np.asarray(th, float), np.asarray(th, float))),
    "M4 完全耦合":        (2, lambda th: (th[0], th[1])),
}
K_LO, K_HI = -0.05, 1.20


def _best_on_grid(N, D, Q, L, base, dim, fmap, lo, hi, step):
    E, A, al, B_, be = base
    Q = np.clip(np.asarray(Q, float), 1e-3, None)
    lnQ = np.log(Q)
    tN0 = A * np.asarray(N, float)**-al
    tD0 = B_ * np.asarray(D, float)**-be
    y = np.asarray(L, float) - E
    if dim == 0:
        r = y - tN0 - tD0
        return 0.0, 0.0, float((r*r).sum())
    axes = [np.arange(lo[i], hi[i] + 1e-12, step) for i in range(dim)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, dim)
    if dim == 1:
        kN, kD = fmap(grid[:, 0])
        kN, kD = np.asarray(kN, float), np.asarray(kD, float)
    else:
        kN, kD = grid[:, 0], grid[:, 1]
    # pred = tN0 * exp(-kN lnQ) + tD0 * exp(-kD lnQ)
    TN = tN0[:, None] * np.exp(-lnQ[:, None] * kN[None, :])
    TD = tD0[:, None] * np.exp(-lnQ[:, None] * kD[None, :])
    sse = ((y[:, None] - TN - TD)**2).sum(0)
    j = int(np.argmin(sse))
    return float(kN[j]), float(kD[j]), float(sse[j])


def fit_family(N, D, Q, L, base, name, step=0.02, refine=2):
    dim, fmap = FAMILY[name]
    lo = np.full(dim, K_LO); hi = np.full(dim, K_HI); st = step
    if dim == 0:
        kN, kD, sse = _best_on_grid(N, D, Q, L, base, 0, fmap, lo, hi, st)
        return dict(kN=kN, kD=kD, sse=sse)
    best = None
    for _ in range(refine + 1):
        kN, kD, sse = _best_on_grid(N, D, Q, L, base, dim, fmap, lo, hi, st)
        th = np.array([kN] if dim == 1 and name.startswith("M1") else
                      ([kD] if dim == 1 and name.startswith("M2") else
                       ([kN] if dim == 1 else [kN, kD])))
        # 用 fmap 的逆：M1 的自由参数是 kD，M2 是 kN，M3 是公共 κ
        if name.startswith("M1"):
            th = np.array([kD])
        elif name.startswith("M3"):
            th = np.array([kN])
        elif dim == 1:
            th = np.array([kN])
        else:
            th = np.array([kN, kD])
        best = (sse, th, kN, kD)
        lo = np.maximum(th - st, K_LO); hi = np.minimum(th + st, K_HI); st /= 5
    return dict(kN=best[2], kD=best[3], sse=best[0])


def cv_folds_random(n, k=5, rng=RNG):
    idx = rng.permutation(n)
    return [(np.setdiff1d(idx, idx[i::k]), idx[i::k]) for i in range(k)]


def cv_folds_by_group(g):
    return [(np.where(g != u)[0], np.where(g == u)[0]) for u in np.unique(g)]


def fit_lin_floor(N, D, Q, L, base):
    """对照：旧线性乘子 + 地板，闭式给定 (κN,κD) 时的 k0，κ 仍网格。"""
    E, A, al, B_, be = base
    tN0, tD0 = A*N**-al, B_*D**-be
    best = None
    for kN in np.arange(0, 1.201, 0.02):
        for kD in np.arange(0, 1.201, 0.02):
            r = L - E - tN0*(1+kN*(1-Q)) - tD0*(1+kD*(1-Q))
            x = 1 - Q
            k0 = float((x*r).sum() / (x*x).sum())
            rr = r - k0*x
            sse = float((rr*rr).sum())
            if best is None or sse < best[0]:
                best = (sse, kN, kD, k0)
    return dict(kN=best[1], kD=best[2], k0=best[3], sse=best[0])


if __name__ == "__main__":
    for sub in ("tables", "figures", "interface"):
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    # ================= Step-1 基线（B1） =================
    b1 = pd.read_csv(f"{B}/pythia_training_log_existing.csv").dropna(subset=["val_loss"])
    N1 = b1.N_params_B.values * 1e9
    D1 = b1.D_tokens_B.values * 1e9
    L1 = b1.val_loss.values
    ratio = b1.C_FLOPs_1e21.values * 1e21 / (6 * N1 * D1)
    sse, al, be, (E0, A0, B0) = fit_baseline(
        N1, D1, L1, np.arange(0.30, 0.381, 0.002), np.arange(0.24, 0.321, 0.002))
    base = (float(E0), float(A0), float(al), float(B0), float(be))
    E0, A0, B0 = base[0], base[1], base[3]
    Lhat1 = predict_pow(N1, D1, np.ones_like(L1), base, 0, 0)
    hoff = 1.69 + 406.4 * N1**-0.34 + 410.7 * D1**-0.28
    print(f"[基线 B1] E={E0:.4f} A={A0:.1f} α={al:.3f} B={B0:.1f} β={be:.3f} "
          f"RMSE={rmse(L1, Lhat1):.6f} R²={r2_score(L1, Lhat1):.6f}; "
          f"C/(6ND) 中位比 {np.median(ratio):.4f}; 对 Hoffmann 常数 RMSE {rmse(L1, hoff):.6f}")
    pd.DataFrame([dict(
        n=len(L1), E=E0, A=A0, alpha=al, B=B0, beta=be,
        rmse=rmse(L1, Lhat1), r2=r2_score(L1, Lhat1),
        C_over_6ND_median=float(np.median(ratio)),
        C_over_6ND_q25=float(np.quantile(ratio, .25)),
        C_over_6ND_q75=float(np.quantile(ratio, .75)),
        rmse_vs_hoffmann=float(rmse(L1, hoff)),
        note="B1 的 val_loss 与 Hoffmann 2022 常数吻合到 1e-4，按公式生成数据使用，不称为独立训练观测")
    ]).to_csv(f"{OUT}/tables/T20_baseline_fit.csv", index=False, encoding="utf-8-sig")

    # ================= Step-2 幂次嵌套族（B7，含 B6） =================
    b6 = pd.read_csv(f"{B}/supplementary_NQ_experiment.csv")
    nq = pd.read_csv(f"{B}/supplementary_NQ_experiment_expanded.csv")
    keys = ["N_params_B", "D_tokens_B", "Q_score"]
    merged = b6.merge(nq, on=keys, how="left", indicator=True)
    n_hit = int((merged["_merge"] == "both").sum())
    print(f"[B6⊂B7] B6 {len(b6)} 点中 {n_hit} 个 (N,D,Q) 出现在 B7；拟合用 B7，B6 不重复估计")
    N = nq.N_params_B.values * 1e9
    D = nq.D_tokens_B.values * 1e9
    Q = nq.Q_score.values
    L = nq.val_loss.values
    n = len(L)
    folds_N = cv_folds_by_group(nq.N_params_B.values)
    big = nq.N_params_B.values >= 11.0

    rows, fits = [], {}
    for name in FAMILY:
        ft = fit_family(N, D, Q, L, base, name)
        fits[name] = ft
        k = FAMILY[name][0]
        pred = predict_pow(N, D, Q, base, ft["kN"], ft["kD"])
        cv = []
        for tr, te in folds_N:
            f_ = fit_family(N[tr], D[tr], Q[tr], L[tr], base, name, step=0.04, refine=1)
            cv.append(rmse(L[te], predict_pow(N[te], D[te], Q[te], base, f_["kN"], f_["kD"])))
        f_ex = fit_family(N[~big], D[~big], Q[~big], L[~big], base, name, step=0.04, refine=1)
        rows.append(dict(
            form=name, family="power", k=k, kN=ft["kN"], kD=ft["kD"], k0=0.0,
            in_rmse=rmse(L, pred), sse=ft["sse"],
            aic=n*np.log(ft["sse"]/n) + 2*k, bic=n*np.log(ft["sse"]/n) + k*np.log(n),
            cv_rmse_leaveN=float(np.mean(cv)),
            rmse_extrap_12B=rmse(L[big], predict_pow(N[big], D[big], Q[big], base, f_ex["kN"], f_ex["kD"]))))
        print(f"  {name}: κN={ft['kN']:.3f} κD={ft['kD']:.3f} RMSE={rows[-1]['in_rmse']:.4f} "
              f"CVleaveN={rows[-1]['cv_rmse_leaveN']:.4f}")

    sens = fit_lin_floor(N, D, Q, L, base)
    pred_s = predict_lin_floor(N, D, Q, base, sens["kN"], sens["kD"], sens["k0"])
    rows.append(dict(
        form="对照 线性乘子+地板", family="sensitivity", k=3,
        kN=sens["kN"], kD=sens["kD"], k0=sens["k0"],
        in_rmse=rmse(L, pred_s), sse=sens["sse"],
        aic=n*np.log(sens["sse"]/n) + 6, bic=n*np.log(sens["sse"]/n) + 3*np.log(n),
        cv_rmse_leaveN=np.nan, rmse_extrap_12B=np.nan))
    ftab = pd.DataFrame(rows)
    ftab.to_csv(f"{OUT}/tables/T21_form_selection.csv", index=False, encoding="utf-8-sig")
    print("\n[T21 形式]")
    print(ftab.round(4).to_string(index=False))

    power = ftab[ftab.family == "power"].set_index("form")
    def ftest(r_name, f_name):
        sr, sf = power.loc[r_name], power.loc[f_name]
        df1 = int(sf.k - sr.k)
        df2 = int(n - sf.k)
        F = ((sr.sse - sf.sse) / df1) / (sf.sse / df2)
        p = float(1 - _fdist.cdf(F, df1, df2)) if _fdist is not None else np.nan
        return dict(restricted=r_name, full=f_name, F=float(F), df1=df1, df2=df2, p_value=p)
    pairs = [("M0 经典", "M1 有效数据(κN=0)"), ("M0 经典", "M2 有效参数(κD=0)"),
             ("M0 经典", "M3 整体乘子(κN=κD)"), ("M1 有效数据(κN=0)", "M4 完全耦合"),
             ("M2 有效参数(κD=0)", "M4 完全耦合"), ("M3 整体乘子(κN=κD)", "M4 完全耦合")]
    ntab = pd.DataFrame([ftest(a, b_) for a, b_ in pairs])
    ntab.to_csv(f"{OUT}/tables/T21b_nested_tests.csv", index=False, encoding="utf-8-sig")
    print("\n[T21b 嵌套 F]"); print(ntab.round(6).to_string(index=False))

    # 裁决：leave-N CV 不差于最优 2% 的模型里，取 F 检验仍拒绝其嵌套限制的最大模型；
    # M3 对 M4 的 p<0.01 时保留两个 κ，否则收缩为 M3。
    best_cv = float(power.cv_rmse_leaveN.min())
    p_m3 = float(ntab[(ntab.restricted == "M3 整体乘子(κN=κD)") & (ntab.full == "M4 完全耦合")].p_value.iloc[0])
    m4_ok = float(power.loc["M4 完全耦合", "cv_rmse_leaveN"]) <= best_cv * 1.02
    MAIN = "M4 完全耦合" if (m4_ok and p_m3 < 0.01) else "M3 整体乘子(κN=κD)"
    kN, kD = fits[MAIN]["kN"], fits[MAIN]["kD"]
    print(f"\n[主模型 {MAIN}] L = E + [A N^-α Q^-{kN:.3f} + B D^-β Q^-{kD:.3f}] · m(p)")

    # 无模型诊断：ΔL(Q=0.1→1) 是否随 N 变
    diag = []
    for (n_, d_), s in nq.groupby(["N_params_B", "D_tokens_B"]):
        s = s.set_index("Q_score")
        if 0.1 in s.index and 1.0 in s.index:
            diag.append(dict(N=n_, D=d_, dL_obs=float(s.val_loss[0.1] - s.val_loss[1.0])))
    diag = pd.DataFrame(diag)
    Nn, Dd = diag.N.values * 1e9, diag.D.values * 1e9
    diag["dL_main"] = predict_pow(Nn, Dd, np.full(len(diag), 0.1), base, kN, kD) - predict_pow(Nn, Dd, np.ones(len(diag)), base, kN, kD)
    diag.to_csv(f"{OUT}/tables/T21c_deltaL_diagnostic.csv", index=False, encoding="utf-8-sig")
    print(f"[诊断] ΔL 范围 [{diag.dL_obs.min():.3f},{diag.dL_obs.max():.3f}] "
          f"corr(ΔL,lnN)={np.corrcoef(diag.dL_obs, np.log(diag.N))[0,1]:.3f} "
          f"corr(ΔL,lnD)={np.corrcoef(diag.dL_obs, np.log(diag.D))[0,1]:.3f}")

    m1 = np.isclose(Q, 1.0)
    r1 = L[m1] - predict_pow(N[m1], D[m1], np.ones(m1.sum()), base, 0, 0)
    bsm = [RNG.choice(r1, len(r1), replace=True).mean() for _ in range(2000)]
    pd.DataFrame([
        dict(test="H_Q1 均值残差", value=float(r1.mean())),
        dict(test="H_Q1 CI_lo", value=float(np.quantile(bsm, .025))),
        dict(test="H_Q1 CI_hi", value=float(np.quantile(bsm, .975))),
        dict(test="H_Q1 RMSE", value=float(np.sqrt((r1**2).mean()))),
        dict(test="selected_model", value=MAIN),
    ]).to_csv(f"{OUT}/tables/T21d_shape_degeneration_tests.csv", index=False, encoding="utf-8-sig")
    print(f"[H Q=1 退化] n={m1.sum()} 均值残差 {r1.mean():+.4f} "
          f"95%CI [{np.quantile(bsm,.025):+.4f},{np.quantile(bsm,.975):+.4f}]")

    b8 = pd.read_csv(f"{B}/supplementary_NQ_experiment_large.csv")
    g8 = b8.groupby("Q_score").val_loss.mean()
    b8row = dict(n=len(b8), mean_L_at_Qmin=float(g8.iloc[0]), mean_L_at_Q1=float(g8.loc[1.0]),
                 frac_below_E=float((b8.val_loss < E0).mean()),
                 verdict="Loss 随 Q 上升且大量低于 E，与 B6/B7 方向相反，不用于拟合")
    pd.DataFrame([b8row]).to_csv(f"{OUT}/tables/T21e_B8_check.csv", index=False, encoding="utf-8-sig")
    print(f"[B8] Qmin 均值 {b8row['mean_L_at_Qmin']:.3f} vs Q=1 {b8row['mean_L_at_Q1']:.3f}；"
          f"低于 E 比例 {b8row['frac_below_E']:.2f} → 排除")

    # ================= Step-3 配比通道（只读问题一） =================
    coef = pd.read_csv(f"{IF1}/P1_f_p_coefficients.csv", index_col=0)
    ps = pd.read_csv(f"{IF1}/P1_p_star.csv")
    dom17 = [c.replace("beta_clr_", "") for c in coef.columns if c.startswith("beta_clr_")]
    assert dom17 == ps.domain.tolist()
    Bc = coef[["intercept"] + [f"beta_clr_{d}" for d in dom17]].to_numpy()
    beta1 = Bc[:, 1:]

    def phi_of(P):
        """1M 上 13 域等权的 clr 偏离（均匀配比处为 0）。单位是 ln L。"""
        Z = clr(np.atleast_2d(P))
        return (Z @ beta1.T).mean(1)

    npz_path = f"{CACHE1}/step6_scale.npz"
    if os.path.isfile(npz_path):
        st6 = np.load(npz_path)
        a_hat, rho_hat = st6["a_hat"], st6["rho_hat"]
        prow = []
        for tag, Nval in [("1M", 1e6), ("60M", 6e7), ("1B", 1e9)]:
            BN = a_hat + rho_hat * np.log(Nval / 1e6)
            betaN = BN[:, 1:]
            s_v = (betaN * beta1).sum(1) / (beta1**2).sum(1)
            w_v = (beta1**2).sum(1)
            s_obs = float((s_v * w_v).sum() / w_v.sum())
            cosv = float(np.mean([
                (betaN[v] @ beta1[v]) / (np.linalg.norm(betaN[v]) * np.linalg.norm(beta1[v]) + 1e-15)
                for v in range(13)]))
            Lbar = float(np.exp(BN[:, 0]).mean())
            prow.append(dict(scale=tag, N=Nval, s_obs=s_obs, s_min=float(s_v.min()), s_max=float(s_v.max()),
                             cos_beta_vs_1M=cosv, Lbar_uniform=Lbar))
        scale_src = "问题一 cache/step6_scale.npz"
    else:
        # 问题一三尺度重拟合的汇总（系数向量尺度与均匀配比处的平均损失）。缓存缺失时用这份已核对结果，不回读附件 A。
        prow = [
            dict(scale="1M", N=1e6, s_obs=1.003128843576256, s_min=0.850774558929466, s_max=1.1057470699849468,
                 cos_beta_vs_1M=0.9937155072367896, Lbar_uniform=5.695298432824765),
            dict(scale="60M", N=6e7, s_obs=0.8660952257964334, s_min=0.6920775052294164, s_max=1.4655026303861258,
                 cos_beta_vs_1M=0.9776878598199314, Lbar_uniform=3.360245238458359),
            dict(scale="1B", N=1e9, s_obs=0.7719331827736269, s_min=0.46908965863985763, s_max=1.8879102997569628,
                 cos_beta_vs_1M=0.9356301300770795, Lbar_uniform=2.3634571545684144),
        ]
        scale_src = "问题一 Step6 汇总（step6_scale.npz 不在磁盘上；数值来自该步已输出的三尺度系数尺度）"
    ptab = pd.DataFrame(prow)
    ptab["reducible_share"] = (ptab.Lbar_uniform - E0) / ptab.Lbar_uniform
    ptab["s_red"] = ptab.s_obs / ptab.reducible_share
    # s_red(N) = s0 + s1 log10(N/1e6)，使可约损失乘子复现问题一的对数损失尺度
    s_lin = np.polyfit(np.log10(ptab.N / 1e6), ptab.s_red, 1)
    ptab["s_red_fit"] = np.polyval(s_lin, np.log10(ptab.N / 1e6))
    ptab["s_obs_if_const_red"] = ptab.s_red.iloc[0] * ptab.reducible_share / ptab.reducible_share.iloc[0]
    ptab.to_csv(f"{OUT}/tables/T28_p_channel.csv", index=False, encoding="utf-8-sig")
    phi_star = float(phi_of(ps.p_star_eqweight.values)[0])
    phi_uni = float(phi_of(ps.p_uniform.values)[0])
    print(f"\n[T28 配比] 来源：{scale_src}")
    print(ptab.round(4).to_string(index=False))
    print(f"  φ(p*)={phi_star:.4f}（相对均匀配比的 1M lnL 偏离）；φ(uniform)={phi_uni:.4f}")
    print(f"  s_red(N)={s_lin[1]:.3f}+({s_lin[0]:.3f})·log10(N/1e6)")
    print("  假设：若配比只乘可约损失且 s_red 恒定，则 s_obs 应随 reducible_share 同比下降；"
          f"1B 上 s_obs={ptab.s_obs.iloc[2]:.3f}，恒定 s_red 预言 {ptab.s_obs_if_const_red.iloc[2]:.3f}，故 s_red 保留尺度项")

    # ================= Step-4 bootstrap =================
    boots = []
    for _ in range(200):
        i1 = RNG.integers(0, len(L1), len(L1))
        _, a_, b_, (E_, A_, B_) = fit_baseline(N1[i1], D1[i1], L1[i1], [al], [be])
        i2 = RNG.integers(0, n, n)
        f_ = fit_family(N[i2], D[i2], Q[i2], L[i2], (E_, A_, a_, B_, b_), MAIN, step=0.05, refine=1)
        boots.append([E_, A_, B_, f_["kN"], f_["kD"]])
    bt = pd.DataFrame(boots, columns=["E", "A", "B", "kappa_N", "kappa_D"])
    ci = bt.quantile([.025, .5, .975]).T
    ci.columns = ["CI_lo", "median", "CI_hi"]
    ci.loc["alpha"] = [al, al, al]
    ci.loc["beta"] = [be, be, be]
    ci.to_csv(f"{OUT}/tables/T22_params_ci.csv", encoding="utf-8-sig")
    print("\n[T22]"); print(ci.round(4))
    print(f"  P(κN>κD)={(bt.kappa_N > bt.kappa_D).mean():.3f}  P(κN>0)={(bt.kappa_N > 0).mean():.3f}  P(κD>0)={(bt.kappa_D > 0).mean():.3f}")

    # ================= Step-5 验证 =================
    rows = []
    def check(name, nature, use, dfN, dfD, dfL, extra=None):
        pred = predict_pow(dfN, dfD, np.ones(len(dfL)), base, kN, kD)
        r = dict(source=name, nature=nature, use=use, n=len(dfL),
                 corr=float(np.corrcoef(dfL, pred)[0, 1]),
                 spearman=spearman(dfL, pred),
                 bias_mean=float((dfL - pred).mean()), rmse=rmse(dfL, pred))
        if extra:
            r.update(extra)
        rows.append(r)

    cer = pd.read_csv(f"{B}/cerebras_training_log.csv").dropna(subset=["val_loss"])
    check("B2 cerebras", "半合成", "族外趋势；不得称为直接观测",
          cer.N_params_B.values*1e9, cer.D_tokens_B.values*1e9, cer.val_loss.values)
    tr = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{B}/training_trajectories/*.csv"))])
    check("B3 trajectories", "插值", "轨迹形状；与 B1 同源，不独立",
          tr.N_params_B.values*1e9, tr.D_tokens_B.values*1e9, tr.val_loss.values)
    sb = pd.read_csv(f"{B}/scaling_baseline.csv")
    lb1 = np.column_stack([np.log10(N1), np.log10(D1)])
    dmin = np.array([
        np.sqrt(((lb1 - np.array([np.log10(nn*1e9), np.log10(dd*1e9)]))**2).sum(1)).min()
        for nn, dd in zip(sb.N_params_B, sb.D_tokens_B)])
    sb["dist_to_B1_logND"] = dmin
    nn_tab = sb.groupby("family").agg(n=("val_loss", "size"), min_dist=("dist_to_B1_logND", "min"),
                                       median_dist=("dist_to_B1_logND", "median")).reset_index()
    nn_tab["near_duplicate_of_B1"] = nn_tab.min_dist < 0.05
    nn_tab.to_csv(f"{OUT}/tables/T23b_B4_neighbor_to_B1.csv", index=False, encoding="utf-8-sig")
    far = sb[sb.dist_to_B1_logND >= 0.05]
    check("B4 全部", "标注真实", "跨族趋势",
          sb.N_params_B.values*1e9, sb.D_tokens_B.values*1e9, sb.val_loss.values,
          dict(n_near_duplicate_B1=int((dmin < 0.05).sum())))
    check("B4 去掉 B1 近邻", "标注真实", "盲测：剔除 log(N,D) 距离 < 0.05 dex",
          far.N_params_B.values*1e9, far.D_tokens_B.values*1e9, far.val_loss.values)
    pub = pd.read_csv(f"{B}/published_scaling_data.csv")
    check("B5 published", "文献整理", "只比较趋势，分词器与验证集不同",
          pub.N_params_B.values*1e9, pub.D_tokens_B.values*1e9, pub.val_loss.values)
    b9 = pd.read_csv(f"{B}/supplementary_large_models.csv")
    b10 = pd.read_csv(f"{B}/supplementary_large_baseline.csv")
    check("B10 估算", "估算，非观测", "百亿以上对照",
          b10.N_params_B.values*1e9, b10.D_tokens_B.values*1e9, b10.val_loss.values)
    cross = pd.DataFrame(rows)
    cross.to_csv(f"{OUT}/tables/T23_cross_source.csv", index=False, encoding="utf-8-sig")
    print("\n[T23]"); print(cross.drop(columns=["use"]).round(4).to_string(index=False))

    summ = json.load(open(f"{IF1}/P1_summary.json", encoding="utf-8"))
    Q0_main = float(summ["Qbar_pstar_eqweight"])
    b9v = b9.dropna(subset=["N_params_B", "D_tokens_B"]).copy()
    b9v = b9v[(b9v.N_params_B > 0) & (b9v.D_tokens_B > 0)]
    m910 = b9v.merge(b10[["family", "val_loss"]], left_on="model_name", right_on="family", how="left")
    for q_, tag in [(1.0, "Q1"), (Q0_main, "Q0")]:
        m910[f"L_{tag}"] = predict_pow(m910.N_params_B.values*1e9, m910.D_tokens_B.values*1e9,
                                       np.full(len(m910), q_), base, kN, kD)
    m910["gap_Q0_vs_Q1"] = m910.L_Q0 - m910.L_Q1
    cols = ["model_name", "N_params_B", "D_tokens_B", "val_loss", "L_Q1", "L_Q0", "gap_Q0_vs_Q1"]
    keep = [c for c in cols if c in m910.columns or c in ("L_Q1", "L_Q0", "gap_Q0_vs_Q1")]
    m910[keep].rename(columns={"val_loss": "B10_estimated_loss"}).to_csv(
        f"{OUT}/tables/T23c_B9_B10_extrapolation.csv", index=False, encoding="utf-8-sig")
    both = m910.dropna(subset=["val_loss"])
    print(f"[B9/B10] 有效 (N,D) {len(b9v)}/{len(b9)}；B10 对 Q=1 预测 RMSE "
          f"{rmse(both.val_loss.values, both.L_Q1.values):.4f}（B10 本身是估算）")

    cred = pd.DataFrame([
        dict(file="B1", nature="公式生成（与 Hoffmann 2022 差 1e-4）", role="E,A,α,B,β", n=len(b1)),
        dict(file="B2", nature="半合成", role="族外趋势", n=len(cer)),
        dict(file="B3", nature="由 B1 插值", role="轨迹，不独立", n=len(tr)),
        dict(file="B4", nature="标注真实；部分与 B1 近邻", role="跨族；盲测剔除近邻", n=len(sb)),
        dict(file="B5", nature="文献点", role="趋势", n=len(pub)),
        dict(file="B6", nature="半合成", role="含于 B7，不重复拟合", n=len(b6)),
        dict(file="B7", nature="半合成", role="κN,κD 的唯一拟合来源", n=n),
        dict(file="B8", nature="半合成且 Loss 随 Q 上升", role="排除", n=len(b8)),
        dict(file="B9", nature="真实元数据", role="百亿以上 (N,D)", n=len(b9)),
        dict(file="B10", nature="估算", role="对照，非观测", n=len(b10)),
        dict(file="B11/B12", nature="辅助", role="未进入拟合", n=np.nan),
    ])
    cred.to_csv(f"{OUT}/tables/T23d_credibility_boundary.csv", index=False, encoding="utf-8-sig")

    qd = pd.read_csv(f"{IF1}/P1_Q_domain.csv")
    qmap = pd.DataFrame(dict(
        domain=qd.mixture_domain, q_final=qd.Q_final,
        Q_B=qd.Q_final, mapping_type=qd.mapping_type, CI_lo=qd.CI_lo, CI_hi=qd.CI_hi))
    qmap.to_csv(f"{OUT}/tables/T28b_Q_mapping_domains.csv", index=False, encoding="utf-8-sig")

    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    for d_, c in zip(sorted(diag.D.unique()), ["#08519c", "#3182bd", "#6baed6", "#fd8d3c", "#a63603"]):
        s = diag[diag.D == d_].sort_values("N")
        ax.semilogx(s.N, s.dL_obs, "o", color=c, ms=5, label=f"D={d_:.0f}B")
        ax.semilogx(s.N, s.dL_main, "-", color=c, lw=1.5)
    ax.set_xlabel("N（十亿）"); ax.set_ylabel("L(Q=0.1)−L(Q=1)")
    ax.set_title("质量收益随 N 下降（实线=主模型）"); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3)
    ax = axes[1]
    ax.scatter(bt.kappa_N, bt.kappa_D, s=8, alpha=.35, color="#3182bd")
    ax.plot([kN], [kD], "r*", ms=14)
    ax.set_xlabel("κN"); ax.set_ylabel("κD"); ax.set_title(f"bootstrap（{MAIN}）"); ax.grid(alpha=.3)
    ax = axes[2]
    xx = np.log10(ptab.N / 1e6)
    ax.plot(xx, ptab.s_obs, "ko-", label="s_obs（ln L 系数尺度）")
    ax.plot(xx, ptab.s_obs_if_const_red, "r--", label="若 s_red 恒定")
    ax.plot(xx, ptab.s_red, "gs-", label="s_red（可约损失乘子）")
    ax.set_xticks(xx); ax.set_xticklabels(ptab.scale)
    ax.set_title("配比通道：数据不支持恒定 s_red"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/F22_form_selection.png"); plt.close(fig)

    params = dict(
        E=E0, A=A0, alpha=float(al), B=B0, beta=float(be),
        kappa_N=float(kN), kappa_D=float(kD), k0=0.0,
        q_shape="pow",
        selected_model=MAIN,
        form="L = E + [A·N^{-α}·Q^{-κ_N} + B·D^{-β}·Q^{-κ_D}] · exp(s_red(N)·φ̃(p))",
        degeneration="Q=1 且 φ̃=0（B 的隐含配比）时精确等于 E+A N^{-α}+B D^{-β}",
        family="幂次嵌套族 M0–M4；线性+地板仅为对照，见 T21/T21b",
        selection_rule="leave-N CV 不差于最优 2% 时，若 M3⊂M4 的 F 检验 p<0.01 则取 M4，否则取 M3",
        scale_source=scale_src,
        p_channel=dict(
            phi_def="φ(p)=13 域等权 mean(β_1M·clr(p))，均匀配比处为 0，单位是 1M 的 ln L",
            phi_pstar=phi_star, phi_uniform=phi_uni,
            reference="m=1 对应附件 B 的隐含配比（Pythia/The Pile），不是 p*",
            s_red_rule="s_red(N)=s0+s1·log10(N/1e6)",
            s0=float(s_lin[1]), s1_per_decade=float(s_lin[0]),
            s_at={r.scale: dict(s_obs=round(r.s_obs, 4), s_red=round(r.s_red, 4)) for r in ptab.itertuples()},
            note="s_obs 下降慢于可约份额，故不把 s_red 收成常数"),
        Q_anchor="Q_B = Q_A（问题一领域质量，已在 (0,1]）",
        Q0_main=Q0_main,
        noise_sigma=float(power.loc[MAIN, "in_rmse"]),
        b6_points_in_b7=n_hit,
        legacy_not_used=dict(kN=sens["kN"], kD=sens["kD"], k0=sens["k0"],
                             rmse=float(rmse(L, pred_s)),
                             why="加性地板使 E 依赖 Q，不满足 Q=1 退化"),
    )
    # 兼容旧字段名：s_p 供 q2_03 读取对数损失尺度（不是 s_red）
    params["s_p"] = dict(s0=float(np.polyfit(np.log10(ptab.N/1e6), ptab.s_obs, 1)[1]),
                         s1_per_decade=float(np.polyfit(np.log10(ptab.N/1e6), ptab.s_obs, 1)[0]),
                         rule="s_obs(N)=max(s0+s1·log10(N/1e6), 0)，乘在问题一的 ∂lnL/∂p 上")
    with open(f"{OUT}/interface/P2_scaling_law.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    np.savez(f"{OUT}/interface/P2_bootstrap.npz", samples=bt.to_numpy(), cols=np.array(list(bt.columns)))
    print("\nQ2-A done. 主模型:", MAIN)
