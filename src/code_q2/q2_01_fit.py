# -*- coding: utf-8 -*-
"""
q2_01_fit.py — 问题二：广义标度律拟合、检验与接口
数据边界：只读附件 B；附件 A 只通过问题一接口（outputs_q1/interface）进入。
主式族（Q=1 且 p=p_ref 时精确退化为 Chinchilla）：
    L = E + k0·(1−Q) + [A N^{-α} h(Q;κ_N) + B D^{-β} h(Q;κ_D)] · exp(s_red · φ̃(p))
    h：pow Q^{-κ} / exp e^{κ(1-Q)} / lin 1+κ(1-Q)；h(1)=1，κ≥0 时 h 单调不增
    k0(1−Q)：质量地板，k0≥0；k0=0 时 N,D→∞ 的极限为 E，k0>0 时为 E+k0(1−Q)
    φ̃(p) = φ(p) − φ(p_ref)，φ 为问题一 1M clr 系数在 13 验证域上的等权平均，p_ref = A4 训练配比均值
数据划分：B6（360 点）训练与选形；B7 中不在 B6 的 90 点只作留出评估；全 B7 只作对照。
配比强度 s_red 只用 1M 训练行定；A6–A11 的校准斜率只用于评估两个预先设定的假设。
"""
import os, sys, json, glob, re
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code_q1"))
from q1_00_common import ROOT as _ROOT, setup_cjk_matplotlib, r2_score, rmse, spearman, clr

ROOT = _ROOT if os.path.isdir(_ROOT) else os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_parent = os.path.dirname(ROOT)
ATT = next((p for p in (os.path.join(_parent, "附件"), os.path.join(_parent, "real_attachments"))
            if os.path.isdir(os.path.join(p, "B_scaling_laws"))), None)
if ATT is None:
    raise FileNotFoundError(f"找不到 B_scaling_laws：已查 {_parent}/附件 与 real_attachments")
B = f"{ATT}/B_scaling_laws"
IF1 = f"{ROOT}/outputs_q1/interface"
OUT = f"{ROOT}/outputs_q2"
RNG = np.random.default_rng(2026)
from scipy.stats import f as _fdist

SHAPES = ("pow", "exp", "lin")
MODELS = ("M0", "M1", "M2", "M3", "M4")
MODEL_NAME = {"M0": "经典", "M1": "有效数据(κN=0)", "M2": "有效参数(κD=0)",
              "M3": "整体乘子(κN=κD)", "M4": "完全耦合"}
MODEL_DIM = {"M0": 0, "M1": 1, "M2": 1, "M3": 1, "M4": 2}
REDUCE = {"M4": ("M1", "M2", "M3")}
K_HI = {"pow": 1.2, "exp": 1.2, "lin": 3.0}
PARSIMONY_P = 0.01
CV_TOL = 1.02


def h(Q, k, shape):
    Q = np.asarray(Q, float)
    k = np.asarray(k, float)
    if shape == "pow":
        return np.power(Q, -k)
    if shape == "exp":
        return np.exp(k * (1 - Q))
    return 1 + k * (1 - Q)


def predict(N, D, Q, base, kN, kD, shape, k0=0.0):
    E, A, al, B_, be = base
    return E + A * N**-al * h(Q, kN, shape) + B_ * D**-be * h(Q, kD, shape) + k0 * (1 - np.asarray(Q, float))


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


def _kappa_pairs(model, grid):
    z = np.zeros_like(grid)
    if model == "M1":
        return z, grid
    if model == "M2":
        return grid, z
    if model == "M3":
        return grid, grid
    raise ValueError(model)


def _floor_resid(R, x, floor):
    """R: (n, m) 残差；floor 时对每列求 k0=max(xᵀr/xᵀx, 0) 并扣除。"""
    if not floor:
        return R, np.zeros(R.shape[1])
    k0 = np.maximum((x @ R) / (x @ x), 0.0)
    return R - x[:, None] * k0[None, :], k0


def fit_model(N, D, Q, L, base, model, shape, floor=False, step=0.02, refine=2):
    """网格加密最小二乘；κ ∈ [0, K_HI]；地板 k0 在每个 κ 上闭式求解并截断为非负。"""
    E, A, al, B_, be = base
    tN0, tD0, y, x = A * N**-al, B_ * D**-be, L - E, 1 - Q
    if model == "M0":
        R, k0 = _floor_resid((y - tN0 - tD0)[:, None], x, floor)
        return dict(kN=0.0, kD=0.0, k0=float(k0[0]), sse=float((R[:, 0]**2).sum()))
    lo, hi, st = np.zeros(MODEL_DIM[model]), np.full(MODEL_DIM[model], K_HI[shape]), step
    best = None
    for _ in range(refine + 1):
        axes = [np.arange(lo[i], hi[i] + 1e-12, st) for i in range(len(lo))]
        if model == "M4":
            gN, gD = [a.ravel() for a in np.meshgrid(*axes, indexing="ij")]
        else:
            gN, gD = _kappa_pairs(model, axes[0])
        pred = tN0[:, None] * h(Q[:, None], gN[None, :], shape) + tD0[:, None] * h(Q[:, None], gD[None, :], shape)
        R, k0 = _floor_resid(y[:, None] - pred, x, floor)
        sse = (R**2).sum(0)
        j = int(np.argmin(sse))
        best = (float(sse[j]), float(gN[j]), float(gD[j]), float(k0[j]))
        th = np.array([best[1], best[2]]) if model == "M4" else np.array([best[2] if model == "M1" else best[1]])
        lo, hi = np.maximum(th - st, 0.0), np.minimum(th + st, K_HI[shape])
        st /= 5
    return dict(kN=best[1], kD=best[2], k0=best[3], sse=best[0])


def folds_by(g):
    return [(np.where(g != u)[0], np.where(g == u)[0]) for u in np.unique(g)]


def ftest(sse_r, k_r, sse_f, k_f, n):
    df1, df2 = k_f - k_r, n - k_f
    F = ((sse_r - sse_f) / df1) / (sse_f / df2)
    return float(F), int(df1), int(df2), float(1 - _fdist.cdf(F, df1, df2))


def model_id(m, floor):
    return f"{m}+F" if floor else m


def _plt():
    plt = setup_cjk_matplotlib()
    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
    return plt


if __name__ == "__main__":
    for sub in ("tables", "figures", "interface"):
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    # ================= Step-1 基线（B1） =================
    b1 = pd.read_csv(f"{B}/pythia_training_log_existing.csv").dropna(subset=["val_loss"])
    N1, D1, L1 = b1.N_params_B.values * 1e9, b1.D_tokens_B.values * 1e9, b1.val_loss.values
    ratio = b1.C_FLOPs_1e21.values * 1e21 / (6 * N1 * D1)
    _, al, be, (E0, A0, B0) = fit_baseline(N1, D1, L1, np.arange(0.30, 0.381, 0.002), np.arange(0.24, 0.321, 0.002))
    base = (float(E0), float(A0), float(al), float(B0), float(be))
    E0, A0, B0 = base[0], base[1], base[3]
    L1hat = predict(N1, D1, 1.0, base, 0, 0, "lin")
    hoff = 1.69 + 406.4 * N1**-0.34 + 410.7 * D1**-0.28
    print(f"[B1] E={E0:.4f} A={A0:.2f} α={al:.3f} B={B0:.2f} β={be:.3f}  RMSE={rmse(L1, L1hat):.2e} "
          f"vsHoffmann={rmse(L1, hoff):.2e}  C/(6ND) 中位 {np.median(ratio):.4f}")
    pd.DataFrame([dict(n=len(L1), E=E0, A=A0, alpha=al, B=B0, beta=be, rmse=rmse(L1, L1hat),
                       r2=r2_score(L1, L1hat), C_over_6ND_median=float(np.median(ratio)),
                       C_over_6ND_q25=float(np.quantile(ratio, .25)), C_over_6ND_q75=float(np.quantile(ratio, .75)),
                       rmse_vs_hoffmann=rmse(L1, hoff),
                       note="与 Hoffmann 2022 常数吻合到 1e-4，按公式生成数据使用")]
                 ).to_csv(f"{OUT}/tables/T20_baseline_fit.csv", index=False, encoding="utf-8-sig")

    # ================= Step-2 数据划分 =================
    keys = ["N_params_B", "D_tokens_B", "Q_score"]
    b6 = pd.read_csv(f"{B}/supplementary_NQ_experiment.csv")
    b7 = pd.read_csv(f"{B}/supplementary_NQ_experiment_expanded.csv")
    mm = b7.merge(b6[keys + ["val_loss"]], on=keys, how="left", suffixes=("", "_b6"), indicator=True)
    same = mm[mm._merge == "both"]
    assert len(same) == len(b6), "B6 不完全包含于 B7"
    assert np.allclose(same.val_loss, same.val_loss_b6), "B6 与 B7 同点 Loss 不一致"
    hold = mm[mm._merge == "left_only"][b7.columns].reset_index(drop=True)
    print(f"[划分] B6 训练 {len(b6)}；B7 新增留出 {len(hold)}，Q∈{sorted(hold.Q_score.unique())}，"
          f"覆盖 {hold.groupby(['N_params_B','D_tokens_B']).ngroups} 个 (N,D) 格")

    def arr(df):
        return df.N_params_B.values * 1e9, df.D_tokens_B.values * 1e9, df.Q_score.values, df.val_loss.values
    N, D, Q, L = arr(b6)
    Nh, Dh, Qh, Lh = arr(hold)
    Na, Da, Qa, La = arr(b7)
    n = len(L)
    fN = folds_by(b6.N_params_B.values)
    big = b6.N_params_B.values >= 11.0

    # ================= Step-3 形状 × 嵌套族 × 地板（仅 B6） =================
    rows, fits = [], {}
    for floor in (False, True):
        for shape in SHAPES:
            for m in MODELS:
                if m == "M0" and shape != "lin":
                    continue
                ft = fit_model(N, D, Q, L, base, m, shape, floor)
                fits[(shape, m, floor)] = ft
                k = MODEL_DIM[m] + int(floor)
                cv = []
                for tr, te in fN:
                    f_ = fit_model(N[tr], D[tr], Q[tr], L[tr], base, m, shape, floor, step=0.04, refine=1)
                    cv.append(rmse(L[te], predict(N[te], D[te], Q[te], base, f_["kN"], f_["kD"], shape, f_["k0"])))
                fx = fit_model(N[~big], D[~big], Q[~big], L[~big], base, m, shape, floor, step=0.04, refine=1)
                ph = predict(Nh, Dh, Qh, base, ft["kN"], ft["kD"], shape, ft["k0"])
                rows.append(dict(q_shape="—" if m == "M0" else shape, model=model_id(m, floor),
                                 name=MODEL_NAME[m] + ("+地板" if floor else ""), floor=floor, k=k,
                                 kN=ft["kN"], kD=ft["kD"], k0=ft["k0"],
                                 train_rmse=float(np.sqrt(ft["sse"] / n)), sse=ft["sse"],
                                 aic=n * np.log(ft["sse"] / n) + 2 * k, bic=n * np.log(ft["sse"] / n) + k * np.log(n),
                                 cv_rmse_leaveN=float(np.mean(cv)),
                                 rmse_extrap_12B=rmse(L[big], predict(N[big], D[big], Q[big], base, fx["kN"], fx["kD"], shape, fx["k0"])),
                                 holdout_rmse=rmse(Lh, ph), holdout_bias=float(np.mean(ph - Lh))))
    ftab = pd.DataFrame(rows)
    ftab.to_csv(f"{OUT}/tables/T21_form_selection.csv", index=False, encoding="utf-8-sig")
    print("\n[T21 形状×嵌套族×地板，B6 训练；holdout 为 B7 新增 90 点，未参与选择]")
    print(ftab.drop(columns=["sse", "floor"]).round(4).to_string(index=False))

    def row(shape, m, floor):
        s = "—" if m == "M0" else shape
        return ftab[(ftab.q_shape == s) & (ftab.model == model_id(m, floor))].iloc[0]

    trows = []
    for shape in SHAPES:
        for floor in (False, True):
            for r_, f_ in [("M0", "M1"), ("M0", "M2"), ("M0", "M3"), ("M1", "M4"), ("M2", "M4"), ("M3", "M4")]:
                a, b_ = row(shape, r_, floor), row(shape, f_, floor)
                F, d1, d2, p = ftest(a.sse, a.k, b_.sse, b_.k, n)
                trows.append(dict(q_shape=shape, restricted=model_id(r_, floor), full=model_id(f_, floor),
                                  F=F, df1=d1, df2=d2, p_value=p))
        for m in MODELS[1:]:
            a, b_ = row(shape, m, False), row(shape, m, True)
            F, d1, d2, p = ftest(a.sse, a.k, b_.sse, b_.k, n)
            trows.append(dict(q_shape=shape, restricted=m, full=model_id(m, True), F=F, df1=d1, df2=d2, p_value=p))
    ntab = pd.DataFrame(trows)
    ntab.to_csv(f"{OUT}/tables/T21b_nested_tests.csv", index=False, encoding="utf-8-sig")
    print("\n[T21b 嵌套 F 检验，B6]"); print(ntab.round(5).to_string(index=False))

    # CV 规则只作对照，不决定主式：
    # 1) 形状：各形状最优 leave-N CV 与全局最优相差不超过 2% 时优先线性，否则取 CV 最小者；
    # 2) 该形状内逐步收缩（去地板，或 M4→M1/M2/M3），F 检验 p≥0.01 且 CV≤1.02×当前则接受。
    # 主式固定为幂次 M4+地板：N、D 上的质量乘子为 Q^{-κ}，地板 k0(1-Q) 保留，两项 κ 都不收缩掉。
    body = ftab[ftab.model != "M0"]
    by_shape = body.groupby("q_shape").cv_rmse_leaveN.min()
    ok = by_shape[by_shape <= by_shape.min() * CV_TOL].index.tolist()
    SHAPE = "lin" if "lin" in ok else str(by_shape.idxmin())
    cur = body[body.q_shape == SHAPE].sort_values("cv_rmse_leaveN").iloc[0]
    cur_m, cur_f = cur.model.replace("+F", ""), bool(cur.floor)
    while True:
        cands = []
        if cur_f:
            cands.append((cur_m, False))
        for r_ in REDUCE.get(cur_m, ()):
            cands.append((r_, cur_f))
        acc = []
        now = row(SHAPE, cur_m, cur_f)
        for m_, f_ in cands:
            r = row(SHAPE, m_, f_)
            _, _, _, p = ftest(r.sse, r.k, now.sse, now.k, n)
            if p >= PARSIMONY_P and r.cv_rmse_leaveN <= now.cv_rmse_leaveN * CV_TOL:
                acc.append((r.cv_rmse_leaveN, m_, f_))
        if not acc:
            break
        _, cur_m, cur_f = min(acc)
    auto_shape, auto_m, auto_f = SHAPE, cur_m, cur_f
    print(f"[CV 规则会选出] {auto_shape}-{model_id(auto_m, auto_f)}，不作为主式")
    SHAPE, MODEL, FLOOR = "pow", "M4", True
    sel = row(SHAPE, MODEL, FLOOR)
    ft = fits[(SHAPE, MODEL, FLOOR)]
    kN, kD, k0 = ft["kN"], ft["kD"], ft["k0"]
    alt_f = not FLOOR
    alt = row(SHAPE, MODEL, alt_f)
    ft_alt = fits[(SHAPE, MODEL, alt_f)]
    print(f"\n[选择] 形状={SHAPE} 模型={model_id(MODEL, FLOOR)}：κN={kN:.4f} κD={kD:.4f} k0={k0:.4f}  "
          f"CV={sel.cv_rmse_leaveN:.4f}  留出 RMSE={sel.holdout_rmse:.4f} 偏差={sel.holdout_bias:+.4f}")
    print(f"[敏感性 {model_id(MODEL, alt_f)}] κN={ft_alt['kN']:.4f} κD={ft_alt['kD']:.4f} k0={ft_alt['k0']:.4f}  "
          f"CV={alt.cv_rmse_leaveN:.4f}  留出 RMSE={alt.holdout_rmse:.4f}")

    def P_main(N_, D_, Q_):
        return predict(N_, D_, Q_, base, kN, kD, SHAPE, k0)

    # ================= Step-4 留出评估与诊断 =================
    pred_h = P_main(Nh, Dh, Qh)
    hold = hold.assign(pred=pred_h, resid=Lh - pred_h)
    hold.to_csv(f"{OUT}/tables/T21f_holdout_points.csv", index=False, encoding="utf-8-sig")
    hsum = []
    for col in ("Q_score", "N_params_B", "D_tokens_B"):
        for v, s in hold.groupby(col):
            hsum.append(dict(by=col, value=v, n=len(s), rmse=float(np.sqrt((s.resid**2).mean())),
                             mean_resid=float(s.resid.mean())))
    hsum.append(dict(by="all", value=np.nan, n=len(hold), rmse=float(np.sqrt((hold.resid**2).mean())),
                     mean_resid=float(hold.resid.mean())))
    hsum = pd.DataFrame(hsum)
    hsum.to_csv(f"{OUT}/tables/T21g_holdout_by_group.csv", index=False, encoding="utf-8-sig")
    print("[留出 按 Q]"); print(hsum[hsum.by.isin(["Q_score", "all"])].round(4).to_string(index=False))

    # 噪声代理：同一 (N,D) 格内对 Q 做二次拟合的残差标准差（8 点、3 参数）
    cell_sd = []
    for _, s in b6.groupby(["N_params_B", "D_tokens_B"]):
        c = np.polyfit(s.Q_score, s.val_loss, 2)
        r = s.val_loss - np.polyval(c, s.Q_score)
        cell_sd.append(float(np.sqrt((r**2).sum() / max(len(s) - 3, 1))))
    noise_sigma = float(np.median(cell_sd))
    print(f"[噪声代理] {noise_sigma:.4f}")

    # 形状残差检验：在主式上再加 (1−Q)^2 项
    x = 1 - Q
    r_main = L - P_main(N, D, Q)
    cq = float((x**2 @ r_main) / (x**2 @ x**2))
    sse_q = float(((r_main - cq * x**2)**2).sum())
    Fq, _, _, pq = ftest(float(r_main @ r_main), sel.k, sse_q, sel.k + 1, n)
    print(f"[形状检验] 主式加 (1−Q)^2：系数 {cq:+.4f}，F={Fq:.2f}，p={pq:.3f}")

    fa = fit_model(Na, Da, Qa, La, base, MODEL, SHAPE, FLOOR)
    print(f"[对照 全 B7 450 点] κN={fa['kN']:.4f} κD={fa['kD']:.4f} k0={fa['k0']:.4f} RMSE={np.sqrt(fa['sse']/len(La)):.4f}")

    diag = []
    for (n_, d_), s in b6.groupby(["N_params_B", "D_tokens_B"]):
        s = s.set_index("Q_score")
        diag.append(dict(N=n_, D=d_, dL_obs=float(s.val_loss[0.1] - s.val_loss[1.0])))
    diag = pd.DataFrame(diag)
    Nn, Dd = diag.N.values * 1e9, diag.D.values * 1e9
    diag["dL_main"] = P_main(Nn, Dd, np.full(len(diag), 0.1)) - P_main(Nn, Dd, np.ones(len(diag)))
    diag.to_csv(f"{OUT}/tables/T21c_deltaL_diagnostic.csv", index=False, encoding="utf-8-sig")
    cN = float(np.corrcoef(diag.dL_obs, np.log(diag.N))[0, 1])
    cD = float(np.corrcoef(diag.dL_obs, np.log(diag.D))[0, 1])
    print(f"[ΔL 诊断] 范围 [{diag.dL_obs.min():.3f},{diag.dL_obs.max():.3f}] corr lnN {cN:.3f} lnD {cD:.3f}")

    m1 = np.isclose(Q, 1.0)
    r1 = L[m1] - predict(N[m1], D[m1], 1.0, base, 0, 0, "lin")
    bsm = [RNG.choice(r1, len(r1)).mean() for _ in range(4000)]
    ci1 = (float(np.quantile(bsm, .025)), float(np.quantile(bsm, .975)))
    print(f"[Q=1 退化] n={m1.sum()} 均值残差 {r1.mean():+.4f} 95%CI [{ci1[0]:+.4f},{ci1[1]:+.4f}]")
    pd.DataFrame([
        dict(test="selected_shape", value=SHAPE), dict(test="selected_model", value=model_id(MODEL, FLOOR)),
        dict(test="Q1_mean_resid", value=float(r1.mean())), dict(test="Q1_CI_lo", value=ci1[0]),
        dict(test="Q1_CI_hi", value=ci1[1]), dict(test="Q1_rmse", value=float(np.sqrt((r1**2).mean()))),
        dict(test="noise_sigma_proxy", value=noise_sigma),
        dict(test="quad_term_coef", value=cq), dict(test="quad_term_p", value=pq),
        dict(test="corr_dL_lnN", value=cN), dict(test="corr_dL_lnD", value=cD),
        dict(test="fullB7_kN", value=fa["kN"]), dict(test="fullB7_kD", value=fa["kD"]), dict(test="fullB7_k0", value=fa["k0"]),
    ]).to_csv(f"{OUT}/tables/T21d_shape_degeneration_tests.csv", index=False, encoding="utf-8-sig")

    b8 = pd.read_csv(f"{B}/supplementary_NQ_experiment_large.csv")
    g8 = b8.groupby("Q_score").val_loss.mean()
    m8 = b8.merge(b7, on=keys, suffixes=("_8", "_7"))
    b8row = dict(n=len(b8), mean_L_at_Qmin=float(g8.iloc[0]), mean_L_at_Q1=float(g8.loc[1.0]),
                 frac_below_E=float((b8.val_loss < E0).mean()), n_same_points_B7=len(m8),
                 corr_with_B7_same_points=float(np.corrcoef(m8.val_loss_8, m8.val_loss_7)[0, 1]) if len(m8) > 2 else np.nan,
                 verdict="Loss 随 Q 上升且大量低于 E，与 B6/B7 方向相反，不用于拟合")
    pd.DataFrame([b8row]).to_csv(f"{OUT}/tables/T21e_B8_check.csv", index=False, encoding="utf-8-sig")
    print(f"[B8] Qmin {b8row['mean_L_at_Qmin']:.3f} vs Q=1 {b8row['mean_L_at_Q1']:.3f}；低于 E {b8row['frac_below_E']:.2f}")

    # ================= Step-5 bootstrap（B1 行 × B6 (N,D) 格簇） =================
    cell_idx = list(b6.groupby(["N_params_B", "D_tokens_B"]).indices.values())
    boots = []
    for _ in range(300):
        i1 = RNG.integers(0, len(L1), len(L1))
        _, a_, b_, (E_, A_, B_) = fit_baseline(N1[i1], D1[i1], L1[i1], [al], [be])
        i2 = np.concatenate([cell_idx[j] for j in RNG.integers(0, len(cell_idx), len(cell_idx))])
        f_ = fit_model(N[i2], D[i2], Q[i2], L[i2], (E_, A_, a_, B_, b_), MODEL, SHAPE, FLOOR, step=0.04, refine=2)
        boots.append([E_, A_, B_, f_["kN"], f_["kD"], f_["k0"]])
    bt = pd.DataFrame(boots, columns=["E", "A", "B", "kappa_N", "kappa_D", "k0"])
    ci = bt.quantile([.025, .5, .975]).T
    ci.columns = ["CI_lo", "median", "CI_hi"]
    ci.loc["alpha"] = [al] * 3
    ci.loc["beta"] = [be] * 3
    ci["note"] = "B1 行重抽 × B6 (N,D) 格簇重抽，300 次；α,β 固定；未含形状与模型选择不确定性"
    ci.to_csv(f"{OUT}/tables/T22_params_ci.csv", encoding="utf-8-sig")
    print("\n[T22]"); print(ci.drop(columns="note").round(4))
    print(f"  P(κN>κD)={(bt.kappa_N > bt.kappa_D).mean():.3f}  P(k0>0)={(bt.k0 > 0).mean():.3f}")

    # ================= Step-6 验证体系（Q=1，与质量项无关） =================
    rows = []
    def check(name, nature, use, dN, dD, dL, extra=None):
        p_ = P_main(dN, dD, np.ones(len(dL)))
        r = dict(source=name, nature=nature, use=use, n=len(dL), corr=float(np.corrcoef(dL, p_)[0, 1]),
                 spearman=spearman(dL, p_), bias_mean=float((dL - p_).mean()), rmse=rmse(dL, p_))
        if extra:
            r.update(extra)
        rows.append(r)
    cer = pd.read_csv(f"{B}/cerebras_training_log.csv").dropna(subset=["val_loss"])
    check("B2 cerebras", "半合成", "族外趋势；非观测", cer.N_params_B.values*1e9, cer.D_tokens_B.values*1e9, cer.val_loss.values)
    tr = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{B}/training_trajectories/*.csv"))])
    check("B3 trajectories", "插值（B1 派生）", "轨迹形状；不独立", tr.N_params_B.values*1e9, tr.D_tokens_B.values*1e9, tr.val_loss.values)
    sb = pd.read_csv(f"{B}/scaling_baseline.csv")
    lb1 = np.column_stack([np.log10(N1), np.log10(D1)])
    dmin = np.array([np.sqrt(((lb1 - np.array([np.log10(a*1e9), np.log10(b*1e9)]))**2).sum(1)).min()
                     for a, b in zip(sb.N_params_B, sb.D_tokens_B)])
    sb["dist_to_B1_logND"] = dmin
    nn_tab = sb.groupby("family").agg(n=("val_loss", "size"), min_dist=("dist_to_B1_logND", "min"),
                                      median_dist=("dist_to_B1_logND", "median")).reset_index()
    nn_tab["near_duplicate_of_B1"] = nn_tab.min_dist < 0.05
    nn_tab.to_csv(f"{OUT}/tables/T23b_B4_neighbor_to_B1.csv", index=False, encoding="utf-8-sig")
    far = sb[sb.dist_to_B1_logND >= 0.05]
    check("B4 全部", "标注真实", "跨族趋势", sb.N_params_B.values*1e9, sb.D_tokens_B.values*1e9, sb.val_loss.values,
          dict(n_near_duplicate_B1=int((dmin < 0.05).sum())))
    check("B4 去掉 B1 近邻", "标注真实", "盲测口径", far.N_params_B.values*1e9, far.D_tokens_B.values*1e9, far.val_loss.values)
    pub = pd.read_csv(f"{B}/published_scaling_data.csv")
    check("B5 published", "文献整理", "趋势；分词器/验证集不同", pub.N_params_B.values*1e9, pub.D_tokens_B.values*1e9, pub.val_loss.values)
    b9 = pd.read_csv(f"{B}/supplementary_large_models.csv")
    b10 = pd.read_csv(f"{B}/supplementary_large_baseline.csv")
    check("B10 估算", "估算，非观测", "百亿以上对照", b10.N_params_B.values*1e9, b10.D_tokens_B.values*1e9, b10.val_loss.values)
    cross = pd.DataFrame(rows)
    cross.to_csv(f"{OUT}/tables/T23_cross_source.csv", index=False, encoding="utf-8-sig")
    print("\n[T23]"); print(cross.drop(columns=["use"]).round(4).to_string(index=False))

    summ = json.load(open(f"{IF1}/P1_summary.json", encoding="utf-8"))
    Q0_main = float(summ["Qbar_pstar_eqweight"])
    Q_ref = float(summ["Qbar_train_mean"])
    b9v = b9.dropna(subset=["N_params_B", "D_tokens_B"])
    b9v = b9v[(b9v.N_params_B > 0) & (b9v.D_tokens_B > 0)].copy()
    m910 = b9v.merge(b10[["family", "val_loss"]], left_on="model_name", right_on="family", how="left")
    for q_, tag in [(1.0, "Q1"), (Q0_main, "Q0")]:
        m910[f"L_{tag}"] = P_main(m910.N_params_B.values*1e9, m910.D_tokens_B.values*1e9, np.full(len(m910), q_))
    m910["gap_Q0_vs_Q1"] = m910.L_Q0 - m910.L_Q1
    m910["gap_share_of_reducible"] = m910.gap_Q0_vs_Q1 / (m910.L_Q1 - E0)
    m910["floor_share_of_gap"] = k0 * (1 - Q0_main) / m910.gap_Q0_vs_Q1
    keep = [c for c in ["model_name", "N_params_B", "D_tokens_B", "publication_date", "accessibility", "val_loss",
                        "L_Q1", "L_Q0", "gap_Q0_vs_Q1", "gap_share_of_reducible", "floor_share_of_gap"] if c in m910.columns]
    m910[keep].rename(columns={"val_loss": "B10_estimated_loss"}).to_csv(
        f"{OUT}/tables/T23c_B9_B10_extrapolation.csv", index=False, encoding="utf-8-sig")
    both = m910.dropna(subset=["val_loss"])
    print(f"[B9/B10] 有效 {len(b9v)}/{len(b9)}；B10 与 Q=1 预测 RMSE {rmse(both.val_loss.values, both.L_Q1.values):.4f}；"
          f"Q0 缺口中位 {m910.gap_Q0_vs_Q1.median():.3f}（Q=1 可约损失的 {m910.gap_share_of_reducible.median()*100:.1f}%，"
          f"其中地板占 {m910.floor_share_of_gap.median()*100:.0f}%）")

    # ================= Step-7 B11/B12 来源审计 =================
    b11 = pd.read_csv(f"{B}/open_model_family_metadata.csv")
    b12 = pd.read_csv(f"{B}/pythia_checkpoint_index.csv")
    def size_to_B(s):
        m_ = re.match(r"([\d.]+)([mb])", str(s).lower())
        return float(m_.group(1)) * (1e-3 if m_.group(2) == "m" else 1.0) if m_ else np.nan
    b12["size_B"] = b12.model_size.map(size_to_B)
    b12_sizes = sorted(b12.size_B.dropna().unique())
    aud = []
    for nB, s in b1.groupby("N_params_B"):
        near = min(b12_sizes, key=lambda z: abs(np.log(z) - np.log(nB)))
        steps12 = set(b12.loc[b12.size_B == near, "step"].astype(int))
        steps1 = set(s.steps.astype(int))
        tok = s.D_tokens_B.values * 1e9 / (s.steps.values * s.batch_tokens_M.values * 1e6)
        nominal = str(b12.loc[b12.size_B == near, "model_size"].iloc[0]).lower()
        aud.append(dict(N_params_B=nB, pythia_nominal_B=near, n_B1_rows=len(s),
                        n_B12_checkpoints=len(steps12), frac_B1_steps_in_B12=len(steps1 & steps12) / len(steps1),
                        D_over_steps_x_batch_median=float(np.median(tok)),
                        in_B11=bool(b11.model_repo.str.lower().str.contains(f"pythia-{nominal}$").any())))
    aud = pd.DataFrame(aud)
    fam11, fam4 = set(b11.family.str.lower()), set(sb.family.str.lower())
    aud_f = pd.DataFrame([dict(item="B4 族中出现在 B11 的比例", value=len(fam4 & fam11) / len(fam4)),
                          dict(item="B11 族", value=";".join(sorted(fam11))),
                          dict(item="B11 抓取错误行", value=int(b11["error"].notna().sum()))])
    aud.to_csv(f"{OUT}/tables/T23e_B12_checkpoint_audit.csv", index=False, encoding="utf-8-sig")
    aud_f.to_csv(f"{OUT}/tables/T23f_B11_family_audit.csv", index=False, encoding="utf-8-sig")
    print("\n[B11/B12 审计]"); print(aud.round(4).to_string(index=False)); print(aud_f.to_string(index=False))

    # ================= Step-8 配比通道（只读问题一接口） =================
    coef = pd.read_csv(f"{IF1}/P1_f_p_coefficients.csv", index_col=0)
    ps = pd.read_csv(f"{IF1}/P1_p_star.csv").set_index("domain")
    dom17 = [c.replace("beta_clr_", "") for c in coef.columns if c.startswith("beta_clr_")]
    assert set(dom17) == set(ps.index), "P1 接口域名不一致"
    beta1 = coef[[f"beta_clr_{d}" for d in dom17]].to_numpy()
    def phi(p):
        return float((clr(np.atleast_2d(p)) @ beta1.T).mean())
    phi_ref = phi(ps.loc[dom17, "p_train_mean"].to_numpy())
    phi_star = phi(ps.loc[dom17, "p_star_eqweight"].to_numpy()) - phi_ref
    phi_uni = phi(ps.loc[dom17, "p_uniform"].to_numpy()) - phi_ref

    cal = pd.read_csv(f"{IF1}/P1_scale_calibration.csv")
    pool = cal.groupby(["scale", "N", "role"], sort=False).apply(
        lambda s: pd.Series(dict(pooled_slope=s.cov_yg.sum() / s.var_g.sum(),
                                 slope_min=s.calib_slope.min(), slope_max=s.calib_slope.max(),
                                 mean_obs_loss=s.mean_obs_loss.mean()))).reset_index()
    # 可约份额：扣除 E 与 RegMix 训练语料质量 Q_ref 下的地板（跨附件假设：同一 E、同一地板）
    floor_ref = k0 * (1 - Q_ref)
    pool["share_reducible"] = (pool.mean_obs_loss - E0 - floor_ref) / pool.mean_obs_loss
    tr_row = pool[pool.role == "train"].iloc[0]
    s_train = float(tr_row.pooled_slope)
    share_train = float(tr_row.share_reducible)
    s_red = s_train / share_train
    pool["pred_H_red"] = s_red * pool.share_reducible
    pool["pred_H_tot"] = s_train
    pool["err_H_red"] = pool.pred_H_red - pool.pooled_slope
    pool["err_H_tot"] = pool.pred_H_tot - pool.pooled_slope
    gap = (pool.pred_H_tot - pool.pred_H_red).where(lambda z: z.abs() > 0.05)
    pool["position_red0_tot1"] = (pool.pooled_slope - pool.pred_H_red) / gap
    pool.to_csv(f"{OUT}/tables/T28_p_channel.csv", index=False, encoding="utf-8-sig")
    print("\n[T28 配比通道] 参数只由 1M 训练行定；其余行为冻结系数的评估")
    print(pool.round(4).to_string(index=False))
    xs = pool[(pool.role == "eval") & (pool.N > 1e6)]
    inside = bool(xs.position_red0_tot1.between(0, 1).all())
    pos = {r.scale: round(float(r.position_red0_tot1), 3) for r in xs.itertuples()}
    print(f"  s_red={s_red:.4f}（训练斜率 {s_train:.4f} ÷ 训练可约份额 {share_train:.4f}，地板扣除 {floor_ref:.4f}）；"
          f"φ̃(p*)={phi_star:.4f}；60M/1B 落在两假设之间：{inside}，相对位置 {pos}")

    qd = pd.read_csv(f"{IF1}/P1_Q_domain.csv")
    pd.DataFrame(dict(domain=qd.mixture_domain, q_final=qd.Q_final, Q_B_main=qd.Q_final,
                      Q_B_alt=qd.Q_final / Q0_main, alt_exceeds_1=qd.Q_final / Q0_main > 1,
                      mapping_type=qd.mapping_type, CI_lo=qd.CI_lo, CI_hi=qd.CI_hi)
                 ).to_csv(f"{OUT}/tables/T28b_Q_mapping_domains.csv", index=False, encoding="utf-8-sig")

    # ================= 图 =================
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    for d_, c in zip(sorted(diag.D.unique()), ["#08519c", "#3182bd", "#6baed6", "#fd8d3c", "#a63603"]):
        s = diag[diag.D == d_].sort_values("N")
        ax.semilogx(s.N, s.dL_obs, "o", color=c, ms=5, label=f"D={d_:.0f}B")
        ax.semilogx(s.N, s.dL_main, "-", color=c, lw=1.5)
    ax.set_xlabel("N（十亿）"); ax.set_ylabel("L(Q=0.1)−L(Q=1)")
    ax.set_title(f"质量收益随 N 下降（实线：{SHAPE}-{model_id(MODEL, FLOOR)}）"); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3)
    ax = axes[1]
    for q_, c in [(0.5, "#fd8d3c"), (0.7, "#2b8cbe")]:
        s = hold[hold.Q_score == q_]
        ax.scatter(s.val_loss, s.pred, s=14, color=c, label=f"留出 Q={q_}")
    lims = [hold.val_loss.min() - .05, hold.val_loss.max() + .05]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("观测"); ax.set_ylabel("预测")
    ax.set_title(f"B7 新增 90 点留出：RMSE {sel.holdout_rmse:.3f}"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax = axes[2]
    xx = np.log10(pool.N)
    ax.plot(xx, pool.pooled_slope, "ko-", label="观测校准斜率")
    ax.plot(xx, pool.pred_H_red, "gs--", label="H_red：配比乘可约损失（主）")
    ax.plot(xx, pool.pred_H_tot, "r^:", label="H_tot：配比乘总损失（上界）")
    ax.set_xticks(xx); ax.set_xticklabels(pool.scale, fontsize=8)
    ax.set_ylabel("配比效应尺度"); ax.set_title("配比通道：两假设都只用 1M 训练定参数")
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/F22_form_selection.png"); plt.close(fig)

    # ================= 接口 =================
    hform = {"pow": "Q^{-κ}", "exp": "exp(κ(1-Q))", "lin": "1+κ(1-Q)"}[SHAPE]
    if SHAPE == "pow":
        form = "L = E + k0(1-Q) + [A·N^{-α}·Q^{-κ_N} + B·D^{-β}·Q^{-κ_D}]·exp(s_red·φ̃(p))"
    else:
        form = (f"L = E + k0(1-Q) + [A·N^{{-α}}·h(Q;κ_N) + B·D^{{-β}}·h(Q;κ_D)]"
                f"·exp(s_red·φ̃(p)),  h={hform}")
    params = dict(
        version="v7",
        E=E0, A=A0, alpha=float(al), B=B0, beta=float(be),
        kappa_N=float(kN), kappa_D=float(kD), k0=float(k0),
        q_shape=SHAPE, selected_model=model_id(MODEL, FLOOR), floor=bool(FLOOR), h_form=hform,
        form=form,
        degeneration="Q=1 且 p=p_ref 时 L = E + A N^-α + B D^-β；N,D→∞ 时 L→E+k0(1−Q)",
        selection_rule="主式固定为幂次 M4+地板：h=Q^{-κ}，地板 k0(1-Q) 保留，两项 κ 不收缩。CV 优先线性的规则只写入 cv_rule_choice，不决定主式",
        cv_rule_choice=f"{auto_shape}-{model_id(auto_m, auto_f)}",
        training_data="B6 360 点", holdout="B7 中不在 B6 的 90 点（Q=0.5/0.7）",
        holdout_rmse=float(sel.holdout_rmse), holdout_bias=float(sel.holdout_bias),
        cv_rmse_leaveN=float(sel.cv_rmse_leaveN), noise_sigma_proxy=noise_sigma,
        fullB7_comparison=dict(kappa_N=fa["kN"], kappa_D=fa["kD"], k0=fa["k0"], note="含训练点，仅对照"),
        nofloor_sensitivity=dict(
            model=model_id(MODEL, alt_f), q_shape=SHAPE,
            kappa_N=float(ft_alt["kN"]), kappa_D=float(ft_alt["kD"]), k0=float(ft_alt["k0"]),
            cv_rmse_leaveN=float(alt.cv_rmse_leaveN), holdout_rmse=float(alt.holdout_rmse),
            holdout_bias=float(alt.holdout_bias),
            note="与主式同形状同约束、地板取反；下游须用本组参数做敏感性"),
        p_channel=dict(
            status="exploratory",
            structure="配比乘子只作用于随规模变化的部分 A N^-α h_N + B D^-β h_D；E 与地板不受配比乘子影响",
            phi_def="φ(p)=mean_v(β_v^{1M}·clr(p))；φ̃=φ(p)−φ(p_ref)",
            p_ref="P1_p_star.p_train_mean（A4 训练配比均值，作为附件 B 隐含 Pile 配比的代理）",
            phi_tilde_pstar=phi_star, phi_tilde_uniform=phi_uni,
            s_red_main=float(s_red),
            s_red_rule="H_red：s_red 恒定 = 1M 训练校准斜率 / 1M 训练可约份额（扣除 E 与 k0(1−Q_ref)）",
            upper_rule="H_tot：配比乘总损失，等价于对 L 整体乘 exp(s_train·φ̃)",
            s_train=s_train, share_train=share_train, Q_ref=Q_ref, floor_ref=float(floor_ref),
            evaluation_inside_bracket_60M_1B=inside, evaluation_position_red0_tot1=pos,
            evidence="A6–A11 只算冻结 1M 系数的校准斜率，未重估任何参数；见 T28"),
        Q_anchor="Q_B = Q_A（问题一领域质量）",
        Q0_main=Q0_main,
        q_mapping=dict(
            main=dict(name="identity", formula="Q_B = Q_A", parameters={}, support=[0.0, 1.0],
                      clip_policy="error", evidence_level="题面语义锚点假设", source="doc/Q2/问题二_SPEC.md"),
            alt=dict(name="pstar_anchor", formula="Q_B = Q_A / Qbar_pstar", parameters=dict(Qbar_pstar=Q0_main),
                     support=[0.0, 1.0], clip_policy="error",
                     evidence_level="敏感性假设；部分领域超出 B 数据支持域", source="doc/Q2/问题二_SPEC.md")),
        q_domain_range=[float(qd.Q_final.min()), float(qd.Q_final.max())],
    )
    with open(f"{OUT}/interface/P2_scaling_law.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    np.savez(f"{OUT}/interface/P2_bootstrap.npz", samples=bt.to_numpy(), cols=np.array(list(bt.columns)))
    print(f"\nQ2-A done. 主模型 {SHAPE}-{model_id(MODEL, FLOOR)}")
