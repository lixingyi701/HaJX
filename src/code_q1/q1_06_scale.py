# -*- coding: utf-8 -*-
"""
q1_06_scale.py — Step6：跨尺度检验、尺度修正与外推表稳健性
配比系数只在 A4/A5（1M 训练）上拟合后冻结。
  ① A6–A11：冻结 1M clr+Huber 模型直接预测 → 逐域 R²/RMSE/Spearman、排名衰减（P1_rank_decay）
  ② P1_scale_calibration：冻结 1M 系数的逐域校准斜率（仅评估统计量）
  ③ 冻结 1M GBDT 直接预测 60M/1B
  ⑤ 尺度修正（按规模留一档）：ln L_v = b_v^1M + ρ_v t + s(N)·g_v(p)，t=ln(N/1e6)；
     s(N) 两种形式并列：对数线性 1+κt（v7 起，对照）与幂律 (N/1e6)^(-λ)（v9 候选，λ=-ln s(60M)/t60）；
     ρ_v 与缩放参数只由 1M 训练与 60M（A8/A9）确定，1B（A10/A11）只作独立检验
  ④ A12–A15（估算/外推、非观测）：冻结模型、尺度修正模型与"同配比 1M 观测 Loss"并列；
     A12/A14 是 A4 的子集，A13/A15 由三尺度幂律外推生成，只作一致性对照
"""
import os, sys, pickle
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q1_00_common import (CACHE, TABLES, IFACE, FIGS, clr, r2_score, rmse, mae, spearman,
                          gbdt_predict, setup_cjk_matplotlib)
from q1_05_mixture import load_pair, predict_all

SCALES = {"1M_heldout": ("test_mixture_1m.csv", "test_pile_loss_1m.csv", 1e6),
          "60M": ("test_mixture_60m.csv", "test_pile_loss_60m.csv", 6e7),
          "1B": ("test_mixture_1B.csv", "test_pile_loss_1B.csv", 1e9)}
ESTS = {"10B_est": ("est_mixture_10b.csv", "est_pile_loss_10b.csv", 1e10),
        "70B_est": ("est_mixture_70b.csv", "est_pile_loss_70b.csv", 7e10)}
TRAIN = ("train_mixture_1m.csv", "train_pile_loss_1m.csv")


def gbdt_all(models, Z):
    return np.exp(np.stack([gbdt_predict(m, Z) for m in models], 1))


def pooled_level(G, lnY, slope=None):
    """汇总斜率 Σcov/Σvar（或给定斜率）与逐域截距 b_v = ȳ_v − s·ḡ_v"""
    gc, yc = G - G.mean(0), lnY - lnY.mean(0)
    s = float((gc * yc).sum() / (gc * gc).sum()) if slope is None else float(slope)
    return s, lnY.mean(0) - s * G.mean(0)


def level_metrics(Y, lnYh, w):
    Yh = np.exp(lnYh)
    return dict(MAE_domain_mean=float(np.abs(Yh - Y).mean()), MAE_target=mae(Y @ w, Yh @ w),
                R2_mean=float(np.mean([r2_score(Y[:, v], Yh[:, v]) for v in range(Y.shape[1])])),
                spearman_target=spearman(Y @ w, Yh @ w),
                mean_pred_loss=float(Yh.mean()), mean_obs_loss=float(Y.mean()))


if __name__ == "__main__":
    st5 = np.load(f"{CACHE}/step5_mixture.npz", allow_pickle=True)
    B_clr, w_eval = st5["B_clr"], st5["w_eval"]
    DOM13 = st5["DOM13"].tolist()
    with open(f"{CACHE}/step5_gbdt.pkl", "rb") as f:
        gb_1m = pickle.load(f)

    # ---- ① 冻结 1M 线性模型直接预测三个检验尺度 ----
    rows, decay, data = [], [], {}
    for tag, (mf, lf, N) in SCALES.items():
        P, Y, _, lcols, _ = load_pair(mf, lf)
        doms = [c.replace("metric/the_pile_", "").replace("_val_loss", "") for c in lcols]
        assert doms == DOM13, f"{tag} loss 列不一致: {doms}"
        Z = clr(P)
        Yh = np.exp(predict_all(B_clr, Z))
        sp = spearman(Y @ w_eval, Yh @ w_eval)
        rows.append(dict(scale=tag, N=N, n_configs=len(P), spearman_target=sp,
                         spearman_domain_mean=float(np.mean([spearman(Y[:, v], Yh[:, v]) for v in range(13)])),
                         R2_mean=float(np.mean([r2_score(Y[:, v], Yh[:, v]) for v in range(13)])),
                         RMSE_mean=float(np.mean([rmse(Y[:, v], Yh[:, v]) for v in range(13)])),
                         MAE_mean=float(np.mean([mae(Y[:, v], Yh[:, v]) for v in range(13)]))))
        decay.append(dict(scale=tag, N=N, spearman=sp))
        data[tag] = (P, Y, Z)
    val_tab = pd.DataFrame(rows)
    val_tab.to_csv(f"{TABLES}/T6_cross_scale_validation.csv", index=False)
    print("冻结 1M 线性模型直接预测（绝对 Loss 有尺度平移，R² 为负属预期）:")
    print(val_tab.round(4).to_string(index=False))
    pd.DataFrame(decay).to_csv(f"{IFACE}/P1_rank_decay.csv", index=False)

    # ---- ② 冻结 1M 系数的校准统计（仅评估） ----
    cal_sets = {"1M_train": (*TRAIN, 1e6, "train")}
    cal_sets.update({t: (m, l, n, "eval") for t, (m, l, n) in SCALES.items()})
    cal_rows = []
    for tag, (mf, lf, N, role) in cal_sets.items():
        P, Y, _, lcols, _ = load_pair(mf, lf)
        assert [c.replace("metric/the_pile_", "").replace("_val_loss", "") for c in lcols] == DOM13
        Z = clr(P)
        for v, d in enumerate(DOM13):
            g = Z @ B_clr[v, 1:]
            y = np.log(Y[:, v])
            gc, yc = g - g.mean(), y - y.mean()
            cal_rows.append(dict(scale=tag, N=N, role=role, n_configs=len(P), domain=d,
                                 cov_yg=float((gc*yc).mean()), var_g=float((gc*gc).mean()),
                                 calib_slope=float((gc*yc).sum()/(gc*gc).sum()),
                                 mean_obs_loss=float(Y[:, v].mean()), spearman=spearman(y, g)))
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(f"{IFACE}/P1_scale_calibration.csv", index=False)
    pooled = cal.groupby(["scale", "N", "role"], sort=False).apply(
        lambda s: pd.Series(dict(pooled_slope=s.cov_yg.sum()/s.var_g.sum(),
                                 mean_obs_loss=s.mean_obs_loss.mean()))).reset_index()
    print("\nP1_scale_calibration（冻结 1M 系数的校准斜率，仅评估）:")
    print(pooled.round(4).to_string(index=False))

    # ---- ③ 冻结 1M GBDT 直接预测 60M/1B ----
    g_rows = []
    for tag in ("1M_heldout", "60M", "1B"):
        P, Y, Z = data[tag]
        Yh = gbdt_all(gb_1m, Z)
        g_rows.append(dict(test_scale=tag, model="1M_GBDT_frozen",
                           spearman_target=spearman(Y @ w_eval, Yh @ w_eval),
                           MAE_target=mae(Y @ w_eval, Yh @ w_eval)))
    gtab = pd.DataFrame(g_rows)
    gtab.to_csv(f"{TABLES}/T6_gbdt_cross_scale.csv", index=False)
    print("\n冻结 1M GBDT 直接预测:"); print(gtab.round(4).to_string(index=False))

    # ---- ⑤ 尺度修正：1M 训练 + 60M 标定，1B 独立检验 ----
    b1m, Bet = B_clr[:, 0], B_clr[:, 1:]
    tN = lambda N: np.log(N / 1e6)
    G = {tag: data[tag][2] @ Bet.T for tag in data}          # 冻结配比部分 g_v(p)
    lnY = {tag: np.log(data[tag][1]) for tag in data}
    s60, b60 = pooled_level(G["60M"], lnY["60M"])
    _, b60_int = pooled_level(G["60M"], lnY["60M"], slope=1.0)
    t60 = tN(6e7)
    kappa, rho, rho_int = (s60 - 1) / t60, (b60 - b1m) / t60, (b60_int - b1m) / t60
    lam = -np.log(s60) / t60        # 幂律 s(N)=(N/1e6)^(-λ)，与 κ 同源（只读 1M+60M）

    def corrected(Gm, N, slope=True, form="linear"):
        t = tN(N)
        if not slope:
            return b1m + rho_int * t + Gm
        s = (1 + kappa * t) if form == "linear" else (N / 1e6) ** (-lam)
        return b1m + rho * t + s * Gm

    s1b_obs, b1b_obs = pooled_level(G["1B"], lnY["1B"])
    sc_rows = []
    for tag, role in (("60M", "标定集（同集）"), ("1B", "独立检验")):
        N = SCALES[tag][2]; Y = data[tag][1]
        for name, lnh in (("冻结1M", b1m + G[tag]),
                          ("仅截距修正", corrected(G[tag], N, slope=False)),
                          ("截距+缩放修正(对数线性)", corrected(G[tag], N, form="linear")),
                          ("截距+缩放修正(幂律)", corrected(G[tag], N, form="power"))):
            sc_rows.append(dict(scale=tag, N=N, role=role, method=name, **level_metrics(Y, lnh, w_eval)))
    sc_rows.append(dict(scale="1B", N=1e9, role="同集拟合（误差下界参照，不作结论）", method="1B同集截距+斜率",
                        **level_metrics(data["1B"][1], b1b_obs + s1b_obs * G["1B"], w_eval)))
    sc_tab = pd.DataFrame(sc_rows)
    sc_tab.to_csv(f"{TABLES}/T6_scale_correction.csv", index=False)
    par = pd.DataFrame(dict(domain=DOM13, b_1M=b1m, b_60M=b60, rho=rho, rho_intercept_only=rho_int,
                            b_1B_insample=b1b_obs))
    par["kappa"] = kappa; par["s_60M"] = s60
    par["s_1B_pred"] = 1 + kappa * tN(1e9); par["s_1B_insample"] = s1b_obs
    par["lambda"] = lam; par["s_1B_pred_power"] = (1e9 / 1e6) ** (-lam)
    par.to_csv(f"{TABLES}/T6_scale_correction_params.csv", index=False)
    print(f"\n尺度修正（ρ_v、κ、λ 只由 1M 训练 + 60M 定）：s(60M)={s60:.4f}, κ={kappa:.5f}, λ={lam:.5f}, "
          f"s(1B) 预测 对数线性 {1 + kappa * tN(1e9):.4f} / 幂律 {(1e9/1e6)**(-lam):.4f} / 同集 {s1b_obs:.4f}")
    print(sc_tab.round(4).to_string(index=False))

    # ---- ④ A12–A15 估算表：冻结模型 + 尺度修正 + 同配比 1M 观测基准 ----
    Ptr, Ytr, _, _, idx_tr = load_pair(*TRAIN)
    tr_by_idx = pd.DataFrame(Ytr, index=idx_tr)
    ext_rows, dom_rows = [], []
    for tag, (mf, lf, N) in ESTS.items():
        P, Y, _, lcols, idx = load_pair(mf, lf)
        assert [c.replace("metric/the_pile_", "").replace("_val_loss", "") for c in lcols] == DOM13
        assert set(idx) <= set(idx_tr), f"{tag} 配比不全在 A4 中"
        Y1m = tr_by_idx.loc[idx].to_numpy()
        Ptr_sub = Ptr[pd.Index(idx_tr).get_indexer(idx)]
        same_mix = bool(np.allclose(Ptr_sub, P, atol=1e-6))
        Z = clr(P)
        preds = {"冻结1M线性": np.exp(predict_all(B_clr, Z)), "冻结1M_GBDT": gbdt_all(gb_1m, Z),
                 "1M线性+尺度修正-对数线性(1M+60M标定)": np.exp(corrected(Z @ Bet.T, N, form="linear")),
                 "1M线性+尺度修正-幂律(1M+60M标定)": np.exp(corrected(Z @ Bet.T, N, form="power")),
                 "同配比1M观测Loss": Y1m}
        Lt = Y @ w_eval
        for name, Yh in preds.items():
            ext_rows.append(dict(est_table=tag, N=N, n=len(P), method=name,
                                 spearman_target=spearman(Lt, Yh @ w_eval),
                                 spearman_domain_mean=float(np.mean([spearman(Y[:, v], Yh[:, v]) for v in range(13)])),
                                 MAE_domain_mean=float(np.abs(Yh - Y).mean()),
                                 MAE_target=mae(Lt, Yh @ w_eval),
                                 mean_pred_loss=float(Yh.mean()), mean_est_loss=float(Y.mean()),
                                 mixtures_equal_A4_subset=same_mix))
        for v, d in enumerate(DOM13):
            dom_rows.append(dict(est_table=tag, domain=d,
                                 spearman_frozen_linear=spearman(Y[:, v], preds["冻结1M线性"][:, v]),
                                 spearman_frozen_gbdt=spearman(Y[:, v], preds["冻结1M_GBDT"][:, v]),
                                 spearman_obs_1M=spearman(Y[:, v], Y1m[:, v])))
    ext_tab = pd.DataFrame(ext_rows)
    ext_tab.to_csv(f"{TABLES}/T6_extrapolation.csv", index=False)
    dom_tab = pd.DataFrame(dom_rows)
    dom_tab.to_csv(f"{TABLES}/T6_domain_stability_rank.csv", index=False)
    print("\nA12–A15（估算/外推、非观测；A12/A14 为 A4 子集）:")
    print(ext_tab.round(4).to_string(index=False))
    d10 = dom_tab[dom_tab.est_table == "10B_est"].sort_values("spearman_frozen_linear", ascending=False)
    print("10B 估算表逐域 Spearman（冻结线性） top3 / bottom3:")
    print(d10.head(3).round(3).to_string(index=False)); print(d10.tail(3).round(3).to_string(index=False))

    for stale in (f"{CACHE}/step6_scale.npz", f"{TABLES}/T6_coef_drift.csv"):
        if os.path.exists(stale):
            os.remove(stale)

    plt = setup_cjk_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    dd = pd.DataFrame(decay)
    ax.semilogx(dd["N"], dd["spearman"], "o-", lw=2, ms=8, color="#2b8cbe", label="冻结 1M 线性")
    gg = gtab.merge(dd[["scale", "N"]], left_on="test_scale", right_on="scale")
    ax.semilogx(gg["N"], gg["spearman_target"], "s--", lw=1.5, color="#fd8d3c", label="冻结 1M GBDT")
    for _, r in dd.iterrows():
        ax.annotate(r["scale"], (r["N"], r["spearman"]), textcoords="offset points", xytext=(6, -12))
    ax.set_xlabel("模型规模 N（参数量）"); ax.set_ylabel("目标损失排名 Spearman ρ")
    ax.set_ylim(0, 1.02); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("配比排序的跨尺度可迁移性（均为冻结 1M 模型直接预测）")
    fig.tight_layout(); fig.savefig(f"{FIGS}/F6_rank_decay.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    obs = sc_tab[sc_tab.method == "冻结1M"].set_index("scale")
    xo = [1e6, 6e7, 1e9]
    yo = [float(Ytr.mean()), obs.loc["60M", "mean_obs_loss"], obs.loc["1B", "mean_obs_loss"]]
    ax.semilogx(xo, yo, "ko", ms=8, label="观测（1M 训练 / 60M / 1B）")
    corr = sc_tab[sc_tab.method == "截距+缩放修正(对数线性)"].set_index("scale")
    corr_p = sc_tab[sc_tab.method == "截距+缩放修正(幂律)"].set_index("scale")
    ext_c = ext_tab[ext_tab.method == "1M线性+尺度修正-对数线性(1M+60M标定)"]
    ext_p = ext_tab[ext_tab.method == "1M线性+尺度修正-幂律(1M+60M标定)"]
    fit_1m = float(np.exp(predict_all(B_clr, clr(Ptr))).mean())
    xc = xo + ext_c.N.tolist()
    yc = [fit_1m, corr.loc["60M", "mean_pred_loss"], corr.loc["1B", "mean_pred_loss"]] + ext_c.mean_pred_loss.tolist()
    ax.semilogx(xc, yc, "s--", color="#2b8cbe", label="尺度修正-对数线性（只用 1M+60M 标定）")
    xp = xo + ext_p.N.tolist()
    yp = [fit_1m, corr_p.loc["60M", "mean_pred_loss"], corr_p.loc["1B", "mean_pred_loss"]] + ext_p.mean_pred_loss.tolist()
    ax.semilogx(xp, yp, "v-.", color="#31a354", label="尺度修正-幂律（只用 1M+60M 标定）")
    ax.semilogx(xo, [fit_1m, obs.loc["60M", "mean_pred_loss"], obs.loc["1B", "mean_pred_loss"]],
                "^:", color="#fd8d3c", label="冻结 1M（不修正）")
    ext_e = ext_c.drop_duplicates("est_table")
    ax.semilogx(ext_e.N, ext_e.mean_est_loss, "D", mfc="none", mec="#a63603", ms=8,
                label="A13/A15 估算值（非观测）")
    ax.axvspan(6e8, 1.6e9, color="#fee391", alpha=.4, lw=0)
    ax.annotate("1B：独立检验", (1e9, yo[2]), textcoords="offset points", xytext=(-30, -18), fontsize=8)
    ax.set_xlabel("模型规模 N（参数量）"); ax.set_ylabel("13 域平均 Loss")
    ax.set_title("尺度修正：1M+60M 标定，1B 独立检验，10B/70B 仅作一致性对照")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{FIGS}/F6_scale_correction.png"); plt.close(fig)
    print("Step6 done.")
