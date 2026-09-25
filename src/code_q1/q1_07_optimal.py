# -*- coding: utf-8 -*-
"""
q1_07_optimal.py — Step7：最优配比 p* 与五份下游接口产出
  ① 单纯形 Dirichlet 采样 10^5 → 信赖域（clr 马氏距离 ≤ 训练 95% 分位、份额 ≤ 训练最大值）
     → 线性 clr+Huber 主模型预测 + top-k 平均 → p*；GBDT 同域另选 p* 作稳健性对照
     双口径：13 域等权 / pile_cc 单域（web 下游代理，RegMix 原文口径）
  ② 与基准（均匀配比、训练配比均值）的预测收益对比，线性与 GBDT 交叉评估
  ③ λ 汇总、\bar Q(p*) 计算 → P1 接口文件齐套
  ④ 统一重绘全部图表（含 Step2 域级质量箱线图修正字体）
"""
import os, sys, pickle
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q1_00_common import (RT, CACHE, TABLES, IFACE, FIGS, clr, predict_lin,
                          spearman, setup_cjk_matplotlib, IND_COLS)
from q1_05_mixture import load_pair, predict_all

RNG = np.random.default_rng(1234)

def sample_simplex(n, dim, rng, center=None, concs=(0.5, 1.0, 5.0, 20.0)):
    """RegMix 式采样：以训练配比均值为中心的多浓度 Dirichlet（贴近实验分布，
    避免把模型外推到训练配比空间之外的区域）"""
    center = np.full(dim, 1/dim) if center is None else center
    parts = []
    for c in concs:
        alpha = np.maximum(center * dim * c, 1e-3)
        parts.append(rng.dirichlet(alpha, size=n // len(concs)))
    return np.vstack(parts)

def trust_region(Z_tr, P_tr, q=0.95, ridge=1e-3):
    """clr 空间到训练配比的马氏距离 ≤ 训练自身距离的 q 分位，且每域份额 ≤ 训练最大值"""
    mu = Z_tr.mean(0)
    Ci = np.linalg.inv(np.cov(Z_tr.T) + ridge * np.eye(Z_tr.shape[1]))
    def dist(Z):
        D = Z - mu
        return np.einsum("ij,jk,ik->i", D, Ci, D)
    thr = float(np.quantile(dist(Z_tr), q))
    pmax = P_tr.max(0)
    def inside(P):
        return (dist(clr(P)) <= thr) & (P <= pmax + 1e-12).all(1)
    return inside, dist, thr

def topk_avg(P, L, k=100):
    idx = np.argsort(L)[:k]
    p = P[idx].mean(0); return p / p.sum()

def tv(a, b):
    return float(np.abs(a - b).sum() / 2)

if __name__ == "__main__":
    st5 = np.load(f"{CACHE}/step5_mixture.npz", allow_pickle=True)
    B_clr, w_eval, Qd = st5["B_clr"], st5["w_eval"], st5["Qd"]
    DOM17, DOM13 = st5["DOM17"].tolist(), st5["DOM13"].tolist()

    # ---- ① 采样 + 信赖域 + 线性主模型选 p*；GBDT 在同一信赖域内另选作对照 ----
    P_tr, Y_tr, pcols, _, _ = load_pair("train_mixture_1m.csv", "train_pile_loss_1m.csv")
    Z_tr, lnY_tr = clr(P_tr), np.log(Y_tr)
    Pc = sample_simplex(99999, 17, RNG, center=P_tr.mean(0))
    Zc = clr(Pc)
    inside, dist, thr = trust_region(Z_tr, P_tr)
    mask = inside(Pc)
    Pm, Zm = Pc[mask], Zc[mask]
    with open(f"{CACHE}/step5_gbdt.pkl", "rb") as f:
        gb_models = pickle.load(f)
    from q1_00_common import gbdt_predict
    lnYh_lin = predict_all(B_clr, Zc)
    lnYh_gb = np.stack([gbdt_predict(m, Zc) for m in gb_models], axis=1)
    icc = DOM13.index("pile_cc")
    Lt_lin, Lcc_lin = np.exp(lnYh_lin) @ w_eval, np.exp(lnYh_lin[:, icc])
    Lt_gb = np.exp(lnYh_gb) @ w_eval

    p_star_eq = topk_avg(Pm, Lt_lin[mask])
    p_star_cc = topk_avg(Pm, Lcc_lin[mask])
    p_star_gb = topk_avg(Pm, Lt_gb[mask])
    for name, p in (("p*_线性等权", p_star_eq), ("p*_线性pile_cc", p_star_cc), ("p*_GBDT等权", p_star_gb)):
        assert inside(p[None, :])[0], f"{name} 落在信赖域外"
    top_free = np.argsort(Lt_lin)[:100]
    tr_free = dict(mahal_median=float(np.median(dist(Zc[top_free]))),
                   L_eq_lin_median=float(np.median(Lt_lin[top_free])))
    rho_cross = spearman(Lt_lin[mask], Lt_gb[mask])
    print(f"信赖域：马氏距离阈值 {thr:.2f}（训练 95% 分位），保留 {mask.sum()}/{len(Pc)} 个候选")
    print(f"无信赖域时线性前 100 候选：马氏距离中位数 {tr_free['mahal_median']:.1f}，"
          f"预测等权 Loss 中位数 {tr_free['L_eq_lin_median']:.3f}")
    print(f"信赖域内线性 vs GBDT 目标损失秩相关 {rho_cross:.3f}；"
          f"p*_线性 与 p*_GBDT 全变差 {tv(p_star_eq, p_star_gb):.3f}")

    def pred_target(p, w, model):
        z = clr(p[None, :])
        ln = predict_all(B_clr, z) if model == "lin" else np.stack([gbdt_predict(m, z) for m in gb_models], 1)
        return float((np.exp(ln) @ w).item())

    # ---- ② 基准比较：线性主口径 + GBDT 交叉评估 ----
    p_uniform = np.full(17, 1/17)
    p_pile = P_tr.mean(0) / P_tr.mean(0).sum()   # 512 组训练配比均值 ≈ Dirichlet 先验中心
    e_cc = np.eye(13)[icc]
    configs = [("p*_等权口径(线性)", p_star_eq), ("p*_pile_cc口径(线性)", p_star_cc),
               ("p*_GBDT对照", p_star_gb), ("均匀配比", p_uniform), ("训练配比均值(近似人工)", p_pile)]
    bench = pd.DataFrame([dict(config=n,
                               L_eq_lin=pred_target(p, w_eval, "lin"), L_cc_lin=pred_target(p, e_cc, "lin"),
                               L_eq_gbdt=pred_target(p, w_eval, "gb"), L_cc_gbdt=pred_target(p, e_cc, "gb"),
                               TV_vs_pstar_lin=tv(p, p_star_eq), mahal=float(dist(clr(p[None, :]))[0]),
                               Qbar=float((p * Qd).sum())) for n, p in configs])
    iu = [n for n, _ in configs].index("均匀配比")
    bench["gain_vs_uniform_lin_%"] = (1 - bench.L_eq_lin / bench.loc[iu, "L_eq_lin"]) * 100
    bench["gain_vs_uniform_gbdt_%"] = (1 - bench.L_eq_gbdt / bench.loc[iu, "L_eq_gbdt"]) * 100
    bench.to_csv(f"{TABLES}/T7_pstar_benchmark.csv", index=False)
    print(bench.round(4).to_string(index=False))

    # bootstrap：固定线性 p* 的收益区间（系数不确定性；选择与评估同源，偏乐观）；
    # 每次在同一信赖域内重选 p*，记录与主 p* 的全变差（选择稳定性）
    gains, tvs = [], []
    from q1_00_common import huber_regression
    z_fixed = clr(np.vstack([p_star_eq, p_uniform]))
    for _ in range(200):
        idx = RNG.integers(0, len(Z_tr), len(Z_tr))
        Bb = np.stack([huber_regression(Z_tr[idx], lnY_tr[idx, v]) for v in range(13)])
        pred = np.exp(predict_all(Bb, z_fixed)) @ w_eval
        gains.append((1 - pred[0] / pred[1]) * 100)
        tvs.append(tv(topk_avg(Pm, np.exp(predict_all(Bb, Zm)) @ w_eval), p_star_eq))
    gains, tvs = np.array(gains), np.array(tvs)
    print(f"p* 相对均匀配比收益（线性 bootstrap）: {np.median(gains):.2f}% "
          f"[{np.quantile(gains, .025):.2f}, {np.quantile(gains, .975):.2f}] (95% CI)；"
          f"重选 p* 全变差中位数 {np.median(tvs):.3f}，P95 {np.quantile(tvs, .95):.3f}")
    pd.Series({"median_gain_%": float(np.median(gains)),
               "CI_lo": float(np.quantile(gains, .025)),
               "CI_hi": float(np.quantile(gains, .975)),
               "TV_reselect_median": float(np.median(tvs)),
               "TV_reselect_p95": float(np.quantile(tvs, .95))}).to_csv(
        f"{TABLES}/T7_gain_ci.csv")

    # ---- ③ 接口 P1-p* ----
    Qbar_eq = float((p_star_eq * Qd).sum())
    Qbar_cc = float((p_star_cc * Qd).sum())
    out = pd.DataFrame({"domain": DOM17,
                        "p_star_eqweight": p_star_eq,
                        "p_star_pilecc": p_star_cc,
                        "p_star_gbdt_eqweight": p_star_gb,
                        "p_uniform": p_uniform,
                        "p_train_mean": p_pile,
                        "Q_domain": Qd})
    out.to_csv(f"{IFACE}/P1_p_star.csv", index=False)
    with open(f"{IFACE}/P1_summary.json", "w", encoding="utf-8") as f:
        import json
        json.dump({"Qbar_pstar_eqweight": Qbar_eq, "Qbar_pstar_pilecc": Qbar_cc,
                   "Qbar_pstar_gbdt": float((p_star_gb * Qd).sum()),
                   "Qbar_uniform": float((p_uniform * Qd).sum()),
                   "Qbar_train_mean": float((p_pile * Qd).sum()),
                   "gain_vs_uniform_pct_median": float(np.median(gains)),
                   "gain_vs_uniform_pct_linear_point": float(bench.loc[0, "gain_vs_uniform_lin_%"]),
                   "gain_vs_uniform_pct_gbdt_crosscheck": float(bench.loc[0, "gain_vs_uniform_gbdt_%"]),
                   "TV_pstar_linear_vs_gbdt": tv(p_star_eq, p_star_gb),
                   "TV_bootstrap_reselect_median": float(np.median(tvs)),
                   "pstar_model": "clr+Huber 线性（主）；GBDT 同信赖域对照",
                   "trust_region": f"clr 马氏距离 ≤ {thr:.4f}（A4 训练 95% 分位，协方差加 1e-3 岭），且份额 ≤ A4 各域最大值",
                   "w_eval": "13 验证域等权（主口径）"}, f, ensure_ascii=False, indent=2)
    print(f"Q̄(p*_eq)={Qbar_eq:.5f}, Q̄(p*_cc)={Qbar_cc:.5f}, Q̄(p*_GBDT)={(p_star_gb * Qd).sum():.5f}")

    # ---- ④ 图 ----
    plt = setup_cjk_matplotlib()
    # 图 7a：p* 条形对比
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(17); wd = 0.27
    order = np.argsort(p_star_eq)[::-1]
    ax.bar(x - wd, p_star_eq[order], wd, label="p*（线性，13域等权，主口径）", color="#2b8cbe")
    ax.bar(x, p_star_gb[order], wd, label="p*（GBDT 对照，13域等权）", color="#9ecae1")
    ax.bar(x + wd, p_star_cc[order], wd, label="p*（线性，pile_cc 口径）", color="#fd8d3c")
    ax.set_xticks(x, [DOM17[i] for i in order], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("配比份额"); ax.legend(fontsize=8)
    ax.set_title("最优配比 $p^*$：信赖域内 Dirichlet 采样 + top-100 平均")
    fig.tight_layout(); fig.savefig(f"{FIGS}/F7_p_star.png"); plt.close(fig)

    # 图 7b：重绘 Step2 域级质量箱线（字体修正）
    sc = np.load(f"{CACHE}/step2_scores.npz")
    A1 = pd.read_pickle(f"{CACHE}/A1_indicators.pkl")
    Q1 = sc["Q1"]
    doms = sorted(A1["domain"].unique())
    data = [Q1[A1.index.get_indexer(A1[A1["domain"] == d].index)] for d in doms]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bp = ax.boxplot(data, labels=doms, showfliers=False, patch_artist=True)
    for b_ in bp["boxes"]: b_.set_facecolor("#9ecae1")
    ax.set_ylabel("综合质量分 $Q$")
    ax.set_title("A1 七域质量分布（CRITIC 加权 Huber 中心）")
    fig.tight_layout(); fig.savefig(f"{FIGS}/F2_domain_quality_box.png"); plt.close(fig)

    # 图 7c：质量 vs 训练边际价值散点（"质量≠训练价值"证据图）
    Beta = B_clr[:, 1:]
    Bc = Beta - Beta.mean(axis=1, keepdims=True)
    nu = -(w_eval @ Bc)          # 边际价值（等权目标下）
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(Qd, nu, s=60, c="#2b8cbe")
    for d, qx, ny in zip(DOM17, Qd, nu):
        ax.annotate(d, (qx, ny), fontsize=7, xytext=(4, 4), textcoords="offset points")
    rho = spearman(Qd, nu)
    ax.axhline(0, ls=":", c="grey")
    ax.set_xlabel("域级质量分 $\\hat Q_d$"); ax.set_ylabel("训练边际价值 $\\nu_d$")
    ax.set_title(f"内在质量与训练价值的解耦检验：Spearman ρ = {rho:.2f}")
    fig.tight_layout(); fig.savefig(f"{FIGS}/F7_quality_vs_value.png"); plt.close(fig)
    pd.DataFrame({"domain": DOM17, "Q_d": Qd, "nu_d": nu}).to_csv(
        f"{TABLES}/T7_quality_vs_value.csv", index=False)
    print(f"质量-价值解耦: Spearman={rho:.3f}")
    print("Step7 done.")
