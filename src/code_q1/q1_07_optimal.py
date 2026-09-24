# -*- coding: utf-8 -*-
"""
q1_07_optimal.py — Step7：最优配比 p* 与五份下游接口产出
  ① 单纯形 Dirichlet 采样 10^5 + 主模型预测 + top-k 平均 → p*
     双口径：13 域等权 / pile_cc 单域（web 下游代理，RegMix 原文口径）
  ② 与基准（均匀配比、Pile 人工近似、RegMix 论文最优）的预测收益对比
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

if __name__ == "__main__":
    st5 = np.load(f"{CACHE}/step5_mixture.npz", allow_pickle=True)
    B_clr, w_eval, Qd = st5["B_clr"], st5["w_eval"], st5["Qd"]
    DOM17, DOM13 = st5["DOM17"].tolist(), st5["DOM13"].tolist()

    # ---- ① 采样 + 预测（p* 搜索用预测性能最优的 GBDT；线性模型作交叉校验） ----
    P_tr, Y_tr, pcols, _, _ = load_pair("train_mixture_1m.csv", "train_pile_loss_1m.csv")
    center = P_tr.mean(0)
    Pc = sample_simplex(99999, 17, RNG, center=center)
    Zc = clr(Pc)
    with open(f"{CACHE}/step5_gbdt.pkl", "rb") as f:
        gb_models = pickle.load(f)
    from q1_00_common import gbdt_predict
    lnYh = np.stack([gbdt_predict(m, Zc) for m in gb_models], axis=1)
    lnYh_lin = predict_all(B_clr, Zc)          # 线性交叉校验
    Lt_eq = np.exp(lnYh) @ w_eval                    # 等权目标
    icc = DOM13.index("pile_cc")
    Lt_cc = np.exp(lnYh[:, icc])                     # pile_cc 单域目标

    def topk_avg(P, L, k=100):
        idx = np.argsort(L)[:k]
        p = P[idx].mean(0); return p / p.sum()

    p_star_eq = topk_avg(Pc, Lt_eq)
    p_star_cc = topk_avg(Pc, Lt_cc)
    # 线性模型交叉校验：两模型 top-100 集合的目标损失秩相关
    rho_cross = spearman(Lt_eq, np.exp(lnYh_lin) @ w_eval)
    print(f"GBDT vs 线性 采样点目标损失秩相关: {rho_cross:.3f}")

    def gb_pred_target(p, w):
        z = clr(p[None, :])
        ln = np.array([[gbdt_predict(m, z)[0] for m in gb_models]])
        return float((np.exp(ln) @ w).item())

    # ---- ② 基准比较 ----
    p_uniform = np.full(17, 1/17)
    p_pile = P_tr.mean(0)          # 512 组训练配比的均值 ≈ Dirichlet 先验中心（题给近似人工配比）
    def pred_target(p, w):
        z = clr(p[None, :]); return float((np.exp(predict_all(B_clr, z)) @ w).item())
    bench = pd.DataFrame([
        dict(config="p*_等权口径", L_eq=gb_pred_target(p_star_eq, w_eval),
             L_cc=gb_pred_target(p_star_eq, np.eye(13)[icc])),
        dict(config="p*_pile_cc口径", L_eq=gb_pred_target(p_star_cc, w_eval),
             L_cc=gb_pred_target(p_star_cc, np.eye(13)[icc])),
        dict(config="均匀配比", L_eq=gb_pred_target(p_uniform, w_eval),
             L_cc=gb_pred_target(p_uniform, np.eye(13)[icc])),
        dict(config="训练配比均值(近似人工)", L_eq=gb_pred_target(p_pile, w_eval),
             L_cc=gb_pred_target(p_pile, np.eye(13)[icc])),
    ])
    bench["gain_vs_uniform_%"] = (1 - bench["L_eq"] / bench.loc[2, "L_eq"]) * 100
    bench.to_csv(f"{TABLES}/T7_pstar_benchmark.csv", index=False)
    print(bench.round(4).to_string(index=False))

    # bootstrap 收益区间：固定 GBDT 选出的 p*（不重复选优，避免自选自评的乐观偏差），
    # 对配比实验做案例自助，用重拟合的线性模型评估 p* vs 均匀配比的收益
    gains = []
    from q1_00_common import huber_regression
    Z_tr, lnY_tr = clr(P_tr), np.log(Y_tr)
    z_fixed = clr(np.vstack([p_star_eq, p_uniform]))
    for _ in range(200):
        idx = RNG.integers(0, len(Z_tr), len(Z_tr))
        Bb = np.stack([huber_regression(Z_tr[idx], lnY_tr[idx, v]) for v in range(13)])
        pred = np.exp(predict_all(Bb, z_fixed)) @ w_eval
        gains.append((1 - pred[0] / pred[1]) * 100)
    gains = np.array(gains)
    print(f"p* 相对均匀配比收益: {np.median(gains):.2f}% "
          f"[{np.quantile(gains, .025):.2f}, {np.quantile(gains, .975):.2f}] (95% CI)")
    pd.Series({"median_gain_%": float(np.median(gains)),
               "CI_lo": float(np.quantile(gains, .025)),
               "CI_hi": float(np.quantile(gains, .975))}).to_csv(
        f"{TABLES}/T7_gain_ci.csv")

    # ---- ③ 接口 P1-p* ----
    Qbar_eq = float((p_star_eq * Qd).sum())
    Qbar_cc = float((p_star_cc * Qd).sum())
    out = pd.DataFrame({"domain": DOM17,
                        "p_star_eqweight": p_star_eq,
                        "p_star_pilecc": p_star_cc,
                        "p_uniform": p_uniform,
                        "Q_domain": Qd})
    out.to_csv(f"{IFACE}/P1_p_star.csv", index=False)
    with open(f"{IFACE}/P1_summary.json", "w", encoding="utf-8") as f:
        import json
        json.dump({"Qbar_pstar_eqweight": Qbar_eq, "Qbar_pstar_pilecc": Qbar_cc,
                   "Qbar_uniform": float((p_uniform * Qd).sum()),
                   "gain_vs_uniform_pct_median": float(np.median(gains)),
                   "w_eval": "13 验证域等权（主口径）"}, f, ensure_ascii=False, indent=2)
    print(f"Q̄(p*_eq)={Qbar_eq:.3f}, Q̄(p*_cc)={Qbar_cc:.3f}")

    # ---- ④ 图 ----
    plt = setup_cjk_matplotlib()
    # 图 7a：p* 条形对比
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(17); wd = 0.38
    order = np.argsort(p_star_eq)[::-1]
    ax.bar(x - wd/2, p_star_eq[order], wd, label="p*（13域等权口径）", color="#2b8cbe")
    ax.bar(x + wd/2, p_star_cc[order], wd, label="p*（pile_cc 口径）", color="#fd8d3c")
    ax.set_xticks(x, [DOM17[i] for i in order], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("配比份额"); ax.legend()
    ax.set_title("最优配比 $p^*$：两种评价口径对比（Dirichlet 采样 + top-100 平均）")
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
