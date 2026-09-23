# -*- coding: utf-8 -*-
"""
q1_06_scale.py — Step6：跨尺度验证与外推稳健性（针对 SQ4/尺度边界）
  ① A6–A11：1M留出/60M/1B 逐域 R²/RMSE/Spearman → 排名衰减曲线（P1-rank_decay）
  ② A12–A15：est 表（幂律外推生成、非观测）
     - 1M 系数直接预测 → 系统偏差
     - 三尺度截距/系数漂移拟合 b_v(N)=b_v0+ρ_v·ln(N/1e6) → 修正后预测
     - Spearman + 灰色关联度：各验证域外推结论稳健性排序
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q1_00_common import (RT, CACHE, TABLES, IFACE, FIGS, clr, huber_regression,
                          predict_lin, r2_score, rmse, mae, spearman,
                          setup_cjk_matplotlib)
from q1_05_mixture import load_pair, predict_all

SCALES = {"1M_heldout": ("test_mixture_1m.csv", "test_pile_loss_1m.csv", 1e6),
          "60M": ("test_mixture_60m.csv", "test_pile_loss_60m.csv", 6e7),
          "1B": ("test_mixture_1B.csv", "test_pile_loss_1B.csv", 1e9)}
ESTS = {"10B_est": ("est_mixture_10b.csv", "est_pile_loss_10b.csv", 1e10),
        "70B_est": ("est_mixture_70b.csv", "est_pile_loss_70b.csv", 7e10)}

def grey_relation(ref, cmp_, rho=0.5):
    """灰色关联度（均值化后）"""
    r = ref / ref.mean(); c = cmp_ / cmp_.mean()
    d = np.abs(r - c)
    return float(((d.min() + rho*d.max()) / (d + rho*d.max())).mean())

if __name__ == "__main__":
    st5 = np.load(f"{CACHE}/step5_mixture.npz", allow_pickle=True)
    B_clr, w_eval = st5["B_clr"], st5["w_eval"]
    DOM13 = st5["DOM13"].tolist()

    # ---- ① 三尺度验证 ----
    rows, decay = [], []
    per_scale_data = {}
    for tag, (mf, lf, N) in SCALES.items():
        P, Y, _, lcols, _ = load_pair(mf, lf)
        # 60M/1B 的 loss 列名应与 1M 相同——核对
        doms = [c.replace("metric/the_pile_", "").replace("_val_loss", "") for c in lcols]
        assert doms == DOM13, f"{tag} loss 列不一致: {doms}"
        Z = clr(P)
        Yh = np.exp(predict_all(B_clr, Z))
        Lt, Lth = Y @ w_eval, Yh @ w_eval
        sp = spearman(Lt, Lth)
        per_r2 = [r2_score(Y[:, v], Yh[:, v]) for v in range(len(DOM13))]
        rows.append(dict(scale=tag, N=N, n_configs=len(P),
                         spearman_target=sp,
                         R2_mean=float(np.mean(per_r2)),
                         RMSE_mean=float(np.mean([rmse(Y[:, v], Yh[:, v]) for v in range(len(DOM13))])),
                         MAE_mean=float(np.mean([mae(Y[:, v], Yh[:, v]) for v in range(len(DOM13))]))))
        decay.append(dict(scale=tag, N=N, spearman=sp))
        per_scale_data[tag] = (P, Y, Z)
    val_tab = pd.DataFrame(rows)
    val_tab.to_csv(f"{TABLES}/T6_cross_scale_validation.csv", index=False)
    print(val_tab.to_string(index=False))

    # ---- 排名衰减接口 P1-rank_decay ----
    pd.DataFrame(decay).to_csv(f"{IFACE}/P1_rank_decay.csv", index=False)

    # ---- ② 尺度漂移模型：对每个验证域拟合 截距+系数 的对数尺度线性漂移 ----
    # 用三尺度各自重拟合的回归系数（60M/1B 上重拟合）观察漂移
    B_by_scale = {"1M_heldout": B_clr}   # 1M 用训练集拟合的主系数
    for tag in ("60M", "1B"):
        P, Y, Z = per_scale_data[tag]
        B_by_scale[tag] = np.stack([huber_regression(Z, np.log(Y[:, v]))
                                    for v in range(len(DOM13))])
    Ns = np.array([SCALES[t][2] for t in B_by_scale])
    lnN = np.log(Ns / 1e6)
    # 漂移拟合：对每个 (v, 参数k)，最小二乘直线 param = a + rho*lnN
    Bs = np.stack([B_by_scale[t] for t in B_by_scale])   # (3, 13, 18)
    A_ = np.column_stack([np.ones(3), lnN])
    coef_drift = np.linalg.lstsq(A_, Bs.reshape(3, -1), rcond=None)[0]  # (2, 13*18)
    a_hat = coef_drift[0].reshape(Bs.shape[1:])
    rho_hat = coef_drift[1].reshape(Bs.shape[1:])
    # 漂移强度表：截距漂移 vs 斜率漂移
    drift_tab = pd.DataFrame({
        "domain": DOM13,
        "rho_intercept": rho_hat[:, 0],
        "rho_beta_maxabs": np.abs(rho_hat[:, 1:]).max(axis=1),
        "rho_beta_meanabs": np.abs(rho_hat[:, 1:]).mean(axis=1)})
    drift_tab.to_csv(f"{TABLES}/T6_coef_drift.csv", index=False)
    print("系数漂移（截距 ρ 前5）:\n",
          drift_tab.sort_values("rho_intercept").head().round(4).to_string(index=False))

    # ---- ③ est 表外推 ----
    ext_rows = []
    for tag, (mf, lf, N) in ESTS.items():
        P, Y, _, lcols, _ = load_pair(mf, lf)
        doms = [c.replace("metric/the_pile_", "").replace("_val_loss", "") for c in lcols]
        Z = clr(P)
        # 直接套 1M 系数
        Yh0 = np.exp(predict_all(B_clr, Z))
        # 尺度修正系数
        B_corr = a_hat + rho_hat * np.log(N / 1e6)
        Yh1 = np.exp(predict_all(B_corr, Z))
        Lt = Y @ w_eval
        for name, Yh in [("直接套用1M系数", Yh0), ("尺度漂移修正后", Yh1)]:
            Lth = Yh @ w_eval
            ext_rows.append(dict(
                est_table=tag, method=name,
                spearman_target=spearman(Lt, Lth),
                RMSE_target=rmse(Lt, Lth), MAE_target=mae(Lt, Lth),
                grey_relation=grey_relation(Lt, Lth)))
    ext_tab = pd.DataFrame(ext_rows)
    ext_tab.to_csv(f"{TABLES}/T6_extrapolation.csv", index=False)
    print(ext_tab.round(4).to_string(index=False))

    # 各验证域外推稳健性排序（10B est 上逐域 Spearman）
    P, Y, _, lcols, _ = load_pair(*ESTS["10B_est"][:2])
    Z = clr(P)
    B_corr = a_hat + rho_hat * np.log(1e10 / 1e6)
    Yh1 = np.exp(predict_all(B_corr, Z))
    dom_stab = pd.DataFrame({
        "domain": DOM13,
        "spearman_10Best": [spearman(Y[:, v], Yh1[:, v]) for v in range(len(DOM13))],
        "grey_10Best": [grey_relation(Y[:, v], Yh1[:, v]) for v in range(len(DOM13))]
    }).sort_values("spearman_10Best", ascending=False)
    dom_stab.to_csv(f"{TABLES}/T6_domain_stability_rank.csv", index=False)
    print("外推稳健域 top3/bottom3:\n", dom_stab.head(3).round(3).to_string(index=False),
          "\n", dom_stab.tail(3).round(3).to_string(index=False))

    np.savez_compressed(f"{CACHE}/step6_scale.npz",
                        a_hat=a_hat, rho_hat=rho_hat)

    # ---- 图：排名保持衰减曲线 ----
    plt = setup_cjk_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    dd = pd.DataFrame(decay)
    ax.semilogx(dd["N"], dd["spearman"], "o-", lw=2, ms=8, color="#2b8cbe")
    for _, r in dd.iterrows():
        ax.annotate(r["scale"], (r["N"], r["spearman"]),
                    textcoords="offset points", xytext=(6, -12))
    ax.set_xlabel("模型规模 N（参数量）"); ax.set_ylabel("目标损失排名 Spearman ρ")
    ax.set_ylim(0, 1.02)
    ax.set_title("配比排序可迁移性随规模的衰减（1M 系数直接预测）")
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{FIGS}/F6_rank_decay.png"); plt.close(fig)
    print("Step6 done.")
