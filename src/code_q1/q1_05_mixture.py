# -*- coding: utf-8 -*-
"""
q1_05_mixture.py — Step5：配比→Loss 建模（针对 SQ3）
主模型：clr 变换 + 对数空间 Huber 线性回归  ln L_v = b_v + Σ β_{v,i} z_i
改进证据（N1）：与"裸配比 OLS"（RegMix 原式）在检验集上对比
非线性对照：迷你 GBDT（替代 LightGBM，无外网环境自实现）
迁移矩阵：T[i,v] = ∂lnL_v/∂p_i（clr 链式法则折算到单纯形切空间）
λ 交互项（N3）：ln L_v = ... + λ_v·Σ p_i(1-Q̂_{d(i)})，嵌套检验 + bootstrap CI
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q1_00_common import (RT, CACHE, TABLES, IFACE, FIGS, clr, ols, huber_regression,
                          predict_lin, gbdt_fit, gbdt_predict, r2_score, rmse, mae,
                          spearman, setup_cjk_matplotlib)

RNG = np.random.default_rng(2026)

def load_pair(mix_file, loss_file):
    """读配比+Loss 表，按 index 内连接对齐（1B 表 64vs63 行的处理）"""
    M = pd.read_csv(f"{RT}/{mix_file}")
    L = pd.read_csv(f"{RT}/{loss_file}")
    df = M.merge(L, on="index", how="inner")
    pcols = [c for c in M.columns if c != "index"]
    lcols = [c for c in L.columns if c != "index"]
    P = df[pcols].to_numpy(dtype=float)
    Y = df[lcols].to_numpy(dtype=float)
    return P, Y, pcols, lcols, df["index"].to_numpy()

def fit_all(Z, lnY, reg="huber"):
    """13 个验证域各拟合一条回归；返回 (13, 18) 系数阵（截距在前）"""
    f = huber_regression if reg == "huber" else ols
    return np.stack([f(Z, lnY[:, v]) for v in range(lnY.shape[1])])

def predict_all(B, Z):
    return np.stack([predict_lin(B[v], Z) for v in range(B.shape[0])], axis=1)

def eval_model(name, Yh_ln, Y, w_eval):
    """整体与逐域指标；目标损失 = 评价权重加权"""
    Yh = np.exp(Yh_ln)
    per = dict(R2=[r2_score(Y[:, v], Yh[:, v]) for v in range(Y.shape[1])],
               RMSE=[rmse(Y[:, v], Yh[:, v]) for v in range(Y.shape[1])],
               MAE=[mae(Y[:, v], Yh[:, v]) for v in range(Y.shape[1])])
    Lt, Lth = Y @ w_eval, Yh @ w_eval
    return dict(model=name,
                R2_mean=float(np.mean(per["R2"])), RMSE_mean=float(np.mean(per["RMSE"])),
                MAE_mean=float(np.mean(per["MAE"])),
                target_spearman=spearman(Lt, Lth), target_r2=r2_score(Lt, Lth)), per

# =====================================================================
if __name__ == "__main__":
    # ---- 数据 ----
    P_tr, Y_tr, pcols, lcols, _ = load_pair("train_mixture_1m.csv", "train_pile_loss_1m.csv")
    P_te, Y_te, *_ = load_pair("test_mixture_1m.csv", "test_pile_loss_1m.csv")
    print(f"train {P_tr.shape}, test1m {P_te.shape}; 每行配比和范围 "
          f"[{P_tr.sum(1).min():.3f},{P_tr.sum(1).max():.3f}]")
    DOM17 = [c.replace("train_the_pile_", "") for c in pcols]
    DOM13 = [c.replace("metric/the_pile_", "").replace("_val_loss", "") for c in lcols]

    # 评价权重 w：13 验证域等权（主口径；数据无 Pile token 份额可依，等权最中立，
    # 副口径在 Step7 用 pile_cc 单域做敏感性）
    w_eval = np.full(len(lcols), 1/len(lcols))

    Z_tr, Z_te = clr(P_tr), clr(P_te)
    lnY_tr = np.log(Y_tr)

    # ---- ① 主模型 clr+Huber vs 裸 OLS（N1 改进证据） vs GBDT ----
    results, per_domain = [], {}
    B_clr = fit_all(Z_tr, lnY_tr, "huber")
    r, per = eval_model("clr+Huber(主模型)", predict_all(B_clr, Z_te), Y_te, w_eval)
    results.append(r); per_domain["clr_huber"] = per

    B_raw = np.stack([ols(P_tr[:, 1:], lnY_tr[:, v]) for v in range(len(lcols))])  # 裸配比OLS(去1列防完全共线)
    Yh_raw = np.stack([predict_lin(B_raw[v], P_te[:, 1:]) for v in range(len(lcols))], 1)
    r, _ = eval_model("裸配比OLS(RegMix原式)", Yh_raw, Y_te, w_eval)
    results.append(r)

    gb_models = [gbdt_fit(Z_tr, lnY_tr[:, v], n_trees=250, lr=0.06) for v in range(len(lcols))]
    Yh_gb = np.stack([gbdt_predict(m, Z_te) for m in gb_models], 1)
    r, _ = eval_model("GBDT(非线性对照)", Yh_gb, Y_te, w_eval)
    results.append(r)

    res_tab = pd.DataFrame(results)
    res_tab.to_csv(f"{TABLES}/T5_model_comparison.csv", index=False)
    print(res_tab.to_string(index=False))

    # ---- ② 迁移矩阵 T（17×13）----
    # clr 链式：∂lnL_v/∂p_i = (β_{v,i} - mean_k β_{v,k})/p_i；
    # 在参考点 p=均匀配比处评估，得到无量纲化的"份额转移弹性"
    Beta = B_clr[:, 1:]                     # (13,17)
    Bc = Beta - Beta.mean(axis=1, keepdims=True)
    p0 = np.full(17, 1/17)
    T = (Bc / p0[None, :]).T                # (17,13): ∂lnL_v/∂p_i at uniform
    T_tab = pd.DataFrame(T, index=DOM17, columns=DOM13)
    T_tab.to_csv(f"{TABLES}/T5_transfer_matrix.csv")
    # 自域效应 vs 跨域外溢
    diag = {d: float(T_tab.loc[d, d]) for d in DOM13 if d in DOM17}
    print("自域效应(应为负=加自己域降自己域loss):",
          {k: round(v, 2) for k, v in sorted(diag.items(), key=lambda x: x[1])[:5]})

    # ---- ③ λ 质量交互项（N3）----
    br = np.load(f"{CACHE}/step4_bridge.npz", allow_pickle=True)
    qmap = dict(zip(br["domains"].tolist(), br["Q_final"]))
    Qd = np.array([qmap[d] for d in DOM17])
    print("17域质量分对齐:", {d: round(q, 3) for d, q in zip(DOM17, Qd)})
    qpen_tr = (P_tr * (1 - Qd)[None, :]).sum(1)   # Σ p_i (1-Q_d)
    qpen_te = (P_te * (1 - Qd)[None, :]).sum(1)

    lam_rows = []
    for v in range(len(lcols)):
        X0, X1 = Z_tr, np.column_stack([Z_tr, qpen_tr])
        b0 = huber_regression(X0, lnY_tr[:, v])
        b1 = huber_regression(X1, lnY_tr[:, v])
        # bootstrap λ 置信区间
        lams = []
        for _ in range(400):
            idx = RNG.integers(0, len(X1), len(X1))
            lams.append(huber_regression(X1[idx], lnY_tr[idx, v])[-1])
        lo, hi = np.quantile(lams, [.025, .975])
        # 检验集增益
        p0_ = predict_lin(b0, Z_te); p1_ = predict_lin(b1, np.column_stack([Z_te, qpen_te]))
        lam_rows.append(dict(domain=DOM13[v], lam=float(b1[-1]), CI_lo=float(lo), CI_hi=float(hi),
                             sig=bool(lo > 0 or hi < 0),
                             dRMSE_test=rmse(np.log(Y_te[:, v]), p1_) - rmse(np.log(Y_te[:, v]), p0_)))
    lam_tab = pd.DataFrame(lam_rows)
    lam_tab.to_csv(f"{TABLES}/T5_lambda_interaction.csv", index=False)
    n_sig = int(lam_tab["sig"].sum())
    print(f"λ 交互项: {n_sig}/13 个验证域显著;  λ 中位数={lam_tab['lam'].median():.3f}")
    print(lam_tab.round(4).to_string(index=False))

    # ---- 缓存 ----
    np.savez_compressed(f"{CACHE}/step5_mixture.npz",
                        B_clr=B_clr, w_eval=w_eval, Qd=Qd,
                        DOM17=np.array(DOM17), DOM13=np.array(DOM13))
    import pickle
    with open(f"{CACHE}/step5_gbdt.pkl", "wb") as f:
        pickle.dump(gb_models, f)

    # ---- 接口 P1-f(p)：系数表 ----
    coef = pd.DataFrame(B_clr, index=DOM13,
                        columns=["intercept"] + [f"beta_clr_{d}" for d in DOM17])
    coef.to_csv(f"{IFACE}/P1_f_p_coefficients.csv")
    T_tab.to_csv(f"{IFACE}/P1_transfer_matrix.csv")

    # ---- 图：迁移矩阵热力图 ----
    plt = setup_cjk_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = np.abs(T).max()
    im = ax.imshow(T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(DOM13)), DOM13, rotation=90, fontsize=8)
    ax.set_yticks(range(len(DOM17)), DOM17, fontsize=8)
    ax.set_xlabel("验证域 v"); ax.set_ylabel("训练配比域 i")
    ax.set_title("迁移矩阵 $T_{iv}=\\partial \\ln L_v/\\partial p_i$（均匀配比处；蓝=增配降损）")
    fig.colorbar(im, shrink=.8)
    fig.tight_layout(); fig.savefig(f"{FIGS}/F5_transfer_matrix.png"); plt.close(fig)
    print("Step5 done.")
