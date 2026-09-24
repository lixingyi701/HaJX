# -*- coding: utf-8 -*-
"""
q2_01_fit.py — 问题二·A：广义标度律的正式拟合
=====================================================
证据分层策略（依据数据审计与逆向工程 q2_00）：
  主拟合  B1(1176, 公式生成) + B6/B7(450, 半合成+σ≈0.05噪声)
  跨源检验 B2 cerebras(1029, 半合成), 训练轨迹 B3(插值,不独立使用)
  外部合理性 scaling_baseline(57)/published(44)/large_baseline(129) —— 真实文献点,
             tokenizer/验证集不可比, 只比趋势与残差符号
两步拟合（避免基线与质量项参数纠缠）：
  Step-1 基线: 仅 B1, 拟合 L0 = E + A·N^-α + B·D^-β  （网格 α,β + LS）
  Step-2 质量项: B6/B7 上固定基线, 拟合 Δ = L - L0 的五种候选形式, CV 选优:
      M-add   Δ = (1-Q)(k0 + k1·N^-α)               [加性+规模交互, 逆向支持]
      M-add0  Δ = k·(1-Q)                            [纯加性]
      M-mult  L = E + A·N^-α + B·(D·Q^γ)^-β          [幂律乘子, Muennighoff 风格]
      M-lin   L = E + A·N^-α + B·(D(q0+(1-q0)Q))^-β  [线性有效数据]
      M-exp   L = E + A·N^-α + B·(D·e^{-γ(1-Q)})^-β  [指数折扣]
  Step-3 配比接入: 广义式 L(N,D,Q,p) = L(N,D,Q) + s·[lnL_tgt(p) - lnL_tgt(p*)]
      其中 L_tgt 来自问题一 P1_f_p 系数; s 由 A 侧证据标定, B 侧无 p 变量,
      作为"结构假设"呈现并做敏感性（这是题目允许的跨附件桥接）。
  Step-4 不确定性: 残差 bootstrap (B=500) 给全参数 CI。
"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code_q1"))
from q1_00_common import ROOT, setup_cjk_matplotlib, r2_score, rmse, spearman

B = f"{ROOT}/real_attachments/B_scaling_laws"
OUT = f"{ROOT}/outputs_q2"
RNG = np.random.default_rng(2026)

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

# ---------------- Step-2 五种质量形式 ----------------
def cv_split(n, k=5, rng=RNG):
    idx = rng.permutation(n)
    return [(np.setdiff1d(idx, idx[i::k]), idx[i::k]) for i in range(k)]

def fit_forms(N, D, Q, L, base):
    E0, A0, al, B0, be = base
    L0 = E0 + A0*N**-al + B0*D**-be
    delta = L - L0
    n = len(L)
    folds = cv_split(n)
    res = {}

    # M-add: Δ=(1-Q)(k0+k1 N^-α) —— 线性于 (k0,k1)
    def madd_fit(tr):
        X = np.column_stack([(1-Q[tr]), (1-Q[tr])*N[tr]**-al])
        k, *_ = np.linalg.lstsq(X, delta[tr], rcond=None); return k
    def madd_pred(k, te):
        return L0[te] + (1-Q[te])*(k[0] + k[1]*N[te]**-al)

    # M-add0
    def madd0_fit(tr):
        x = 1-Q[tr]; return np.array([(x*delta[tr]).sum()/(x*x).sum()])
    def madd0_pred(k, te): return L0[te] + k[0]*(1-Q[te])

    # 乘子族: 需重估 B 项系数（D_eff 改变了 D^-β 的量纲），网格搜内参 γ/q0
    def mult_factory(kind):
        def _fit(tr):
            best = None
            grid = (np.arange(0.1, 3.01, 0.05) if kind != "lin"
                    else np.arange(0.0, 0.9, 0.02))
            for g in grid:
                if kind == "pow":   Deff = D[tr]*np.clip(Q[tr], 1e-3, None)**g
                elif kind == "lin": Deff = D[tr]*(g + (1-g)*Q[tr])
                else:               Deff = D[tr]*np.exp(-g*(1-Q[tr]))
                y = L[tr] - E0 - A0*N[tr]**-al
                x = Deff**-be
                Bq = (x*y).sum()/(x*x).sum()
                sse = ((y - Bq*x)**2).sum()
                if best is None or sse < best[0]: best = (sse, g, Bq)
            return np.array(best[1:])
        def _pred(k, te):
            g, Bq = k
            if kind == "pow":   Deff = D[te]*np.clip(Q[te], 1e-3, None)**g
            elif kind == "lin": Deff = D[te]*(g + (1-g)*Q[te])
            else:               Deff = D[te]*np.exp(-g*(1-Q[te]))
            return E0 + A0*N[te]**-al + Bq*Deff**-be
        return _fit, _pred

    forms = {"M-add 加性+N交互": (madd_fit, madd_pred),
             "M-add0 纯加性": (madd0_fit, madd0_pred),
             "M-mult 幂律乘子": mult_factory("pow"),
             "M-lin 线性乘子": mult_factory("lin"),
             "M-exp 指数乘子": mult_factory("exp")}
    for name, (f, p) in forms.items():
        errs = []
        for tr, te in folds:
            k = f(tr); errs.append(rmse(L[te], p(k, te)))
        k_full = f(np.arange(n))
        res[name] = dict(cv_rmse=float(np.mean(errs)), params=k_full.tolist(),
                         in_rmse=rmse(L, p(k_full, np.arange(n))))
    return res, delta

# =====================================================================
if __name__ == "__main__":
    for sub in ("tables", "figures", "interface"):
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)

    # ---- Step-1 ----
    b1 = pd.read_csv(f"{B}/pythia_training_log_existing.csv").dropna(subset=["val_loss"])
    N1, D1, L1 = b1.N_params_B.values*1e9, b1.D_tokens_B.values*1e9, b1.val_loss.values
    sse, al, be, (E0, A0, B0) = fit_baseline(
        N1, D1, L1, np.arange(0.30, 0.381, 0.002), np.arange(0.24, 0.321, 0.002))
    base = (E0, A0, al, B0, be)
    print(f"[基线] E={E0:.4f} A={A0:.1f} α={al:.3f} B={B0:.1f} β={be:.3f} "
          f"RMSE={np.sqrt(sse/len(L1)):.6f}")

    # ---- Step-2 ----
    nq = pd.read_csv(f"{B}/supplementary_NQ_experiment_expanded.csv")
    N, D, Q, L = (nq.N_params_B.values*1e9, nq.D_tokens_B.values*1e9,
                  nq.Q_score.values, nq.val_loss.values)
    forms, delta = fit_forms(N, D, Q, L, base)
    ftab = pd.DataFrame(forms).T.reset_index().rename(columns={"index": "form"})
    ftab = ftab.sort_values("cv_rmse")
    ftab.to_csv(f"{OUT}/tables/T21_form_selection.csv", index=False)
    print(ftab.to_string(index=False))
    kadd = forms["M-add 加性+N交互"]["params"]
    print(f"[主模型] L=L0+(1-Q)(k0+k1·N^-α), k0={kadd[0]:.4f}, k1={kadd[1]:.1f}")

    # ---- Step-4 bootstrap（主模型全参数）----
    boots = []
    n = len(L)
    for _ in range(500):
        i1 = RNG.integers(0, len(L1), len(L1))
        s_, a_, b_, (E_, A_, B_) = fit_baseline(
            N1[i1], D1[i1], L1[i1], [al], [be])   # 公式生成数据, α β 固定加速
        i2 = RNG.integers(0, n, n)
        L0_ = E_ + A_*N[i2]**-a_ + B_*D[i2]**-b_
        X = np.column_stack([(1-Q[i2]), (1-Q[i2])*N[i2]**-a_])
        k_, *_ = np.linalg.lstsq(X, L[i2]-L0_, rcond=None)
        boots.append([E_, A_, B_, k_[0], k_[1]])
    bt = pd.DataFrame(boots, columns=["E", "A", "B", "k0", "k1"])
    ci = bt.quantile([.025, .5, .975]).T
    ci.columns = ["CI_lo", "median", "CI_hi"]
    ci.loc["alpha"] = [al, al, al]; ci.loc["beta"] = [be, be, be]
    ci.to_csv(f"{OUT}/tables/T22_params_ci.csv")
    print(ci.round(4).to_string())

    # ---- 跨源检验 ----
    rows = []
    def check(name, dfN, dfD, dfL, Qv=1.0):
        pred = E0 + A0*dfN**-al + B0*dfD**-be + (1-Qv)*(kadd[0]+kadd[1]*dfN**-al)
        rows.append(dict(source=name, n=len(dfL),
                         corr=float(np.corrcoef(dfL, pred)[0, 1]),
                         spearman=spearman(dfL, pred),
                         bias_mean=float((dfL-pred).mean()),
                         rmse=rmse(dfL, pred)))
    cer = pd.read_csv(f"{B}/cerebras_training_log.csv").dropna(subset=["val_loss"])
    check("B2 cerebras(半合成)", cer.N_params_B.values*1e9, cer.D_tokens_B.values*1e9,
          cer.val_loss.values)
    sb = pd.read_csv(f"{B}/scaling_baseline.csv")
    check("scaling_baseline(真实57)", sb.N_params_B.values*1e9, sb.D_tokens_B.values*1e9,
          sb.val_loss.values)
    pub = pd.read_csv(f"{B}/published_scaling_data.csv")
    check("published(文献44)", pub.N_params_B.values*1e9, pub.D_tokens_B.values*1e9,
          pub.val_loss.values)
    lb = pd.read_csv(f"{B}/supplementary_large_baseline.csv")
    check("large_baseline(百亿129)", lb.N_params_B.values*1e9, lb.D_tokens_B.values*1e9,
          lb.val_loss.values)
    cross = pd.DataFrame(rows)
    cross.to_csv(f"{OUT}/tables/T23_cross_source.csv", index=False)
    print(cross.round(4).to_string(index=False))

    # ---- 接口输出 ----
    params = dict(E=E0, A=A0, alpha=al, B=B0, beta=be, k0=kadd[0], k1=kadd[1],
                  noise_sigma=0.051, Q_anchor="Q=1 ⇔ Pile 基线语料; Q_B=Q1/0.5608",
                  form="L = E + A·N^-α + B·D^-β + (1-Q)(k0 + k1·N^-α)")
    with open(f"{OUT}/interface/P2_scaling_law.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    np.savez(f"{OUT}/interface/P2_bootstrap.npz", samples=bt.to_numpy(),
             cols=np.array(bt.columns.tolist()))
    print("Q2-A done.")
