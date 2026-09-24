# -*- coding: utf-8 -*-
"""
q1_02_quality.py — Step2：质量评分 Q
主线：域内标准化 + CRITIC 客观赋权 + 加权 Huber 中心估计。
对照：CRITIC 加权算术平均、PCA、TOPSIS、K-means。
硬规则：
  - 所有变换参数（缩尾端点、MinMax 锚点、单峰化中位数、填补值、CRITIC 权重、delta）在 A1 上估计后冻结，
    A2/A3 用同一套参数复算 —— 否则"管线差异"混进"抽样偏差"（提示词 §2.5）。
  - 域内标准化针对 DSIR 跨域不可比与域基线差（提示词 §1、§2.3）。
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q1_00_common import (CACHE, TABLES, IFACE, IND_COLS, DIRECTION, GROUPS,
                          winsorize, minmax, spearman, pearson, ks_2samp,
                          cohens_d, setup_cjk_matplotlib, FIGS)

RNG = np.random.default_rng(2026)

def weighted_median(Z, w):
    order = np.argsort(Z, axis=1)
    sorted_z = np.take_along_axis(Z, order, axis=1)
    sorted_w = np.broadcast_to(w, Z.shape)
    sorted_w = np.take_along_axis(sorted_w, order, axis=1)
    k = (np.cumsum(sorted_w, axis=1) >= sorted_w.sum(axis=1, keepdims=True) / 2).argmax(axis=1)
    return sorted_z[np.arange(len(Z)), k]

def weighted_huber_center(Z, w, delta, chunk=20000, iterations=28):
    """对每行在 [0,1] 解 sum_j w_j clip(q-z_j,-delta,delta)=0。"""
    if not 0 < delta <= 1:
        raise ValueError("Huber delta must be in (0,1]")
    result = np.empty(len(Z))
    for start in range(0, len(Z), chunk):
        z = Z[start:start+chunk]
        wc = w[start:start+chunk] if np.ndim(w) == 2 else w
        lo = np.zeros(len(z)); hi = np.ones(len(z))
        for _ in range(iterations):
            mid = (lo + hi) / 2
            grad = (np.clip(mid[:, None] - z, -delta, delta) * wc).sum(axis=1)
            lo = np.where(grad < 0, mid, lo)
            hi = np.where(grad >= 0, mid, hi)
        result[start:start+chunk] = (lo + hi) / 2
    return result

# ---------- 变换管线（参数在 A1 上拟合，冻结后应用于 A2/A3） ----------
class QualityPipeline:
    def __init__(self):
        self.params = {}          # 每域每指标: winsor 端点/单峰中位数/minmax 锚点
        self.critic_w = None
        self.global_params = {}
        self.dsir_resid_params = {}  # P2修复: DSIR对log(wc)的残差化参数
        self.impute_medians = {}  # A1 域内缺失填充值
        self.delta = None

    # -- step A: 单峰化 + 方向统一 + 缩尾（域内） --
    def _transform_one(self, x, ind, dom, fit):
        key = (dom, ind)
        d = DIRECTION[ind]
        x = x.astype(float).copy()
        if fit:
            p = {}
            if d == 0:  # 适宜区间型：log1p 后取距域内中位数偏离
                xl = np.log1p(np.clip(x, 0, None)) if np.nanmin(x) >= 0 else x
                med = np.nanmedian(xl)
                v = -np.abs(xl - med)          # 越接近中位数越好
                p["log"] = np.nanmin(x) >= 0; p["med"] = med
            else:
                heavy = ind.startswith("dsir") or ind in ("rps_doc_word_count", "rps_doc_num_sentences")
                xl = np.log1p(np.clip(x, 0, None)) if (heavy and np.nanmin(x) >= 0) else x
                v = xl if d == 1 else -xl
                p["log"] = heavy and np.nanmin(x) >= 0
            v, wz = winsorize(v)
            v, mm = minmax(v)
            p["winsor"] = wz; p["mm"] = mm
            self.params[key] = p
            return v
        p = self.params.get(key)
        if p is None:   # A2/A3 出现 A1 未见过的域（不会发生：arxiv/github 均在 A1）
            p = self.params[("__global__", ind)]
        if d == 0:
            xl = np.log1p(np.clip(x, 0, None)) if p["log"] else x
            v = -np.abs(xl - p["med"])
        else:
            xl = np.log1p(np.clip(x, 0, None)) if p["log"] else x
            v = xl if d == 1 else -xl
        v = np.clip(v, p["winsor"][0], p["winsor"][1])
        v, _ = minmax(v, p["mm"][0], p["mm"][1])
        return v

    def transform(self, df, fit=False):
        """df 需含 IND_COLS + domain；返回 [0,1] 的 25 维标准化矩阵 Z"""
        Z = np.full((len(df), len(IND_COLS)), np.nan)
        _DSIR_COLS = {"dsir_books", "dsir_wiki", "dsir_math"}
        for dom, sub in df.groupby("domain"):
            idx = sub.index.to_numpy()
            pos = df.index.get_indexer(idx)
            # DSIR 测量校正：域内对 log(word_count) 残差化，与冲突类型无关。
            wc_log = np.log1p(sub["rps_doc_word_count"].to_numpy())
            dsir_residualized = {}
            for dsir_col in _DSIR_COLS:
                vals = sub[dsir_col].to_numpy()
                rkey = (dom, dsir_col)
                if fit:
                    mask = ~(np.isnan(vals) | np.isnan(wc_log))
                    if mask.sum() > 10:
                        X = np.column_stack([np.ones(mask.sum()), wc_log[mask]])
                        beta, *_ = np.linalg.lstsq(X, vals[mask], rcond=None)
                    else:
                        beta = np.array([0.0, 0.0])
                    self.dsir_resid_params[rkey] = beta
                else:
                    beta = self.dsir_resid_params.get(rkey, np.array([0.0, 0.0]))
                pred = beta[0] + beta[1] * wc_log
                dsir_residualized[dsir_col] = vals - pred
            for j, ind in enumerate(IND_COLS):
                arr = dsir_residualized[ind] if ind in _DSIR_COLS else sub[ind].to_numpy()
                Z[pos, j] = self._transform_one(arr, ind, dom, fit)
            for j, ind in enumerate(IND_COLS):
                key = (dom, ind)
                if fit:
                    self.impute_medians[key] = float(np.nanmedian(Z[pos, j]))
                fill = self.impute_medians.get(key, .5)
                missing = np.isnan(Z[pos, j])
                Z[pos[missing], j] = fill
        return Z

    # -- step B: CRITIC 权重（A1 全体样本上估计一次） --
    def fit_critic(self, Z):
        sd = Z.std(axis=0, ddof=1)
        R = np.corrcoef(Z.T)
        C = sd * (1 - R).sum(axis=1)
        self.critic_w = C / C.sum()
        return self.critic_w

    def fit_delta(self, Z):
        """A1 上的稳健尺度规则：1.345 × 1.4826 × 逐篇加权 MAD 的中位数。"""
        center = weighted_median(Z, self.critic_w)
        dev = np.abs(Z - center[:, None])
        mad = weighted_median(dev, self.critic_w)
        self.delta = float(np.clip(1.345 * 1.4826 * np.median(mad), .01, 1.0))
        return self.delta

    def score(self, Z):
        return weighted_huber_center(Z, self.critic_w, self.delta)

# ---------- 对照方法（传统数模组） ----------
def pca_score(Z, var_target=0.80):
    Zc = (Z - Z.mean(0)) / (Z.std(0, ddof=1) + 1e-12)
    C = np.cov(Zc.T)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    ratio = vals / vals.sum()
    m = int(np.searchsorted(np.cumsum(ratio), var_target) + 1)
    # 载荷方向修正：使每个主成分与"指标均值方向"正相关（可解释性）
    load = vecs[:, :m].copy()
    for k in range(m):
        if load[:, k].sum() < 0: load[:, k] *= -1
    scores = Zc @ load
    q = scores @ ratio[:m]
    q, _ = minmax(q)
    return q, load, ratio[:m], m

def topsis(Z, w):
    V = Z * w[None, :]
    best, worst = V.max(0), V.min(0)
    dp = np.sqrt(((V - best)**2).sum(1)); dm = np.sqrt(((V - worst)**2).sum(1))
    return dm / (dp + dm + 1e-12)

def kmeans(Z, k=3, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    # k-means++ 初始化
    cent = [Z[rng.integers(len(Z))]]
    for _ in range(k - 1):
        d2 = np.min([( (Z - c)**2).sum(1) for c in cent], axis=0)
        cent.append(Z[rng.choice(len(Z), p=d2/d2.sum())])
    C = np.array(cent)
    for _ in range(iters):
        lab = np.argmin(((Z[:, None, :] - C[None])**2).sum(-1), axis=1)
        newC = np.array([Z[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(newC, C): break
        C = newC
    return lab, C

def bootstrap_ci(x, stat=np.median, B=2000, alpha=0.05, rng=RNG):
    n = len(x)
    batch = max(1, min(B, 2_000_000 // n))
    values = []
    for start in range(0, B, batch):
        idx = rng.integers(0, n, size=(min(batch, B-start), n))
        values.extend(stat(x[i]) for i in idx)
    s = np.asarray(values)
    return float(np.quantile(s, alpha/2)), float(np.quantile(s, 1-alpha/2))

# =====================================================================
if __name__ == "__main__":
    # pickle 保存可从模块路径重新加载的类，而非只在脚本 __main__ 中存在的类。
    sys.modules["q1_02_quality"] = sys.modules[__name__]
    QualityPipeline.__module__ = "q1_02_quality"
    os.makedirs(TABLES, exist_ok=True); os.makedirs(IFACE, exist_ok=True)
    A1 = pd.read_pickle(f"{CACHE}/A1_indicators.pkl")
    A2 = pd.read_pickle(f"{CACHE}/A2_indicators.pkl")
    A3 = pd.read_pickle(f"{CACHE}/A3_indicators.pkl")

    pipe = QualityPipeline()
    Z1 = pipe.transform(A1, fit=True)
    w = pipe.fit_critic(Z1)
    pd.Series(w, index=IND_COLS, name="critic_weight").to_csv(f"{TABLES}/T2_critic_weights.csv")

    delta = pipe.fit_delta(Z1)
    pd.DataFrame([{"dataset":"A1", "delta":delta,
                   "rule":"1.345 × 1.4826 × median_x weighted-MAD_j(z_j(x))"}]
                 ).to_csv(f"{TABLES}/T2_delta_selection.csv", index=False)
    print(f"A1 冻结 Huber delta={delta:.6f}")

    # DSIR 是测量预处理诊断，不根据冲突类型更改任何评分权重。
    dsir_rows = []
    for dom, sub in A1.groupby("domain"):
        wc = np.maximum(sub["rps_doc_word_count"].to_numpy(dtype=float), 1)
        wc_log = np.log1p(wc)
        for ind in ("dsir_books", "dsir_wiki", "dsir_math"):
            perword = sub[ind].to_numpy(dtype=float)
            beta = pipe.dsir_resid_params[(dom, ind)]
            residual = perword - beta[0] - beta[1] * wc_log
            dsir_rows.append(dict(domain=dom, indicator=ind, n=len(sub),
                                  rho_raw_vs_wc=spearman(perword * wc, wc),
                                  rho_perword_vs_wc=spearman(perword, wc),
                                  rho_residual_vs_wc=spearman(residual, wc)))
    pd.DataFrame(dsir_rows).to_csv(f"{TABLES}/T2_dsir_length_diagnostic.csv", index=False)

    # ---- 主线 Q 与对照 ----
    Q1 = pipe.score(Z1)
    Q1_mean = Z1 @ w
    Q1_pca, load, ratio, m = pca_score(Z1)
    Q1_top = topsis(Z1, w)
    lab, cent = kmeans(Z1, k=3, seed=42)
    # 聚类只给相对档位：按簇中心的 Q 均值排序命名 优/中/差
    cent_q = [Q1[lab == j].mean() for j in range(3)]
    order = np.argsort(cent_q)[::-1]
    lab_named = np.select([lab == order[0], lab == order[1], lab == order[2]],
                          ["优质", "普通", "劣质"], default="普通")
    corr = {
        "CRITIC-Huber vs CRITIC算术均值": spearman(Q1, Q1_mean),
        "CRITIC-Huber vs PCA": spearman(Q1, Q1_pca),
        "CRITIC-Huber vs TOPSIS": spearman(Q1, Q1_top),
    }
    pd.Series(corr, name="Spearman").to_csv(f"{TABLES}/T2_method_agreement.csv")
    pd.DataFrame(load, index=IND_COLS,
                 columns=[f"PC{k+1}(方差{r:.1%})" for k, r in enumerate(ratio)]
                 ).to_csv(f"{TABLES}/T2_pca_loadings.csv")
    print("方法一致性:", {k: round(v, 4) for k, v in corr.items()})
    print(f"PCA 取 {m} 个主成分, 累积方差 {ratio.sum():.1%}")
    print("K-means 三档占比:", pd.Series(lab_named).value_counts(normalize=True).round(3).to_dict())

    # 只改变每篇距离其中位数最远的一个指标，衡量少数极端指标的实际影响。
    med = weighted_median(Z1, w)
    extreme_j = np.abs(Z1 - med[:, None]).argmax(axis=1)
    W_loo = np.broadcast_to(w, Z1.shape).copy()
    W_loo[np.arange(len(Z1)), extreme_j] = 0
    W_loo /= W_loo.sum(axis=1, keepdims=True)
    # weighted_huber_center 支持每篇独立权重矩阵（与原主评分同一求解器）。
    Q1_leave = weighted_huber_center(Z1, W_loo, delta)
    Q1_mean_leave = (Z1 * W_loo).sum(axis=1)
    extremeness = np.abs(Z1[np.arange(len(Z1)), extreme_j] - med)
    extreme_cut = float(np.quantile(extremeness, .9))
    influence_rows = []
    for group, mask in [("A1_all", np.ones(len(Z1), dtype=bool)),
                        ("A1_extreme_top10pct", extremeness >= extreme_cut)]:
        for method, base, leave in [("Huber", Q1, Q1_leave),
                                    ("CRITIC_arithmetic", Q1_mean, Q1_mean_leave)]:
            diff = np.abs(base[mask] - leave[mask])
            influence_rows.append(dict(group=group, method=method, n=int(mask.sum()),
                                       extreme_cut=extreme_cut,
                                       median_abs_change=float(np.median(diff)),
                                       p90_abs_change=float(np.quantile(diff, .9))))
    pd.DataFrame(influence_rows).to_csv(f"{TABLES}/T2_extreme_influence.csv", index=False)

    # ---- 同一冻结管线跑 A2/A3 ----
    Z2 = pipe.transform(A2, fit=False); Q2 = pipe.score(Z2)
    Z3 = pipe.transform(A3, fit=False); Q3 = pipe.score(Z3)

    # ---- 域级聚合（中位数主口径 + token 加权副口径 + bootstrap CI） ----
    rows = []
    def dom_stats(df, Q, tag):
        for dom, sub in df.groupby("domain"):
            pos = df.index.get_indexer(sub.index.to_numpy())
            q = Q[pos]; wc = sub["rps_doc_word_count"].to_numpy()
            lo, hi = bootstrap_ci(q)
            tokw = float((q * wc).sum() / wc.sum())
            rows.append(dict(dataset=tag, domain=dom, n=len(q),
                             Q_med=float(np.median(q)), CI_lo=lo, CI_hi=hi,
                             Q_mean=float(q.mean()), Q_tokenw=tokw,
                             P10=float(np.quantile(q, .1)), P90=float(np.quantile(q, .9))))
    dom_stats(A1, Q1, "A1"); dom_stats(A2, Q2, "A2"); dom_stats(A3, Q3, "A3")
    dom_tab = pd.DataFrame(rows)
    dom_tab.to_csv(f"{TABLES}/T2_domain_quality.csv", index=False)
    print(dom_tab.to_string(index=False))

    # ---- 抽样代表性检验：A1(arxiv/github 子集) vs A2/A3 全量 ----
    ks_rows = []
    for tag, ext_df, ext_Q, dom in [("A2", A2, Q2, "arxiv"), ("A3", A3, Q3, "github")]:
        sub = A1[A1["domain"] == dom]
        pos = A1.index.get_indexer(sub.index.to_numpy())
        qa = Q1[pos]
        D, p = ks_2samp(qa, ext_Q)
        d = cohens_d(qa, ext_Q)
        ks_rows.append(dict(domain=dom, KS_D=D, KS_p=p, cohens_d=d,
                            n_sample=len(qa), n_full=len(ext_Q)))
        # 指标级 KS（找差异来源）
        Zsub = Z1[pos]; Zext = Z2 if tag == "A2" else Z3
        ind_ks = {IND_COLS[j]: ks_2samp(Zsub[:, j], Zext[:, j])[0] for j in range(25)}
        pd.Series(ind_ks, name="KS_D").sort_values(ascending=False).to_csv(
            f"{TABLES}/T2_ks_by_indicator_{dom}.csv")
    pd.DataFrame(ks_rows).to_csv(f"{TABLES}/T2_sampling_check.csv", index=False)
    print(pd.DataFrame(ks_rows).to_string(index=False))

    # ---- 缓存到下一步 ----
    np.savez_compressed(f"{CACHE}/step2_scores.npz",
                        Z1=Z1, Q1=Q1, Z2=Z2, Q2=Q2, Z3=Z3, Q3=Q3,
                        w=w, delta=delta, lab=lab, Q1_pca=Q1_pca, Q1_top=Q1_top,
                        Q1_mean=Q1_mean)
    with open(f"{CACHE}/pipeline.pkl", "wb") as f:
        pickle.dump(pipe, f)

    # ---- 图：域级质量分布（小提琴式箱线） ----
    plt = setup_cjk_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    doms = sorted(A1["domain"].unique())
    data = [Q1[A1.index.get_indexer(A1[A1["domain"] == d].index)] for d in doms]
    bp = ax.boxplot(data, labels=doms, showfliers=False, patch_artist=True)
    for b in bp["boxes"]: b.set_facecolor("#9ecae1")
    ax.set_ylabel("综合质量分 $Q$"); ax.set_title("A1 七域质量分布（CRITIC 加权 Huber 中心）")
    fig.tight_layout(); fig.savefig(f"{FIGS}/F2_domain_quality_box.png"); plt.close(fig)

    # ---- 样本级接口 P1-Q_sample ----
    out = pd.DataFrame({"dataset": pd.concat([A1["dataset"], A2["dataset"], A3["dataset"]]),
                        "domain": pd.concat([A1["domain"], A2["domain"], A3["domain"]]),
                        "Q": np.concatenate([Q1, Q2, Q3])})
    out.to_csv(f"{IFACE}/P1_Q_sample.csv.gz", index=False, compression="gzip")
    print("Step2 done.")
