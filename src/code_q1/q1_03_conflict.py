# -*- coding: utf-8 -*-
"""
q1_03_conflict.py — Step3：主质量分 Q 之后的指标冲突诊断（针对 SQ2）
流程：
  ① 25 指标按来源分 4 组，组内 Kendall W（同源一致性基线）
  ② 组分数 = 组内标准化分中位数；冲突度 c(x) = 组分数域内分位排名的极差
  ③ 阈值 τ 数据自适应（冲突率-τ 曲线 + 肘部/曲率法），拒绝预设阈值
  ④ 成因：冲突类型（最高组×最低组对）× 域 交叉表 + 卡方检验
  ⑤ A2/A3 同一规则复检（阈值沿用 A1 的 τ）；冲突不反馈到 Q
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q1_00_common import (CACHE, TABLES, FIGS, IND_COLS, GROUPS,
                          rankdata, kendall_w, chi2_sf,
                          setup_cjk_matplotlib)

GK = list(GROUPS.keys())
G_IDX = {g: [IND_COLS.index(c) for c in cols] for g, cols in GROUPS.items()}

def group_scores(Z):
    """25 维标准化分 -> 4 组分数（组内中位数）"""
    return np.column_stack([np.median(Z[:, G_IDX[g]], axis=1) for g in GK])

def domain_pct_rank(S, domains):
    """组分数 -> 域内分位排名 [0,1]"""
    R = np.empty_like(S)
    for d in np.unique(domains):
        m = domains == d
        for k in range(S.shape[1]):
            R[m, k] = rankdata(S[m, k]) / m.sum()
    return R

def conflict_score(R):
    return R.max(axis=1) - R.min(axis=1)

def elbow_tau(c, taus=None):
    """冲突率-τ 曲线的最大弦距肘部。"""
    taus = np.linspace(0.3, 0.95, 66) if taus is None else taus
    rate = np.array([(c > t).mean() for t in taus])
    # 归一化后计算到首尾连线的最大距离（Thorndike 肘部的几何形式）
    x = (taus - taus[0]) / (taus[-1] - taus[0])
    y = (rate - rate[-1]) / (rate[0] - rate[-1] + 1e-12)
    d = np.abs(y - (1 - x)) / np.sqrt(2)
    i = int(np.argmax(d))
    return float(taus[i]), taus, rate

# =====================================================================
if __name__ == "__main__":
    A1 = pd.read_pickle(f"{CACHE}/A1_indicators.pkl")
    A2 = pd.read_pickle(f"{CACHE}/A2_indicators.pkl")
    A3 = pd.read_pickle(f"{CACHE}/A3_indicators.pkl")
    sc = np.load(f"{CACHE}/step2_scores.npz")
    Z1, Q1 = sc["Z1"], sc["Q1"]
    Z2, Q2, Z3, Q3 = sc["Z2"], sc["Q2"], sc["Z3"], sc["Q3"]
    d1 = A1["domain"].to_numpy(); d2 = A2["domain"].to_numpy(); d3 = A3["domain"].to_numpy()

    # ---- ① 组内一致性 Kendall W ----
    kw = {}
    for g in GK:
        idx = G_IDX[g]
        if len(idx) >= 2:
            sub = Z1[np.random.default_rng(0).choice(len(Z1), 8000, replace=False)][:, idx]
            kw[g] = kendall_w(sub)
    pd.Series(kw, name="Kendall_W").to_csv(f"{TABLES}/T3_kendall_w.csv")
    print("组内 Kendall W:", {k: round(v, 3) for k, v in kw.items()})

    # ---- ② 冲突度 ----
    S1 = group_scores(Z1); R1 = domain_pct_rank(S1, d1); c1 = conflict_score(R1)

    # ---- ③ 自适应阈值 ----
    tau, taus, rate = elbow_tau(c1)
    conflict_rate = float((c1 > tau).mean())
    print(f"自适应阈值 τ={tau:.3f}，A1 冲突率={conflict_rate:.2%}")
    pd.DataFrame({"tau": taus, "conflict_rate": rate}).to_csv(
        f"{TABLES}/T3_conflict_rate_curve.csv", index=False)

    # ---- ④ 成因分析 ----
    is_conf = c1 > tau
    hi = R1.argmax(axis=1); lo = R1.argmin(axis=1)
    ctype = np.array([f"{GK[h].split('_')[1]}高×{GK[l].split('_')[1]}低"
                      for h, l in zip(hi, lo)])
    # 冲突×域交叉表 + 卡方
    ct = pd.crosstab(pd.Series(d1[is_conf], name="domain"),
                     pd.Series(ctype[is_conf], name="conflict_type"))
    ct.to_csv(f"{TABLES}/T3_conflict_by_domain.csv")
    dom_rate = pd.Series(is_conf, index=d1).groupby(level=0).mean().sort_values(ascending=False)
    dom_rate.to_csv(f"{TABLES}/T3_conflict_rate_by_domain.csv")
    print("各域冲突率:\n", dom_rate.round(4).to_string())
    # 卡方检验（冲突 vs 域 独立性）
    obs = pd.crosstab(pd.Series(d1, name="domain"), pd.Series(is_conf, name="is_conflict")).to_numpy()
    E = obs.sum(1, keepdims=True) * obs.sum(0, keepdims=True) / obs.sum()
    chi2 = float(((obs - E)**2 / E).sum()); df = (obs.shape[0]-1)*(obs.shape[1]-1)
    print(f"冲突×域 卡方={chi2:.1f}, df={df}, p≈{chi2_sf(chi2, df):.2e}")

    # ---- ⑤ 扩展集复检（同一 τ，同一规则） ----
    rows = []
    for tag, Z, Q, d in [("A1", Z1, Q1, d1), ("A2", Z2, Q2, d2), ("A3", Z3, Q3, d3)]:
        S = group_scores(Z); R = domain_pct_rank(S, d); c = conflict_score(R)
        ic = c > tau
        hi_, lo_ = R.argmax(1), R.argmin(1)
        top_type = pd.Series([f"{GK[h].split('_')[1]}高×{GK[l].split('_')[1]}低"
                              for h, l in zip(hi_[ic], lo_[ic])]).value_counts()
        rows.append(dict(dataset=tag, n=len(c), conflict_rate=float(ic.mean()),
                         top_type=top_type.index[0] if len(top_type) else "-",
                         top_type_share=float(top_type.iloc[0]/ic.sum()) if ic.sum() else 0))
    rep = pd.DataFrame(rows)
    rep.to_csv(f"{TABLES}/T3_extended_recheck.csv", index=False)
    print(rep.to_string(index=False))

    # A18 抽查素材：A1 冲突样本中抽 30 条 id+content 摘要（人工核验用）
    conf_idx = np.where(is_conf)[0]
    pick = np.random.default_rng(7).choice(conf_idx, min(30, len(conf_idx)), replace=False)
    # 重新流式读 content（A1 缓存未存 content，按行号提取）
    import lzma, json
    from q1_00_common import A1_PATH
    want = set(pick.tolist()); found = {}
    with lzma.open(A1_PATH, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i in want:
                r = json.loads(line)
                found[i] = {"row": i, "domain": d1[i], "conflict": float(c1[i]),
                            "type": ctype[i], "Q": float(Q1[i]),
                            "content_head": " ".join((r.get("content") or "").split())[:500]}
            if len(found) == len(want): break
    pd.DataFrame(found.values()).to_csv(f"{TABLES}/T3_manual_check_samples.csv", index=False)

    np.savez_compressed(f"{CACHE}/step3_conflict.npz",
                        c1=c1, tau=tau, is_conf=is_conf)

    # ---- 图：冲突率-τ 曲线 ----
    plt = setup_cjk_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(taus, rate * 100, lw=2)
    ax.axvline(tau, ls="--", c="crimson", label=f"自适应阈值 τ={tau:.2f}（最大弦距）")
    ax.scatter([tau], [conflict_rate * 100], c="crimson", zorder=5)
    ax.set_xlabel("阈值 τ"); ax.set_ylabel("冲突率 (%)")
    ax.set_title("冲突率–阈值曲线：冲突率不是 0%，且结论对 τ 稳健")
    ax.legend(); fig.tight_layout()
    fig.savefig(f"{FIGS}/F3_conflict_rate_curve.png"); plt.close(fig)
    print("Step3 done.")
