# -*- coding: utf-8 -*-
"""
q1_03_conflict_v2.py — Step3 v2：基于六来源的质量指标冲突诊断
设计原则：
  ① 单指标分析为基础，按六个较细来源汇总（RPS、DSIR、轻量分类器、FineWeb-Edu、QuRating、ModernBERT）
  ② 来源内部一致性 + 来源间分歧度，避免指标数量主导
  ③ 区分单指标异常 vs 多来源对立
  ④ A1 自适应阈值，A2/A3 冻结复检
  ⑤ 与原四组方法对比稳定性
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q1_00_common import (CACHE, TABLES, FIGS, IND_COLS, GROUPS,
                          rankdata, kendall_w, spearman, chi2_sf,
                          setup_cjk_matplotlib)

# ==================== 六来源分组定义 ====================
SOURCES_V2 = {
    "S1_RPS规则": [
        "rps_doc_word_count", "rps_doc_num_sentences", "rps_doc_mean_word_length",
        "rps_doc_unigram_entropy", "rps_doc_frac_no_alph_words", "rps_doc_frac_unique_words",
        "rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram",
        "rps_lines_uppercase_letter_fraction",
        "rps_lines_ending_with_terminal_punctution_mark",
        "rps_lines_numerical_chars_fraction"
    ],
    "S2_DSIR": ["dsir_books", "dsir_wiki", "dsir_math"],
    "S3_轻量分类器": ["fluency_en", "ad_en"],
    "S4_FineWeb-Edu": ["fineweb_edu"],
    "S5_QuRating": ["qurater_writing", "qurater_expertise", "qurater_facts", "qurater_edu"],
    "S6_ModernBERT": ["modernbert_professionalism", "modernbert_readability",
                      "modernbert_reasoning", "modernbert_cleanliness"]
}

SK = list(SOURCES_V2.keys())
S_IDX = {s: [IND_COLS.index(c) for c in cols] for s, cols in SOURCES_V2.items()}

# ==================== 核心函数 ====================
def domain_percentile_rank(x, domains):
    """标量值 -> 域内百分位排名 [0,1]"""
    r = np.empty_like(x)
    for d in np.unique(domains):
        m = domains == d
        r[m] = rankdata(x[m]) / m.sum()
    return r

def source_scores_v2(Z, Q, domains):
    """
    25维标准化分 Z + 主分 Q + 域标签 -> 六来源分数
    返回：
      - r_ind: (n, 25) 各指标域内百分位排名
      - r_Q: (n,) 主分Q域内百分位排名
      - r_src: (n, 6) 六来源平均排名
      - delta_ind: (n, 25) 各指标与主分的偏离度
    """
    n = len(Z)
    r_ind = np.empty_like(Z)
    for j in range(25):
        r_ind[:, j] = domain_percentile_rank(Z[:, j], domains)

    r_Q = domain_percentile_rank(Q, domains)
    delta_ind = np.abs(r_ind - r_Q[:, None])

    r_src = np.column_stack([r_ind[:, S_IDX[s]].mean(axis=1) for s in SK])
    return r_ind, r_Q, r_src, delta_ind

def conflict_metrics_v2(r_src):
    """
    六来源排名 -> 三种冲突度指标
    返回：c_range, c_std, c_dir
    """
    c_range = r_src.max(axis=1) - r_src.min(axis=1)
    c_std = r_src.std(axis=1, ddof=0)

    # 方向不一致：存在某来源<0.3且另一来源>0.7
    low = (r_src < 0.3).any(axis=1)
    high = (r_src > 0.7).any(axis=1)
    c_dir = (low & high).astype(float)

    return c_range, c_std, c_dir

def elbow_tau(c, taus=None):
    """冲突率-τ 曲线的最大弦距肘部"""
    taus = np.linspace(0.3, 0.95, 66) if taus is None else taus
    rate = np.array([(c > t).mean() for t in taus])
    x = (taus - taus[0]) / (taus[-1] - taus[0])
    y = (rate - rate[-1]) / (rate[0] - rate[-1] + 1e-12)
    d = np.abs(y - (1 - x)) / np.sqrt(2)
    i = int(np.argmax(d))
    return float(taus[i]), taus, rate

def source_internal_consistency(Z, domains, sample_size=8000, seed=0):
    """
    计算六来源内部一致性：Kendall W, 平均成对Spearman, 指标-来源均值相关
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), min(sample_size, len(Z)), replace=False)
    Z_sub = Z[idx]
    d_sub = domains[idx]

    # 转域内百分位
    r_ind_sub = np.empty_like(Z_sub)
    for j in range(25):
        r_ind_sub[:, j] = domain_percentile_rank(Z_sub[:, j], d_sub)

    results = []
    for s in SK:
        idx_list = S_IDX[s]
        k = len(idx_list)

        if k >= 2:
            # Kendall W
            sub_mat = r_ind_sub[:, idx_list]
            kw = kendall_w(sub_mat)

            # 平均成对Spearman
            pairs = []
            for i in range(k):
                for j in range(i+1, k):
                    rho = spearman(sub_mat[:, i], sub_mat[:, j])
                    if not np.isnan(rho):
                        pairs.append(rho)
            mean_pair_rho = float(np.mean(pairs)) if pairs else np.nan

            # 指标-来源均值相关
            src_mean = sub_mat.mean(axis=1)
            ind_src_corrs = []
            for i in range(k):
                rho = spearman(sub_mat[:, i], src_mean)
                if not np.isnan(rho):
                    ind_src_corrs.append(rho)
            mean_ind_src_corr = float(np.mean(ind_src_corrs)) if ind_src_corrs else np.nan
        else:
            kw = np.nan
            mean_pair_rho = np.nan
            mean_ind_src_corr = np.nan

        results.append({
            "source": s,
            "n_indicators": k,
            "kendall_W": kw,
            "mean_pairwise_spearman": mean_pair_rho,
            "mean_indicator_source_corr": mean_ind_src_corr
        })

    return pd.DataFrame(results)

def single_vs_multi_anomaly(r_ind, r_Q, r_src, tau_ind, tau_src):
    """
    区分单指标异常 vs 多来源对立
    单指标异常：某指标δ>tau_ind，但其所属来源内其他指标δ<tau_ind
    多来源对立：存在两来源|r_s - r_s'|>tau_src
    """
    delta_ind = np.abs(r_ind - r_Q[:, None])
    n = len(delta_ind)

    single_anomaly = np.zeros(n, dtype=bool)
    multi_source = np.zeros(n, dtype=bool)

    for i in range(n):
        # 单指标异常检测
        for s in SK:
            idx_list = S_IDX[s]
            if len(idx_list) > 1:
                for j_pos, j in enumerate(idx_list):
                    if delta_ind[i, j] > tau_ind:
                        # 检查同源其他指标是否都不异常
                        other_idx = [idx_list[k] for k in range(len(idx_list)) if k != j_pos]
                        if all(delta_ind[i, oi] < tau_ind for oi in other_idx):
                            single_anomaly[i] = True
                            break
            if single_anomaly[i]:
                break

        # 多来源对立检测
        for si in range(len(SK)):
            for sj in range(si+1, len(SK)):
                if abs(r_src[i, si] - r_src[i, sj]) > tau_src:
                    multi_source[i] = True
                    break
            if multi_source[i]:
                break

    return single_anomaly, multi_source

# =====================================================================
if __name__ == "__main__":
    print("=== 六来源冲突诊断 v2 ===")

    # ---- 加载数据 ----
    A1 = pd.read_pickle(f"{CACHE}/A1_indicators.pkl")
    A2 = pd.read_pickle(f"{CACHE}/A2_indicators.pkl")
    A3 = pd.read_pickle(f"{CACHE}/A3_indicators.pkl")
    sc = np.load(f"{CACHE}/step2_scores.npz")
    Z1, Q1 = sc["Z1"], sc["Q1"]
    Z2, Q2, Z3, Q3 = sc["Z2"], sc["Q2"], sc["Z3"], sc["Q3"]
    d1 = A1["domain"].to_numpy()
    d2 = A2["domain"].to_numpy()
    d3 = A3["domain"].to_numpy()

    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)

    # ---- ① 来源内部一致性 ----
    print("\n① 计算六来源内部一致性...")
    consist = source_internal_consistency(Z1, d1)
    consist.to_csv(f"{TABLES}/T3_source_internal_consistency.csv", index=False)
    print(consist.to_string(index=False))

    # ---- ② A1 冲突度与阈值 ----
    print("\n② A1 冲突度与阈值确定...")
    r_ind1, r_Q1, r_src1, delta_ind1 = source_scores_v2(Z1, Q1, d1)
    c_range1, c_std1, c_dir1 = conflict_metrics_v2(r_src1)

    # 极差阈值（主阈值）
    tau_range, taus, rate_curve = elbow_tau(c_range1)
    conf_rate_range = float((c_range1 > tau_range).mean())

    # 标准差阈值（辅助）
    tau_std = float(np.percentile(c_std1, 80))
    conf_rate_std = float((c_std1 > tau_std).mean())

    # 方向不一致率
    conf_rate_dir = float(c_dir1.mean())

    # 单指标阈值
    tau_ind = float(np.percentile(delta_ind1, 90))

    metrics_a1 = pd.DataFrame([{
        "n_docs": len(Q1),
        "conflict_rate_range": conf_rate_range,
        "conflict_rate_std": conf_rate_std,
        "conflict_rate_dir": conf_rate_dir,
        "tau_range": tau_range,
        "tau_std": tau_std,
        "tau_ind": tau_ind
    }])
    metrics_a1.to_csv(f"{TABLES}/T3_conflict_metrics_A1.csv", index=False)
    print(f"τ_range={tau_range:.3f}, 冲突率(极差)={conf_rate_range:.2%}")
    print(f"τ_std={tau_std:.3f}, 冲突率(标准差)={conf_rate_std:.2%}")
    print(f"τ_ind={tau_ind:.3f}, 方向不一致率={conf_rate_dir:.2%}")

    # 保存阈值曲线
    pd.DataFrame({"tau": taus, "conflict_rate": rate_curve}).to_csv(
        f"{TABLES}/T3_conflict_rate_curve_6sources.csv", index=False)

    # ---- ③ A1 冲突类型分布（按域和来源配对） ----
    print("\n③ A1 冲突类型分布...")
    is_conf1 = c_range1 > tau_range
    hi1 = r_src1.argmax(axis=1)
    lo1 = r_src1.argmin(axis=1)

    conflict_rows = []
    for d in np.unique(d1):
        m = (d1 == d) & is_conf1
        if m.sum() == 0:
            continue
        pairs = [(SK[hi1[i]].split('_')[1], SK[lo1[i]].split('_')[1])
                 for i in np.where(m)[0]]
        pair_counts = pd.Series(pairs).value_counts()
        for (sh, sl), cnt in pair_counts.items():
            conflict_rows.append({
                "domain": d,
                "source_high": sh,
                "source_low": sl,
                "count": int(cnt),
                "share": float(cnt / m.sum())
            })

    if conflict_rows:
        pd.DataFrame(conflict_rows).to_csv(
            f"{TABLES}/T3_conflict_by_domain_source.csv", index=False)

    # 各域冲突率
    dom_rate1 = pd.Series(is_conf1, index=d1).groupby(level=0).mean().sort_values(ascending=False)
    dom_rate1.to_csv(f"{TABLES}/T3_conflict_rate_by_domain_v2.csv")
    print("各域冲突率(极差):\n", dom_rate1.round(4).to_string())

    # ---- ④ 单指标异常 vs 多来源对立 ----
    print("\n④ 单指标异常 vs 多来源对立...")
    single1, multi1 = single_vs_multi_anomaly(r_ind1, r_Q1, r_src1, tau_ind, tau_range)
    both1 = single1 & multi1
    neither1 = ~(single1 | multi1)

    anom_stats = []
    for tag, single, multi, both, neither in [
        ("A1", single1, multi1, both1, neither1)
    ]:
        anom_stats.append({
            "dataset": tag,
            "n_single_anomaly": int(single.sum()),
            "n_multi_source": int(multi.sum()),
            "n_both": int(both.sum()),
            "n_neither": int(neither.sum())
        })

    # ---- ⑤ A2/A3 复检 ----
    print("\n⑤ A2/A3 冻结复检...")
    recheck_rows = []
    for tag, Z, Q, d in [("A1", Z1, Q1, d1), ("A2", Z2, Q2, d2), ("A3", Z3, Q3, d3)]:
        r_ind, r_Q, r_src, delta_ind = source_scores_v2(Z, Q, d)
        c_range, c_std, c_dir = conflict_metrics_v2(r_src)

        ic_range = c_range > tau_range
        ic_std = c_std > tau_std

        hi_ = r_src.argmax(1)
        lo_ = r_src.argmin(1)

        pairs = [(SK[hi_[i]].split('_')[1], SK[lo_[i]].split('_')[1])
                 for i in np.where(ic_range)[0]]
        if pairs:
            top_pair = pd.Series(pairs).value_counts()
            top_pair_name = f"{top_pair.index[0][0]}高×{top_pair.index[0][1]}低"
            top_pair_share = float(top_pair.iloc[0] / len(pairs))
        else:
            top_pair_name = "-"
            top_pair_share = 0.0

        recheck_rows.append({
            "dataset": tag,
            "n": len(Q),
            "conflict_rate_range": float(ic_range.mean()),
            "conflict_rate_std": float(ic_std.mean()),
            "top_source_pair": top_pair_name,
            "top_pair_share": top_pair_share
        })

        # 补充单指标异常统计
        single, multi = single_vs_multi_anomaly(r_ind, r_Q, r_src, tau_ind, tau_range)
        both = single & multi
        neither = ~(single | multi)
        anom_stats.append({
            "dataset": tag,
            "n_single_anomaly": int(single.sum()),
            "n_multi_source": int(multi.sum()),
            "n_both": int(both.sum()),
            "n_neither": int(neither.sum())
        })

    pd.DataFrame(recheck_rows).to_csv(f"{TABLES}/T3_extended_recheck_6sources.csv", index=False)
    pd.DataFrame(anom_stats).to_csv(f"{TABLES}/T3_single_vs_multi_anomaly.csv", index=False)
    print(pd.DataFrame(recheck_rows).to_string(index=False))

    # ---- ⑥ 与原四组方法对比 ----
    print("\n⑥ 与原四组方法对比...")
    # 加载旧四组冲突结果
    old_conf = np.load(f"{CACHE}/step3_conflict.npz")
    is_conf_old = old_conf["is_conf"]

    # 文档级对比
    from sklearn.metrics import cohen_kappa_score
    overlap = (is_conf_old & is_conf1).sum() / (is_conf_old | is_conf1).sum()
    kappa = cohen_kappa_score(is_conf_old, is_conf1)

    # 冲突文档的Q分布 KS检验
    from q1_00_common import ks_2samp
    D_ks, p_ks = ks_2samp(Q1[is_conf1], Q1[~is_conf1])

    old_vs_new = []
    old_rate = float(is_conf_old.mean())
    new_rate = float(is_conf1.mean())
    old_vs_new.append({
        "dataset": "A1",
        "old_conflict_rate": old_rate,
        "new_conflict_rate": new_rate,
        "overlap_jaccard": float(overlap),
        "cohens_kappa": float(kappa),
        "KS_p_value": float(p_ks)
    })

    pd.DataFrame(old_vs_new).to_csv(f"{TABLES}/T3_old_vs_new_conflict.csv", index=False)
    print(f"旧四组冲突率={old_rate:.2%}, 新六来源冲突率={new_rate:.2%}")
    print(f"Jaccard重叠={overlap:.3f}, Cohen's Kappa={kappa:.3f}, KS p={p_ks:.3f}")

    # ---- ⑦ 阈值敏感性 ----
    print("\n⑦ 阈值敏感性分析...")
    sens_rows = []
    for shift in [-0.10, -0.05, 0.00, 0.05, 0.10]:
        tau_test = tau_range + shift
        for tag, c_range, n in [("A1", c_range1, len(Q1))]:
            rate = float((c_range > tau_test).mean())
            sens_rows.append({
                "dataset": tag,
                "tau_shift": shift,
                "conflict_rate_range": rate
            })
    pd.DataFrame(sens_rows).to_csv(f"{TABLES}/T3_threshold_sensitivity.csv", index=False)

    # ---- 保存核心结果 ----
    np.savez_compressed(f"{CACHE}/step3_conflict_v2.npz",
                        c_range1=c_range1, c_std1=c_std1, c_dir1=c_dir1,
                        tau_range=tau_range, tau_std=tau_std, tau_ind=tau_ind,
                        is_conf1=is_conf1, r_src1=r_src1, r_ind1=r_ind1)

    # ---- 图：冲突率-阈值曲线 ----
    print("\n⑧ 生成图表...")
    plt = setup_cjk_matplotlib()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(taus, rate_curve * 100, lw=2, label="六来源极差冲突率")
    ax.axvline(tau_range, ls="--", c="crimson", label=f"肘部阈值 τ={tau_range:.2f}")
    ax.scatter([tau_range], [conf_rate_range * 100], c="crimson", s=80, zorder=5)
    ax.set_xlabel("阈值 τ")
    ax.set_ylabel("冲突率 (%)")
    ax.set_title("六来源冲突率曲线（最大弦距肘部法）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/F3_conflict_rate_curve_6sources.png", dpi=150)
    plt.close(fig)

    # 图：来源一致性热力图（成对相关）
    fig, ax = plt.subplots(figsize=(10, 8))
    # 计算25指标的成对Spearman相关
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Z1), min(8000, len(Z1)), replace=False)
    r_ind_sub = np.empty((len(idx), 25))
    for j in range(25):
        r_ind_sub[:, j] = domain_percentile_rank(Z1[idx, j], d1[idx])

    corr_mat = np.full((25, 25), np.nan)
    for i in range(25):
        for j in range(25):
            corr_mat[i, j] = spearman(r_ind_sub[:, i], r_ind_sub[:, j])

    im = ax.imshow(corr_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(25))
    ax.set_yticks(range(25))
    ax.set_xticklabels([c[:15] for c in IND_COLS], rotation=90, fontsize=7)
    ax.set_yticklabels([c[:15] for c in IND_COLS], fontsize=7)
    ax.set_title("25指标域内百分位排名成对Spearman相关（A1样本）")
    fig.colorbar(im, ax=ax)

    # 分块边界线（六来源）
    boundaries = np.cumsum([len(S_IDX[s]) for s in SK[:-1]]) - 0.5
    for b in boundaries:
        ax.axhline(b, color="black", lw=1.5)
        ax.axvline(b, color="black", lw=1.5)

    fig.tight_layout()
    fig.savefig(f"{FIGS}/F3_source_consistency_heatmap.png", dpi=150)
    plt.close(fig)

    # 图：冲突文档vs非冲突文档的Q分布
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(Q1[is_conf1], bins=50, alpha=0.6, label="冲突文档", density=True, color="red")
    ax.hist(Q1[~is_conf1], bins=50, alpha=0.6, label="非冲突文档", density=True, color="blue")
    ax.set_xlabel("主质量分 Q")
    ax.set_ylabel("密度")
    ax.set_title(f"冲突文档 vs 非冲突文档的Q分布（KS p={p_ks:.3f}）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/F3_Q_distribution_by_conflict.png", dpi=150)
    plt.close(fig)

    # 图：六来源分歧分布（箱线图）
    fig, ax = plt.subplots(figsize=(8, 5))
    data_conf = [r_src1[is_conf1, i] for i in range(6)]
    data_non = [r_src1[~is_conf1, i] for i in range(6)]

    positions = np.arange(6) * 2
    bp1 = ax.boxplot(data_conf, positions=positions - 0.35, widths=0.6,
                     patch_artist=True, boxprops=dict(facecolor="red", alpha=0.6),
                     medianprops=dict(color="darkred", linewidth=2))
    bp2 = ax.boxplot(data_non, positions=positions + 0.35, widths=0.6,
                     patch_artist=True, boxprops=dict(facecolor="blue", alpha=0.6),
                     medianprops=dict(color="darkblue", linewidth=2))

    ax.set_xticks(positions)
    ax.set_xticklabels([s.split('_')[1] for s in SK], rotation=45, ha="right")
    ax.set_ylabel("来源域内百分位排名")
    ax.set_title("六来源分数分布：冲突文档 vs 非冲突文档")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["冲突", "非冲突"])
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{FIGS}/F3_source_disagreement_distribution.png", dpi=150)
    plt.close(fig)

    print("\n✓ Step3 v2 完成。六来源冲突诊断结果已保存。")
