# -*- coding: utf-8 -*-
"""Step3: 在固定主 Q 之后进行单指标、六来源冲突候选诊断。

所有经验分布、阈值仅由 A1 拟合并冻结用于 A2/A3。诊断不修改 Q、权重或任何 P1 接口。
原四组方法按其历史规则完整复算，但输出明确标为 legacy candidate。
"""
import os
import sys
import json
import lzma
import pickle
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q1_00_common import (CACHE, TABLES, FIGS, A1_PATH, IND_COLS, GROUPS,
                          RPS_FIELDS, DSIR_FIELDS, QURATER_DIMS, MB_FIELDS,
                          rankdata, kendall_w, setup_cjk_matplotlib)

SOURCES = {
    "RPS": RPS_FIELDS,
    "DSIR": DSIR_FIELDS,
    "轻量分类器": ["fluency_en", "ad_en"],
    "FineWeb-Edu": ["fineweb_edu"],
    "QuRating": QURATER_DIMS,
    "ModernBERT": MB_FIELDS,
}
SNAMES = list(SOURCES)
SIDX = {s: [IND_COLS.index(c) for c in cols] for s, cols in SOURCES.items()}
assert sorted(j for idx in SIDX.values() for j in idx) == list(range(25))
SOURCE_TO_LEGACY = {"RPS": "规则型", "DSIR": "DSIR", "轻量分类器": "轻量分类器",
                    "FineWeb-Edu": "模型评分器", "QuRating": "模型评分器", "ModernBERT": "模型评分器"}
LEGACY_NAMES = list(GROUPS)
LIDX = {s: [IND_COLS.index(c) for c in cols] for s, cols in GROUPS.items()}
QS = (0.90, 0.95, 0.975)  # 仅是高分歧候选尾部的操作点，绝非真值率
STANCE_TAIL = 0.20        # A1 来源中心百分位的两端 20%；另做敏感性对照


def mid_ecdf(x, sorted_ref):
    """A1 经验分布中的中秩百分位；大量缩尾并列值按 (left+right)/2 处理。"""
    return (np.searchsorted(sorted_ref, x, "left")
            + np.searchsorted(sorted_ref, x, "right")) / (2 * len(sorted_ref))


def fit_reference(Z1, d1):
    return {(d, j): np.sort(Z1[d1 == d, j])
            for d in np.unique(d1) for j in range(len(IND_COLS))}


def indicator_ranks(Z, domains, ref):
    R = np.empty_like(Z)
    for d in np.unique(domains):
        if (d, 0) not in ref:
            raise ValueError(f"A1 未见过的质量域 {d}，不能自适应重建参照")
        mask = domains == d
        for j in range(len(IND_COLS)):
            R[mask, j] = mid_ecdf(Z[mask, j], ref[d, j])
    return R


def source_medians(R):
    return np.column_stack([np.median(R[:, SIDX[s]], axis=1) for s in SNAMES])


def source_internal_span(R):
    """来源内 P90-P10；单指标 FineWeb-Edu 无内部跨度，记 NA 而非零。"""
    out = np.full((len(R), len(SNAMES)), np.nan)
    for g, s in enumerate(SNAMES):
        if len(SIDX[s]) >= 2:
            v = R[:, SIDX[s]]
            out[:, g] = np.quantile(v, .9, axis=1) - np.quantile(v, .1, axis=1)
    return out


def source_internal_opposition(R):
    """同一来源内同时存在 A1 参照下低端和高端指标；单指标来源不定义。"""
    out = np.full((len(R), len(SNAMES)), np.nan)
    for g, s in enumerate(SNAMES):
        if len(SIDX[s]) >= 2:
            v = R[:, SIDX[s]]
            out[:, g] = ((v <= STANCE_TAIL).any(1) &
                         (v >= 1 - STANCE_TAIL).any(1)).astype(float)
    return out


def leave_one_out_deviation(R):
    """每指标相对于同来源其余指标中位数的偏离；来源规模 <3 无法唯一归因。"""
    out = np.full_like(R, np.nan)
    for s, idx in SIDX.items():
        if len(idx) >= 3:
            for j in idx:
                peers = [k for k in idx if k != j]
                out[:, j] = np.abs(R[:, j] - np.median(R[:, peers], axis=1))
    return out


def source_outlier_stats(dev):
    """各来源最大单指标偏离及其指标；相同最大值的并列不归为单指标。"""
    val = np.full((len(dev), len(SNAMES)), np.nan)
    idx_out = np.full((len(dev), len(SNAMES)), -1, dtype=int)
    unique = np.zeros((len(dev), len(SNAMES)), dtype=bool)
    for g, s in enumerate(SNAMES):
        if len(SIDX[s]) >= 3:
            sub = dev[:, SIDX[s]]
            k = np.argmax(sub, axis=1)
            mx = sub[np.arange(len(sub)), k]
            val[:, g] = mx
            idx_out[:, g] = np.asarray(SIDX[s])[k]
            unique[:, g] = (np.isclose(sub, mx[:, None], rtol=0, atol=1e-12).sum(1) == 1)
    return val, idx_out, unique


def fit_six_thresholds(R1, d1):
    M1 = source_medians(R1)
    center_ref = {(d, g): np.sort(M1[d1 == d, g])
                  for d in np.unique(d1) for g in range(len(SNAMES))}
    U1 = source_center_percentiles(M1, d1, center_ref)
    B1 = U1.max(1) - U1.min(1)
    span1 = source_internal_span(R1)
    dev1 = leave_one_out_deviation(R1)
    out1, _, _ = source_outlier_stats(dev1)
    between = {(d, q): float(np.quantile(B1[d1 == d], q))
               for d in np.unique(d1) for q in QS}
    internal = {(d, g): float(np.nanquantile(span1[d1 == d, g], .95))
                for d in np.unique(d1) for g, s in enumerate(SNAMES)
                if len(SIDX[s]) >= 2}
    single = {(d, g): float(np.nanquantile(out1[d1 == d, g], .95))
              for d in np.unique(d1) for g, s in enumerate(SNAMES)
              if len(SIDX[s]) >= 3}
    indicator_deviation = {(d, j): float(np.nanquantile(dev1[d1 == d, j], .95))
                           for d in np.unique(d1) for s in SNAMES if len(SIDX[s]) >= 3
                           for j in SIDX[s]}
    return dict(center_ref=center_ref, between=between, internal=internal,
                single=single, indicator_deviation=indicator_deviation, indicator_ref=None)


def source_center_percentiles(M, domains, ref):
    U = np.empty_like(M)
    for d in np.unique(domains):
        mask = domains == d
        for g in range(len(SNAMES)):
            U[mask, g] = mid_ecdf(M[mask, g], ref[d, g])
    return U


def diagnose_six(R, domains, fit):
    M = source_medians(R)
    U = source_center_percentiles(M, domains, fit["center_ref"])
    B = U.max(1) - U.min(1)
    span = source_internal_span(R)
    internal_opposition = source_internal_opposition(R)
    dev = leave_one_out_deviation(R)
    out, out_j, unique = source_outlier_stats(dev)
    peer_flag = np.zeros_like(R, dtype=bool)
    cutoff = np.array([fit["between"][d, .95] for d in domains])
    between = B > cutoff
    high = U >= 1 - STANCE_TAIL
    low = U <= STANCE_TAIL
    nh, nl = high.sum(1), low.sum(1)
    multi = between & (nh >= 2) & (nl >= 2)
    sparse = between & (nh >= 1) & (nl >= 1) & ~multi
    other_between = between & ~(multi | sparse)
    internal = np.zeros_like(span, dtype=bool)
    outlier = np.zeros_like(out, dtype=bool)
    for g, s in enumerate(SNAMES):
        if len(SIDX[s]) >= 2:
            internal[:, g] = span[:, g] > np.array([fit["internal"][d, g] for d in domains])
        if len(SIDX[s]) >= 3:
            for j in SIDX[s]:
                peer_flag[:, j] = dev[:, j] > np.array(
                    [fit["indicator_deviation"][d, j] for d in domains])
            outlier[:, g] = ((out[:, g] > np.array([fit["single"][d, g] for d in domains]))
                             & unique[:, g] & (peer_flag[:, SIDX[s]].sum(1) == 1)
                             & peer_flag[np.arange(len(R)), out_j[:, g]])
    # 被该唯一偏离指标撑大的本来源跨度允许存在；别的来源若也高分散则不算“独有”。
    single_only = (outlier.sum(1) == 1) & ~between & ~(internal & ~outlier).any(1)
    return dict(R=R, M=M, U=U, B=B, span=span, dev=dev, out=out,
                out_j=out_j, outlier=outlier, peer_flag=peer_flag, internal=internal,
                internal_opposition=internal_opposition,
                between=between, multi=multi, sparse=sparse,
                other_between=other_between, single_only=single_only,
                high=high, low=low, nh=nh, nl=nl)


def legacy_group_score(Z):
    return np.column_stack([np.median(Z[:, LIDX[s]], axis=1) for s in LEGACY_NAMES])


def legacy_domain_rank(S, domains):
    R = np.empty_like(S)
    for d in np.unique(domains):
        m = domains == d
        for k in range(S.shape[1]):
            R[m, k] = rankdata(S[m, k]) / m.sum()
    return R


def legacy_elbow(c):
    taus = np.linspace(.3, .95, 66)
    rate = np.array([(c > t).mean() for t in taus])
    x = (taus - taus[0]) / (taus[-1] - taus[0])
    y = (rate - rate[-1]) / (rate[0] - rate[-1] + 1e-12)
    gap = np.abs(y - (1-x)) / np.sqrt(2)
    return float(taus[np.argmax(gap)]), taus, rate


def legacy_diagnose(Z, domains, group_ref, matched_cutoff, tau):
    S = legacy_group_score(Z)
    old_R = legacy_domain_rank(S, domains)  # 历史口径：A2/A3 各自重建域内秩
    frozen_R = np.empty_like(S)             # 公平对照：四组也冻结 A1 经验参照
    for d in np.unique(domains):
        m = domains == d
        for k in range(S.shape[1]):
            frozen_R[m, k] = mid_ecdf(S[m, k], group_ref[d, k])
    c_old = old_R.max(1)-old_R.min(1)
    c_frozen = frozen_R.max(1)-frozen_R.min(1)
    hi, lo = old_R.argmax(1), old_R.argmin(1)
    types = np.array([f"{LEGACY_NAMES[h].split('_')[1]}高×{LEGACY_NAMES[l].split('_')[1]}低"
                      for h, l in zip(hi, lo)])
    matched = c_frozen > np.array([matched_cutoff[d] for d in domains])
    return dict(c_old=c_old, c_frozen=c_frozen, old=c_old>tau,
                frozen=c_frozen>tau, matched=matched, types=types)


def rank_correlation(a, b):
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def ratio(mask):
    return float(np.mean(mask))


def cohen_kappa(a, b):
    pa, pb = ratio(a), ratio(b)
    observed = ratio(a == b)
    expected = pa * pb + (1 - pa) * (1 - pb)
    return float((observed - expected) / (1 - expected)) if expected < 1 else np.nan


def export_tables(data, fit, old_tau, old_curve):
    d1 = data["A1"]["domains"]
    # A1 精确参照存二进制缓存；此表给出人可读的分位端点与冻结阈值。
    refs = []
    for d in np.unique(d1):
        for j, name in enumerate(IND_COLS):
            z = fit["indicator_ref"][d, j]
            refs.append(dict(domain=d, kind="indicator", name=name, n_A1=len(z),
                             p10=np.quantile(z,.1), p50=np.quantile(z,.5), p90=np.quantile(z,.9),
                             cutoff_p90=np.nan, cutoff_p95=np.nan, cutoff_p975=np.nan))
            if (d, j) in fit["indicator_deviation"]:
                refs.append(dict(domain=d, kind="indicator_LOO_deviation", name=name, n_A1=len(z),
                                 p10=np.nan, p50=np.nan, p90=np.nan, cutoff_p90=np.nan,
                                 cutoff_p95=fit["indicator_deviation"][d,j], cutoff_p975=np.nan))
        refs.append(dict(domain=d, kind="between_six_sources", name="B=max U-min U",
                         n_A1=int((d1==d).sum()), p10=np.nan, p50=np.nan, p90=np.nan,
                         cutoff_p90=fit["between"][d,.9], cutoff_p95=fit["between"][d,.95],
                         cutoff_p975=fit["between"][d,.975]))
        refs.append(dict(domain=d, kind="legacy_four_matched_tail", name="c_four_A1_frozen",
                         n_A1=int((d1==d).sum()), p10=np.nan, p50=np.nan, p90=np.nan,
                         cutoff_p90=np.nan, cutoff_p95=fit["legacy_matched_cutoff"][d],
                         cutoff_p975=np.nan))
        for g, s in enumerate(SNAMES):
            center = fit["center_ref"][d, g]
            refs.append(dict(domain=d, kind="source_center", name=s,
                             n_A1=len(center),p10=np.quantile(center,.1),
                             p50=np.quantile(center,.5),p90=np.quantile(center,.9),
                             cutoff_p90=np.nan,cutoff_p95=np.nan,cutoff_p975=np.nan))
            refs.append(dict(domain=d, kind="source_internal_span", name=s,
                             n_A1=int((d1==d).sum()), p10=np.nan,p50=np.nan,p90=np.nan,
                             cutoff_p90=np.nan, cutoff_p95=fit["internal"].get((d,g),np.nan), cutoff_p975=np.nan))
            refs.append(dict(domain=d, kind="source_max_single_deviation", name=s,
                             n_A1=int((d1==d).sum()), p10=np.nan,p50=np.nan,p90=np.nan,
                             cutoff_p90=np.nan, cutoff_p95=fit["single"].get((d,g),np.nan), cutoff_p975=np.nan))
    pd.DataFrame(refs).to_csv(f"{TABLES}/T3_A1_frozen_reference.csv",index=False)

    rng = np.random.default_rng(0)
    sub = rng.choice(len(d1), min(8000,len(d1)),replace=False)
    W = {s: kendall_w(data["A1"]["six"]["R"][sub][:, SIDX[s]]) if len(SIDX[s])>=2 else np.nan
         for s in SNAMES}
    oldW_rank = {s: kendall_w(data["A1"]["six"]["R"][sub][:, LIDX[s]]) if len(LIDX[s])>=2 else np.nan
                 for s in LEGACY_NAMES}
    oldW_Z = {s: kendall_w(data["A1"]["Z"][sub][:, LIDX[s]]) if len(LIDX[s])>=2 else np.nan
              for s in LEGACY_NAMES}
    pd.DataFrame([dict(source=s,n_indicators=len(SIDX[s]),kendall_W_A1_rank_8000=W[s])
                  for s in SNAMES]).to_csv(f"{TABLES}/T3_six_source_consistency.csv",index=False)
    pd.DataFrame([dict(legacy_group=s,n_indicators=len(LIDX[s]),
                       kendall_W_A1_rank_8000=oldW_rank[s],kendall_W_A1_legacy_Z_8000=oldW_Z[s])
                  for s in LEGACY_NAMES]).to_csv(f"{TABLES}/T3_legacy_four_consistency.csv",index=False)

    summary=[]; source_rows=[]; indicator_rows=[]; comparison=[]; types=[]; full_types=[]; pairs=[]; sensitivity=[]
    for tag, entry in data.items():
        d=entry["domains"]; now=entry["six"]; old=entry["legacy"]
        scopes=[("ALL",np.ones(len(d),dtype=bool))]+[(dom,d==dom) for dom in np.unique(d)]
        for dom, mask in scopes:
            n=int(mask.sum()); b=now["between"][mask]; oldb=old["old"][mask]
            newb=now["B"][mask]; multi=now["multi"][mask]
            shared=oldb&b; union=oldb|b
            summary.append(dict(dataset=tag,domain=dom,n=n,
                                high_disagreement_candidate_rate=ratio(b),
                                multi_source_opposition_rate=ratio(multi),
                                sparse_source_opposition_rate=ratio(now["sparse"][mask]),
                                other_high_disagreement_rate=ratio(now["other_between"][mask]),
                                any_internal_split_candidate_rate=ratio(now["internal"][mask].any(1)),
                                any_internal_opposition_rate=ratio(np.nan_to_num(now["internal_opposition"][mask]).any(1)),
                                single_indicator_only_rate=ratio(now["single_only"][mask]),
                                legacy_four_candidate_rate=ratio(oldb),
                                legacy_four_A1_frozen_candidate_rate=ratio(old["frozen"][mask]),
                                legacy_four_matched_tail_candidate_rate=ratio(old["matched"][mask])))
            comparison.append(dict(dataset=tag,domain=dom,n=n,
                                   old_only_rate=ratio(oldb&~b),shared_rate=ratio(shared),
                                   new_only_rate=ratio(b&~oldb),jaccard=float(shared.sum()/union.sum()) if union.any() else np.nan,
                                   kappa=cohen_kappa(oldb,b),
                                   spearman_old_score_vs_B=rank_correlation(old["c_old"][mask],newb),
                                   frozen_four_rate=ratio(old["frozen"][mask]),
                                   frozen_four_jaccard=float((old["frozen"][mask]&b).sum()/(old["frozen"][mask]|b).sum())
                                   if (old["frozen"][mask]|b).any() else np.nan,
                                   matched_four_rate=ratio(old["matched"][mask]),
                                   matched_four_shared_rate=ratio(old["matched"][mask]&b),
                                   matched_four_only_rate=ratio(old["matched"][mask]&~b),
                                   matched_six_only_rate=ratio(b&~old["matched"][mask]),
                                   matched_four_kappa=cohen_kappa(old["matched"][mask],b),
                                   matched_four_jaccard=float((old["matched"][mask]&b).sum()/(old["matched"][mask]|b).sum())
                                   if (old["matched"][mask]|b).any() else np.nan))
            for g,s in enumerate(SNAMES):
                values=now["span"][mask,g]
                source_rows.append(dict(dataset=tag,domain=dom,source=s,n_indicators=len(SIDX[s]),n=n,
                                        median_internal_span=float(np.nanmedian(values)) if len(SIDX[s])>=2 else np.nan,
                                        p90_internal_span=float(np.nanquantile(values,.9)) if len(SIDX[s])>=2 else np.nan,
                                        internal_split_candidate_rate=ratio(now["internal"][mask,g]) if len(SIDX[s])>=2 else np.nan,
                                        within_source_opposition_rate=ratio(now["internal_opposition"][mask,g].astype(bool)) if len(SIDX[s])>=2 else np.nan,
                                        single_indicator_source_candidate_rate=ratio(now["outlier"][mask,g]) if len(SIDX[s])>=3 else np.nan,
                                        kendall_W_A1_rank_8000=W[s]))
            for j,name in enumerate(IND_COLS):
                source=next(s for s,idx in SIDX.items() if j in idx)
                r=now["R"][mask,j]; dev=now["dev"][mask,j]
                chosen=np.zeros(len(now["R"]),dtype=bool)
                g=SNAMES.index(source)
                if len(SIDX[source])>=3:
                    chosen=(now["outlier"][:,g] & (now["out_j"][:,g]==j))
                indicator_rows.append(dict(dataset=tag,domain=dom,source=source,indicator=name,n=n,
                                           rank_median=float(np.median(r)),rank_p10=float(np.quantile(r,.1)),
                                           rank_p90=float(np.quantile(r,.9)),
                                           low_tail_rate=ratio(r<=.2),high_tail_rate=ratio(r>=.8),
                                           median_LOO_deviation=float(np.nanmedian(dev)) if len(SIDX[source])>=3 else np.nan,
                                           indicator_LOO_candidate_rate=ratio(now["peer_flag"][mask,j]) if len(SIDX[source])>=3 else np.nan,
                                           selected_source_outlier_rate=ratio(chosen[mask]) if len(SIDX[source])>=3 else np.nan))
            vc=pd.Series(old["types"][mask&old["old"]]).value_counts()
            types.append(dict(dataset=tag,domain=dom,old_top_type=vc.index[0] if len(vc) else "-",
                              old_top_type_share=float(vc.iloc[0]/vc.sum()) if len(vc) else 0))
            for h in LEGACY_NAMES:
                for l in LEGACY_NAMES:
                    if h == l: continue
                    typ=f"{h.split('_')[1]}高×{l.split('_')[1]}低"
                    count=int(vc.get(typ,0))
                    full_types.append(dict(dataset=tag,domain=dom,legacy_type=typ,count=count,
                                           share_among_legacy_candidates=float(count/vc.sum()) if vc.sum() else 0))
        # 每篇文件的高/低来源配对总权重为 1；来源多者不会多算文件数。
        H=now["high"] & now["between"][:,None]
        L=now["low"] & now["between"][:,None]
        den=np.maximum(H.sum(1)*L.sum(1),1)
        for dom, mask in scopes:
            for gh,sh in enumerate(SNAMES):
                for gl,sl in enumerate(SNAMES):
                    if gh==gl:continue
                    weight=float(np.sum(((H[:,gh]&L[:,gl])/den)[mask]))
                    pairs.append(dict(dataset=tag,domain=dom,high_source=sh,low_source=sl,
                                      weighted_documents=weight,
                                      share_of_between_candidates=weight/max(now["between"][mask].sum(),1)))
        for q in QS:
            thresh=np.array([fit["between"][dom,q] for dom in d])
            for tail in (.10,.20,.25):
                h=(now["U"]>=1-tail).sum(1);l=(now["U"]<=tail).sum(1)
                candidate=now["B"]>thresh
                sensitivity.append(dict(dataset=tag,A1_tail_quantile=q,stance_tail=tail,
                                        high_disagreement_candidate_rate=ratio(candidate),
                                        multi_source_opposition_rate=ratio(candidate&(h>=2)&(l>=2))))
    pd.DataFrame(summary).to_csv(f"{TABLES}/T3_six_source_candidate_summary.csv",index=False)
    pd.DataFrame(source_rows).to_csv(f"{TABLES}/T3_source_internal.csv",index=False)
    pd.DataFrame(indicator_rows).to_csv(f"{TABLES}/T3_indicator_diagnostics.csv",index=False)
    pd.DataFrame(comparison).to_csv(f"{TABLES}/T3_legacy_stability.csv",index=False)
    pd.DataFrame(types).to_csv(f"{TABLES}/T3_legacy_four_types.csv",index=False)
    old_types = pd.DataFrame(full_types)
    old_types.to_csv(f"{TABLES}/T3_legacy_four_type_by_domain.csv",index=False)
    pair_table = pd.DataFrame(pairs)
    pair_table.to_csv(f"{TABLES}/T3_source_opposition_pairs.csv",index=False)
    pair_table["legacy_direction"] = [
        ("模型评分器内部" if SOURCE_TO_LEGACY[h] == SOURCE_TO_LEGACY[l]
         else f"{SOURCE_TO_LEGACY[h]}高×{SOURCE_TO_LEGACY[l]}低")
        for h,l in zip(pair_table.high_source,pair_table.low_source)]
    mapped = pair_table.groupby(["dataset","domain","legacy_direction"],as_index=False)[
        "share_of_between_candidates"].sum().rename(
            columns={"share_of_between_candidates":"new_weighted_share"})
    mapped = mapped.merge(old_types[["dataset","domain","legacy_type","share_among_legacy_candidates"]],
                          how="outer",left_on=["dataset","domain","legacy_direction"],
                          right_on=["dataset","domain","legacy_type"])
    mapped["direction"] = mapped.legacy_direction.fillna(mapped.legacy_type)
    mapped["new_weighted_share"] = mapped.new_weighted_share.fillna(0)
    mapped["legacy_four_share"] = mapped.share_among_legacy_candidates.fillna(0)
    mapped["within_old_model_group"] = mapped.direction.eq("模型评分器内部")
    mapped[["dataset","domain","direction","new_weighted_share","legacy_four_share",
            "within_old_model_group"]].to_csv(f"{TABLES}/T3_direction_stability.csv",index=False)
    pd.DataFrame(sensitivity).to_csv(f"{TABLES}/T3_threshold_sensitivity.csv",index=False)
    tau, taus, rate=old_curve
    pd.DataFrame(dict(tau=taus,legacy_four_candidate_rate=rate)).to_csv(
        f"{TABLES}/T3_legacy_four_curve.csv",index=False)
    pd.DataFrame([dict(dataset=tag,n=len(e["domains"]),legacy_four_candidate_rate=ratio(e["legacy"]["old"]),
                       legacy_top_type=pd.Series(e["legacy"]["types"][e["legacy"]["old"]]).value_counts().index[0])
                  for tag,e in data.items()]).to_csv(f"{TABLES}/T3_legacy_four_recheck.csv",index=False)
    return pd.DataFrame(summary),pd.DataFrame(comparison),pd.DataFrame(source_rows),pd.DataFrame(pairs),pd.DataFrame(sensitivity)


def export_manual(data):
    e=data["A1"];n=e["six"];old=e["legacy"];d=e["domains"]
    categories={
        "多来源对立":n["multi"],
        "单指标偏离且无来源对立":n["single_only"],
        "新六来源独有":n["between"]&~old["old"],
        "旧四组独有":old["old"]&~n["between"],
    }
    rng=np.random.default_rng(7); selected=set(); tags={}
    for label,mask in categories.items():
        available=np.array([i for i in np.where(mask)[0] if i not in selected])
        if not len(available):continue
        pick=rng.choice(available,min(8,len(available)),replace=False)
        for i in pick:tags[int(i)]=label
        selected.update(int(i) for i in pick)
    found=[]
    with lzma.open(A1_PATH,"rt",encoding="utf-8",errors="replace") as f:
        for i,line in enumerate(f):
            if i not in tags:continue
            raw=json.loads(line)
            hi=[SNAMES[g] for g in np.where(n["high"][i])[0]]
            lo=[SNAMES[g] for g in np.where(n["low"][i])[0]]
            flagged=[IND_COLS[n["out_j"][i,g]] for g in range(len(SNAMES)) if n["outlier"][i,g]]
            found.append(dict(row=i,domain=d[i],category=tags[i],Q=float(e["Q"][i]),
                              between_B=float(n["B"][i]),legacy_four_candidate=bool(old["old"][i]),
                              high_sources=";".join(hi),low_sources=";".join(lo),
                              isolated_indicator_candidates=";".join(flagged),
                              content_head=" ".join((raw.get("content") or "").split())[:500]))
            if len(found)==len(tags):break
    full = pd.DataFrame(found).sort_values("row")
    full.to_csv(f"{CACHE}/step3_manual_samples_with_text.csv",index=False)
    # 跟踪表只含行号与诊断元数据；附件原文摘要仅在被忽略的本地缓存中。
    full.drop(columns=["content_head"]).to_csv(f"{TABLES}/T3_manual_check_samples.csv",index=False)


def export_figures(data, summary, compare, source_rows, pairs, sensitivity, old_curve):
    plt=setup_cjk_matplotlib()
    tags=list(data)
    # 来源内部的绝对跨度可显示 RPS 异质性；P95 候选率在 A1 接近 5% 属阈值定义。
    fig,ax=plt.subplots(figsize=(9,4.5))
    p=source_rows[source_rows.domain=="ALL"]
    x=np.arange(len(SNAMES));width=.25
    for t,tag in enumerate(tags):
        v=p[p.dataset==tag].set_index("source").reindex(SNAMES)
        ax.bar(x+(t-1)*width,v.median_internal_span.to_numpy(),width,label=tag)
    ax.set_xticks(x,SNAMES,rotation=25,ha="right")
    ax.set_ylabel("来源内指标排名 P90−P10 的文档中位数")
    ax.set_title("RPS 内部跨度较大；FineWeb-Edu 单指标不定义内部跨度")
    ax.legend();fig.tight_layout();fig.savefig(f"{FIGS}/F3_source_internal.png");plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,4.5))
    sub=sensitivity[sensitivity.stance_tail==.2]
    for tag in tags:
        p=sub[sub.dataset==tag].sort_values("A1_tail_quantile")
        ax.plot(100*p.A1_tail_quantile,100*p.high_disagreement_candidate_rate,"o-",label=tag)
    ax.set_xlabel("A1 的 B 阈值分位 (%)");ax.set_ylabel("高分歧候选比例 (%)")
    ax.set_title("阈值敏感性：A1 比例由操作点定义，并非真实冲突率")
    ax.legend();fig.tight_layout();fig.savefig(f"{FIGS}/F3_threshold_sensitivity.png");plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,4.2))
    p=compare[compare.domain=="ALL"].set_index("dataset").loc[tags]
    bottom=np.zeros(len(tags))
    for col,label,color in [("old_only_rate","仅旧四组","#9ecae1"),("shared_rate","共同候选","#3182bd"),
                             ("new_only_rate","仅新六来源","#fdae6b")]:
        v=100*p[col].to_numpy();ax.bar(tags,v,bottom=bottom,label=label,color=color);bottom+=v
    ax.set_ylabel("文档比例 (%)");ax.set_title("旧四组与新六来源高分歧候选的重合")
    ax.legend();fig.tight_layout();fig.savefig(f"{FIGS}/F3_legacy_overlap.png");plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,4.2))
    bottom=np.zeros(len(tags))
    matched_only=p.matched_four_rate.to_numpy()-p.matched_four_shared_rate.to_numpy()
    new_only=p.new_only_rate.to_numpy()+p.shared_rate.to_numpy()-p.matched_four_shared_rate.to_numpy()
    for v,label,color in [(matched_only,"仅匹配四组","#9ecae1"),
                          (p.matched_four_shared_rate.to_numpy(),"共同候选","#3182bd"),
                          (new_only,"仅六来源","#fdae6b")]:
        ax.bar(tags,100*v,bottom=100*bottom,label=label,color=color)
        bottom+=v
    ax.set_ylabel("文档比例 (%)")
    ax.set_title("以 A1 前 5% 预算冻结的旧四组与六来源候选重合")
    ax.legend();fig.tight_layout();fig.savefig(f"{FIGS}/F3_matched_overlap.png");plt.close(fig)

    mat=np.zeros((len(SNAMES),len(SNAMES)))
    p=pairs[(pairs.dataset=="A1") & (pairs.domain=="ALL")]
    for row in p.itertuples():mat[SNAMES.index(row.high_source),SNAMES.index(row.low_source)]=row.share_of_between_candidates
    fig,ax=plt.subplots(figsize=(7,5.8))
    im=ax.imshow(mat*100,cmap="Blues",vmin=0)
    ax.set_xticks(range(len(SNAMES)),SNAMES,rotation=35,ha="right")
    ax.set_yticks(range(len(SNAMES)),SNAMES)
    ax.set_xlabel("低端来源");ax.set_ylabel("高端来源")
    ax.set_title("A1 高分歧候选的来源对立方向（每篇总权重为 1）")
    fig.colorbar(im,ax=ax,label="候选中的加权占比 (%)")
    fig.tight_layout();fig.savefig(f"{FIGS}/F3_source_opposition.png");plt.close(fig)

    tau,taus,rate=old_curve
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(taus,100*rate,lw=2)
    ax.axvline(tau,ls="--",c="crimson",label=f"旧四组阈值 τ={tau:.2f}")
    ax.set_xlabel("旧四组秩极差阈值");ax.set_ylabel("旧四组候选比例 (%)")
    ax.set_title("历史四组口径：仅用于方法对照")
    ax.legend();fig.tight_layout();fig.savefig(f"{FIGS}/F3_legacy_four_curve.png");plt.close(fig)


if __name__=="__main__":
    A={t:pd.read_pickle(f"{CACHE}/{t}_indicators.pkl") for t in ("A1","A2","A3")}
    sc=np.load(f"{CACHE}/step2_scores.npz")
    data={t:dict(Z=sc[f"Z{t[-1]}"],Q=sc[f"Q{t[-1]}"],domains=A[t].domain.to_numpy()) for t in A}
    for tag,e in data.items():
        assert len(e["Z"])==len(e["Q"])==len(e["domains"])
    d1=data["A1"]["domains"];Z1=data["A1"]["Z"]
    indicator_ref=fit_reference(Z1,d1)
    R1=indicator_ranks(Z1,d1,indicator_ref)
    fit=fit_six_thresholds(R1,d1)
    fit["indicator_ref"]=indicator_ref

    oldS1=legacy_group_score(Z1)
    group_ref={(d,k):np.sort(oldS1[d1==d,k]) for d in np.unique(d1) for k in range(len(LEGACY_NAMES))}
    oldR1=legacy_domain_rank(oldS1,d1)
    oldc1=oldR1.max(1)-oldR1.min(1)
    # 与六来源同为 A1 各域前 5% 的选择预算，避免把阈值差误读为分组差。
    matched_cutoff={d:float(np.quantile(oldc1[d1==d],.95)) for d in np.unique(d1)}
    fit["legacy_matched_cutoff"]=matched_cutoff
    with open(f"{CACHE}/step3_a1_reference.pkl","wb") as f:pickle.dump(fit,f)
    old_tau,old_taus,old_rate=legacy_elbow(oldc1)
    for tag,e in data.items():
        R=R1 if tag=="A1" else indicator_ranks(e["Z"],e["domains"],indicator_ref)
        e["six"]=diagnose_six(R,e["domains"],fit)
        e["legacy"]=legacy_diagnose(e["Z"],e["domains"],group_ref,matched_cutoff,old_tau)
        print(tag,"six-source high-disagreement candidate",ratio(e["six"]["between"]),
              "multi-source opposition",ratio(e["six"]["multi"]),
              "legacy four-group candidate",ratio(e["legacy"]["old"]),flush=True)
    summary,comparison,source_rows,pairs,sensitivity=export_tables(data,fit,old_tau,(old_tau,old_taus,old_rate))
    export_manual(data)
    export_figures(data,summary,comparison,source_rows,pairs,sensitivity,(old_tau,old_taus,old_rate))
    # 缓存只保存诊断；主 Q 和 P1 接口从未写入。
    np.savez_compressed(f"{CACHE}/step3_conflict.npz",
                        B1=data["A1"]["six"]["B"],B2=data["A2"]["six"]["B"],
                        B3=data["A3"]["six"]["B"],
                        six_candidate_A1=data["A1"]["six"]["between"],
                        six_candidate_A2=data["A2"]["six"]["between"],
                        six_candidate_A3=data["A3"]["six"]["between"],
                        old_four_c1=oldc1,old_four_tau=old_tau)
    print("Step3 done: 主 Q 保持原值；所有比例是规则筛出的候选比例，而非真值冲突率。")
