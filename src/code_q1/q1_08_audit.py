# -*- coding: utf-8 -*-
"""旧条件几何评分仅用于方法对照；绝不写入 P1 主接口。"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q1_00_common import CACHE, TABLES, IND_COLS, GROUPS, spearman

def legacy_score(Z, w, thresholds):
    dsir_idx = [IND_COLS.index(c) for c in GROUPS["G2_DSIR"]]
    rule_idx = [IND_COLS.index(c) for c in GROUPS["G1_规则型"]]
    dsir = Z[:, dsir_idx].mean(1); rule = Z[:, rule_idx].mean(1)
    W = np.broadcast_to(w, Z.shape).copy()
    mask1 = (dsir > thresholds["dsir_p75"]) & (rule < thresholds["rule_p25"])
    mask2 = (rule > thresholds["rule_p75"]) & (dsir < thresholds["dsir_p25"])
    W[np.ix_(mask1, rule_idx)] *= .5
    W[np.ix_(mask2, dsir_idx)] *= .5
    W /= W.sum(1, keepdims=True)
    return np.exp((W * np.log(np.clip(Z, 1e-3, 1))).sum(1))

if __name__ == "__main__":
    sc = np.load(f"{CACHE}/step2_scores.npz")
    Z, Q, w = sc["Z1"], sc["Q1"], sc["w"]
    A1 = pd.read_pickle(f"{CACHE}/A1_indicators.pkl")
    dsir_idx = [IND_COLS.index(c) for c in GROUPS["G2_DSIR"]]
    rule_idx = [IND_COLS.index(c) for c in GROUPS["G1_规则型"]]
    dsir, rule = Z[:, dsir_idx].mean(1), Z[:, rule_idx].mean(1)
    thresholds = {"dsir_p25":np.quantile(dsir,.25), "dsir_p75":np.quantile(dsir,.75),
                  "rule_p25":np.quantile(rule,.25), "rule_p75":np.quantile(rule,.75)}
    Q_old = legacy_score(Z, w, thresholds)
    rows = []
    for dom, sub in A1.groupby("domain"):
        pos = A1.index.get_indexer(sub.index)
        rows.append(dict(domain=dom, n=len(pos), old_conditional_geometric=float(np.median(Q_old[pos])),
                         new_weighted_huber=float(np.median(Q[pos]))))
    out = pd.DataFrame(rows)
    out["old_rank"] = out["old_conditional_geometric"].rank(ascending=False, method="min").astype(int)
    out["new_rank"] = out["new_weighted_huber"].rank(ascending=False, method="min").astype(int)
    out["delta_new_minus_old"] = out["new_weighted_huber"]-out["old_conditional_geometric"]
    out.sort_values("new_rank").to_csv(f"{TABLES}/T8_old_new_domain_ranking.csv", index=False)
    pd.DataFrame([{"comparison":"A1旧条件几何 vs 新Huber", "sample_spearman":spearman(Q_old,Q),
                   "domain_rank_spearman":spearman(out["old_rank"].to_numpy(),out["new_rank"].to_numpy())}]
                 ).to_csv(f"{TABLES}/T8_old_new_agreement.csv", index=False)
    print(out.sort_values("new_rank").to_string(index=False))
