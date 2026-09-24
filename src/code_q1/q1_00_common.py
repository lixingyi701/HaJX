# -*- coding: utf-8 -*-
"""
q1_00_common.py — 问题一公共工具库（纯 numpy/pandas 实现，无 scipy/sklearn 依赖）
2026 研数模 F 题问题一。所有统计工具自实现，保证任意环境可复现。
"""
import os, json, lzma, math
import numpy as np
import pandas as pd

# ---------------- 路径 ----------------
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])  # src: 所有代码和 outputs_q* 的共同父目录
A_DIR = str(Path(ROOT).parent / "附件" / "A_data_value")
OUT = f"{ROOT}/outputs_q1"
CACHE = f"{OUT}/cache"
TABLES = f"{OUT}/tables"
FIGS = f"{OUT}/figures"
IFACE = f"{OUT}/interface"

A1_PATH = f"{A_DIR}/slimpajama_quality_signal_sample.jsonl.xz"
A2_PATH = f"{A_DIR}/slimpajama_quality_extended/arxiv_part-6777d8857c6e-000486.jsonl.xz"
A3_PATH = f"{A_DIR}/slimpajama_quality_extended/github_part-6777d8857c6e-000275.jsonl.xz"
A18_PATH = f"{A_DIR}/regmix_domain_sample.jsonl.xz"
RT = f"{A_DIR}/regmix_tables"
MAP_PATH = f"{A_DIR}/domain_mapping_guide.csv"

# ---------------- 25 指标展开定义 ----------------
# 硬事实（提示词 v2 §1）：
#   modernbert_*(6) -> softmax 期望分 sum(k*sigma_k), k=0..5
#   qurater(4)      -> 拆 4 维（QuRating: writing_style, required_expertise, facts_trivia, educational_value）
#   ad_en/fluency_en(2) -> softmax P(label=1)；ad 标签1=无广告(正向)，fluency 标签1=流畅(正向)
#   fineweb_edu(1)  -> 直接取值
QURATER_DIMS = ["qurater_writing", "qurater_expertise", "qurater_facts", "qurater_edu"]
MB_FIELDS = ["modernbert_professionalism", "modernbert_readability",
             "modernbert_reasoning", "modernbert_cleanliness"]
RPS_FIELDS = [
    "rps_doc_word_count", "rps_doc_num_sentences", "rps_doc_mean_word_length",
    "rps_doc_unigram_entropy", "rps_doc_frac_no_alph_words", "rps_doc_frac_unique_words",
    "rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram",
    "rps_lines_uppercase_letter_fraction",
    "rps_lines_ending_with_terminal_punctution_mark",  # 数据源拼写如此
    "rps_lines_numerical_chars_fraction",
]
DSIR_FIELDS = ["dsir_books", "dsir_wiki", "dsir_math"]
IND_COLS = (["fineweb_edu", "fluency_en", "ad_en"] + QURATER_DIMS + MB_FIELDS
            + DSIR_FIELDS + RPS_FIELDS)          # 恰 25 个指标
assert len(IND_COLS) == 25

# 指标方向：+1 越高越好；-1 越低越好；0 “适宜区间型”(距域内中位数偏离改单峰)
DIRECTION = {
    "fineweb_edu": 1, "fluency_en": 1, "ad_en": 1,      # ad 标签1=无广告 => P 越高越好
    "qurater_writing": 1, "qurater_expertise": 1, "qurater_facts": 1, "qurater_edu": 1,
    "modernbert_professionalism": 1, "modernbert_readability": 1,
    "modernbert_reasoning": 1, "modernbert_cleanliness": 1,
    "dsir_books": 1, "dsir_wiki": 1, "dsir_math": 1,     # 负对数比，越大(越不负)越好；域内标准化后使用
    "rps_doc_word_count": 0,          # 两端皆差 -> 单峰化
    "rps_doc_num_sentences": 0,       # 同上
    "rps_doc_mean_word_length": 0,    # 倒U
    "rps_doc_unigram_entropy": 0,     # 过高可能是噪声 -> 单峰化
    "rps_doc_frac_no_alph_words": -1,
    "rps_doc_frac_unique_words": 1,
    "rps_doc_frac_chars_top_2gram": -1,
    "rps_doc_frac_chars_top_3gram": -1,
    "rps_lines_uppercase_letter_fraction": -1,
    "rps_lines_ending_with_terminal_punctution_mark": 1,
    "rps_lines_numerical_chars_fraction": -1,
}

# 25 指标的来源分组（冲突分析用）
GROUPS = {
    "G1_规则型": RPS_FIELDS,
    "G2_DSIR": DSIR_FIELDS,
    "G3_轻量分类器": ["fluency_en", "ad_en"],            # WanjuanCC 系
    "G4_模型评分器": ["fineweb_edu"] + QURATER_DIMS + MB_FIELDS,  # PRRC+QuRating+FineWeb
}

# ---------------- 基础数学工具 ----------------
def softmax(v):
    v = np.asarray(v, dtype=float)
    e = np.exp(v - v.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def expand_record(r):
    """单条 json 记录 -> 25 指标 dict（缺失置 nan）"""
    out = {}
    fe = r.get("fineweb_edu")
    out["fineweb_edu"] = float(fe[0]) if isinstance(fe, list) and fe else np.nan
    for f in ("fluency_en", "ad_en"):
        v = r.get(f)
        out[f] = float(softmax(v)[1]) if isinstance(v, list) and len(v) == 2 else np.nan
    qv = r.get("qurater")
    if isinstance(qv, list) and len(qv) == 4:
        for name, x in zip(QURATER_DIMS, qv):
            out[name] = float(x)
    else:
        for name in QURATER_DIMS: out[name] = np.nan
    for f in MB_FIELDS:
        v = r.get(f)
        if isinstance(v, list) and len(v) == 6:
            p = softmax(v)
            out[f] = float((np.arange(6) * p).sum())   # 0–5 期望分
        else:
            out[f] = np.nan
    for f in DSIR_FIELDS + RPS_FIELDS:
        v = r.get(f)
        out[f] = float(v) if isinstance(v, (int, float)) else np.nan
    return out

def stream_jsonl_xz(path, domain=None, keep_content=False):
    """流式读取 .jsonl.xz -> 生成器 (25指标dict, domain, n_words, content?)"""
    with lzma.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: r = json.loads(line)
            except json.JSONDecodeError: continue
            row = expand_record(r)
            d = r.get("_source_domain", domain)
            c = r.get("content") if keep_content else None
            yield row, d, c

# ---------------- 稳健统计 ----------------
def winsorize(x, lo=0.01, hi=0.99):
    a, b = np.nanquantile(x, lo), np.nanquantile(x, hi)
    return np.clip(x, a, b), (a, b)

def minmax(x, lo=None, hi=None):
    lo = np.nanmin(x) if lo is None else lo
    hi = np.nanmax(x) if hi is None else hi
    rng = hi - lo
    return np.clip((x - lo) / rng, 0, 1) if rng > 0 else np.full_like(x, .5), (lo, hi)

def rankdata(x):
    """平均秩（处理并列）"""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sx = x[order]
    i = 0; n = len(x)
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]: j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return ranks

def spearman(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3: return np.nan
    ra, rb = rankdata(a[m]), rankdata(b[m])
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra*rb).sum()/den) if den > 0 else np.nan

def pearson(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3: return np.nan
    a, b = a[m]-a[m].mean(), b[m]-b[m].mean()
    den = np.sqrt((a**2).sum()*(b**2).sum())
    return float((a*b).sum()/den) if den > 0 else np.nan

def ks_2samp(a, b):
    """双样本 KS 统计量 + 渐近 p 值（Kolmogorov 分布）"""
    a = np.sort(a[~np.isnan(a)]); b = np.sort(b[~np.isnan(b)])
    n1, n2 = len(a), len(b)
    data = np.concatenate([a, b])
    cdf1 = np.searchsorted(a, data, side="right") / n1
    cdf2 = np.searchsorted(b, data, side="right") / n2
    D = np.abs(cdf1 - cdf2).max()
    en = math.sqrt(n1 * n2 / (n1 + n2))
    t = (en + 0.12 + 0.11 / en) * D
    p = 2 * sum((-1) ** (k - 1) * math.exp(-2 * (k * t) ** 2) for k in range(1, 101))
    return float(D), float(min(max(p, 0.0), 1.0))

def cohens_d(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    n1, n2 = len(a), len(b)
    s = math.sqrt(((n1-1)*a.var(ddof=1) + (n2-1)*b.var(ddof=1)) / (n1+n2-2))
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0

def kendall_w(mat):
    """Kendall 一致性系数 W；mat: (n_samples, k_judges) 列为评委(指标)"""
    n, k = mat.shape
    R = np.column_stack([rankdata(mat[:, j]) for j in range(k)])
    Ri = R.sum(axis=1)
    S = ((Ri - Ri.mean())**2).sum()
    return float(12*S / (k**2 * (n**3 - n)))

def chi2_sf(x, df):
    """卡方生存函数（Wilson–Hilferty + 正态近似，报告用途足够）"""
    if x <= 0: return 1.0
    z = ((x/df)**(1/3) - (1 - 2/(9*df))) / math.sqrt(2/(9*df))
    return float(0.5 * math.erfc(z / math.sqrt(2)))

# ---------------- 回归工具 ----------------
def ols(X, y):
    """最小二乘（带截距），返回 coef(含截距在前), 预测函数"""
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return beta

def huber_regression(X, y, delta=1.35, iters=50, tol=1e-8):
    """Huber IRLS 稳健回归（带截距）"""
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    for _ in range(iters):
        r = y - X1 @ beta
        s = np.median(np.abs(r - np.median(r))) / 0.6745 + 1e-12
        u = np.abs(r / s)
        w = np.where(u <= delta, 1.0, delta / u)
        W = np.sqrt(w)
        beta_new, *_ = np.linalg.lstsq(X1 * W[:, None], y * W, rcond=None)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new; break
        beta = beta_new
    return beta

def ridge(X, y, alpha=1.0):
    X1 = np.column_stack([np.ones(len(X)), X])
    A = X1.T @ X1 + alpha * np.eye(X1.shape[1]); A[0, 0] -= alpha  # 截距不惩罚
    return np.linalg.solve(A, X1.T @ y)

def predict_lin(beta, X):
    return np.column_stack([np.ones(len(X)), X]) @ beta

# ---------------- 成分数据 ----------------
def clr(P, eps=1e-4):
    """centered log-ratio；零分量用乘性替换 eps 后闭合"""
    P = np.asarray(P, dtype=float)
    P = np.where(P <= 0, eps, P)
    P = P / P.sum(axis=1, keepdims=True)
    logP = np.log(P)
    return logP - logP.mean(axis=1, keepdims=True)

# ---------------- 评价指标 ----------------
def r2_score(y, yh):
    ssr = ((y - yh)**2).sum(); sst = ((y - y.mean())**2).sum()
    return float(1 - ssr/sst) if sst > 0 else np.nan

def rmse(y, yh): return float(np.sqrt(((y - yh)**2).mean()))
def mae(y, yh):  return float(np.abs(y - yh).mean())

# ---------------- 迷你梯度提升（非线性对照，替代 LightGBM） ----------------
class Stump:
    __slots__ = ("f", "t", "l", "r")
def _fit_stump(X, g, n_bins=16):
    n, p = X.shape
    best = (np.inf, 0, 0.0, 0.0, 0.0)
    total_sum, total_cnt = g.sum(), n
    for f in range(p):
        x = X[:, f]
        qs = np.unique(np.quantile(x, np.linspace(0.05, 0.95, n_bins)))
        for t in qs:
            m = x <= t
            c1 = m.sum()
            if c1 < 8 or c1 > n - 8: continue
            s1 = g[m].sum(); s2 = total_sum - s1
            loss = -(s1*s1/c1 + s2*s2/(total_cnt-c1))
            if loss < best[0]:
                best = (loss, f, t, s1/c1, s2/(total_cnt-c1))
    st = Stump(); _, st.f, st.t, st.l, st.r = best
    return st
def gbdt_fit(X, y, n_trees=300, lr=0.05):
    F = np.full(len(y), y.mean()); trees = []
    for _ in range(n_trees):
        st = _fit_stump(X, y - F)
        pred = np.where(X[:, st.f] <= st.t, st.l, st.r)
        F += lr * pred; trees.append(st)
    return {"base": y.mean(), "lr": lr, "trees": trees}
def gbdt_predict(model, X):
    F = np.full(len(X), model["base"])
    for st in model["trees"]:
        F += model["lr"] * np.where(X[:, st.f] <= st.t, st.l, st.r)
    return F

# ---------------- 中文绘图 ----------------
def setup_cjk_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Hiragino Sans GB", "Arial Unicode MS", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    return plt
