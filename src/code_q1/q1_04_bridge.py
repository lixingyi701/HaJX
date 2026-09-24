# -*- coding: utf-8 -*-
"""
q1_04_bridge.py — Step4：质量线→配比线桥接（17 配方域质量分 Q̂_d）
方法（创新点 N-bridge）：
  ① 自实现规则特征算子 Φ(text) -> 11 维（对应 rps_* 定义）
  ② 在 A1 上用 content 重算 Φ（与应用端同源，消除实现差），
     以 Step2 主分 Q 为标签训练校准函数 ĝ（岭回归 + GBDT 对照，域分层 5 折 CV）
  ③ 对 A18 的 17 配方域文本算 Φ→ĝ→文档分，聚合成 Q̂_d（中位数 + token 加权 + bootstrap CI）
  ④ direct/near_direct 6 域：Q̂_d 与 Step2 域级分互验 → 报告映射误差
输出：P1-Q_domain（17 域，带 CI 与来源等级）
"""
import os, sys, re, math, lzma, json, collections
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q1_00_common import (CACHE, TABLES, IFACE, FIGS, A1_PATH, A18_PATH, MAP_PATH,
                          ridge, predict_lin, gbdt_fit, gbdt_predict,
                          spearman, pearson, r2_score, setup_cjk_matplotlib)

RNG = np.random.default_rng(42)
PHI_COLS = ["word_count", "num_sentences", "mean_word_length", "unigram_entropy",
            "frac_no_alph_words", "frac_unique_words", "frac_chars_top_2gram",
            "frac_chars_top_3gram", "lines_uppercase_frac", "lines_terminal_punct",
            "lines_numerical_frac"]
WORD_RE = re.compile(r"\S+")
ALPHA_RE = re.compile(r"[A-Za-z]")
SENT_RE = re.compile(r"[.!?]+[\s\"')\]]|[.!?]+$")

def phi(text, max_chars=40000):
    """规则质量特征（RedPajama-v2 rps_* 的同名复算版）"""
    t = text[:max_chars]
    words = WORD_RE.findall(t)
    n_w = len(words)
    if n_w == 0:
        return None
    n_sent = len(SENT_RE.findall(t))
    mwl = sum(len(w) for w in words) / n_w
    cnt = collections.Counter(w.lower() for w in words)
    tot = sum(cnt.values())
    ent = -sum((c/tot) * math.log(c/tot) for c in cnt.values())
    no_alpha = sum(1 for w in words if not ALPHA_RE.search(w)) / n_w
    uniq = len(cnt) / n_w
    # top n-gram 字符占比
    def top_ng_frac(n):
        if n_w < n: return 0.0
        ng = collections.Counter(tuple(words[i:i+n]) for i in range(n_w - n + 1))
        (top, c), = ng.most_common(1)
        return c * sum(len(x) for x in top) / max(len(t), 1)
    lines = [l for l in t.split("\n") if l.strip()]
    n_l = max(len(lines), 1)
    upper = sum(sum(ch.isupper() for ch in l) / max(sum(ch.isalpha() for ch in l), 1)
                for l in lines) / n_l
    term = sum(1 for l in lines if l.rstrip().endswith((".", "!", "?", '."', ".'"))) / n_l
    numer = sum(sum(ch.isdigit() for ch in l) / max(len(l), 1) for l in lines) / n_l
    return dict(word_count=n_w, num_sentences=n_sent, mean_word_length=mwl,
                unigram_entropy=ent, frac_no_alph_words=no_alpha,
                frac_unique_words=uniq, frac_chars_top_2gram=top_ng_frac(2),
                frac_chars_top_3gram=top_ng_frac(3), lines_uppercase_frac=upper,
                lines_terminal_punct=term, lines_numerical_frac=numer)

def phi_design(F):
    """特征工程：log1p 重尾特征 + 原值"""
    X = F.copy()
    for c in ("word_count", "num_sentences"):
        X[c] = np.log1p(X[c])
    return X.to_numpy(dtype=float)

# =====================================================================
if __name__ == "__main__":
    # ---- ① 在 A1 上重算 Φ（每域抽样上限 4000，共约 2.4 万篇，够拟合校准） ----
    cache_f = f"{CACHE}/step4_a1_phi.pkl"
    if os.path.exists(cache_f):
        a1phi = pd.read_pickle(cache_f)
    else:
        cap = {d: 4000 for d in ["c4", "commoncrawl", "github", "stackexchange", "wikipedia"]}
        cap.update({"arxiv": 1419, "book": 171})
        seen = collections.Counter(); rows = []
        with lzma.open(A1_PATH, "rt", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                try: r = json.loads(line)
                except json.JSONDecodeError: continue
                d = r.get("_source_domain")
                if seen[d] >= cap.get(d, 0): continue
                p = phi(r.get("content") or "")
                if p is None: continue
                p["row"] = i; p["domain"] = d
                rows.append(p); seen[d] += 1
        a1phi = pd.DataFrame(rows)
        a1phi.to_pickle(cache_f)
    print(f"A1 复算 Φ: {len(a1phi)} 篇", dict(collections.Counter(a1phi['domain'])))

    # 标签与样本级接口完全同源：A1 的 CRITIC 加权 Huber 主分 Q。
    score = np.load(f"{CACHE}/step2_scores.npz")
    Q1 = score["Q1"]
    a1_source = pd.read_pickle(f"{CACHE}/A1_indicators.pkl")
    assert a1phi["row"].max() < len(Q1)
    assert (a1_source["domain"].iloc[a1phi["row"].to_numpy()].to_numpy()
            == a1phi["domain"].to_numpy()).all(), "A1 原文行号与评分缓存未对齐"
    y = Q1[a1phi["row"].to_numpy()]
    X = phi_design(a1phi[PHI_COLS])
    dom = a1phi["domain"].to_numpy()

    # ---- ② 域分层 5 折 CV 选模型 ----
    folds = np.zeros(len(X), dtype=int)
    for d in np.unique(dom):
        idx = np.where(dom == d)[0]; RNG.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % 5
    cv = {"ridge": [], "gbdt": []}
    for k in range(5):
        tr, te = folds != k, folds == k
        fold_mu, fold_sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        Xtr, Xte = (X[tr]-fold_mu)/fold_sd, (X[te]-fold_mu)/fold_sd
        b = ridge(Xtr, y[tr], alpha=1.0)
        cv["ridge"].append(r2_score(y[te], predict_lin(b, Xte)))
        g = gbdt_fit(Xtr, y[tr], n_trees=250, lr=0.06)
        cv["gbdt"].append(r2_score(y[te], gbdt_predict(g, Xte)))
    print("校准函数 CV R²:", {k: round(float(np.mean(v)), 4) for k, v in cv.items()})
    use_gbdt = np.mean(cv["gbdt"]) > np.mean(cv["ridge"])
    model_name = "GBDT" if use_gbdt else "Ridge"
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Xs = (X - mu) / sd
    if use_gbdt:
        model = gbdt_fit(Xs, y, n_trees=300, lr=0.06)
        pred_fn = lambda Xn: gbdt_predict(model, Xn)
    else:
        beta = ridge(Xs, y, alpha=1.0)
        pred_fn = lambda Xn: predict_lin(beta, Xn)
    cv_r2 = float(np.mean(cv["gbdt" if use_gbdt else "ridge"]))
    pd.DataFrame(cv).to_csv(f"{TABLES}/T4_calibration_cv.csv", index=False)

    # ---- ③ A18：17 配方域文本 → Φ → ĝ ----
    cache18 = f"{CACHE}/step4_a18_phi.pkl"
    if os.path.exists(cache18):
        a18 = pd.read_pickle(cache18)
    else:
        CAP18 = 3000
        seen = collections.Counter(); rows = []
        with lzma.open(A18_PATH, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                try: r = json.loads(line)
                except json.JSONDecodeError: continue
                d = r.get("_source_domain") or r.get("meta", {}).get("pile_set_name")
                if d is None: continue
                if seen[d] >= CAP18: continue
                p = phi(r.get("text") or r.get("content") or "")
                if p is None: continue
                p["domain"] = d
                rows.append(p); seen[d] += 1
        a18 = pd.DataFrame(rows)
        a18.to_pickle(cache18)
    print(f"A18 复算 Φ: {len(a18)} 篇, {a18['domain'].nunique()} 域")
    X18 = (phi_design(a18[PHI_COLS]) - mu) / sd
    q18 = np.clip(pred_fn(X18), 0, 1)
    a18["Qhat"] = q18

    # ---- ④ 域级聚合 + 与 Step2 官方分互验 ----
    mapping = pd.read_csv(MAP_PATH)
    dom_q = pd.read_csv(f"{TABLES}/T2_domain_quality.csv")
    # 映射域点估计与区间均来自 Step2 的 A1 主分中位数 bootstrap。
    a1_domain = dom_q[dom_q["dataset"] == "A1"].set_index("domain")
    off = a1_domain["Q_med"].to_dict()

    rows = []
    for d, sub in a18.groupby("domain"):
        q = sub["Qhat"].to_numpy(); wc = sub["word_count"].to_numpy()
        B = 2000; n = len(q)
        bs = np.array([np.median(q[RNG.integers(0, n, n)]) for _ in range(B)])
        mrow = mapping[mapping["mixture_domain"] == d]
        mtype = mrow["mapping_type"].iloc[0] if len(mrow) else "unknown"
        qd = mrow["quality_domain"].iloc[0] if len(mrow) else "(none)"
        rows.append(dict(mixture_domain=d, n=n,
                         Q_med=float(np.median(q)),
                         CI_calibrated_lo=float(np.quantile(bs, .025)),
                         CI_calibrated_hi=float(np.quantile(bs, .975)),
                         Q_tokenw_calibrated=float((q*wc).sum()/wc.sum()),
                         mapping_type=mtype,
                         Q_mapped_A1=off.get(qd, np.nan),
                         mapped_quality_domain=qd,
                         evidence="校准推断" if mtype == "inferred" else "A1映射主分"))
    bridge = pd.DataFrame(rows).sort_values("Q_med", ascending=False)
    print(bridge.to_string(index=False))

    # 互验：6 个映射域上 校准分 vs 官方分
    both = bridge.dropna(subset=["Q_mapped_A1"])
    val = dict(n=len(both),
               pearson=pearson(both["Q_med"].to_numpy(), both["Q_mapped_A1"].to_numpy()),
               spearman=spearman(both["Q_med"].to_numpy(), both["Q_mapped_A1"].to_numpy()),
               mae=float(np.abs(both["Q_med"] - both["Q_mapped_A1"]).mean()))
    print("6 映射域互验:", {k: round(v, 4) if isinstance(v, float) else v for k, v in val.items()})
    pd.Series(val).to_csv(f"{TABLES}/T4_bridge_validation.csv")

    # ---- 接口 P1-Q_domain ----
    # 最终 17 域质量分：映射域用官方分（更可信），inferred 用校准分；统一带 CI
    final = bridge.copy()
    final["Q_final"] = np.where(final["Q_mapped_A1"].notna(),
                                final["Q_mapped_A1"], final["Q_med"])
    final["CI_lo"] = [float(a1_domain.loc[d, "CI_lo"]) if d in a1_domain.index else cal
                      for d, cal in zip(final["mapped_quality_domain"], final["CI_calibrated_lo"])]
    final["CI_hi"] = [float(a1_domain.loc[d, "CI_hi"]) if d in a1_domain.index else cal
                      for d, cal in zip(final["mapped_quality_domain"], final["CI_calibrated_hi"])]
    final["CI_source"] = np.where(final["Q_mapped_A1"].notna(),
                                  "A1主分中位数bootstrap", "A18桥接预测中位数bootstrap")
    final["Q_tokenw"] = [float(a1_domain.loc[d, "Q_tokenw"]) if d in a1_domain.index else cal
                          for d, cal in zip(final["mapped_quality_domain"], final["Q_tokenw_calibrated"])]
    final["cv_r2_calibration"] = cv_r2
    final["calibration_model"] = model_name
    assert ((final["CI_lo"] <= final["Q_final"]) & (final["Q_final"] <= final["CI_hi"])).all()
    final.to_csv(f"{TABLES}/T4_bridge_17domains.csv", index=False)
    final.to_csv(f"{IFACE}/P1_Q_domain.csv", index=False)
    np.savez_compressed(f"{CACHE}/step4_bridge.npz",
                        domains=final["mixture_domain"].to_numpy(),
                        Q_final=final["Q_final"].to_numpy())

    # ---- 图 ----
    plt = setup_cjk_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 5))
    b = final.sort_values("Q_final")
    colors = {"direct": "#2b8cbe", "near_direct": "#74a9cf", "inferred": "#fd8d3c"}
    xerr_lo = np.maximum(b["Q_final"] - b["CI_lo"], 0)
    xerr_hi = np.maximum(b["CI_hi"] - b["Q_final"], 0)
    ax.barh(b["mixture_domain"], b["Q_final"],
            xerr=[xerr_lo, xerr_hi],
            color=[colors.get(t, "grey") for t in b["mapping_type"]], alpha=.85)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=v, label=k) for k, v in colors.items()], loc="lower right")
    ax.set_xlabel("域级质量分 $\\hat Q_d$")
    ax.set_title(f"17 配方域质量分（校准 {model_name}, CV R²={cv_r2:.2f}；映射域采用官方分）")
    fig.tight_layout(); fig.savefig(f"{FIGS}/F4_bridge_17domains.png"); plt.close(fig)
    print("Step4 done.")
