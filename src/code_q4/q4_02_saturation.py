# -*- coding: utf-8 -*-
"""
q4_02_saturation.py — 问题四·Step5：C8 逐任务聚合 —— 任务饱和度分析 + 去饱和综合分

做法
 1. 逐任务按随机基线归一化 ŝ = max(0,(s−c)/(1−c))（c = 1/选项数；MATH/IFEval c=0；与榜单 v2 规则一致），
    先用 6 个 family 的重算分与 C1 六维对照，确认解析口径正确。
 2. 时间轴 = C1 Submission Date（C8 为同一批上榜评测）；月度累计前沿 F_t(m) = 截至 m 月 top-5 均值。
 3. 每任务：起点/终点前沿、月斜率、剩余空间 headroom = 1−F_end、剩余空间关闭率 = 斜率/(1−F_start)、
    规模敏感度（base 模型 ŝ 与 log N 的 Spearman）、中位 stderr（噪声门槛）。
 4. 分类：饱和 / 快速增长 / 低位停滞 / 常规增长。
 5. 去饱和综合分：剔除饱和任务，其余按起点剩余空间加权，family 等权；与 C1 Average 对照前沿增速。
产出：interface/P4_task_saturation.csv, tables/T5_*.csv, figures/F9, F10
"""
import glob, json
import numpy as np
import pandas as pd
from q4_00_common import (C8_DIR, CACHE, TAB, FIG, IFACE, DIMS, AVG, load_c1,
                          setup_cjk_matplotlib)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code_q1"))
from q1_00_common import spearman  # noqa: E402

FAM2DIM = {"ifeval": "IFEval", "bbh": "BBH", "math": "MATH Lvl 5", "gpqa": "GPQA",
           "musr": "MUSR", "mmlu_pro": "MMLU-PRO"}
MUSR_CHOICES = {"musr_murder_mysteries": 2, "musr_object_placements": 5, "musr_team_allocation": 3}


def chance_levels():
    """从一份 C8 JSON 的 configs 读 BBH 各子任务选项数；其余按榜单规则"""
    fp = sorted(glob.glob(f"{C8_DIR}/01-ai_Yi-1.5-34B/*.json"))[0]
    cfg = json.load(open(fp))["configs"]
    c = {}
    for t, v in cfg.items():
        t2 = t.replace("leaderboard_", "")
        if t2.startswith("bbh_"):
            c[t2] = 1 / len(v["doc_to_choice"])
    c.update({k: 1 / n for k, n in MUSR_CHOICES.items()})
    c.update({"gpqa_diamond": .25, "gpqa_extended": .25, "gpqa_main": .25, "mmlu_pro": .1})
    return c


def top_k_mean(x, k=5):
    x = np.sort(np.asarray(x))[::-1]
    return x[:k].mean() if len(x) else np.nan


if __name__ == "__main__":
    L = pd.read_pickle(f"{CACHE}/c8_long.pkl")
    e = load_c1().drop_duplicates("Model")
    ch = chance_levels()
    L["chance"] = L.task.map(ch).fillna(0.0)
    L["s_norm"] = ((L.score - L.chance) / (1 - L.chance)).clip(lower=0)
    L["se_norm"] = L.stderr / (1 - L.chance)
    L = L.merge(e[["Model", "sub_date", "group", "N"] + DIMS + [AVG]],
                left_on="model", right_on="Model", how="inner")
    L["date"] = L.sub_date.fillna(pd.to_datetime(L.eval_time.str[:10]))
    L["month"] = L.date.dt.to_period("M")
    print(f"C8 ∩ C1: {L.model.nunique()} 模型, {L.task.nunique()} 叶子任务")

    # ---------- 1. 解析口径校验：重算 family 分 vs C1 六维 ----------
    # MATH/GPQA 在榜单上是按样本合并计分（子集按样本数加权），BBH/MUSR 为子任务等权
    L["wn"] = np.where(L.family.isin(["math", "gpqa"]), L.n_samples.fillna(1).astype(float), 1.0)
    g_ = L.assign(sw=L.s_norm * L.wn).groupby(["model", "family"])
    fam_w = (g_.sw.sum() / g_.wn.sum()).unstack() * 100
    chk = []
    for f, dim in FAM2DIM.items():
        y = e.set_index("Model").loc[fam_w.index, dim]
        ok = y.notna() & fam_w[f].notna()
        chk.append(dict(family=f, C1维度=dim, pearson=np.corrcoef(fam_w[f][ok], y[ok])[0, 1],
                        MAE=float(np.abs(fam_w[f][ok] - y[ok]).mean()), n=int(ok.sum())))
    CHK = pd.DataFrame(chk)
    CHK.to_csv(f"{TAB}/T5_c8_vs_c1_check.csv", index=False, encoding="utf-8-sig")
    print(CHK.round(3).to_string(index=False))

    # ---------- 2-4. 逐任务前沿与饱和度 ----------
    months = pd.period_range("2024-06", "2025-03", freq="M")
    t_idx = np.arange(len(months))
    base = L[L.group == "base"]
    rows, front = [], {}
    for task, d in L.groupby("task"):
        F = np.array([top_k_mean(d.loc[d.month <= m, "s_norm"]) for m in months])
        front[task] = F
        slope = np.polyfit(t_idx, F, 1)[0]
        b = base[base.task == task]
        rows.append(dict(task=task, family=d.family.iat[0], chance=d.chance.iat[0],
                         n_models=d.model.nunique(), mean_norm=d.s_norm.mean(),
                         F_start=F[0], F_end=F[-1], gain=F[-1] - F[0], slope_per_month=slope,
                         headroom_end=1 - F[-1],
                         closure_rate_per_month=slope / max(1 - F[0], 1e-6),
                         se_median=d.se_norm.median(),
                         scale_rho_base=spearman(np.log10(b.N.values), b.s_norm.values)
                         if len(b) > 10 else np.nan))
    S = pd.DataFrame(rows)
    q_hi = S.closure_rate_per_month.quantile(2 / 3)
    def cls(r):
        if r.F_end >= 0.85 or (r.headroom_end < 0.2 and r.gain < 2 * r.se_median):
            return "饱和"
        if r.F_end < 0.35 and r.gain < 2 * r.se_median:
            return "低位停滞"
        if r.closure_rate_per_month >= q_hi:
            return "快速增长"
        return "常规增长"
    S["status"] = S.apply(cls, axis=1)
    S = S.sort_values(["status", "closure_rate_per_month"], ascending=[True, False])
    print(S[["task", "F_start", "F_end", "gain", "closure_rate_per_month", "se_median",
             "scale_rho_base", "status"]].round(3).to_string(index=False))
    print(S.status.value_counts())

    # ---------- 5. 去饱和综合分 ----------
    keep = S[S.status != "饱和"].set_index("task")
    w = (1 - keep.F_start).clip(lower=0.05)
    w = w / w.groupby(keep.family).transform("sum")        # family 内归一
    nf = keep.family.nunique()
    L2 = L[L.task.isin(keep.index)].copy()
    L2["w"] = L2.task.map(w) / nf
    desat = (L2.s_norm * L2.w).groupby(L2.model).sum() * 100
    # 等权"全任务"重算分（family 等权，任务等权）作为中间对照
    L["w_all"] = 1 / L.groupby(["model", "family"]).task.transform("count") / 6
    allw = (L.s_norm * L.w_all).groupby(L.model).sum() * 100
    M = e.set_index("Model").loc[desat.index, ["sub_date", "group", "N", AVG]].copy()
    M["S_desat"] = desat; M["S_alltask"] = allw.loc[desat.index]
    M["month"] = M.sub_date.dt.to_period("M")

    comp = []
    for col in (AVG, "S_alltask", "S_desat"):
        F = np.array([top_k_mean(M.loc[M.month <= m, col]) for m in months])
        sl = np.polyfit(t_idx, F, 1)[0]
        comp.append(dict(metric=col, F_start=F[0], F_end=F[-1], slope_per_month=sl,
                         rel_gain_pct=(F[-1] / F[0] - 1) * 100,
                         closure_rate_per_month=sl / (100 - F[0])))
        front["__" + col] = F
    for grp in ("base", "chat"):
        Mg = M[(M.group == grp) & M.sub_date.notna() & (M.N > 0)]
        X = np.c_[np.ones(len(Mg)), np.log10(Mg.N), (Mg.sub_date - pd.Timestamp("2024-06-01")).dt.days / 30.44]
        for col in (AVG, "S_desat"):
            b_, *_ = np.linalg.lstsq(X, Mg[col].values, rcond=None)
            comp.append(dict(metric=f"{col}|{grp} 回归", n=len(Mg), coef_log10N=b_[1],
                             coef_month=b_[2], month_equiv_dex=b_[2] / b_[1] * 12))
    COMP = pd.DataFrame(comp)
    COMP.to_csv(f"{TAB}/T5_desat_vs_average.csv", index=False, encoding="utf-8-sig")
    print(COMP.round(4).to_string(index=False))
    print("去饱和分 vs Average 相关:", np.corrcoef(M.S_desat, M[AVG])[0, 1])

    S["weight_in_desat"] = S.task.map(w / nf).fillna(0.0)
    S.to_csv(f"{IFACE}/P4_task_saturation.csv", index=False, encoding="utf-8-sig")
    M.reset_index().rename(columns={"index": "Model"}).to_csv(
        f"{TAB}/T5_model_desat_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"month": months.astype(str), **front}).to_csv(
        f"{TAB}/T5_task_frontier_monthly.csv", index=False, encoding="utf-8-sig")

    # ---------------- 作图 ----------------
    plt = setup_cjk_matplotlib()
    colst = {"饱和": "#d7301f", "快速增长": "#31a354", "常规增长": "#2b8cbe", "低位停滞": "#737373"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    ax = axes[0]
    for st, d in S.groupby("status"):
        ax.scatter(d.F_start, d.closure_rate_per_month * 100, s=40, color=colst[st], label=f"{st}（{len(d)}）")
    for r in S.itertuples():
        if r.status in ("饱和", "低位停滞") or r.closure_rate_per_month >= S.closure_rate_per_month.quantile(.85):
            ax.annotate(r.task.replace("bbh_", "").replace("_hard", ""), (r.F_start, r.closure_rate_per_month * 100),
                        fontsize=7, xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("2024-06 前沿（去随机基线归一化）"); ax.set_ylabel("剩余空间月关闭率 (%)")
    ax.set_title("39 个叶子任务：起点 × 增速"); ax.legend(fontsize=8)
    ax = axes[1]
    fams = ["bbh", "math", "gpqa", "musr", "ifeval", "mmlu_pro"]
    cmap = {"bbh": "#2b8cbe", "math": "#d7301f", "gpqa": "#756bb1", "musr": "#31a354",
            "ifeval": "#fd8d3c", "mmlu_pro": "#636363"}
    for r in S.itertuples():
        ax.plot(months.astype(str), front[r.task], color=cmap[r.family], lw=.8, alpha=.6)
    for f in fams:
        ax.plot([], [], color=cmap[f], label=f)
    ax.set_xticks(range(0, len(months), 3)); ax.set_xticklabels(months.astype(str)[::3])
    ax.set_ylabel("累计前沿（top-5 均值，归一化）"); ax.set_title("逐任务前沿随时间（C1 提交日）")
    ax.legend(fontsize=8, ncol=3)
    fig.suptitle("图9  C8 逐任务饱和度分析")
    fig.tight_layout(); fig.savefig(f"{FIG}/F9_task_saturation.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for col, lab in ((AVG, "C1 Average"), ("S_alltask", "C8 全任务重算"), ("S_desat", "C8 去饱和综合分")):
        F = front["__" + col]
        axes[0].plot(months.astype(str), F / F[0], marker="o", ms=3, label=lab)
    axes[0].set_xticks(range(0, len(months), 3)); axes[0].set_xticklabels(months.astype(str)[::3])
    axes[0].set_ylabel("前沿（相对 2024-06）"); axes[0].legend(); axes[0].set_title("三种能力口径的前沿增速")
    axes[1].scatter(M[AVG], M.S_desat, s=4, alpha=.4)
    axes[1].set_xlabel("C1 Average"); axes[1].set_ylabel("去饱和综合分")
    axes[1].set_title(f"模型级对照（r = {np.corrcoef(M.S_desat, M[AVG])[0, 1]:.3f}）")
    fig.suptitle("图10  去饱和综合分 vs 榜单 Average")
    fig.tight_layout(); fig.savefig(f"{FIG}/F10_desat_vs_avg.png"); plt.close(fig)
    print("done")
