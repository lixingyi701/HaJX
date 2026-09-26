# -*- coding: utf-8 -*-
"""
q4_05_forecast.py — 问题四·Step4：开源前沿能力情景预测（12 / 24 个月）+ 回测

链条（每一环都复用前面已估计的量，不新增自由参数）：
  算力情景 g（dex/yr）→ 前沿算力 lc(T+h) = lc_T + g·h
  → 问题三最优配置下的 Loss 降幅  ΔL = L*(lc_T+g·h) − L*(lc_T)   （κ = 6 + η·L_ctx，Q=1）
  → 前沿 L̂(T+h) = L̂_T + ΔL          （"锚定增量"：现有前沿并非算力最优，只借用 L* 的斜率）
  → 桥接 + 技术项：S(T+h) = S_T + [f(L̂_{T+h}, t_T+h) − f(L̂_T, t_T)]
       技术形式①  M1b 加性：f = lo + amp·σ(k(L−L0)) + τ·t   （线性技术进步，乐观）
       技术形式②  M1c 有效 Loss：f = lo + amp·σ(k(L − δt − L0))  （技术进步也受桥接饱和约束，保守）
  chat 前沿 = chat 当前前沿 + base 模型增量 + 后训练增益增量 ΔG（线性趋势 / 停滞 两种）
情景：g = 0.60（C4 开源权重最大算力 2019–2025 斜率 0.58、Top-5 0.60，基线）、0.30（减半）、0.15（1/4）
不确定性：蒙特卡洛 4000 次 —— 问题二标度律 bootstrap × 分解 bootstrap（簇重抽样 120 组）
          × 后训练增益 bootstrap × 前沿残差噪声（wRMSE/√TOPK）；两种技术形式各半 ⇒ 50%/90% 区间
上界：纯规模天花板 lo+amp（L→E 时桥接上限）与"算力无限"极限情形一并报告
回测：① 发布日轴：只用 t<2024.5 的 base 重估 M1b/M1c，锚 2024.5 前沿，预测 2025.25 前沿；
      ② 提交日轴（C1 月度前沿，全部类型）：2024-06~12 线性趋势外推 2025-01~03，作为朴素对照。
产出：interface/P4_frontier_forecast.csv；tables/T8_*.csv；figures/F14
"""
import json
import numpy as np
import pandas as pd
from q4_00_common import (TAB, FIG, IFACE, AVG, ETA, PAR, law_loss, setup_cjk_matplotlib)
from q4_04_decomp import (build_panel, frontier_state, fit_add, fit_eff, f_add, f_eff, T_REF, TOPK)
from q4_00_common import ROOT

RNG = np.random.default_rng(11)
T0 = 2025.25                       # 数据截止（C2 最晚发布 2025-03 前后）；模型在此锚定
T_NOW = 2026.75                    # "现在"=2026-09（竞赛时间）；距数据截止约 1.5 年，本身为模型外推
DT = T_NOW - T0                    # 1.5 年
HORIZONS = (2.5, 3.5)              # 距数据锚点 2.5/3.5 年 ⇒ 2027-09 / 2028-09（自 2026-09 起 12/24 个月）
SCEN = {"基线 0.60 dex/yr": 0.60, "减半 0.30 dex/yr": 0.30, "1/4 0.15 dex/yr": 0.15}
LCTX = 4096
KAPPA = 6 + ETA * LCTX
NGRID = np.logspace(8, 13.5, 1201)


def Lstar(lc, p=PAR):
    lc = np.atleast_1d(lc)
    D = 10 ** lc[:, None] / (KAPPA * NGRID[None])
    return law_loss(NGRID[None], D, 1.0, p).min(1)


def p2_draws():
    z = np.load(f"{ROOT}/outputs_q2/interface/P2_bootstrap.npz", allow_pickle=True)
    cols = list(z["cols"])
    out = []
    for s in z["samples"]:
        p = dict(PAR); p.update({("E" if c == "E" else c): float(v) for c, v in zip(cols, s)})
        out.append(p)
    return out


def delta_S(form, p, L0_, t0_, L1_, t1_):
    f = f_add if form == "M1b" else f_eff
    return f(p, L1_, t1_ - T_REF) - f(p, L0_, t0_ - T_REF)


def simulate(state, g, h, J, P2D, n=4000, noise=0.0, G=None, posttrain="linear"):
    """返回 n 个 S(T+h) 抽样及分量（规模增量、技术增量）"""
    bM1b, bM1c = J["boot_M1b"], J["boot_M1c"]
    res = np.empty(n); sc = np.empty(n); te = np.empty(n)
    lc1 = state["lc"] + g * h
    for i in range(n):
        p2 = P2D[RNG.integers(len(P2D))]
        dL = Lstar(lc1, p2)[0] - Lstar(state["lc"], p2)[0]
        form = "M1b" if i % 2 == 0 else "M1c"
        pb = (bM1b if form == "M1b" else bM1c)[RNG.integers(len(bM1b))]
        L0_, t0_ = state["L"], state["t"]
        tot = delta_S(form, pb, L0_, t0_, L0_ + dL, t0_ + h)
        s_only = 0.5 * (delta_S(form, pb, L0_, t0_, L0_ + dL, t0_) +
                        delta_S(form, pb, L0_, t0_ + h, L0_ + dL, t0_ + h))
        dG = 0.0
        if G is not None:
            gb = G[RNG.integers(len(G))]
            dG = gb[1] * h if posttrain == "linear" else 0.0
        res[i] = state["S"] + tot + dG + RNG.normal(0, noise)
        sc[i] = s_only; te[i] = tot - s_only + dG
    return res, sc, te


def qs(x):
    return dict(p05=np.percentile(x, 5), p25=np.percentile(x, 25), p50=np.percentile(x, 50),
                p75=np.percentile(x, 75), p95=np.percentile(x, 95))


if __name__ == "__main__":
    # ---------------- 情景依据：C4 开源权重模型年度最大训练算力 ----------------
    from q4_00_common import load_c4_lm
    lm = load_c4_lm()
    ow = lm[lm.open & lm.C.notna() & lm.pub.notna()].assign(yr=lambda d: d.pub.dt.year)
    ow = ow[(ow.yr >= 2019) & (ow.yr <= 2025)]
    al = lm[lm.C.notna() & lm.pub.notna()].assign(yr=lambda d: d.pub.dt.year)
    ct = []
    for yr, d in ow.groupby("yr"):
        lcs = np.sort(np.log10(d.C.values))[::-1]
        ct.append(dict(year=yr, n_open=len(d), open_max_lc=lcs[0], open_top5_mean_lc=lcs[:5].mean(),
                       all_max_lc=np.log10(al[al.yr == yr].C.max())))
    CT = pd.DataFrame(ct)
    sl = dict(open_max_2019_2025=np.polyfit(CT.year, CT.open_max_lc, 1)[0],
              open_top5_2019_2025=np.polyfit(CT.year, CT.open_top5_mean_lc, 1)[0],
              open_max_2022_2025=np.polyfit(CT.year[CT.year >= 2022], CT.open_max_lc[CT.year >= 2022], 1)[0],
              all_max_2019_2025=np.polyfit(CT.year, CT.all_max_lc, 1)[0])
    CT.to_csv(f"{TAB}/T8_compute_trend.csv", index=False, encoding="utf-8-sig")
    print(CT.round(2).to_string(index=False)); print("算力斜率 dex/yr:", {k: round(v, 3) for k, v in sl.items()})

    J = json.load(open(f"{IFACE}/P4_decomp_models.json"))
    P2D = p2_draws()
    P, e = build_panel()
    base = P[P.group == "base"].reset_index(drop=True)
    chat = P[P.group == "chat"].reset_index(drop=True)
    noise = J["wRMSE_M1b"] / np.sqrt(TOPK)
    sb, sc_ = frontier_state(base, T0), frontier_state(chat, T0)
    # 后训练增益 bootstrap（与 q4_04 同式重算，便于抽样）
    PR = pd.read_csv(f"{TAB}/T7_chat_posttrain_pairs.csv").dropna(subset=["t"])
    Xg = np.c_[np.ones(len(PR)), PR.t - T_REF]
    Gb = []
    for _ in range(500):
        ii = RNG.integers(0, len(PR), len(PR))
        Gb.append(np.linalg.lstsq(Xg[ii], PR.gain.values[ii], rcond=None)[0])
    Gb = np.array(Gb)
    print(f"锚点 base: S={sb['S']:.2f} L̂={sb['L']:.3f} lc={sb['lc']:.2f} t={sb['t']:.2f} [{sb['models']}]")
    print(f"锚点 chat: S={sc_['S']:.2f} L̂={sc_['L']:.3f} lc={sc_['lc']:.2f} t={sc_['t']:.2f} [{sc_['models']}]")
    print(f"κ={KAPPA:.3f}; L*(lc) 斜率 @24.6: {(Lstar(24.7) - Lstar(24.5))[0] / 0.2:.4f} /dex")

    rows, sims = [], {}
    for sname, g in SCEN.items():
        for h in HORIZONS:
            for grp, st, G, pt in (("base", sb, None, "-"), ("chat", sc_, Gb, "linear"), ("chat", sc_, Gb, "plateau")):
                x, s_, t_ = simulate(st, g, h, J, P2D, noise=noise, G=G, posttrain=pt)
                sims[(sname, h, grp, pt)] = x
                q = qs(x)
                rows.append(dict(scenario=sname, g_dex_per_yr=g, horizon_months=int((h - DT) * 12), date=T0 + h,
                                 group=grp, posttrain=pt, S_anchor=st["S"], lc_path=st["lc"] + g * h,
                                 **q, mean=x.mean(), scale_part_p50=np.median(s_), tech_part_p50=np.median(t_),
                                 tech_share_p50=np.median(t_) / (np.median(s_) + np.median(t_))))
    FC = pd.DataFrame(rows)
    # 按技术形式拆开（base，看模型不确定性来源）
    rows2 = []
    for sname, g in SCEN.items():
        for h in HORIZONS:
            for form in ("M1b", "M1c"):
                pts = []
                for i in range(1500):
                    p2 = P2D[RNG.integers(len(P2D))]
                    pb = J[f"boot_{form}"][RNG.integers(len(J[f"boot_{form}"]))]
                    dL = Lstar(sb["lc"] + g * h, p2)[0] - Lstar(sb["lc"], p2)[0]
                    pts.append(sb["S"] + delta_S(form, pb, sb["L"], sb["t"], sb["L"] + dL, sb["t"] + h))
                rows2.append(dict(scenario=sname, horizon_months=int(h * 12), tech_form=form, **qs(np.array(pts))))
    FORM = pd.DataFrame(rows2)
    # 上界：纯规模天花板 & 算力"无限"
    pm, pc = J["M1b"], J["M1c"]
    caps = dict(M1b_scale_ceiling_at_T0=pm["lo"] + pm["amp"] + pm["tau"] * (sb["t"] - T_REF),
                M1c_ceiling=pc["lo"] + pc["amp"],
                infinite_compute_24m_M1b=sb["S"] + delta_S("M1b", pm, sb["L"], sb["t"], PAR["E"] + 0.0, sb["t"] + DT + 2),
                bridge_C6_hi=json.load(open(f"{IFACE}/P4_bridge_params.json"))["params"]["base|Average"][1])
    print("上界:", {k: round(v, 2) for k, v in caps.items()})

    # ---------------- 回测① 发布日轴 ----------------
    TB = 2024.5
    tr = base[base.t < TB]
    L, t, y, w = tr.Lh.values, tr.t.values - T_REF, tr[AVG].values, tr.w.values
    pb_tr, pc_tr = fit_add(L, t, y, w), fit_eff(L, t, y, w)
    s0, s1 = frontier_state(base, TB), frontier_state(base, T0)
    h_bt = T0 - TB
    g_act = (s1["lc"] - s0["lc"]) / h_bt
    bt = []
    for lab, g in (("情景 0.60", 0.60), ("情景 0.30", 0.30), (f"实际算力增速 {g_act:.2f}", g_act)):
        dL = Lstar(s0["lc"] + g * h_bt)[0] - Lstar(s0["lc"])[0]
        for form, p in (("M1b", pb_tr), ("M1c", pc_tr)):
            pred = s0["S"] + delta_S(form, p, s0["L"], s0["t"], s0["L"] + dL, s0["t"] + h_bt)
            bt.append(dict(backtest="发布日轴 2024.5→2025.25", setting=lab, tech_form=form, n_train=len(tr),
                           S_anchor=s0["S"], pred=pred, actual=s1["S"], error=pred - s1["S"]))
    # "神谕规模"：用前沿实际 L̂ 变化
    for form, p in (("M1b", pb_tr), ("M1c", pc_tr)):
        pred = s0["S"] + delta_S(form, p, s0["L"], s0["t"], s1["L"], s0["t"] + h_bt)
        bt.append(dict(backtest="发布日轴 2024.5→2025.25", setting="实际前沿 L̂（神谕规模）", tech_form=form,
                       n_train=len(tr), S_anchor=s0["S"], pred=pred, actual=s1["S"], error=pred - s1["S"]))
    # 朴素：前沿阶梯序列线性外推（2023.5–2024.5）
    tt = np.arange(2023.5, TB + 1e-9, 0.25); ff = [frontier_state(base, T)["S"] for T in tt]
    nb = np.polyfit(tt, ff, 1)
    bt.append(dict(backtest="发布日轴 2024.5→2025.25", setting="朴素线性外推(前沿序列)", tech_form="-",
                   n_train=len(tt), S_anchor=s0["S"], pred=np.polyval(nb, T0), actual=s1["S"],
                   error=np.polyval(nb, T0) - s1["S"]))
    # ---------------- 回测② 提交日轴（C1 月度前沿） ----------------
    M = pd.read_csv(f"{TAB}/T5_task_frontier_monthly.csv")
    fa = M["__" + AVG].values; mi = np.arange(len(fa))
    tr_i = mi < 7                                   # 2024-06..2024-12
    nb2 = np.polyfit(mi[tr_i], fa[tr_i], 1)
    for j in mi[~tr_i]:
        bt.append(dict(backtest="提交日轴 C1 月度前沿(全类型 top-5)", setting=f"线性趋势 → {M.month[j]}",
                       tech_form="-", n_train=7, S_anchor=fa[6], pred=np.polyval(nb2, j), actual=fa[j],
                       error=np.polyval(nb2, j) - fa[j]))
    BT = pd.DataFrame(bt)
    print(BT.round(2).to_string(index=False))
    print(f"回测训练集重估：M1b τ={pb_tr['tau']:.2f}, M1c δ={pc_tr['delta']:.3f}；实际前沿算力增速 {g_act:.2f} dex/yr")

    print(FC[["scenario", "horizon_months", "group", "posttrain", "S_anchor", "p05", "p25", "p50", "p75", "p95",
              "tech_share_p50"]].round(2).to_string(index=False))
    print(FORM.round(2).to_string(index=False))

    FC.to_csv(f"{IFACE}/P4_frontier_forecast.csv", index=False, encoding="utf-8-sig")
    FORM.to_csv(f"{TAB}/T8_forecast_by_techform.csv", index=False, encoding="utf-8-sig")
    BT.to_csv(f"{TAB}/T8_backtest.csv", index=False, encoding="utf-8-sig")
    json.dump(dict(T0=T0, T_NOW=T_NOW, DT=DT, kappa=KAPPA, L_ctx=LCTX, anchor_base=sb, anchor_chat=sc_, noise_sd=noise,
                   caps=caps, g_actual_backtest=g_act, compute_slopes=sl,
                   backtest_fit=dict(M1b=pb_tr, M1c=pc_tr)),
              open(f"{TAB}/T8_forecast_meta.json", "w"), ensure_ascii=False, indent=1, default=float)

    # ---------------- 作图 ----------------
    plt = setup_cjk_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cols = {"基线 0.60 dex/yr": "#d7301f", "减半 0.30 dex/yr": "#fd8d3c", "1/4 0.15 dex/yr": "#2b8cbe"}
    for ax, grp, st, df_, pt in ((axes[0], "base", sb, base, "-"), (axes[1], "chat", sc_, chat, "linear")):
        hist = [(T, frontier_state(df_, T)["S"]) for T in np.arange(2023.0, T0 + 1e-9, 0.125)]
        ax.step(*zip(*hist), where="post", color="k", lw=2, label=f"{grp} 累计前沿（top-{TOPK}）")
        ax.scatter(df_.t, df_[AVG], s=8, c="gray", alpha=.4)
        for sname, g in SCEN.items():
            xs = [T0] + [T0 + h for h in HORIZONS]
            q = [qs(np.array([st["S"]]))] + [qs(sims[(sname, h, grp, pt)]) for h in HORIZONS]
            ax.plot(xs, [a["p50"] for a in q], color=cols[sname], marker="o", label=sname)
            ax.fill_between(xs, [a["p05"] for a in q], [a["p95"] for a in q], color=cols[sname], alpha=.12)
            ax.fill_between(xs, [a["p25"] for a in q], [a["p75"] for a in q], color=cols[sname], alpha=.22)
        if grp == "base":
            ax.axhline(caps["M1b_scale_ceiling_at_T0"], color="gray", ls=":", lw=1)
            ax.text(2023.05, caps["M1b_scale_ceiling_at_T0"] + .6, "纯规模天花板（M1b，t=T0）", fontsize=7)
        else:
            q = qs(sims[("基线 0.60 dex/yr", 3.5, "chat", "plateau")])
            ax.errorbar(T0 + 3.55, q["p50"], yerr=[[q["p50"] - q["p05"]], [q["p95"] - q["p50"]]], fmt="s",
                        color="purple", capsize=3, label="基线·后训练增益停滞")
        ax.set_xlabel("时间（发布日）"); ax.set_ylabel("Average"); ax.legend(fontsize=7, loc="upper left")
        ax.set_title(f"{grp} 开源前沿预测（中位数，50%/90% 区间）")
    ax = axes[2]
    b1 = BT[BT.backtest.str.startswith("发布日")]
    lab = [f"{r.setting}\n{r.tech_form}" for r in b1.itertuples()]
    ax.barh(range(len(b1)), b1.error, color=["#2b8cbe" if f == "M1b" else "#31a354" if f == "M1c" else "gray"
                                             for f in b1.tech_form])
    ax.set_yticks(range(len(b1))); ax.set_yticklabels(lab, fontsize=7)
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel(f"预测 − 实际（实际 {s1['S']:.1f}，锚点 {s0['S']:.1f}）")
    ax.set_title("回测：t<2024.5 训练 → 预测 2025.25 base 前沿")
    fig.suptitle("图14  开源模型前沿能力情景预测与回测")
    fig.tight_layout(); fig.savefig(f"{FIG}/F14_forecast.png"); plt.close(fig)
    print("done")
