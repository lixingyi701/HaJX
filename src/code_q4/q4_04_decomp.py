# -*- coding: utf-8 -*-
"""
q4_04_decomp.py — 问题四·Step3：规模扩张 vs 非规模技术进步的分解

面板：C2（C1+Epoch 发布日/组织）× C4（训练算力 C、数据量 D、参数 N）联表，
      时间轴 = Epoch 发布日（真实发布），2021 起；D 缺失时 D = C/(6N)。
      base 与 chat 分开；chat 仅保留"官方/原厂"组织（与 base 同组织），剔除社区微调洪流。
      同一 C4 模型被 C1 多行命中时按 1/重复数 加权（防止重复计数）。

三种互相独立的口径（同一前沿窗口上比较"技术进步占比"）：
  M1 结构模型（主）：S = f(L̂(N,D)) + τ·(t−2023)，f 为 sigmoid 桥接，L̂ 为问题二标度律（参数固定）
       M1a  f 直接取 C6 桥接（Step 2）不重估，τ 由残差回归 —— 严格"桥接即插即用"
       M1b  f 形状在面板上与 τ 联合重估（加性技术项，主口径）
       M1c  有效 Loss 型：S = f(L̂ − δ·(t−2023))，技术进步 = 每年等效降低 Loss δ（自带饱和）
       前沿分解：Shapley（两种顺序平均），规模 = 只让 L̂ 按前沿实际演进，技术 = 只让 t 演进
  M2 效率前沿法（Epoch 口径）：在帕累托有效模型上 S = a + b·log10C + c·t，
       c/b = "每年算力等效倍数"（dex/yr），技术占比 = cΔt / (bΔlogC + cΔt)
  M3 分位数回归：同式在全部 base 上做 q=0.9（前沿）与 q=0.5（中位数）
另：C1 9 个月窗口回归（q4_02，提交日轴）作窗口内交叉核对；C3 历史点降权（0.2）敏感性；
    chat：配对 (base, chat) 算后训练增益 G 随发布时间的趋势（另一类非规模技术进步）。
产出：interface/P4_decomposition.csv、P4_decomp_models.json；tables/T7_*.csv；figures/F12, F13
"""
import json, re
import numpy as np
import pandas as pd
from q4_00_common import (C_DIR, TAB, FIG, IFACE, CACHE, AVG, law_loss, load_c1, load_c4_lm,
                          join_c1_c4, year_frac, wls, quantile_reg, setup_cjk_matplotlib)

RNG = np.random.default_rng(7)
T_REF = 2023.0
WINDOWS = [(2023.5, 2025.0), (2024.0, 2025.0), (2024.5, 2025.25)]   # 主窗口：2023 年中 → 2024 年末（此前 base 样本过稀）
TOPK = 3


# ---------------------------------------------------------------- 面板
def build_panel():
    e = load_c1(); lm = load_c4_lm()
    m = join_c1_c4(e, lm)
    P = m[m.D_use.notna() & m.group.isin(["base", "chat"]) & (m.N > 0)].copy()
    P["Lh"] = law_loss(P.N, P.D_use)
    P["lc"] = np.log10(P.C_use)
    P["t"] = year_frac(P.pub_date)
    P = P[(P.t >= 2021) & (P.D_use > 1e9)]                 # gpt2-large 误配 D=1e8 等剔除
    P["org"] = P.Model.str.split("/").str[0]
    orgs = set(P[P.group == "base"].org) - {"braindao"}     # braindao 为转存
    P = P[(P.group == "base") | P.org.isin(orgs)].copy()
    P["w"] = 1.0 / P.groupby(["group", "c4_model"]).Model.transform("count")
    return P.reset_index(drop=True), e


def Lstar_of_lc(lc, kappa=6.0):
    """算力 C=10^lc 下的最优 Loss（问题三同一成本式 C = κ·N·D，Q=1，N 网格搜索）"""
    N = np.logspace(8, 13, 2001)
    return float(np.min(law_loss(N, 10 ** lc / (kappa * N))))


def sig(L, k, L0):
    return 1.0 / (1.0 + np.exp(np.clip(k * (L - L0), -50, 50)))


# ---------------------------------------------------------------- 拟合器（向量化网格 + 闭式线性部分）
def fit_add(L, t, y, w, levels=3):
    """M1b：y = lo + amp·σ(k(L−L0)) + τ·t ；对 (k,L0) 网格，(lo,amp,τ) 闭式 WLS，要求 amp>0"""
    kg = np.exp(np.linspace(np.log(1), np.log(80), 40)); Lg = np.linspace(1.5, 2.5, 51)
    for _ in range(levels):
        K, L0 = [a.ravel() for a in np.meshgrid(kg, Lg)]
        x = sig(L[None], K[:, None], L0[:, None])                         # (G,n)
        X = np.stack([np.ones_like(x), x, np.broadcast_to(t, x.shape)], -1)  # (G,n,3)
        XtW = X * w[None, :, None]
        A = np.einsum("gni,gnj->gij", XtW, X) + 1e-9 * np.eye(3)
        bvec = np.einsum("gni,n->gi", XtW, y)
        beta = np.linalg.solve(A, bvec[..., None])[..., 0]
        sse = (w * (y - np.einsum("gni,gi->gn", X, beta)) ** 2).sum(1)
        sse[beta[:, 1] <= 0] = np.inf
        i = int(np.argmin(sse))
        best = dict(lo=beta[i, 0], amp=beta[i, 1], k=K[i], L0=L0[i], tau=beta[i, 2], sse=sse[i])
        kg = K[i] * np.exp(np.linspace(-0.25, 0.25, 15)); Lg = L0[i] + np.linspace(-0.04, 0.04, 17)
    return best


def fit_eff(L, t, y, w, levels=2):
    """M1c：y = lo + amp·σ(k(L − δ·t − L0))；(k,L0,δ) 网格，(lo,amp) 闭式"""
    kg = np.exp(np.linspace(np.log(1), np.log(80), 25)); Lg = np.linspace(1.5, 2.5, 41)
    dg = np.linspace(-0.03, 0.15, 37)
    for _ in range(levels):
        K, L0, De = [a.ravel() for a in np.meshgrid(kg, Lg, dg)]
        x = sig(L[None] - De[:, None] * t[None], K[:, None], L0[:, None])
        s0 = w.sum(); s1 = (w * x).sum(1); s11 = (w * x * x).sum(1)
        r0 = (w * y).sum(); r1 = (w * x * y).sum(1)
        det = s0 * s11 - s1 ** 2; det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        amp = (s0 * r1 - s1 * r0) / det; lo = (r0 - amp * s1) / s0
        sse = (w * (y - lo[:, None] - amp[:, None] * x) ** 2).sum(1)
        sse[amp <= 0] = np.inf
        i = int(np.argmin(sse))
        best = dict(lo=lo[i], amp=amp[i], k=K[i], L0=L0[i], delta=De[i], sse=sse[i])
        kg = K[i] * np.exp(np.linspace(-0.25, 0.25, 11)); Lg = L0[i] + np.linspace(-0.04, 0.04, 11)
        dg = De[i] + np.linspace(-0.01, 0.01, 11)
    return best


def f_add(p, L, t):
    return p["lo"] + p["amp"] * sig(L, p["k"], p["L0"]) + p["tau"] * t


def f_eff(p, L, t):
    return p["lo"] + p["amp"] * sig(L - p["delta"] * t, p["k"], p["L0"])


# ---------------------------------------------------------------- 前沿状态 & 分解
def frontier_state(d, T):
    """截至时点 T（发布日 < T）的累计前沿：Average 最高的 TOPK 个模型的均值状态"""
    q = d[d.t < T].nlargest(TOPK, AVG)
    return dict(S=q[AVG].mean(), L=q.Lh.mean(), lc=q.lc.mean(), t=q.t.mean(),
                models=";".join(q.Model))


def shapley(fun, L0, t0, L1, t1):
    """两因素 Shapley：规模(L) 与 技术(t)，两种顺序平均"""
    sc = 0.5 * ((fun(L1, t0) - fun(L0, t0)) + (fun(L1, t1) - fun(L0, t1)))
    te = 0.5 * ((fun(L0, t1) - fun(L0, t0)) + (fun(L1, t1) - fun(L1, t0)))
    return sc, te


def reg_share(b, c, s0, s1):
    """线性口径 S = a + b·lc + c·t 的前沿增益分解"""
    sc = b * (s1["lc"] - s0["lc"]); te = c * (s1["t"] - s0["t"])
    return sc, te


def pareto(d):
    """效率前沿：不存在 '算力更低/相等、发布更早/相等、得分更高' 的模型"""
    keep = []
    for i, r in d.iterrows():
        dom = d[(d.lc <= r.lc) & (d.t <= r.t) & (d[AVG] > r[AVG])]
        if len(dom) == 0:
            keep.append(i)
    return d.loc[keep]


def run_all(d, bridge_p, boot=False):
    """对一个 base 面板 d 计算三口径的参数；返回 dict"""
    L, t, y, w = d.Lh.values, d.t.values - T_REF, d[AVG].values, d.w.values
    out = {}
    # M1a：C6 桥接固定
    lo, hi, k, L0 = bridge_p
    r = y - (lo + (hi - lo) * sig(L, k, L0))
    X = np.c_[np.ones(len(d)), t]; ba = wls(X, r, w)
    out["M1a"] = dict(lo=lo + ba[0], amp=hi - lo, k=k, L0=L0, tau=ba[1])
    out["M1b"] = fit_add(L, t, y, w, levels=2 if boot else 3)
    out["M1c"] = fit_eff(L, t, y, w, levels=2)
    pe = pareto(d)
    X = np.c_[np.ones(len(pe)), pe.lc, pe.t - T_REF]; out["M2"] = wls(X, pe[AVG].values, pe.w.values)
    X = np.c_[np.ones(len(d)), d.lc, t]
    out["M3_q75"] = quantile_reg(X, y, 0.75)
    out["M3_q90"] = quantile_reg(X, y, 0.9); out["M3_q50"] = quantile_reg(X, y, 0.5)
    out["OLS_lc"] = wls(X, y, w)
    return out


def shares(out, d, win):
    s0, s1 = frontier_state(d, win[0]), frontier_state(d, win[1])
    res = {}
    for mname in ("M1a", "M1b"):
        p = out[mname]
        res[mname] = shapley(lambda L, t: f_add(p, L, t - T_REF), s0["L"], s0["t"], s1["L"], s1["t"])
    p = out["M1c"]
    res["M1c"] = shapley(lambda L, t: f_eff(p, L, t - T_REF), s0["L"], s0["t"], s1["L"], s1["t"])
    for mname in ("M2", "M3_q75", "M3_q90", "M3_q50", "OLS_lc"):
        b = out[mname]; res[mname] = reg_share(b[1], b[2], s0, s1)
    return res, s0, s1


if __name__ == "__main__":
    P, e = build_panel()
    base = P[P.group == "base"].reset_index(drop=True)
    chat = P[P.group == "chat"].reset_index(drop=True)
    print(f"面板：base {len(base)} 行（{base.c4_model.nunique()} 个 C4 模型），chat {len(chat)} 行；"
          f"D 来源 {base.D_src.value_counts().to_dict()}")
    BR = json.load(open(f"{IFACE}/P4_bridge_params.json"))
    bridge_p = BR["params"]["base|Average"]

    out = run_all(base, bridge_p)
    for k_, v in out.items():
        print(k_, {a: round(float(b), 4) for a, b in v.items()} if isinstance(v, dict) else np.round(v, 4))

    # 拟合优度
    L, t, y, w = base.Lh.values, base.t.values - T_REF, base[AVG].values, base.w.values
    gof = []
    for mname, fun in (("M1a", lambda: f_add(out["M1a"], L, t)), ("M1b", lambda: f_add(out["M1b"], L, t)),
                       ("M1c", lambda: f_eff(out["M1c"], L, t)),
                       ("仅规模(M1b 去τ重估)", None), ("OLS_lc", lambda: np.c_[np.ones(len(base)), base.lc, t] @ out["OLS_lc"])):
        if fun is None:
            p0 = fit_add(L, np.zeros_like(t), y, w); yh = f_add(p0, L, np.zeros_like(t))
        else:
            yh = fun()
        gof.append(dict(model=mname, wRMSE=float(np.sqrt(np.sum(w * (y - yh) ** 2) / w.sum())),
                        R2=float(1 - np.sum(w * (y - yh) ** 2) / np.sum(w * (y - np.average(y, weights=w)) ** 2))))
    GOF = pd.DataFrame(gof); print(GOF.round(3).to_string(index=False))

    # ---------------- bootstrap（按 C4 模型簇重抽样） ----------------
    clusters = base.c4_model.unique()
    boots = []
    for bi in range(120):
        cs = RNG.choice(clusters, len(clusters), replace=True)
        db = pd.concat([base[base.c4_model == c] for c in cs], ignore_index=True)
        try:
            ob = run_all(db, bridge_p, boot=True)
        except Exception:
            continue
        boots.append(ob)
    print("bootstrap 成功次数", len(boots))
    pickle_path = f"{CACHE}/decomp_boots.pkl"
    pd.to_pickle(boots, pickle_path)

    # ---------------- 前沿分解表 ----------------
    rows = []
    for win in WINDOWS:
        res, s0, s1 = shares(out, base, win)
        bres = [shares(ob, base, win)[0] for ob in boots]
        for mname, (sc, te) in res.items():
            sh = te / (sc + te) if abs(sc + te) > 1e-9 else np.nan
            bs = np.array([b_[mname][1] / (b_[mname][0] + b_[mname][1]) for b_ in bres
                           if abs(b_[mname][0] + b_[mname][1]) > 1e-9])
            rows.append(dict(window=f"{win[0]:.2f}→{win[1]:.2f}", method=mname,
                             S_start=s0["S"], S_end=s1["S"], dS_actual=s1["S"] - s0["S"],
                             L_start=s0["L"], L_end=s1["L"], lc_start=s0["lc"], lc_end=s1["lc"],
                             t_start=s0["t"], t_end=s1["t"],
                             scale_contrib=sc, tech_contrib=te, dS_model=sc + te,
                             tech_share=sh, tech_share_p05=np.nanpercentile(bs, 5) if len(bs) else np.nan,
                             tech_share_p95=np.nanpercentile(bs, 95) if len(bs) else np.nan,
                             frontier_start=s0["models"], frontier_end=s1["models"]))
    DEC = pd.DataFrame(rows)
    main = DEC[DEC.window == f"{WINDOWS[0][0]:.2f}→{WINDOWS[0][1]:.2f}"].set_index("method")
    print(DEC[["window", "method", "dS_actual", "scale_contrib", "tech_contrib", "tech_share",
               "tech_share_p05", "tech_share_p95"]].round(3).to_string(index=False))
    core = main.loc[["M1b", "M2", "M3_q75"], "tech_share"]
    spread = float(core.max() - core.min())
    print(f"三口径（M1b/M2/M3_q75）技术占比极差 = {spread * 100:.1f} pp")

    # 算力等效速率（dex/yr）：M2/M3 为 c/b；M1b 为 τ / (df/dlogC 在前沿)
    Lf, tf = main.loc["M1b", "L_end"], main.loc["M1b", "t_end"]
    pb = out["M1b"]
    lcf = main.loc["M1b", "lc_end"]
    dLdlc = (Lstar_of_lc(lcf + 0.05) - Lstar_of_lc(lcf - 0.05)) / 0.1
    dfdL = (f_add(pb, Lf + 1e-3, 0) - f_add(pb, Lf - 1e-3, 0)) / 2e-3
    rate = dict(M1b=pb["tau"] / (dfdL * dLdlc), M1c=out["M1c"]["delta"] / abs(dLdlc), M2=out["M2"][2] / out["M2"][1],
                M3_q75=out["M3_q75"][2] / out["M3_q75"][1],
                M3_q90=out["M3_q90"][2] / out["M3_q90"][1], M3_q50=out["M3_q50"][2] / out["M3_q50"][1],
                OLS_lc=out["OLS_lc"][2] / out["OLS_lc"][1])
    print("算力等效技术进步速率 (dex/yr):", {k_: round(float(v), 3) for k_, v in rate.items()})

    # ---------------- 9 个月窗口核对（q4_02 T5） ----------------
    T5 = pd.read_csv(f"{TAB}/T5_desat_vs_average.csv")
    win9 = T5[T5.metric.str.contains("回归")][["metric", "n", "coef_log10N", "coef_month", "month_equiv_dex"]]

    # ---------------- C3 历史点降权敏感性（logN 口径，C3 只有参数量） ----------------
    c3 = pd.read_csv(f"{C_DIR}/leaderboard_extended_timeseries.csv")
    h = c3[c3.Source.str.startswith("Hist")].copy()
    dims = ["IFEval", "BBH", "MATH_Lvl5", "GPQA", "MUSR", "MMLU_PRO"]
    r_c3 = float(np.corrcoef(h.Average, h[dims].mean(1))[0, 1])
    h = h.assign(t=h.Year + 0.5, lN=np.log10(h.Params_B * 1e9), S=h.Average, w=0.2)
    bb = base.assign(lN=np.log10(base.N), S=base[AVG])[["t", "lN", "S", "w"]]
    sens = []
    for lab, dd in (("仅C2×C4 base", bb), ("+C3历史点(权0.2)", pd.concat([bb, h[["t", "lN", "S", "w"]]])),
                    ("+C3历史点(权1.0)", pd.concat([bb, h[["t", "lN", "S"]].assign(w=1.0)]))):
        X = np.c_[np.ones(len(dd)), dd.lN, dd.t - T_REF]
        b_ = wls(X, dd.S.values, dd.w.values)
        sens.append(dict(setting=lab, n=len(dd), coef_log10N=b_[1], coef_year=b_[2], dex_per_year=b_[2] / b_[1]))
    SENS = pd.DataFrame(sens); SENS["C3_Avg_vs_dimmean_r"] = r_c3
    print(SENS.round(3).to_string(index=False))

    # ---------------- chat：后训练增益 G(t)（配对） ----------------
    bmap = e[e.group == "base"].drop_duplicates("Model").set_index("Model")
    pairs = []
    for mdl, s, pubd in e[e.group == "chat"].drop_duplicates("Model")[["Model", AVG, "pub_date"]].values:
        stem = re.sub(r"-(Instruct|instruct|Chat|chat|it)(-v?[\d.]+)?$", "", mdl)
        for cand in (stem, stem + "-Base", stem + "-base", stem + "-hf"):
            if cand in bmap.index and cand != mdl:
                pb_ = bmap.loc[cand]
                pairs.append(dict(chat=mdl, base=cand, S_chat=s, S_base=pb_[AVG], gain=s - pb_[AVG],
                                  pub=pb_.pub_date if pd.notna(pb_.pub_date) else pubd, N=pb_.N))
                break
    PR = pd.DataFrame(pairs); PR["t"] = year_frac(PR.pub)
    prt = PR[PR.t.notna()]
    Xg = np.c_[np.ones(len(prt)), prt.t - T_REF]
    g_b = wls(Xg, prt.gain.values)
    gb_boot = []
    for _ in range(500):
        ii = RNG.integers(0, len(prt), len(prt)); gb_boot.append(wls(Xg[ii], prt.gain.values[ii]))
    gb_boot = np.array(gb_boot)
    print(f"后训练增益：{len(PR)} 对（{len(prt)} 对有发布日），均值 {PR.gain.mean():.2f}；"
          f"G(t) = {g_b[0]:.2f} + {g_b[1]:.2f}·(t−2023)")
    PR.to_csv(f"{TAB}/T7_chat_posttrain_pairs.csv", index=False, encoding="utf-8-sig")
    # chat 面板上同样拟合 M1b（参考）
    pc = fit_add(chat.Lh.values, chat.t.values - T_REF, chat[AVG].values, chat.w.values)
    print("chat M1b:", {a: round(float(b), 3) for a, b in pc.items()})

    # chat 前沿分解：规模 + 预训练技术（沿用 base 的 M1b） + 后训练技术 ΔG
    rows_c = []
    for win in WINDOWS:
        s0, s1 = frontier_state(chat, win[0]), frontier_state(chat, win[1])
        sc, te = shapley(lambda L, t: f_add(out["M1b"], L, t - T_REF), s0["L"], s0["t"], s1["L"], s1["t"])
        dG = g_b[1] * (s1["t"] - s0["t"])
        tot = sc + te + dG
        rows_c.append(dict(window=f"{win[0]:.2f}→{win[1]:.2f}", S_start=s0["S"], S_end=s1["S"],
                           dS_actual=s1["S"] - s0["S"], scale=sc, pretrain_tech=te, posttrain_tech=dG,
                           dS_model=tot, tech_share=(te + dG) / tot if tot else np.nan,
                           frontier_start=s0["models"], frontier_end=s1["models"]))
    DC = pd.DataFrame(rows_c); print(DC.round(3).drop(columns=["frontier_start", "frontier_end"]).to_string(index=False))

    # ---------------- 输出 ----------------
    DEC["group"] = "base"
    DEC.to_csv(f"{IFACE}/P4_decomposition.csv", index=False, encoding="utf-8-sig")
    DC.to_csv(f"{TAB}/T7_chat_decomposition.csv", index=False, encoding="utf-8-sig")
    GOF.to_csv(f"{TAB}/T7_decomp_fit_quality.csv", index=False, encoding="utf-8-sig")
    SENS.to_csv(f"{TAB}/T7_c3_sensitivity.csv", index=False, encoding="utf-8-sig")
    win9.to_csv(f"{TAB}/T7_c1_window_check.csv", index=False, encoding="utf-8-sig")
    base.to_csv(f"{TAB}/T7_panel_base.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([dict(method=k_, dex_per_year=v) for k_, v in rate.items()]).to_csv(
        f"{TAB}/T7_compute_equiv_rate.csv", index=False, encoding="utf-8-sig")
    J = dict(T_REF=T_REF, TOPK=TOPK,
             M1a=out["M1a"], M1b={k_: float(v) for k_, v in out["M1b"].items()},
             M1c={k_: float(v) for k_, v in out["M1c"].items()},
             M2=list(map(float, out["M2"])), M3_q75=list(map(float, out["M3_q75"])), M3_q90=list(map(float, out["M3_q90"])),
             M3_q50=list(map(float, out["M3_q50"])),
             chat_M1b={k_: float(v) for k_, v in pc.items()},
             posttrain_G=list(map(float, g_b)), posttrain_G_boot_sd=gb_boot.std(0).tolist(),
             wRMSE_M1b=float(GOF.set_index("model").loc["M1b", "wRMSE"]),
             tech_share_spread_pp=spread * 100, compute_equiv_rate=rate,
             boot_M1b=[{k_: float(v) for k_, v in ob["M1b"].items()} for ob in boots],
             boot_M1c=[{k_: float(v) for k_, v in ob["M1c"].items()} for ob in boots])
    json.dump(J, open(f"{IFACE}/P4_decomp_models.json", "w"), ensure_ascii=False, indent=1)

    # ---------------- 作图 ----------------
    plt = setup_cjk_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    sc_ = ax.scatter(base.Lh, base[AVG], c=base.t, cmap="viridis", s=26, label="base（C2×C4）")
    ax.scatter(chat.Lh, chat[AVG], c=chat.t, cmap="viridis", marker="^", s=26, alpha=.6, label="chat（原厂）")
    Lg = np.linspace(1.8, 2.5, 200)
    for yr, ls in ((2022, ":"), (2023.5, "--"), (2025, "-")):
        ax.plot(Lg, f_add(out["M1b"], Lg, yr - T_REF), "k", ls=ls, lw=1.6, label=f"M1b 在 t={yr}")
    ax.invert_xaxis(); ax.set_xlabel("标度律预测 Loss  L̂(N,D)"); ax.set_ylabel("Average")
    ax.set_title("同 L̂ 下得分随发布时间上移 = 非规模技术进步"); ax.legend(fontsize=7)
    plt.colorbar(sc_, ax=ax, label="发布时间")
    ax = axes[1]
    pe = pareto(base)
    ax.scatter(base.lc, base[AVG], c="lightgray", s=16, label="base 全部")
    ax.scatter(pe.lc, pe[AVG], c=pe.t, cmap="viridis", s=36, edgecolor="k", label="效率前沿（帕累托）")
    lcg = np.linspace(20, 25.2, 50)
    for yr, ls in ((2022, ":"), (2023.5, "--"), (2025, "-")):
        ax.plot(lcg, out["M2"][0] + out["M2"][1] * lcg + out["M2"][2] * (yr - T_REF), "k", ls=ls, lw=1.3)
        ax.plot(lcg, out["M3_q75"][0] + out["M3_q75"][1] * lcg + out["M3_q75"][2] * (yr - T_REF),
                "#d7301f", ls=ls, lw=1)
    ax.plot([], [], "k", label="M2 效率前沿回归"); ax.plot([], [], "#d7301f", label="M3 q=0.75")
    ax.set_ylim(0, 45); ax.set_xlabel("log10 训练算力 C（FLOP）"); ax.set_ylabel("Average")
    ax.set_title("效率前沿法与分位数回归"); ax.legend(fontsize=7)
    ax = axes[2]
    for yr_, grp in (("C3 历史（Average 口径不一致，降权）", h),):
        ax.scatter(grp.Year + 0.5, grp.Average, marker="o", facecolors="none", edgecolors="#fd8d3c", s=30, label=yr_)
    ax.scatter(base.t, base[AVG], s=10, c="#2b8cbe", alpha=.5, label="C2×C4 base")
    ax.scatter(chat.t, chat[AVG], s=10, c="#31a354", alpha=.5, marker="^", label="C2×C4 chat")
    fr = [(T, frontier_state(base, T)["S"]) for T in np.arange(2021.5, 2025.3, 0.25)]
    ax.step(*zip(*fr), where="post", color="#2b8cbe", lw=2, label="base 累计前沿（top-3）")
    ax.set_xlabel("发布时间（Epoch）"); ax.set_ylabel("Average"); ax.legend(fontsize=7)
    ax.set_title("长期时间轴：C3 + C2×C4")
    fig.suptitle("图12  规模 vs 技术进步：结构模型、效率前沿与分位数回归")
    fig.tight_layout(); fig.savefig(f"{FIG}/F12_decomp_panel.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    ax = axes[0]
    labs = {"M1a": "M1a 桥接即插", "M1b": "M1b 结构(主)", "M1c": "M1c 有效Loss",
            "M2": "M2 效率前沿", "M3_q75": "M3 q=0.75", "M3_q90": "M3 q=0.9(不稳)", "M3_q50": "M3 q=0.5", "OLS_lc": "OLS 均值"}
    mm = main.reindex(list(labs))
    xs = np.arange(len(mm))
    ax.bar(xs, mm.scale_contrib, color="#2b8cbe", label="规模扩张贡献")
    ax.bar(xs, mm.tech_contrib, bottom=np.where(mm.tech_contrib > 0, mm.scale_contrib.clip(lower=0), 0),
           color="#31a354", label="非规模技术进步贡献")
    ax.axhline(main.dS_actual.iat[0], color="k", ls="--", lw=1, label=f"实际前沿增益 {main.dS_actual.iat[0]:.1f}")
    for x_, r in zip(xs, mm.itertuples()):
        ax.text(x_, max(r.scale_contrib, 0) + max(r.tech_contrib, 0) + 0.5, f"{r.tech_share * 100:.0f}%",
                ha="center", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(list(labs.values()), rotation=20, fontsize=8)
    ax.set_ylabel("Average 分"); ax.set_title(f"base 前沿 {WINDOWS[0][0]}→{WINDOWS[0][1]}：各口径分解（百分数=技术占比）")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.scatter(prt.t, prt.gain, s=22, c="#31a354")
    tg = np.linspace(2023, 2025.2, 20)
    ax.plot(tg, g_b[0] + g_b[1] * (tg - T_REF), "k--", label=f"G(t)={g_b[0]:.1f}+{g_b[1]:.1f}·(t−2023)")
    ax.axhline(0, color="gray", lw=.8)
    ax.set_xlabel("发布时间"); ax.set_ylabel("chat − base（Average）")
    ax.set_title(f"后训练增益随时间上升（{len(prt)} 对原厂配对）"); ax.legend(fontsize=8)
    fig.suptitle("图13  前沿增益分解与后训练技术进步")
    fig.tight_layout(); fig.savefig(f"{FIG}/F13_decomp_bars.png"); plt.close(fig)
    print("done")
