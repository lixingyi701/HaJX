# -*- coding: utf-8 -*-
"""
q4_03_bridge.py — 问题四·Step2：Loss → Benchmark 桥接映射（C6，按可比性分层）

映射形式：S = lo + (hi−lo) / (1 + exp(k (L − L0)))  （得分有下界=随机/格式底噪、上界≤100）
拟合：给定 (k, L0) 时 S 对 (lo, hi) 线性 ⇒ 二维网格 + 两级加密 + 线性 WLS（无需 scipy）
      约束 0 ≤ lo ≤ hi ≤ 100
可比性分层：High（Pythia，同模型同验证集）权重 1；Medium（不同验证集、近似）权重 0.5；
      另做 Medium 权重 0.25/1.0 敏感性、High-only 拟合（只能锁定底噪）。
类型：主桥接只用 base（pretrained）——Loss 是预训练概念；chat 另拟合，差值即"后训练增益"。
检验：留一交叉验证（sigmoid vs 线性 vs 常数），bootstrap 200 次给参数不确定性（供 Step4 传播）。
产出：interface/P4_bridge_params.json, tables/T6_*.csv, figures/F11
"""
import json, re
import numpy as np
import pandas as pd
from q4_00_common import (C_DIR, TAB, FIG, IFACE, CACHE, law_loss, wls,
                          sigmoid_bridge, setup_cjk_matplotlib)

RNG = np.random.default_rng(2026)
DIMC = {"Average": "LB_Average", "IFEval": "LB_IFEval", "BBH": "LB_BBH", "MATH": "LB_MATH",
        "GPQA": "LB_GPQA", "MUSR": "LB_MUSR", "MMLU-PRO": "LB_MMLU_PRO"}


def _lohi_grid(x, S, w):
    """x: (G, n)。对每个网格点闭式 WLS 求 (lo, hi) 并投影到 0≤lo≤hi≤100，返回 lo, hi, sse"""
    u = 1 - x
    a11 = (w * u * u).sum(1); a12 = (w * u * x).sum(1); a22 = (w * x * x).sum(1)
    r1 = (w * u * S).sum(1); r2 = (w * x * S).sum(1)
    det = a11 * a22 - a12 ** 2
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    lo = (a22 * r1 - a12 * r2) / det
    hi = (a11 * r2 - a12 * r1) / det
    bad = (lo < 0) | (lo > hi) | (hi > 100)
    lo_b = np.clip(lo, 0, 100)
    hi_b = np.clip((r2 - lo_b * a12) / np.maximum(a22, 1e-12), lo_b, 100)
    lo = np.where(bad, lo_b, lo); hi = np.where(bad, hi_b, hi)
    sse = (w * (S - lo[:, None] - (hi - lo)[:, None] * x) ** 2).sum(1)
    return lo, hi, sse


def fit_sigmoid(L, S, w, levels=3):
    L, S, w = map(lambda a: np.asarray(a, float), (L, S, w))
    kg = np.exp(np.linspace(np.log(0.5), np.log(60), 50))
    Lg = np.linspace(1.2, 3.2, 81)
    for _ in range(levels):
        K, L0 = [a.ravel() for a in np.meshgrid(kg, Lg)]
        x = 1 / (1 + np.exp(np.clip(K[:, None] * (L[None] - L0[:, None]), -50, 50)))
        lo, hi, sse = _lohi_grid(x, S, w)
        i = int(np.argmin(sse))
        best = np.array([lo[i], hi[i], K[i], L0[i]])
        kg = K[i] * np.exp(np.linspace(-0.3, 0.3, 21))
        Lg = L0[i] + np.linspace(-0.05, 0.05, 21)
    return best


def wrmse(y, yh, w):
    return float(np.sqrt(np.sum(w * (y - yh) ** 2) / np.sum(w)))


def load_c6():
    b = pd.read_csv(f"{C_DIR}/loss_benchmark_bridge_expanded.csv")
    b["tier"] = np.where(b.Loss_Comparability.str.startswith("High"), "High", "Medium")
    # 类型：C6 行名带 Instruct/Chat/-it 的为后训练模型，其余为 base（与 C1 Type 一致性已抽查）
    b["group"] = np.where(b.Model.str.contains(r"instruct|chat|-it\b|-it-", case=False, regex=True),
                          "chat", "base")
    # 模型系列（同一 tokenizer/验证集口径）：去掉规模后缀
    b["series"] = [re.split(r"-\d+(?:\.\d+)?[bBmM]", m.split("/")[1])[0] for m in b.Model]
    return b


def within_series_slope(d, col="LB_Average"):
    """系列固定效应：S = α_series + γ·L，只用组内变异 ⇒ 排除跨系列 Loss 口径不可比"""
    s = d.groupby("series").filter(lambda g: g.Val_Loss.nunique() >= 2)
    Xd = pd.get_dummies(s.series).values.astype(float)
    X = np.c_[Xd, s.Val_Loss.values]
    beta = wls(X, s[col].values, s.w.values)
    return float(beta[-1]), len(s), s.series.nunique()


if __name__ == "__main__":
    b = load_c6()
    b["w"] = np.where(b.tier == "High", 1.0, 0.5)
    print(pd.crosstab(b.group, b.tier))
    # 陷阱核对：相关系数符号
    r_all = np.corrcoef(b.Val_Loss, b.LB_Average)[0, 1]
    r_hi = np.corrcoef(b[b.tier == "High"].Val_Loss, b[b.tier == "High"].LB_Average)[0, 1]
    r_med = np.corrcoef(b[b.tier == "Medium"].Val_Loss, b[b.tier == "Medium"].LB_Average)[0, 1]
    # High 层 Val_Loss 与问题二标度律 L̂ 的一致性（验证"同一尺度"）
    hi = b[b.tier == "High"]
    Lhat_hi = law_loss(hi.N_params_B.values * 1e9, hi.D_tokens_B.values * 1e9)
    print(f"r(Loss,Avg): 全体 {r_all:.3f} | High {r_hi:.3f} | Medium {r_med:.3f}; "
          f"High 层 Val_Loss vs 标度律 L̂ 最大差 {np.abs(Lhat_hi - hi.Val_Loss).max():.4f}")
    gam = {}
    for grp in ("base", "chat"):
        gam[grp] = within_series_slope(b[b.group == grp])
        print(f"系列内斜率 dS/dL ({grp}): {gam[grp][0]:.2f}（{gam[grp][1]} 行, {gam[grp][2]} 个系列）")

    rows, params = [], {}
    fits = {}
    for grp in ("base", "chat", "all"):
        d = b if grp == "all" else b[b.group == grp]
        for dim, col in DIMC.items():
            p = fit_sigmoid(d.Val_Loss, d[col], d.w)
            yh = sigmoid_bridge(d.Val_Loss, p)
            Xl = np.c_[np.ones(len(d)), d.Val_Loss]
            bl = np.linalg.lstsq(Xl * np.sqrt(d.w.values)[:, None], d[col] * np.sqrt(d.w.values), rcond=None)[0]
            rows.append(dict(group=grp, dim=dim, n=len(d), lo=p[0], hi=p[1], k=p[2], L0=p[3],
                             wRMSE=wrmse(d[col], yh, d.w),
                             RMSE_High=wrmse(d[d.tier == "High"][col], yh[(d.tier == "High").values],
                                             np.ones((d.tier == "High").sum())) if (d.tier == "High").any() else np.nan,
                             RMSE_Medium=wrmse(d[d.tier == "Medium"][col], yh[(d.tier == "Medium").values],
                                               np.ones((d.tier == "Medium").sum())),
                             wRMSE_linear=wrmse(d[col], Xl @ bl, d.w),
                             slope_at_1p9=-(p[1] - p[0]) * p[2] * np.exp(p[2] * (1.9 - p[3]))
                             / (1 + np.exp(p[2] * (1.9 - p[3]))) ** 2))
            params[f"{grp}|{dim}"] = p.tolist()
            fits[(grp, dim)] = p
    T = pd.DataFrame(rows)
    print(T.round(3).to_string(index=False))

    # ---------- 留一交叉验证（base|Average 主桥接） ----------
    d = b[b.group == "base"].reset_index(drop=True)
    loo = []
    for i in range(len(d)):
        tr = d.drop(i)
        p = fit_sigmoid(tr.Val_Loss, tr.LB_Average, tr.w, levels=2)
        Xl = np.c_[np.ones(len(tr)), tr.Val_Loss]
        bl = np.linalg.lstsq(Xl, tr.LB_Average, rcond=None)[0]
        loo.append(dict(Model=d.Model[i], tier=d.tier[i], y=d.LB_Average[i],
                        pred_sigmoid=sigmoid_bridge(d.Val_Loss[i], p),
                        pred_linear=bl[0] + bl[1] * d.Val_Loss[i],
                        pred_const=np.average(tr.LB_Average, weights=tr.w)))
    LOO = pd.DataFrame(loo)
    LOO.to_csv(f"{TAB}/T6_bridge_loo.csv", index=False, encoding="utf-8-sig")
    loo_s = {m: float(np.sqrt(((LOO.y - LOO[f"pred_{m}"]) ** 2).mean()))
             for m in ("sigmoid", "linear", "const")}
    print("LOO RMSE (base|Average):", {k: round(v, 3) for k, v in loo_s.items()})

    # ---------- 可比性分层敏感性 ----------
    sens = []
    for wm in (0.25, 0.5, 1.0):
        ww = np.where(d.tier == "High", 1.0, wm)
        p = fit_sigmoid(d.Val_Loss, d.LB_Average, ww)
        sens.append(dict(setting=f"Medium权重={wm}", lo=p[0], hi=p[1], k=p[2], L0=p[3],
                         S_at_1p85=sigmoid_bridge(1.85, p), S_at_2p0=sigmoid_bridge(2.0, p)))
    ph = fit_sigmoid(hi.Val_Loss, hi.LB_Average, np.ones(len(hi)))
    sens.append(dict(setting="仅High(Pythia 7点)", lo=ph[0], hi=ph[1], k=ph[2], L0=ph[3],
                     S_at_1p85=sigmoid_bridge(1.85, ph), S_at_2p0=sigmoid_bridge(2.0, ph)))
    SENS = pd.DataFrame(sens)
    SENS["note"] = ""
    SENS.loc[SENS.setting.str.startswith("仅High"), "note"] = \
        "High 层 Loss∈[2.09,2.60] 全落在底噪区，Avg 仅 5.1–6.1：只能锁定下界 lo，上升段不可识别"
    SENS.to_csv(f"{TAB}/T6_bridge_tier_sensitivity.csv", index=False, encoding="utf-8-sig")
    print(SENS.round(3).to_string(index=False))

    # ---------- bootstrap（供 Step4 误差传播） ----------
    boots = []
    for _ in range(200):
        idx = RNG.integers(0, len(d), len(d))
        dd = d.iloc[idx]
        boots.append(fit_sigmoid(dd.Val_Loss, dd.LB_Average, dd.w, levels=2))
    boots = np.array(boots)
    np.save(f"{CACHE}/bridge_boot_base_avg.npy", boots)
    p_main = fits[("base", "Average")]
    resid_sd = float(np.sqrt(np.average((d.LB_Average - sigmoid_bridge(d.Val_Loss, p_main)) ** 2, weights=d.w)))

    T.to_csv(f"{TAB}/T6_bridge_fits.csv", index=False, encoding="utf-8-sig")
    J = dict(form="S = lo + (hi-lo)/(1+exp(k(L-L0)))", params_order=["lo", "hi", "k", "L0"],
             weights=dict(High=1.0, Medium=0.5), main="base|Average", params=params,
             resid_sd_base_avg=resid_sd, loo_rmse=loo_s,
             corr_loss_avg=dict(all=r_all, High=r_hi, Medium=r_med),
             within_series_slope={g: dict(slope=v[0], n=v[1], n_series=v[2]) for g, v in gam.items()},
             boot_p05=np.percentile(boots, 5, 0).tolist(), boot_p95=np.percentile(boots, 95, 0).tolist())
    json.dump(J, open(f"{IFACE}/P4_bridge_params.json", "w"), ensure_ascii=False, indent=2)

    # ---------------- 作图 ----------------
    plt = setup_cjk_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    Lg = np.linspace(1.6, 2.9, 200)
    ax = axes[0]
    mk = {("base", "High"): ("o", "#d7301f"), ("base", "Medium"): ("o", "#2b8cbe"),
          ("chat", "Medium"): ("^", "#31a354")}
    for (g, t), dd in b.groupby(["group", "tier"]):
        m, c = mk.get((g, t), ("s", "gray"))
        ax.scatter(dd.Val_Loss, dd.LB_Average, marker=m, color=c, s=28, label=f"{g}|{t}（{len(dd)}）")
    for bs in boots[:60]:
        ax.plot(Lg, sigmoid_bridge(Lg, bs), color="#2b8cbe", alpha=.05)
    ax.plot(Lg, sigmoid_bridge(Lg, p_main), color="#2b8cbe", lw=2.5, label="base 桥接（主）")
    ax.plot(Lg, sigmoid_bridge(Lg, fits[("chat", "Average")]), color="#31a354", lw=2, ls="--",
            label="chat 桥接")
    ax.set_xlabel("验证 Loss（C6）"); ax.set_ylabel("Open LLM Leaderboard Average")
    ax.set_title(f"Average 桥接（r={r_all:.2f}，负相关）"); ax.legend(fontsize=7)
    ax.invert_xaxis()
    ax = axes[1]
    cols = plt.cm.tab10(np.arange(6))
    for c, dim in zip(cols, list(DIMC)[1:]):
        ax.plot(Lg, sigmoid_bridge(Lg, fits[("base", dim)]), color=c, lw=2, label=dim)
        dd = b[b.group == "base"]
        ax.scatter(dd.Val_Loss, dd[DIMC[dim]], color=c, s=6, alpha=.5)
    ax.set_xlabel("验证 Loss"); ax.set_ylabel("维度得分"); ax.invert_xaxis()
    ax.set_title("分维度桥接（base）：MATH/GPQA 起跳晚"); ax.legend(fontsize=7)
    ax = axes[2]
    ax.scatter(LOO.y, LOO.pred_sigmoid, s=18, label=f"sigmoid  RMSE={loo_s['sigmoid']:.2f}")
    ax.scatter(LOO.y, LOO.pred_linear, s=18, marker="x", label=f"线性  RMSE={loo_s['linear']:.2f}")
    ax.plot([0, 45], [0, 45], "k:", lw=1)
    ax.set_xlabel("实际 Average"); ax.set_ylabel("留一预测"); ax.legend(fontsize=8)
    ax.set_title("留一交叉验证（base）")
    fig.suptitle("图11  Loss→Benchmark 桥接映射（High 权重 1，Medium 权重 0.5）")
    fig.tight_layout(); fig.savefig(f"{FIG}/F11_bridge.png"); plt.close(fig)
    print("done")
