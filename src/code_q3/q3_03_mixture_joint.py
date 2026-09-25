# -*- coding: utf-8 -*-
"""
q3_03_mixture_joint.py — 问题三·Step6：配比 p 的处理（主口径固定 p*，副口径联立）

联立模型（副口径）：
    L_joint(N,D,Q,p) = L(N,D,Q) · m(p)
    m(p)  = Σ_v w_v L̂_v(p) / Σ_v w_v L̂_v(p*)             —— 问题一配比模型（GBDT on clr(p)，
            即搜索 p* 时所用的同一模型），w_v = 13 验证域等权（与 P1 主口径一致）
    Q0(p) = min( Σ_i p_i Q_B,i , 1 )                        —— 配比决定基线质量（B 口径）
    其余成本结构不变：提质从 Q0(p) 起算。
含义：p 同时通过两条渠道影响目标——①配比效应 m(p)（迁移矩阵）②改变提质起点 Q0(p)。
做法：以 p* 为中心的多浓度 Dirichlet 采样 4000 个 p（不外推到训练配比空间之外），
     对每个 p 解一次 (N,Q) 最优（求解器已向量化），比较最优 p 与 p* 的差距。
产出：tables/T3_mixture_joint.csv, figures/F8_mixture_joint.png
"""
import numpy as np
import pandas as pd
from q3_00_model import G_LIST, TAB, FIG, ROOT, solve, setup_cjk_matplotlib
import sys, os
sys.path.insert(0, os.path.join(ROOT, "code_q1"))
from q1_00_common import clr  # noqa: E402

RNG = np.random.default_rng(2026)
IF1 = f"{ROOT}/outputs_q1/interface"

if __name__ == "__main__":
    ps = pd.read_csv(f"{IF1}/P1_p_star.csv")

    dom = ps.domain.tolist()

    import pickle
    from q1_00_common import gbdt_predict  # noqa: E402
    with open(f"{ROOT}/outputs_q1/cache/step5_gbdt.pkl", "rb") as f:
        gb_models = pickle.load(f)          # 问题一搜索 p* 所用的 13 个验证域 GBDT（clr 输入）
    Q_B = ps.Q_domain.values          # 主口径 Q_B = q̄，不再除以 0.5608
    p_star = ps.p_star_eqweight.values
    p_uni = ps.p_uniform.values

    def m_raw(P):
        """13 验证域等权的预测 Loss。用 GBDT 而非线性 clr 模型：线性模型在某分量→0 时
        clr→−∞，会外推出 m≈0.01 的荒谬值；树模型输出有界，且与问题一求 p* 的模型一致。"""
        Z = clr(P)
        return np.exp(np.stack([gbdt_predict(mm, Z) for mm in gb_models], 1)).mean(1)
    m_star = m_raw(p_star[None, :])[0]
    def m(P): return m_raw(P) / m_star
    def Q0p(P): return np.minimum(P @ Q_B, 1.0)

    parts = [p_star[None, :], p_uni[None, :]]
    for conc in (5, 20, 50, 200):          # 浓度越大越贴近 p*
        parts.append(RNG.dirichlet(np.maximum(p_star * conc, 1e-3), 1000))
    P = np.vstack(parts)
    mP, Q0P = m(P), Q0p(P)
    print(f"采样 {len(P)} 个配比；m(p) 范围 [{mP.min():.3f}, {mP.max():.3f}]，"
          f"Q0(p) 范围 [{Q0P.min():.3f}, {Q0P.max():.3f}]；m(uniform)={mP[1]:.4f}")

    rows, best, Js = [], {}, {}
    for g in G_LIST:
        for lc in (19, 22, 24):
            r = solve(np.full(len(P), 10.0 ** lc), g, 2048, Q0P)
            J = r["L"] * mP
            Js[(g, lc)] = J
            k = int(np.argmin(J))
            top = np.argsort(J)[:100]                            # 与问题一同法：top-100 平均
            p100 = P[top].mean(0); p100 /= p100.sum()
            r100 = solve([10.0 ** lc], g, 2048, Q0p(p100[None, :]))
            J100 = r100["L"][0] * m(p100[None, :])[0]
            dist = np.abs(P[k] - p_star).sum() / 2              # 全变差距离
            rows.append(dict(g=g, log10C=lc,
                             L_joint_pstar=J[0], L_joint_best=J[k],
                             gain_pct=(J[0] - J[k]) / J[0] * 100,
                             gain_from_m_pct=(1 - mP[k]) * 100,
                             gain_top100_pct=(J[0] - J100) / J[0] * 100,
                             gain_scale_channel_pct=(r["L"][0] - r["L"][k]) / r["L"][0] * 100,
                             L_joint_uniform=J[1],
                             TV_best_vs_pstar=dist,
                             TV_top100_vs_pstar=np.abs(p100 - p_star).sum() / 2,
                             Q0_pstar=Q0P[0], Q0_best=Q0P[k],
                             m_best=mP[k], Qopt_best=r["Q"][k], regime_pstar=int(r["regime"][0]),
                             regime_best=int(r["regime"][k])))
            best[(g, lc)] = (P[k], J, r)
    T = pd.DataFrame(rows)
    # 可分离性检验：不同 (g, C) 下联立目标对 4000 个 p 的排序是否一致
    from q1_00_common import spearman
    keys = list(Js)
    rho = [spearman(Js[a], Js[b]) for i, a in enumerate(keys) for b in keys[i + 1:]]
    rho_m = [spearman(Js[a], mP) for a in keys]
    T["rho_rank_min_across_gC"] = min(rho)
    T["rho_J_vs_m"] = rho_m
    print(f"可分离性：不同 (g,C) 间联立目标的秩相关最小值 {min(rho):.4f}；"
          f"与纯配比效应 m(p) 的秩相关 {min(rho_m):.4f}~{max(rho_m):.4f}")
    T.to_csv(f"{TAB}/T3_mixture_joint.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(T.round(4).to_string(index=False))

    # 最优 p 的领域构成（取对数型 1e22 为代表），与 p* 对照
    pk = best[("对数型", 22)][0]
    comp = pd.DataFrame({"domain": dom, "p_star": p_star, "p_joint_best": pk,
                         "Q_B": Q_B}).sort_values("p_star", ascending=False)
    comp.to_csv(f"{TAB}/T3_mixture_joint_composition.csv", index=False, encoding="utf-8-sig")

    plt = setup_cjk_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    J = best[("指数型", 19)][1]
    sc = axes[0].scatter(Q0P, mP, c=J, s=3, cmap="viridis_r")
    axes[0].scatter(Q0P[0], mP[0], c="red", s=60, marker="*", label="p*")
    axes[0].scatter(Q0P[1], mP[1], c="k", s=40, marker="x", label="均匀配比")
    fig.colorbar(sc, ax=axes[0], label="联立目标 L·m(p)（指数型, C=1e19）")
    axes[0].set_xlabel("配比决定的基线质量 Q₀(p)"); axes[0].set_ylabel("配比效应 m(p)（p*=1）")
    axes[0].set_title("两条渠道的权衡：Q₀(p) 与 m(p)"); axes[0].legend()
    x = np.arange(len(comp)); w = .4
    axes[1].bar(x - w / 2, comp.p_star, w, label="p*（问题一）")
    axes[1].bar(x + w / 2, comp.p_joint_best, w, label="联立最优（对数型, 1e22）")
    axes[1].set_xticks(x); axes[1].set_xticklabels(comp.domain, rotation=70, fontsize=8)
    axes[1].set_ylabel("配比"); axes[1].legend()
    axes[1].set_title("联立最优配比与 p* 对照")
    fig.suptitle("图8  配比联立的副口径：最优 p 与预算、成本函数无关（可分离）⇒ 固定 p 合理")
    fig.tight_layout(); fig.savefig(f"{FIG}/F8_mixture_joint.png"); plt.close(fig)
    print("done")
