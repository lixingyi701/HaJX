# -*- coding: utf-8 -*-
"""
q3_01_optimize.py — 问题三主网格、关键 18 组互验与 F1–F4。

主网格：3 种 g × 7 档预算 × C7 五档 × 2 个同映射 Q0 = 210。
关键 18 组：主口径 Q0(p*) × 3 种 g × {1e19,1e22,1e24} × {8192,32768}。
互验：剖面黄金分割、粗网格、多起点 SLSQP。SLSQP 不替换主求解器。
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from q3_00_model import (
    PAR, ETA, LCTX_CRIT, Q0_MAIN, Q0_WEB, M_STAR, H_TOT_SCALE, EVIDENCE, G_FUNCS, G_LIST,
    LCTX_C7, TAB, FIG, IFACE, loss, solve, N_star_closed, profile_Fq, profile_loss_at_Q,
    c_crit, CLASS_TOL, setup_cjk_matplotlib,
)

C_MAIN = 10.0 ** np.arange(19, 26)
Q0S = {"主口径p*加权": Q0_MAIN, "副口径网页pile_cc": Q0_WEB}
KEY_C = (19, 22, 24)
KEY_L = (8192, 32768)


def _row_from_solve(q0name, Q0, g, Lc, r, i):
    return dict(
        scenario_id=f"{q0name}|{g}|1e{int(round(np.log10(r['C'][i])))}|L{Lc}",
        q_mapping="identity", model_shape="M4+F", evidence=EVIDENCE,
        Q0口径=q0name, Q0=Q0, g=g, L_ctx=Lc, m_star=M_STAR,
        C=r["C"][i], log10C=float(np.log10(r["C"][i])),
        N_opt=r["N"][i], D_opt=r["D"][i], Q_opt=r["Q"][i], L_opt=r["L"][i],
        D_over_N=r["DN"][i],
        C_train=r["C_train"][i], C_attn=r["C_attn"][i], C_quality=r["C_quality"][i],
        total_cost=r["total_cost"][i], residual=r["residual"][i],
        budget_use=r["budget_use"][i], budget_rel_resid=r["budget_rel_resid"][i],
        share_train=r["s_train"][i], share_attn=r["s_attn"][i], share_Q=r["s_Q"][i],
        state=r["state"][i], regime=int(r["regime"][i]),
        tied=bool(r["tied"][i]), bound_hit=bool(r["bound_hit"][i]),
    )


def run_grid(m=None):
    rows = []
    for q0name, Q0 in Q0S.items():
        for g in G_LIST:
            for Lc in LCTX_C7:
                r = solve(C_MAIN, g, Lc, Q0, m=m)
                for i in range(len(C_MAIN)):
                    rows.append(_row_from_solve(q0name, Q0, g, Lc, r, i))
    return pd.DataFrame(rows)


def brute_force(C, g, Lc, Q0, nN_per_dec=40, nQ=100, log_lo=6.0, log_hi=13.0):
    """粗网格。若最优点贴在 N 边界上，自动放宽一次再搜。"""
    def _once(lo, hi):
        n_grid = 10.0 ** np.linspace(lo, hi, int(round((hi - lo) * nN_per_dec)) + 1)[:, None]
        q_grid = np.linspace(Q0, 1.0, nQ)[None, :]
        gf = G_FUNCS[g][0]
        dg = np.maximum(gf(q_grid) - gf(Q0), 0.0)
        d = C / ((6.0 + ETA * Lc) * n_grid + dg)
        lm = loss(n_grid, d, q_grid)
        i, j = np.unravel_index(np.argmin(lm), lm.shape)
        return n_grid[i, 0], q_grid[0, j], lm[i, j], i, len(n_grid)

    n_star, q_star, l_star, i, nlen = _once(log_lo, log_hi)
    on_bound = i == 0 or i == nlen - 1
    if on_bound:
        n_star, q_star, l_star, i, nlen = _once(1.0, 17.0)
        on_bound = i == 0 or i == nlen - 1
    return n_star, q_star, l_star, on_bound


def solve_slsqp(C, g, Lc, Q0):
    """多起点 SLSQP，变量 (log10 N, Q)，D 由预算等式反解。"""
    kappa = 6.0 + ETA * Lc
    gf = G_FUNCS[g][0]
    g0 = float(gf(Q0))

    def obj(x):
        log_n, q = x
        n = 10.0 ** log_n
        dg = float(gf(q) - g0)
        d = C / (kappa * n + dg)
        return float(loss(n, d, q))

    q_mid = 0.5 * (Q0 + 1.0)
    starts = []
    for q in (Q0, q_mid, 1.0):
        n_guess = float(N_star_closed(C, q, kappa))
        starts.append([np.log10(n_guess), q])
    starts.append([np.clip(starts[0][0] - 0.5, 1.0, 17.0), Q0])
    starts.append([np.clip(starts[1][0] + 0.5, 1.0, 17.0), 1.0])
    best = None
    n_ok = 0
    for x0 in starts:
        res = minimize(obj, np.asarray(x0, float), method="SLSQP",
                       bounds=[(1.0, 17.0), (float(Q0), 1.0)],
                       options={"ftol": 1e-14, "maxiter": 300, "disp": False})
        if np.isfinite(res.fun):
            n_ok += 1
            if best is None or res.fun < best.fun:
                best = res
    if best is None:
        raise RuntimeError(f"SLSQP 无有限结果: g={g}, C={C}, L={Lc}")
    log_n, q = best.x
    n = 10.0 ** log_n
    dg = float(gf(q) - g0)
    d = C / (kappa * n + dg)
    return dict(N=n, Q=q, D=d, L=float(loss(n, d, q)), success=bool(best.success), n_starts=n_ok)


def run_scan():
    cs = 10.0 ** np.linspace(17, 26, 181)
    out = []
    for q0name, Q0 in Q0S.items():
        for g in G_LIST:
            for Lc in LCTX_C7:
                r = solve(cs, g, Lc, Q0)
                df = pd.DataFrame({k: np.asarray(v) for k, v in r.items()})
                df["log10C"] = np.log10(df["C"])
                df["Q0口径"] = q0name
                df["g"] = g
                df["L_ctx"] = Lc
                out.append(df)
    return pd.concat(out, ignore_index=True)


def finite_diff_check(opt):
    """在下界、内点、上界代表点上核对剖面导数。"""
    rows = []
    main = opt[opt.Q0口径 == "主口径p*加权"]
    picks = []
    for state, fallback_q in (("LOWER", None), ("INTERIOR", None), ("UPPER", None)):
        sub = main[main.state == state]
        if len(sub):
            rec = sub.iloc[0]
            picks.append((state, rec.g, float(rec.C), int(rec.L_ctx), float(rec.Q_opt), float(rec.Q0)))
    # 即使某全局状态未出现，也在三个质量位置核对剖面导数。
    picks.append(("profile_Q0", "对数型", 1e22, 2048, Q0_MAIN, Q0_MAIN))
    picks.append(("profile_mid", "对数型", 1e22, 2048, 0.5 * (Q0_MAIN + 1.0), Q0_MAIN))
    picks.append(("profile_Q1", "对数型", 1e22, 2048, 1.0, Q0_MAIN))
    for tag, g, c, lc, q, q0 in picks:
        analytic = profile_Fq(c, g, lc, q, q0)
        eps = 1e-5
        q_hi = min(1.0, q + eps)
        q_lo = max(q0, q - eps)
        if q_hi - q_lo < 1e-8:
            rows.append(dict(tag=tag, g=g, Q=q, rel_err=np.nan, note="端点步长不足"))
            continue
        fd = (profile_loss_at_Q(c, g, lc, q_hi, q0) - profile_loss_at_Q(c, g, lc, q_lo, q0)) / (q_hi - q_lo)
        abs_err = abs(fd - analytic)
        rel = abs_err / (abs(analytic) + 1e-8)
        rows.append(dict(tag=tag, g=g, C=c, L_ctx=lc, Q=q, Fq=analytic, fd=fd,
                         abs_err=abs_err, rel_err=rel))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    failures = []
    T = run_grid()
    T.to_csv(f"{TAB}/T1_optimal_all.csv", index=False, encoding="utf-8-sig")
    T.to_csv(f"{IFACE}/P3_optimal_config.csv", index=False, encoding="utf-8-sig")
    key_show = T[(T.L_ctx == 2048) & T.log10C.round().isin(KEY_C)]
    key_show.to_csv(f"{TAB}/T1_key_budgets_Lctx2048.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(key_show[["Q0口径", "g", "log10C", "N_opt", "D_opt", "Q_opt", "L_opt",
                    "state", "budget_use"]].to_string(index=False))

    share_err = (T.share_train + T.share_attn + T.share_Q - 1.0).abs().max()
    resid = T.budget_rel_resid.abs().max()
    print(f"份额和误差 {share_err:.3e}；预算相对残差 {resid:.3e}")
    if share_err > 1e-9 or resid > 1e-9:
        failures.append("预算残差或份额和超出 1e-9")
    if (T.N_opt <= 0).any() or (T.D_opt <= 0).any() or (T.Q_opt < T.Q0 - 1e-12).any() or (T.Q_opt > 1 + 1e-12).any():
        failures.append("出现非正规模或质量越界")
    if T.bound_hit.any():
        failures.append("主网格最优点贴住 logN 搜索边界")

    # 诊断：L=2048 的两起点粗网格，不代替关键 18 组。
    diag = []
    for q0name, Q0 in Q0S.items():
        for g in G_LIST:
            for lc in KEY_C:
                c = 10.0 ** lc
                nb, qb, lb, on_b = brute_force(c, g, 2048, Q0)
                r = solve([c], g, 2048, Q0)
                diag.append(dict(Q0口径=q0name, g=g, log10C=lc, L_ctx=2048,
                                 L_grid=lb, L_solver=r["L"][0], L_gap=lb - r["L"][0],
                                 grid_bound=on_b))
    diag = pd.DataFrame(diag)
    diag.to_csv(f"{TAB}/T1_solver_vs_grid_L2048.csv", index=False, encoding="utf-8-sig")

    # 关键 18 组：粗网格 + SLSQP
    rows18 = []
    for g in G_LIST:
        for lc in KEY_C:
            for Lctx in KEY_L:
                c = 10.0 ** lc
                prof = solve([c], g, Lctx, Q0_MAIN)
                nb, qb, lb, on_b = brute_force(c, g, Lctx, Q0_MAIN)
                alt = solve_slsqp(c, g, Lctx, Q0_MAIN)
                gap_grid = lb - prof["L"][0]
                gap_slsqp = alt["L"] - prof["L"][0]
                state_prof = prof["state"][0]
                # SLSQP 的状态用同一容差。
                q_alt = alt["Q"]
                if q_alt <= Q0_MAIN + CLASS_TOL:
                    state_alt = "LOWER"
                elif q_alt >= 1.0 - CLASS_TOL:
                    state_alt = "UPPER"
                else:
                    state_alt = "INTERIOR"
                disagree = state_prof != state_alt and state_prof != "TIED"
                rows18.append(dict(
                    g=g, log10C=lc, L_ctx=Lctx, Q0=Q0_MAIN,
                    L_profile=prof["L"][0], L_grid=lb, L_slsqp=alt["L"],
                    gap_grid=gap_grid, gap_slsqp=gap_slsqp,
                    state_profile=state_prof, state_slsqp=state_alt,
                    Q_profile=prof["Q"][0], Q_slsqp=q_alt, N_profile=prof["N"][0], N_slsqp=alt["N"],
                    grid_bound=on_b, slsqp_success=alt["success"], state_disagree=disagree,
                ))
    cross = pd.DataFrame(rows18)
    cross.to_csv(f"{TAB}/T1_key18_crosscheck.csv", index=False, encoding="utf-8-sig")
    print("关键 18 组 gap_grid", cross.gap_grid.min(), cross.gap_grid.max(),
          "gap_slsqp", cross.gap_slsqp.min(), cross.gap_slsqp.max())
    if (cross.gap_grid < -2e-5).any() or (cross.gap_slsqp < -2e-5).any():
        failures.append("关键 18 组上剖面解劣于粗网格或 SLSQP，超过 2e-5")
    if cross.grid_bound.any():
        failures.append("关键 18 组的粗网格仍贴住 N 边界")
    tied_cross = cross[cross.state_disagree & (cross.gap_slsqp.abs() <= 2e-5)]
    if len(tied_cross):
        print("状态不一致但损失差在 2e-5 内，按并列记录：", len(tied_cross))

    # Δg=0 闭式：直接比较固定 Q 的一维搜索，不依赖外层是否离开 Q0。
    cf = []
    for Lc in LCTX_C7:
        kappa = 6.0 + ETA * Lc
        for lc in KEY_C:
            c = 10.0 ** lc
            for q in (Q0_MAIN, 1.0):
                r = solve([c], "对数型", Lc, Q0=q, nQ=5, levels=1)
                # Q0=q 且只允许该点附近时，用闭式对照的是 Δg=0 的 N。
                n_closed = float(N_star_closed(c, q, kappa))
                # 强制 Δg=0 的剖面，避免 Q 离开起点。
                from q3_00_model import _profile
                lv, x = _profile(c, q, 0.0, kappa, PAR)
                n_num = float(10.0 ** np.reshape(x, ()))
                cf.append(dict(L_ctx=Lc, log10C=lc, Q=q, N_numeric=n_num,
                               N_closed=n_closed, rel_err=abs(n_num / n_closed - 1.0)))
    cf = pd.DataFrame(cf)
    cf.to_csv(f"{TAB}/T1_closed_form_check.csv", index=False, encoding="utf-8-sig")
    print("闭式相对误差最大", cf.rel_err.max())
    if cf.rel_err.max() > 1e-5:
        failures.append("无提质闭式 N* 相对误差超过 1e-5")

    # m=1 对照与 H_tot 缩放。不写入主接口。
    m1_rows = []
    for g in G_LIST:
        for Lc in LCTX_C7:
            r_main = solve(C_MAIN, g, Lc, Q0_MAIN)
            r_m1 = solve(C_MAIN, g, Lc, Q0_MAIN, m=1.0)
            r_ht = solve(C_MAIN, g, Lc, Q0_MAIN, m=1.0, loss_scale=H_TOT_SCALE)
            for i in range(len(C_MAIN)):
                m1_rows.append(dict(
                    g=g, L_ctx=Lc, log10C=float(np.log10(C_MAIN[i])),
                    N_mstar=r_main["N"][i], Q_mstar=r_main["Q"][i], L_mstar=r_main["L"][i],
                    N_m1=r_m1["N"][i], Q_m1=r_m1["Q"][i], L_m1=r_m1["L"][i],
                    N_htot=r_ht["N"][i], Q_htot=r_ht["Q"][i], L_htot=r_ht["L"][i],
                    htot_scale=H_TOT_SCALE,
                ))
    m1 = pd.DataFrame(m1_rows)
    m1.to_csv(f"{TAB}/T1_m1_htot_control.csv", index=False, encoding="utf-8-sig")
    cfg_gap = np.maximum(np.abs(m1.N_htot / m1.N_m1 - 1.0), np.abs(m1.Q_htot - m1.Q_m1)).max()
    loss_gap = (m1.L_htot / (m1.L_m1 * H_TOT_SCALE) - 1.0).abs().max()
    print(f"H_tot 与 m=1 配置相对差 {cfg_gap:.3e}，损失缩放相对差 {loss_gap:.3e}")
    # 正的损失缩放不改变最优配置。黄金分割在缩放后的浮点比较上允许 1e-6 量级的舍入差。
    if cfg_gap > 1e-6 or loss_gap > 1e-8:
        failures.append("H_tot 与 m=1 不同配置，或损失未按常数缩放")

    # 边界状态
    v_low, s_low = c_crit("对数型", 2048, Q0_MAIN, which="leave", lo=8.0, hi=12.0, return_status=True)
    v_one, s_one = c_crit("对数型", 2048, 1.0, which="leave", return_status=True)
    v_wide, s_wide = c_crit("对数型", 8192, Q0_MAIN, which="leave", return_status=True)
    boundary = [dict(case="low_budget_window", status=str(s_low[0]), log10C=v_low[0]),
                dict(case="Q0_eq_1", status=str(s_one[0]), log10C=v_one[0])]
    if str(s_wide[0]) == "FOUND" and np.isfinite(v_wide[0]):
        v_past, s_past = c_crit("对数型", 8192, Q0_MAIN, which="leave",
                                lo=float(v_wide[0]) + 0.5, hi=26.0, return_status=True)
        boundary.append(dict(case="window_after_leave", status=str(s_past[0]), log10C=v_past[0]))
        if str(s_past[0]) != "NOT_BRACKETED":
            failures.append(f"越过已知离开点后的窗口状态是 {s_past[0]}，期望 NOT_BRACKETED")
    else:
        boundary.append(dict(case="wide_leave", status=str(s_wide[0]), log10C=v_wide[0]))
    pd.DataFrame(boundary).to_csv(f"{TAB}/T1_boundary_status.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(boundary).to_string(index=False))
    if str(s_low[0]) != "NOT_IDENTIFIED":
        failures.append(f"低预算窗口状态是 {s_low[0]}，期望 NOT_IDENTIFIED")
    if str(s_one[0]) != "NO_QUALITY_RANGE":
        failures.append(f"Q0=1 状态是 {s_one[0]}，期望 NO_QUALITY_RANGE")

    fd = finite_diff_check(T)
    fd.to_csv(f"{TAB}/T1_profile_derivative.csv", index=False, encoding="utf-8-sig")
    print(fd.to_string(index=False))
    # 边界处差分是单侧的，F_Q 接近 0 时相对误差会放大。用绝对误差验收。
    if fd.abs_err.dropna().max() > 1e-4:
        failures.append("剖面导数有限差分绝对误差超过 1e-4")

    S = run_scan()
    S.to_pickle(f"{TAB}/T1_scan.pkl")
    S.to_csv(f"{TAB}/T1_scan.csv", index=False, encoding="utf-8-sig")

    plt = setup_cjk_matplotlib()
    col = {"指数型": "#d7301f", "幂函数型": "#2b8cbe", "对数型": "#31a354"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, (q0name, Q0) in zip(axes, Q0S.items()):
        for g in G_LIST:
            s = S[(S.Q0口径 == q0name) & (S.g == g) & (S.L_ctx == 2048)]
            ax.plot(s.log10C, s.Q, color=col[g], lw=2, label=g)
        ax.axhline(Q0, color="gray", ls=":", lw=1)
        ax.text(17.1, Q0, f" Q₀={Q0:.3f}", va="bottom", fontsize=9, color="gray")
        for v in KEY_C:
            ax.axvline(v, color="k", alpha=.15)
        ax.set_xlabel("log₁₀ C (FLOPs)")
        ax.set_ylabel("最优质量 Q*")
        ax.set_title(f"{q0name}（Q₀={Q0:.3f}，L_ctx=2048）")
        ax.legend(loc="best")
    fig.suptitle("图1  预算响应：Q* 随 log₁₀ C 的变化")
    fig.tight_layout()
    fig.savefig(f"{FIG}/F1_Qstar_vs_C.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.8), sharex=True, sharey=True)
    for r_, (q0name, Q0) in enumerate(Q0S.items()):
        for c_, g in enumerate(G_LIST):
            ax = axes[r_, c_]
            s = S[(S.Q0口径 == q0name) & (S.g == g) & (S.L_ctx == 2048)].sort_values("log10C")
            ax.stackplot(s.log10C, s.s_train, s.s_attn, s.s_Q,
                         colors=["#9ecae1", "#fdae6b", "#a1d99b"],
                         labels=["训练", "注意力", "提质"])
            use_lo, use_hi = float(s.budget_use.min()), float(s.budget_use.max())
            ax.text(0.02, 0.05, f"预算使用率 {use_lo:.6f}–{use_hi:.6f}",
                    transform=ax.transAxes, fontsize=7)
            ax.set_ylim(0.8, 1.02)
            ax.set_title(f"{g} | {q0name}", fontsize=10)
            if r_ == 1:
                ax.set_xlabel("log₁₀ C")
            if c_ == 0:
                ax.set_ylabel("占总消耗份额")
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.suptitle("图2  成本份额（分母=实际总消耗；预算使用率另注）")
    fig.tight_layout()
    fig.savefig(f"{FIG}/F2_cost_shares.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for g in G_LIST:
        s = S[(S.Q0口径 == "主口径p*加权") & (S.g == g) & (S.L_ctx == 2048)]
        axes[0].plot(s.log10C, np.log10(s.N), color=col[g], label=g)
        axes[1].plot(s.log10C, np.log10(s.D), color=col[g], label=g)
        axes[2].plot(s.log10C, s.DN, color=col[g], label=g)
    s = S[(S.Q0口径 == "主口径p*加权") & (S.g == "对数型") & (S.L_ctx == 2048)]
    kappa = 6 + ETA * 2048
    axes[0].plot(s.log10C, np.log10(N_star_closed(10 ** s.log10C, 1.0, kappa)),
                 "k--", lw=1, label="闭式(Q=1,Δg=0)")
    axes[0].set_ylabel("log₁₀ N*")
    axes[1].set_ylabel("log₁₀ D*")
    axes[2].set_ylabel("D*/N*")
    for ax in axes:
        ax.set_xlabel("log₁₀ C")
        ax.legend(fontsize=8)
    fig.suptitle(f"图3  预算响应：N*、D*、D*/N*（主口径 Q₀={Q0_MAIN:.3f}，L_ctx=2048）")
    fig.tight_layout()
    fig.savefig(f"{FIG}/F3_NDratio.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    lgrid = np.logspace(np.log10(1024), np.log10(262144), 60)
    for lc, ls in zip(KEY_C, (":", "-", "--")):
        c = 10.0 ** lc
        ns = np.array([solve([c], "对数型", L, Q0_MAIN)["N"][0] for L in lgrid])
        n0 = solve([c], "对数型", 0, Q0_MAIN)["N"][0]
        axes[0].plot(lgrid, ns / n0, "k" + ls, lw=1.5, label=f"数值 C=1e{lc}")
    axes[0].plot(lgrid, (6 / (6 + ETA * lgrid)) ** (PAR["be"] / (PAR["al"] + PAR["be"])),
                 color="#d7301f", lw=4, alpha=.35, label="解析 [6/(6+ηL)]^{β/(α+β)}")
    axes[0].axvline(LCTX_CRIT, color="purple", ls="-.", lw=1)
    axes[0].text(LCTX_CRIT * 1.08, 0.62, "L_crit=30000", color="purple", fontsize=9)
    for L in LCTX_C7:
        axes[0].axvline(L, color="gray", alpha=.25)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("L_ctx")
    axes[0].set_ylabel("N*(L_ctx)/N*(L_ctx=0)")
    axes[0].set_title("长上下文改变规模的数值解与无提质解析对照")
    axes[0].legend(fontsize=8)
    heat = T[(T.Q0口径 == "主口径p*加权") & (T.g == "对数型")].pivot(
        index="L_ctx", columns="log10C", values="L_opt")
    im = axes[1].imshow(heat.values, aspect="auto", cmap="viridis_r")
    axes[1].set_xticks(range(heat.shape[1]))
    axes[1].set_xticklabels([f"1e{int(c)}" for c in heat.columns])
    axes[1].set_yticks(range(heat.shape[0]))
    axes[1].set_yticklabels(heat.index)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            axes[1].text(j, i, f"{heat.values[i, j]:.3f}", ha="center", va="center", fontsize=7,
                         color="w" if heat.values[i, j] > np.nanmedian(heat.values) else "k")
    axes[1].axhline(2.5, color="purple", ls="-.", lw=1)
    axes[1].set_xlabel("预算 C")
    axes[1].set_ylabel("L_ctx（C7）")
    axes[1].set_title("最优 Loss（对数型，主口径；紫线在 8192 与 32768 之间）")
    fig.colorbar(im, ax=axes[1], shrink=.8)
    fig.suptitle("图4  上下文长度敏感性")
    fig.tight_layout()
    fig.savefig(f"{FIG}/F4_Lctx_sensitivity.png")
    plt.close(fig)

    if failures:
        print("验收未通过：")
        for item in failures:
            print(" -", item)
        sys.exit(1)
    print("q3_01 done")
