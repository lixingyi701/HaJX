# -*- coding: utf-8 -*-
"""
q4_99_check_report.py — 《问题三四结果汇报.md》关键数字自动核对

逐条从 outputs_q3 / outputs_q4 的 CSV/JSON 重新读取，与汇报中写的数字比对（按汇报的显示精度给容差）。
全部通过打印 ALL PASS；否则列出不一致项。重跑上游脚本后可用本脚本确认汇报是否需要更新。
"""
import json, os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
Q3T, Q3I = f"{ROOT}/outputs_q3/tables", f"{ROOT}/outputs_q3/interface"
Q4T, Q4I = f"{ROOT}/outputs_q4/tables", f"{ROOT}/outputs_q4/interface"
rd = lambda p: pd.read_csv(p)
res = []


def chk(name, got, want, tol):
    ok = abs(float(got) - want) <= tol
    res.append((ok, name, float(got), want))


# ---------------- 问题三 ----------------
cc = rd(f"{Q3T}/T2_ccrit_three_criteria.csv")
m = cc[(cc.L_ctx == 2048)].set_index(["Q0口径", "g"])
for (q, g), want in {("主口径p*加权", "指数型"): 20.610, ("主口径p*加权", "幂函数型"): 20.483,
                     ("主口径p*加权", "对数型"): 18.523, ("副口径网页c4", "指数型"): 20.100,
                     ("副口径网页c4", "幂函数型"): 20.199, ("副口径网页c4", "对数型"): 18.449}.items():
    chk(f"C_crit 离开Q0 {q} {g}", m.loc[(q, g), "A_leave_Q0"], want, 6e-4)
chk("三口径极差最大", cc.spread_ABC_dex.max(), 0.385, 6e-4)
chk("主口径三口径极差最大", cc[cc.Q0口径 == "主口径p*加权"].spread_ABC_dex.max(), 0.0627, 1e-3)
chk("131k 指数型 C_crit", cc[(cc.Q0口径 == "主口径p*加权") & (cc.g == "指数型") & (cc.L_ctx == 131072)].A_leave_Q0.iat[0],
    19.99, 6e-3)
bt = rd(f"{Q3T}/T2_bootstrap_ccrit.csv").set_index(["Q0口径", "g"])
chk("bootstrap p05 主指数", bt.loc[("主口径p*加权", "指数型"), "p05"], 20.594, 6e-4)
chk("bootstrap p95 主指数", bt.loc[("主口径p*加权", "指数型"), "p95"], 20.628, 6e-4)
chk("区制一致率最小", bt.filter(like="regime_agree").values.min(), 1.0, 1e-9)
kb = rd(f"{Q3T}/T1_key_budgets_Lctx2048.csv")
r = kb[(kb.Q0口径 == "主口径p*加权") & (kb.g == "指数型")].set_index("log10C")
chk("1e22 N*", r.loc[22, "N_opt"] / 1e9, 5.06, 6e-3)
chk("1e22 L*", r.loc[22, "L_opt"], 2.144, 6e-4)
chk("1e24 D/N", r.loc[24, "D_over_N"], 96.7, 0.06)
chk("1e22 s_Q", r.loc[22, "share_Q"] * 100, 0.87, 6e-3)
opt = rd(f"{Q3I}/P3_optimal_config.csv")
chk("最优解组数", len(opt), 210, 0)
chk("Q*=Q0 角点组数", (opt.regime == "Q0角点(不提质)").sum(), 34, 0)
chk("内点组数", opt.regime.str.startswith("内点").sum(), 3, 0)
cf = rd(f"{Q3T}/T1_closed_form_check.csv")
chk("闭式互验相对误差上界(×1e7)", ((cf.N_numeric - cf.N_closed).abs() / cf.N_closed).max() * 1e7, 1.83, 0.6)
sv = rd(f"{Q3T}/T1_solver_vs_grid.csv")
chk("网格互验 Loss 差上界(×1e5)", (sv.L_grid - sv.L_solver).abs().max() * 1e5, 1.58, 0.6)
lc = rd(f"{Q3I}/P3_sensitivity_Lctx.csv")
lc = lc[(lc.Q0口径 == "主口径p*加权") & (lc.g == "指数型") & (lc.log10C == 22)].set_index("L_ctx")
chk("131k N 比", lc.loc[131072, "N_ratio_vs2048"], 0.480, 6e-4)
chk("131k 注意力占比%", lc.loc[131072, "share_attn"] * 100, 81.1, 0.06)
chk("32k 注意力占比%", lc.loc[32768, "share_attn"] * 100, 51.9, 0.06)
t3 = rd(f"{Q3T}/T3_mixture_joint.csv")
chk("联立增益最小%", t3.gain_pct.min(), 13.69, 6e-3)
chk("联立增益最大%", t3.gain_pct.max(), 13.78, 6e-3)
chk("规模通道最大%", t3.gain_scale_channel_pct.max(), 0.105, 6e-4)
chk("秩相关最小", t3.rho_rank_min_across_gC.min(), 0.976, 6e-4)
chk("TV 最优", t3.TV_best_vs_pstar.iat[0], 0.143, 6e-4)

# ---------------- 问题四 ----------------
sat = rd(f"{Q4I}/P4_task_saturation.csv").status.value_counts()
for k, v in {"饱和": 10, "快速增长": 9, "常规增长": 17, "低位停滞": 3}.items():
    chk(f"任务数 {k}", sat.get(k, 0), v, 0)
ds = rd(f"{Q4T}/T5_desat_vs_average.csv").set_index("metric")
chk("Average 相对增幅%", ds.loc["Average ⬆️", "rel_gain_pct"], 29.5, 0.06)
chk("去饱和相对增幅%", ds.loc["S_desat", "rel_gain_pct"], 48.2, 0.06)
chk("base 月份等效 dex/yr", ds.loc["Average ⬆️|base 回归", "month_equiv_dex"], 0.72, 6e-3)
chk("chat 月份等效 dex/yr", ds.loc["Average ⬆️|chat 回归", "month_equiv_dex"], 0.44, 6e-3)
B = json.load(open(f"{Q4I}/P4_bridge_params.json"))
chk("r(Loss,Avg)", B["corr_loss_avg"]["all"], -0.510, 6e-4)
chk("LOO sigmoid", B["loo_rmse"]["sigmoid"], 8.08, 6e-3)
chk("LOO linear", B["loo_rmse"]["linear"], 8.28, 6e-3)
chk("LOO const", B["loo_rmse"]["const"], 9.51, 6e-3)
chk("桥接 hi", B["params"]["base|Average"][1], 21.57, 6e-3)
J = json.load(open(f"{Q4I}/P4_decomp_models.json"))
chk("M1b τ", J["M1b"]["tau"], 2.45, 6e-3)
chk("M1c δ", J["M1c"]["delta"], 0.035, 6e-4)
chk("方法极差 pp", J["tech_share_spread_pp"], 32.8, 0.06)
chk("后训练斜率", J["posttrain_G"][1], 3.35, 6e-3)
D = rd(f"{Q4I}/P4_decomposition.csv")
D = D[D.window == "2023.50→2025.00"].set_index("method")
for mth, want in {"M1b": 12.9, "M1c": 23.1, "M2": 45.7, "M3_q75": 37.5, "M3_q50": 48.5, "OLS_lc": 29.5}.items():
    chk(f"技术占比% {mth}", D.loc[mth, "tech_share"] * 100, want, 0.06)
chk("前沿 S 起点", D.loc["M1b", "S_start"], 12.54, 6e-3)
chk("前沿 S 终点", D.loc["M1b", "S_end"], 37.30, 6e-3)
fq = rd(f"{Q4T}/T7_decomp_fit_quality.csv").set_index("model")
chk("M1b R2", fq.loc["M1b", "R2"], 0.870, 6e-4)
ch = rd(f"{Q4T}/T7_chat_decomposition.csv")
chk("chat 技术占比%", ch.tech_share.iat[0] * 100, 26, 0.6)
F = rd(f"{Q4I}/P4_frontier_forecast.csv")
f = F.set_index(["scenario", "horizon_months", "group", "posttrain"])
for key, want in {("基线 0.60 dex/yr", 12, "base", "-"): (43.1, 37.4, 52.8),
                  ("基线 0.60 dex/yr", 24, "base", "-"): (46.2, 37.5, 68.5),
                  ("1/4 0.15 dex/yr", 24, "base", "-"): (44.1, 37.5, 54.9),
                  ("基线 0.60 dex/yr", 24, "chat", "linear"): (61.6, 51.8, 79.6),
                  ("基线 0.60 dex/yr", 24, "chat", "plateau"): (54.8, 46.4, 70.1)}.items():
    for col, w in zip(("p50", "p05", "p95"), want):
        chk(f"预测 {key[0][:2]} {key[1]}m {key[2]}/{key[3]} {col}", f.loc[key, col], w, 0.06)
chk("基线 24m 技术占比%", f.loc[("基线 0.60 dex/yr", 24, "base", "-"), "tech_share_p50"] * 100, 55, 0.6)
chk("1/4 24m 技术占比%", f.loc[("1/4 0.15 dex/yr", 24, "base", "-"), "tech_share_p50"] * 100, 77, 0.6)
BT = rd(f"{Q4T}/T8_backtest.csv")
chk("回测 M1b 0.60 误差", BT[(BT.setting == "情景 0.60") & (BT.tech_form == "M1b")].error.iat[0], -0.78, 6e-3)
chk("回测 朴素外推误差", BT[BT.setting.str.startswith("朴素")].error.iat[0], 2.65, 6e-3)
chk("提交日轴 3 月误差", BT.error.iat[-1], 7.91, 6e-3)

bad = [x for x in res if not x[0]]
for ok, n, g, w in res:
    print(f"{'OK ' if ok else 'XX '} {n:<34s} 实际 {g:>12.5g}  汇报 {w}")
print(f"\n共 {len(res)} 项，不一致 {len(bad)} 项")
print("ALL PASS" if not bad else "有不一致，请更新汇报")
