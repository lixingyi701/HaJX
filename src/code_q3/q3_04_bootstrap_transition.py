# -*- coding: utf-8 -*-
"""
q3_04_bootstrap_transition.py — Q3-9：P1 不确定性传播到临界预算区间

将问题一的域质量和配比不确定性传播到问题三的临界预算识别中：
  ① 加载 P1 的 Q_domain 和 p* bootstrap 样本
  ② 对每个 bootstrap 样本重新计算 Q0(p*) = Σᵢ pᵢ*Qᵢ
  ③ 对每个 bootstrap 样本重新识别三种成本函数下的临界预算
  ④ 汇总临界预算的 bootstrap 区间（5%–95%）
  ⑤ 更新 P3_structural_transition.json 添加 P1 传播区间

输出：
  - P3_bootstrap_ccrit_with_P1.csv: 包含 P1 不确定性的临界预算 bootstrap
  - P3_structural_transition.json: 更新添加 P1+P2 联合区间
"""
import os, sys, json
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code_q1"))
from q1_00_common import ROOT

OUT = f"{ROOT}/outputs_q3"
IFACE = f"{OUT}/interface"

# 从 q3_00_model 导入求解器
from q3_00_model import PAR, ETA, Q0_MAIN, G_LIST, LCTX_C7, c_crit, write_manifest

if __name__ == "__main__":
    print("开始 Q3-9：P1 不确定性传播到临界预算...")

    # ============ 1. 加载 P1 bootstrap 样本 ============
    print("\n加载 P1 bootstrap 接口...")

    P1_Q = np.load(f"{ROOT}/outputs_q1/interface/P1_bootstrap_Q_domain.npz")
    P1_p = np.load(f"{ROOT}/outputs_q1/interface/P1_bootstrap_p_star.npz")

    Q_boot = P1_Q["samples"]  # (300, 17)
    p_boot = P1_p["samples"]  # (300, 17)
    domains = P1_Q["domains"]
    n_boot_full = int(P1_Q["n_boot"])

    # 为加速计算，使用全部样本（向量化，无需子样本）
    n_boot = n_boot_full
    print(f"  - 使用全部 {n_boot} 个 bootstrap 样本（向量化计算）")

    print(f"  - 加载 {n_boot_full} 次 bootstrap 样本，使用子样本 {n_boot} 次")
    print(f"  - Q_domain shape: {Q_boot.shape}")
    print(f"  - p_star shape: {p_boot.shape}")

    # ============ 2. 加载 P2 bootstrap 样本 ============
    print("\n加载 P2 bootstrap 接口...")

    P2_boot = np.load(f"{ROOT}/outputs_q2/interface/P2_bootstrap.npz")
    P2_samples = P2_boot["samples"]  # (300, 6) for E,A,B,kappa_N,kappa_D,k0
    P2_cols = list(P2_boot["cols"])

    print(f"  - P2 参数 bootstrap: {P2_samples.shape}")
    print(f"  - P2 参数列: {P2_cols}")

    # ============ 3. 向量化计算临界预算的 bootstrap 分布 ============
    print("\n向量化计算临界预算的 bootstrap 分布...")
    print("  说明：c_crit 支持 p 各参数为长度 n 的数组，一次调用算完全部样本")

    # 主口径：L_ctx=2048
    Lctx = 2048

    # 对齐 P1 与 P2 的样本数（取较小者）
    n_use = min(n_boot, P2_samples.shape[0])
    Q_boot = Q_boot[:n_use]
    p_boot = p_boot[:n_use]
    P2_use = P2_samples[:n_use]
    n_boot = n_use
    print(f"  - 对齐后使用 {n_boot} 个样本")

    # 每个样本的 Q0 = Σᵢ pᵢ*Qᵢ（向量化点积）
    Q0_vec = np.einsum("bi,bi->b", p_boot, Q_boot)  # (n_boot,)

    # 每个样本的 P2 参数：构造参数字典，各参数为长度 n_boot 的数组
    p_vec = dict(PAR)
    for i, col in enumerate(P2_cols):
        p_vec[col] = P2_use[:, i]

    # 结果存储
    results = []
    for g in G_LIST:
        # 向量化：一次算完 n_boot 个样本的临界预算
        cc_leave = c_crit(g, Lctx, Q0_vec, p=p_vec, which="leave")
        cc_full = c_crit(g, Lctx, Q0_vec, p=p_vec, which="full")
        for b in range(n_boot):
            results.append({
                "bootstrap_id": b,
                "g": g,
                "L_ctx": Lctx,
                "Q0_bootstrap": float(Q0_vec[b]),
                "ccrit_leave_Q0": float(cc_leave[b]) if np.isfinite(cc_leave[b]) else None,
                "ccrit_reach_Q1": float(cc_full[b]) if np.isfinite(cc_full[b]) else None
            })
        n_leave = int(np.isfinite(cc_leave).sum())
        n_full = int(np.isfinite(cc_full).sum())
        print(f"  {g}: 离开Q0有效 {n_leave}/{n_boot}, 到达Q1有效 {n_full}/{n_boot}")

    # ============ 4. 汇总统计 ============
    print("\n汇总 bootstrap 结果...")

    df = pd.DataFrame(results)
    df.to_csv(f"{IFACE}/P3_bootstrap_ccrit_with_P1.csv", index=False)

    # 计算各成本函数的区间
    summary = {}
    for g in G_LIST:
        subset = df[df["g"] == g].dropna(subset=["ccrit_leave_Q0", "ccrit_reach_Q1"])

        if len(subset) > 0:
            summary[g] = {
                "leave_Q0": {
                    "mean": float(subset["ccrit_leave_Q0"].mean()),
                    "std": float(subset["ccrit_leave_Q0"].std()),
                    "p05": float(subset["ccrit_leave_Q0"].quantile(0.05)),
                    "p50": float(subset["ccrit_leave_Q0"].quantile(0.50)),
                    "p95": float(subset["ccrit_leave_Q0"].quantile(0.95)),
                    "n_valid": int(len(subset))
                },
                "reach_Q1": {
                    "mean": float(subset["ccrit_reach_Q1"].mean()),
                    "std": float(subset["ccrit_reach_Q1"].std()),
                    "p05": float(subset["ccrit_reach_Q1"].quantile(0.05)),
                    "p50": float(subset["ccrit_reach_Q1"].quantile(0.50)),
                    "p95": float(subset["ccrit_reach_Q1"].quantile(0.95)),
                    "n_valid": int(len(subset))
                },
                "Q0_bootstrap": {
                    "mean": float(subset["Q0_bootstrap"].mean()),
                    "std": float(subset["Q0_bootstrap"].std()),
                    "p05": float(subset["Q0_bootstrap"].quantile(0.05)),
                    "p95": float(subset["Q0_bootstrap"].quantile(0.95))
                }
            }
        else:
            summary[g] = {"note": "所有 bootstrap 样本计算失败"}

    # ============ 5. 更新 P3_structural_transition.json ============
    print("\n更新 P3_structural_transition.json...")

    # 已有文件可能由 q3_02 在 Windows 上以 GBK 写出（默认编码），
    # 故读取时先试 utf-8，失败再回退 gbk；统一以 utf-8 重写。
    _json_path = f"{IFACE}/P3_structural_transition.json"
    try:
        with open(_json_path, "r", encoding="utf-8") as f:
            trans_data = json.load(f)
    except UnicodeDecodeError:
        with open(_json_path, "r", encoding="gbk") as f:
            trans_data = json.load(f)

    # 添加 P1+P2 联合不确定性区间
    trans_data["bootstrap_with_P1_Q3_9"] = {
        "description": "包含 P1 域质量和配比不确定性 + P2 参数不确定性的临界预算区间",
        "n_bootstrap": n_boot,
        "L_ctx": Lctx,
        "Q0_caliber": "主口径p*加权",
        "summary": summary
    }

    with open(f"{IFACE}/P3_structural_transition.json", "w", encoding="utf-8") as f:
        json.dump(trans_data, f, ensure_ascii=False, indent=2)

    # ============ 6. 打印报告 ============
    print("\n===== Q3-9 完成：P1 不确定性传播报告 =====")
    print(f"Bootstrap 样本数: {n_boot}")
    print(f"上下文长度: {Lctx} tokens")
    print()

    for g in G_LIST:
        if g in summary and "leave_Q0" in summary[g]:
            s = summary[g]
            print(f"{g}:")
            print(f"  离开 Q0: {s['leave_Q0']['p50']:.3f} [{s['leave_Q0']['p05']:.3f}, {s['leave_Q0']['p95']:.3f}] (log10 C)")
            print(f"  到达 Q1: {s['reach_Q1']['p50']:.3f} [{s['reach_Q1']['p05']:.3f}, {s['reach_Q1']['p95']:.3f}] (log10 C)")
            print(f"  Q0(p*): {s['Q0_bootstrap']['mean']:.5f} ± {s['Q0_bootstrap']['std']:.5f}")
            print(f"  有效样本: {s['leave_Q0']['n_valid']}/{n_boot}")
            print()

    print(f"输出文件:")
    print(f"  - {IFACE}/P3_bootstrap_ccrit_with_P1.csv")
    print(f"  - {IFACE}/P3_structural_transition.json (已更新)")

    # ============ 7. 重写权威运行清单 ============
    # q3_04 是 Q3 流水线的最后一步，且刚刚修改了 P3_structural_transition.json。
    # q3_01 / q3_02 先后写过 manifest，但 q3_02 会覆盖 q3_01 的输出列表，
    # 且此后 q3_04 又改动了 structural_transition.json 使其哈希过期。
    # 故在此以全部 Q3 接口产物重写一次，保证 manifest 哈希对全链条新鲜、可核验。
    print("\n重写权威运行清单 P3_run_manifest.json（覆盖全部 Q3 接口产物）...")
    write_manifest([
        "P3_optimal_config.csv",
        "P3_sensitivity_Lctx.csv",
        "P3_structural_transition.json",
        "P3_dn_slope_diagnostic.csv",
        "P3_dn_slope_kink.csv",
        "P3_extrapolation_by_config.csv",
        "P3_extrapolation_warning.json",
        "P3_bootstrap_ccrit_with_P1.csv",
    ])
    print("  - P3_run_manifest.json 已重写（含 Q3-9/Q3-10/外推产物，全部哈希新鲜）")

    print("\nQ3-9 完成！P1 不确定性已传播到临界预算区间。")
