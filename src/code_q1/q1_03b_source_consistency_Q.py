# -*- coding: utf-8 -*-
"""
q1_03b_source_consistency_Q.py — 基于来源内部一致性（Kendall W）调整CRITIC权重后重算Q

设计原则：
  ① 来源级 Kendall W 作先验可靠性权重，乘以原 CRITIC 权重后重归一化
  ② 不依赖任何单篇文档的冲突状态——是全局、非条件的权重调整
  ③ Q_v2 作实验性对照，不替换 P1 主接口的主 Q
  ④ 同时报告 A1/A2/A3，验证结论跨集是否成立
  ⑤ 指标顺序与 q1_00_common.IND_COLS 对齐（共 25 项）

IND_COLS 顺序（来自 q1_00_common.py）：
  [0]  fineweb_edu       → S4_FineWeb-Edu  (W=1.0，单指标)
  [1]  fluency_en        → S3_轻量分类器   (W=0.691)
  [2]  ad_en             → S3_轻量分类器
  [3]  qurater_writing   → S5_QuRating     (W=0.675)
  [4]  qurater_expertise
  [5]  qurater_facts
  [6]  qurater_edu
  [7]  modernbert_professionalism → S6_ModernBERT (W=0.579)
  [8]  modernbert_readability
  [9]  modernbert_reasoning
  [10] modernbert_cleanliness
  [11] dsir_books        → S2_DSIR         (W=0.967)
  [12] dsir_wiki
  [13] dsir_math
  [14..24] rps_*         → S1_RPS规则      (W=0.140)
"""
import os
import csv
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q1_00_common import (CACHE, TABLES, FIGS, IND_COLS,
                          kendall_w, spearman, setup_cjk_matplotlib)

# ── 来源 Kendall W（从 T3_six_source_consistency.csv 读取） ──────────────────
SOURCE_W_DEFAULTS = {
    "RPS":   0.1401257752293671,
    "DSIR":  0.9673472190210503,
    "轻量分类器": 0.6912276790445732,
    "FineWeb-Edu": 1.0,   # 单指标，无组内分歧定义，设为1
    "QuRating": 0.6754718689739256,
    "ModernBERT": 0.5785964515059016,
}

# 指标→来源名称映射（按 IND_COLS 顺序，共25项）
IND_SOURCE = (
    ["FineWeb-Edu"]       # [0]  fineweb_edu
    + ["轻量分类器"] * 2  # [1-2]  fluency_en, ad_en
    + ["QuRating"] * 4    # [3-6]  qurater_*
    + ["ModernBERT"] * 4  # [7-10] modernbert_*
    + ["DSIR"] * 3        # [11-13] dsir_*
    + ["RPS"] * 11        # [14-24] rps_*
)
assert len(IND_SOURCE) == 25


def load_source_kendall_w(csv_path):
    """从 T3_six_source_consistency.csv 读取各来源 Kendall W，单指标来源设为1.0"""
    w_map = dict(SOURCE_W_DEFAULTS)  # fallback
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = row["source"].strip()
                val = row.get("kendall_W_A1_rank_8000", "").strip()
                if val:
                    try:
                        w_map[src] = float(val)
                    except ValueError:
                        pass
                else:
                    w_map[src] = 1.0  # NaN → 1.0（FineWeb-Edu 单指标）
    return w_map


def build_source_consistency_weights(w_critic, source_w_map):
    """
    w_tilde_j = w_critic_j * W_{s(j)}，归一化为和=1
    参数：
      w_critic: (25,) 原 CRITIC 权重
      source_w_map: dict[来源名称 → Kendall W]
    返回：
      w_tilde: (25,) 调整后归一化权重
      multiplier: (25,) 每指标乘数（用于报告）
    """
    multiplier = np.array([source_w_map.get(s, 1.0) for s in IND_SOURCE])
    w_raw = w_critic * multiplier
    w_tilde = w_raw / w_raw.sum()
    return w_tilde, multiplier


def huber_center(Z, w, delta, n_bisect=60):
    """
    逐篇求解 Huber 加权中心：
      argmin_{q in [0,1]} sum_j w_j * rho_delta(z_j - q)
    一阶条件：sum_j w_j * clip(q - z_j, -delta, delta) = 0
    二分法求解，Z: (n, 25)，w: (25,)，delta: 标量
    返回 Q: (n,)
    """
    lo = np.zeros(Z.shape[0])
    hi = np.ones(Z.shape[0])
    for _ in range(n_bisect):
        mid = (lo + hi) / 2.0
        # f(mid) = sum_j w_j * clip(mid - z_j, -delta, delta)
        diff = mid[:, None] - Z          # (n, 25)
        clipped = np.clip(diff, -delta, delta)
        f = (clipped * w[None, :]).sum(axis=1)
        lo = np.where(f < 0, mid, lo)
        hi = np.where(f > 0, mid, hi)
    return (lo + hi) / 2.0


def domain_median(Q, domains):
    """返回 {domain: median_Q} 字典"""
    result = {}
    for d in np.unique(domains):
        m = domains == d
        result[d] = float(np.median(Q[m]))
    return result


def weighted_spearman(x, y):
    """Spearman 相关（基于 rankdata）"""
    return spearman(x, y)


# ── 主程序 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== q1_03b：来源一致性加权 Q_v2 计算 ===")

    # 1. 加载数据
    sc = np.load(f"{CACHE}/step2_scores.npz")
    Z1, Q1_orig = sc["Z1"], sc["Q1"]
    Z2, Q2_orig = sc["Z2"], sc["Q2"]
    Z3, Q3_orig = sc["Z3"], sc["Q3"]
    w_critic = sc["w"]    # (25,) 原 CRITIC 权重
    delta = float(sc["delta"])
    print(f"  原 delta={delta:.6f}，CRITIC权重范围=[{w_critic.min():.4f}, {w_critic.max():.4f}]")

    # 2. 读域标签
    import pandas as pd
    A1 = pd.read_pickle(f"{CACHE}/A1_indicators.pkl")
    A2 = pd.read_pickle(f"{CACHE}/A2_indicators.pkl")
    A3 = pd.read_pickle(f"{CACHE}/A3_indicators.pkl")
    d1 = A1["domain"].to_numpy()
    d2 = A2["domain"].to_numpy()
    d3 = A3["domain"].to_numpy()

    # 3. 读 Kendall W
    consist_csv = f"{TABLES}/T3_six_source_consistency.csv"
    src_w_map = load_source_kendall_w(consist_csv)
    print("  来源 Kendall W：")
    for s, w in src_w_map.items():
        print(f"    {s}: {w:.4f}")

    # 4. 构建调整权重
    w_v2, multiplier = build_source_consistency_weights(w_critic, src_w_map)
    print(f"\n  权重调整统计（乘数 × 原权重 → 归一化）：")
    print(f"    RPS均值: 原{w_critic[14:25].mean():.4f} → 新{w_v2[14:25].mean():.4f}")
    print(f"    DSIR均值: 原{w_critic[11:14].mean():.4f} → 新{w_v2[11:14].mean():.4f}")
    print(f"    QuRating均值: 原{w_critic[3:7].mean():.4f} → 新{w_v2[3:7].mean():.4f}")
    print(f"    ModernBERT均值: 原{w_critic[7:11].mean():.4f} → 新{w_v2[7:11].mean():.4f}")
    print(f"    FineWeb-Edu: 原{w_critic[0]:.4f} → 新{w_v2[0]:.4f}")

    # 5. 计算 Q_v2（三集复用同一 delta，A2/A3 冻结参数）
    print("\n  计算 Q_v2（Huber 60步二分法）...")
    Q1_v2 = huber_center(Z1, w_v2, delta)
    Q2_v2 = huber_center(Z2, w_v2, delta)
    Q3_v2 = huber_center(Z3, w_v2, delta)
    print(f"  A1: Q_orig 中位数={np.median(Q1_orig):.4f}, Q_v2 中位数={np.median(Q1_v2):.4f}")
    print(f"  A2: Q_orig 中位数={np.median(Q2_orig):.4f}, Q_v2 中位数={np.median(Q2_v2):.4f}")
    print(f"  A3: Q_orig 中位数={np.median(Q3_orig):.4f}, Q_v2 中位数={np.median(Q3_v2):.4f}")

    # 6. 全局 Spearman 相关
    rho1 = weighted_spearman(Q1_orig, Q1_v2)
    rho2 = weighted_spearman(Q2_orig, Q2_v2)
    rho3 = weighted_spearman(Q3_orig, Q3_v2)
    print(f"\n  Q_orig vs Q_v2 Spearman: A1={rho1:.4f}, A2={rho2:.4f}, A3={rho3:.4f}")

    # 7. 域级中位数对比
    rows = []
    for tag, Q_orig, Q_v2, domains in [
        ("A1", Q1_orig, Q1_v2, d1),
        ("A2", Q2_orig, Q2_v2, d2),
        ("A3", Q3_orig, Q3_v2, d3),
    ]:
        med_orig = domain_median(Q_orig, domains)
        med_v2 = domain_median(Q_v2, domains)
        # 域级 Spearman（名次稳定性）
        doms = sorted(med_orig.keys())
        orig_arr = np.array([med_orig[d] for d in doms])
        v2_arr = np.array([med_v2[d] for d in doms])
        dom_rho = weighted_spearman(orig_arr, v2_arr) if len(doms) > 1 else np.nan
        for d in doms:
            diff = med_v2[d] - med_orig[d]
            rows.append({
                "dataset": tag,
                "domain": d,
                "Q_orig_median": round(med_orig[d], 6),
                "Q_v2_median": round(med_v2[d], 6),
                "diff": round(diff, 6),
                "doc_spearman_Q_orig_v2": round(
                    weighted_spearman(
                        Q_orig[domains == d],
                        Q_v2[domains == d]
                    ), 4
                ),
                "domain_level_spearman": round(dom_rho, 4) if not np.isnan(dom_rho) else "",
            })

    # 8. 冲突文档 vs 非冲突文档中 Q_v2-Q 差值
    step3 = np.load(f"{CACHE}/step3_conflict.npz")
    # old_four_c1 是连续冲突得分，old_four_tau 是二值化阈值（= 0.71）
    tau = float(step3["old_four_tau"])
    is_conf_orig = step3["old_four_c1"] >= tau   # A1 原四组冲突掩码，rate ≈ 0.191
    # 六来源高分歧候选（bool，rate ≈ 0.050）
    if "six_candidate_A1" in step3:
        is_conf_v2 = step3["six_candidate_A1"].astype(bool)
    else:
        is_conf_v2 = None

    diff1 = Q1_v2 - Q1_orig
    print(f"\n  冲突标记形状: old_four_c1={is_conf_orig.shape}, diff1={diff1.shape}")
    print(f"  冲突文档数: {is_conf_orig.sum()}, 非冲突: {(~is_conf_orig).sum()}")

    def safe_conflict_row(label, mask, diff):
        sub = diff[mask]
        if len(sub) == 0:
            return {"group": label, "n": 0,
                    "diff_median": "", "diff_mean": "",
                    "diff_p10": "", "diff_p90": ""}
        return {
            "group": label,
            "n": int(mask.sum()),
            "diff_median": round(float(np.median(sub)), 6),
            "diff_mean": round(float(np.mean(sub)), 6),
            "diff_p10": round(float(np.percentile(sub, 10)), 6),
            "diff_p90": round(float(np.percentile(sub, 90)), 6),
        }

    conflict_diff_rows = []
    for label, mask in [("原四组冲突", is_conf_orig),
                         ("原四组非冲突", ~is_conf_orig)]:
        conflict_diff_rows.append(safe_conflict_row(label, mask, diff1))
    if is_conf_v2 is not None:
        for label, mask in [("六来源高分歧候选", is_conf_v2),
                             ("六来源非候选", ~is_conf_v2)]:
            conflict_diff_rows.append(safe_conflict_row(label, mask, diff1))

    # 9. 权重对比表
    weight_rows = []
    for j, col in enumerate(IND_COLS):
        weight_rows.append({
            "indicator": col,
            "source": IND_SOURCE[j],
            "source_kendall_W": round(src_w_map.get(IND_SOURCE[j], 1.0), 6),
            "w_critic_orig": round(float(w_critic[j]), 6),
            "w_v2_adjusted": round(float(w_v2[j]), 6),
            "multiplier": round(float(multiplier[j]), 6),
            "weight_ratio_v2_over_orig": round(float(w_v2[j] / w_critic[j]), 4),
        })

    # 10. 整体摘要行
    summary_rows = []
    for tag, Q_orig, Q_v2, rho in [
        ("A1", Q1_orig, Q1_v2, rho1),
        ("A2", Q2_orig, Q2_v2, rho2),
        ("A3", Q3_orig, Q3_v2, rho3),
    ]:
        summary_rows.append({
            "dataset": tag,
            "n": len(Q_orig),
            "Q_orig_median": round(float(np.median(Q_orig)), 6),
            "Q_v2_median": round(float(np.median(Q_v2)), 6),
            "global_spearman_orig_v2": round(rho, 6),
            "mean_abs_diff": round(float(np.mean(np.abs(Q_v2 - Q_orig))), 6),
            "p90_abs_diff": round(float(np.percentile(np.abs(Q_v2 - Q_orig), 90)), 6),
        })

    # 11. 写出 CSV
    os.makedirs(TABLES, exist_ok=True)

    def write_csv(path, rows):
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(f"{TABLES}/T3_Q_v2_comparison.csv", rows)
    write_csv(f"{TABLES}/T3_Q_v2_conflict_diff.csv", conflict_diff_rows)
    write_csv(f"{TABLES}/T3_Q_v2_weight_comparison.csv", weight_rows)
    write_csv(f"{TABLES}/T3_Q_v2_summary.csv", summary_rows)
    print(f"\n  已写出:")
    print(f"    {TABLES}/T3_Q_v2_comparison.csv    (域级中位数对比)")
    print(f"    {TABLES}/T3_Q_v2_conflict_diff.csv (冲突文档差值分布)")
    print(f"    {TABLES}/T3_Q_v2_weight_comparison.csv (权重对比)")
    print(f"    {TABLES}/T3_Q_v2_summary.csv       (总体摘要)")

    # 12. 图：权重对比条形图
    print("\n  生成图表...")
    plt = setup_cjk_matplotlib()

    # 图1：原/新权重对比
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(25)
    ax.bar(x - 0.2, w_critic, 0.4, label="原 CRITIC 权重", alpha=0.8, color="steelblue")
    ax.bar(x + 0.2, w_v2, 0.4, label="调整后权重 (×KendallW)", alpha=0.8, color="darkorange")
    # 来源分隔线
    boundaries = [1, 3, 7, 11, 14]  # fineweb|轻量|qurater|modernbert|dsir|rps
    labels_src = ["FW", "轻量", "QuR", "MB", "DSIR", "RPS"]
    prev = 0
    for i, (b, lbl) in enumerate(zip(boundaries + [25], labels_src)):
        mid = (prev + b - 1) / 2
        ax.axvspan(prev - 0.5, b - 0.5, alpha=0.05,
                   color=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"][i])
        ax.text(mid, max(w_v2.max(), w_critic.max()) * 1.02, lbl,
                ha="center", fontsize=8)
        if i < len(boundaries):
            ax.axvline(b - 0.5, color="gray", lw=0.8, ls="--")
        prev = b
    ax.set_xticks(x)
    ax.set_xticklabels([c[:12] for c in IND_COLS], rotation=90, fontsize=6)
    ax.set_ylabel("权重值")
    ax.set_title("CRITIC权重 vs 来源一致性调整后权重（25指标）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/F3_Q_v2_weight_comparison.png", dpi=150)
    plt.close(fig)

    # 图2：A1 Q_orig vs Q_v2 散点（子样本）
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(Q1_orig), min(5000, len(Q1_orig)), replace=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(Q1_orig[sample_idx], Q1_v2[sample_idx],
               s=1, alpha=0.3, color="steelblue")
    ax.plot([0, 1], [0, 1], "r--", lw=1, label="y=x")
    ax.set_xlabel("Q_orig（原 CRITIC+Huber）")
    ax.set_ylabel("Q_v2（来源一致性加权）")
    ax.set_title(f"A1 Q_orig vs Q_v2（Spearman={rho1:.4f}，n=5000子样本）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/F3_Q_v2_scatter_A1.png", dpi=150)
    plt.close(fig)

    # 图3：各域Q_orig vs Q_v2 中位数对比（A1）
    med_orig_a1 = domain_median(Q1_orig, d1)
    med_v2_a1 = domain_median(Q1_v2, d1)
    doms_a1 = sorted(med_orig_a1.keys(),
                     key=lambda d: med_orig_a1[d], reverse=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    xi = np.arange(len(doms_a1))
    ax.bar(xi - 0.2, [med_orig_a1[d] for d in doms_a1], 0.4,
           label="Q_orig", alpha=0.85, color="steelblue")
    ax.bar(xi + 0.2, [med_v2_a1[d] for d in doms_a1], 0.4,
           label="Q_v2", alpha=0.85, color="darkorange")
    ax.set_xticks(xi)
    ax.set_xticklabels(doms_a1, rotation=30, ha="right")
    ax.set_ylabel("域中位数 Q")
    ax.set_title("A1 各域 Q_orig vs Q_v2 中位数（按Q_orig降序）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/F3_Q_v2_domain_median.png", dpi=150)
    plt.close(fig)

    # 图4：冲突文档差值分布
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(diff1[is_conf_orig], bins=60, alpha=0.6,
            label=f"原四组冲突文档 (n={is_conf_orig.sum()})", density=True, color="red")
    ax.hist(diff1[~is_conf_orig], bins=60, alpha=0.6,
            label=f"原四组非冲突文档 (n={(~is_conf_orig).sum()})", density=True, color="blue")
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("Q_v2 − Q_orig")
    ax.set_ylabel("密度")
    ax.set_title("A1 冲突/非冲突文档的权重调整效果（Q_v2 − Q_orig）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/F3_Q_v2_conflict_diff_dist.png", dpi=150)
    plt.close(fig)

    print("\n  已生成图：")
    print(f"    {FIGS}/F3_Q_v2_weight_comparison.png")
    print(f"    {FIGS}/F3_Q_v2_scatter_A1.png")
    print(f"    {FIGS}/F3_Q_v2_domain_median.png")
    print(f"    {FIGS}/F3_Q_v2_conflict_diff_dist.png")

    print("\n✓ q1_03b 完成。Q_v2 为实验性对照分，不替换 P1 主接口主 Q。")
    print(f"  关键结论：A1/A2/A3 全局 Spearman {rho1:.4f}/{rho2:.4f}/{rho3:.4f}")
    print(f"  A1 Q_v2 域中位数：" +
          ", ".join(f"{d}={med_v2_a1[d]:.4f}" for d in doms_a1))
