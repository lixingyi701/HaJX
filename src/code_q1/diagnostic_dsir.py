# -*- coding: utf-8 -*-
"""
diagnostic_dsir.py - P2 DSIR归一化修复效果诊断
验证DSIR归一化是否生效,以及与word_count的相关性
"""
import numpy as np
import pandas as pd

def spearmanr(x, y):
    """计算Spearman相关系数(纯numpy实现)"""
    n = len(x)
    rx = np.argsort(np.argsort(x)) + 1  # 排名
    ry = np.argsort(np.argsort(y)) + 1
    d_sq = ((rx - ry) ** 2).sum()
    rho = 1 - (6 * d_sq) / (n * (n**2 - 1))
    return rho, 0.0  # 返回(相关系数, p值占位)

# 加载修复后的数据
A1 = pd.read_pickle("/sessions/dazzling-wizardly-clarke/mnt/version1/outputs_q1/cache/A1_indicators.pkl")
A2 = pd.read_pickle("/sessions/dazzling-wizardly-clarke/mnt/version1/outputs_q1/cache/A2_indicators.pkl")
A3 = pd.read_pickle("/sessions/dazzling-wizardly-clarke/mnt/version1/outputs_q1/cache/A3_indicators.pkl")

print("=" * 60)
print("P2 DSIR归一化修复效果诊断报告")
print("=" * 60)

# 1. 检查DSIR数值范围
print("\n1. DSIR数值范围检查（验证归一化是否生效）:")
for dsir_col in ["dsir_books", "dsir_wiki", "dsir_math"]:
    if dsir_col in A1.columns:
        print(f"\n{dsir_col}:")
        print(f"  A1: min={A1[dsir_col].min():.6f}, max={A1[dsir_col].max():.6f}, mean={A1[dsir_col].mean():.6f}")
        print(f"  A3: min={A3[dsir_col].min():.6f}, max={A3[dsir_col].max():.6f}, mean={A3[dsir_col].mean():.6f}")
        # 如果归一化生效,数值应该在合理范围内(通常<10)
        # 如果未归一化,累计值可能很大(几百到几千)

# 2. 计算DSIR与word_count的相关性
print("\n2. DSIR与word_count的Spearman相关性:")
for dsir_col in ["dsir_books", "dsir_wiki", "dsir_math"]:
    if dsir_col in A1.columns and "rps_doc_word_count" in A1.columns:
        # A2 (arxiv)
        corr_a2, p_a2 = spearmanr(A2[dsir_col], np.log1p(A2["rps_doc_word_count"]))
        # A3 (github)
        corr_a3, p_a3 = spearmanr(A3[dsir_col], np.log1p(A3["rps_doc_word_count"]))

        print(f"\n{dsir_col} vs log(word_count):")
        print(f"  A2 (arxiv):  ρ={corr_a2:.4f}, p={p_a2:.2e}")
        print(f"  A3 (github): ρ={corr_a3:.4f}, p={p_a3:.2e}")
        print(f"  目标: |ρ| < 0.15 (修复成功)")

        if abs(corr_a3) > 0.5:
            print(f"  ⚠️ 警告: A3相关性{corr_a3:.2f}仍然很高,DSIR归一化可能未生效!")

# 3. 按文档长度分层检查DSIR均值
print("\n3. DSIR均值按文档长度分层:")
print("\nA3 (github) - dsir_wiki:")
bins = [0, 20, 100, 500, 2000, 10000, np.inf]
labels = ["<20词", "20-100词", "100-500词", "500-2000词", "2000-10000词", ">10000词"]
A3["length_bin"] = pd.cut(A3["rps_doc_word_count"], bins=bins, labels=labels)

if "dsir_wiki" in A3.columns:
    for label in labels:
        subset = A3[A3["length_bin"] == label]
        if len(subset) > 0:
            mean_dsir = subset["dsir_wiki"].mean()
            print(f"  {label:15s}: n={len(subset):6d}, dsir_wiki均值={mean_dsir:.6f}")

    # 如果归一化生效,各长度段的DSIR均值应该相近
    # 如果未归一化,短文档DSIR均值会明显更低

# 4. 加载修复前的冲突率数据进行对比
print("\n4. 冲突率变化:")
print("  修复前: 21.5% (基于规格文档)")
print("  修复后: 19.98% (A1), 24.36% (A2), 21.72% (A3)")
print("  下降幅度: 1.5个百分点")
print("  预期下降: 10个百分点")
print("  实际效果: 远低于预期")

print("\n" + "=" * 60)
print("诊断结论:")
print("=" * 60)
