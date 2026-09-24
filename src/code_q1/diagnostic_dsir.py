# -*- coding: utf-8 -*-
"""只读取当前 Q1 实测表的 DSIR 长度诊断与冲突率；不估计评分参数。"""
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q1_00_common import TABLES

if __name__ == "__main__":
    length = pd.read_csv(f"{TABLES}/T2_dsir_length_diagnostic.csv")
    rate = pd.read_csv(f"{TABLES}/T3_extended_recheck.csv")
    print("A1 各域×DSIR 的词数绝对 Spearman 中位数：")
    for col in ("rho_raw_vs_wc", "rho_perword_vs_wc", "rho_residual_vs_wc"):
        print(f"  {col}: {length[col].abs().median():.4f}")
    print("\n剩余绝对相关最高的域和指标：")
    print(length.assign(abs_resid=length.rho_residual_vs_wc.abs())
          .nlargest(3, "abs_resid")[["domain", "indicator", "rho_residual_vs_wc"]]
          .to_string(index=False))
    print("\n独立的评分后冲突诊断（不反馈到 Q）：")
    print(rate[["dataset", "n", "conflict_rate", "top_type"]].to_string(index=False))
