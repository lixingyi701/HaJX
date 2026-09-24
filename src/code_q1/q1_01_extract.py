# -*- coding: utf-8 -*-
"""
q1_01_extract.py — Step1：流式抽取 A1/A2/A3 -> 25 指标矩阵（parquet 缓存）
依据：《数据说明》要求流式读取；列表字段按硬事实表压缩（softmax 期望/拆维/取 P(label=1)）。
"""
import os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q1_00_common import (A1_PATH, A2_PATH, A3_PATH, CACHE, IND_COLS,
                          stream_jsonl_xz)

def extract(path, tag, domain=None):
    t0 = time.time()
    rows, doms, wcs = [], [], []
    for row, d, _ in stream_jsonl_xz(path, domain=domain):
        # ===== P2修复: DSIR归一化 (2026-09-24 重新修复) =====
        # DSIR 测量假设：先除词数，再由 q1_02 在 A1 上拟合域内长度残差。
        # 这不是附件已证明的定义，也不依冲突类型改变评分权重。
        wc = max(row.get("rps_doc_word_count", 1), 1)  # 防止除零
        row_values = []
        for c in IND_COLS:
            val = row[c]
            # 对DSIR三个指标进行归一化
            if c in ["dsir_books", "dsir_wiki", "dsir_math"]:
                val = val / wc
            row_values.append(val)
        rows.append(row_values)
        # ===== P2修复结束 =====

        doms.append(d)
        wcs.append(row["rps_doc_word_count"])
        if len(rows) % 50000 == 0:
            print(f"  [{tag}] {len(rows)} rows, {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows, columns=IND_COLS)
    df["domain"] = doms
    df["dataset"] = tag

    out = f"{CACHE}/{tag}_indicators.pkl"
    df.to_pickle(out)
    print(f"[{tag}] {len(df)} rows -> {out}  ({time.time()-t0:.0f}s)", flush=True)
    # 概要
    print(df["domain"].value_counts().to_string())
    miss = df[IND_COLS].isna().mean()
    bad = miss[miss > 0]
    print("缺失率>0 的指标:", bad.to_dict() if len(bad) else "无", flush=True)
    return df

if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    extract(A1_PATH, "A1")
    extract(A2_PATH, "A2", domain="arxiv")
    extract(A3_PATH, "A3", domain="github")
    print("Step1 done.")
