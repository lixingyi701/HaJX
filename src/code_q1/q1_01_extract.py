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
        rows.append([row[c] for c in IND_COLS])
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
