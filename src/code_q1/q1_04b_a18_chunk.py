# -*- coding: utf-8 -*-
"""q1_04b_a18_chunk.py — A18 规则特征分片断点续算
用法: python3 q1_04b_a18_chunk.py <start_line> <end_line>
输入: /tmp/q1work/a18.jsonl（已解压）
输出: /tmp/q1work/a18_phi_<start>_<end>.pkl
"""
import sys, os, json, collections
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q1_04_bridge import phi

CAP18 = 3000
COUNTER_F = "/tmp/q1work/a18_counter.json"

def main(start, end):
    seen = collections.Counter()
    if os.path.exists(COUNTER_F):
        seen.update(json.load(open(COUNTER_F)))
    rows = []
    with open("/tmp/q1work/a18.jsonl", "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i < start: continue
            if i >= end: break
            try: r = json.loads(line)
            except json.JSONDecodeError: continue
            d = r.get("_source_domain")
            if d is None or seen[d] >= CAP18: continue
            p = phi(r.get("text") or "")
            if p is None: continue
            p["domain"] = d
            rows.append(p); seen[d] += 1
    pd.DataFrame(rows).to_pickle(f"/tmp/q1work/a18_phi_{start}_{end}.pkl")
    json.dump(dict(seen), open(COUNTER_F, "w"))
    print(f"chunk [{start},{end}) -> {len(rows)} rows; 总计数 {sum(seen.values())}")

if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
