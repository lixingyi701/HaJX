# -*- coding: utf-8 -*-
"""跨问接口和论文就绪检查。只检查可机读约束；模型验证见各问 Spec。"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

SRC = Path(__file__).resolve().parents[1]
REPO = SRC.parent
DOC = REPO / "doc"
checks = []


def check(name, ok, detail=""):
    checks.append((bool(ok), name, detail))


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run():
    p1_path = SRC / "outputs_q1/interface/P1_summary.json"
    p2_path = SRC / "outputs_q2/interface/P2_scaling_law.json"
    p3_path = SRC / "outputs_q3/interface/P3_optimal_config.csv"
    p4_path = SRC / "outputs_q4/interface/P4_frontier_forecast.csv"
    for path in (p1_path, p2_path, p3_path, p4_path):
        check(f"接口存在: {path.name}", path.is_file(), str(path))
    if not all(path.is_file() for path in (p1_path, p2_path, p3_path, p4_path)):
        return

    p1, p2 = load_json(p1_path), load_json(p2_path)
    q0 = float(p1["Qbar_pstar_eqweight"])
    check("P1/P2 主质量基线一致", math.isclose(q0, float(p2["Q0_main"]), abs_tol=1e-10),
          f"P1={q0:.10f}, P2={p2['Q0_main']:.10f}")
    check("P2 使用当前质量耦合参数", all(k in p2 for k in ("kappa_N", "kappa_D", "k0")))
    check("P2 主质量映射机读", p2.get("q_mapping", {}).get("main", {}).get("name") == "identity")

    p3 = load_csv(p3_path)
    main = [r for r in p3 if r.get("Q0口径") == "主口径p*加权"]
    check("P3 主网格为 210 行", len(p3) == 210, f"实际 {len(p3)}")
    check("P3 主质量基线一致", bool(main) and all(
        math.isclose(float(r["Q0"]), q0, abs_tol=1e-10) for r in main))
    check("P3 预算份额归一", bool(p3) and all(
        abs(sum(float(r[k]) for k in ("share_train", "share_attn", "share_Q")) - 1) < 1e-9
        for r in p3))

    manifest_path = SRC / "outputs_q3/interface/P3_run_manifest.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        inputs = manifest.get("inputs_sha256", {})
        valid = bool(inputs)
        for rel, want in inputs.items():
            path = REPO / rel
            valid &= path.is_file() and sha256(path) == want
        check("P3 输入哈希与运行清单一致", valid)
    else:
        check("P3 运行清单存在", False, "缺 P3_run_manifest.json；不能证明上下游同轮")

    p4 = load_csv(p4_path)
    check("P4 预测有输出", bool(p4))
    if p4:
        earliest = min(float(r["date"]) for r in p4)
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        current_year = now.year + (now.timetuple().tm_yday - 1) / 365.25
        check("P4 目标日期仍属于未来", earliest > current_year,
              f"最早目标年 {earliest:.2f}；旧结果只能称历史情景")

    final = DOC / "final_report"
    abstract = (final / "abstract.tex").read_text(encoding="utf-8")
    q1_section = (final / "sec5_q1.tex").read_text(encoding="utf-8")
    check("论文 Q1 主方法已更新", "加权几何平均" not in abstract + q1_section,
          "摘要/§5.2 仍将旧几何平均写作主分" if "加权几何平均" in abstract + q1_section else "")
    check("论文 Q3 基线已更新", "Q_0=0.988" not in abstract,
          "摘要仍使用旧 Q3 主质量基线" if "Q_0=0.988" in abstract else "")


if __name__ == "__main__":
    run()
    for ok, name, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    failed = sum(not ok for ok, _, _ in checks)
    print(f"\n{len(checks)} 项检查，{failed} 项未通过。该脚本不替代各 Spec 的模型与留出验收。")
    raise SystemExit(1 if failed else 0)
