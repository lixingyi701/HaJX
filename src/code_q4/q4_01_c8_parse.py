# -*- coding: utf-8 -*-
"""
q4_01_c8_parse.py — 问题四·Step5a：C8 逐任务原始结果解析（1 958 个 JSON → 长表缓存）

每个 JSON 取 38 个叶子任务（BBH 24 + GPQA 3 + MATH 7 + MUSR 3 + MMLU-PRO 1）+ IFEval 的主指标与 stderr：
  BBH/GPQA/MUSR → acc_norm；MATH → exact_match；MMLU-PRO → acc；
  IFEval → (prompt_level_strict + inst_level_strict)/2（与榜单一致）
同一模型多次评测只保留最新一份。分块续跑（每次 ≤ 170 s），缓存到 cache/c8_long.pkl。
"""
import glob, json, os, sys, time, pickle
import pandas as pd
from q4_00_common import C8_DIR, CACHE

METRIC = {"bbh": "acc_norm", "gpqa": "acc_norm", "musr": "acc_norm",
          "math": "exact_match", "mmlu_pro": "acc"}
GROUPS = {"leaderboard", "leaderboard_bbh", "leaderboard_gpqa", "leaderboard_math_hard",
          "leaderboard_musr"}
PART = f"{CACHE}/c8_parts"
os.makedirs(PART, exist_ok=True)


def family(task):
    t = task.replace("leaderboard_", "")
    for k in ("bbh", "gpqa", "musr", "math", "mmlu_pro", "ifeval"):
        if t.startswith(k):
            return k
    return None


def parse(fp):
    j = json.load(open(fp))
    rows = []
    for task, v in j["results"].items():
        if task in GROUPS:
            continue
        fam = family(task)
        if fam is None:
            continue
        if fam == "ifeval":
            s = (v["prompt_level_strict_acc,none"] + v["inst_level_strict_acc,none"]) / 2
            se = v.get("prompt_level_strict_acc_stderr,none")
        else:
            s = v.get(f"{METRIC[fam]},none")
            se = v.get(f"{METRIC[fam]}_stderr,none")
        rows.append(dict(model=j.get("model_name"), task=task.replace("leaderboard_", ""),
                         family=fam, score=s, stderr=se if isinstance(se, (int, float)) else None,
                         n_samples=(j.get("n-samples", {}).get(task, {}) or {}).get("effective"),
                         eval_time=os.path.basename(fp)[8:27]))
    return rows


if __name__ == "__main__":
    files = sorted(glob.glob(f"{C8_DIR}/*/results_*.json"))
    chunk = 250
    t0 = time.time()
    for ci in range(0, len(files), chunk):
        out = f"{PART}/part_{ci:05d}.pkl"
        if os.path.exists(out):
            continue
        rows = []
        for fp in files[ci:ci + chunk]:
            try:
                rows += parse(fp)
            except Exception as ex:                       # 个别损坏文件记录后跳过
                print("skip", fp, ex)
        pickle.dump(rows, open(out, "wb"))
        print(f"chunk {ci} done, {time.time() - t0:.0f}s", flush=True)
        if time.time() - t0 > 140:
            print("时间将尽，下次续跑"); sys.exit(0)
    parts = sorted(glob.glob(f"{PART}/part_*.pkl"))
    L = pd.DataFrame([r for p in parts for r in pickle.load(open(p, "rb"))])
    L = L.sort_values("eval_time").drop_duplicates(["model", "task"], keep="last")
    L.to_pickle(f"{CACHE}/c8_long.pkl")
    print(L.shape, L.model.nunique(), "models,", L.task.nunique(), "tasks")
    print(L.groupby("family").task.nunique())
