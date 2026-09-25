# -*- coding: utf-8 -*-
"""
q4_00_common.py — 问题四公共口径：路径、数据读取、四项口径定义、标度律接口

【四项口径（赛题逐条点名，统一在此定义，后续脚本只引用不再改）】
1. 能力度量：主口径 = C1 `Average ⬆️`（Open LLM Leaderboard v2 六维等权，已扣随机基线，0-100）；
            副口径 = C8 逐任务重算的"去饱和综合分"（q4_02 产出）。
2. 开源口径：Hub 记录不等于权重可下载；需与 C2 开放权重标记及 C4 可访问性核对，
            再按 Hub License 分三档
            permissive（apache/mit/bsd/cc-by/openrail…）、community（llama*/gemma/qwen/other…）、
            noncommercial（cc-by-nc*…）。C4 以 `Model accessibility` 含 "Open weights" 为开源。
3. 模型类型：base = 🟢 pretrained + 🟩 continuously pretrained；
            chat = 💬 chat + 🔶 fine-tuned；merges / multimodal / other 剔除（非独立技术样本）。
4. 时间轴：主 = C2 `Epoch_AI_Publication_Date`（真实发布日）；C1 `Submission Date` 仅是上榜日
          （滞后于发布，窗口仅 2024-06~2025-03），用于月度窗口分析与回测；C3 历史点用 Year。
"""
import os, sys, json, re
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code_q1"))
from q1_00_common import ROOT, setup_cjk_matplotlib  # noqa: E402,F401

C_DIR = os.path.join(os.path.dirname(ROOT), "附件", "C_efficiency_evolution")
C8_DIR = f"{C_DIR}/detailed_results"
OUT = f"{ROOT}/outputs_q4"
TAB, FIG, IFACE, CACHE = f"{OUT}/tables", f"{OUT}/figures", f"{OUT}/interface", f"{OUT}/cache"
for _d in (TAB, FIG, IFACE, CACHE):
    os.makedirs(_d, exist_ok=True)

P2 = json.load(open(f"{ROOT}/outputs_q2/interface/P2_scaling_law.json"))
PAR = dict(E=P2["E"], A=P2["A"], al=P2["alpha"], B=P2["B"], be=P2["beta"],
           k0=P2["k0"], kappa_N=P2["kappa_N"], kappa_D=P2["kappa_D"])
ETA = 2e-4
DIMS = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
AVG = "Average ⬆️"


def law_loss(N, D, Q=1.0, p=PAR):
    """问题二广义标度律（参数固定不重估）"""
    return (p["E"] + p["A"] * (1 + p["kappa_N"] * (1 - Q)) * N ** -p["al"]
            + p["B"] * (1 + p["kappa_D"] * (1 - Q)) * D ** -p["be"]
            + (1 - Q) * p["k0"])


def type_group(t):
    t = str(t)
    if "pretrained" in t:            # 🟢 pretrained / 🟩 continuously pretrained
        return "base"
    if "chat" in t or "fine-tuned" in t:
        return "chat"
    return "excluded"                # merges / multimodal / other


_PERM = ("apache", "mit", "bsd", "cc-by-4.0", "cc-by-sa", "openrail", "bigscience", "cc0",
         "unlicense", "artistic", "afl", "wtfpl", "gpl", "lgpl", "mpl", "ecl", "odc", "pddl")


def license_tier(lic):
    s = str(lic).lower()
    if s in ("nan", "", "unknown"):
        return "unknown"
    if "nc" in s.split("-") or "non-commercial" in s or "noncommercial" in s or "cc-by-nc" in s:
        return "noncommercial"
    if any(k in s for k in _PERM):
        return "permissive"
    return "community"               # llama2/3.x、gemma、qwen、other 等带使用条款的社区许可


def load_c1():
    """C2 = C1 + Epoch 列；返回带口径列的 DataFrame"""
    e = pd.read_csv(f"{C_DIR}/leaderboard_enhanced.csv")
    e["sub_date"] = pd.to_datetime(e["Submission Date"], errors="coerce")
    e["pub_date"] = pd.to_datetime(e["Epoch_AI_Publication_Date"], errors="coerce")
    e["group"] = e.Type.map(type_group)
    e["lic_tier"] = e["Hub License"].map(license_tier)
    e["N"] = e["#Params (B)"] * 1e9
    return e


def load_c4_lm():
    c4 = pd.read_csv(f"{C_DIR}/epoch_all_ai_models.csv", low_memory=False)
    c4["pub"] = pd.to_datetime(c4["Publication date"], errors="coerce")
    lm = c4[c4.Domain.str.contains("Language", na=False)].copy()
    lm["open"] = lm["Model accessibility"].str.contains("Open weights", na=False)
    lm["C"] = pd.to_numeric(lm["Training compute (FLOP)"], errors="coerce")
    lm["Dtok"] = pd.to_numeric(lm["Training dataset size (total)"], errors="coerce")
    lm["Npar"] = pd.to_numeric(lm["Parameters"], errors="coerce")
    return lm


def join_c1_c4(e, lm):
    """C2 的 Epoch 发布日 + 组织 ⇒ 在 C4 中定位同一模型（同日同组织内按参数量最接近），
    取训练算力 C、数据量 D、参数 N。D 缺失时由 D = C/(6N) 反解（赛题允许的近似）。"""
    k = e[e.pub_date.notna()].copy()
    m = k.merge(lm[["Model", "pub", "Organization", "Npar", "C", "Dtok", "Model accessibility"]]
                .rename(columns={"Model": "c4_model"}),
                left_on=["pub_date", "Epoch_AI_Organization"], right_on=["pub", "Organization"],
                how="left")
    m["par_gap"] = np.abs(np.log(m.Npar / m.N))
    m = m.sort_values("par_gap").drop_duplicates("Model").sort_index()
    ok = m.par_gap < np.log(1.5)                     # 参数量相差 50% 以内才视为同一模型
    for c in ("c4_model", "Npar", "C", "Dtok", "Model accessibility"):
        m.loc[~ok, c] = np.nan
    m["D_src"] = np.where(m.Dtok.notna(), "C4数据量", np.where(m.C.notna(), "C/(6N)反解", "缺失"))
    m["D_use"] = m.Dtok.where(m.Dtok.notna(), m.C / (6 * m.N))
    m["C_use"] = m.C.where(m.C.notna(), 6 * m.N * m.Dtok)
    return m


def year_frac(d):
    d = pd.to_datetime(d)
    return d.dt.year + (d.dt.dayofyear - 1) / 365.25


# ---------------- 小工具（无 scipy） ----------------
def wls(X, y, w=None):
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return beta


def quantile_reg(X, y, q=0.5, iters=200, eps=1e-6):
    """分位数回归（IRLS 近似 check loss）"""
    b = wls(X, y)
    for _ in range(iters):
        r = y - X @ b
        w = np.where(r >= 0, q, 1 - q) / np.maximum(np.abs(r), eps)
        b_new = wls(X, y, w)
        if np.max(np.abs(b_new - b)) < 1e-8:
            break
        b = b_new
    return b


def sigmoid_bridge(L, p):
    """S = lo + (hi-lo) / (1 + exp(k (L - L0)))，得分有上下界"""
    lo, hi, k, L0 = p
    return lo + (hi - lo) / (1 + np.exp(k * (np.asarray(L) - L0)))
