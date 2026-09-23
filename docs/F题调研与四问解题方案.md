# 2026 研究生数学建模 F 题：调研与四问解题方案

> 题目：《算力约束下提升大语言模型能力的资源配置建模》
> 本文档定位：**赛前调研 + 建模思路蓝图**，供队伍理解题意、规划建模路线、检索文献之用。
> 依据赛题《AI 使用规范》，本文档属于"资料整理与概念理解"环节，**核心建模、推导与论证须由队伍独立完成并在论文中披露 AI 工具使用情况**。本文给出的所有结论性数字均来自附件数据说明或文末列出的可核验文献，凡属建议/方案性内容均明确标注。

---

## 0. 阅读须知与合规红线（文档要求速查）

### 0.1 AI 使用规范（赛题第五节，强制）

1. AI 仅可用于**概念理解、资料整理、代码调试**；不得直接复制 AI 生成的建模方案作为论文主体。
2. 对 AI 输出须**核验并理解**，队伍对正确性负责。
3. 理论、公式、实验结论必须引用**正式文献或可核验数据**，不得以 AI 输出作为学术依据。
4. 论文末尾必须**披露所用 AI 工具、使用环节及贡献范围**，未披露取消资格。

→ 建议论文附"AI 工具披露"小节，模板见附录 D。

### 0.2 数据使用强制要求（附录 A，每问都有"必用"清单）

| 问题 | 必用数据 | 关键约束 |
|---|---|---|
| 一 | A1 全量（51,230 条）+ A2/A3 全量（17,523 + 203,752 条）；A4–A15 配比建模，A6–A11 检验，A12–A15 外推 | 质量评价须给域级 $Q$ 并与抽样集对照；冲突分析覆盖抽样集并在扩展集复检 |
| 二 | B1 主拟合；B2 或 B3 验证；B4、B5 跨族/文献验证；B6（或 B7/B8）质量项；B9、B10 百亿以上外推 | 半合成数据（B2、B6–B8）须标注可信度边界，不得表述为直接观测 |
| 三 | C7（上下文长度可行域）+ 前两问输出 | $L_{ctx}$ 外生非寻优；须解析给出 $L_{ctx}^{crit}=6/\eta$ 并做敏感性；至少三档预算 $10^{19}/10^{22}/10^{24}$ |
| 四 | C1（或 C2）+ C3；C4 算力/数据量/开源权重字段；**C8 至少一项逐任务聚合分析** | 须说明开源口径、pretrained vs chat/finetuned 区分、时间轴口径；桥接数据按可比性分层 |

### 0.3 ⚠️ 《数据说明》PDF 中的"陷阱提示"（务必避开）

数据说明每页页眉混入了一批**看似省事但明显错误/误导的建议**，与赛题背景和科学常识矛盾，可能是命题方设置的反作弊/反照抄标记。**不可采用**，并建议在论文中主动说明为何不采用：

| 陷阱说法 | 为什么错 | 正确处理 |
|---|---|---|
| "冲突定义为 z 分数之差 > 0.5，全样本冲突率约 0%，说明各指标一致性极高" | 先用定义算出 0% 再下"无冲突"结论是循环论证；22 个异质指标（如广告含量 vs 教育价值）必有冲突样本 | 应使用稳健的冲突度量（如 IQR 缩放的 robust z 分数、密度峰值），给出冲突率随阈值的变化曲线，而非预设结论 |
| "ε_N=+0.32、ε_D=+0.18、ε_Q=+0.47 均为正，表明扩大规模与提升质量都会使 Loss 上升" | 与标度律 $L=E+AN^{-\alpha}+BD^{-\beta}$ 直接矛盾；弹性定义漏了负号 | 弹性应定义为 $\varepsilon_N=-\frac{\partial L}{\partial N}\frac{N}{L}>0$ 表示"Loss 下降的贡献率" |
| "Loss 与 Benchmark 相关系数 r=+0.68，为正相关，即 Loss 越高 Benchmark 越高" | 交叉熵损失越低能力越强，正常应为**负**相关 | 自己用 C5/C6 算相关系数并报告符号；若为正值须检查口径（如是否混入不同验证集） |
| "C=10²² FLOPs 时最优解 N\*=1.2×10⁻³ B、最优质量恒为 Q\*=1，无需比较成本函数" | N 比最小模型还小 60 倍，明显荒谬；质量成本随 g(Q) 形式差异巨大，Q 恒为角点解需验证而非预设 | 用 KKT 条件检验 Q 是否内点/角点，三种 g 形式逐一求解比较 |
| "配比建模 OLS 直接回归，训练 R²=0.93，外推直接套用原系数，无需重新估计" | 训练/外推表对应 10B/70B 尺度，参数跨 4 个数量级，系数必然漂移 | 外推须引入尺度项（见问题一 3.4）或做稳健性对比 |
| "可用 K-means 聚类代替方向统一""分类正确率超 80% 即可，无需 R²""穷举网格+罚函数系数取 1"等 | 均违背赛题明文要求（指标须统一为"越高越好"；须给出定量模型与检验） | 按题目要求做连续定量模型 |

**正确姿态：凡是"建议不用验证、直接套结论"的话术，一律视为待检验假设。**

---

## 1. 赛题主线与数据地图

四问是递进的：问题一产出 **质量分 $Q$、配比–损失关系**；问题二把 $Q$ 与 $\mathbf p$ 嵌入广义标度律 $L(N,D,Q,\mathbf p)$；问题三在算力约束 $6ND + D[g(Q)-g(Q_0)]_+ + \eta NDL_{ctx}\le C$ 下求最优 $(N,D,Q,\mathbf p)$ 并识别结构性转移；问题四把 Loss 翻译成 Benchmark 得分（C5/C6 桥接），分解"规模扩张 vs 非规模技术进步"并预测开源前沿。

**数据编号速查**（路径均相对 `F题/附件/`）：

- A1 `A_data_value/slimpajama_quality_signal_sample.jsonl.xz`（51,230 条 × 27 字段，7 质量域）
- A2/A3 `A_data_value/slimpajama_quality_extended/{arxiv,github}_*.jsonl.xz`
- A4–A15 `A_data_value/regmix_tables/`（train 512 组 / test 256+256+64 组 / est 63+63 组，配比表与 Loss 表按 `index` 对应）
- A16 `A_data_value/domain_mapping_guide.csv`（17 配方域 → 7 质量域映射）
- B1 `B_scaling_laws/pythia_training_log_existing.csv`（1,176 行，0.07–12B）
- B2–B12 同目录（cerebras 日志、pythia 插值轨迹、scaling_baseline 57 点、published 44 点、NQ 半合成 360/450/1704 点、100B+ 模型 132 点 + 预估 128 点）
- C1/C2 `C_efficiency_evolution/leaderboard_{cleaned,enhanced}.csv`（4,576 行，6 维 Benchmark）
- C3 `leaderboard_extended_timeseries.csv`；C4 `epoch_all_ai_models.csv`（3,523 行 × 57 列）
- C5/C6 `loss_benchmark_bridge{,_expanded}.csv`（43/75 条，含 Loss_Comparability 分层）
- C7 `model_architecture_metadata.csv`（45 模型 × 7 列，含 max_position_embeddings）
- C8 `C_efficiency_evolution/detailed_results/`（1,863 模型子目录，逐任务 JSON，**必用**）

单位注意：$N$、$D$ 以 $10^9$ 为单位（1.04 = 1.04×10⁹），代入 $C=6ND$ 时须乘 $10^{18}$；$C_{FLOPs\_1e21}$ 以 $10^{21}$ FLOPs 为单位。A1 大文件用 `lzma.open` 流式读取，content 字段仅在线提取统计量。

---

## 2. 近两年（2024.09–2026.09）大模型厂商技术报告调研

### 2.1 总表

调研口径：以**公开发布的技术报告 / arXiv 论文**为准，覆盖 2024 年 9 月（Llama 3.1/Qwen2.5 之后）至 2026 年 9 月的主流开源/开放权重模型。"规模"列给出（总参数量/激活参数量/预训练 token 量）。

| 厂商 | 报告 / 模型 | 时间 | 规模 | 与四问相关的关键技术点 |
|---|---|---|---|---|
| DeepSeek | DeepSeek-V3（arXiv:2412.19437） | 2024-12 | 671B/37B，14.8T tokens，2.664M H800 时 | MLA 压缩 KV 缓存；MoE 无辅助损失负载均衡；**FP8 混合精度训练**；MTP 多 token 预测；DualPipe 通信重叠 → 问题三算力模型、问题四效率演进 |
| DeepSeek | DeepSeek-R1 / R1-0528 | 2025-01 / 2025-05 | 基于 V3 | 纯 RL（GRPO）涌现推理能力；后训练算力 ≈ 预训练 → 问题四"非规模技术进步"案例 |
| DeepSeek | DeepSeek-V3.2（DSA 稀疏注意力） | 2025-09 | 685B/37B | token 级稀疏注意力，降低长上下文开销 → 问题三 $C_{attn}$ 削减机制 |
| DeepSeek | DeepSeek-V4（arXiv:2606.19348） | 2026-04 | 百万 token 上下文 | CSA+HCA 混合稀疏注意力；流形约束超连接 mHC；**Muon 优化器** → 问题三 $L_{ctx}$ 敏感性、问题四 |
| Alibaba | Qwen3（arXiv:2505.09388） | 2025-04–05 | 235B-A22B MoE 等 8 档，36T tokens | 三阶段预训练数据配比（通用→ STEM/代码/推理加强）；混合思考/非思考模式；高质量多语言数据 → 问题一领域配比、问题四 |
| Alibaba | Qwen3-Next（80B-A3B）/ Qwen3.5（397B-A17B，Gated DeltaNet） | 2025-09 / 2026 | 1M 上下文 | 线性注意力 + 稀疏注意力混合架构 → 问题三长文本开销 |
| Moonshot | Kimi K2（arXiv:2507.20534） | 2025-07 | 1.04T/32B，15.5T tokens | **MuonClip（QK-Clip）零 loss spike**；WSD 学习率；退火阶段 400B+60B tokens；YaRN 扩上下文 → 问题四稳定性/效率案例 |
| Moonshot | Kimi K2.5（arXiv:2602.02276） | 2026-02 | 多模态，15T+ | 联合预训练/长上下文中训练分阶段数据配比 → 问题一配比思想 |
| Zhipu | GLM-4.5（arXiv:2508.06471） | 2025-07/08 | 355B/32B，23T tokens | 多阶段预训练 + mid-training 领域数据注入；expert iteration 后训练 → 问题一、问题四 |
| Zhipu | GLM-5 / GLM-5.1 | 2026-02 | 744B/40B，28.5T tokens | 引入 DSA；异步 RL 框架 slime；**华为昇腾 910B 全流程训练**（算力受限下的工程效率） → 问题三、问题四 |
| MiniMax | MiniMax-Text-01 / MiniMax-M1 | 2025-01 / 2025-06 | 456B/45.9B，1M 上下文 | Lightning Attention（线性注意力）；CISPO RL → 问题三长上下文开销 |
| MiniMax | MiniMax-M2 / M2.5 | 2025–2026 | — | 面向真实环境的 RL → 问题四 |
| Meta | Llama 4 Scout / Maverick | 2025-04 | 17B-16E / 17B-128E MoE | iRoPE 架构；专家混合 + 长上下文 → 问题三 |
| Mistral | Mistral Small 3.1（24B）/ Mistral 3 系列 | 2025-03 起 | 24B dense | 小模型高效率路线 → 问题四效率前沿对照 |
| Google | Gemma 3（27B 等） | 2025-03 | 27B/12B/4B/1B | 蒸馏自更大模型；知识蒸馏 → 问题四"非规模进步" |
| NVIDIA | Nemotron 系列（Nano 2 / H / Cascade，arXiv:2512.13607 等） | 2025 全年 | 7B–140B | 模型剪枝 + 蒸馏（Nemotron-T）；**Nemotron-CC 数据工程**：分类器集成过滤 + 合成改写（6.3T tokens，4.4T 去重 + 1.9T 合成） → 问题一质量建模的直接参照 |
| 字节 | Seed1.5/Seed2.0（公开报告） | 2025 | — | — |
| 蚂蚁 | Ling-1T | 2025 | 1T/50B，**FP8 训练的最大模型** | 国产芯片 FP8 训练 → 问题三 |
| 小米 | MiMo-V2-Flash | 2026 | 309B/15B | 混合注意力 → 问题三 |
| StepFun | Step-3.5-Flash | 2026 | 196B/11B | 三路 MTP；推理模型低成本化 → 问题四 |
| Allen AI | OLMo 2 / Dolmino Mix | 2025 | 7B–32B | 全开放数据配方（DCLM + 领域数据混合消融） → 问题一配比建模参照 |
| AI2/UW | OpenScholar（Nature 650:857–863, 2026；arXiv:2411.14199） | 2026-02（Nature 版） | 8B + 45M 论文检索库 | **赛题背景直接引用**：8B 专门化模型凭数据质量/领域聚焦超越 GPT-4o（引用准确率持平人类专家，GPT-4o 幻觉率 78–90%） → 问题一、问题四动机 |

### 2.2 调研结论（对赛题的四点支撑）

**（a）数据质量与配比已成为第一等公民（→问题一）。** 主流报告均披露大规模数据工程：Nemotron-CC 用 FineWeb-Edu 与 DCLM 双分类器集成过滤并做合成改写；DCLM（Li et al., 2024）证明"模型分类器过滤 + 去重"的语料在同预算下显著优于原始网页；FineWeb-Edu（Penedo et al., 2024）用 1.3T tokens 即可对标远大数据量；Qwen3/GLM-4.5/Kimi K2 都采用"通用预训练 → 领域加强 mid-training/annealing"的两段式配比。这些正是 A1–A3 的 SlimPajama-Meta-rater 22 维质量信号与 A4–A15 的 RegMix 配比实验所抽象的对象。

**（b）经典标度律已被系统性扩展（→问题二）。** 除 Kaplan/Hoffmann 外：Muennighoff et al. (2023) 的**数据受限标度律**把重复数据的边际效用衰减写成 $D'=U_D[1+R_D^*(1-e^{-R_D/R_D^*})]$，约 4 个 epoch 内重复数据几乎等效新数据，之后收益骤降（2026 年 Prescriptive Scaling Laws, arXiv:2605.01640 进一步给出加性过拟合惩罚形式并在外部数据上验证）。这为"质量项应做成**有效数据量乘子**"提供了直接先例：低质量数据 ≈ 打折的有效 token。

**（c）算力效率改进层出不穷（→问题三）。** 降低 $C_{train}$ 的手段：FP8 混合精度（DeepSeek-V3、Ling-1T）、MoE（用激活参数计 FLOPs）、MTP、Muon 系优化器；降低 $C_{attn}$ 的手段：MLA（DeepSeek-V2/V3）、GQA、线性/稀疏注意力（Lightning Attention、NSA/DSA、CSA+HCA、Qwen3.5 Gated DeltaNet）。赛题给出 $\eta=2\times10^{-4}$ 的简化代理，实际文献表明稀疏注意力可把等效 $\eta$ 降低一个数量级以上——这正好支撑问题三"讨论成本函数/敏感性"的要求：$\eta$ 是架构效率的代理参数，值得做敏感性分析。

**（d）能力增长 = 规模扩张 + 效率演进（→问题四）。** Epoch AI 估计 LLM 达到同等 next-token loss 所需算力每年约 **3×** 下降（等效算力增长）；用"追赶前沿"（同能力水平的最小训练算力随时间下降）口径的估计达 **16–60×/年**（2025-12 LessWrong 分析，基于 Epoch 算力估计 + Artificial Analysis Intelligence Index）；算力受限背景真实存在（GLM-5 在昇腾集群上完成训练、DeepSeek 用 H800）。这给出问题四分解两类贡献的先验区间：**算力（规模）贡献占大头，但算法/数据效率贡献每年数倍、不可忽略**。

---

## 3. 问题一：数据质量评价、质量冲突消解与领域配比建模

### 3.1 任务拆解

三个子任务共用一条流水线：**指标预处理 → 综合评分 $Q$ → 冲突分析 → 域级聚合与跨体系映射 → 配比–损失建模**。数据：质量信号 A1–A3（22 指标，7 质量域 vs 17 配方域），配方实验 A4–A15（512 训练组 + 3 套检验组 + 2 套外推组）。

### 3.2 子任务 1：质量综合评价模型（方案）

**第一步：22 维指标预处理。**（文献参照：DCLM/FineWeb-Edu 的信号工程设计思想）

1. **列表型指标压缩**：8 个多维列表字段（rps_doc_*、rps_lines_* 等）先压为标量。可选：均值、分位数加权（如 0.25/0.5/0.75 分位线性组合）、或截尾均值。建议用**中位数**（对重尾稳健），并在附录给出不同压缩方式的稳健性对比。
2. **缺失/异常处理**：按域分组报告缺失率；异常值用 IQR 缩尾（winsorize 1%/99%）。
3. **方向统一**（题目强制）：Min-Max 归一化到 $[0,1]$，负向指标（如 ad_en、rps_doc_frac_no_alph_words、rps_lines_numerical_chars_fraction 等"越低越好"项）做补转换 $x\mapsto 1-x$。所有指标最终"越高越好"。

**第二步：综合评分（给出两套并比较）。**

- **方案 1a（客观赋权）**：熵权-TOPSIS 或 CRITIC 赋权 + 加权几何平均。几何平均对"短板指标"更敏感（某一维极差会把总分拉低），符合"垃圾进垃圾出"的直觉。
- **方案 1b（数据驱动降维）**：对 22 维做 PCA，取前 2–3 个主成分，以方差贡献率为权重合成 $Q\in[0,1]$。22 维指标间相关性高（如 fluency/readability 簇、rps_doc_* 簇），PCA 可压缩冗余；附录须给出载荷矩阵并**解释主成分含义**（对照域知识：教育价值/语言质量/噪声程度）。
- 比较两方案的 Spearman 相关；若一致性高则任选其一为主线，另一个作稳健性。

**第三步：聚合与对照（题目强制）。** 样本级 $Q$ → 域级 $\bar Q_d$ 用**token 加权或中位数**（长尾语料下中位数更稳健，两种都报）。A2/A3 与 A1 须用**同一套预处理与同一组权重**计算域级 $Q$，做对照表（7 域 × {抽样集, arxiv 扩展, github 扩展}）+ 两两差异的效应量。预期发现：抽样集按域抽样，扩展集为该域全量，分布差异本身揭示**抽样偏差**——这恰是命题让"对照"的目的，论文应讨论。

**可靠性验算（用 A18）**：从 regmix_domain_sample 抽各领域原始文本，人工/规则抽检高分样本与低分样本的 content，报告命中率（如"教育价值高且广告含量低"的样本抽查 20 条，专家打分一致率）。

### 3.3 子任务 2：质量冲突消解（方案）

**冲突定义**（须自定义并论证，勿抄 PDF 页眉）：对每篇文本，先将各指标转为稳健 z 分数 $z_i=(x_i-\text{med}_d)/\text{IQR}_d$（按域分组标准化，消除域间基线差），定义冲突得分

$$c(x)=\max_i z_i(x)-\min_i z_i(x),\qquad \text{冲突样本}=\{x: c(x)>\tau\},$$

阈值 $\tau$ 不拍脑袋：画 $c(x)$ 的分布直方图 + 核密度，用**双峰检测/肘部法**或 POT（超越阈值的峰值）法自适应选 $\tau$，并报告冲突率随 $\tau$ 的曲线。成因分析：对冲突样本按域、按指标对（如 fineweb_edu vs ad_en）做交叉表与卡方检验，识别"教育价值高 × 广告多"（营销式科普）、"代码质量高 × 语言流畅度低"（GitHub 领域天然冲突）等结构。

**消解规则（建立相应的综合评价模型）**：

- **稳健加权**：冲突样本的极端指标降权（如用 Huber 权重 $w_i=\min(1, k/|z_i|$)），相当于稳健主成分/稳健因子分析。
- **分簇融合**：对 22 维指标做因子分析提取 3–4 个公共因子（教育价值、语言质量、噪声、结构），同一文本不同因子得分差过大即冲突；融合时对冲突因子取**调和平均**（惩罚两极），对一致因子取算术平均。这样"消解"不是删掉矛盾信息，而是用非线性融合规则表达"质量由短板决定"。
- **扩展集复检**：对 A2/A3 跑同一管线，比较冲突率与成因排序是否一致（题目强制）。预期 arxiv 域冲突率低于 github 域，因为 github 天然存在"内容价值高、语言形式差"。

### 3.4 子任务 3：领域配比建模（方案）

数据：A4（512 组 17 域配比）配对 A5（13 个验证域 Loss；nih_exporter/enron_emails/europarl/philpapers 无 Loss 列，说明处理方式：保留为解释变量但无法单独评估，或合并入相近域）。

**关键预处理：clr 变换。** 配比是成分数据（闭合效应），直接 OLS 会有伪相关。用中心化对数比变换：$z_i=\ln(p_i/g(\mathbf p))$，$g(\mathbf p)$ 为几何均值。也可用 Aitchison 距离做近邻分析。

**基线模型（RegMix 思想，ICML 2024）**：对每个验证域 $v$ 的 Loss 拟合

$$L_v(\mathbf p)=b_v+\sum_i \beta_{v,i}\, z_i + \varepsilon,\qquad \text{或用幂形式 } L_v = \exp\!\big(b_v+\textstyle\sum_i \beta_{v,i} z_i\big).$$

用 512 组训练拟合，在 A6–A11 三套检验集（1M/60M/1B 规模）上验证，报告每域 R²/RMSE 与平均排名。RegMix 论文报告线性模型已能较好预测配比效果——但注意这是 1M 参数小模型的实验。

**是否引入 $Q$（须论证，建议引入）**：通过 A16 把 17 配方域映射到 7 质量域（direct 直接采信、near_direct 微调、inferred 降级使用并注明），把问题一的域级 $\bar Q_d$ 作为该域配比的**交互项**或**先验惩罚**：

$$L_v(\mathbf p)=b_v+\sum_i \beta_{v,i} z_i + \lambda \sum_i p_i\, (1-\bar Q_{d(i)}) + \varepsilon,$$

$\lambda>0$ 表示质量低的数据要"打折"贡献。检验 $\lambda$ 的显著性即可回答"质量是否独立影响配比–损失关系"。

**外推稳健性（A12–A15，题目强制）**：外推表是 10B/70B 尺度预估 Loss（幂律外推生成，非观测）。做法：把 1M 拟合的系数直接预测外推表，与 est_pile_loss 对比；预期系统性偏差 → 引入**尺度修正项**（如乘子 $(N/10^9)^{\kappa}$ 或对系数做 $(N/N_0)^{\rho}$ 漂移），用 1M/60M/1B 三套检验集校准尺度依赖后，再看外推误差是否收窄。用灰色关联度或排名相关性（Spearman）给出"外推结论稳健性"排序，**不得**直接宣称系数不变。

---

## 4. 问题二：跨维度数据融合与广义标度律

### 4.1 建模目标

构造 $L(N,D,Q,\mathbf p)$，要求：① $Q{=}1$（理想数据）时退化为经典 Chinchilla 形式（题目明确鼓励此约束）；② $\mathbf p$ 以合理方式进入；③ 用 B1 拟合主参数，B2/B3 族外/轨迹验证，B4/B5 跨族验证，B6–B8 标定 $Q$ 参数，B9/B10 百亿外推。

### 4.2 候选形式（给出三个层次，建议以 b 为主线）

**（a）有效数据乘子型（首选，文献支撑最强）**

$$L(N,D,Q,\mathbf p)=E+\frac{A}{N^{\alpha}}+\frac{B}{\big(D\cdot h(Q,\mathbf p)\big)^{\beta}},\qquad h(Q,\mathbf p)=Q^{\,\nu}\cdot \prod_i \Big(\frac{p_i}{p_i^*}\Big)^{\!-\theta_i \mathbb 1[p_i<p_i^*]},$$

- $h(Q,\mathbf p)\in(0,1]$：数据质量与配比偏离最优都会**打折有效 token 数**；$Q{=}1$ 且 $\mathbf p=\mathbf p^*$ 时 $h=1$，退化为经典律。
- $h$ 取 $Q^\nu$ 的依据：Muennighoff 数据受限律中"重复数据按指数衰减折成有效数据"已有先例；质量折扣同构。$\nu$ 用 B6–B8（含 $Q_{score}$ 列的半合成实验）拟合；B6–B8 须标注半合成、给可信度边界。
- 配比惩罚只罚"低于最优"的一侧，$\mathbf p^*$ 由问题一配比模型给出（RegMix 最优域）。$\theta_i$ 用 A4/A5 拟合（但题目要求本问"不重复读取 A"，配比信息经由问题一的 $\mathbf p^*$ 与配比效应量引入，说明该假设即可）。

**（b）不可约损失修正型（创新点，可与 a 叠加）**

$$E(Q)=E_{\infty}+\frac{c}{(1+\kappa Q)^{\rho}},$$

数据质量决定 Loss 地板（脏数据带来不可约误差，如标注噪声/广告文本的模式不可学）。$Q\to1$ 时 $E\to E_\infty+c(1+\kappa)^{-\rho}$ 仍有限，若坚持严格退化，可令此项仅作对比模型。

**（c）完全乘积型（备查）**：$L=E+A N^{-\alpha} Q^{\nu_N}+B D^{-\beta} Q^{\nu_D}$，把质量同时折入两项；缺点是 $Q\to0$ 时 $L\to E$（脏数据反而无损），不合理，不建议。

### 4.3 参数估计与验证设计

1. **B1 拟合**：经典四项参数 $(E,A,\alpha,B,\beta)$ 用 Huber 损失在 log 空间非线性最小二乘（比直接 log 线性化更稳，因为 $\ln(L-E)$ 对 $E$ 敏感）。以最终步收敛点为锚（loss 尾段趋稳）可缓解中间检查点的"未收敛偏差"。
2. **质量参数**：B6–B8（360/450/1704 个 N-D-Q 点）拟合 $\nu$：固定 $(E,A,\alpha,B,\beta)$（量级兼容时）或联合重拟合；B8 含更大参数外推，专门用于检验 $\nu$ 的尺度稳定性。报告 $\nu$ 的置信区间，并明确"半合成，噪声由真实标度律校准叠加，结论只作边界参考"。
3. **验证矩阵**：B2（Cerebras 半合成轨迹，族外）与 B3（Pythia 插值轨迹）至少用一个；B4（12 族 57 收敛点）与 B5（Kaplan/Hoffmann 文献 44 点）做跨族外部验证，报告预测区间覆盖率（如 95% 预测带覆盖实测的比例）。
4. **假设检验**：提出"质量弹性与数据弹性同号且量级相当"等可检验假设，用 B6–B8 的偏相关/弹性分解检验。

### 4.4 边际效用、弹性与"质量–参数替代"推导（题目核心得分点）

定义（注意负号，勿抄页眉错误）：

$$\varepsilon_N=\frac{\partial \ln L}{\partial \ln N}=-\alpha\frac{AN^{-\alpha}}{L}<0,\quad \varepsilon_D=-\beta\frac{BD^{-\beta}h^{-\beta}}{L}<0,\quad \varepsilon_Q=-\nu\frac{\ln h\cdot BD^{-\beta}h^{-\beta}}{L}<0.$$

弹性均为负（Loss 随投入下降），其绝对值即"该因素的贡献率"。

**"质量提升 0.1 等价于参数增加多少"的可计算条件**：保持 Loss 不变，对模型 (a) 求全微分并令 $dL=0$：

$$\frac{dN}{N}=\frac{\nu\ln\!\big(\tfrac{Q+0.1}{Q}\big)}{\alpha}\cdot\frac{BD^{-\beta}h^{-\beta}}{AN^{-\alpha}}.$$

右边全部可用拟合参数与当前 $(N,D,Q)$ 算出——这就是"可计算条件"的显式表达；当 $\alpha A N^{-\alpha}\approx \beta B D^{-\beta}h^{-\beta}$（Chinchilla 最优邻域）时简化为 $\Delta N/N\approx \tfrac{\nu}{\alpha}\ln(1+0.1/Q)$。**替代强度**（Q 与 N 的边际替代率随规模的漂移）建议用 CES 型生产函数刻画画出等 Loss 曲线：$L$ 取等值时 $(N^\phi+Q^\phi)^{1/\phi}$ 的 $\phi$ 估计即替代弹性参数（Morishima 弹性作补充）。

**领域间替代/互补**：用问题一配比模型的系数符号与交叉项检验：两域 $z_i z_j$ 交互项显著为负 → 互补；为正 → 替代。报告 17 域的相关/替代矩阵热力图。

---

## 5. 问题三：算力约束下的多维资源联合优化与结构性转移

### 5.1 优化模型

$$\min_{N,D,Q,\mathbf p}\; L(N,D,Q,\mathbf p)\quad \text{s.t.}\quad \underbrace{6ND}_{C_{train}}+\underbrace{D\,[g(Q)-g(Q_0)]_+}_{C_Q}+\underbrace{\eta N D L_{ctx}}_{C_{attn}}\le C,$$

其中 $\eta=2\times10^{-4}$，$L_{ctx}$ 外生给定（可行域依据 C7 的 max_position_embeddings 分布确定），三种 $g$ 形式（指数/幂/对数渐进）逐一求解并比较。$\mathbf p$ 处理理由：建议**固定 $\mathbf p=\mathbf p^*$（问题一最优配比）作主线**——配比不直接消耗算力，且问题一已单独研究；同时把 $\mathbf p$ 作敏感性变量（在最优邻域 ±20% 扰动）验证结论稳健性，兼顾题目"可作联立变量"的要求。

**$L_{ctx}$ 的临界值（题目要求解析给出）**：令 $C_{attn}=C_{train}$ 得 $L_{ctx}^{crit}=6/\eta=3\times10^{4}$ tokens。$L_{ctx}<3\times10^4$ 时训练开销主导；超过则注意力开销主导。结合 C7（45 个主流开源模型的 max_position_embeddings）给出实际可行域（如 4K–1M 的分布分位数），在域内取低/中/高档 + 临界点附近加密做敏感性分析，**不得仅凭主观假设**——题目明文要求。

### 5.2 求解方法（建议三轨并行）

1. **解析/KKT 轨（论文加分）**：预算约束在最优点取等号（$L$ 关于各变量递减）。把 $D$ 用约束消去，对 $(N,Q)$ 两维做内点搜索；推导 KKT 条件并讨论 $Q$ 的解结构——**用实际参数检验 $Q^*$ 是内点还是角点**（对数渐进型 $g$ 下质量成本低，$Q$ 可能拉满；指数型 $g(Q)=10^7 e^{6Q}$ 下 $g(1)-g(Q_0)$ 极大，$Q^*$ 几乎 surely 为内点）。这正是要推翻页眉陷阱"各档预算最优质量恒为 1"的地方：按附录 B 参数，指数型 $g(1)=10^7\cdot e^6\approx4\times10^9$ FLOPs/token，超过 $6N$，直接拉满 Q 会让质量成本吞掉全部预算。
2. **数值轨**：对 $\log_{10}N\in[-3,3]$、$\log_{10}D\in[-3,3]$、$Q\in(0,1]$ 取对数网格粗搜 + SLSQP/信赖域细化；$\mathbf p$ 用单纯形投影。小模型约束（如 $N\ge10^7$ 防退化解）须说明依据（Pythia 最小 70M）。
3. **验证轨**：用变步长有限差分校验梯度，报告最优解对网格密度的收敛性。

### 5.3 三档预算与结构性转移识别

对 $C\in\{10^{19},10^{22},10^{24}\}$（FLOPs）求解（单位换算：$N,D$ 以 $10^9$ 计时 $C=6ND$ 须乘 $10^{18}$）。

**结构性转移的数学定义**（给出明确定义，勿只画图说话）：

- **份额定义**：$s_N=C_{train}/C$、$s_Q=C_Q/C$、$s_L=\eta N D L_{ctx}/C$，及 $D/N$ 比。
- **转移判据**：存在预算区间 $[C_1,C_2]$ 使得某资源份额的排序发生交换，或份额变化超过阈值 $\Delta s>\delta$（如 0.2）且经扰动检验显著。识别方法：对 $C$ 取对数网格（$10^{18}$–$10^{25}$，每 0.25 dex 一点）扫最优解 → 用 **CUSUM/贝叶斯变点检测**在份额序列上定位转移点 → 用参数 bootstrap（重抽样 B1、B6 拟合残差）检验转移点显著性。

**预期叙事**（以实际求解为准）：低预算时数据/质量投入占比高（小模型喂太多参数浪费，Chinchilla 最优要求 $D/N\approx20$ token/参数）；随预算上升，$Q$ 成本函数形态决定质量投入的可持续性；$L_{ctx}$ 高档时注意力份额挤占训练预算，最优 $N$ 下移。若三档预算下结构无显著转移，也要如实报告并解释（如 $g$ 使质量始终优先），这才是"分析"。

### 5.4 成本函数选择的影响

三种 $g$ 各解一遍，比较 $(N^*,D^*,Q^*,L^*)$ 的差异：指数型惩罚高质量区 → $Q^*$ 内点且随预算缓升；幂函数型 $g=\gamma Q^4$ → 质量成本集中在高端；对数渐进型 → 质量便宜、$Q^*\to1$。给出"成本函数不确定性下最优解的包络带"，这比单一结论更符合建模竞赛的稳健性要求。

---

## 6. 问题四：技术演进分析与前沿预测

### 6.1 口径声明（题目强制，先定义再分析）

- **综合能力度量**：六维（IFEval/BBH/MATH Lvl 5/GPQA/MUSR/MMLU-PRO）算术均分为主线；附加权方案（等权 vs 标准化后等权）作稳健性。
- **开源筛选**：以 C1 的 Hub License + C4 的 open weights 字段双重口径；区分"开放权重可复现"（Apache/MIT/Llama 社区许可）与"仅 API"。
- **类型区分**：pretrained base 与 chat/finetuned 分层建模（C1 有 Type 字段），两者前沿不可混算。
- **时间轴**：发布日期（Epoch AI 匹配日期）为主线，说明不采用提交日期的原因（提交≠能力可获得）。

### 6.2 规模扩张 vs 非规模技术进步的分解（核心）

**前沿面回归法（推荐主线）**：对每个时间点 $t$，取能力–算力平面的 Pareto 前沿（用 C4 的算力字段），拟合

$$\ln \text{Score}_f(t)=\beta_N \ln N_f(t)\ \big(\text{或}\ \ln C_f(t)\big)+\beta_t\, t+\gamma+\varepsilon,$$

- $\beta_N\,\Delta\ln N$ 贡献 = **规模扩张贡献**；$\beta_t\,\Delta t$ 贡献 = **非规模技术进步贡献**（架构/数据/对齐，如 MLA、MuonClip、RL 后训练——见第 2 节调研表）。
- 贡献占比 = 各自解释的能力增幅 ÷ 总增幅；报告逐年分解表。
- **对照方法**：Solow 余值（$\Delta T=\Delta\text{Score}-\alpha\Delta\ln N$）与 Malmquist 指数，三法互证。

### 6.3 C8 逐任务聚合（题目强制，至少一项）

对 C8 逐任务 JSON 按目录取最新可解析记录，**按单一任务**（建议 MATH Lvl 5 与 GPQA 各做一遍）重算逐任务前沿时间序列，重复 6.2 分解。预期：数学/推理任务的技术进步斜率 $\beta_t$ 显著高于知识型任务（MMLU-PRO），因为 RL 后训练主要提升推理——这一"逐任务不一致"本身就是非规模技术进步存在且不均匀的有力证据。跳过 4 个截断 JSON 并说明。

### 6.4 Loss–Benchmark 桥接与预测

1. **桥接**：用 C6（75 条，按 Loss_Comparability 分层）。高可比层（Pythia 系，Loss 口径与附件 B 一致）拟合映射 $B=\phi(L)$（建议单调递减的负指数或对数线性形式）；中可比层只作外部校验不参拟合。报告映射 $R^2$、残差分布；映射误差传播：把参数 bootstrap 得到的 $L$ 预测分布经 $\phi$ 变换，得到 Benchmark 预测的**区间**。
2. **前沿预测（12/24 个月）**：两条路线交叉验证——
   - **路线 A（数据驱动）**：开源 chat 模型综合能力前沿的时间序列（月度前沿分位数）+ HP 滤波分离趋势与周期 → 趋势外推，按算力增长放缓情景（基准：Epoch 口径近年训练算力年增约 4–5×；放缓情景：1.5–2×）设两种情景，用 6.2 的分解把"算力贡献"按情景缩放、"技术贡献"按历史 $\beta_t$ 保守外推。
   - **路线 B（模型驱动）**：问题二三的广义标度律预测未来算力下的可达 Loss → 经 $\phi$ 换算为 Benchmark 分数前沿。
3. **不确定性分析**：参数 bootstrap + 情景分析 + 映射误差传播，给出 12/24 月前沿分数的 80%/95% 预测区间，并明确"开源前沿 ≠ 闭源前沿"的口径边界。

### 6.5 ⚠️ 本问必须自己重算的三个量

- Loss–Benchmark 相关系数（页眉声称 r=+0.68 为正，违背常识，**自己算并核对符号与样本构成**）；
- 规模/技术贡献占比（页眉称 $b_N=-6.2$、$b_t=0.02$"几乎无技术进步"，与第 2 节调研的 Epoch/LessWrong 证据矛盾，**用自己的分解验证**）；
- 前沿预测的基准情景（不要用未说明依据的马尔可夫等级转移糊弄）。

---

## 附录 A：数据利用清单（论文附录模板）

| 编号 | 文件 | 用在哪一问 | 使用方式 |
|---|---|---|---|
| A1–A3 | 质量信号三集 | 问题一 1、2 | 全量流式处理，22 指标统一方向，域级 $Q$ 对照 |
| A4–A5 | train 配比/Loss | 问题一 3 | clr 变换回归拟合 |
| A6–A11 | test 三套 | 问题一 3 | 配比模型检验 |
| A12–A15 | est 外推两套 | 问题一 3 | 外推稳健性（标注非观测） |
| A16/A17/A18 | 映射/摘要/原始文本 | 问题一 | 跨体系映射；A18 抽查验算 |
| B1 | Pythia 日志 | 问题二 | 主拟合（真实） |
| B2/B3 | Cerebras/插值轨迹 | 问题二 | 族外/轨迹验证（B2 半合成，标注） |
| B4/B5 | 跨族/文献点 | 问题二 | 外部验证（真实） |
| B6–B8 | NQ 半合成 | 问题二 | 质量参数 $\nu$（标注半合成与边界） |
| B9/B10 | 100B+ 参数/预估 Loss | 问题二 | 百亿外推讨论（B10 估算，标注） |
| B11/B12 | 元数据/索引 | 问题二 | 辅助 |
| C7 | 架构元数据 | 问题三 | $L_{ctx}$ 可行域依据 |
| C1/C2、C3、C4 | 榜单/时序/Epoch | 问题四 | 能力度量、时间轴、算力与开源口径 |
| C5/C6 | Loss–Benchmark 桥 | 问题四 | 分层桥接映射 |
| C8 | 逐任务 JSON | 问题四 | ≥1 项逐任务聚合（必用） |

## 附录 B：参考文献（已核验，供引用）

1. Hoffmann et al. Training Compute-Optimal Large Language Models (Chinchilla). NeurIPS 2022.
2. Kaplan et al. Scaling Laws for Neural Language Models. arXiv:2001.08361.
3. Muennighoff et al. Scaling Data-Constrained Language Models. NeurIPS 2023（数据受限标度律；datablations 仓库）.
4. Xia et al. RegMix: Data Mixture as Regression for Language Model Pre-training. ICML 2024.
5. Biderman et al. Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. ICML 2023.
6. Bahri et al. Explaining neural scaling laws. PNAS 121(27), 2024.
7. Penedo et al. The FineWeb Datasets / FineWeb-Edu. 2024（arXiv:2406.17557 及 edu 报告；38B edu token 对标 350B 未过滤数据）.
8. Li et al. DataComp-LM: In Search of the Next Generation of Language Datasets. NeurIPS 2024（DCLM，fastText 分类器过滤）.
9. Soldaini et al. Dolma: An Open Corpus of Three Trillion Tokens. 2024.
10. Su et al. Nemotron-CC. 2024/2025（分类器集成 + 合成改写，6.3T tokens）.
11. DeepSeek-AI. DeepSeek-V3 Technical Report. arXiv:2412.19437, 2024-12.
12. DeepSeek-AI. DeepSeek-R1. arXiv:2501.12948, 2025-01.
13. DeepSeek-AI. DeepSeek-V4: Towards Highly Efficient Million-Token Context. arXiv:2606.19348, 2026-04.
14. Qwen Team. Qwen3 Technical Report. arXiv:2505.09388, 2025-05.
15. Kimi Team. Kimi K2: Open Agentic Intelligence. arXiv:2507.20534, 2025-07.
16. Kimi Team. Kimi K2.5. arXiv:2602.02276, 2026-02.
17. GLM-4.5 Team. GLM-4.5. arXiv:2508.06471, 2025-07.
18. Zhipu AI. GLM-5 模型卡与技术博客. 2026-02（744B/40B，28.5T tokens，DSA，slime 异步 RL）.
19. MiniMax. MiniMax-M1. arXiv:2506.13585, 2025-06（456B MoE，Lightning Attention，1M 上下文）.
20. Meta. The Llama 4 Herd. 2025-04.
21. NVIDIA. Nemotron-T / Nemotron-Cascade（arXiv:2512.13607）等. 2025.
22. Asai et al. Synthesizing scientific literature with retrieval-augmented language models. Nature 650, 857–863 (2026)（OpenScholar-8B；DOI:10.1038/s41586-025-10072-4）.
23. Erdil & Besiroglu / Epoch AI. Algorithmic progress in language models（~3×/年等效算力；epoch.ai）.
24. "Catch-up algorithmic progress might be 60×/year"（LessWrong, 2025-12；Epoch 算力 + Artificial Analysis Index 方法）.
25. Schaeffer et al. Are Emergent Abilities of Large Language Models a Mirage? NeurIPS 2023.
26. Prescriptive Scaling Laws for Data Constrained Training. arXiv:2605.01640, 2026-05（重复数据加性惩罚律）.
27. Lilian Weng. Scaling Laws, Carefully. 2026-06（综述博客，可作背景不作学术依据）.

## 附录 C：建议的代码与实验清单（可执行路线）

1. `q1_preprocess.py`：lzma 流式读 A1–A3 → 22 指标清洗/方向统一/域分组稳健 z → 落盘 parquet。
2. `q1_score.py`：CRITIC/熵权 + PCA 双管线 → 域级 $Q$ 对照表 + 冲突率–阈值曲线 + A18 抽查。
3. `q1_regmix.py`：clr 变换 → 13 验证域回归 → A6–A11 检验 → A12–A15 外推（尺度修正）→ 替代/互补矩阵。
4. `q2_scaling.py`：B1 Huber 非线性拟合 → B6–B8 标 $\nu$ → B2/B3/B4/B5 验证矩阵 → 弹性与替代条件推导（符号化求导）→ B9/B10 百亿外推。
5. `q3_optimize.py`：三 $g$ × 三预算 × $L_{ctx}$ 网格（依据 C7）→ KKT 检验 → 份额序列 + 变点检测。
6. `q4_frontier.py`：C1/C4 口径过滤 → 前沿面分解（规模 vs 技术）→ C8 逐任务聚合 → C6 分层桥接 → 双路线预测 + 区间。

## 附录 D：AI 工具披露模板（放论文末尾）

> 本文在资料整理阶段使用了 Kimi Work（Moonshot AI）辅助检索近两年大模型技术报告与数据集文档、整理赛题附件清单；**核心建模、公式推导、参数估计与结论均由参赛队独立完成**；AI 输出内容均经人工核验；未使用 AI 直接生成论文主体内容。

---

*文档生成时间：2026-09-23。技术报告调研覆盖 2024-09 至 2026-09 公开发布资料；引用条目均经检索核验，正式写作时请按期刊/会议格式补全卷期页码。*
