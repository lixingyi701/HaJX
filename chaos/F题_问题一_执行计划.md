# F 题问题一执行计划：数据质量评价、质量冲突消解与领域配比建模

> 依据：赛题正文（docx）问题一条款与附录 A/B、《数据说明》（pdf）附件 A 一节、`提示词_问题一_v2.md` 的输出要求。
> 参考：`问题一方法详解.md`、`F题_执行路线_修订版.md`（两份均为方案草稿，本计划已对其中可核查的数据事实逐条实测，结果见 §2；与实测不符处以本计划为准）。
> 环境实测：Python 3.12.7，pandas / numpy / scipy / scikit-learn / statsmodels 可用；**lightgbm、xgboost 未安装**（用 `sklearn.ensemble.HistGradientBoostingRegressor` 替代，或 `pip install lightgbm`）。A1 全量流式读取约 10 s，A3（20 万条）约 15 s，A18（13.8 万条）约 20 s，均无内存压力。

---

## 0. 目标与验收口径

### 0.1 官方三项子任务与硬性要求（逐条对照验收）

| 编号 | 官方要求（原文） | 验收证据 |
|---|---|---|
| R1 | 质量指标预处理，方向统一为"越高越好"（负向指标 Min-Max 后取 \(1-x\)） | 25 指标"变换—方向"表；处理后所有指标与 fineweb_edu 的相关方向一致性检查 |
| R2 | 综合评价模型 → 样本级 / 语料级 / 领域级 \(Q\)，说明聚合方式 | 样本级 \(Q\) 文件；7 质量域 + 17 配方域 \(Q_d\) 表（含区间） |
| R3 | **全量**质量信号：A1 + A2 + A3 全部记录；给出扩展集域级 \(Q\) 并与抽样集对照 | A2/A3 逐条处理计数（17,523 / 203,752）；arxiv、github 域"A1 子集 vs 全量"对照表 + KS 检验 |
| R4 | 定义冲突、分析成因、建立消解后的综合评价；覆盖抽样集，在扩展集检验主要结论 | 冲突率–阈值曲线；域 × 指标组交叉表；消解前后 \(Q\) 对照；A2/A3 复验表 |
| R5 | 用 A4–A15 建立 17 域配比 \(p\) 与 Loss 的定量关系；分析各域及组合的影响；论证是否/如何引入 \(Q\) | 13 个验证域回归模型；17×13 边际效应矩阵；嵌套模型检验 |
| R6 | 单纯形约束 \(p_i\ge0,\ \sum p_i=1\) 须合理处理 | 成分数据变换 + 零值处理说明；最优配比搜索在单纯形内进行 |
| R7 | A6–A11 检验；A12–A15 讨论外推稳健性 | 1M / 60M / 1B 三尺度 Spearman、R²、RMSE；外推表只报排名稳定性，并标注"est = 幂律外推，非观测" |
| R8 | 用原始文本检验评分可靠性（A18 可选） | 高分 / 低分 / 冲突样本各抽 20–30 条人工核验记录 |
| R9 | 论文附数据利用清单；半合成 / 外推数据标注 | 附录 A 清单；A12–A15 标注 |

### 0.2 与下游三问的接口（问题一必须落盘的五份产出）

| 产出 | 内容 | 下游用途 |
|---|---|---|
| P1-Q_sample | 样本级 \(Q\)（A1 + A2 + A3 共 272,505 条）+ 25 维标准化分 + 冲突标记 | 问题四按能力维度解释；论文附录 |
| P1-Q_domain | 7 质量域 + 17 配方域的 \(Q_d\)、95% 区间、来源等级（direct / near_direct / inferred） | 问题二 \(\bar Q(p)=\sum_i p_iQ_{d(i)}\)；问题三基线 \(Q_0\) |
| P1-f(p) | 配比 → 13 域 Loss 模型（系数、变换、零值规则）+ 17×13 边际效应矩阵 \(T\) | 问题二替代 / 互补分析；问题三 Loss 中的 \(p\) 项 |
| P1-p* | 最优配比 \(p^*\) 与 \(\bar Q(p^*)\)，以及基线配比（Pile 原始 / RegMix 人工）的 \(\bar Q\) | 问题三默认配比与 \(Q_0\)；问题二参照点 |
| P1-rank_decay | 排名一致性随规模 \(N\) 的衰减曲线（1M→60M→1B→est 10B/70B） | 问题二决定 \(p\) 进入标度律的形式；问题三外推不确定性 |

---

## 1. 科学问题与可判定假设

方案围绕三个科学问题组织（不是流水账）：

- **SQ1 信号合成与域偏差**：22 个异质指标如何合成跨域可比的质量分？
  H1：规则型指标与模型型指标的基线在 github / arxiv 等域存在系统性偏移（实测：github 域 `fluency` 概率中位数 0.56 vs c4 0.99；`frac_no_alph_words` 中位数 34.8% vs c4 20.3%）；域内标准化后跨组冲突率显著下降。
- **SQ2 冲突的定义与消解**：指标互相矛盾时怎么处理？
  H2：冲突集中在跨来源组（规则型 vs 模型型），且方向在域内有结构性偏向；"降权而非删除"的消解规则在 A2/A3 复验下结论不变。
- **SQ3 配比效应及其尺度边界**：配比→Loss 规律从 1M 到 60M / 1B 保持多少？质量分能否解释各域边际贡献？
  H3：排名相关随 \(N\) 单调下降但仍高（实测线性基线：1M 0.96 → 60M 0.96 → 1B 0.92，见 §2.3）；绝对 Loss 存在尺度平移，必须做尺度校准；\(\bar Q(p)\) 只能部分解释域边际效应。

---

## 2. 数据事实核查（本次实测，覆盖参考文档中的全部关键断言）

### 2.1 质量信号 A1–A3

| 事实 | 实测结果 | 对方案的影响 |
|---|---|---|
| A1 规模与域分布 | 51,230 条；arxiv 1,419 / book **171** / c4 10,000 / commoncrawl 9,640 / github 10,000 / stackexchange 10,000 / wikipedia 10,000 | book 域级统计必须带 bootstrap 区间 |
| 8 个列表型字段长度 | `modernbert_*` 长 6（全部记录）；`qurater` 长 4；`ad_en`、`fluency_en` 长 2；`fineweb_edu` 长 1 | 参考文档的"logits / 多评分器"判断成立：22 字段展开为 **25 指标** |
| modernbert 6 维 logits 的 argmax 分布 | cleanliness 集中在 4–5 档，reasoning 集中在 1–2 档，professionalism 分布分散 | softmax 期望分 \(\sum_k k\sigma_k\in[0,5]\) 合理；arxiv 域多项饱和至 5.00 |
| `ad_en` / `fluency_en` 二分类概率 | \(P(\text{label}=1)\)：ad 中位数 0.994（P10 = 0.756）；fluency 中位数 0.898，github 域 0.56 | 概率高度饱和 → 直接 Min-Max 后分布极偏；**改用 log-odds**（\(\ell_1-\ell_0\)）再做分位数 / Min-Max |
| DSIR 三项与文档长度的关系 | 与 `rps_doc_word_count` 的 Spearman = **−0.89 / −0.89 / −0.91**；量级 −24 ~ −1.67×10⁶ | 原始 DSIR 几乎就是"负的文档长度"，**必须按词数归一**（每词平均 log 比，均值 ≈ −10），再做域内标准化 |
| 规则型指标量纲 | `frac_*`、`uppercase`、`terminal_punct`、`numerical_chars` 为百分数（0–100）；`word_count` 1–5.5×10⁵，`num_sentences` 1–2.5×10⁴，重尾 | 重尾指标先 log，再 Min-Max |
| 缺失值 | 22 个指标字段在 A1 中**无缺失** | 缺失处理仅需在 A2/A3 上检查后说明 |
| A2 / A3 字段 | 24 字段（22 指标 + id + sub_path），无 `_source_domain`、`content`；`sub_path` 为空 | 域由文件名推断；A2/A3 无法做原文核验，核验只能用 A1 与 A18 |
| A1 与 A2/A3 的关系 | A1 的 arxiv 1,419 条 **100% 出现在 A2**，github 10,000 条 **100% 出现在 A3**（按 id）；`_source_path` 与 A2/A3 文件名一致；manifest 标注 "Range-sampled … from one file per domain" | "扩展集 vs 抽样集"对照的正确解释是**抽样代表性检验**（区间抽样 vs 全文件），而非两组独立数据；A2/A3 无重复 id |
| A16 映射 | direct 3（arxiv、github、stackexchange）；near_direct 3（wikipedia_en→wikipedia、gutenberg_pg_19→book、pile_cc→commoncrawl）；inferred 11，quality_domain 为 `(none)` | 11 个配方域没有任何质量信号，必须自建关联（走 A18） |
| A17 | 17 行：domain、source_path、sample_rows、sample_bytes_requested、avg_text_chars | 用于解释 word_count 类指标的域偏；`sample_rows` 与 A18 域计数一致 |
| A18 | 138,034 条，字段 `text`、`_source_domain`、`_source_path`；域计数：stackexchange 30,378 / pubmed_abstracts 29,895 / pile_cc 18,191 / wikipedia_en 17,511 / github 15,117 / uspto 11,415 / freelaw 5,100 / pubmed_central 2,629 / dm_mathematics 1,922 / nih_exporter 1,884 / arxiv 1,719 / enron 1,010 / hackernews 944 / europarl 157 / gutenberg 79 / philpapers 67 / **ubuntu_irc 16** | 小域文档极长（ubuntu_irc 均 22.5 万字符），需**切段**（如 2,000 字符 / 段）后再算规则型信号，以逼近 SlimPajama 文档粒度并扩大样本 |

### 2.2 配比实验 A4–A15

| 事实 | 实测结果 | 对方案的影响 |
|---|---|---|
| 配比行和 | 均值 0.99988，范围 0.996–1.003（千分位舍入） | 建模前重新闭合（按行归一） |
| **配比零值** | 各域零值比例 **30%–69%**（enron 68.8%、philpapers 59.8%、nih 58.0%……）；最小正值 0.001 | **clr 不能直接用**（\(\ln 0\)），必须做零值替换（乘性替换 \(\delta=0.0005\)，即检测限一半）或改用 \(\ln(p_i+\varepsilon)\)。参考文档未提及此问题 |
| Loss 列 | 13 域；nih_exporter、enron_emails、europarl、philpapers 无 Loss 列 | 4 域仅作解释变量 |
| 检验集结构 | **test 1M 与 test 60M 的 256 组配比完全相同**；test 1B 的 64 组与 train / test 1M 均不重叠 | 1M→60M 的排名不变性可**无需模型直接测**（同配比 Loss 的 Spearman ≥ 0.98，13 域均如此）；1B 只能经模型评估 |
| Loss 尺度平移 | 域均值：1M ≈ 5.0–6.5，60M ≈ 2.5–4.7，1B ≈ 1.1–3.1 | 用 1M 系数直接预测 60M/1B 的绝对 Loss 时 R² 为大负值（实测 −5 ~ −300）；须报告 Spearman 与"尺度校准后 R²"，不能只报原始 R² |
| est 表 | est_mixture_10b / 70b 的 63 组 **完全等于 train 的子集**（index 相同）；est Loss 均值 10B ≈ 1.3–2.3，70B ≈ 1.0–1.8 | 同一 63 组配比，观测 1M Loss 与 est 10B Loss 的 Spearman 在 pile_cc 仅 **0.10**、hackernews **0.02**、wikipedia 0.33，而真实 1M→60M 同域 Spearman 为 0.99；est 10B 与 est 70B 之间 Spearman ≈ 0.98–0.99 | est 表隐含的"排名大幅改变"与真实观测（1M→60M 排名几乎不变）矛盾，属出题方外推生成的产物；只能用于稳健性讨论并明确标注，不得当作验证 |

### 2.3 配比→Loss 线性基线（Ridge，目标 \(\ln L\)，13 域平均）

| 特征变换 | 训练 R² | Spearman 1M | Spearman 60M | Spearman 1B |
|---|---|---|---|---|
| 原始 \(p\)（RegMix 线性做法） | 0.670 | 0.846 | 0.844 | 0.749 |
| clr（零值乘性替换 \(\delta=5\times10^{-4}\)） | 0.807 | 0.907 | 0.903 | 0.894 |
| \(\ln(p_i+10^{-3})\) | **0.911** | **0.961** | **0.957** | **0.917** |

结论：对配比取对数型变换是提升可预测性的关键；原始 \(p\) 直接线性回归明显更差，且 1B 上退化最快。主线取 \(\ln(p+\varepsilon)\) 或 clr（二者都属 Aitchison 几何的近似），并以 GBM 作非线性对照（RegMix 原文 LightGBM Spearman 约 0.97–0.98）。

### 2.4 对两份参考文档的修正清单

1. 补充**配比零值问题**：clr 前必须零值替换；主线改为 \(\ln(p+\varepsilon)\) 与 clr 双轨对照。
2. 补充 **DSIR 长度归一**：仅"域内标准化"不够，原始 DSIR 与长度相关 −0.9，须先除以词数。
3. 补充 **概率饱和问题**：`ad_en`/`fluency_en` 用 log-odds 而非概率进入 Min-Max。
4. 修正验证口径：1M 与 60M 检验集为同一批配比，可直接测排名不变性；绝对 Loss 有尺度平移，"R² 直接套用"不成立。
5. 修正 est 表定位：est 与真实观测的排名相关性在部分域接近 0，稳健性讨论须点明这一矛盾。
6. `Meta-rater Table 11 权重`：本地无法核验其数值，仅在团队能查到原文时作对照，不写入主线。
7. 时间估计：A1–A3、A18 全量读取均在 1 分钟内，"大文件"不构成瓶颈。

---

## 3. 执行路线（Stage 0–6）

每阶段按"输入 → 步骤 → 方法与理由 → 输出 → 检查点"写。全部脚本放在 `F题/code/q1/`，中间结果放 `F题/outputs/q1/`（Parquet / CSV），图放 `F题/figs/q1/`。

### Stage 0：工程骨架（约 1 h）

- 目录：`code/q1/{00_io.py, 01_expand.py, 02_quality.py, 03_conflict.py, 04_a18_rule.py, 05_mixture.py, 06_export.py}`。
- `00_io.py`：`lzma.open` 逐行 `json.loads`；A1 只保留 `content` 长度与 `content` 的哈希（原文核验时按 id 再回读）；A2/A3 从文件名解析域；输出 `a1_raw.parquet`、`a2_raw.parquet`、`a3_raw.parquet`（含展开后的 25 指标原值）。
- 检查点：记录数 51,230 / 17,523 / 203,752；A1 arxiv、github 子集 id 全部命中 A2/A3。

### Stage 1：25 指标展开与预处理（对应 R1，约 3 h）

**步骤 1.1 展开（22 → 25）**

| 字段 | 处理 | 输出指标 |
|---|---|---|
| `modernbert_{cleanliness,readability,reasoning,professionalism}` | \(\sigma=\mathrm{softmax}(\ell)\)，\(s=\sum_{k=0}^{5}k\sigma_k\) | 4 个，取值 [0,5] |
| `qurater` | 拆 4 维：writing_style、required_expertise、facts_trivia、educational_value（QuRating 原始顺序，须在论文注明"按数据源文档顺序"） | 4 个 |
| `ad_en`、`fluency_en` | log-odds \(\ell_1-\ell_0\)（label 1 = 无广告 / 流畅；实测各域中位数为正，与语义一致） | 2 个 |
| `fineweb_edu` | 取唯一元素 | 1 个 |
| 14 个标量字段 | 原值 | 14 个 |

**步骤 1.2 变换与方向（写成一张 25 行的"变换—方向"表放论文附录）**

| 指标组 | 变换 | 方向 |
|---|---|---|
| `dsir_books / wiki / math` | 先除以 `rps_doc_word_count`（每词 log 比），再 1%/99% 域内缩尾 | 正向 |
| `rps_doc_word_count`、`rps_doc_num_sentences` | \(\log(1+x)\)；非单调 → 取 \(-|\log x-\mathrm{med}_d\log x|\) 转成单峰 | 转换后正向 |
| `rps_doc_mean_word_length` | 同上单峰化（两端皆差） | 转换后正向 |
| `rps_doc_unigram_entropy`、`rps_doc_frac_unique_words`、`rps_lines_ending_with_terminal_punctution_mark` | 域内缩尾 | 正向 |
| `rps_doc_frac_no_alph_words`、`rps_doc_frac_chars_top_2gram / top_3gram`、`rps_lines_uppercase_letter_fraction`、`rps_lines_numerical_chars_fraction` | 域内缩尾 | **负向**，Min-Max 后取 \(1-x\) |
| 4 个 modernbert、4 个 qurater、fineweb_edu、ad、fluency | 无（log-odds 缩尾） | 正向 |

理由：题目强制"越高越好"；重尾指标不经 log 时 Min-Max 会把 99% 的样本压到极小区间；非单调指标按"距域内中位数偏离"处理是 RedPajama 规则型信号用于过滤时的实际语义（过短/过长皆差）。

**步骤 1.3 域内 Min-Max**：每个指标在其质量域内做 Min-Max（端点用域内 0.5%/99.5% 分位）；输出 \(x_{ij}\in[0,1]\)。**端点在 A1 上估计并冻结**，A2/A3 复用同一端点（否则对照结果混入管线差异）。

理由：DSIR 是相对源域的 log 比，规则型信号在 github/arxiv 有结构性基线差（§2.1），全局标准化会把域间差异误判为质量差。附录同时给出"全局 Min-Max"版本作敏感性对照。

检查点：25 指标全部与 `fineweb_edu` 的域内 Spearman 符号为非负（方向统一的直接证据）；缩尾前后分布图。

### Stage 2：质量评分 \(Q\)（对应 R2、R3，约 4 h）

**2.1 权重（主线 CRITIC）**
\[
C_j=\sigma_j\sum_{k\ne j}(1-r_{jk}),\qquad w_j=\frac{C_j}{\sum_k C_k},
\]
在 A1 的 25 维标准化矩阵上估计（\(r_{jk}\) 用 Spearman，抗非线性）。对照：等权；PCA 第一主成分载荷（须解释载荷，不得把方差贡献率当权重）；若团队能核到 Meta-rater 原文权重则再加一列。报告三种权重下样本级 \(Q\) 的 Spearman。

理由：CRITIC 同时奖励区分度、惩罚冗余（fluency 与 readability 高相关时自动降权），客观可复现；22 维人工定权不可靠。

**2.2 合成**：主线加权几何平均 \(Q(x)=\prod_j \max(x_j,\epsilon)^{w_j}\)，\(\epsilon=0.01\)；对照算术平均。理由：几何平均体现"短板支配"（高教育价值 + 高广告 → 低分），与 FineWeb-Edu / DCLM 硬过滤实践一致；\(\epsilon\) 保证 \(Q\in(0,1]\)，衔接问题三成本函数定义域。

**2.3 聚合**：域级报中位数、均值、P10/P90、bootstrap 95% 区间（book 171 条必须给区间）；另给按 `word_count` 加权均值（"按 token 抽样的期望质量"口径）。

**2.4 A2/A3 全量复算与对照（R3 硬要求）**
- 用 A1 上冻结的变换端点与 CRITIC 权重，对 A2（17,523）、A3（203,752）逐条计算 \(Q\)。
- 对照表：arxiv、github 两域 × {A1 子集, A2/A3 全量, A2/A3 去掉 A1 子集后的剩余部分} × {中位数, 均值, P10, P90, 区间}。
- 抽样代表性检验：25 指标逐个做 A1 子集 vs 全文件的 KS 检验 + Cohen's d；结论写成"区间抽样是否代表全文件"，而非"两组数据是否一致"。

**2.5 17 配方域的 \(Q_d\)（走 A18，对应 A16 的 11 个 inferred 域）**
1. 用纯 Python 在 A1 的 `content` 上复现 11 个 `rps_*` 规则型信号；与 A1 给定值比对（Spearman > 0.95 视为复现成功；差异记录为"实现口径差"）。
2. 在 A1 上拟合 \(Q\approx\phi(\text{11 个规则型信号})\)（HistGradientBoosting 或带域随机效应的线性模型），在 A2/A3 上验证外推误差；报告 \(\phi\) 的 R² 与残差区间。
3. 对 A18 的 17 域文本（长文本切 2,000 字符段）计算规则型信号 → \(\hat Q\) → 域级中位数与 bootstrap 区间。
4. 6 个 direct/near_direct 域同时有"直接 \(Q_d\)"与"经 \(\phi\) 的 \(\hat Q_d\)"，二者差异作为 inferred 域的**系统误差估计**并加宽区间。
5. 输出 P1-Q_domain：17 行 × {Q_d, CI_low, CI_high, source_level ∈ {direct, near_direct, inferred}, n_docs, n_segments}。

理由：A16 明确 11 域无质量信号；A18 是题目唯一给出的 17 域原文；只有规则型信号能无 GPU 复算，模型型信号（modernbert/qurater/fineweb_edu）无法复现，故用 \(\phi\) 校准而非直接等同。

检查点：7 质量域 \(Q_d\) 排序与直觉一致（预期 arxiv/wikipedia 高、c4/commoncrawl 中、github 低）并能用 A18 抽查解释；A2/A3 处理条数与文件条数一致。

### Stage 3：质量冲突消解（对应 R4，约 4 h）

**3.1 指标分组**（按来源）：G1 规则型 11 个（rps_*）；G2 DSIR 3 个；G3 WanjuanCC 类 6 个（4 modernbert + ad + fluency）；G4 PRRC/QuRating/FineWeb 5 个（4 qurater + fineweb_edu）。组内 Kendall W（预期组内 W 高、组间低，否则调整分组并说明）。组分 \(g_k(x)\) = 组内标准化分的均值。

**3.2 冲突定义**（禁止预设阈值）
- 样本级冲突度：\(c(x)=\max_k r_k(x)-\min_k r_k(x)\)，\(r_k\) 为组分在**域内**的分位排名。
- 阈值 \(\tau\)：画冲突率–\(\tau\) 曲线；\(\tau\) 取 \(c\) 的域内分布肘点或 POT（广义帕累托）形状参数稳定的起点；同时报告 \(\tau\in\{0.3,0.4,0.5,0.6\}\) 下结论是否变化。
- 结构性冲突：域 \(d\) 内，组对 \((k,l)\) 的符号 \(\mathrm{sign}(r_k-r_l)\) 显著偏向一侧（二项检验），记为该域的"系统性偏差"。

理由：《数据说明》夹带的"z 分数差 > 0.5，冲突率 ≈ 0%"是把结论藏进定义的循环论证；阈值须由数据决定并给出敏感性。

**3.3 成因分析**
- 域 × 组对交叉表 + 卡方检验；冲突率按域排序（预期 github > arxiv > stackexchange > 其他）。
- 机制解释（结合 A17 平均长度与规则型定义）：github 代码使 `frac_no_alph_words`、`uppercase` 偏高、`fluency` 偏低，但 `professionalism` 高（实测 github 中位数 4.43）；arxiv LaTeX 使 `terminal_punct` 低（33.6%）但模型型评分全部饱和为 5；book 域长文本使 DSIR 未归一时极端。
- 高冲突样本的指标对联合分布图：fineweb_edu 高 × ad 低（软广科普）、reasoning 高 × fluency 低（代码/公式）。

**3.4 消解规则**（降权而非删除）
- 规则 A（样本级 Huber 降权）：对 \(|z_j|>\tau_z\) 的指标 \(w_j\leftarrow w_j\min(1,\tau_z/|z_j|)\)，重新归一后合成 \(\tilde Q\)。
- 规则 B（域级结构性降权）：对存在结构性冲突的（域, 组）对，该组权重乘 \(1-\rho_{d,k}\)，\(\rho\) 为结构性偏向强度。
- 输出 \(\tilde Q\) 与 \(Q\) 的对照：变化最大的样本 top 50 送 A18/A1 原文核验。
- 备选（简述）：因子分析 + 调和平均；D-S 证据理论。

**3.5 扩展集复验（R4 硬要求）**：同一分组、同一 \(\tau\)、同一规则在 A2/A3 上重算：冲突率、top 冲突组对、结构性冲突方向是否复现；给出"A1 子集 vs 全文件"冲突率差异（预期区间抽样稀释极端样本，全量冲突率更高）。

检查点：冲突率–\(\tau\) 曲线在合理区间内结论稳定；消解后 \(\tilde Q\) 与 \(Q\) 的 Spearman 报告；核验样本一致率 ≥ 80%。

### Stage 4：领域配比建模（对应 R5–R7，约 6 h）

**4.1 数据与预处理**
- A4/A5 按 `index` 合并，行归一闭合。目标取 \(\ln L_v\)（13 个验证域各建一个模型）。
- 特征三轨：(i) 原始 \(p\)（RegMix 复现基线）；(ii) \(\ln(p_i+\varepsilon)\)，\(\varepsilon=10^{-3}\)（配比精度）；(iii) clr 于乘性零值替换（\(\delta=5\times10^{-4}\)）之后。零值替换敏感性：\(\delta\in\{10^{-4},5\times10^{-4},10^{-3}\}\)。
- 17 维全部保留（含 4 个无 Loss 域）；敏感性：把 4 域合并为"other"。

**4.2 模型**
- 主线：Ridge / Huber 线性回归（\(\alpha\) 由 CV 定），可解释；对照：HistGradientBoostingRegressor（捕捉交互）。
- 所有模型用 A4/A5 训练，A6–A11 只做评估，绝不参与拟合。

**4.3 边际效应矩阵 \(T\)（17×13）**
\(T_{iv}=\partial \ln L_v/\partial p_i\)，在基线配比（RegMix 人工配比与 \(p^*\)）处数值求导（对数变换下解析式 \(\beta_{v,i}/(p_i+\varepsilon)\)）。报告：自域效应对角、跨域迁移（如 pile_cc 对多数域的负 Loss 贡献）、4 个无 Loss 域的外溢；热力图。

**4.4 引入 \(Q\) 的嵌套检验**
- 构造 \(\bar Q(p)=\sum_i p_iQ_{d(i)}\)（用 P1-Q_domain；inferred 域取区间上下界做两次）。
- 模型 M0：仅配比；M1：配比 + \(\lambda(1-\bar Q(p))\)；M2：配比 + \(\sum_i \beta_i p_i Q_{d(i)}\)。
- 以检验集 Spearman/RMSE 与 bootstrap 下 \(\lambda\) 的区间判断质量项是否有独立解释力；注意 \(\bar Q(p)\) 是 \(p\) 的线性函数，与 clr 特征共线，须报 VIF 并用 Ridge。
- 结论写法：若 \(\lambda\) 显著 → 为问题二"质量作有效数据乘子"提供附件内证据；若不显著 → 说明 17 域 \(Q_d\) 的映射误差与共线性使附件 A 无法单独识别质量效应，问题二改由 B6–B8 识别。两种情形都要写。

**4.5 最优配比 \(p^*\)**
- 目标：13 域 \(\ln L\) 的均值（等权）与 RegMix 口径（pile_cc 域）各算一次。
- 单纯形内 Dirichlet 采样 \(10^5\)–\(10^6\) 组（浓度参数按 A4 拟合）→ 模型预测 → top-128 平均得 \(p^*\)；再用 SLSQP 在单纯形约束下局部精化。
- 输出 \(p^*\)、\(\bar Q(p^*)\)、基线配比的 \(\bar Q\)，以及 bootstrap（重采样训练集重拟合）下 \(p^*\) 各分量的区间。

**4.6 三尺度验证与排名衰减（R7）**
- 每域、每尺度报告：Spearman、Kendall τ、原始 R²、**尺度校准后 R²**（对预测值做一次仿射校准 \(\hat y\to a_N+b_N\hat y\)，\(a_N,b_N\) 在该尺度上用一半样本估、另一半评估）、RMSE。
- 模型无关的排名不变性：test 1M 与 test 60M 同配比 Loss 直接 Spearman（实测 ≥ 0.98）作为上界参照。
- 1B（64 组，样本少）用 bootstrap 给 Spearman 区间。
- P1-rank_decay：横轴 \(\log_{10}N\in\{6, 7.8, 9\}\)，纵轴各域 Spearman，附 est 10B/70B 两点（虚线、标注"外推"）。

**4.7 外推表 A12–A15 的稳健性讨论**
- 明确标注：est Loss 由出题方按 1M/60M/1B 幂律外推生成，非观测；est_mixture 是 train 的 63 行子集。
- 只做三件事：(a) \(p^*\) 在 est 10B/70B 预测口径下的排名是否仍在前列；(b) 各域"观测 1M Loss vs est Loss"的 Spearman（实测 pile_cc 0.10、hackernews 0.02），与"观测 1M vs 60M"（0.99）对比，说明 est 表隐含的排名重排缺乏观测支持；(c) 用 1M→60M→1B 拟合系数随 \(N\) 的漂移 \(\beta_{v,i}(N)\)，外推到 10B/70B 与 est 对比，给出"哪些域结论跨尺度稳、哪些不稳"的排序。
- 结论措辞：稳健性结论以真实三尺度为主，est 表仅作方向性参考。

检查点：主线模型 13 域平均 Spearman 1M ≥ 0.95、60M ≥ 0.95、1B ≥ 0.90（基线已达 0.96/0.96/0.92）；GBM 对照 ≥ 线性；\(T\) 矩阵符号与 RegMix 原文定性结论（pile_cc 关联最强）一致。

### Stage 5：原文核验（对应 R8，约 1.5 h，可与 Stage 4 并行）

- 从 A1 抽：\(Q\) 最高 20、最低 20、消解前后变化最大 20、随机 20；从 A18 各 inferred 域抽 5 条。
- 三人独立打 1–5 分，算与 \(Q\) 的 Spearman 及一致率；记录典型误判（如 LaTeX 源码被规则型信号低估）。

### Stage 6：产出落盘与接口文档（约 1 h）

- `outputs/q1/P1_Q_sample.parquet`（id, source_set, domain, 25 维标准化分, Q, Q_tilde, conflict_score, is_conflict）
- `outputs/q1/P1_Q_domain.csv`（17 + 7 行）
- `outputs/q1/P1_f_p.json`（变换类型、\(\varepsilon/\delta\)、13 组系数、\(T\) 矩阵）
- `outputs/q1/P1_p_star.csv`、`outputs/q1/P1_rank_decay.csv`
- `outputs/q1/README.md`：字段说明 + 问题二/三/四如何读取。

---

## 4. 图表清单（标题自带结论）

| 编号 | 图/表 | 预期标题句式 |
|---|---|---|
| 图 1 | 问题一数据流图（A1–A3/A18 → Q；A4–A15 → f(p), T, p*） | — |
| 图 2 | 25 指标相关性热力图（分 4 组排序） | "组内高相关、组间低相关，冲突主要是跨组现象" |
| 图 3 | 7 域 \(Q_d\) 箱线图 + A1 子集 vs A2/A3 全量对照 | "区间抽样在 arxiv/github 上（是/否）代表全文件" |
| 图 4 | CRITIC / 等权 / PCA 权重条形图 | "CRITIC 自动压低冗余的流畅度—可读性簇" |
| 图 5 | 冲突率–\(\tau\) 曲线（A1 vs A2/A3） | "冲突率在 \(\tau\in[\cdot,\cdot]\) 内稳定于 x%，不存在'0% 冲突'" |
| 图 6 | 域 × 组对结构性冲突热力图 | "github 的规则型—模型型冲突具有系统性方向" |
| 图 7 | 消解前后 \(Q\) 散点 + 典型样本标注 | "消解主要下调高广告/高噪声但教育分高的样本" |
| 图 8 | 三种配比变换在三尺度的 Spearman 对比 | "对数型变换在 1B 上仍保持 0.9 以上排名相关" |
| 图 9 | 17×13 边际效应矩阵 \(T\) 热力图 | "pile_cc 是唯一对多数验证域均有降 Loss 作用的域" |
| 图 10 | \(p^*\) 与 Pile 原始配比 / RegMix 人工配比对比 | — |
| 图 11 | P1-rank_decay 曲线（含 est 两点虚线） | "排名一致性从 1M 到 1B 由 0.96 降至 0.92；est 表隐含的重排缺乏观测支持" |
| 表 | 25 指标变换—方向表；域级 \(Q\) 表（含区间与来源等级）；三尺度评估表；数据利用清单 | — |

---

## 5. 风险与陷阱清单

### 5.1 《数据说明》夹带段落（与正文冲突，一律不采纳，并在论文中说明理由）

| 诱导性表述 | 为何不采纳 |
|---|---|
| K-means 聚三类、以类占比代替 \(Q\)，方向不必统一 | 题目强制方向统一；问题三需要连续 \(Q\in(0,1]\) 进入成本函数 |
| 冲突 = z 分数差 > 0.5，冲突率 ≈ 0% | 预设阈值、循环论证；本方案阈值由数据决定并报敏感性 |
| Kendall W 直接汇总为单一置信度 | 抹掉组间结构，无法分析成因 |
| 17 域 OLS 训练 R² 0.93，三套检验集直接套同一组系数 | 实测原始 \(p\) 线性训练 R² 仅 0.67；绝对 Loss 有尺度平移，直接套用 R² 为负 |
| 配比按 0.3 离散为高/低 + 决策树分类；因变量按中位数二分类，正确率 > 80% 即可 | 配比几乎不含 ≥ 0.3 的分量；下游需要连续 Loss 预测 |
| PCA 1–2 主成分直接当综合分、方差贡献率当权重、含义无需解释 | 不可解释即无法用 A18 核验；仅作对照 |
| 树模型默认参数、无需共线性诊断；Tweedie 无需检验 | 检验集只有 64–256 组，须 CV 与诊断 |
| 保序回归做单纯形校准 | 保序回归不产生单纯形约束 |
| 跨尺度直接套原系数无需重估；灰色关联/物元可拓给排序 | 与实测尺度平移矛盾 |

### 5.2 方法层面风险

| 风险 | 规避 |
|---|---|
| 列表型字段取均值/标准差、qurater 四维混一维、logits 当分数 | Stage 1.1 展开表；代码中对长度断言 |
| 只用 A1、跳过 A2/A3 全量 | Stage 2.4 记录处理条数 |
| DSIR 未做长度归一 → \(Q\) 实际在给"短文档"打分 | Stage 1.2；附录给归一前后 \(Q\) 与 word_count 的相关 |
| 概率饱和导致 ad/fluency 权重失真 | log-odds 进管线 |
| 配比零值使 clr 失效 | 零值替换 + \(\ln(p+\varepsilon)\) 双轨 + \(\delta\) 敏感性 |
| \(\bar Q(p)\) 与配比特征共线，\(\lambda\) 不可识别 | VIF、Ridge、bootstrap 区间；两种结论写法预备 |
| 把 est 表当验证 | 只报排名稳定性，标注"外推、非观测" |
| A18 小域（ubuntu_irc 16 条）\(Q_d\) 不可信 | 切段 + bootstrap 区间 + 标注 inferred，问题二/三使用时按区间做敏感性 |
| A2/A3 无原文，无法核验 | 核验限定在 A1 与 A18，论文如实说明 |
| 宣称复现 Meta-rater 代理训练 | 明确说明无 GPU，只复现规则型信号与权重合成 |

---

## 6. 时间分配（对应总路线 Day 1，0–24 h，三人）

| 时段 | 建模 A | 编程 B | 写作 C |
|---|---|---|---|
| 0–1 h | 确认本计划、分配 | Stage 0 骨架，验证读取 | 问题一重述、SQ1–SQ3、假设与符号 |
| 1–4 h | 定 25 指标变换—方向表、分组方案 | Stage 1 展开与预处理，输出分布图 | 数据说明章节（A1–A18 用途表、列表展开公式） |
| 4–8 h | CRITIC/合成/聚合口径；A18 校准方案 | Stage 2（含 A2/A3 全量、KS）；Stage 2.5 规则型信号复现 | 质量评价模型章节初稿 |
| 8–12 h | 冲突阈值判定、消解规则 | Stage 3 冲突分析与复验 | 冲突消解章节初稿 |
| 12–18 h | 配比模型选型、\(T\) 解读、\(Q\) 嵌套结论 | Stage 4 全流程（三轨变换、GBM、\(p^*\)、三尺度、est） | 配比建模章节初稿 |
| 18–21 h | 原文核验（三人各 30 条） | Stage 5 抽样与记录、Stage 6 落盘 | 检验章节、图表标题 |
| 21–24 h | 复核五份产出与问题二/三接口 | 出图定稿 | 结果解读、风险清单、数据利用清单 |

原则：先跑通最小闭环（等权 \(Q\) + 原始 \(p\) 线性），再逐步替换为主线方法；任何阶段卡住不得超过 2 h，超时用备选方案先落盘。

---

## 7. 与《提示词_问题一_v2》十段结构的对应

| 提示词要求的输出段 | 本计划对应内容 |
|---|---|
| ① 问题重述与分析（SQ1–SQ3） | §1 |
| ② 符号与假设 | Stage 1–4 各公式；假设：域内可比、A1 抽样代表性待检、est 非观测 |
| ③ 数据与预处理（A1–A18 用途表、展开公式、方向/变换/域内标准化） | §2、Stage 0–1 |
| ④ 模型（质量评价 / 冲突 / 配比） | Stage 2、3、4 |
| ⑤ 算法伪代码（标注流式读取） | Stage 0 读取；Stage 2.4、2.5、4.5 流程 |
| ⑥ 检验计划（KS、三尺度 Spearman、外推口径） | Stage 2.4、4.6、4.7 |
| ⑦ 图表清单 | §4 |
| ⑧ 结果解读模板 | 各 Stage"检查点"与图表标题句式 |
| ⑨ 与 Q2–Q4 接口 | §0.2、Stage 6 |
| ⑩ 风险清单 | §5 |

---

## 附：本次核查所用的临时脚本要点（不入库，供复现）

- A1/A2/A3：`lzma.open(..., 'rt')` 逐行解析；统计域计数、列表长度、id 交集、DSIR 与 word_count 的 Spearman、softmax 期望分与 argmax 分布。
- A4–A15：行和、零值比例、最小正值；test 1M 与 60M 配比矩阵 `allclose`；est_mixture 与 train 按 index 对齐比对；同配比 Loss 的跨尺度 Spearman。
- 基线：RidgeCV，目标 \(\ln L\)，三种特征变换，13 域 × 三尺度 Spearman/R²。
