# 附件 A 数据集详解（基于实际文件查验）

> 对 A1–A18 逐一解释含义。标注【实查】的内容来自对文件的实际读取（2026-09-23），比《数据说明》的概述更细。
> 附件 A 分两条子线：**质量信号线**（A1–A3 + A16–A18，回答"数据好不好"）和**配方实验线**（A4–A15 + A17，回答"怎么搭配数据"）。

---

## 第一条子线：质量信号（A1–A3）

**大白话**：有人（SlimPajama-Meta-rater 项目）用 22 把不同的"尺子"给互联网文本量了身材——有的量"像不像教材"（fineweb_edu），有的量"广告多不多"（ad_en），有的量"读起来顺不顺"（fluency_en）。我们要做的是把这 22 把尺子的读数合成一个质量分 $Q$。

### A1 `slimpajama_quality_signal_sample.jsonl.xz` —— 质量信号抽样集

**是什么**：从 SlimPajama 语料（一个开源的大型预训练语料库）上，按 7 个域**分层抽样**的 51,230 篇文档，每篇附带 22 个质量指标的读数。来源是 OpenDataLab 发布的 SlimPajama-Meta-rater 数据集。

**7 个质量域**：arxiv（论文）、book（书籍）、c4（谷歌公开的网页语料）、commoncrawl（通用网页爬取）、github（代码）、stackexchange（问答社区）、wikipedia（百科）。由 `_source_domain` 字段标识。

**字段结构（27 个）**【实查】：

- 5 个辅助字段：`id`（文档编号）、`content`（**原始正文**，抽查用，不必入库）、`sub_path`、`_source_domain`（域）、`_source_path`（来源文件）；
- **14 个标量指标**：
  - `dsir_books` / `dsir_wiki` / `dsir_math`：**DSIR 重要性权重**（Xie et al. 2023 方法），衡量文档与书/百科/数学分布的接近程度。【实查】取值为**大号负数**（如 −1961、−1629），应为负对数空间权重——**越大（越不负）越好**，预处理时须统一尺度；
  - `rps_doc_word_count`（词数）、`rps_doc_num_sentences`（句数）、`rps_doc_mean_word_length`（平均词长）、`rps_doc_unigram_entropy`（一元词熵）；
  - `rps_doc_frac_no_alph_words`（非字母词占比）、`rps_doc_frac_unique_words`（词表丰富度）、`rps_doc_frac_chars_top_2gram` / `top_3gram`（最高频 2/3-gram 字符占比，**模板化程度**）；
  - `rps_lines_uppercase_letter_fraction`（大写行占比）、`rps_lines_ending_with_terminal_punctution_mark`（以句末标点结尾的行数）、`rps_lines_numerical_chars_fraction`（数字字符占比）。【实查】注意：部分 rps 字段是**百分数**（如 28.96、12.5），不是 0–1 小数，各字段量纲差异大，必须归一化；
- **8 个列表型指标**：`fineweb_edu`（1 个数）、`fluency_en`（2 个数）、`ad_en`（2 个数）、`qurater`（4 个数）、`modernbert_cleanliness / readability / reasoning / professionalism`（各 6 个数）。【实查】列表长度在域内固定，可视为该指标的**多维子分**（如 modernbert 的 6 维可能对应 6 个评测侧面），压缩为标量时先搞清每维含义再取中位数/均值。

**性质**：真实数据；问题一质量主样本，**必用**。

**例子**【实查】：一条 arxiv 文档，content 是 MoSe₂/WS₂ 的 DFT 计算小节（LaTeX 源码），fineweb_edu 子分 1.33、fluency_en 子分 [−0.40, 0.45]、dsir_math −1629（比对 book/wiki 的 −1961/−1934 更不负 → 数学相关度最高）。这类"指标方向不一、量纲混杂"正是问题一预处理要解决的。

### A2 / A3 `slimpajama_quality_extended/` —— arxiv 与 github 域的**全量**质量信号

**是什么**：同一套 22 指标，但不再是抽样，而是 arxiv 域全部 17,523 篇（A2）和 github 域全部 203,752 篇（A3）。每条 24 个字段（比 A1 少了 content、_source_domain、_source_path——域由文件名推断）。

**为什么要给两个域的全量**：题目要求"使用全量质量信号"，并且要做**抽样集 vs 全量集的对照**——抽样集按域等比例抽取，会稀释极端样本；arxiv（论文，相对干净）与 github（代码托管，天然噪声大）又是性格差异最大的两个域，适合做冲突分析的对照组。

**性质**：真实；**必用**。

**怎么用**：与 A1 **完全同一套**预处理与权重算域级 $\bar Q$，做三集对照表；同一管线算冲突率，检验"抽样集的冲突结论在全量集是否成立"。

### A16 `domain_mapping_guide.csv` —— 两套域体系的"翻译表"

**是什么**：质量信号只有 7 个域，配方实验有 17 个域，两边对不上。这张表是人工整理的翻译表，17 行，每行给出：配方域名 → 质量域名 → 映射类型（direct 同名直接对应 / near_direct 语义近似 / inferred 无对应、需自行推断）+ 备注。

【实查】实际的映射结构：

- **direct（3 个）**：arxiv→arxiv、github→github、stackexchange→stackexchange；
- **near_direct（3 个）**：wikipedia_en→wikipedia、gutenberg_pg_19→book、pile_cc→commoncrawl；
- **inferred（11 个）**：dm_mathematics、freelaw、nih_exporter、pubmed_central、philpapers、enron_emails、ubuntu_irc、europarl、hackernews、pubmed_abstracts、uspto_backgrounds——**全部无质量域可挂**。

**含义的深意**：17 个配方域里只有 6 个能拿到"官方"质量分，这是题目"若需将质量信息引入配比建模，应自行建立跨体系关联并说明规则与假设"的原因。方案处理：direct/near_direct 直接用；inferred 的 11 域需自建规则（如用 A18 原始文本算启发式质量特征，或按主题挂到近似质量域），并在论文中分层给出结论的上下界。

### A17 `regmix_domain_summary.csv` —— 配方 17 域的文本抽样摘要

**是什么**：17 行 × 5 列：每个配方域抽 80 MB 文本后的统计——抽样行数、平均文本长度（字符）等。【实查】例如 gutenberg 平均 38 万字符/篇（整本书）、ubuntu_irc 平均 22.5 万字符（聊天日志）、pubmed_abstracts 平均 1330 字符（摘要）。

**用途**：背景描述与辅助分析（如 token 加权域级 $Q$ 时的权重参考：平均长度差异 300 倍，说明"按文档等权"和"按 token 加权"会给出很不同的域级分）。本版未接入求解。

### A18 `regmix_domain_sample.jsonl.xz` —— 配方域的原始文本样例

**是什么**：138,034 篇 RegMix 各域（The Pile 17 域体系）的原始文本，字段为 text + _source_domain，约 0.16 GB 压缩。**注意它的域是 17 个配方域**，不是质量信号的 7 域——两边体系不同，抽查时要说明用的是哪套口径。

**用途（可选但建议做）**：① 检验评分可靠性——从高分/低分样本各抽 ~20 条人工判定；② 确定模糊指标方向（如 unigram_entropy 过高的样本到底是不是噪声文本）。

---

## 第二条子线：配方实验（A4–A15）

**大白话**：RegMix 项目（ICML 2024，Sail SG / Sea AI Lab）拿 1M 参数的小模型做了 512 次训练实验：每次换一种"17 科课时表"（各领域数据占比），训完测 13 个"科目"的验证 Loss。这给了我们"配比 → Loss"的配对数据，可以回归出"哪个领域加一分，哪科成绩涨多少"。

### A4 `train_mixture_1m.csv` —— 训练配比表

**是什么**：512 行 × 18 列。每行一个实验编号 `index` + 17 列 `train_the_pile_*` 配比（0–1，每行和 ≈1，千分位舍入误差）。

【实查】17 个域：arxiv、freelaw、nih_exporter、pubmed_central、wikipedia_en、dm_mathematics、github、philpapers、stackexchange、enron_emails、gutenberg_pg_19、pile_cc、ubuntu_irc、europarl、hackernews、pubmed_abstracts、uspto_backgrounds。

【实查例子】index=1：pile_cc 0.787 + gutenberg 0.209 + philpapers 0.004（接近"纯网页+书"的配方）；index=2：github 0.304 + pile_cc 0.299 + wikipedia 0.047 等 17 域全有非零占比（均衡配方）。512 组覆盖了从"单域为主"到"全域混合"的配比空间。

### A5 `train_pile_loss_1m.csv` —— 训练 Loss 表

**是什么**：512 行 × 14 列。`index` 对应 A4 + **13 个验证域的交叉熵损失** `metric/the_pile_*_val_loss`。

【实查】13 个验证域：arxiv、freelaw、pubmed_central、wikipedia_en、dm_mathematics、github、stackexchange、gutenberg_pg_19、pile_cc、ubuntu_irc、hackernews、（表头截断，另有 2 个，共 13）。

**一个关键事实**：17 个训练域里有 4 个（nih_exporter、enron_emails、europarl、philpapers）**没有对应 Loss 列**——即"只能调配、无法单独考核"的科目。论文须说明处理方式（方案：保留为解释变量 + 合并敏感性分析）。

**因变量理解**：这是 1M 参数模型在 Pile 各子域验证集上的交叉熵，量级约 4–7（【实查】index=2 的 arxiv_loss=4.74，dm_math=4.61）。注意配比实验里"训练配比"与"验证域"是两套角色：用某些域的数据训练，在所有域上考试——所以模型能告诉我们"练什么对考什么有用"。

### A6/A7、A8/A9、A10/A11 —— 三套检验集（1M / 60M / 1B）

**是什么**：与 A4/A5 完全同构的**留出检验集**：1M 规模 256 组、60M 规模 256 组、1B 规模 64 组（配比表 + 对应 Loss 表各一个文件，按 index 配对）。

**含义的深意**：它们回答两个问题——① 配比模型泛化到没见过的配方好不好（同尺度检验）；② 配比效应是否随模型规模变化（1M→60M→1B 跨 3 个数量级）。后者直接支撑 A12–A15 的外推修正。

### A12/A13、A14/A15 —— 外推表（10B / 70B）

**是什么**：10B 和 70B 两个更大尺度上，63 组配比（**取自 train 配比的子集**，非独立新实验）+ 对应的**预估 Loss**。注意 A13/A15 的 Loss 是从 1M/60M/1B 三尺度幂律外推**生成的**，不是观测值。

**为什么要给"假答案"**：它让队伍能量化"小尺度拟合的系数直接推大尺度"有多大偏差——这是"讨论外推结论的稳健性"的素材，而不是可以直接抄的参考答案。论文必须标注其"外推/非观测"性质。

### A17（已在上方）属于这条线的统计背景。

---

## 两条子线怎么接上（回顾）

```
质量线：A1/A2/A3 → 22 指标合成 → 域级 Q̄（7 质量域）
                              ↓ A16 翻译（仅 6/17 配方域有官方映射，11 个要自建）
配方线：A4+A5 训练 → clr 回归 → L(p) 模型 → A6–A11 检验 → A12–A15 外推
                              ↑ 加入质量交互项 λ·Σpᵢ(1−Q̄)（题目让"论证是否引入"）
```

输出三个成果交给后面三问：域级质量分 $\bar Q$（→问题二的质量乘子、问题三的 $Q_0$）、最优配比 $\mathbf p^*$ 与配比效应量（→问题二的 $h(Q,\mathbf p)$、问题三的配比设定）、跨尺度外推证据（→问题二 B9/B10 百亿外推的先验）。

---

## 实查发现的三个预处理要点（写代码前注意）

1. **量纲混乱**：dsir_* 是千位级负数（负对数权重），rps 部分字段是百分数（0–100），modernbert/qurater 子分是小数——归一化前**不要假设任何字段的量纲**，逐字段检查分布；
2. **列表字段的维度语义**：8 个列表长度固定（1/2/4/6），压缩为标量前先确认每维含义（可对照各指标原论文：QuRating 是 4 维质量问题、modernbert 相关分数通常对应多个侧面），取中位数是默认稳健选择；
3. **域字段不对称**：A1 有 `_source_domain` 字段，A2/A3 要靠文件名推断域——合并三集时先统一加域标签列，否则对照表会出错。
