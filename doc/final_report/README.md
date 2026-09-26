# 完整论文状态

本目录集中存放完整论文的 LaTeX 源文件、图、格式模板及已有 PDF。Q1–Q3 已按现行 Spec 完成同步并写入论文；Q4 仍是以 2025.25 为锚点的历史情景预测，尚未按当前日期重跑，故 PDF 暂不作为本轮最终提交版。

## LaTeX 文件索引

`main.tex` 是编译入口，按摘要、正文第 1–12 节、参考文献、附录 A 的顺序逐个 `\input`。每个正文一级章节独占一个文件：

| 部分 | 标题 | 文件 |
|---|---|---|
| 摘要 | 摘要 | `abstract.tex` |
| 第 1 节 | 问题重述 | `sec01_problem_restatement.tex` |
| 第 2 节 | 问题分析 | `sec02_problem_analysis.tex` |
| 第 3 节 | 模型假设 | `sec03_model_assumptions.tex` |
| 第 4 节 | 符号说明 | `sec04_notation.tex` |
| 第 5 节 | 数据来源与接口约定 | `sec05_data_sources_interfaces.tex` |
| 第 6 节 | 问题一：数据质量与领域配比 | `sec06_q1.tex` |
| 第 7 节 | 问题二：含质量与配比的广义标度律 | `sec07_q2.tex` |
| 第 8 节 | 问题三：算力约束下的资源配置优化 | `sec08_q3.tex` |
| 第 9 节 | 问题四：Benchmark 检验与能力演进 | `sec09_q4.tex` |
| 第 10 节 | 模型的检验与灵敏度分析 | `sec10_validation_sensitivity.tex` |
| 第 11 节 | 模型的评价与推广 | `sec11_evaluation_extension.tex` |
| 第 12 节 | 结论 | `sec12_conclusion.tex` |
| 参考文献 | 参考文献 | `references.tex` |
| 附录 A | 补充材料与程序 | `appendix.tex` |

运行 `./compile.sh` 会执行两遍 XeLaTeX；带时间戳的 PDF 存入 `pdf/`。结构拆分后的最近一次单遍编译产物为 `pdf/main_0926-2030.pdf`（45 页 A4）。

已同步：摘要为四问递进表述；Q1 三项任务、Q2 幂次主式与 B6 训练/B7 新增留出、Q3 幂次主式（地板＋配比乘子 $m_*$）的三档预算最优配置、成本份额、临界预算均已按落盘接口写入正文；四问检验表、结论与附录代码节选同步更正；缺失的参考文献条目（shafer、subramanyam、softq、lovelace）已补齐。

仍未关闭：Q4 预测起点 $T_0=2025.25$ 为历史锚点（数据截止 2025-03-13），需先定口径再重跑或重述；Q3 的 P1 逐次重抽样接口缺失，全链条不确定区间未闭合；`P3_run_manifest.json` 的 P1 输入哈希与当前文件有字节级漂移（$Q_0$、pile_cc 数值未变，建议重跑刷新 manifest 或确认无碍）。

修订顺序：先关闭 [一致性核查](../项目整理与Spec同步核查.md) 与四问 Spec 中的剩余阻断项（尤其 Q4 预测口径）；按当前接口重跑/复核受影响数字；检查 `\input`/`\includegraphics` 路径；重编译并逐页核查 PDF。历史修订记录见 `报告修订记录.md`。
