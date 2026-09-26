# 完整论文状态

本目录集中存放完整论文的 LaTeX 源文件、图、格式模板及已有 PDF。

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

运行 `./compile.sh` 会执行两遍 XeLaTeX；带时间戳的 PDF 存入 `pdf/`。修改单个问题章节后，用 `./compile_chapter.sh q1|q2|q3|q4` 做单章编译：脚本复用 `main.tex` 导言区、引入对应章节及 `references.tex`，产物存入 `pdf/qN/`；若仍有未解析的引用，脚本会保留日志并报错。单章不含其他正文，跨章内容仍需整本编译核对。结构拆分后的最近一次单遍编译产物为 `pdf/main_0926-2030.pdf`（45 页 A4），最近一次完整编译为 `pdf/main_0926-2049.pdf`（46 页 A4）。
