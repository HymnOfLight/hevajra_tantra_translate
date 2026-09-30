# hevajra_tantra_translate — 《喜金刚本续》多见证偏移矩阵

以梵文《喜金刚本续》（Hevajratantra）为参照，把它在梵、藏、汉、西夏、蒙等语言中的十余种见证放进同一张矩阵，用与动机无关、可机械复算的偏移向量度量每个见证相对梵文金标准的"漂移"，并把漂移分解为"可由已知梵文异读解释"、"与独立同源译本共享"、"残余"三部分。宋代汉译 T0892 只是矩阵中的一列。

## 文档

| 文件 | 内容 |
|---|---|
| [`docs/01_研究设计与独创性.md`](docs/01_研究设计与独创性.md) | 研究定位；与德重弘志 2026 及其待刊《喜金刚》研究的五层区分；对《评估》十项建议的逐条处置；研究问题与可证伪结论；工作包与关口；许可 |
| [`docs/02_矩阵与偏移度量.md`](docs/02_矩阵与偏移度量.md) | 三层矩阵的形式定义；"金标准是分布"；单元格记录；偏移向量 \(d_{cov}, d_{len}, d_{lit}, d_{ord}, d_{split}, \mathbf d_{comp}\)；三分分解；统计报告规范；证据门与弃判；LLM 过度归因实验 |
| [`docs/03_远古文本处理方法论.md`](docs/03_远古文本处理方法论.md) | 从本项目提炼的"见证矩阵法"：十条原则、八步流水线、最小度量集合、谬误清单、人机分工、复现包规范、迁移指南 |
| [`data/witnesses.yaml`](data/witnesses.yaml) | 见证清单（矩阵的列）：梵文校勘本与写本、德格本及其修订层、宋译、西夏时期与明抄本汉译、西夏文、蒙文、注疏引文、现代译本（仅桥梁） |
| [`data/concordance/chapters.yaml`](data/concordance/chapters.yaml) | 章级同本系概表（G0 关口输入），含 L1 篇幅比证据与待核事项 |
| [`data/anchors/terms.yaml`](data/anchors/terms.yaml) | 梵–藏–汉锚点词表（B0 对齐基线的唯一跨语信号） |

## 代码

纯 Python（3.10+），唯一依赖 PyYAML。

```bash
pip install -e ".[dev]"
python -m hevajra_matrix fetch --raw data/raw          # 下载 CBETA T0892 XML 与德格本 rgyud 'bum nga 卷（记录 SHA-256）
python -m hevajra_matrix run   --raw data/raw --out out # 切分 → 章级门控 → 对齐（含 NULL）→ 矩阵 → 统计 → 报告
pytest
```

模块一览：

| 模块 | 职责 |
|---|---|
| `ingest/cbeta.py` | CBETA TEI P5 → 带大正藏页栏行坐标的子句级片段；保留校勘记与大正藏梵文脚注（现成锚点） |
| `ingest/derge.py` | Esukhia 德格本文本 → 按 Toh 切片、shad 分段、叶.行坐标、品尾题记识别、韵散/真言分类、前置题名与礼敬语标为 meta |
| `ingest/sanskrit.py` | 研究者自备的梵文 TSV（偈号 + 文本 + LACUNA/ABSENT 标记） |
| `normalize.py` | 三种语言的长度归一化、数词解析、音译识别 |
| `anchors.py` | 跨语锚点抽取与相似度 |
| `align.py` | 章级门控 + Gale–Church 长度 + 锚点 + 允许 1:0 / 0:1 的动态规划；相似度后端可插拔（嵌入模型可替换） |
| `matrix.py` | 行（参照单元）× 列（见证）× 单元格（状态 + 偏移向量）；L1/L2 导出 |
| `metrics.py` | 三分分解、品区组自助法、匹配对照、见证距离与 UPGMA |
| `attribution.py` | 证据门 → 无动机标签的归因分布 → 弃判；LLM 过度归因反事实实验框架 |
| `report.py` | SVG 热图、状态条、Markdown 摘要 |
| `pipeline.py` / `cli.py` | 端到端运行；合并品（如汉第 11 品 = I.11 + II.1）按概表自动处理 |

## 两种运行模式

- **gold**：`data/reference/` 下存在梵文 TSV 时，梵文为参照，藏译与汉译都被测量，三分分解完整。
- **provisional**：无梵文文本时以德格本为临时参照，所有输出打 `reference_grade=provisional`，只描述汉译相对藏译的关系。当前仓库状态下运行即为此模式。

## 当前状态（v0.1）

- 流水线在真实 T0892 与 Toh 417–418 上端到端运行（约 10 秒）：3,045 个临时参照单元、20 个汉文品全部映射、3 个品（II.1、II.12 并入相邻汉品；II.9 待判）按概表处理。
- L1 篇幅比已给出三条待核的结构假说（见 `chapters.yaml`）：汉第 11 品含 II.1；汉第 20 品含 II.12；II.9（抽字法品）在汉译中缺失或压缩入第 18 品。
- B0 对齐是长度主导的基线，NULL 检测参数（`prior_null`、`anchor_weight`）需在试标金标准上重调并预注册；`out/alignment_review.csv` 为人工修正表。
- 组件层（L3）标注、多列梵文、其他甘珠尔版本、西夏/明汉译尚未接入；接口已按 `witnesses.yaml` 预留。

## 发布边界

仓库只含坐标、标签、数值、代码与合理长度的短引文。CBETA（CC BY-NC-SA）、德格本电子文本（许可待核）、梵文校勘本（版权）、84000（CC BY-NC-ND）的全文均不入库；`data/raw/` 与 `data/reference/*.tsv` 已被 `.gitignore` 排除。
