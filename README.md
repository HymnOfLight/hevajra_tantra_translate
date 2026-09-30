# hevajra_tantra_translate — 《喜金刚本续》多见证偏移矩阵

以梵文《喜金刚本续》（Hevajratantra）为参照，把它在梵、藏、汉、西夏、蒙等语言中的十余种见证放进同一张矩阵，用与动机无关、可机械复算的偏移向量度量每个见证相对梵文金标准的"漂移"，并把漂移分解为"可由已知梵文异读解释"、"与独立同源译本共享"、"残余"三部分。宋代汉译 T0892 只是矩阵中的一列。

## 文档

| 文件 | 内容 |
|---|---|
| [`docs/01_研究设计与独创性.md`](docs/01_研究设计与独创性.md) | 研究定位；与德重弘志 2026 及其待刊《喜金刚》研究的五层区分；对《评估》十项建议的逐条处置；研究问题与可证伪结论；工作包与关口；许可 |
| [`docs/02_矩阵与偏移度量.md`](docs/02_矩阵与偏移度量.md) | 三层矩阵的形式定义；"金标准是分布"；单元格记录；偏移向量 \(d_{cov}, d_{len}, d_{lit}, d_{ord}, d_{split}, \mathbf d_{comp}\)；三分分解；统计报告规范；证据门与弃判；LLM 过度归因实验 |
| [`docs/03_远古文本处理方法论.md`](docs/03_远古文本处理方法论.md) | 从本项目提炼的"见证矩阵法"：十条原则、八步流水线、最小度量集合、谬误清单、人机分工、复现包规范、迁移指南 |
| [`docs/04_LLM角色与5090部署.md`](docs/04_LLM角色与5090部署.md) | 开源大模型的九个角色（嵌入检索、对齐复核、命题切分、组件预填、主题预标、带弃判归因、跨模型反事实、记忆污染探针、逐格说明）及其代码强制的约束；单卡 RTX 5090 上 Qwen3 / Gemma 3 / MITRA 的部署与选型；评估门槛 |
| [`docs/05_迁移研究：老子诸本与候选文本.md`](docs/05_迁移研究：老子诸本与候选文本.md) | 《老子》诸本（郭店/马王堆/北大简/王弼/河上公/傅奕/想尔注/引文）作为"无源语言、同语言多见证"的反向迁移；`d_lex` 与章序度量；可证伪的预期读数；二十余种候选古代文本及选择标准 |
| [`data/witnesses.yaml`](data/witnesses.yaml) | 见证清单（矩阵的列）：梵文校勘本与写本、德格本及其修订层、宋译、西夏时期与明抄本汉译、西夏文、蒙文、注疏引文、现代译本（仅桥梁） |
| [`data/concordance/chapters.yaml`](data/concordance/chapters.yaml) | 章级同本系概表（G0 关口输入），含 L1 篇幅比证据与待核事项 |
| [`data/anchors/terms.yaml`](data/anchors/terms.yaml) | 梵–藏–汉锚点词表（B0 对齐基线的唯一跨语信号） |
| [`data/llm/models.yaml`](data/llm/models.yaml) | 5090 模型配置：生成模型 profile（vLLM / transformers / mock）、嵌入模型、各任务默认模型、显存预算与许可 |
| [`data/transfer/laozi/`](data/transfer/laozi/) | 《老子》14 列见证清单、郭店/马王堆章序、异文归一表（避讳/通假/异体/虚词分类）、文本格式说明 |

## 代码

纯 Python（3.10+），核心唯一依赖 PyYAML；大模型相关功能通过本地 OpenAI 兼容服务（vLLM / Ollama / llama.cpp）调用，不引入 GPU 依赖。

```bash
pip install -e ".[dev]"
python -m hevajra_matrix fetch --raw data/raw          # 下载 CBETA T0892 XML 与德格本 rgyud 'bum nga 卷（记录 SHA-256）
python -m hevajra_matrix run   --raw data/raw --out out # 切分 → 章级门控 → 对齐（含 NULL）→ 矩阵 → 统计 → 报告
pytest

# 在 5090 上：vllm serve Qwen/Qwen3-32B-AWQ --served-model-name qwen3-32b-awq --max-model-len 8192
python -m hevajra_matrix run --similarity hybrid --embedding-model BAAI/bge-m3      # B1 嵌入 + 锚点混合对齐（需 pip install -e ".[embeddings]"）
python -m hevajra_matrix llm-judge     --llm qwen3-32b-awq --review out/alignment_review.csv   # 对可疑珠出裁决 → review 表
python -m hevajra_matrix llm-extract   --llm qwen3-32b-awq --witness zh_T0892_song            # 组件预填 → out/derived/components.jsonl
python -m hevajra_matrix llm-attribute --llm gemma3-27b    --witness zh_T0892_song            # 固定标签集归因 + 动机词降级
python -m hevajra_matrix llm-probe     --llm mitra-qwen3.5 --passages out/cells.csv           # 记忆污染探针
python -m hevajra_matrix llm-counterfactual --llm qwen3-32b-awq,gemma3-27b --items items.csv  # 跨模型过度归因实验
# 任一 llm-* 子命令加 --llm mock 即可无 GPU 干跑

# 迁移研究：《老子》诸本（文本自备，格式见 data/transfer/laozi/README.md）
python -m hevajra_matrix transfer --config data/transfer/laozi --texts data/transfer/laozi/texts --out out/laozi
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
| `pipeline.py` / `cli.py` | 端到端运行；合并品（如汉第 11 品 = I.11 + II.1）按概表自动处理；`--similarity anchors|embedding|hybrid` |
| `llm/backends.py` | `LLMBackend` 协议；vLLM/Ollama/llama.cpp（OpenAI 兼容，JSON schema 约束）、进程内 transformers（bf16 / 4-bit）、mock；每次调用的哈希审计日志；`data/llm/models.yaml` 读取 |
| `llm/embeddings.py` | 句向量相似度后端（bge-m3 / MITRA-E / Qwen3-Embedding）与"锚点 ∨ 加权嵌入"混合后端，直接插入对齐 DP |
| `llm/tasks.py` | 九个角色 R1–R9 的任务函数，全部带机械约束：子串校验、封闭标签集、动机词降级、只写派生层 |
| `transfer/sinitic.py` | 同语言多见证矩阵（《老子》）：`#N` 章标记文本格式、□ 残字 → LACUNA、异文归一、字符 n-gram 相似度、`d_lex`、非单调重连、章序 Kendall τ、选编本未收章 → NA |

## 三种参照等级

- **gold**：`data/reference/` 下存在梵文 TSV 时，梵文为参照，藏译与汉译都被测量，三分分解完整。
- **provisional**：无梵文文本时以德格本为临时参照，所有输出打 `reference_grade=provisional`，只描述汉译相对藏译的关系。当前仓库状态下运行即为此模式。
- **coordinate_only**（迁移研究）：参照只是一个坐标系（《老子》通行本 81 章），根本没有金标准；报告的是见证间离散度 σ_S、章序偏移与 `d_lex`，而非任何一列"相对源文本的偏移"。

## 当前状态（v0.1）

- 流水线在真实 T0892 与 Toh 417–418 上端到端运行（约 10 秒）：3,045 个临时参照单元、20 个汉文品全部映射、3 个品（II.1、II.12 并入相邻汉品；II.9 待判）按概表处理。
- L1 篇幅比已给出三条待核的结构假说（见 `chapters.yaml`）：汉第 11 品含 II.1；汉第 20 品含 II.12；II.9（抽字法品）在汉译中缺失或压缩入第 18 品。
- B0 对齐是长度主导的基线，NULL 检测参数（`prior_null`、`anchor_weight`）需在试标金标准上重调并预注册；`out/alignment_review.csv` 为人工修正表。
- 组件层（L3）标注、多列梵文、其他甘珠尔版本、西夏/明汉译尚未接入；接口已按 `witnesses.yaml` 预留。
- LLM 层（v0.2）：九个角色的任务函数与五个 `llm-*` 子命令全部可用 `--llm mock` 无 GPU 干跑并有测试覆盖；5090 上的真实运行需按 `docs/04 §3` 起 vLLM 服务，尚未在真机上跑吞吐与评估门槛。
- 迁移层（v0.2）：《老子》流水线在三章示例夹具上端到端运行（郭店未收章 → NA、"民之從事…"→ ABSENT、絕巧棄利前移由非单调重连检出、马王堆德在道前 τ=0.67、恆/常避讳被归一解释）；全量运行需研究者自备释文。

## 发布边界

仓库只含坐标、标签、数值、代码与合理长度的短引文。CBETA（CC BY-NC-SA）、德格本电子文本（许可待核）、梵文校勘本（版权）、84000（CC BY-NC-ND）、出土简帛释文（版权）的全文均不入库；`data/raw/`、`data/reference/*.tsv`、`data/transfer/*/texts/` 已被 `.gitignore` 排除。模型权重不入库；模型输出只发布派生层的标签、分布与哈希日志。
