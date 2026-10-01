# 输出文件与术语对照

报告与输出文件一律为英文。本表给出中文对照。字段的完整定义见 [`docs/data-formats.md`](../docs/data-formats.md)。

## 1 运行目录 `runs/<UTC 时间>-<git 短哈希>/`

| 文件 | 内容 |
|---|---|
| `manifest.json` | 本次运行的清单：代码版本、git 提交、输入文件与配置的 SHA-256、工具指纹、实际服务的模型、缓存命中数 |
| `ingest/` | 切分结果、大正藏脚注、德格异文、题记元数据、切分报告 |
| `alignments/*.jsonl` | 各对齐器的结果：`claude.*`（Claude 对勘）、`dp_zero`（纯长度对照 P1）、`dp_anchor`（锚点基线 B0）、`shuffled`（洗牌对照 P2）、外部导入的对齐 |
| `collation/` | 每个窗口、每次重复的已核验提案与诊断 |
| `matrix/cells.csv` | 矩阵长表：每行一个（参照单元，见证）单元格 |
| `matrix/units.csv`、`wide_status.csv` | 单元表；按状态的宽表 |
| `matrix/stale_verdicts.csv` | 因原文指纹变化而**未被套用**的人工裁决 |
| `evaluation/scores.json`、`gate.json`、`sentinels.json` | 各对齐器在金标准上的得分；门控结果与报告级别；哨兵核对结果 |
| `stats/estimates.json` | 各估计量（含区间、覆盖的误差来源、无法估计的原因） |
| `summary.md`、`*.svg` | 分级报告与图 |
| `llm_audit.jsonl` | 每次 LLM 调用的审计记录（不含原文） |
| `review/` | 导出的复核表格（含原文，不提交） |

## 2 术语

| 英文 | 中文 | 说明 |
|---|---|---|
| reference unit | 参照单元 | 矩阵的行；现阶段是德格本的句/偈行，接入梵文后改为梵文单元 |
| witness | 见证 | 矩阵的列，如宋译 T0892、德格本 |
| link / alignment | 链接 / 对齐 | 参照单元与汉文片段的对应 |
| relation | 关系 | equivalent 等价、paraphrase 意译、expanded 增补、generalised 概括、substitution 替换、reversal 反转、category_name_omitted 类名略去、transliterated 音写、abridged 部分、no_counterpart 无对应 |
| witness-only | 仅见于见证 | addition 增写、translator_note 译者夹注、paratext 副文本、belongs_elsewhere 属于别处 |
| status | 状态 | PRESENT 在场、PARTIAL 部分、ABSENT 缺失、UNALIGNED 未判定（附原因）、LACUNA 物理残缺、NA 不覆盖 |
| D_any / D_cov | 任何偏移 / 覆盖偏移 | 前者含改写与音写，后者只算部分与缺失 |
| grade A/B/C/X | 证据等级 | A 人工裁决；B 三次重复一致且核验全通过；C 多数或有非致命标记；X 核验失败或无多数 |
| gold (dev / test / test_second) | 金标准（开发集 / 测试集 / 第二标注者） | 只接受盲标 |
| control: P1 / B0 / P2 | 对照：纯长度 / 锚点基线 / 洗牌 | Claude 必须显著优于所有对照 |
| perturbation | 扰动测试 | wrong window 错窗、deletion 删句、negation 去否定词 |
| sentinel | 哨兵事实 | 每次运行都核对的已知原文事实；verified 已签署 / proposed 待确认 / retired 已撤回 |
| gate G0–G4 | 门控 | G0 完整性、G1 效度、G2 工具完整性、G3 校正、G4 按估计量 |
| report level 0/1/2 | 报告级别 | 0 描述级（DESCRIPTIVE）、1 已验证（VALIDATED）、2 已校正（CALIBRATED） |
| confirmatory / exploratory | 确认性 / 探索性 | 预注册一致且测试集只评过一次才算确认性 |
| two-phase estimation | 两阶段估计 | 机器阳性核验＋机器阴性分层抽检，后验预测插补 |
| stratum | 分层 | `pos:<类>:<主题组>`、`neg:<等级>:<主题组>`、`unresolved` |
| Manski bounds | Manski 界 | 未解决单元"全算偏移/全不算"给出的上下界 |
| NOT_ESTIMABLE | 无法估计 | 打印原因，绝不打印 0 |
| instrument digest | 工具指纹 | 模型、提示词、schema、effort、参数的哈希 |
| substituted_model | 模型被替换 | 服务器端 fallback 改用了别的模型；只作复核提示 |
| ledger | 评估账本 | 测试集每被评估一次追加一行 |
| E1–E6 | 估计量 | E1 偏移比例、E2 音节加权比例、E3 分解、E4 敏感-中性差 Δ、E5 仅见于汉文的材料、E6 实验效应 |
| attested_vorlage / attested_translator | 夹注证明的底本缺文 / 译者替换说明 | 前者可归为底本，后者属译者一侧，不减少残余 |
| shared_revised | 与修订后藏译共享 | 只有德格本（经宣奴贝修订）作共同见证时的标签 |
