# 输出文件与术语对照

报告与输出文件一律为英文。本表给出中文对照。字段的完整定义见 [`docs/data-formats.md`](../docs/data-formats.md)。

## 1 运行目录 `runs/<UTC 时间>-<git 短哈希>/`

| 文件 | 内容 |
|---|---|
| `manifest.json` | 本次运行的清单：代码版本、git 提交、输入文件与配置的 SHA-256、工具指纹、实际服务的模型、缓存命中数 |
| `ingest/` | 切分结果、大正藏脚注、德格异文、题记元数据、切分报告、G0 结果 `g0.json`、切分阶段哨兵结果 `sentinels.jsonl` |
| `alignments/*.jsonl` | 各对齐器的结果：`claude.*`（Claude 对勘）、`dp_zero`（纯长度对照 P1）、`dp_anchor`（锚点基线 B0）、`shuffled`（洗牌对照 P2）、外部导入的对齐 |
| `collation/` | 每次重复的未判定原因、诊断、替代模型的提示（`r<k>.json`）；最近一次对勘的重复、窗口、请求键、提示词版本，以及所用提示词与 schema 的 sha（`replicates.json`，G2 只检查这些请求是否被替换；`prompt` 缺省即 collate.v1；`build`、`evaluate`、`perturb` 和揭示表导入据此拒绝用另一份文本对勘的运行）；共识等级；重复间一致性（`integrity.json`，collate.v2 另有惯例的 Fleiss κ `convention_kappa`，只描述） |
| `matrix/cells.csv` | 矩阵长表：每行一个（参照单元，见证）单元格；`flags` 列可含惯例标记 `standing_rendering`、`transcribed_name` 和 `convention_unsupported` |
| `matrix/units.csv`、`wide_status.csv` | 单元表；按状态的宽表 |
| `matrix/stale_verdicts.csv` | 因原文指纹变化而**未被套用**的人工裁决 |
| `evaluation/scores.<集>.json` | 每次评估各自的得分文件（`scores.test.json`、`scores.dev.json`、`scores.<集>_baselines.json`），互不覆盖；collate.v2 起另有关系 κ `relation_kappa`、只算两个码本编码一致单元的关系 κ `relation_kappa_shared`（金标准有链接、惯例标注为 none、关系不是 generalised 或 category_name_omitted；v1/v2 采用规则比较的就是它）、`dany_conv_kappa`，这三项对所有来源都计算（不产出惯例的来源以自身的 D_any 读作 D_any_conv）；惯例的精确率/召回率/F1（总体与按惯例分项）和 `convention_unsupported_rate` 只对产出惯例的来源计算，对照、v1 共识等为 null；每个对齐器另有 `reliability`：置信度的 Brier 分数、常数预测的 Brier 分数、是否胜过常数、各档可靠性表（无置信度的对照为 null） |
| `evaluation/scores.json`、`gate.json`、`sentinels.jsonl` | 门控所依据的那次评估的得分；门控结果与报告级别（`gating`、`gold_set`、`baselines_only` 说明它来自哪次评估：只有 Claude 在测试集上的评估才门控）；哨兵核对结果（每行一条） |
| `evaluation/perturbations.json` | 扰动测试：错窗误链率、删句召回；`perturb --negation` 时另有去否定的极性召回 `negation`（只报告，不门控） |
| `stats/estimates.json` | 各估计量（含区间、覆盖的误差来源、无法估计的原因）；含惯例的敏感性变体 `E1_any_conv`（只有 final 列；惯例与类别按分层从 phase-2 裁决联合插补，只用人工惯例，v1 运行同样有效）；另有先验敏感性 `E1_<主结局>_prior`；E3 可计算时另有 `E3.phi_V`、`E3.phi_S` |
| `stats/details.json` | 改判率、Manski 界（`any`、`cov`；collate.v2 运行另有 `any_conv`）、E1_any_conv 的插补依据 `any_conv`（phase-2 裁决数、其中未标注惯例的个数（按无计）、记有惯例的个数、无偏移且记有惯例的个数、记有人工惯例的已核实单元数、由金标准或解决裁决决定而未标注惯例的已核实单元数（按无计）及其中无偏移的个数）、矩阵的惯例计数 `conventions`（固定译名、专名音写、未获核实、人工判定；只作描述）、未校正分层 `uncalibrated`、夹注上的仅见于汉文声明数 `ingest_notes`、Δ 的诊断（置换 p、TOST、重叠、机器标签的朴素 Δ 与衰减、匹配、按品/长度三分位的误判表、叙述框架句负对照、`power`）；分解 `e3`（写本列、被搁置的校勘本 `editions_ignored`、共同见证、各类计数与逐单元归类、O/E/O−E 与 φ、A2 界 `bound`）；梵文参照模式下的见证距离与 UPGMA 树 `distance` |
| `stats/power.json` | 主题标签齐全后，在真实标签上模拟的最小可检测效应与功效曲线（G4 取它与预注册 `stats.mde` 的较大者） |
| `experiments/overattribution/`（预实验在 `pilot/` 子目录） | 试次表 `trials.jsonl`、回答 `responses.jsonl`、结果 `results.json`（含 `phase`、`outcome_basis`、`human_codes_set_aside`）、盲编码表 `human_coding_sheet.csv`、其私钥 `human_sample.json` |
| `summary.md`、`*.svg` | 分级报告与图 |
| `llm_audit.jsonl` | 每次 LLM 调用的审计记录（不含原文） |
| `review/` | 导出的复核表格（含原文，不提交）；解决表多一列 `hint_other_model`（另一模型的提示，导入时剥去） |
| `witnesses/<见证>/` | 梵文参照模式下德格本一列的全部输出，布局与上面相同（`alignments/`、`matrix/`、`evaluation/`、`stats/`、`summary.md` 等）；`ingest/`、清单、审计日志共用 |

## 1b 运行目录之外

| 文件 | 内容 |
|---|---|
| `data/annotations/verdicts/<见证>/<批次>.csv` | 已提交的人工裁决（27 列，含揭示时记录的机器极性判断 `machine_polarity_flip`，以及惯例三列 `machine_convention`、`blind_convention`、`final_convention`；旧的 23、24 列文件仍可读取） |
| `data/annotations/gold/<见证>/<集>.csv` | 已提交的金标准（15 列，含 `convention` 与 `note`；旧的 13 列文件仍可读取，惯例读作"未标注"） |
| `data/codebook/collate.v2/standing_renderings.yaml` | collate.v2 的译名表；只有 `verified` 条目进入提示词、核验 V12 和复核表格旁的 `standing_renderings.csv` |
| `data/annotations/verdicts/<见证>/plan_<批次>.csv`、`strata_<批次>.json` | 核验/抽检计划与抽样时冻结的分层（只有 id 与分层，无原文；与裁决一起提交，新运行目录也读取；旧运行目录里的 `review/plan_*.csv` 仍作后备） |
| `data/ledger/test_evaluations.jsonl` | 测试集评估账本；完全相同的重复评分不追加（工具指纹、预注册、门控结果相同，已入账的每个指标都在且相等；新记录只可多出 collate.v2 新增指标族的键）。梵文参照模式下按参照与见证分账：`test_evaluations.<参照>.<见证>.jsonl` |
| `data/annotations/by_reference/<参照>/` | 梵文参照模式下的金标准、裁决与主题标签（布局同 `data/annotations/`） |
| `data/reference/<梵文参照>.tsv`、`data/reference/readings/<品>.tsv` | 研究者自备的梵文参照与写本异读（不提交）；前者存在即进入梵文参照模式，后者供分解 E3 |
| `runs/llm-cache/` | LLM 响应缓存（含原文，不提交） |
| `runs/llm-cache/llm_spend.jsonl` | 跨运行累计的花费账本，每次未命中缓存的调用一行；预算上限按它累计 |

## 2 术语

| 英文 | 中文 | 说明 |
|---|---|---|
| reference unit | 参照单元 | 矩阵的行；现阶段是德格本的句/偈行，接入梵文后（梵文参照模式）改为梵文单元 |
| witness | 见证 | 矩阵的列，如宋译 T0892、德格本 |
| link / alignment | 链接 / 对齐 | 参照单元与汉文片段的对应 |
| relation | 关系 | equivalent 等价、paraphrase 意译、expanded 增补、generalised 概括（含列表合并为总称）、substitution 替换、reversal 反转、category_name_omitted 类名略去（列表各项保留、名字换成序数）、transliterated 音写（内容词、说明文字、真言）、abridged 部分、no_counterpart 无对应 |
| convention | 惯例 | 与关系并列记录的译法惯例（collate.v2）：none 无、standing_rendering 固定译名（只限已核实的译名表条目）、transcribed_name 专名与种子字音写；在链接和单元格上记为同名标记 |
| standing renderings / glossary | 译名表 | 见证全经统一使用的译名；只有已核实条目生效 |
| convention_unsupported | 惯例未获核实 | 核验 V12 未能确认所声明的惯例；共识中持有投票关系的多数读法都没有声明投票惯例时，共识也会加上它；非致命，证据等级降为 C |
| prompt version (collate.v1 / collate.v2) | 提示词版本 | 由 `tasks.collate.prompt` 选择；默认 v1；两个版本是不同的测量工具 |
| witness-only | 仅见于见证 | addition 增写、translator_note 译者夹注、paratext 副文本、belongs_elsewhere 属于别处 |
| status | 状态 | PRESENT 在场、PARTIAL 部分、ABSENT 缺失、UNALIGNED 未判定（附原因）、LACUNA 物理残缺、NA 不覆盖 |
| D_any / D_cov | 任何偏移 / 覆盖偏移 | 前者含改写与内容音写，后者只算部分与缺失；两者都不计惯例，主结果只能是二者之一 |
| D_any_conv / E1_any_conv | 含惯例的任何偏移 | D_any 之外，带惯例的单元也算偏移；只作 E1 的敏感性分析，从不作主结果；它比 collate.v1 的 D_any 宽得多（单元里有任何音写、半音半义或固定译名的名字都算），不能直接当作德重 2026 的口径，与德重可比的是 D_any；惯例与类别联合插补，只用人工惯例，所以 v1 运行同样有效；没有人工记下无偏移单元的惯例时等于 E1_any |
| grade A/B/C/X | 证据等级 | A 人工裁决；B 三次重复一致且核验全通过；C 多数或有非致命标记（惯例标记本身不降级，`convention_unsupported` 降为 C）；X 核验失败或无多数 |
| gold (dev / test / test_second) | 金标准（开发集 / 测试集 / 第二标注者） | 只接受盲标 |
| control: P1 / B0 / P2 | 对照：纯长度 / 锚点基线 / 洗牌 | Claude 必须显著优于所有对照 |
| perturbation | 扰动测试 | wrong window 错窗、deletion 删句、negation 去否定词 |
| sentinel | 哨兵事实 | 每次运行都核对的已知原文事实；verified 已签署 / proposed 待确认 / retired 已撤回 |
| gate G0–G4 | 门控 | G0 完整性、G1 效度、G2 工具完整性、G3 校正、G4 按估计量 |
| report level 0/1/2 | 报告级别 | 0 描述级（DESCRIPTIVE）、1 已验证（VALIDATED）、2 已校正（CALIBRATED） |
| confirmatory / exploratory | 确认性 / 探索性 | 预注册一致且测试集只评过一次才算确认性 |
| two-phase estimation | 两阶段估计 | 机器阳性核验＋机器阴性分层抽检，后验预测插补 |
| stratum | 分层 | `pos:<类>:<主题组>`、`neg:<等级>:<主题组>`、`unresolved`、`pos:witness_only`；抽样时冻结 |
| uncalibrated stratum | 未校正分层 | 有未核验单元却没有任何第二阶段样本裁决的分层；其估计只会是先验，故相关估计量打印 `NOT_ESTIMABLE`，G3 不通过 |
| Manski bounds | Manski 界 | 未解决单元"全算偏移/全不算"给出的上下界 |
| NOT_ESTIMABLE | 无法估计 | 打印原因，绝不打印 0 |
| instrument digest | 工具指纹 | 模型、提示词、schema、effort、参数的哈希；collate.v2 的提示词含已核实的译名表，核实一个条目就会改变指纹 |
| substituted_model | 模型被替换 | 服务器端 fallback 改用了别的模型；只作复核提示 |
| ledger | 评估账本 | 测试集每被评估一次追加一行 |
| E1–E6 | 估计量 | E1 偏移比例、E2 音节加权比例、E3 分解、E4 敏感-中性差 Δ、E5 仅见于汉文的材料、E6 实验效应 |
| attested_vorlage / attested_translator | 夹注证明的底本缺文 / 译者替换说明 | 前者可归为底本，后者属译者一侧，不减少残余 |
| shared_revised | 与修订后藏译共享 | 只有德格本（经宣奴贝修订）作共同见证时的标签 |
| Sanskrit reference mode | 梵文参照模式 | 有梵文参照 TSV 时，以梵文单元为行，德格本与汉文同为见证列 |
| co-witness | 共同见证 | 分解中判断"共享"偏移所用的另一个见证：汉文用德格本，德格本用汉文 |
| manuscript / edition | 写本 / 校勘本 | 只有写本算分解的写本列；校勘本不是独立见证，从不计数 |
| A2 bound | A2 界 | 未经两列人工核验的"共享"按"非共享"计，给出共享类的下界 |
| Brier score | Brier 分数 | 置信度（high/medium/low 的名义概率 0.9/0.7/0.5）对"状态正确"的平方误差均值；须低于常数预测才算有信息 |
| reliability table | 可靠性表 | 每档置信度的单元数、正确数与准确率 |
| witness distance / UPGMA | 见证距离 / 平均连锁树 | 梵文参照、德格本、汉文三者之间的描述性距离与树 |
