# 人工复核指南

本文写给做金标准、核验、抽检和主题标注的人。所有工作都在电子表格里完成，不需要专门软件。字段的权威定义见英文文档 [`docs/data-formats.md`](../docs/data-formats.md)，本文只讲怎么做、为什么这样做。

---

## 0 三条基本规矩

1. **先盲判，再看机器。** 盲表里没有任何机器结论；导入盲判之后，才导出带机器结论的"揭示表"。两次判断都会保存，改判率就是自动化偏差的指标。
2. **只按文本判断，不判断动机。** 关系码描述的是"汉文怎样呈现这一单元"（保留、概括、替换、反转、略去类名、音写、部分、无对应），不写"为什么"。动机讨论不进入任何表格。
3. **原文不入库。** 导出的表格带原文，只放在 `runs/<运行目录>/review/` 下（不提交）；导入时程序自动剥去原文列，引文截短到汉文 ≤ 30 字、藏文 ≤ 60 字，再写入 `data/annotations/`。

表格是 UTF-8 带 BOM 的 CSV，可直接用 Excel、Numbers 或 LibreOffice 打开。每张表旁边有 `witness_text/<品>.txt`，是该汉文品的全文（每行"片段 id、类型、文字"），方便搜索。

---

## 1 关系码（每个参照单元填一个）

| 关系码 | 判断方法 | 例 |
|---|---|---|
| `equivalent` | 意思完整保留 | — |
| `paraphrase` | 保留，但措辞自由 | — |
| `expanded` | 保留，并有增补 | — |
| `generalised` | 用上位词概括 | "狗肉"→"肉" |
| `substitution` | 指称被换成别的东西 | II.9"杀"→"降伏" |
| `reversal` | 肯定/否定、命令/禁止翻转（同时把 `polarity_flip` 填 1） | II.3"你当杀生"→"不應殺害" |
| `category_name_omitted` | 列表成员保留，类名被略去或换成序数 | I.7 十二处 |
| `transliterated` | 只音写、不意译 | — |
| `abridged` | 只保留了一部分 | — |
| `no_counterpart` | 在映射的汉文品及相邻品里都找不到对应 | — |
| `lacuna` | 汉文此处物理残缺（不是没译） | — |
| `unresolved` | 判断不了（会单独统计，不进分母） | — |

可选标记（`flags`，多个用分号隔开）：`scope_list`（整张列表的现象）、`uniform_across_list`（整表包括中性成员都做了同样处理，这类单元不进入敏感/中性对比）、`instruction_as_mantra`（说明文字被当成真言音写）、`reordered`（顺序被调整）、`unsure`（不确定）。

仅见于汉文的片段（参照里没有的材料）在汉文侧表格的 `witness_only` 列填：`addition`（增写）、`translator_note`（译者夹注）、`paratext`（题记等副文本）、`belongs_elsewhere`（其实属于别处的内容）。

---

## 2 金标准盲标（最重要的人工工作）

金标准分两类，**都必须盲标**（标注者看不到任何机器输出）：

- **开发集**：I.1、I.7、II.3（17b.6 前后各 20 单元）、II.9 第 1–56 单元、II.11 末尾 20 单元＋II.12。只用于改提示词和调参数，**从不用于门控**。
- **测试窗**：用预注册的随机种子抽出的 12 个 25 单元的窗口，共 300 单元；其中 4 窗再请第二位标注者独立标一遍，用来算人与人一致性。测试集在冻结预注册之前不得用来评估 Claude。

操作：

```bash
hevajra-matrix ingest                            # 先切分原文
hevajra-matrix sample windows                    # 抽测试窗（只抽一次，写入 data/annotations/gold/<见证>/windows.csv）
hevajra-matrix review export --task gold --set dev     # 或 --set test / --set test_second
# 填表：runs/<运行>/review/gold/<集>_<窗>.ref.csv 与 .wit.csv
hevajra-matrix review import --task gold --file <填好的 .ref.csv> --annotator 姓名 --minutes 用时
```

- 参照侧表（`.ref.csv`）每行一个参照单元：在 `links` 填对应汉文片段的句柄（汉文侧表 `handle` 列，多个用空格隔开；无对应就留空），在 `relation` 填关系码。
- 汉文侧表（`.wit.csv`）只需要给仅见于汉文的片段填 `witness_only`。
- 在整品范围内搜索对应，不要只看相邻几行：II.9 的开头就在汉文第 18 品后半。

---

## 3 核验、抽检与"解决"

金标准之外，全经的机器判断通过"两阶段"方式校正。

**先做主题标注（第 4 节），再抽样。** 分层（`pos:<类>:<主题组>`、`neg:<等级>:<主题组>`）在抽样那一刻冻结，随计划一起保存；之后再导入的主题标签不会改变已冻结单元的分层。若在主题标注完成之前抽样，未标注的单元一律按"其他"分层，并一直如此，敏感段落就得不到自己的误差率。`sample` 在主题标签不完整时会打印警告。

| 任务 | 抽什么 | 命令 |
|---|---|---|
| `verify` 核验 | 机器判为偏移的单元：罕见要类（无对应、反转、替换、类名略去、越窗、仅见于汉文、X 级）**全部**核验，常见类（部分、概括、散文音写）按预注册比例抽样，每层至少 min(该层单元数, 20)。落在夹注上的"仅见于汉文"声明不核验（夹注在切分时已分类） | `hevajra-matrix sample verification` |
| `audit` 抽检 | 机器判为无偏移的单元，按"证据等级 B/C × 主题组敏感/其他"分 4 层，每层至少 40、C 级加倍 | `hevajra-matrix sample audit` |
| `resolve` 解决 | 机器未能判断的单元（被拒、截断、核验失败、模型被替换、无多数） | 包含在核验计划里 |

计划（`plan_<批次>.csv`）与冻结的分层（`strata_<批次>.json`）写在 `data/annotations/verdicts/<见证>/` 下、与裁决放在一起，请一并提交；新的运行目录也会读取它们。计划只抽一次（`--force` 才会重抽）。

流程（`--hours` 循环：每次出一批，填完导入后再出下一批）：

```bash
hevajra-matrix review export --task verify --hours 4 --competence bo,zh   # 按优先级出 4 小时的题（已盲判的条目跳过）
# 盲判：填 <批次>.blind.csv 的 blind_relation / blind_wit_loci / blind_flags
hevajra-matrix review import --task verify --file <填好的盲表> --annotator 姓名 --minutes 用时
hevajra-matrix review export --task reveal --batch <批次>                # 导出揭示表（只含尚未揭示的条目）
# 看机器结论后填 final_relation / final_wit_loci / final_flags；如果改判，在 revised_reason 写原因
hevajra-matrix review reveal --file <填好的揭示表> --annotator 姓名
hevajra-matrix review status                                              # 看各层完成度
# 再次 review export --task verify --hours 4，得到下一批；如此循环直到各层完成
```

同一批次的盲表和揭示表文件名固定（`<批次>.blind.csv`、`<批次>.reveal.csv`）。表中所有条目都已导入后，下一批会直接覆盖它；只要表中还有**未导入**的条目，导出就会拒绝覆盖并列出这些条目——先导入，或确认放弃后加 `--force`。

`resolve` 任务没有揭示步骤：盲判即终裁。解决表多一列 `hint_other_model`：如果服务器端 fallback 让另一个模型回答了该窗口，这里显示那个模型提出的关系和汉文片段（明确标为"FROM ANOTHER MODEL"，不是测量工具的结论，从不计入测量），仅供参考；导入时这一列被剥去。

**重要：** 抽检样本的分层是在"只有机器结论"的矩阵上算的，所以请先跑一次不带人工裁决的 `build`，再 `sample`；导入裁决后再 `build`，裁决会以 A 级覆盖机器结论。

---

## 4 主题标注

主题只用参照文本（藏文）判断，**标注时不要看汉文**，否则会污染"敏感/中性"这个暴露变量。

- 词表：`sexual`（性）、`female_agent`（女性）、`flesh_food`（肉食）、`harm`（伤害）、`theft`（盗）、`impure_substance`（不净物）、`bone_corpse`（人骨/尸）、`ritual`（仪礼）——对应德重 2026 的八类；另有 `neutral`（中性，必须单独出现）、`frame`（叙述框架句）、`mantra_control`（真言，作音写的负对照）。定义见 `data/codebook/topics.yaml`。
- 可以先让 Claude 预标（`hevajra-matrix topics prelabel`，只发送藏文），第一编码者在预标基础上确认。
- **第二编码者的表格不含预标列**（`--task topics_second`），以免被预标带偏；主题一致性 κ 只在两位人类之间计算。

```bash
hevajra-matrix topics prelabel
hevajra-matrix review export --task topics            # 第一编码者
hevajra-matrix review export --task topics_second     # 第二编码者（20% 单元，无预标）
hevajra-matrix review import --task topics --file <表> --annotator 姓名
```

---

## 5 哨兵事实的签署

`data/sentinels/sentinels.yaml` 里状态为 `proposed` 的条目（S1b 十二类名、S2 重排、S7、S8、S9、S10）是设计阶段的新读法。请专家核对原文后，把 `status` 改为 `verified` 并在 `verified_by` 写上真名；只有 `verified` 的哨兵会阻断门控。不同意的改为 `retired` 并在 `source` 里注明理由。

---

## 6 预计工时（约 90 小时，可分人分语言进行）

| 顺序 | 工作 | 工时 | 所需语言 |
|---|---|---|---|
| 1 | 开发集盲标 | 约 18 | 藏＋汉 |
| 2 | 测试窗盲标 | 约 15 | 藏＋汉 |
| 3 | 第二标注者（4 窗） | 约 5 | 藏＋汉 |
| 4 | 哨兵签署 | 约 1 | 藏＋汉 |
| 5 | 主题确认（＋第二编码者） | 约 8＋2 | 藏 |
| 6 | 解决被拒/未对齐单元 | 视情况 | 藏＋汉 |
| 7 | 罕见类核验 | 约 8 | 藏＋汉（底本问题需梵） |
| 8 | 阴性抽检 | 约 10 | 藏＋汉 |
| 9 | 常见类抽样核验 | ≤ 11 | 藏＋汉 |
| 10 | 实验人工编码 | 约 5 | 英 |

做完前三项（约 38 小时）即可达到"已验证"级报告。导入时填 `--minutes`，可以统计真实耗时、修正以后的预算。
