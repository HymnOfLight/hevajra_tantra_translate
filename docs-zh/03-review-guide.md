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
| `generalised` | 用上位词概括；或列表中几项被合并成一个总称（如"三身"），这时每一项都填 `generalised`、链接到总称所在的那一行 | "狗肉"→"肉"；"三身三業" |
| `substitution` | 指称被换成别的东西 | II.9"杀"→"降伏" |
| `reversal` | 肯定/否定、命令/禁止翻转（同时把 `polarity_flip` 填 1） | II.3"你当杀生"→"不應殺害" |
| `category_name_omitted` | 列表每项在汉文里仍各占一位，只是名字换成了序数或计数 | I.7 十二处；"第一門、第二門" |
| `transliterated` | 内容词、说明文字或真言只音写、不意译（专名、种子字、字母的音写不在此列，见 1a 节） | "金剛妃之 bhaga"→"金剛喻施婆倪" |
| `abridged` | 只保留了一部分 | — |
| `no_counterpart` | 在映射的汉文品及相邻品里都找不到对应（在这个范围以外找到对应的填 `unresolved`，见 §2） | — |
| `lacuna` | 汉文此处物理残缺（不是没译） | — |
| `unresolved` | 判断不了（会单独统计，不进分母） | — |

可选标记（`flags`，多个用分号隔开）：`scope_list`（整张列表的现象）、`uniform_across_list`（整表包括中性成员都做了同样处理，这类单元不进入敏感/中性对比）、`instruction_as_mantra`（说明文字被当成真言音写）、`reordered`（顺序被调整）、`unsure`（不确定）。

藏文说明真言怎样拼出（取某字、加某字、末尾加 svāhā 之类），汉文只给出拼好的真言：按现行规则判 `transliterated`，标 `instruction_as_mantra`，`note` 栏写"真言拼法"，以便专家定下规则之后复查（[`09-collate-v2-draft.md`](09-collate-v2-draft.md) Q10）。

仅见于汉文的片段（参照里没有的材料）在汉文侧表格的 `witness_only` 列填：`addition`（增写）、`translator_note`（译者夹注）、`paratext`（题记等副文本）、`belongs_elsewhere`（其实属于别处的内容）。

---

## 1a 惯例（`convention`，与关系码分开填）

有些差异只是译法上的习惯，不是内容的改变。这类情况另记在"惯例"一栏，和关系码分开：

| 惯例 | 何时填 |
|---|---|
| 空 | 没有惯例（导入时读作 `none`） |
| `?` | 暂不判定（导入时读作"未标注"，这个单元不进入惯例指标，也不进入 `relation_kappa_shared`）。但到 `build`、`stats` 时仍留着的 `?` 在矩阵、D_any_conv 和 E1_any_conv 里按"无惯例"计（`details.json` 的 `any_conv` 会报告数目），所以正式测量之前所有 `?` 都要换成确定的惯例。只能用于有链接的关系，用于两种情况：（一）参照单元里有 Hevajra 一名（ཀྱེའི་རྡོ་རྗེ、ཀྱེ་ཡི་རྡོ་རྗེ、ཀྱེ་རྡོ་རྗེ、ཀྱེ་རྡོར 等），所链接的汉文用了任何由"空智"构成的名字（空智金剛、大悲空智金剛、金剛空智、空智主、空智大金剛王等），而译名表条目尚未核实：关系码按"名字已被翻译"来填，`note` 栏写"空智"；汉文里没有这个名字的单元照常判。（二）属于下面列出的待专家决定的情况（半音、释词、类名），而您拿不准：关系码照常填 |
| `standing_rendering` | 单元里出现**译名表**所列的原词，汉文用了表中的译名（如 Hevajra 作"空智金剛"）。只能用于译名表中**已核实**的条目 |
| `transcribed_name` | 汉文把**专名**（尊名、人名、地名、脉名、法器名）或**被当作声音的音节和字母**（种子字、字母表中的字母）按音写出，单元其余部分是意译的 |

填写规则：

- 填了惯例的单元，关系码按"名字已被翻译"来填：只差译名或音写的单元，通常是 `equivalent` 或 `paraphrase`。
- 惯例只免除名字或译名这一处差异。单元里其他内容的缺失、反转或替换照常记录：名字是音写的、同时少了一个短语，就填 `transcribed_name` 加 `abridged`。惯例**永远不会**把缺失、反转或替换变成"等价"。
- 惯例只能用在有链接的关系上：`no_counterpart`、`lacuna`、`unresolved` 以及仅见于汉文的行不能填惯例。
- 内容词、说明文字和真言的音写仍是关系码 `transliterated`，不填惯例。类名（ḍākinī "拏吉尼"、yoginī "瑜儗尼" 等）被音写，目前按内容音写处理，即 `transliterated`，惯例栏留空，`note` 栏写"类名"：类名不是专名，即使指神祇也一样（[`09-collate-v2-draft.md`](09-collate-v2-draft.md) Q3，待专家在开发集上确认）。collate.v2 提示词对 Claude 写的是同一条规则。
- **半音半义的名字**：同一个原名，一部分意译、另一部分音写（如"金剛薩埵"：金剛意译，薩埵音写），暂按 `transcribed_name` 填，并在 `note` 栏写"半音"（09 Q7，待专家决定；collate.v2 提示词写的是同一条规则）。汉文另加的、不对应原名任何部分的词不算音写成分：意译的女神名后加上"菩薩"，名字仍是意译的。整个名字都是音写的（如"三昧耶薩埵"，三昧耶即 samaya 的音写）是普通的 `transcribed_name`，不必写"半音"。
- **解释词义的单元**（单元在解释某个词的含义，汉文作"X 者，謂……"并保留了 X 的音写）：被解释的词是内容，按上文类名、内容词的规则填 `transliterated`，惯例栏留空，在 `note` 栏写"释词"（09 Q8）。这一条只管词：种子字、字母，或逐个音节解释的名字中的音节（如 Heruka 的 he、ru、ka），即使单元说明了它代表什么，仍记 `transcribed_name`。
- 种子字或字母后面紧接着释义（"某字者，即某尊"），照常判：字母本身记 `transcribed_name`，释义部分按关系码记。
- **两种惯例同时出现**（同一单元既有译名表的名字，又有另一个音写的名字）：以 `standing_rendering` 为准。Hevajra 条目尚未核实时填 `?`，`note` 栏同时写"空智"和"半音"（或"音写"）；核实后以 `standing_rendering` 为准，不采用或译名不在条目之内时改为 `transcribed_name`。
- **专家对 Hevajra 条目有结论之后**（核实、修改后核实或不采用），研究者会逐一复查所有含 Hevajra 的单元（不论惯例栏填的是什么）：汉文译名在核实条目之内的，改为 `standing_rendering`；条目不采用、或这个译名不在条目之内的：单元另有音写的名字（`note` 栏写了"半音"或"音写"）时把 `?` 改为 `transcribed_name`，否则改为空（`none`）；并按普通规则重判关系。
- `?`、`;` 和各个代码请用半角（英文）字符输入；全角的"？""；"程序也认。
- 惯例不要写进 `flags` 栏；`convention_unsupported` 是程序的标记，人工不填。

**译名表。** 每张金标准表、盲表和揭示表旁边都有 `standing_renderings.csv`，列出已核实的条目（`entry`、`reference_forms`、`witness_renderings`，多个形式用 ` | ` 分隔）。这份表与提示词里给 Claude 的是同一份；没有已核实条目时只有表头，这时不能填 `standing_rendering`；遇到待核实的候选条目（目前只有 Hevajra，见 `data/codebook/collate.v2/standing_renderings.yaml`），按上面 `?` 的规则填。注意：留空读作 `none`（"没有惯例"），不是"未标注"。导入时程序会核对：所链接的汉文段里须有该条目的译名，参照单元里须有同一条目的原词（藏文从音节开头匹配，末音节可带黏着助词，只看字母；以 a-chung 结尾的条目形式遇到 -r、-s 取代 a-chung 的写法，如 `དཔར`，不算匹配，需要把这种写法另列为形式）。核对不通过的行会按行号报错，整张表不导入。

关系和惯例分开记录，金标准就不会被将来口径的选择锁死：主结果 D_any 不把惯例算作偏移，"连惯例一起算"的版本（D_any_conv）作为敏感性分析同时报告，两种口径都能从同一份金标准算出来。

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

- 参照侧表（`.ref.csv`）每行一个参照单元：在 `links` 填对应汉文片段的句柄（汉文侧表 `handle` 列，多个用空格隔开；无对应就留空），在 `relation` 填关系码，在 `convention` 填惯例（见 1a 节；没有就留空）。
- 汉文侧表（`.wit.csv`）只需要给仅见于汉文的片段填 `witness_only`。
- 在汉文侧表覆盖的范围内搜索对应（概表映射的汉文品及前后各一品），不要只看相邻几行：II.9 的开头就在汉文第 18 品后半，也在这个范围内。`links` 只接受本区域汉文侧表里的句柄；如果在这个范围以外找到对应，关系填 `unresolved`，`links` 留空，在 `note` 栏写"越窗"和那个汉文片段的编号（如 `T0892:0593b19.1`）。`note` 栏会随金标准一起入库，请只写标记和编号，不要抄原文。`unresolved` 的单元不计分，所以您找到的越窗对应不会算成 Claude 的错误；但范围以外而没有被找到的对应仍会记成 `no_counterpart`，Claude 在那里正确的越窗链接会被算成错误，这是已知的局限。

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
# 盲判：填 <批次>.blind.csv 的 blind_relation / blind_wit_loci / blind_convention / blind_flags
hevajra-matrix review import --task verify --file <填好的盲表> --annotator 姓名 --minutes 用时
hevajra-matrix review export --task reveal --batch <批次>                # 导出揭示表（只含尚未揭示的条目）
# 看机器结论后填 final_relation / final_wit_loci / final_convention / final_flags；如果改判，在 revised_reason 写原因
hevajra-matrix review reveal --file <填好的揭示表> --annotator 姓名
hevajra-matrix review status                                              # 看各层完成度
# 再次 review export --task verify --hours 4，得到下一批；如此循环直到各层完成
```

揭示表的 `machine_convention` 是机器多数决的惯例（collate.v1 的运行一律为 `none`）；程序未能核实的惯例显示为 `<惯例>:unsupported`。`final_*` 各栏预填盲判的结论。盲判的惯例未标注时（盲表填了 `?`，或 collate.v2 以前的盲表），凡关系可带惯例的行，`blind_convention` 与 `final_convention` 都显示为 `?`：不改动就仍是未标注；改为确定的惯例不算改判，不必写 `revised_reason`。盲判的惯例已确定时，只改惯例也算改判（这时 `final_convention` 留空或填 `?` 都读作 `none`），要写 `revised_reason`。揭示之后仍是 `?` 的条目不会再出现在新的揭示表上；要补填，就在已填好的 `<批次>.reveal.csv`（请自留一份副本：同一批次下一次导出揭示表时会覆盖它）里改 `final_convention`，再用 `review reveal --file` 重新导入。重新导入会用表上的内容覆盖所有行的 final 各栏，并把 `final_date` 改为这次导入的日期；盲判各栏不变，补填惯例不必写 `revised_reason`。

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

译名表 `data/codebook/collate.v2/standing_renderings.yaml` 的条目也要这样签署：核对计数和原文段落后，把 `status: proposed` 改为 `verified`，在 `verified_by` 写真名（非空白的名字；其他状态必须是 `null`）。只有 `verified` 的条目才会给 Claude 看、出现在表格旁的 `standing_renderings.csv` 里、被导入时的核对接受。核实条目，或修改、退役、删除已核实条目，就改变了 collate.v2 的提示词，相当于换了测量工具；改动 `proposed` 条目不改变提示词。详见 [`09-collate-v2-draft.md`](09-collate-v2-draft.md) §6。

---

## 5a 梵文参照模式下的复核

接入梵文参照后（见 `02-plan.md` §3），参照单元换成梵文单元（Snellgrove 编号），德格本与汉文是两个并列的见证，各有自己的金标准、核验计划与裁决：

- 表格与人工数据按参照分开存放：`data/annotations/by_reference/<参照>/gold|verdicts|topics/...`（布局与 `data/annotations/` 相同）。以德格本为参照时做的金标准与裁决不会套用到梵文单元上。
- 复核命令默认针对汉文；做德格本一列时加 `--witness bo_derge_D417_418`，例如 `hevajra-matrix review export --task gold --set dev --witness bo_derge_D417_418`。所需语言变为梵＋藏或梵＋汉（`--competence sa,bo` / `sa,zh`）。
- 写本异读（`data/reference/readings/<品>.tsv`）只登记写本（`data/registry/sa_manuscripts.yaml` 里的 id）；校勘本的读法即使录入也不计入分解。某写本在某单元未校时填 `not_collated`，不要填 `absent`。
- 分解的 A2 规则：一个"共享"偏移只有在汉文列与德格本列都经人工确认（A 级）后才在下界中算作共享；因此对进入分解的偏移单元，最好两列都核验。

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
