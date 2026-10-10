# 码本与提示词修订草案（collate.v2）

状态：**已实现，可选用；默认仍是 v1**。§4 的默认做法已全部采用，代码、码本和复核表格都已支持 v2（§6）。`config/llm.yaml` 中 `tasks.collate.prompt` 仍为 `collate.v1`，v1 的提示词、schema、请求键和工具指纹逐字不变，I.1 试跑可以离线逐字重放。改用 v2 要等 §7 的开发集对照通过之后。译名表唯一的条目（Hevajra）仍是 `proposed`，所以目前生效的译名表是**空的**。

起因：I.1 试跑暴露的三个问题，见 [`08-pilot-I1.md`](08-pilot-I1.md) §5。

---

## 0 一句话

增加一个与"关系"并列的维度：**惯例**（`convention`）。

- **关系**只描述内容有没有变：省略、替换、反转等。
- **惯例**单独记录译法上的习惯做法：
  - 全经固定的译名（`standing_rendering`）；
  - 专名和种子字的音写（`transcribed_name`）。

两者分开记录之后，主结果不再把惯例算作偏移；同时还能报告一个"连惯例一起算"的版本，便于与德重 2026 的口径（省略＋改写＋音写）比较。

---

## 1 改动一览

| | v1（现在） | v2 草案 |
|---|---|---|
| 固定译名（Hevajra 作"空智金剛"） | 按普通规则判，常被标为 `substitution` | 若该译名在**已核实的译名表**里，记 `convention: standing_rendering`，关系按"译名等于原词"来判（通常为 `equivalent` 或 `paraphrase`） |
| 专名、种子字、字母的音写（脉名、尊名、地名、"吽"字） | `transliterated`，非真言单元一律算偏移 | 记 `convention: transcribed_name`，关系按"名字已被翻译"来判 |
| 内容词或说明文字被音写（bhaga 作"喻施婆倪"） | `transliterated` | 不变，仍为 `transliterated`，仍算偏移 |
| 真言音写 | `transliterated`，真言单元不算偏移 | 不变 |
| 列表每项保留一位、名字换成序数（"第一門、第二門"） | `category_name_omitted` | 不变 |
| 几项合并成一个总称（"三身三業"） | 定义含糊（"或仅存一个数目"），被标为 `category_name_omitted` | 明确为 `generalised`，各项都链接到总称所在的那一行 |
| 输出字段 | 无 | 每个单元新增 `convention`：`none` / `standing_rendering` / `transcribed_name` |
| 代码核验 | V1–V11 | 新增 V12：`standing_rendering` 必须真的用了译名表里的形式；`transcribed_name` 的汉文引文必须像音写。不符合的标为 `convention_unsupported`（非致命，证据等级降为 C） |

关系→状态的映射**不变**：`equivalent`、`paraphrase` 仍算无偏移。

---

## 2 三条新规则

### 2.1 固定译名（`standing_rendering`）

**规则**：单元里出现译名表所列的原词，而汉文用了表中的译名，就记 `standing_rendering`。判关系时，把译名当作原词本身。

限制有三：
- 只能用于表中**已核实**的条目，代码会核对；
- 表外的译法，无论重复多少次，都按普通规则判；
- 一个单元如果除译名外还有别的变化（例如"听"变成"开示"），关系照样记录那个变化。

**译名表草案**（`standing_renderings.yaml`）只提了一条，状态为待核实：

| 条目 | 藏文出现次数 | 汉文出现次数 | 说明 |
|---|---|---|---|
| Hevajra（ཀྱེའི་/ཀྱེ་ཡི་/ཀྱེ་ རྡོ་རྗེ）→ 空智金剛 / 大悲空智金剛 | 约 45 | 18（其中"大悲空智金剛"11） | 全经统一；藏文出现次数多得多，没有对应名字的地方按普通规则判 |

**考察过、没有列入的**（记在文件里，供专家改判）：

| 条目 | 原因 |
|---|---|
| Heruka（ཧེ་རུ་ཀ，15 次） | 有时作"空智"，有时音写为"呬嚕迦"（3 次），不是统一译名；音写的那些按 2.2 处理 |
| I.1 解释 he 字音义处的"空智" | 原文在解释 he 这个音节，汉文作"空智"，音节解释随之丢失。这算惯例还是内容改变，需要专家判断 |
| ḍākinī（མཁའ་འགྲོ་མ，21 次） | 有时音写为"拏吉尼"（11），有时意译为"空行"（5） |
| yoginī（རྣལ་འབྱོར་མ，49 次） | 只有 6 次音写为"瑜儗尼"。汉文"明妃"出现 105 次，似乎覆盖了好几个原词：这是否属于"概括"，正是对勘要测量的，所以不列入 |

### 2.2 专名与种子字的音写（`transcribed_name`）

**规则**：汉文把**专名**（尊名、人名、地名、脉名、法器名）或**被当作声音的音节和字母**（种子字、字母表中的字母）按音写出，而单元其余部分是意译的，就记 `transcribed_name`。判关系时，把名字当作已被翻译。

`transliterated` 收窄为两种情况：
- 真言；
- **内容词或说明文字**被按音写出。

**试跑中的例子**：
- 脉名：藏文意译为 རྐྱང་མ，汉文按梵文音写为"羅羅拏"，是专名音写。
- 尊名：སྒྲོལ་མ 作"多羅菩薩"，是专名音写。
- 种子字："伊、鑁、摩、野"，是种子字音写。
- 序分中"金剛妃之 bhaga"作"金剛喻施婆倪"：bhaga 是内容词，不是专名，所以仍记 `transliterated`，仍算偏移。这正是"以音译行遮蔽"要测的现象。

### 2.3 列表压缩成总称（`generalised`）

**规则**：
- 列表每一项在汉文里仍各占一位，只是名字换成了序数或计数 → `category_name_omitted`。
- 几项被合并成一个总称（如"三身"）→ 每一项都记 `generalised`，链接到总称那一行。

这两类都算"在场偏移"，所以这条改动**不影响 D_any**，只影响描述性的分类统计。

### 2.4 惯例与其他差异并存时

惯例只免除名字或译名这一处差异。单元里其他内容的缺失、反转或替换照常记录。例如：名字是音写的，同时少了一个短语，就记 `transcribed_name` 加 `abridged`。惯例**永远不会**把缺失、反转或替换变成"等价"。

---

## 3 对 I.1 试跑结果的影响（估算）

以下是按 v2 规则对试跑输出人工重新归类的估算，不是重新跑出来的结果。

| | v1 实际 | v2 估算 |
|---|---|---|
| transliterated 音写 | 15 | 1（bhaga），另有 14 个变为 `transcribed_name`：脉名 11、尊名 2、种子字 1 |
| substitution 替换 | 24 | 约 23（"空智金剛"单独对应 Hevajra 的一处转为惯例；Heruka 与 he 字那几处不在译名表内，照旧） |
| category_name_omitted 类名略去 | 5 | 0（"三身三業"5 项转为 `generalised`） |
| **D_any（任何偏移）** | **68 / 162 = 42%** | **约 53 / 162 ≈ 33%** |
| D_cov（只算缺失与部分） | 23 / 162 | 23 / 162（不变） |
| "连惯例一起算"的 D_any | — | 68 / 162（与 v1 相同，供与德重口径比较） |

单是 I.1 一品，口径的差别就有约 9 个百分点。所以这个决定必须在开发集阶段定下来，并写进预注册。

---

## 4 需要您决定的问题

| # | 问题 | 草案的默认做法 |
|---|---|---|
| Q1 | 是否采用"惯例"这一维度，即主结果不把固定译名和专名音写算作偏移？ | 采用。主结果 D_any 不计惯例；"含惯例"的 D_any 作为敏感性分析同时报告，与德重口径比较 |
| Q2 | 译名表收哪些条目？I.1 解释 he 字音义处的"空智"算惯例吗？ | 只收 Hevajra 一条，由专家核实后签名启用；he 字那处按普通规则判，等专家决定 |
| Q3 | 类名（ḍākinī、yoginī 等）被音写，算"专名音写"还是"内容音写"？ | 暂按内容音写（`transliterated`，算偏移），与 v1 一致；如果专家认为属于汉传佛典的通行做法（类似"菩薩""羅漢"），可改为惯例 |
| Q4 | 列表压缩成总称，用 `generalised`？ | 是。不影响 D_any |
| Q5 | 种子字和字母的音写算惯例？ | 是。它们本来就是声音，只能音写 |
| Q6 | 什么时候切换？ | 开发集盲标**之前**定稿码本；金标准表同时记"关系"和"惯例"两列，这样 v1、v2 两种口径都能从同一份金标准算出来 |

**研究者决定（2026-10-10）：以上默认做法全部采用。** 具体说：主结果 D_any 不计惯例，E1_any_conv 作为敏感性分析报告；译名表只显示和使用已核实条目（Hevajra 一条仍待专家签名，所以目前为空）；类名音写仍是 `transliterated`；列表压缩为 `generalised`；种子字和字母为 `transcribed_name`；金标准和裁决把关系与惯例分开记录。默认提示词仍是 v1，等 §7 的对照之后再决定是否切换。

---

## 5 金标准与人工复核怎么配合

以下内容**已合并进**复核指南（[`03-review-guide.md`](03-review-guide.md) §1、§1a、§2、§3、§5）。表格的惯例列不论运行用的是哪个版本都存在，所以人工标注从现在起就可以同时记关系和惯例。原拟改内容如下：

- **关系码表**：
  - `transliterated` 的说明改为"内容词、说明文字或真言只音写、不意译"；
  - `category_name_omitted` 的说明改为"列表每项保留一位，名字换成序数"；
  - `generalised` 增加"几项合并为一个总称（如'三身'）"。
- **新增"惯例"列**（`convention`），盲表、揭示表和金标准表都加。可填：
  - 空（无）；
  - `standing_rendering`：只用于译名表所列条目；
  - `transcribed_name`：专名、种子字、字母被音写。

  填了惯例的单元，关系码按"名字已被翻译"来填。
- **盲标员看到的译名表**与提示词里给 Claude 的是同一份（只含已核实条目），以 `standing_renderings.csv` 的形式放在每张金标准表、盲表和揭示表旁边。

金标准把"关系"和"惯例"分开记录，就不会被将来口径的选择锁死。

---

## 6 实施情况

### 6.1 已实现的内容

1. **对勘器**（`collate/prompt.py`、`collate/collator.py`）：
   - 提示词版本由配置选择：`tasks.collate.prompt`，取值 `collate.v1`（默认）或 `collate.v2`；其他取值报错；
   - v2 的模板是 `hevajra_matrix/prompts/collate.v2.md`，示例和译名表在 `data/codebook/collate.v2/`；只有已核实的条目进入提示词，放在示例之后的 STANDING RENDERINGS 块里，没有条目时写 `(none)`；
   - v2 的输出 schema 在每个单元的 `relation` 之前增加 `convention`；
   - 链接的来源写作 `<模型>:collate.v2:<重复>`，`collation/replicates.json` 记下 `prompt`（旧文件没有这个键，视为 v1）。
2. **核验**（`collate/verify.py`）：新增 V12。
   - `standing_rendering`：同一个已核实条目的原词出现在藏文引文里、译名出现在汉文引文里（做异体折叠；藏文从音节开头匹配，末音节可带黏着助词）；
   - `transcribed_name`：汉文引文的音写字比例达到 V11 的阈值；
   - 不符合时保留惯例标记，另加 `convention_unsupported`，证据等级降为 C，从不致命。
3. **共识**（`collate/consensus.py`）：惯例按三分之二多数决定，达不到就是 `none`；`collation/integrity.json` 记下重复之间惯例的 Fleiss κ（`convention_kappa`，只描述，不进门控）。
4. **矩阵与统计**：关系→状态的映射不变，惯例不改变 D_any；新增结果变体 `any_conv`，只用于 E1 的 final 列（`E1_any_conv`），以及 Manski 区间；主结果只能是 `any` 或 `cov`，`prereg freeze` 和 `stats` 都会拒绝 `any_conv`。
5. **复核表格**：金标准表、盲表、揭示表增加惯例列，裁决文件增加 `machine_convention`、`blind_convention`、`final_convention`（27 列），金标准文件增加 `convention`（14 列）；旧文件仍可读取，缺的列读作"未标注"。人工填的 `standing_rendering` 用与 V12 相同的规则核对。
6. **评估**：关系一致性照旧计算，另报告 `relation_kappa`、`dany_conv_kappa`、惯例的精确率、召回率、F1 和 `convention_unsupported_rate`；不产出惯例的来源（基线、v1 共识等）这些指标为 n/a。门控 G1–G4 不变，不加惯例门控。
7. **防止混用**：`evaluate`、`perturb` 和揭示表导入会核对配置的版本与该运行实际对勘所用的版本，不一致就报错。

### 6.2 v1 保持不变

默认配置下，v1 的提示词 sha256（`3ab88636…`）、schema sha256（`ed99b249…`）、collate 工具指纹（`d0233b34…`）和 I.1 试跑的两个请求键都与改动前相同；把试跑目录复制一份离线重放，`alignments/claude.r1.jsonl` 和 `collation/r1.json` 逐字相同，`replicates.json` 只多了 `"prompt": "collate.v1"`。v1 的链接和单元格不带惯例标记，所以 v1 运行的 E1_any_conv 等于 E1_any。

### 6.3 怎样切换到 v2

1. 在 `config/llm.yaml` 中把 `tasks.collate.prompt` 改为 `collate.v2`。
2. 用**新的运行目录**跑 `collate`（或 `run`）。v2 的请求键都是新的，缓存里没有，需要真正调用模型；已经用 v1 对勘过的运行目录不能直接接着用 v2 评估。
3. 改回 v1 时也一样：每个运行目录只认它对勘时用的版本。
4. 预注册冻结之后再切换，要用 `hevajra-matrix prereg freeze --amend "<理由>"` 更新工具指纹。
5. 梵文参照的语言对要用 v2，需要先提交 `data/codebook/collate.v2/examples.<参照>-<见证>.yaml` 和 `standing_renderings.<参照>-<见证>.yaml`（条目可以为空），否则报错。

### 6.4 怎样核实译名表的条目

1. 专家核对 `data/codebook/collate.v2/standing_renderings.yaml` 中条目的计数和原文段落。
2. 同意的，把 `status: proposed` 改为 `status: verified`，在 `verified_by` 写真名（`verified_by` 只能、也必须随 `verified` 一起填）；不同意的，改为 `status: retired`，或移到 `considered_not_listed` 并写明理由。
3. 新增条目要写 `id`、`reference_forms`（藏文）、`witness_renderings`（汉文），加载时会校验整个文件，格式错误就报错。
4. 核实、增删条目都会改变 v2 的提示词，相当于换了测量工具：工具指纹改变，缓存键全部改变，开发集对照要重做；冻结之后要 `prereg freeze --amend`。
5. 从此以后，复核表格旁的 `standing_renderings.csv` 也会列出该条目，人工填的 `standing_rendering` 按它核对。

计数、说明和核实人姓名不会出现在提示词里。

### 6.5 尚未做的

- §7 的开发集对照（需要开发集金标准和约 12 次调用）。
- 译名表 Hevajra 一条的专家签名。
- 哨兵可以用 `expect.conventions` 限定惯例（例如 S10 写 `[none]`，防止把 om、hum、ah 当作 `transcribed_name` 免除），但 S10 的范围本身还要研究者细读后决定，所以暂未添加。

改了提示词、schema 和示例，就是**新的测量工具**：工具指纹改变，使用新的缓存键，必须重新过门控。v1 的试跑结果仍可离线重放，不受影响。

---

## 7 验证计划

在开发集上做一次 v1 与 v2 的对照，对比的是同一份盲标金标准：

- 开发集约 450 个单元，约 6 个窗口；每个版本跑 1 次重复，共约 12 次调用，按 API 价格折算约 $6。
- 比较以下三项：
  - 关系一致性（κ）；
  - 两种 D_any 口径与金标准的符合程度；
  - v2 惯例标记的精确率与召回率。
- 另外检查 I.1：上文 §3 预期转为惯例的 14 个单元，是否真的被标出。
- **采用条件**：v2 在关系一致性上不比 v1 差，且惯例标记的精确率 ≥ 0.8。

满足条件后，再冻结预注册（`prereg freeze`）。

---

## 8 文件

| 文件 | 内容 |
|---|---|
| `hevajra_matrix/prompts/collate.v2.md` | v2 提示词全文（英文，与 v1 并列可比） |
| `data/codebook/collate.v2/examples.yaml` | 三个合成示例：序数 vs 总称；固定译名＋反转＋无对应；专名与种子字音写 vs 内容词音写＋真言＋仅见于汉文 |
| `data/codebook/collate.v2/standing_renderings.yaml` | 译名表，含计数，以及考察过但未列入的条目；目前没有已核实条目 |
| `data/codebook/collate.v2/README.md` | 英文说明 |
| `hevajra_matrix/collate/prompt.py` | 版本注册表：两个版本的模板、示例、schema 和译名表 |
| `tests/unit/test_collate_v2.py`、`test_collate_conventions.py`、`test_pipeline_collate_v2.py` | v1 的 sha 钉死；v2 的示例、译名表、V12、共识和流程 |
| `tests/unit/test_review_conventions.py`、`test_evaluation_conventions.py`、`test_stats_conventions.py` 等 | 复核表格、金标准、评估指标和 E1_any_conv |
