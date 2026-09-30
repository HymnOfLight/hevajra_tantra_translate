# hevajra\_tantra\_translate 代码审查与研究价值评估

Sep 30, 2026 · @Ruifeng Cao

结论：研究问题值得做，但仓库目前产不出可信数字。在真实的 T0892 与德格本上，它报告的“汉译缺失率 18.4%”与把相似度换成随机向量时的结果无法区分；四处原文抽查全部出现系统性误判。代码写得整洁、43 个测试全过，毛病在方法、数据与文献层。

标记：【复现】本机跑代码得到；【原文】我对照 CBETA 与德格原文核过；【文献】打开来源核实；【推断】未经验证的判断。审查对象为 commit 1f9474c。

## 一、真实数据上跑出来的是什么

在真实文本上，"缺失"判定主要由长度统计和先验参数决定，而不是由内容证据决定。我按 `fetch` 的地址取回两份原文（CBETA T18n0892.xml 211,008 字节；Esukhia 德格本第 80 函 3,091,634 字节），完整跑了 `run`：16 秒，3,045 个临时参照单元，20 个汉文品全部映射，与 README 一致【复现】。但矩阵里只有汉译一列，三分分解、见证距离、聚类都没有真正运行；摘要表里的"残余 526（0.184）"只是 ABSENT 加 PARTIAL 的另一种写法。

| 指标 | 数值 | 说明 |
| --- | --- | --- |
| 非空珠中锚点相似度为 0 | 1,698 / 2,237（75.9%） | 四分之三的对齐只靠长度 |
| 1:1 珠中相似度为 0 | 1,434 / 1,771（81.0%） |  |
| "偏移"中 PARTIAL 的占比 | 427 / 526（81.2%） | PARTIAL 只按"长度不足基线一半"判定 |
| PARTIAL 中相似度为 0 | 351 / 427（82.2%） | 这些"部分缺失"没有任何内容证据 |
| 3:1 珠 | 216 珠，648 个单元（在场单元的 23%） | 三行藏文配一个汉文小句，是长度模型的产物 |
| 0:1 珠与矩阵孤儿行 | 70 对 67 | 3 行因坐标相同被覆盖丢失 |

安慰剂对照最能说明问题。我把相似度后端换成"恒为 0"（纯长度对齐）和"不含任何内容的随机向量"（模拟嵌入模型的各向异性，底噪余弦约 0.42），其余参数不动【复现，脚本见附件 `repro_checks.py --placebo`】：

| 相似度来源 | 汉译"缺失率"及品区组 95% 区间 | ABSENT | PARTIAL | ABSENT 集合与默认结果的 Jaccard |
| --- | --- | --- | --- | --- |
| 锚点（默认 B0） | 0.184 \[0.128, 0.238\] | 99 | 427 | 1.000 |
| 恒为 0（纯长度） | 0.108 \[0.082, 0.128\] | 0 | 310 | 0.000 |
| 随机向量，种子 1 | 0.244 \[0.175, 0.318\] | 65 | 635 | 0.025 |
| 随机向量，种子 2 | 0.229 \[0.174, 0.278\] | 60 | 597 | 0.000 |
| 随机向量，种子 3 | 0.235 \[0.172, 0.293\] | 74 | 599 | 0.012 |

三点结论。第一，默认结果的区间与三组纯噪声结果的区间都重叠，头条数字无法与"没有内容信息的流水线"区分。第二，具体哪些单元被判 ABSENT 几乎完全取决于后端，集合几乎不相交（Jaccard 不超过 0.025）。第三，只换后端，点估计就在 0.108 与 0.244 之间移动，跨度 0.136，大于默认结果自身的区间宽度 0.110；纯长度结果的区间与三组噪声结果的区间互不重叠。品区组自助法只量了"抽到哪些品"的波动，没有量对齐误差，而对齐误差才是主导的不确定性来源。

另外，纯长度对齐与锚点对齐之间，只有 558 / 2,765（20.2%）的在场单元被配到完全相同的汉文片段，说明锚点确实在改变对齐，但改变的方向无法验证，因为仓库里没有任何一段人工金标准对齐。

## 二、四处原文抽查：每一处都判错

我挑了四个项目自己最关心的位置，对照 CBETA 与德格原文逐句读，再看流水线给出的单元格状态。四处全部判错，而且错的方向正好会污染"敏感主题残余率"这个主估计量。

| 位置 | 原文实际情况 | 流水线判定 | 后果 |
| --- | --- | --- | --- |
| I.7 十二处表末尾（藏 8a–8b；汉 0592a29–b01） | 藏文第 11、12 类为 dur khrod / nye ba'i dur khrod（尸林、近尸林：海边、园林、池沼）。汉文作"十一者眾所樂處或大海邊，十二者華果園林、清淨池沼"：地点都在，只是没出现"尸林/寒林"这个类名【原文】 | 4 个藏文单元判 ABSENT；整张地名表错位 2–3 格，所有地名珠锚点相似度为 0【复现】 | 在"尸林/骨"这类敏感主题上制造假缺失，并把真正的现象（类名被略去）判成"整句缺失" |
| II.3 "你当杀生…"（藏 17b.6；汉 0596b25–27） | 藏文 khyod kyis srog chags bsad pa dang（你当杀生、妄语、不与取、淫他人妇）。汉文作"一者不應殺害眾生…二者無不與故取…三者無欲邪行…四者無虛妄語"：意义整体反转【原文】 | 4 个单元中 3 个 PRESENT、1 个仅因长度被标 PARTIAL，没有任何反转标记；对齐整体错位 3 句，"杀生"一句配给了"持何等戒"【复现】 | 全经最典型的反转在结构层不可见；L3 的 `component_deviation` 只比槽位有无，也看不见 |
| II.9 与汉文第 18 品 | 汉第 18 品后半（"欲作降伏法者…八面者八解脫所生；一十六臂者一十六空所顯…皮骨脂肉血脈等相即四明妃"）逐句对应藏 II.9 前 56 个单元（27b.1–28a.1）；II.9 其余 125 个单元（不含品末题记）是抽字法（28a.1–29a.2），汉文确无【原文，逐句对读，需专家复核】 | 概表设 II.9 为 pin: null，181 个单元全部 UNALIGNED 并被排除；这批汉文材料被硬塞给只有 49 个单元的 II.8，产生 9 个孤儿行【复现】 | 章级门控把一个错误的概表条目静默传播到矩阵；"II.9 缺失或压缩"两个假说其实几分钟就能读出答案 |
| 全经末尾（汉 0601b–c；藏 30a.1–3） | 汉文在"加持金剛蓮華真言"后另有"復說伽陀曰：若不知空智…自他俱利樂"三偈 12 句；德格本在真言与种子字说明后直接进入卷末题记和译者题记，无此三偈【原文】 | 12 个汉文句被配给藏文前一首偈的末 4 句、真言、种子字说明和 3 条译者题记（9 个 PRESENT、3 个 PARTIAL），无一被识别为汉文独有；II.12 覆盖率 0.963【复现】 | 项目自己的核心例证（docs/01 Q1 引 Isaacson 2022 的"末尾三偈"）在输出里完全消失 |

附带两个观察。其一，II.9 开头藏文 snying rjes gsad par bya（应以悲心杀之）与 skad cig gsod par byed（刹那杀之），汉文作"剎那降伏"，是"杀→降伏"的替换；这类现象只有组件层能记录。其二，汉文 0592a27 有译者夹注"七者、八者，梵本元闕"，是夹注明言梵本原缺，属于 Vorlage 解释的直接证据，但 CBETA 摄取器把所有夹注从正文删掉、只计数量【原文】。

## 三、代码缺陷清单

26 处问题，前 2 处让当前输出不可用，接下来 9 处系统性地偏置核心度量。编号 #N 对应附件 `repro_checks.py` 的检查编号，均已在本机复现。

| 严重度 | 位置 | 问题（证据） | 修法 |
| --- | --- | --- | --- |
| 致命 | `align.py` / `matrix.py` | 状态由长度模型和先验决定；PARTIAL 只看长度比；无任何金标准校验，头条数字与噪声不可区分（§1） | 先做 3–4 品人工金标准对齐，报告珠级 P/R 与 NULL 召回；ABSENT 与 PARTIAL 分开报告 |
| 致命 | 全流水线 | 人工复核回不来：`alignment_review.csv` 只写不读；`evidence` 永远是 C，`prior`、`Unit.topic` 从未赋值；RQ3 主估计量 `matched_contrast` 与规则归因 `attribute` 不被任何命令调用 | 加 `apply_review()` 把人工裁决写回单元格；加主题表载入；把 Δ 接进 `run` |
| 严重 | `anchors.py` | 藏文子串匹配无音节边界：ཤ（肉）命中 218 段而真实音节只有 15 个；སྟོང（千）命中"空性"；རྩ、དབང、གར 同类。词表缺"寒林"（汉文 7 次）、"髑髅"用了简体（原文"髏"），无地名；"日"命中嚩日囉，"三"命中三摩地（#7、#14） | 以 tsheg 为界匹配；用繁体原形；并入已抽取却闲置的 69 条大正藏梵汉对照；Jaccard 加 IDF 权重 |
| 严重 | `ingest/derge.py` | 卷末译者题记（Gayādhara、Śākya ye shes、gZhon nu dpal 三条）被当作 II.12 正文参与对齐；Esukhia 的 `{原字,校字}` 标记 26 处原样留在文本里（#12、#13） | 在最终 rdzogs so 处截断；解析 `{a,b}` 取校字并记录异文 |
| 严重 | `ingest/cbeta.py` | 所有夹注一律删除，包括"七者、八者，梵本元闕"和"以石碌代之""或以白蒿汁代之"这类夹注（#11） | 非音注夹注保留为 `note` 片段，作为 Vorlage/替换的直接证据 |
| 严重 | `matrix.cells_from_beads` | 合并品组里，单元位置按单品算、见证位置按整组算，d\_ord 失真：I.11 0.281、II.1 0.225、II.11 0.196、II.12 0.397，其余各品不超过 0.093（§1 运行结果） | 合并组内统一重算单元位置 |
| 严重 | `attribution.asserts_motive` | 否认动机也算"断言动机"："This is not censorship"、"並非刪略"均返回 True；R6 的 motive\_flag、R7 的过度归因率、R9 全受影响（#1） | 让模型输出结构化字段，再由独立分类器加人工抽检判定；至少处理否定 |
| 严重 | `llm/tasks.py` R7/R8 | Qwen3 默认开启思考，代码没有关；探针只给 36 个 token、反事实 200 个；未闭合的 `<think>` 不会被剥离，探针召回会趋近 0，反事实的正则扫到的是思维链（#8） | 传 `chat_template_kwargs={"enable_thinking": false}`；加大 token 预算；只对最终答案计分 |
| 严重 | `chapters.yaml` + 章级门控 | II.9 设为 pin: null，181 单元被排除；实际上汉第 18 品含 II.9 前 56 单元（§2）。硬门控让概表错误静默传播 | 允许品内切分映射；门控改软约束，或至少输出越界诊断 |
| 严重 | `metrics.decompose` | 同源见证不可计数的单元先计入偏移总数、再被跳过，三类之和小于总数，与 docs/02 "互斥且穷尽"矛盾（#9） | 增设 insufficient 类并单列报告 |
| 严重 | `llm/tasks.component_deviation` | 只比较槽位有无，替换、概括、反转都记为保留；docs/02 §4.6 的 ret/gen/sub/lit/om/add 六码在代码里不存在（#5） | 按槽位比较内容并输出六码；否定翻转单列 |
| 中等 | `registry.chapter_lookup` | 合并品只能映射到最后一个参照品：pin 11→II.1，pin 20→II.12（#2） | 返回一对多映射 |
| 中等 | `attribution.attribute` / `evidence_gate` | UNALIGNED 单元被归为 unexplained\_at\_translation；`min_sanskrit=1` 使证据门几乎恒开，与 docs/02 §9 不符（#3） | 非偏移状态一律 abstain；按文档实现门槛 |
| 中等 | `ids.make_orphan_id` | 同一坐标行上的两个孤儿片段 ID 相同，后者覆盖前者（70 珠只剩 67 行）；合并组的孤儿行不进 `structure()` 统计 | ID 加片段序号；孤儿行按单品归属 |
| 中等 | `llm/embeddings.HybridSimilarityBackend` | 取 max(锚点, w·嵌入)，嵌入有底噪就永不为 0，锚点冲突罚分被静默关闭；NULL 数量随相似度尺度漂移（#6，§1） | 每个后端先用打乱配对做 z 标准化，再重调先验 |
| 中等 | `cli` 的 llm-extract / llm-probe | 读的是 `cells.csv` 的 text 字段，已截断到 80 字；组件抽取在残文上做，探针默认 min\_chars=80 只剩恰好 80 字的格子 | 从原始片段取全文 |
| 中等 | `transfer.VariantTable.explained` | 按位置逐字比较，任何插入（如"也"）之后的异文全部错位漏记，直接影响 docs/05 预期读数 2（#4） | 先做字级对齐再比 |
| 中等 | `llm-counterfactual` 文档命令 | 两个 profile 都指向 localhost:8000，32 GB 放不下两个模型，README 的一次性双模型命令跑不通 | 逐模型分次运行，各自 base\_url |
| 中等 | `data/llm/models.yaml` | `--guided-decoding-backend` 已在 vLLM 0.12 移除，`vllm serve` 会报错；5090 实际需要 vLLM ≥ 0.9.2、bitsandbytes ≥ 0.45.3；MITRA 仓库名应为 `buddhist-nlp/…`，MITRA-E 是 9.24B 参数、3584 维，不是 bge-m3 量级；用 sentence-transformers 加载会漏掉其必需的前缀与末 token 池化；bge-m3 覆盖藏文无依据【文献/源码】 | 按 §4 逐项更新 |
| 中等 | `llm/tasks.TOPICS` | 与 docs/02 §6 的主题集完全不同，也没有 `matched_contrast` 默认需要的 neutral 标签（#10） | 统一为一套词表 |
| 中等 | `transfer/sinitic.py` | 小句由现代标点切分；残缺段跨两个参照小句时，一句被标 LACUNA、另一句被错配为 PRESENT（我构造的马王堆第 1 章样例） | 残缺段允许吸收多句；报告标点来源 |
| 轻微 | `normalize.sa_numerals` | 子串匹配：lakṣaṇa→100000、advitīya→2、kaṣṭa→8、rātri 与 kṣatriya→3（#7）；仅梵文模式受影响 | 按词形边界匹配 |
| 轻微 | `llm/embeddings.py` | 每个候选珠用纯 Python 算均值与余弦：8 维玩具向量已多耗 23 秒，线性外推 1024 维约 50 分钟、4096 维约 3 小时一跑【推断】 | numpy 批量化，缓存前缀和 |
| 轻微 | 版本与清单 | `__version__` 为 0.1.0 而 pyproject 为 0.2.0；manifest 没有代码版本或 git 哈希，与 docs/02 §10 承诺不符 | 写入版本与提交哈希 |
| 轻微 | 打包 | `data/` 不在包内，DATA\_DIR 依赖可编辑安装 | 用 package data 或显式路径参数 |
| 轻微 | `tests/` | 43 个测试全过，但都在手造的理想夹具上；没有一项在真实数据上断言结果 | 把 §2 四处原文事实写成回归测试 |

工程本身不差：模块划分清楚，德格摄取器在真实数据上正确识别了 11 + 12 个品题及其序数，CBETA 坐标保留完整，受限文本一律不入库。问题集中在"信号从哪来"和"结果如何验证"。

## 四、文献与事实核对

19 条中 5 条有实质错误，4 条夸大或范围扩大，另有 2 条大正藏自身的关键事实仓库完全没提。最需要立刻改的是德重的数字：它是项目区分自身与先行研究的依据。

| 仓库说法 | 核对结果 | 来源 |
| --- | --- | --- |
| 德重 2026："275/250 偈，23%" | **错**。IBK 74(2)（2026.3）pp. 838–843：全经 1,100 偈，261 偈（约 24%）有自主规制，其中 243 偈（约 93%）落入八类（性 55、女性 54、肉 19、伤害 33、盗 5、不净物 45、人骨 8、仪礼 52）；手法为省略、改写、音写【文献，本人打开 PDF 核对】 | [论文 PDF](https://researchmap.jp/tokushigehiroshi/published_papers/54360757/attachment_file.pdf) |
| 德重 2026-09-04 学会报告（喜金刚） | 属实：题为「Hevajratantraの漢訳における自主規制について」，书面版未刊，抢先风险真实【文献，出自作者 researchmap】 | [报告列表](https://researchmap.jp/tokushigehiroshi/presentations) |
| Isaacson 2022，pp. 162–186，"已证明"末尾三偈与部分梵本相合而藏无 | 页码错，应为 161–186："Studies in the Transmission of the Hevajratantra (I)"，Almogi 编 *Evolution of Scriptures, Formation of Canons*（Indian and Tibetan Studies 13, Hamburg 2022）。内容主张找不到摘要或书评可核；"汉有藏无"一半我在原文里看到了（§2），"与梵本相合"未核实，"已证明"措辞过强 | [书目](https://sakyaresearch.org/sources/5897) |
| Conlon 2022 暂定校本 | 存在，但不是出版物：saktumiva 上的 TEI 在线本，自称仍在修订；5 个梵文写本（C、K、Na、Nb、P）加两种印本，不含藏汉；II.12 页面目前不存在 | [在线校本](https://mail.saktumiva.org/wiki/conlon/hevajratantra/heta_1.4/htec_1.4) |
| 鈴關宥俊 1938–39，《智山学报》12–13 | 属实：「佛説大悲空智金剛大教王儀軌經の研究：特に梵藏漢三譯對照の結果に就て」，12 号 pp. 133–167、13 号 pp. 152–200。人名读法未能确认 | [12 号 DOI](https://doi.org/10.18963/chisangakuho.1938.12_133)、[13 号 DOI](https://doi.org/10.18963/chisangakuho.1939.13_152) |
| Willemen 1983 "diplomatic effort"；宋译 1054–55 | 书目属实；该短语见于出版社简介。84000 与 Szántó 2015 作 1055 年；据维基百科转引 Willemen，进呈于至和元年末 | [馆藏记录](https://find.library.upenn.edu/catalog/99798593503681)、[Szántó 2015](https://openphilology.eu/publications-peter-daniel-szanto/papers_2015g_hevajratantra.pdf) |
| 84000 英译 Toh 417–418（Mical，v1.0.0，2025，CC BY-NC-ND 4.0）；gZhon nu dpal 即贡译师（1392–1481） | 版本信息属实。84000 经人名规范档把此人对应到 'Gos lo tsā ba gZhon nu dpal（1392–1481），导言本身未论证；题记我在德格 30a.3 核过，措辞是"补译缺漏并校正"。"贡译师童祥"非通行译名，通行作廓（桂）译师宣奴贝 | [84000](https://84000.co/translation/toh417)、[人名档](https://scholar.84000.co/canon/authors-translators/843f7658-9351-4866-8381-0f294bd4ddff) |
| 藏译约 1040 年代，与宋译相隔约十年 | **无依据**：84000 只说 11 世纪，Szántó 说与汉译大约同时；卓弥卒年本有 1043、1072 两说 | 同上 |
| 沈卫荣：宋、西夏、明三种汉译比较 | 存在：「宋、西夏、明三种汉译《吉祥喜金刚本续》的比较研究」，沈卫荣主编《汉藏佛学研究：文本、人物、图像和历史》，中国藏学出版社 2013 | [书目](https://book.douban.com/subject/25793157/) |
| 安海燕：明抄本汉译《吉祥喜金刚本续王》 | **张冠李戴**：同书安文研究国图藏明抄本《〈吉祥喜金刚本续王〉后分注疏》，是对后分的注释，不是根本续译本；`zh_ming_ms` 须重新比定 | 同上 |
| 西夏文 Tang. 326，354–355 为目录号 | 基本属实：Tang. 326 是圣彼得堡架号，Kychanov 1999 目录号 354–355 | [BabelStone 索引](https://www.babelstone.co.uk/Tangut/1999_Pressmark_index.html) |
| 松長 1998:244 "漢民族社会の倫理観"作为宋译伦理审查的先行表述 | 出处为《秘密集会タントラの研究》（著作集 5，法藏馆 1998），该句针对汉译《秘密集会》T885；仓库扩大为宋代密教译经的一般假说【文献，经德重 2026 转引】 | 德重 PDF |
| Sen 2002、Orzech 2006、黄启江 1990/1997、武内孝善 1975/1976、Sinclair 2023 | 均存在。Sinclair 2023 专论宋译阎曼德迦诸续"以音译行审查"，正是 `d_lit` 的直接先例，应正面讨论而不是列为背景 | [Sinclair 2023](https://www.ebsco.com/articles/religion-and-philosophy/c7eafa8f-f677-58ff-974d-785614986470/censorship-through-transliteration-in-song-yamantaka-tantra-translations)、[Orzech 2006](https://eprints.gla.ac.uk/84874) |
| CATSS 的 = / =? / =?? 可直接映射到三分分解 | **夸大**：三者是"希伯来底本不同"这一个解释的把握程度；译法原因另用 =% 与 {…} 编码，没有"与另一译本共享"或"残余"类。最多能说 E\_V 是 = 类的机械化【文献经转述，发表前须对照原件】 | [CATSS 手册](https://accordancefiles1.com/exchange/downloads/documents/MT-LXX_Parallel_Manual.pdf) |
| FOCI 2023 把差异直接视为审查 | 部分属实：Streisand、Wustrow、Houmansadr 标记"可能被删"后人工确认，但没有确认率，也没有译者正常变异模型 | [FOCI 2023](https://petsymposium.org/foci/2023/foci-2023-0001.php) |
| MITRA：`dharmamitra/mitra-qwen3.5`、MITRA-E 与 bge-m3 同量级 | 仓库名错：应为 `buddhist-nlp/mitra-qwen35-translate`（Qwen3.5-9B，Apache-2.0，需 vLLM ≥ 0.27）；MITRA-E 为 `buddhist-nlp/gemma-2-mitra-e`，9.24B 参数、3584 维 | [MITRA 论文](https://arxiv.org/abs/2601.06400)、[模型卡](https://huggingface.co/buddhist-nlp/mitra-qwen35-translate) |
| （未提及）大正藏 T892 的底本 | 校注 0587005 为"【原】明本，東京帝國大學梵本三百三十五號"：汉文这一列本身是经过宋→明传抄的见证，"见证优先"原则在汉文列上没有执行【原文】 | [CBETA XML](https://raw.githubusercontent.com/cbeta-org/xml-p5/master/T/T18/T18n0892.xml) |
| （未提及）大正藏对第十二品的比定 | 校注 0595012 把"熾盛拏吉尼所說成就品第十二"对到东大梵本的 dvitīyakalpasya prathamaḥ paṭalaḥ，与概表"pin 12 = II.2、pin 11 = I.11 + II.1"冲突；至少说明该梵本此处分品与 Snellgrove 不同，I.11 与 II.1 的"合并"可能是底本特征而非译者合并【原文；解释为推断】 | 同上 |
| 老子数据：马王堆乙本"文帝时期（避邦讳、不避恆讳）"；章序"24→21→22" | 自相矛盾：不避"恆"恰恰说明抄于文帝即位（前 180）之前。章序应为 21→24→22，且漏了 80、81 章移到 66 章之后这一最大局部移位；docs/05 预期 2 说乙本差异"大部分来自恆→常、邦→國"，与"乙本避邦"矛盾【推断：依通行说法与文件内部逻辑，本次未能联网核对原书】 | — |

## 五、研究价值评估

问题本身有价值，也有一块真正的空白；但这块空白要靠梵文写本数据和专家精读来填，不是靠当前这套代码。按现状，这是一份写得很认真的研究计划加一个未经验证的原型，还不是研究成果。

| 组成部分 | 先行研究 | 新颖性 | 当前完成度 |
| --- | --- | --- | --- |
| 梵藏汉句级对齐 | MITRA（2026，174 万句对）、SansTib（2022）、DharmaNexus 已收 T892 | 不新；B0（长度 + 约 60 个锚点）远弱于现成工具 | 能跑，未验证 |
| 宋译删改的"发现" | Willemen 1983；Sinclair 2023（音译审查）；德重 2026（T885 全表）及 2026-09 喜金刚报告 | 作为发现已被抢先 | — |
| 汉译作为梵文传本的独立见证 | Karashima 1992 一类方法；Isaacson 2022（按仓库的描述） | 不新 | — |
| 每个缺文单元的三分分解 + 匹配中性对照 | 未找到同类工作 | **真正的新点** | 函数写了但没接入；梵文写本与第二部甘珠尔全缺 |
| LLM 是否过度归因"审查"的反事实实验 | 未找到同类工作 | **新，而且最便宜** | 计分规则目前无效（§3） |
| 以藏译汉的西夏、明代译本分离"目标文化"与"源语" | 沈卫荣 2013 有三种汉译比较 | 设计新 | 可行性存疑：仓库所举明抄本其实是后分注疏；西夏时期汉译的底本与可得性未核 |
| 老子迁移 | 刘笑敢 2006 五本对勘等人工研究 | 低：章序、避讳断代都是已知事实 | 只在三章玩具夹具上跑过；"可证伪预期"多是已知结果，只能当健全性检查 |

**规模与工具不匹配。** 仓库估计全经约 750 偈，德格本实测 3,045 个可对齐单元。一位通梵藏汉的学者几个月就能做完逐句对照；§2 的 II.9 归属和末尾三偈，我对着原文十几分钟就读出来了。仓库却为此设计了九个 LLM 角色和整套 5090 部署。真正的瓶颈是专家工时与写本数据，自动化能省的只是检索和记账。

**三分分解在动手前要先解决四个方法问题。**

1. "存在某个梵文见证缺此单元"就算可由底本解释，这个比例会随见证数单调上升；校勘本也不是独立见证（Farrow–Menon 基本承袭 Snellgrove）。应以写本为单位，并按写本与汉译底本的亲疏加权。
2. "共享"缺零假设。两译若各自独立地以 p、q 的概率省略，偶然共享率就是 p·q，应报告观测减期望。同一难段两列都对不上的相关对齐错误也会抬高共享。
3. 同源见证被污染。德格本经 gZhon nu dpal"补译缺漏"，恰好系统性地补上藏译的缺文，使"残余"偏大。先接入至少一部 Them spangs ma 系甘珠尔（如 Stok、London）分离修订层，否则分解结果不可解释。
4. 单元层看不见最重要的现象。II.3 的反转、II.9 的"杀→降伏"、I.7 的类名略去，都在组件层；组件层恰恰是 600–1,000 小时的人工瓶颈。

**统计设计。** 主估计量 Δ 只有在对齐与标注误差远小于效应时才有意义。§1 显示，仅换相似度后端，估计值就在 0.108 到 0.244 之间摆动，这个量级足以淹没中小效应。要么只用人工核定的状态计算 Δ，要么在金标准样本上估计误判率并做校正。另外，敏感主题按品高度聚集（II.3 密语、II.4、I.7），品内匹配后可用的中性对照可能很少，应先做功效估算。

**抢先风险。** 德重的喜金刚书面版很可能随 IBK 下一期刊出逐偈清单（按其 T885 的节奏推断）。仓库应把区分点落在"分解"和"LLM 过度归因"上；"连续多维偏移向量"对文献学读者的说服力有限，且各维度本身都是标准做法。

**我的打分（10 分制，判断而非测量）。**

| 维度 | 分 | 理由 |
| --- | --- | --- |
| 问题价值 | 7 | 底本差异与译场取舍的区分是真问题，材料窗口正好（Conlon 在线校本、84000 英译、MITRA） |
| 新颖性 | 5 | 分解与 LLM 实验是新的；对齐、删改发现、偏移维度都不新 |
| 当前证据价值 | 1 | 真实数据输出与噪声不可区分，无金标准 |
| 工程质量 | 6 | 结构清晰、可复算、许可意识好；验证缺位，文档与代码多处不符 |
| 按现有资源的可行性 | 3 | 缺梵文写本层、第二部甘珠尔、藏文标注员；LLM 层先于数据层建成 |

**可发表性。** 现状不可投。完成下一节的 P0 后，可投 NLP4DH、LT4HALA、ALP 一类工作坊的方法或资源论文；LLM 过度归因实验单独可成一篇短文。以分解结果为主的文献学论文需要写本层与专家标注，按仓库自己的工时估算是一到两年的量级。

## 六、建议路线

先让流水线证明自己能对齐，再谈矩阵、分解和大模型。顺序如下。

**P0：在报告任何数字之前（约二到四周）**

- [ ] 做 3–4 品人工金标准对齐，建议 I.1、I.7、II.3、II.11–12（覆盖列表、反转、合并、结尾异文），含全部 NULL 与 n:m；报告 B0、纯长度、MITRA-E 三者的珠级 P/R 与 NULL 召回
- [ ] 修 §3 的摄取与度量缺陷：译者题记截断、`{a,b}` 标记、夹注保留、合并组 d\_ord、孤儿 ID、`chapter_lookup`、`decompose` 的 insufficient 类、`attribute` 对非偏移状态弃判
- [ ] 重写锚点词表：藏文按音节为界匹配；补"寒林""髑髏"与二十四处地名；并入已抽取的 69 条大正藏梵汉对照；加 IDF 权重
- [ ] 改概表：II.9 前 56 单元并入汉第 18 品，其余标待定缺失；引用大正藏校注 0595012，讨论 I.11 + II.1 是否为底本分品
- [ ] 摘要里撤下"汉译缺失率"，ABSENT 与 PARTIAL 分列，并附安慰剂基线
- [ ] 改正 §4 的引文，尤其德重的数字、Isaacson 页码、安海燕条目
- [ ] 实现 `apply_review()` 让人工裁决回写矩阵；把 §2 四处原文事实写成回归测试

**P1：数据层（约一到三个月）**

- [ ] 梵文层：联系 Conlon，取得其在线 TEI 校本（5 个写本）的使用许可，作为写本级"金标准分布"，E\_V 按写本计
- [ ] 藏文层：接入一部 Them spangs ma 系甘珠尔（如 Stok 或 London），分离 gZhon nu dpal 补译层
- [ ] 分解加零模型：共享率报告"观测减期望"，并对对齐误差做敏感性分析
- [ ] 从沈卫荣 2013 原文入手，核实西夏时期与明代根本续汉译是否存在、能否取得；取不到就删去 RQ2 与 Q3 的相关设计

**P2：之后**

- [ ] LLM 过度归因实验：关闭 Qwen3 思考；以结构化字段加人工抽检计分；题目只取有写本证据的真值案例（如"七者、八者，梵本元闕"这类夹注）；逐模型分次运行
- [ ] R3、R5、R9 与 5090 部署文档暂缓，等金标准与写本层就位
- [ ] 老子迁移只作方法健全性测试，不作研究贡献申报；修正乙本年代与章序说明

附件 `repro_checks.py` 复现本报告的全部代码结论：在仓库根目录 `pip install -e .` 后运行 `python repro_checks.py <raw目录> --placebo`（约 3 分钟）。

## 来源

原始数据与代码（本机下载、运行）：

- [HymnOfLight/hevajra\_tantra\_translate](https://github.com/HymnOfLight/hevajra_tantra_translate)，commit 1f9474c
- [CBETA T18n0892.xml](https://raw.githubusercontent.com/cbeta-org/xml-p5/master/T/T18/T18n0892.xml)
- [Esukhia 德格甘珠尔](https://github.com/Esukhia/derge-kangyur)，text/080 卷（{D417}–{D419}）

文献与技术资料（德重 PDF 由我本人打开核对，其余由本次会话的检索子代理打开）：

- [德重 2026 论文 PDF](https://researchmap.jp/tokushigehiroshi/published_papers/54360757/attachment_file.pdf)、[德重报告列表](https://researchmap.jp/tokushigehiroshi/presentations)
- [Isaacson 2022 书目](https://sakyaresearch.org/sources/5897)、[Conlon 在线校本](https://mail.saktumiva.org/wiki/conlon/hevajratantra/heta_1.4/htec_1.4)
- [鈴關 1938](https://doi.org/10.18963/chisangakuho.1938.12_133)、[鈴關 1939](https://doi.org/10.18963/chisangakuho.1939.13_152)
- [Willemen 1983 馆藏记录](https://find.library.upenn.edu/catalog/99798593503681)、[Szántó 2015](https://openphilology.eu/publications-peter-daniel-szanto/papers_2015g_hevajratantra.pdf)
- [84000 Toh 417](https://84000.co/translation/toh417)、[84000 人名规范档](https://scholar.84000.co/canon/authors-translators/843f7658-9351-4866-8381-0f294bd4ddff)
- [沈卫荣主编 2013](https://book.douban.com/subject/25793157/)、[BabelStone 西夏目录索引](https://www.babelstone.co.uk/Tangut/1999_Pressmark_index.html)
- [Sinclair 2023](https://www.ebsco.com/articles/religion-and-philosophy/c7eafa8f-f677-58ff-974d-785614986470/censorship-through-transliteration-in-song-yamantaka-tantra-translations)、[Orzech 2006](https://eprints.gla.ac.uk/84874)、[Sen 2002 书目](https://dlbs.liberal.ntu.edu.tw/en/search/search_detail.jsp?seq=278751)、[黄启江 1990 书目](https://dlbs.liberal.ntu.edu.tw/en/search/search_detail.jsp?seq=262478)、[Karashima 1992 书目](https://dlbs.liberal.ntu.edu.tw/en/search/search_detail.jsp?seq=681781)
- [CATSS 平行文本手册](https://accordancefiles1.com/exchange/downloads/documents/MT-LXX_Parallel_Manual.pdf)、[FOCI 2023](https://petsymposium.org/foci/2023/foci-2023-0001.php)
- [MITRA 论文](https://arxiv.org/abs/2601.06400)、[mitra-qwen35-translate 模型卡](https://huggingface.co/buddhist-nlp/mitra-qwen35-translate)、[SansTib 2022](https://aclanthology.org/2022.lrec-1.724)、[DharmaNexus 中的 T892](https://dharmamitra.org/db/zh/ZH_T18_0892/text)
- [Qwen3-32B 模型卡](https://huggingface.co/Qwen/Qwen3-32B)、[vLLM 结构化输出文档](https://docs.vllm.ai/en/latest/features/structured_outputs.html)、[vLLM sm\_120 支持 PR](https://github.com/vllm-project/vllm/pull/19794)、[CINO 论文（XLM-R 与藏文）](https://arxiv.org/abs/2202.13558)

本次联网检索额度已用尽，马王堆甲乙本年代与章序两点未能再核原书，已在 §4 标为推断。
