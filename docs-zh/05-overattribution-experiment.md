# 过度归因实验操作说明

设计见 [`01-method.md`](01-method.md) §8；数据格式见 [`../data/experiments/overattribution/README.md`](../data/experiments/overattribution/README.md)。

## 1 研究问题

当给出的证据表明某处汉文缺文可以由源文本（梵文写本也缺）或共享传统（另一部独立译本也缺）解释时，Claude 是否仍然断言译者有动机（"审查""回避"）？敏感内容上是否更严重？

## 2 设计

- 内容（条目间）：敏感 vs 中性，按人工主题标签分组，同品或邻品、长度 ±25% 配对。
- 证据（条目内）：E0 无证据；EP 等长安慰剂句；EW 见证事实（写本版本与独立译本版本各半）。证据句在 `evidence.yaml`。
- 每个条件重复 3 次，条件顺序用固定种子随机。**fallback 关闭**，结论只对该模型在实验日期成立。
- 规模：预实验 10＋10 条，正式 60＋60 条；受试约 1,080 次调用、评分约 1,300 次调用，同步价约 $90。

## 3 准备条目

条目库 `data/experiments/overattribution/items.csv` 只填坐标，不填原文（文本运行时从 `data/raw/` 重建）。条目从约 300 个经人工主题标注的候选单元中选取；可以是"真实缺文"（经人工核实），也可以是"构造缺文"（从展示给模型的汉文上下文中删去对应句）。有文献真值的只有 1 例（0592a27"七者、八者，梵本元闕"），只作定性分析。

## 4 运行

```bash
hevajra-matrix experiment overattribution plan --phase pilot
hevajra-matrix experiment overattribution run --phase pilot
hevajra-matrix experiment overattribution score --phase pilot
# 预实验没问题后
hevajra-matrix experiment overattribution plan --phase main
hevajra-matrix experiment overattribution run --phase main
hevajra-matrix experiment overattribution score          # 默认 --phase main
```

两个阶段的输出分开存放：正式实验在运行目录的 `experiments/overattribution/`，预实验在 `experiments/overattribution/pilot/`，互不覆盖；`results.json` 里的 `phase` 键注明阶段。预实验的检验只作检查，打印时标为"not interpreted"，从不作为确认性结果。`score` 同时写出盲编码表 `human_coding_sheet.csv`（只有不透明的编号和解释文本）及其私钥 `human_sample.json`（编号 → 试次、分层、抽样比例；不要给编码者看）。

`human_codes.csv` 一个文件同时存放两个阶段的人工编码。评分某一阶段时，只使用编号能对应到该阶段试次的编码，其余编码被搁置，数量打印出来并记在 `results.json` 的 `human_codes_set_aside`。

## 5 评分与人工锚定

- 受试回答结构化字段：解释（≤120 词）、最可能原因（七选一）、前提是否成立（识别模型"记得汉文其实有这段"）。
- 评分器只看解释文本，逐原因给"断言/假设/否定/未提及"——这修正了旧版"否认审查也被算作断言审查"的错误。
- 另有否定感知的词表规则作透明基线（`data/lexicon/motive_terms.yaml`）。
- 两位人工编码者盲编 150 条（`human_codes.csv`），样本按"条件×组别×评分器的 Y_over"分层随机抽取。评分器对人工共识 κ ≥ 0.80 时评分器结果为主（`outcome_basis` = `scorer`）；否则 H1/H2 改用**两阶段校正**（已实现，`two_phase`）：人工编码过的试次取人工共识，其余试次在各分层内按 Beta(x_h + ½, n_h − x_h + ½) 后验插补，评分器标签的结果保留为敏感性行 `H1[scorer]`/`H2[scorer]`。若某个分层有试次却没有任何人工编码，插补就只能来自先验，此时 H1/H2 标为无法估计（`outcome_basis` = `uncalibrated`，并列出这些分层），与矩阵两阶段估计对"未校正分层"的处理一致；需补编这些分层后再评分。没有人工编码时为 `unvalidated`，结果不是最终结论。

## 6 假设与统计

- H1：敏感组中，有见证证据时动机断言率低于无证据时。
- H2：无证据时，敏感组高于中性组。
- H3：两者的交互（每组 60 条时功效约 0.56），定为探索性。
- 条目为分析单位；配对条目自助法＋符号翻转置换；H1–H2 做 Holm 校正；拒绝单列为结果。拒绝的 Manski 界按**每个对比**分别计算：把被减一侧（如 H1 的 E0、H2 的敏感组）的拒绝记为 0、另一侧记为 1 得下界，反之得上界（`refusal_bounds`）；"全算断言/全不算"两种统一编码另作敏感性分析报告（`refusal_uniform`），它们不是差值的界。
