# 梵文参照文本（不入库）

梵文校勘本受版权保护，仓库只保存偈号坐标，不保存文本。请研究者自备电子文本，放在本目录下，文件名 = `data/witnesses.yaml` 中的见证 `id`：

```
data/reference/sa_snellgrove1959.tsv      ← 默认行标识来源（reference_scheme）
data/reference/sa_tripathi_negi2001.tsv
data/reference/sa_conlon2022.tsv
data/reference/sa_farrow_menon1992.tsv
```

格式（制表符分隔，`#` 开头为注释）：

```
I.1.p01	evaṃ mayā śrutam ekasmin samaye bhagavān …
I.1.1	vajrasattvo bhavet kasmāt …
I.1.2	…
I.2.5		LACUNA      ← 该校勘本报告此处写本脱叶
I.2.6		ABSENT      ← 该校勘本根本没有此偈（另一校勘本有）
```

- 行标识规则见 `docs/02_矩阵与偏移度量.md` §1.1；偈号按 Snellgrove 1959。
- 其他校勘本若偈号与 Snellgrove 不一致，请先做偈号映射表再转成 Snellgrove 编号（映射表本身应保存并公开，它不含受版权文本）。
- 散文命题（`pNN`）按 §1.3 规则切分。
- 存在任一 `sa_*.tsv` 时，流水线自动进入 **gold** 模式：梵文为参照，藏译与汉译都作为被测量的见证列；否则进入 **provisional** 模式，以德格本藏译为临时参照，所有输出标注 `reference_grade=provisional`。

`*.tsv` 已在 `.gitignore` 中，不会被提交。
