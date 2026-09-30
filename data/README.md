# data/

All research data lives here. Values may be in any language (Sanskrit, Tibetan,
Chinese, ...); keys, comments and this README are English.

| Directory | Content | Committed |
|---|---|---|
| `registry/` | `witnesses.yaml` (matrix columns), `concordance.yaml` (chapter concordance v2, many-to-many, unit-range splits) | yes |
| `lexicon/` | Deterministic linguistic resources: cross-lingual anchors, numerals, Derge parser markers, CBETA note classes, negators, variant character forms, motive terms | yes |
| `codebook/` | Topic vocabulary and synthetic few-shot examples for the Claude tasks | yes |
| `sentinels/` | Expert-verified or proposed collation facts, keyed by witness coordinates, checked on every run | yes |
| `annotations/` | Human-authored research data: blind gold alignments, review verdicts, topic labels. Ids and short quotes only | yes |
| `ledger/` | Append-only record of every scoring of the sealed test set | yes |
| `experiments/` | Over-attribution experiment items (coordinates only) and human codes | yes |
| `fixtures/` | Small multilingual inputs for unit tests | yes |
| `raw/` | Fetched source texts (`hevajra-matrix fetch`) with a SHA-256 manifest | **no** |
| `reference/` | Researcher-supplied Sanskrit TSV files (copyright) | **no** (README only) |

## Licence rules

- CBETA (CC BY-NC-SA 4.0), the Esukhia Derge text (licence to be confirmed), Sanskrit
  critical editions (copyright) and the 84000 translation (CC BY-NC-ND 4.0) are never
  committed in full. Committed files hold coordinates, labels, numbers and short quotes
  (at most 30 Chinese characters or 60 Tibetan/Sanskrit characters).
- A unit test rejects runs of more than 60 non-Latin characters under `annotations/`.
- The LLM response cache (`runs/llm-cache/`) contains prompt text and is never committed;
  deposit it privately if a run must be replayable by others.
