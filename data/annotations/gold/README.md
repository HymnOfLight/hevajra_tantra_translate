# Gold alignments: file formats

Blind human alignments of reference units to witness segments. Gold is the yardstick for the
Claude collator and for every control (synthesis 5.1-5.2). Annotators never see machine output.

```
data/annotations/gold/<witness>/windows.csv      every gold window of this witness (dev and test)
data/annotations/gold/<witness>/dev.csv          purposive development gold (never used by a gate)
data/annotations/gold/<witness>/test.csv         probability sample of test windows (gate G1)
data/annotations/gold/<witness>/test_second.csv  second annotator on some test windows (human kappa)
```

`<witness>` is a witness id from `data/registry/witnesses.yaml`, e.g. `zh_T0892_song`. The
files are read by `hevajra_matrix.evaluation.gold.load` and written by `gold.save` /
`gold.write_windows` from the verdicts that `review import` makes of filled blind gold sheets
(`gold.from_verdicts`).

Licence rule: these files hold ids and decisions only, no text. The sheets the annotator
filled in (with text) live under `runs/<id>/review/` and are never committed.

## `<set>.csv`

UTF-8 (a byte-order mark is tolerated). Exactly these 13 columns, in any order:

| Column | Content |
|---|---|
| `set` | `dev`, `test` or `test_second`; must equal the file name |
| `window_id` | the window or dev region the row belongs to; must be listed in `windows.csv` |
| `row_type` | `ref` (a reference unit) or `witness_only` (a witness segment with no reference counterpart) |
| `unit_id` | `ref`: the reference unit id, e.g. `D418:17b.6.2`; `witness_only`: `+<segment id>`, e.g. `+T0892:0601c01.2` |
| `fingerprint` | the text fingerprint when annotated; gold is applied to the matrix only while it matches |
| `wit_ids` | witness segment ids, separated by spaces; empty for `no_counterpart`; for `witness_only` the one segment |
| `relation` | `ref` rows: `equivalent`, `paraphrase`, `generalised`, `abridged`, `expanded`, `substitution`, `reversal`, `category_name_omitted`, `transliterated`, `no_counterpart`, or `lacuna` / `unresolved`; empty for `witness_only` rows |
| `polarity_flip` | `true` or `false` |
| `flags` | any of `scope_list`, `uniform_across_list`, `instruction_as_mantra`, `reordered`, `unsure`, separated by `;` |
| `witness_only_kind` | `witness_only` rows: `addition`, `translator_note`, `paratext` or `belongs_elsewhere`; empty for `ref` rows |
| `annotator` | name or initials |
| `date` | `YYYY-MM-DD` |
| `minutes` | minutes spent on the window, when recorded |

Rules checked on load: every relation except `no_counterpart` names at least one witness
segment and `no_counterpart` names none; a unit appears once; a segment is never both
linked and witness-only. Status (PRESENT / PARTIAL / ABSENT) is never stored: it is derived
from the relation (`hevajra_matrix.matrix.status`). Rows decided `lacuna` or `unresolved` are
kept but not scored.

## `windows.csv`

| Column | Content |
|---|---|
| `window_id` | unique id: `w01`, `w02`, ... for test windows; the region id for dev regions (e.g. `I.7`, `II.9-first56`, `II.11-12`) |
| `first_unit`, `last_unit` | first and last reference unit of the window |
| `n_units` | number of reference units |
| `chapter` | reference chapter; two chapters joined by `+` for a dev region across a merged group (e.g. `II.11+II.12`) |
| `seed` | the draw seed for test windows; empty for dev regions |
| `drawn_at` | date of the draw (test windows) |

## How the sets are made

- **Dev regions** come from `config/preregistration.yaml: gold.dev_regions` and are resolved
  by `evaluation.gold.dev_region_units`. Critique A5 asks for I.1, I.7, II.3 (+/-20 units
  around D418:17b.6), II.9 units 1-56, and the tail of II.11 with II.12 as one merged-group
  region. Dev gold tunes prompts and baselines; it never gates. `evaluate --gold dev
  --baselines-only` scores the DP baselines on it without any API key.
- **Test windows** are drawn once by `evaluation.gold.draw_test_windows` with the seed in
  the preregistration: 12 windows of 25 contiguous units, chapters with probability
  proportional to their size, starts uniform, dev units excluded. The annotator searches the
  whole mapped witness chapter +/- 1.
- **Second annotator**: 4 of the test windows, blind to the first annotator.

Scoring any alignment against gold: `evaluation.gold.score`; intervals are window-cluster
bootstraps (`gold.interval_estimates`, `gold.paired_difference`).
