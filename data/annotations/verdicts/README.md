# Review verdicts: file format

One CSV file per review batch, under a directory named after the witness id in
`data/registry/witnesses.yaml`: `data/annotations/verdicts/<witness>/<batch>.csv`, e.g.
`zh_T0892_song/verify_2026-10-15.csv`. Each row is one human decision about one matrix row
(a reference unit, or an orphan row `+<segment id>` for witness-only material). The files are
written by `hevajra-matrix review import` (`hevajra_matrix.review.sheets.import_` and
`review.verdicts.save`) and read by `review.verdicts.load`. They are applied to the matrix on
every `build` (`hevajra_matrix.matrix.build.build_cells`).

Gold is not stored here. `review import --task gold` writes blind gold sheets to
`data/annotations/gold/<witness>/<set>.csv` (see `data/annotations/gold/README.md`); gold
rows become verdicts only in memory (`evaluation.gold.to_verdicts`), which is what gives them
gold precedence and makes them visible to G1 scoring. Do not copy gold rows into this
directory: `review.verdicts.load` refuses them.

The review plans live here too: `sample verification|audit` writes `plan_<batch>.csv` (the
sampled items with their stratum and inclusion probability) and `strata_<batch>.json` (every
unit's stratum when the plan was drawn, frozen for estimation) beside the verdicts they
calibrate, so every later run finds them. They hold ids and strata only. Commit them with the
verdicts; never edit them by hand (`--force` redraws a plan). `review.verdicts.load` skips the
`plan_*.csv` files, so a batch name may not start with `plan_` or `strata_`. Witness-only
claims on CBETA notes (classified at ingest) are never sampled.

Licence rule: these files hold ids, decisions and short quotes only. The text the annotator
read is in the review sheets under `runs/<id>/review/`, which are never committed. Quotes are
cut to `config/run.yaml: review.quote_max_chars` (30 Chinese / 60 Tibetan characters) and
never exceed 60 characters.

## Columns

Exactly these 24 columns, UTF-8 (a byte-order mark is tolerated). Lists of ids are separated
by spaces, lists of flags by `;`. Files written before `machine_polarity_flip` was added (23
columns) still load; the missing column reads as `false`.

| Column | Content |
|---|---|
| `batch_id` | the batch (the sheet's file name without `.blind.csv`) |
| `item_id` | `<task>:<unit id>`, unique within the batch |
| `task` | `verify`, `audit` or `resolve` (a file with any other task, including `gold`, is refused on load) |
| `stratum` | sampling stratum fixed when the plan was drawn, e.g. `pos:abridged:sensitive`, `neg:C:other`, `unresolved`, `pos:witness_only` |
| `inclusion_prob` | n_h divided by the size of the unverified pool U_h of the stratum when drawn (1 for census strata); required for verify and audit |
| `unit_id` | reference unit id (e.g. `D418:17b.6.2`) or orphan row id (e.g. `+T0892:0601c01.2`) |
| `fingerprint` | the row's text fingerprint when reviewed; the verdict is applied only while it matches |
| `instrument_digest` | digest of the instrument whose output was revealed (empty before reveal) |
| `machine_relation` | the machine proposal shown at reveal (`UNALIGNED:<reason>` when there was none) |
| `machine_status` | status of that proposal (`PRESENT`, `PARTIAL`, `ABSENT`, `UNALIGNED`, `NA`) |
| `machine_polarity_flip` | `true` when that proposal flagged a polarity flip (from the reveal sheet; used by the E4 misclassification table), else `false` |
| `blind_relation` | the decision made before seeing any machine output |
| `blind_wit_ids` | witness segment ids of the blind decision |
| `final_relation` | the decision after reveal (resolve: a copy of the blind one, since resolve has no reveal) |
| `final_wit_ids` | witness segment ids of the final decision |
| `polarity_flip` | `true` or `false` (true for `reversal`) |
| `flags` | any of `scope_list`, `uniform_across_list`, `instruction_as_mantra`, `reordered`, `unsure` |
| `quote_ref` | short quote of the reference unit (re-resolution hint only) |
| `quote_zh` | short quote of the linked witness text (re-resolution hint only) |
| `annotator` | name or initials |
| `blind_date`, `final_date` | `YYYY-MM-DD` |
| `minutes` | minutes spent on the item, when recorded |
| `note` | the annotator's own words, any language; a revision reason is appended as `revised: ...` |

## Decision values

- Reference unit: a relation (`equivalent`, `paraphrase`, `generalised`, `abridged`,
  `expanded`, `substitution`, `reversal`, `category_name_omitted`, `transliterated`,
  `no_counterpart`), or `lacuna` (the witness is physically damaged or lost here), or
  `unresolved` (the annotator cannot decide; the cell stays UNALIGNED). Every relation except
  `no_counterpart` names at least one witness segment; `no_counterpart` names none.
- Orphan row: a witness-only kind (`addition`, `translator_note`, `paratext`,
  `belongs_elsewhere`), or `has_counterpart` when the claim is wrong (then record the link with
  a verdict on the reference unit it translates).

Status is never stored: it is derived from the relation (`hevajra_matrix.matrix.status`).

## How verdicts are used

- **Precedence in the matrix:** gold, then the latest final verdict (by `final_date`), then the
  machine consensus. Verify and audit verdicts count only once their final columns are filled.
- **Stale verdicts:** when a unit's text changes on re-ingest its fingerprint changes; the
  verdict is then never applied but listed in `matrix/stale_verdicts.csv` with its quotes, so
  it can be re-resolved.
- **Estimation (critique A1):** the final columns are primary. The blind columns give the
  pre-registered automation-bias sensitivity estimate, and the blind-to-final revision rate is
  reported per stratum. Assisted (post-reveal) verdicts are never used as G1 gold.
