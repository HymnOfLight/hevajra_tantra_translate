# Review verdicts: file format

One CSV file per review batch, under a directory named after the witness id in
`data/registry/witnesses.yaml`: `data/annotations/verdicts/<witness>/<batch>.csv`, e.g.
`zh_T0892_song/verify_2026-10-15.csv`. Each row is one human decision about one matrix row
(a reference unit, or an orphan row `+<segment id>` for witness-only material). The files are
written by `hevajra-matrix review import` (`hevajra_matrix.review.sheets.import_` and
`review.verdicts.save`) and read by `review.verdicts.load`. They are applied to the matrix on
every `build` (`hevajra_matrix.matrix.build.build_cells`).

Under the Sanskrit reference (`data/reference/README.md`) the reference units are Sanskrit
units, so these files live under `data/annotations/by_reference/<reference>/verdicts/` with the
same layout, one directory per witness (T0892 and the Derge); files made against the Derge units are never applied to them.

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

Exactly these 27 columns, UTF-8 (a byte-order mark is tolerated). Lists of ids are separated
by spaces, lists of flags by `;`. Older files still load: without `machine_polarity_flip`
(written before 2026-10-01) that column reads as `false`; without the three convention columns
`machine_convention`, `blind_convention` and `final_convention` (written before collate.v2)
they read as empty, i.e. not annotated. A file with only some of the three convention columns
is refused.

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
| `machine_convention` | the convention of that proposal shown at reveal (`none`, `standing_rendering` or `transcribed_name`; a v1 instrument always gives `none`; empty before reveal, without a machine cell, or for an UNALIGNED one). A convention the code checks did not support is shown on the sheet as `<convention>:unsupported` and stored without the suffix |
| `blind_relation` | the decision made before seeing any machine output |
| `blind_wit_ids` | witness segment ids of the blind decision |
| `blind_convention` | the convention of the blind decision: `none`, `standing_rendering` or `transcribed_name` (a blank sheet cell is stored as `none`; empty = not annotated) |
| `final_relation` | the decision after reveal (resolve: a copy of the blind one, since resolve has no reveal) |
| `final_wit_ids` | witness segment ids of the final decision |
| `final_convention` | the convention of the final decision (resolve: a copy of the blind one) |
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
- Convention (collate.v2), recorded apart from the relation and never in `flags`:
  `standing_rendering` (the unit contains a term of the verified glossary of the witness's
  standing renderings, rendered as listed) or `transcribed_name` (a proper name, seed syllable
  or letter rendered by sound while the rest is rendered by meaning); otherwise `none`. The
  relation is then judged as if the name or term had been translated; a convention never turns
  a missing, reversed or replaced element into `equivalent`. A convention other than `none`
  needs a linked relation (not `no_counterpart`, `lacuna` or `unresolved`, and never on an
  orphan row). `standing_rendering` is accepted only when the unit contains a reference form
  of a verified glossary entry and the linked witness segments contain a rendering of the same
  entry: `standing_renderings.csv` beside the sheets lists the verified entries (the same ones
  the collate.v2 prompt shows; while none is verified, every `standing_rendering` is refused on
  import). The flag `convention_unsupported` is set by the code checks only.

Status is never stored: it is derived from the relation (`hevajra_matrix.matrix.status`). A
decided convention (final columns; gold: blind) reaches the matrix as the cell flag of the same
name, on a grade-A cell; an empty convention adds none.

## How verdicts are used

- **Precedence in the matrix:** gold, then the latest final verdict (by `final_date`), then the
  machine consensus. Verify and audit verdicts count only once their final columns are filled.
- **Stale verdicts:** when a unit's text changes on re-ingest its fingerprint changes; the
  verdict is then never applied but listed in `matrix/stale_verdicts.csv` with its quotes, so
  it can be re-resolved.
- **Estimation (critique A1):** the final columns are primary. The blind columns give the
  pre-registered automation-bias sensitivity estimate, and the blind-to-final revision rate is
  reported per stratum. Assisted (post-reveal) verdicts are never used as G1 gold.
- **Revisions:** a final decision that differs from the blind one in relation, witness ids,
  flags or convention needs a `revised_reason` on the reveal sheet; an empty convention and
  `none` count as the same. A blind decision, convention included, is fixed once revealed.
