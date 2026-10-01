# Data formats

Every machine-readable file the package reads or writes, with the exact column and field
names used by the code. Module paths are relative to `hevajra_matrix/`. For the module map and
data flow see [architecture.md](architecture.md); for LLM request and response shapes see
[llm-tasks.md](llm-tasks.md).

Conventions (from `core/io.py`):

- UTF-8 everywhere. CSV readers accept a byte-order mark; review sheets are written *with* a BOM
  (so spreadsheets show Chinese and Tibetan), machine files without.
- Whole-file writes go to a temporary sibling and are renamed into place.
- JSON is written with `ensure_ascii=False`; JSONL is one object per line.
- In CSV cells, lists of ids are separated by a space and lists of flags or topics by `;`.
- Floats in the matrix CSVs are rounded to 4 decimals.

Where things live:

| Location | Committed | Content |
|---|---|---|
| `config/` | yes | run parameters, Claude settings, pre-registration |
| `data/registry/`, `data/lexicon/`, `data/codebook/`, `data/sentinels/` | yes | research data, YAML |
| `data/annotations/`, `data/ledger/`, `data/experiments/` | yes | human decisions, review plans and their frozen strata, and the test ledger (ids, labels, strata, short quotes only) |
| `data/raw/`, `data/reference/` | **no** | source texts and Sanskrit TSVs (licensed) |
| `runs/<run>/`, `runs/llm-cache/` | **no** | run outputs, the LLM response cache (contain source text) and the spend ledger `llm_spend.jsonl` at the cache root |

A run directory is named `<UTC yyyymmddThhmmssZ>-<git short hash>` (suffix `-2`, `-3` when the
name is taken).

Under the Sanskrit reference (`data/reference/<witnesses.sanskrit_reference>.tsv` present at
ingest) the run aligns two witnesses. Files of the target (T0892) keep the paths below; the
Derge writes the same files under `witnesses/bo_derge_D417_418/` (e.g.
`witnesses/bo_derge_D417_418/matrix/cells.jsonl`). `ingest/`, `manifest.json`,
`llm_audit.jsonl`, `topics/`, `experiments/` and `claude_check.json` are shared
(`pipeline/context.py: SHARED_ENTRIES`).

---

## 1. Run directory

### `manifest.json`

Rewritten after every stage by `pipeline.context.record_stage` (`core.io.write_manifest`).

| Key | Content |
|---|---|
| `package_version` | `hevajra_matrix.__version__` |
| `git_commit`, `git_dirty` | commit of the checkout and whether it had uncommitted changes (null outside git) |
| `python` | Python version |
| `created_at` | UTC ISO time of this write |
| `config_sha256` | config file name -> sha256 of its bytes |
| `prereg_sha256` | sha256 of `config/preregistration.yaml` |
| `inputs` | label (e.g. `raw:cbeta`, `raw:derge`, `external:<name>`) -> `{path, sha256}` |
| `run_id` | the run directory name |
| `offline` | whether the stage ran with `--offline` |
| `stages` | stages run so far, in order of their latest run |
| `instrument_digests` | task (`collate`, `topics`, `components`, `subject`, `scorer`, and `collate:<ref>-<wit>` per extra language pair with its own examples) -> digest (see [llm-tasks.md](llm-tasks.md)) |
| `llm` | summary of `llm_audit.jsonl`: `calls`, `cache_hits`, `cache_misses`, `served_models` (task -> list), `substituted_calls` (task -> count), `est_usd_uncached` |

### `ingest/`

Written by `ingest` (`pipeline/texts.py`).

**`segments_<witness>.jsonl`**: one `core.types.Segment` per line, all kinds, in document order
(`pipeline/store.py: segment_to_dict`). Under the Sanskrit reference also
`segments_<sanskrit reference>.jsonl`: its units with id, start and end the Snellgrove id
(`I.1.p01`), `local_chapter` = `chapter` = `I.1`, kind `prose` or `verse`, and kind `gap` for a unit
flagged `LACUNA` or `ABSENT` (empty text, `extra.flag`; no matrix row). Its presence is what
puts the run in Sanskrit mode.

| Field | Content |
|---|---|
| `id` | coordinate id: `T0892:0592a29.1`, `D418:17b.6.2`; notes `T0892:0592a27.n1` (`core/ids.py`) |
| `witness` | witness id from `data/registry/witnesses.yaml` |
| `lang` | `zh`, `bo`, `sa` |
| `text` | the segment text (NFC) |
| `start`, `end` | witness line of the first and last character (`0592a29`; Derge `17b.6`) |
| `kind` | `prose`, `verse_line`, `verse`, `mantra` (alignable: `ingest.CONTENT_KINDS`); `head`, `meta`, `colophon`, `paratext`, `note` |
| `local_chapter` | witness chapter: `pin1`..`pin20` (T0892), `D417:1`..`D418:12` (Derge); null outside chapters |
| `chapter` | reference chapter `I.1`..`II.12`, set on reference segments by the concordance; null otherwise |
| `fingerprint` | `core.textnorm.fingerprint(text, lang)`: first 12 hex digits of sha1 over `for_quote(text)` |
| `extra` | string map; CBETA notes carry `note_class` and `host` (the content segment the note follows) |

**`footnotes.jsonl`**: every Taisho footnote (`ingest.Footnote`): `n` (footnote number),
`locus` (line), `segment_id` (anchor segment), `zh_lemma`, `sa_text`, `source` (`taisho`, or
`tokyo335` for the readings marked as the Tokyo 335 Sanskrit manuscript), `text` (full note).
Footnotes are evidence only and never enter a T1 request.

**`variants.csv`**: columns `segment_id, locus, reading, alternative, kind`. Derge `{a,b}` edit
marks (kind `orthographic_variant`) and CBETA apparatus readings.

**`witness_meta.json`**: witness -> metadata. Derge: `chapters`, `paratext`, `reviser`,
`revision_statement`, `toh`, `translators`, `witness`. T0892: `chapters`, `glosses` (Taisho
term glosses), `prefix`, `witness`.

**`report.json`**: witness -> ingest counts, read by gate G0. Derge keys: `segments`,
`content_segments`, `kinds`, `chapters`, `chapter_ordinal_mismatches`, `colophon_roles`,
`paratext`, `variants`, `variant_pairs`, `duplicate_ids`. T0892 keys: `segments`,
`content_segments`, `kinds`, `chapters`, `notes`, `note_classes`, `unclassified_notes`,
`footnotes`, `footnote_sources`, `variants`, `duplicate_ids`. Sanskrit reference keys: `segments`,
`content_segments`, `chapters`, `flagged` (`LACUNA`/`ABSENT` -> count), `duplicate_ids`.

**`g0.json`**: `{"passed": bool, "reasons": [str]}`.

**`sentinels.jsonl`**: one sentinel result per line (`evaluation.sentinels.SentinelResult`):
`sentinel_id, check, stage, sentinel_stage, status, blocking, passed, detail`. `stage` is the
stage checked now, `sentinel_stage` the sentinel's own stage.

### `alignments/*.jsonl`

One `core.types.Alignment` per file (`pipeline/store.py`). The first line is a header
`{"source", "reference", "witness"}`; every further line is one `Link`:

| Field | Content |
|---|---|
| `ref_id` | reference unit id; null for witness-only material |
| `wit_ids` | linked witness segment ids; empty for `no_counterpart` |
| `relation` | a `Relation` value (reference links) or a `WitnessOnlyKind` value (witness-only links) |
| `polarity_flip` | bool |
| `confidence` | `high`, `medium`, `low` (T1 only) or null |
| `quotes` | list of `{side: "ref" or "wit", text, segment_ids}` |
| `flags` | sorted list of flags (see [llm-tasks.md](llm-tasks.md), V-checks) |
| `source` | producer, e.g. `claude-opus-5-5:collate.v1:r2`, `claude:consensus`, `dp:zero`, `dp:anchor`, `placebo:shuffled`, `external:<name>` |

Files: `dp_zero.jsonl` (P1), `dp_anchor.jsonl` (B0), `external_<name>.jsonl`,
`claude.r<k>.jsonl` (verified replicate k), `claude.jsonl` (consensus), `shuffled.jsonl` (P2).
Relation values: `equivalent, paraphrase, generalised, abridged, expanded, substitution,
reversal, category_name_omitted, transliterated, no_counterpart`. Witness-only kinds:
`addition, translator_note, paratext, belongs_elsewhere`. Status is never stored on a link.

### `collation/`

| File | Content |
|---|---|
| `r<k>.json` | per replicate: `unresolved` (unit id -> UNALIGNED reason), `diagnostics` (list of `{kind, ref_ids, wit_ids, detail}`), `hints` (links of substituted-model answers, same fields as an alignment link), `overlap` (`{compared, agreed, disagreeing}`, the V10 statistic) |
| `replicates.json` | `{tags, chapters, windows, request_keys}` of the latest `collate` invocation; `build` reads these replicates; `request_keys` (sorted `LLMRequest.key()` of every request) are the measurement calls whose audit lines G2 checks for substitution (a file without them: every collate call counts) |
| `consensus.json` | `{grades: unit -> B, C or X, reasons: unit -> UNALIGNED reason}` |
| `integrity.json` | `replicates, units, quote_failures, quote_failure_rate, missing_units, missing_rate, fleiss_kappa, mean_link_jaccard, class_jaccard` (gate G2) |
| `diagnostics.jsonl` | `{replicate, kind, ref_ids, wit_ids, detail}` for every replicate's diagnostics |
| `dry_run.json` | `calls, cached_calls, calls_to_send, input_tokens, assumed_output_tokens_per_call, projected_usd, token_count_method (count_tokens or heuristic), windows, replicates, budget_usd` |

UNALIGNED reasons: `unassessed`, `refused:<category>` (`refused:unspecified` without a
category), `truncated`, `invalid`, `substituted_model`, `verification_failed`, `no_majority`,
`unmapped_chapter`, and `unresolved_by_human` (a human decided `unresolved`).

### `matrix/`

Written by `build` (`matrix/export.py`). The CSVs hold ids, statuses and numbers only.

**`cells.csv`**: one row per cell; reference units first (reference order), then orphan rows
(witness order).

| Column | Content |
|---|---|
| `unit_id` | reference unit id, or orphan row id `+<witness segment id>` |
| `row_type` | `unit` or `orphan` |
| `witness` | witness id |
| `chapter` | reference chapter (orphans: attributed chapter, critique B15) |
| `status` | `PRESENT`, `PARTIAL`, `ABSENT`, `UNALIGNED`, `LACUNA`, `NA` (orphans are `NA`) |
| `relation` | relation, or witness-only kind for orphans; empty when UNALIGNED |
| `polarity_flip` | `true` / `false` |
| `grade` | `A` (human), `B`, `C`, `X` |
| `reason` | UNALIGNED reason |
| `wit_ids` | linked witness segment ids |
| `flags` | flags |
| `d_len` | log(witness length / reference length) of the linked segments |
| `d_ord` | signed rank displacement within the concordance chapter group |
| `d_lit` | transliteration density of the linked witness text |
| `source` | `gold:dev`, `gold:test`, `verdict:<batch>`, `claude:consensus`, ... |

**`units.csv`**: `unit_id, row_type, chapter, kind, fingerprint, length`.

**`wide_status.csv`**: `unit_id, chapter`, then one status column per witness id.

**`stale_verdicts.csv`**: `unit_id, batch_id, item_id, task, reason, stored_fingerprint,
current_fingerprint, quote_ref, quote_zh`; `reason` is `fingerprint_changed` or `unit_missing`.

**`cells.jsonl`** (human decisions applied) and **`machine_cells.jsonl`** (consensus only, used
for sampling strata): one `Cell` per line with the fields of `cells.csv` except `row_type`
(`pipeline/store.py: cell_to_dict`).

### `evaluation/`

| File | Content |
|---|---|
| `scores.<set>.json` | one per evaluation, never overwritten by another kind: `scores.test.json`, `scores.dev.json`, `scores.<set>_baselines.json` (`--baselines-only`). Keys: `gold_set`, `baselines_only`, `human_kappa`, `sources`: source -> `{counts, metrics, intervals}`. `counts`: `units, windows, excluded, gold_links, gold_null, gold_witness_only, unresolved`. `metrics`: the scalars `link_precision, link_recall, link_f1, null_precision, null_recall, witness_only_recall, witness_only_kind_agreement, status_kappa, dany_kappa, quote_failure_rate, invalid_handle_rate, refusal_rate`, then keyed metrics `<breakdown>:<key>`: `status_agreement:<status>`, `dany_agreement:dev` / `:nondev`, `relation_recall:<relation>`, `refusal_rate:<topic group>` (`evaluation/scores.py: METRICS, BREAKDOWNS`). `intervals`: metric -> an `Estimate` (below) with its 95% window-cluster bootstrap interval. `reliability` (`evaluation/scores.py: confidence_reliability`; null for a source without confidences): `outcome` (`status_correct`), `n` (units with a confidence), `accuracy`, `brier` (nominal probabilities high 0.9, medium 0.7, low 0.5), `brier_constant` (constant predictor at the observed accuracy), `beats_constant`, `table` (label -> `{n, correct, accuracy, nominal}`) |
| `scores.json` | the same content as the scores file of the evaluation that wrote `gate.json` |
| `gate.json` | `level` (0, 1, 2), `confirmatory`, `passed` (gate -> bool), `reasons`, `deferred_null`, `not_estimable` (estimand -> reason), `scope_note`, `gating` (true only for the Claude consensus scored on test gold), `gold_set`, `baselines_only`. Once a gating evaluation exists, dev-gold and baselines-only evaluations write only their `scores.<set>.json` and leave `gate.json`, `scores.json` and `sentinels.jsonl` alone |
| `sentinels.jsonl` | sentinel results of every stage checked (as in `ingest/`), written with `gate.json` |
| `perturbations.json` | `wrong_window` and `deletion` as `{hits, n}`, `negation` (null: not run), `fraction`, `seed`, `windows`. Deletion scores each unit once: the units a chunk shares with the next chunk are scored in the next one only. `negation` is `{hits, n}` after `perturb --negation` (polarity recall over dev-gold `equivalent` pairs negated on both sides; windows keyed `<window>-neg`), null otherwise; never gated |

### Other run files

| File | Content |
|---|---|
| `llm_audit.jsonl` | one line per LLM call (section 2) |
| `topics/prelabels.jsonl` | T3 hints: `unit_id, topics, cues ([topic, cue] pairs), flags, reason` |
| `review/plan_<batch>.csv`, `review/strata_<batch>.json` | runs made before plans were committed only; now committed (section 3), read here as a fallback for a batch with no committed plan |
| `review/` sheets | section 4; contain source text |
| `stats/estimates.json` | estimand name -> `Estimate`: `name, point, lo, hi, n, scope, sources, level, not_estimable`. Names: `E1_any, E1_cov, E1_absent, E1_partial, E2_any`, the blind-column sensitivity `E1_any_blind, E1_cov_blind, E2_any_blind`, the prior sensitivity `E1_<primary>_prior` (primary outcome of `preregistration.yaml: primary_outcome`, Dirichlet prior symmetric in D; only when the final column has draws), `E3` (when computed: `point` = deviating units decomposed, `n` = countable units; the counts are in `details.json: e3`), `E3.phi_V`, `E3.phi_S` (only when E3 is computed), `E4`, `E5`. E5 counts human-verified witness-only rows except those on CBETA `note` segments |
| `stats/details.json` | `revision` (stratum -> `{stratum, n, relation_changed, outcome_changed}`); `manski` (`any`/`cov` -> `[lo, hi]`); `uncalibrated` (`final`/`blind` -> stratum -> unverified units, when a stratum has no phase-2 sample verdict and E1/E2 are not estimable); `ingest_notes` (`{witness_only_rows}`: witness-only rows on CBETA `note` segments, classified at ingest and left out of verification and E5; descriptive); `e4`: `margin` always, `power` (the content of `stats/power.json`) when computed, and when Delta is computed also `permutation_p, equivalent_within_margin, overlap {n_strata, n_overlap_strata, n_units, n_used, dropped}, naive` (Delta on the machine labels), `attenuation` (corrected minus naive), `matched_rd {difference, pairs}`, `misclassification` (list of `{stratum, factor (chapter or tertile), level, n, errors, rate}`), `negative_control` (an `Estimate` of frame vs neutral); `e3` (`pipeline/e3.py`): `manuscripts` (registry manuscripts with readings), `editions_ignored` (reading columns naming an edition), `cowitness`, `cowitness_revised`, `m_min`, `readings_units`, `unknown_units` (reading ids that are not reference units), `estimable`, `reason`, and once a manuscript column exists `units`, `n_deviating`, `counts` (class -> n; `shared_revised` when the co-witness is revised), `by_unit` (unit -> class), `translator_flagged`, `verified_deviating`, `vorlage` and `shared` (`{n, observed, expected, excess, case_rate, base_rate, phi}`), `bound` (A2: `{counts, by_unit, shared, unverified_shared}` with unverified shared units counted as not shared); `distance` (Sanskrit reference only): `{labels, matrix, newick}` over the reference, the Derge and T0892 |
| `stats/power.json` | written by `stats` once topic labels are complete (`pipeline/e4.py: mde_on_labels`): `mde_on_labels` (smallest Delta on the grid detected with `target_power`; null: none), `target_power, p0, chapters, units, exposed, power_curve` (effect -> power), `n_sim`. Gate G4 uses the larger of it and `stats.mde` (the simulated MDE alone when `stats.mde` is unset) |
| `components/components.jsonl` | T2 slot codes: `ref_id, wit_ids, slot, code, ref_quote, wit_quote, polarity_flip, flags, reason` |
| `components/rendering_profile.csv` | `source_quote, rendering_quote, count, n_units, units, codes, slots` (descriptive) |
| `components/diagnostics.jsonl` | `{kind, ref_ids, wit_ids, detail}` |
| `experiments/overattribution/` | the main phase; the pilot writes the same files under `experiments/overattribution/pilot/` |
| `experiments/overattribution/trials.jsonl` | `trial_id, item_id, arm, condition, evidence, replicate, order` |
| `experiments/overattribution/responses.jsonl` | one `TrialOutcome` per trial (`experiments/overattribution/score.py`): `trial_id, item_id, arm, condition, evidence, replicate, omission_origin, status, scorer_status, refusal_category, served_model, explanation, most_likely, premise_ok, word_count, primary, stances, disputes_premise, y_over, y_any, y_uptake, lexical, repeat_primary, repeat_y_over, flags` |
| `experiments/overattribution/results.json` | `ExperimentResults`: `tests` (each `name, role, estimate, ci_low, ci_high, p_value, n, p_holm, not_estimable`), `refusal_bounds` (test -> Manski `[lo, hi]`: refusals coded 0 on one side of the contrast and 1 on the other), `refusal_uniform` (test -> `[all 0, all 1]`), `cells`, `scorer`, `lexical_kappa_y_any`, `served_models`, `outcome_basis` (`scorer`; `two_phase` when the scorer kappa is below the G4 threshold, H1/H2 then two-phase corrected and the scorer-label rows named `H1[scorer]`/`H2[scorer]`; `uncalibrated` when, below that threshold, a stratum condition x arm x scorer Y_over holds measured trials but no human code: H1/H2 and their sensitivity rows are not estimable and the refusal bounds null, as in `stats.twophase.UncalibratedStrata`; `unvalidated` without human codes of this phase), `uncalibrated` (stratum -> uncoded trials), `human_codes_set_aside` (codes in `human_codes.csv` whose id is no trial of this phase, e.g. the other phase's), `scope`, `phase` (`main` or `pilot`) |
| `experiments/overattribution/human_coding_sheet.csv` | blind coding sheet: `response_id` (opaque sheet id), `explanation`, then the code columns of `human_codes.csv` from `coder` on (empty) |
| `experiments/overattribution/human_sample.json` | private key of the sheet: `seed`, `responses` (`sheet_id, trial_id, stratum`), `inclusion` (stratum -> sampling fraction); never shown to coders |
| `claude_check.json` | `passed, status, requested_model, served_model, fallback_used, stop_reason, refusal_category, usage {input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens}, request_id, from_cache` |
| `summary.md`, `status_strip.svg`, `chapter_heatmap.svg` | the report |

---

## 2. LLM cache and audit log

### Cache entries

`llm/cache.py`. One JSON file per request at `<cache root>/<model>/<key[:2]>/<key>.json`, where
`<key>` is `LLMRequest.key()` (sha256 over every request field, including the replicate tag)
and the cache root is `run.yaml: paths.cache` (default `runs/llm-cache`). Every outcome is
cached (ok, refusal, truncated, invalid); exceptions are not. Never commit the cache: entries
hold model output that may quote licensed text.

```
{
 "cache_format": 1,
 "request": {"task", "model", "effort", "max_tokens", "replicate", "allow_fallback", "prompt_sha"},
 "response": {"key", "status", "data", "raw_text", "requested_model", "served_model",
              "stop_reason", "usage": {"input_tokens", "output_tokens",
              "cache_read_input_tokens", "cache_creation_input_tokens"},
              "fallback_used", "refusal_category", "refusal_explanation", "request_id",
              "created_at", "from_cache"}
}
```

`status` is `ok`, `refusal`, `truncated` or `invalid`; `data` is the parsed JSON object when
`ok`, else null. The request header holds no prompt text.

### `llm_audit.jsonl`

`llm/audit.py: audit_record`. One line per completed call, cache hits included; a call that
raises writes no line. No prompt or response text.

| Key | Content |
|---|---|
| `ts`, `run_id` | UTC time; run directory name |
| `task`, `replicate` | `collate`, `components`, `topics`, `subject`, `scorer`, `check`; replicate tag |
| `prompt_sha`, `schema_sha`, `effort`, `key` | request identity |
| `requested_model`, `served_model`, `fallback_used` | substitution signals |
| `status`, `stop_reason`, `refusal_category` | outcome |
| `usage` | `{in, out, cache_read, cache_write}` tokens |
| `request_id`, `from_cache`, `seconds` | provenance and timing |
| `est_usd` | `estimate_usd(usage, pricing)` at the configured model's prices; a substituted call at the per-key maximum of `pricing_usd_per_mtok` and `fallback_pricing_usd_per_mtok` |

The budget stop sums `est_usd` over the spend ledger `<paths.cache>/llm_spend.jsonl`, shared by
all runs: one line `{ts, run_id, task, key, est_usd, from_cache}` per uncached call. The cap is
cumulative across run directories; the per-run audit log stays the provenance record.

---

## 3. Committed human data

Each directory has a README written for annotators; this section lists the exact formats and
the code that enforces them.

Under the Sanskrit reference every path below `data/annotations/` moves to
`data/annotations/by_reference/<reference>/` with the same layout (gold, verdicts and topic labels
are made against the reference units), and each (reference, witness) pair other than
(Derge, T0892) has its own ledger `data/ledger/test_evaluations.<reference>.<witness>.jsonl`
(`pipeline/human_data.py: annotations_dir`, `ledger_path`).

### Gold (`data/annotations/gold/<witness>/`)

Details: [`data/annotations/gold/README.md`](../data/annotations/gold/README.md). Code:
`evaluation/gold.py`, `evaluation/windows.py`.

- **`<set>.csv`**, `set` in `dev`, `test`, `test_second`; exactly the columns `GOLD_COLUMNS`:
  `set, window_id, row_type, unit_id, fingerprint, wit_ids, relation, polarity_flip, flags,
  witness_only_kind, annotator, date, minutes`. `row_type` is `ref` or `witness_only`;
  witness-only rows use the orphan id `+<segment id>`. Relations as above plus `lacuna` and
  `unresolved` (kept, not scored). Flags: `scope_list, uniform_across_list,
  instruction_as_mantra, reordered, unsure`.
- **`windows.csv`**: `window_id, first_unit, last_unit, n_units, chapter, seed, drawn_at`.
  Test windows have a seed; dev regions have none. Written once by `sample windows`.

### Review verdicts (`data/annotations/verdicts/<witness>/<batch>.csv`)

Details: [`data/annotations/verdicts/README.md`](../data/annotations/verdicts/README.md). Code:
`review/verdicts.py` (`COLUMNS`, `load`, `save`, `validate`, `decision`). Exactly 24 columns (a file
written before `machine_polarity_flip` existed, with the other 23, still loads; the column then
reads `false`):

`batch_id, item_id, task, stratum, inclusion_prob, unit_id, fingerprint, instrument_digest,
machine_relation, machine_status, machine_polarity_flip, blind_relation, blind_wit_ids,
final_relation, final_wit_ids, polarity_flip, flags, quote_ref, quote_zh, annotator, blind_date,
final_date, minutes, note`

`task` is `verify`, `audit` or `resolve`. Decision values for a reference unit are a relation,
`lacuna` or `unresolved`; for an orphan row a witness-only kind or `has_counterpart`. Quotes
are truncated on import to `run.yaml: review.quote_max_chars` (zh 30, bo 60, sa 60) and never
exceed 60 characters. The matrix applies the final columns (resolve copies blind to final on
import); estimation uses `final_*` as primary and `blind_*` for the automation-bias sensitivity
estimate (critique A1). `machine_polarity_flip` is the machine's polarity call from the reveal
sheet (`true`/`false`), kept for the E4 misclassification table.

### Review plans and frozen strata (`data/annotations/verdicts/<witness>/`)

Written by `sample verification|audit` (`pipeline/review.py`, `pipeline/human_data.py`) beside
the verdicts they calibrate; ids and strata only, no text. `review.verdicts.load` skips the
`plan_*.csv` files, so a batch name may not start with `plan_` or `strata_`.

- **`plan_<batch>.csv`** (`review/sampling.py: PLAN_COLUMNS`): `item_id, task, unit_id, stratum,
  inclusion_prob, priority`. Strata: `pos:<class>:<topic>` (class one of `absent, reversal,
  substitution, category_name_omitted, relocated, abridged, generalised, transliterated_prose`;
  topic `sensitive` or `other`), `neg:<B|C>:<topic>`, `unresolved`, `pos:witness_only`.
  `inclusion_prob` = n_h / |U_h| (1.0 for census strata). A plan is drawn once (`--force`
  redraws it). Witness-only claims on CBETA `note` segments are never in a plan.
- **`strata_<batch>.json`**: `{families, topics_complete, strata}`: `strata` maps every unit to
  its stratum when the plan was drawn; `families` (`pos`, `unresolved`, `neg`) are the stratum
  prefixes the plan samples, whose units keep the frozen stratum in every later `stats` and
  `evaluate` (topic labels imported later do not move them).

Runs made before plans were committed hold both files under `<run>/review/`; they are read
for any batch that has no committed plan.

### Topic labels (`data/annotations/topics/<reference>.csv`)

Details: [`data/annotations/topics/README.md`](../data/annotations/topics/README.md). Code:
`topics/labels.py` (`LABEL_COLUMNS`). Exactly 9 columns:

`unit_id, fingerprint, topics, prelabel_topics, coder, date, second_topics, second_coder,
second_date`

Topics are from `data/codebook/topics.yaml` (`sexual, female_agent, flesh_food, harm, theft,
impure_substance, bone_corpse, ritual, neutral, frame, mantra_control`); `neutral` must be
alone; unknown topics are rejected with the line number.

### Evaluation ledger (`data/ledger/test_evaluations.jsonl`)

Details: [`data/ledger/README.md`](../data/ledger/README.md). Code: `evaluation/gate.py`
(`ledger_record`, `ledger_append`, `ledger_summary`). Append-only; one line per scoring of the
Claude consensus on test gold, with exactly the keys `ts, instrument_digest, prereg_sha256,
git_commit, metrics, gate` (`gate` is `pass` or `fail` for G1). Scoring dev gold, or test gold
with `--baselines-only`, writes nothing.

### Over-attribution experiment (`data/experiments/overattribution/`)

Details: [`data/experiments/overattribution/README.md`](../data/experiments/overattribution/README.md).
Code: `experiments/overattribution/design.py`, `analysis.py`.

- **`items.csv`**: `item_id, phase, arm, pair_id, chapter, unit_ids, omission_origin,
  zh_context_from, zh_context_to, synthetic_facts, note` (coordinates only; `unit_ids` separated
  by `;`; `phase` `pilot`/`main`; `arm` `sensitive`/`neutral`; `omission_origin`
  `real`/`constructed`). Committed header-only until the bank is filled.
- **`evidence.yaml`**: `schema_version`, `lines` with exactly the keys `EP`, `EW-V`, `EW-S`, each
  `{text, implies}`; `implies` lists scorer classes for Y_uptake.
- **`human_codes.csv`**: `response_id, coder, source_text, shared_tradition, transmission_loss,
  abridgement, content_motive, external_pressure, primary, disputes_premise, date, note`.
  Stances: `asserted, hypothesised, rejected, not_mentioned`; adjudicated rows use coder
  `consensus`.

---

## 4. Review sheets (run directory, never committed)

Written by `review export` under `runs/<run>/review/` (`review/sheets.py`, `gold_sheets.py`,
`topic_sheets.py`), UTF-8 with BOM. Beside every text-bearing sheet,
`witness_text/<local chapter>.txt` holds each witness chapter as `<segment id> TAB <kind> TAB
<text>` lines for searching.

| Sheet | File | Columns |
|---|---|---|
| gold, reference side | `gold/<set>_<window>.ref.csv` | `row, unit_id, fingerprint, locus, kind, text, links, relation, polarity_flip, flags, note` |
| gold, witness side | `gold/<set>_<window>.wit.csv` | `handle, seg_id, locus, kind, text, witness_only` |
| verify, audit (blind) | `<batch>.blind.csv` | `item_id, task, unit_id, fingerprint, locus, ref_text, zh_context_loci, zh_context, blind_relation, blind_wit_loci, blind_flags, note` |
| resolve | `<batch>_resolve.blind.csv` | the blind columns, then `hint_other_model`: the proposal of a substituted model for the unit (`collation/r<k>.json: hints`), as `FROM ANOTHER MODEL (substituted; not the instrument, never measured): <relation> [<witness loci>]`, distinct hints joined by ` \| `; empty without a hint |
| reveal | `<batch>.reveal.csv` | the blind columns, then `machine_relation, machine_polarity_flip, machine_wit_loci, machine_quotes, machine_grade, final_relation, final_wit_loci, final_flags, revised_reason` |
| topics, first coder | `topics_<batch>.csv` | `unit_id, fingerprint, locus, text, context_before, context_after, prelabel_topics, prelabel_cue, topics, note` |
| topics, second coder | `topics_<batch>.second.csv` | the same without `prelabel_topics` and `prelabel_cue` (critique A6) |

Import (`review import`, `review reveal`, `topics import`) drops every text column
(`text, ref_text, zh_context, context_before, context_after, machine_quotes`) and the
`hint_other_model` column, and writes the committed files of section 3. A blind verify or audit
sheet never contains a `machine_*` column.

Export never overwrites a blind or reveal sheet that holds an item without an imported verdict
(blind: no committed blind decision; reveal: no committed final decision) unless `--force` is
given; once every item on it is imported, the next `--hours` chunk replaces it. A reveal sheet
holds only the batch's items not revealed yet. Review plans and strata: section 3.

---

## 5. Registry, sentinels and lexicon (YAML)

Keys and comments are English; values may be in any script (language policy). Loaders reject
unknown keys.

### `data/registry/witnesses.yaml`

Top-level `witnesses:` list; read by `registry.load_witnesses`. Required fields `id, lang,
layer, role, status, title`; optional `source_lang, independence, date, notes, editor,
translator, reviser, source_url, license, coords, unit_scheme`.

| Field | Values |
|---|---|
| `lang` | `sa`, `bo`, `zh`, `txg` (Tangut), `mn`, `en` |
| `layer` | `primary`, `edition`, `indirect`, `pivot` (pivot never enters statistics) |
| `role` | `gold`, `reference_variant`, `target`, `control`, `pivot`, `indirect` |
| `status` | `available`, `license_restricted`, `to_verify`, `not_digitized` |
| `independence` | `independent`, `derived`, `revised`, `unknown` |

### `data/registry/sa_manuscripts.yaml`

`schema_version: 1`, `manuscripts:` list; `registry.load_manuscripts`. Every field required:
`id, description, group, independence (independent | derived | to_verify), shelfmark, source,
access`. Unknown facts are `to_verify`, never guessed.

### `data/registry/concordance.yaml` (schema version 2)

`registry.load_concordance`.

```yaml
schema_version: 2
reference_scheme: sa_snellgrove1959
witnesses:
  <witness id>: {locals: [<local chapter keys in witness order>]}
chapters:
  - ref: II.9                      # reference chapter, I.1 .. II.12
    sa_title: Mantroddharapatala   # optional
    spans:
      <witness id>:
        - local: pin18             # null exactly when the span is absent
          status: verified         # proposed | verified | absent_candidate | absent
          ref_from: "D418:27b.1"   # optional: the span covers only these reference lines
          ref_to: "D418:28a.1"
          evidence: ["..."]        # why the mapping holds (critique B8)
          source: "..."            # who established it
```

The concordance is many-to-many: `refs_for_local("zh_T0892_song", "pin11")` is `[I.11, II.1]`,
`pin20` is `[II.11, II.12]`; II.9 is split between `pin18` and an `absent_candidate` span. It
only proposes the core window; links outside it are reported as relocations.

### `data/sentinels/sentinels.yaml` (schema version 2)

`evaluation.sentinels.load` / `check`; the file header is the full specification.

```yaml
schema_version: 2
sentinels:
  - id: S7_I7_melapaka_gap_shared
    status: proposed          # proposed | verified | retired
    verified_by: null         # required for verified
    source: "..."
    stage: ingest             # ingest | proposal | final; also checked at every later stage
    check: adjacent           # one of the 8 kinds below
    loci: {first: "D417:8a.6", second: "D417:8a.7"}
    quotes: {first: "...", second: "..."}     # short; pick the segment when a line holds several
    expect: {max_between: 0}
```

| Check | Loci keys | Expect keys |
|---|---|---|
| `note_kind` | `note` | `note_class` |
| `paratext` | `from, to` | `count` |
| `adjacent` | `first, second` | `max_between` |
| `relation_in` | `ref_from, ref_to[, wit_from, wit_to]` | `relations[, polarity_flip][, flags]` |
| `status_in` | as `relation_in` | `statuses` |
| `none_status` | as `relation_in` | `status` |
| `linked_local` | as `relation_in` | `local` |
| `witness_only` | `wit_from, wit_to` | `kinds` |

Only `verified` sentinels block a gate (G0 at ingest, G2 at proposal, G3 at final). The
committed file holds 18 sentinels.

### `data/lexicon/`, `data/codebook/`

| File | Read by | Content |
|---|---|---|
| `lexicon/anchors.yaml` | `align/anchors.py` | cross-lingual anchors: `key, type (name, term, mantra, place), sa, bo, zh, exclude, note` |
| `lexicon/numerals.yaml` | `core/lexicon.py` | numeral words per language with exclusions |
| `lexicon/negators.yaml` | `core/lexicon.py` | negators per language with exclusion contexts (V6, T2, P-negation, lexical baseline) |
| `lexicon/variant_forms.yaml` | `core/lexicon.py` | variant character forms folded before quote matching (V4) |
| `lexicon/derge_markers.yaml` | `ingest/derge.py` | colophon, ordinal and front-matter markers |
| `lexicon/notes_zh.yaml` | `ingest/notes.py` | ordered (class, regex) rules for CBETA inline notes; Tokyo 335 marker |
| `lexicon/motive_terms.yaml` | `experiments/overattribution/lexical.py` | motive terms, negation scope, breakers (en, zh, ja) |
| `codebook/topics.yaml` | `topics/codebook.py` | topic definitions; names and groups must equal the code constants |
| `codebook/collate_examples.yaml` | `collate/collator.py` | the 3 synthetic T1 few-shot examples |
| `codebook/components_examples.yaml` | `collate/components.py` | synthetic T2 examples |

Each file documents its own schema in its header comment.

---

## 6. Configuration (`config/`)

All three files are English only, read by `config.load_settings`, and fingerprinted into every
manifest. Each consumer builds its own frozen parameter dataclass with `config.from_mapping`,
which rejects unknown keys.

### `run.yaml`

| Section | Keys | Read by |
|---|---|---|
| `paths` | `data, raw, reference, runs, cache` (relative to the root) | `Settings.path` |
| `sources` | `cbeta_file, derge_file` | `pipeline/texts.py` |
| `witnesses` | `reference, target, derge_toh`, optional `sanskrit_reference` (default `sa_snellgrove1959`): the Sanskrit reference used when `data/reference/<it>.tsv` exists | `pipeline/` |
| `windows` | `max_ref_units, overlap, neighbours` | `collate/windows.WindowParams` |
| `align` | `prior_11, prior_null, prior_12, prior_22, prior_13, length_weight, length_var, anchor_weight, anchor_conflict` | `align/dp.DPParams` |
| `review` | `minutes` (task -> minutes per item), `competence` (task -> languages), `quote_max_chars` (lang -> chars) | `review/sampling.ReviewParams` |

### `llm.yaml`

`model, max_retries, timeout_s, workers, budget_usd, pricing_usd_per_mtok {input, output,
cache_read, cache_write}`, `fallback_pricing_usd_per_mtok` (same keys; optional), and `tasks.<task>` with `effort, max_tokens, replicates, fallback`
plus task-specific keys (`components.pairs_per_call`, `topics.units_per_call`,
`scorer.double_score_fraction`). Tasks: `collate, components, topics, subject, scorer, check`.
See [llm-tasks.md](llm-tasks.md).

### `preregistration.yaml`

| Key | Content |
|---|---|
| `schema_version` | 1 |
| `frozen` | set to true by `prereg freeze` |
| `instrument_digests` | task -> digest, written by `prereg freeze` |
| `primary_outcome` | `any` (D_any) or `cov` (D_cov) |
| `scope` | lines printed beside every number |
| `gold.dev_regions` | list of `{chapter[, around, radius_units][, first_units][, last_units][, id]}` (`evaluation/windows.py`) |
| `gold.test_windows` | `n_windows, width, second_annotator_windows, seed` |
| `gates.g1` | `min_link_f1, min_status_kappa, min_kappa_ratio_to_human, min_witness_only_gold, min_witness_only_recall, min_null_gold, min_null_precision, min_null_recall, beat_controls, n_boot` |
| `gates.g2` | `min_replicate_kappa, max_quote_failure_rate, max_missing_rate, max_wrong_window_false_link_rate, min_deletion_recall` |
| `gates.g3` | `min_audit_per_stratum, max_manski_width` |
| `gates.g4` | `min_manuscripts_per_unit, min_topic_kappa, max_mde, min_scorer_kappa` |
| `verification` | `census_classes, sampled_classes, cap_units, sampled_fraction, seed`, optional `min_per_sampled_stratum` (default 20): every sampled stratum gets at least min(|U_h|, it) units even when the census exceeds `cap_units` (`review/sampling.py`) |
| `audit` | `total, min_per_stratum, grade_c_oversample, seed` |
| `stats` | `n_draws, n_permutations, seed, mde` (`mde` is also the TOST margin of E4; a legacy `tost_margin` is ignored, with a warning when it differs) |
| `experiment` | `items_per_arm, pilot_items_per_arm, replicates, conditions, seed, human_coded_responses` |
| `amendments` | list of `{date, reason, changed, was_frozen}`, appended by `prereg freeze --amend` |

An optional `gates.g0.expected` (witness -> ingest report key -> count) overrides the G0 counts;
the committed file has none, so they default to `evaluation.gate.DEFAULT_G0_EXPECTED` (T0892: 521
notes, 181 footnotes; Derge: 26 variants, 3 paratext segments).

---

## 7. Researcher-supplied Sanskrit files (`data/reference/`, not committed)

Details: [`data/reference/README.md`](../data/reference/README.md). Code: `ingest/sanskrit.py`.

- **`<witness id>.tsv`**: `unit_id<TAB>text`, `#` comments; a unit with empty text and flag
  `LACUNA` or `ABSENT` in the third column. Unit ids follow Snellgrove (`I.1.p01`, `I.5.12`,
  `I.5.12a`). `ingest` reads `<run.yaml: witnesses.sanskrit_reference>.tsv` (default
  `sa_snellgrove1959`) and makes it the reference of the run.
- **`readings/<chapter>.tsv`**: header `unit_id, ms, status, reading, source, note`; `status`
  is `present, absent, variant, illegible, not_collated`. `unit_id` is a reference unit id
  (Snellgrove; Derge segment ids while the Derge is the reference). `ms` is a manuscript of
  `data/registry/sa_manuscripts.yaml`, or an edition's witness id (set aside, never counted);
  anything else stops `stats`. Read by `stats` for E3 (`pipeline/e3.py`).

Collator examples for the Sanskrit pairs are committed files, not licensed text:
`data/codebook/collate_examples.sa-bo.yaml` and `collate_examples.sa-zh.yaml`, in the format of
`collate_examples.yaml` with `reference_lang: sa` (not committed yet; see
[llm-tasks.md](llm-tasks.md)).
