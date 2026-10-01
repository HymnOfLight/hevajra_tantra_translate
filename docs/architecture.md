# Architecture

Developer reference for `hevajra_matrix` v0.3. It describes the code as it is on this
branch; anything not built yet is marked **deferred**. The researcher-facing method and plan
are in Chinese under [`docs-zh/`](../docs-zh/README.md). Companion documents:
[data formats](data-formats.md), [LLM tasks](llm-tasks.md), [contributing](contributing.md).

## 1. Purpose

The package builds a **deviation matrix** for the *Hevajratantra*: rows are reference units
(provisionally the Derge Tibetan translation, Toh 417-418, cut at the shad), columns are
witnesses (now the Song Chinese translation, Taisho T0892), and each cell records how the
witness renders the unit: retained, rewritten, abridged, absent, or undecided. Every cell has
an evidence grade. From the matrix it estimates how much of the text deviates (E1, E2), where
witness-only material lies (E5), and whether sensitive content deviates more than neutral
content (E4). The decomposition of deviations by origin (E3) is computed from per-manuscript
Sanskrit readings once the Sanskrit reference is ingested (then the Derge is a second witness and
the co-witness of the Chinese); under the Derge reference it prints `NOT_ESTIMABLE`. A separate
experiment (E6) studies whether Claude over-attributes translator motives.

**Reference modes.** Without a Sanskrit text the reference is the provisional Derge and the only
witness is T0892. When `data/reference/<run.yaml: witnesses.sanskrit_reference>.tsv` exists
(default `sa_snellgrove1959`), `ingest` makes the Sanskrit units the reference and both the
Derge and T0892 aligned witnesses. Every per-witness stage then runs once per witness with the
same functions (`RunContext.for_witness`, `RunContext.each_witness`); the target keeps the plain
run-directory layout and the Derge writes the same layout under `<run>/witnesses/<witness>/`.
The concordance needs no change: its Derge spans map I.n / II.n to D417:n / D418:n.

v0.2 (commit `1f9474c`) decided status from segment lengths and could not tell its headline
rate from a placebo. v0.3 replaces that with *validated, evidence-graded collation*.

## 2. The method in one page

**Validate before measuring.** No machine-derived rate is printed unless the instrument passes
pre-registered gates on a sealed, probability-sampled blind test set *and* beats content-free
controls scored on the same units. Until then the report is level 0 (DESCRIPTIVE): counts
labelled "unvalidated instrument output", printed beside the controls.

**Claude proposes, code verifies, humans decide.**

1. *Claude proposes.* Claude Opus 5.5 (`claude-opus-5-5`) reads one reference chunk against the
   full Chinese text and returns, for every unit, the linked witness segments, a relation from
   a closed list, a polarity flag and verbatim quotes (task T1). Three replicates are run.
2. *Code verifies.* Pure functions check every handle and every quote (V1-V11). A failed check
   never becomes a silent PRESENT or ABSENT: the unit becomes UNALIGNED with a reason, or keeps
   its link with a flag. Replicates are combined by a two-thirds vote; agreement and flags give
   the grade (B, C, X).
3. *Humans decide.* Annotators make blind gold for validation, then verify a census or sample
   of machine positives and audit a stratified sample of machine negatives, first blind and
   then after seeing the machine output. Gold and every final human decision are re-applied on
   every build (grade A) while the unit's text fingerprint still matches. Witness-only claims on
   CBETA `note` segments are not verified and not counted in E5: ingest already classifies every
   note deterministically (`review/sampling.py: note_rows`; `stats` reports their number).

**Status comes from the relation only** (`matrix/status.py`); length never decides it. Prevalence
is estimated by a two-phase posterior-predictive imputation over the human verification
strata, so machine error is corrected rather than assumed away. No LLM attributes motives
inside the pipeline: relations describe text, never causes.

## 3. Package and module map

Python >= 3.10, stdlib plus PyYAML; `anthropic` is an optional extra imported lazily in one
module. Every module below is English only (enforced, see [contributing](contributing.md)).

### Shell

| Module | Role | Public entry points |
|---|---|---|
| `cli.py` | argparse command line, error reporting | `main(argv)`, `build_parser()` |
| `__main__.py` | `python -m hevajra_matrix` | - |
| `config.py` | locate and read `config/*.yaml`, sha256 of each | `load_settings(root)`, `find_root()`, `from_mapping(cls, data, where)`, `Settings`, `ConfigError` |
| `prereg.py` | instrument digests, `prereg freeze` | `instruments(settings)`, `instrument_digests(settings)`, `freeze(root, amend)` |
| `registry.py` | witnesses, Sanskrit manuscripts, chapter concordance v2 | `load_witnesses`, `load_manuscripts`, `load_concordance`, `Concordance.locals_for / refs_for_local / core_window / assign_reference_chapters` |

### Pipeline stages (`pipeline/`)

| Module | Stages | Notes |
|---|---|---|
| `pipeline/__init__.py` | `run` | chains ingest, baselines, collate, build, evaluate, stats, report as far as data and key allow; documents the run-directory layout |
| `pipeline/context.py` | - | `RunContext` (reference mode, `witnesses`, `for_witness`, `each_witness`, per-witness `path`), `StageError`, `new_run_dir`, `latest_run_dir`, `make_client`, `load_texts` / `Texts`, `all_segments`, `scope_lines`, `record_stage` (manifest) |
| `pipeline/store.py` | - | JSON forms of segments, alignments, collations, cells, estimates, gate reports, sentinel results (pure `*_to_dict` / `*_from_dict` pairs) |
| `pipeline/texts.py` | `fetch`, `ingest`, `baselines`, `baselines_import` | Claude-free; G0 and ingest-stage sentinels; ingest loads the Sanskrit reference when its TSV exists |
| `pipeline/instrument.py` | `collate` (incl. `--dry-run`), `perturb` (incl. `--negation`), `components_stage`, `claude_check` | every stage that calls Claude except topics and the experiment; `collate_digest` of the run's language pair |
| `pipeline/measure.py` | `build`, `evaluate` | Claude-free; G2-G4 facts (`_integrity`, `_calibration`, `design_mde`), ledger deduplication (`already_ledgered`) |
| `pipeline/results.py` | `stats`, `report` | Claude-free; `witness_tree` (distances and UPGMA once three texts share the reference) |
| `pipeline/e3.py` | - | E3 for `stats` and G4: readings, manuscript columns (editions set aside), co-witness, A2 bound (`e3_inputs`, `e3_stats`, `e3_reason`) |
| `pipeline/e4.py` | - | E4 for `stats`: Delta with its diagnostics (`contrast_with_diagnostics`) and the MDE simulated on the real topic labels (`mde_on_labels`) |
| `pipeline/review.py` | `topics_prelabel`, `sample`, `review_export`, `review_import`, `review_status` | human-work stages |
| `pipeline/human_data.py` | - | reads committed gold, verdicts, topic labels, review plans and frozen strata (committed beside the verdicts; run-directory fallback for old runs); `annotations_dir` (per reference) and `ledger_path` (per reference and witness pair) |
| `pipeline/experiment.py` | `experiment_plan`, `experiment_run`, `experiment_score` | the over-attribution experiment |

### Core (`core/`, no I/O except `core/io.py`)

| Module | Role | Public entry points |
|---|---|---|
| `core/types.py` | shared frozen value types (frozen contract) | `Segment`, `Quote`, `Link`, `Alignment`, `Diagnostic`, `Cell`, `Verdict`, `Estimate`, `InstrumentId`; enums `Status`, `Relation`, `WitnessOnlyKind`, `Grade`, `OutcomeClass`; `REASON_*` |
| `core/ids.py` | coordinate-derived ids | `segment_id`, `note_id`, `orphan_row_id`, `orphan_source`, `assign_ids`, `id_kind`, `parse_segment_id`, `parse_snellgrove_id`, `sort_key`, `REF_CHAPTERS` |
| `core/textnorm.py` | tokenisation, quote keys, fingerprints, numerals | `chunks`, `for_quote`, `quote_in`, `fingerprint`, `fold_variants`, `find_terms`, `bo_syllables`, `length`, `numerals` |
| `core/translit.py` | transliteration density (V11 and `d_lit`) | `transliteration_charset`, `transliteration_density` |
| `core/lexicon.py` | loaders for `data/lexicon/*.yaml` | `load_numerals`, `load_negators`, `load_variant_forms`, `read_lexicon` |
| `core/io.py` | YAML/CSV/JSONL I/O, atomic writes, git, manifest | `read_yaml`, `read_csv`, `write_csv`, `read_jsonl`, `write_jsonl`, `append_jsonl`, `sha256_file`, `git_commit`, `write_manifest` |

### Ingest (`ingest/`)

| Module | Role | Public entry points |
|---|---|---|
| `ingest/__init__.py` | result types | `IngestResult`, `Footnote`, `Variant`, `CONTENT_KINDS` |
| `ingest/cbeta.py` | CBETA TEI P5 (T0892): clauses, mantras, notes, footnotes, apparatus, glosses | `parse(path, witness, data_dir)` |
| `ingest/derge.py` | Esukhia Derge volume text: shad units, chapters, paratext, `{a,b}` marks | `parse(path, toh, witness, data_dir)`, `load_markers` |
| `ingest/notes.py` | 7 classes of CBETA inline notes; footnote source | `classify`, `load_rules`, `footnote_source`, `NOTE_CLASSES` |
| `ingest/sanskrit.py` | researcher-supplied Sanskrit TSVs | `load_reference` (read by `ingest`), `load_readings` (read by `stats` for E3; unit ids are Snellgrove ids or, under the Derge reference, Derge segment ids) |

### Baselines and controls (`align/`)

| Module | Role | Public entry points |
|---|---|---|
| `align/anchors.py` | cross-lingual anchor keys (whole-syllable Tibetan, no 1-character Chinese forms, exclusion contexts) | `load_anchor_lexicon`, `extract` |
| `align/similarity.py` | `Similarity` protocol | `ZeroSimilarity`, `AnchorSimilarity.fit`, `idf_weights`, `weighted_jaccard` |
| `align/dp.py` | monotone bead DP, controls P1 (`dp:zero`) and B0 (`dp:anchor`) | `align`, `align_groups`, `DPParams.from_config` |
| `align/placebo.py` | control P2 (`placebo:shuffled`) | `shuffle_alignment` |
| `align/external.py` | TSV import/export of external alignments (critique A4) | `load_tsv`, `write_tsv` |

### LLM layer (`llm/`)

| Module | Role | Public entry points |
|---|---|---|
| `llm/client.py` | request/response values and the protocol (frozen contract) | `LLMRequest`, `LLMResponse`, `Usage`, `LLMClient`, `LLMError` and subclasses |
| `llm/anthropic_client.py` | the only module that imports `anthropic` (lazily) | `AnthropicClient.complete / count_tokens`, pure `build_kwargs`, `to_response`, `parse_answer`, `map_sdk_error` |
| `llm/cache.py` | content-addressed response cache, offline replay | `CachedClient(inner, root, offline)` |
| `llm/audit.py` | audit log (no text) and budget stop | `AuditedClient`, `estimate_usd`, `recorded_spend` |
| `llm/schema.py` | strict JSON schemas without a jsonschema dependency | `strict`, `for_api`, `validate` |
| `llm/fake.py` | deterministic client and response builders for tests | `FakeClient`, `ok_response`, `refusal_response`, `truncated_response`, `invalid_response`, `substituted_response` |
| `llm/check.py` | `claude-check` | `build_check_request`, `run_check` |

### Collation (`collate/`): T1 and T2

| Module | Role | Public entry points |
|---|---|---|
| `collate/windows.py` | reference chunks + full witness text + core window; prompt-local handles | `plan`, `Window`, `WindowParams` |
| `collate/collator.py` | T1 requests, parsing, thread pool; examples per language pair | `build_request`, `build_requests`, `collate`, `parse`, `SCHEMA`, `TaskSettings`, `load_examples`, `examples_file`, `example_pairs` |
| `collate/verify.py` | V1-V8, V11 on one window | `verify`, `Collation`, `CheckLexicon.load`, `quote_match` |
| `collate/merge.py` | V9, V10, chapter and text merge | `verify_all`, `merge_chapter`, `merge_text`, `compare_overlaps` |
| `collate/consensus.py` | replicate vote, grades, Fleiss kappa | `consensus`, `agreement` |
| `collate/perturb.py` | perturbation placebos and their scores | `wrong_window`, `delete_segments`, `negated_pairs`, `remove_negators`, `false_link_rate`, `deletion_recall`, `negation_recall` |
| `collate/components.py` | T2 component coder (descriptive only): pair selection, run loop, outputs; re-exports the two modules below | `select_pairs`, `code_components`, `verify`, `rendering_profile`, `invention_rate` |
| `collate/components_request.py` | T2 settings, schema, batches with full segment texts, prompts, few-shot examples | `ComponentTaskSettings`, `plan_batches`, `build_requests`, `load_system`, `load_examples` |
| `collate/components_verify.py` | T2 code checks C1-C5 | `SlotCode`, `verify`, `verify_answer` |

### Matrix, review, evaluation, statistics, topics, report

| Module | Role | Public entry points |
|---|---|---|
| `matrix/status.py` | relation -> status -> outcome: the only status derivation (frozen contract) | `RELATION_STATUS`, `status_of`, `outcome_class`, `deviates`, `cell_outcome`, `cell_deviates`, `RELATION_PRIORITY` |
| `matrix/build.py` | cells with precedence gold > final verdict > consensus; `d_len`, `d_lit`, `d_ord`; orphan rows | `build_cells`, `StaleVerdict` |
| `matrix/export.py` | `cells.csv`, `units.csv`, `wide_status.csv`, `stale_verdicts.csv` | `write_matrix` |
| `review/sampling.py` | verification and audit plans, review queue, coverage | `draw_verification`, `draw_audit`, `queue`, `coverage`, `machine_strata`, `note_rows`, `write_plan` / `read_plan` |
| `review/sheets.py` | blind, reveal and resolve sheets (resolve: substituted-model hint column); import strips text | `export`, `import_`, `hint_text` |
| `review/gold_sheets.py` | blind gold sheets | `export_gold`, `import_gold` (via `sheets`) |
| `review/topic_sheets.py` | topic sheets (second coder without prelabels) | `export_topics`, `import_topics`, `merge_topic_labels` |
| `review/verdicts.py` | committed verdict CSVs | `load`, `save`, `validate`, `decision`, `COLUMNS` |
| `evaluation/gold.py` | gold format and scoring of any `Alignment` | `load`, `save`, `score`, `human_kappa`, `from_verdicts`, `to_verdicts` |
| `evaluation/scores.py` | per-unit scores and metrics; confidence reliability | `AlignmentScores`, `METRICS`, `interval_estimates`, `confidence_reliability`, `CONFIDENCE_PROBABILITY` |
| `evaluation/resample.py` | kappa, window-cluster and paired bootstrap, McNemar | `cohen_kappa`, `window_bootstrap`, `paired_difference`, `mcnemar` |
| `evaluation/windows.py` | dev regions and test-window draw | `dev_region_units`, `draw_test_windows` |
| `evaluation/sentinels.py` | 8 sentinel check kinds at 3 stages | `load`, `check`, `applicable` (alignment checks only on their own text pair), `text_prefixes` |
| `evaluation/gate.py` | G0-G4, report level, confirmatory flag, ledger | `evaluate`, `check_g1`, `GateSpecs.from_prereg`, `ledger_append`, `ledger_summary` |
| `stats/twophase.py` | strata, posterior-predictive draws, E1/E2, Manski bounds, revision rate | `stratum_of`, `draws`, `prevalence`, `manski`, `revision_rate`, `uncalibrated_strata`, `UncalibratedStrata` |
| `stats/contrast.py` | Delta (E4), permutation test, TOST, matching, diagnostics; length strata are mid-rank tertiles | `delta`, `tertile_strata`, `permutation_p`, `tost`, `matched_rd`, `overlap_diagnostics`, `misclassification_table` |
| `stats/decompose.py` | E3 classes, O / E / O-E, excess fractions, NOT_ESTIMABLE gate | `decompose`, `excess`, `e3`, `not_estimable_reason` |
| `stats/power.py` | planning simulations | `simulate_delta`, `simulate_experiment` |
| `stats/distance.py` | witness distance and UPGMA (>= 3 witnesses; `stats` writes it under the Sanskrit reference) | `witness_distance`, `distance_matrix`, `average_linkage_newick` |
| `topics/` | codebook, human labels and agreement, T3 pre-labeller | `load_codebook`, `topic_group`, `load_labels`, `human_agreement`, `prelabel_agreement`, `build_request`, `parse_prelabels` |
| `report/markdown.py` | level-gated `summary.md` | `render(inputs, gate)`, `ReportInputs` |
| `report/sections.py` | E3 section; confidence reliability and P-negation section | `e3`, `reliability` |
| `report/svg.py` | status strip and chapter heatmap | `status_strip`, `chapter_heatmap` |

### Experiment (`experiments/overattribution/`)

| Module | Role | Public entry points |
|---|---|---|
| `design.py` | items (coordinates only), evidence lines, seeded trial schedule | `load_items`, `load_evidence`, `trials` |
| `run.py` | T4 subject and T5 scorer requests and the run loop | `subject_request`, `scorer_request`, `run_trials` |
| `score.py` | answer verification and outcomes | `parse_subject`, `parse_scorer`, `trial_outcome` |
| `lexical.py` | negation-aware lexical motive baseline (comparison only) | `load_motive_lexicon`, `lexical_motive` |
| `analysis.py` | H1/H2 with Holm, exploratory H3, contrast-wise Manski refusal bounds, two-phase correction below the G4 scorer kappa (not estimable while a stratum has no human code) | `analyse`, `two_phase`, `refusal_bounds`, `uncalibrated_strata` |
| `human.py` | human sample (condition x arm x scorer Y_over), blind coding sheet with opaque ids, human codes, scorer agreement; codes of the other phase are set aside | `human_sample`, `write_coding_sheet`, `write_sample`, `load_human_codes`, `resolve_codes`, `scorer_agreement` |

The experiment depends only on `core`, `llm` and the topic codebook and never feeds the matrix.

### Prompts

`hevajra_matrix/prompts/{collate,components,topics,subject,scorer}.v1.md`, English, shipped as
package data. Few-shot examples are synthetic and live in `data/codebook/`.

## 4. Data flow

Stages are plain functions `stage(ctx, ...)` with one CLI subcommand each. They communicate
only through files in a run directory `runs/<UTC>-<git7>/` (no DAG engine). A missing input
raises `StageError` naming the command that produces it (exit status 2).

```
fetch    -> data/raw/ (T18n0892.xml, derge_rgyud_bum_nga.txt, manifest.json)
ingest   -> ingest/segments_<witness>.jsonl, footnotes.jsonl, variants.csv,
            witness_meta.json, report.json, g0.json, sentinels.jsonl
  baselines              -> alignments/dp_zero.jsonl (P1), dp_anchor.jsonl (B0)
  baselines import --tsv -> alignments/external_<name>.jsonl
  collate (T1, k reps)   -> alignments/claude.r<k>.jsonl, collation/r<k>.json,
                            collation/replicates.json (incl. the request keys), llm_audit.jsonl,
                            cache entries, <cache>/llm_spend.jsonl
    build                -> alignments/claude.jsonl (consensus), alignments/shuffled.jsonl (P2),
                            collation/consensus.json, integrity.json, diagnostics.jsonl,
                            matrix/cells.csv, units.csv, wide_status.csv, stale_verdicts.csv,
                            matrix/cells.jsonl (human decisions applied), machine_cells.jsonl
      perturb (T1)       -> evaluation/perturbations.json (--negation: also P-negation)
      components (T2)    -> components/components.jsonl, rendering_profile.csv, diagnostics.jsonl
      sample verification|audit -> data/annotations/verdicts/<witness>/plan_<batch>.csv and
                            strata_<batch>.json (committed; ids and strata only)
  topics prelabel (T3)   -> topics/prelabels.jsonl
  sample windows         -> data/annotations/gold/<witness>/windows.csv (committed)
  review export          -> review/ sheets (text; never committed)
  review import|reveal   -> data/annotations/{gold,verdicts,topics}/ (committed; no text)
evaluate -> evaluation/scores.<set>[_baselines].json; when gating (test gold, Claude scored):
            scores.json, gate.json, sentinels.jsonl and one line in
            data/ledger/test_evaluations.jsonl (not appended for an identical re-scoring)
stats    -> stats/estimates.json, stats/details.json (incl. e3, distance), stats/power.json (complete topic labels)
report   -> summary.md, status_strip.svg, chapter_heatmap.svg

experiment overattribution plan|run|score [--phase pilot]
         -> experiments/overattribution/ (pilot: experiments/overattribution/pilot/):
            trials.jsonl, responses.jsonl, results.json, human_coding_sheet.csv, human_sample.json
claude-check -> claude_check.json
every stage  -> manifest.json (rewritten after each stage)
Sanskrit reference: baselines ... report run once per aligned witness (`--witness` picks one);
                    the Derge's files go to witnesses/bo_derge_D417_418/<same paths>
```

`ingest`, `run` and `claude-check` create a new run directory; every other command uses the
latest one unless `--run-dir` is given. Committed human data (`data/annotations/`,
`data/ledger/`) is shared by all runs, including the review plans and their frozen strata
(runs made before plans were committed kept them under `review/`; they are still read for any
batch with no committed plan). The LLM response cache (`run.yaml: paths.cache`,
default `runs/llm-cache/`) is shared by all runs and makes any finished run replayable with
`--offline`. Exact file formats are in [data-formats.md](data-formats.md).

Order of a research campaign (a checklist, not code): ingest and G0; dev gold blind; prompt
and effort development on dev gold; `sample windows`; blind test gold; `prereg freeze`; full
collation with replicates, controls and perturbations; `evaluate` once on test gold (ledgered);
topic labelling (before sampling: strata freeze when a plan is drawn); verification and audit
samples; blind-then-reveal review (`review export --hours` chunks: export, fill, import, repeat);
`stats`, `report`.

## 5. Key invariants

1. **Status is derived in one place.** `matrix/status.py` maps relation (plus polarity flag and
   reference kind) to status and outcome class. A `Link` never stores a status; gold, verdict
   and machine cells all go through the same table. Length enters only the descriptive `d_len`.
2. **Stable coordinate ids and fingerprints.** Segment ids come from witness coordinates
   (`T0892:0592a29.1`, `D418:17b.6.2`, notes `T0892:0592a27.n1`, orphan rows
   `+T0892:0601c01.2`), never from a running counter, so they survive re-ingest. Every segment
   carries `fingerprint` (12 hex digits of sha1 over the quote key). Human decisions are keyed
   by (id, fingerprint) and applied only while the fingerprint matches; stale ones are listed in
   `matrix/stale_verdicts.csv`, never silently applied.
3. **One `Alignment` type.** Claude's replicates and consensus, the DP baselines, the shuffled
   placebo, imported external alignments and human gold are all `core.types.Alignment`, so
   every aligner is scored by the same code on the same units. Baselines and placebos never
   populate the matrix.
4. **Substituted-model output is never measured.** A response whose served model differs from
   the requested one, or that carries a fallback signal, has `LLMResponse.substituted_model`
   True and `usable` False. T1 turns the window's units UNALIGNED(`substituted_model`) and keeps
   the parsed proposal only as a reviewer hint (the resolve sheet's `hint_other_model` column);
   T2 marks such codes `substituted_model` and leaves them out of the rendering profile; T3
   shows them as sheet hints but never commits them; the experiment excludes them from every
   statistic. Gate G2 fails while any collate request that fed the consensus was substituted.
5. **No LLM attributes motives in the pipeline.** Pipeline schemas (T1, T2, T3) have no
   free-text fields; relations describe text, never causes; prompts never ask the model to show
   or explain its reasoning (tests assert both). Motive attribution is only the *object* of the
   over-attribution experiment.
6. **Blindness by construction.** A T1 request is rendered from a `Window` holding one reference
   and one witness, with handles instead of coordinates and no footnotes, Sanskrit or topic
   labels. T3 accepts reference segments only. T5 sees only the subject's explanation. Gold and
   blind review sheets carry no machine output; the second topic coder's sheet has no prelabel.
7. **Human verdicts flow back on every build.** `build` re-applies gold and final verdicts each
   time; precedence is gold > latest final verdict > Claude consensus.
8. **Missing numbers are named, never zero.** An estimand that cannot be computed is an
   `Estimate` with `not_estimable` set, printed as `NOT_ESTIMABLE: <gate>: <reason>`.
9. **Licence boundary.** Committed files hold ids, labels, numbers and short quotes only
   (<= 30 Chinese / 60 Tibetan or Sanskrit characters in verdicts). Source texts, sheets, run
   outputs and the LLM cache live under gitignored `data/raw/` and `runs/`. The audit log holds
   no prompt or response text.
10. **Language policy.** Code, comments, prompts, config and Markdown outside `docs-zh/` are
    English; multilingual material lives in `data/`. Enforced by
    `tests/unit/test_language_policy.py` (see [contributing](contributing.md)).
11. **Functional core, thin shell.** Domain logic is pure functions over frozen dataclasses;
    I/O is confined to `core/io.py`, the loaders, `llm/cache.py`, `llm/audit.py` and `pipeline/`.
    Protocols exist only where two real implementations exist (`LLMClient`, `Similarity`).

## 6. Gates and report levels

Implemented in `evaluation/gate.py`, thresholds from `config/preregistration.yaml: gates`.

| Gate | Pass condition (abridged) |
|---|---|
| G0 integrity | 521 notes and 181 footnotes (T0892), 26 variants and 3 paratext segments (Derge), 0 duplicate ids, 0 unclassified notes, verified ingest sentinels pass |
| G1 validity | on test windows: link F1 >= 0.80 and the paired window-cluster bootstrap of F1(Claude) - F1(best control) has CI_low > 0; status kappa >= 0.70, >= 0.8 x human kappa, and the same paired rule; witness-only recall >= 0.70 when gold has >= 10 witness-only segments; NULL precision/recall >= 0.70 when gold has >= 20 NULLs, else deferred to G3 |
| G2 instrument | replicate Fleiss kappa >= 0.80; quote failures <= 2%; missing units <= 1%; no substituted call among the collate requests that fed the consensus (`collation/replicates.json: request_keys`; perturbation calls do not count); collate digest = preregistered digest; P-wrong-window false links <= 0.10; P-deletion recall >= 0.70 (each unit scored once, chunk overlaps included); verified proposal-stage sentinels pass |
| G3 calibration | a verification plan exists and every stratum reached its planned n; no stratum holds unverified units without a phase-2 sample verdict (an uncalibrated stratum would be estimated from the prior; `twophase.uncalibrated_strata`); every machine-negative stratum audited >= 40 (a fully verified smaller census meets it); deferred NULL precision (blind verdicts on machine-ABSENT units) and NULL recall (two-phase imputation on the blind column) >= 0.70 when G1 deferred them; unresolved units resolved or Manski width <= 0.05; verified final-stage sentinels pass |
| G4 estimands | E3: a Sanskrit manuscript column (registry manuscripts only; editions never count) and a co-witness other than the reference whose matrix is built; E4: topic labels complete, human-human topic kappa >= 0.70, MDE <= 0.10 (the larger of `stats.mde` and the MDE simulated on the labels, `stats/power.json`; the simulated one alone when `stats.mde` is unset); E6: scorer kappa >= 0.80, else two-phase-corrected outcomes |

Level 0 (DESCRIPTIVE) if any of G0-G2 fails, 1 (VALIDATED) if G0-G2 pass, 2 (CALIBRATED) if G3
passes too. A report is *confirmatory* only when the preregistration is frozen, the collate
digest is the preregistered one, the ledger lines for that digest carry the current prereg
sha256, and the test set was scored exactly once for that digest. Re-running `evaluate` on the
same output and gold (e.g. to refresh G2/G3 after review) appends nothing: a ledger line with
the same digest, preregistration, gate outcome and metrics (topic-group breakdowns aside) is
already there (`pipeline/measure.py: already_ledgered`). `report/markdown.py` enforces
what each level may print; human-verified (grade A) counts are printed at every level.

## 7. Review defects of 2026-09-30 and their resolution

The code review of v0.2 ([`docs-zh/reference/2026-09-30-code-review.md`](../docs-zh/reference/2026-09-30-code-review.md))
listed 26 defects. Synthesis section 10 planned a resolution for each; this table records what
was actually built and the test that checks it (test files under `tests/`, function names in
parentheses).

| # | Defect | Resolution as built | Where | Test |
|---|---|---|---|---|
| 1 | Status decided by length and priors; no gold | Content-only status table; blind gold, controls P1/B0/P2/external, gates | `matrix/status.py`, `evaluation/` | `unit/test_matrix_build.py` (`test_status_table_in_built_cells`, `test_length_never_decides_status`); `unit/test_evaluation_gate.py` (`test_g1_failure_gives_level_0`, `test_absolute_floor_applies_even_when_claude_beats_controls`) |
| 2 | Review never flows back; evidence always C; topic unset; Delta never called | Verdicts applied on every `build`; grades from consensus; topic labels loaded; `stats` computes E4 or prints NOT_ESTIMABLE | `matrix/build.py`, `pipeline/measure.py`, `pipeline/results.py` | `unit/test_matrix_build.py` (`test_precedence_gold_over_verdict_over_consensus`); `unit/test_pipeline_review.py` (`test_verification_blind_reveal_and_back_into_the_matrix`); `unit/test_report_markdown.py` (`test_every_estimand_is_printed_or_not_estimable_with_a_gate_reason`) |
| 3 | Tibetan substring anchors; lexicon gaps; false hits inside transliterations | Whole-syllable Tibetan matching, multi-syllable forms and exclusion contexts, traditional forms, the "cold forest" term for smasana, no 1-character Chinese forms, IDF weighting | `align/anchors.py`, `data/lexicon/anchors.yaml` | `unit/test_align_anchors.py` (`test_real_lexicon_cases` over `data/fixtures/align_anchors.yaml`, incl. stong pa nyid, dbang phyug, snying gar; `test_removed_one_sided_anchors_stay_removed`); realdata `integration/test_align_realdata.py` (`test_anchor_asymmetry_is_far_below_v02`, repro check #14) |
| 4 | Derge colophon ingested as content; `{a,b}` left in | Everything after the text-end colophon is `paratext`; `{a,b}` -> b, logged as a variant | `ingest/derge.py` | `unit/test_ingest_derge.py` (`test_edit_marks_resolve_to_b_and_are_logged`, `test_only_the_last_text_end_starts_the_paratext`); realdata `integration/test_ingest_realdata.py` (`test_three_paratext_segments_after_the_final_colophon`, `test_26_edit_marks_in_6_pairs_and_no_brace_left`, `test_s6_colophon_is_paratext`) |
| 5 | CBETA notes all dropped | Notes kept as `note` segments with 7 classes and a host; all 181 Taisho footnotes kept as `Footnote` records | `ingest/cbeta.py`, `ingest/notes.py` | `unit/test_ingest_notes.py` (`test_fixture_covers_the_48_distinct_notes_of_t0892`); realdata `integration/test_ingest_realdata.py` (`test_521_notes_in_seven_classes`, `test_181_footnotes_16_from_tokyo_335`, `test_s5_translator_notes`) |
| 6 | `d_ord` wrong in merged chapter groups | Rank displacement computed once per concordance group | `matrix/build.py` | `unit/test_matrix_build.py` (`test_d_ord_is_zero_for_an_in_order_alignment_of_a_merged_group`, `test_d_ord_measures_a_transposition_and_ignores_n_to_1_ties`) |
| 7 | Motive regex counts denials as assertions | `attribution.py` deleted; structured stance scorer (T5) plus a negation-aware lexical baseline for comparison | `experiments/overattribution/score.py`, `lexical.py` | `unit/test_experiment_score.py` (`test_repro_check_1_denials_are_not_assertions`, `test_lexical_motive_fixture_cases`) |
| 8 | Qwen `<think>` text leaked into parsed output | Local backends deleted; only `text` blocks of a structured-output message are parsed | `llm/anthropic_client.py` (`to_response`) | `unit/test_llm_to_response.py` (`test_text_blocks_are_joined_and_other_blocks_ignored`) |
| 9 | II.9 mapped to no Chinese chapter; hard gating hid errors | Concordance v2 with partial spans and evidence; soft core window; full witness text in every request; V7 relocation diagnostics | `registry.py`, `data/registry/concordance.yaml`, `collate/` | `unit/test_registry.py` (`test_ii9_is_split_with_evidence`); `unit/test_collate_verify.py` (`test_v7_link_from_ii9_to_chinese_18_outside_the_window_is_a_relocation`, `test_v7_concordance_v2_puts_chinese_18_inside_the_ii9_window`); sentinels S3a-S3c (proposal stage, need a Claude run) |
| 10 | `decompose()` not exhaustive | Ordered exhaustive classes `attested_vorlage`, `vorlage_ms`, `shared` / `shared_revised`, `residual`, `insufficient`; sum asserted | `stats/decompose.py` | `unit/test_stats_decompose.py` (`test_decomposition_property_loop`) |
| 11 | Component deviation compared presence only | T2 codes ret/gen/sub/lit/om/add per slot, with polarity rule C5 | `collate/components.py` | `unit/test_components_verify.py` (`test_c5_flip_without_a_negation_slot_fails` and the other C1-C5 tests) |
| 12 | `chapter_lookup` mapped a merged chapter to its last part | `Concordance.refs_for_local` returns every reference chapter (pin11 -> I.11, II.1; pin20 -> II.11, II.12) | `registry.py` | `unit/test_registry.py` (`test_merged_chinese_chapters_map_to_several_reference_chapters`, `test_many_to_many_mapping`) |
| 13 | `attribute` ran on UNALIGNED cells; one Sanskrit witness sufficed | `attribution.py` deleted; `decompose.classify` refuses non-deviating units; `insufficient` below m_min = 2 informative manuscripts; only manuscript readings are inputs (editions are witnesses, not manuscripts) | `stats/decompose.py` | `unit/test_stats_decompose.py` (`test_class_order_rules`) |
| 14 | Orphan id collision (70 blocks became 67 rows) | Orphan row id `+<first segment id>`; witness-only rows in merged groups get a reference-chapter attribution (critique B15) | `core/ids.py`, `matrix/build.py` | `unit/test_core_ids.py` (`test_orphan_rows_on_one_line_are_distinct`); `unit/test_matrix_build.py` (`test_two_orphans_on_one_line_give_two_rows`, `test_orphan_in_merged_group_is_attributed_to_the_nearest_linked_neighbour`) |
| 15 | Hybrid `max()` noise floor | Embedding similarity deleted | `align/similarity.py` | `unit/test_align_similarity.py` (`test_embedding_similarity_is_gone`) |
| 16 | Component extraction read 80-character truncated cell text | T2 reads full segment text from the segment store | `collate/components.py` | `unit/test_components.py` (`test_request_body_holds_the_full_segment_texts`) |
| 17 | `VariantTable.explained` positional | Deleted with the Laozi transfer module (preserved at `1f9474c`) | - | none (code removed) |
| 18 | Counterfactual docs assumed two local model profiles | Deleted; the experiment uses Claude only | `experiments/overattribution/` | none (docs removed) |
| 19 | `models.yaml` vLLM flags and MITRA repository names | File deleted; MITRA facts corrected in the Chinese method document | `docs-zh/01-method.md` | none (documentation) |
| 20 | Topic vocabularies disagreed; no `neutral` | One codebook `data/codebook/topics.yaml`, equal to constants in code, used by T3, labels, E4 and the experiment | `topics/codebook.py` | `unit/test_topics_codebook.py` (`test_vocabulary_is_tokushiges_eight_plus_three_controls`, `test_codebook_that_disagrees_with_the_code_is_refused`); `unit/test_topics_labels.py` (`test_unknown_topic_error_lists_the_allowed_values`) |
| 21 | Tangut/Chinese lacuna logic spanned clauses | Deleted with the transfer module | - | none (code removed) |
| 22 | Sanskrit numerals matched as substrings | Whole-word numeral matching with exclusions | `core/textnorm.py`, `data/lexicon/numerals.yaml` | `unit/test_core_textnorm.py` (`test_numerals` over `data/fixtures/core_numerals.yaml`: laksana, advitiya, kasta, ratri, ksatriya; Tibetan fused-particle look-alikes bcos, brgyas, khrir) |
| 23 | Pure-Python embedding cost | Deleted | - | none (code removed) |
| 24 | `__version__` mismatch; manifest lacked the commit | `__version__` from `importlib.metadata`; manifest with git commit, dirty flag, config and input sha256, instrument digests, served models | `__init__.py`, `core/io.py`, `pipeline/context.py` | `unit/test_core_io.py` (`test_manifest_keys_and_values`); `unit/test_pipeline_e2e.py` (`test_offline_replay_is_all_cache_hits_and_the_manifest_records_it`) |
| 25 | `DATA_DIR` needed an editable install | Root found from the working directory or `--root`; paths from `run.yaml: paths` | `config.py`, `cli.py` | `unit/test_cli.py` (`test_root_option_works_from_any_directory_and_later_stages_reuse_the_latest_run`) |
| 26 | Tests only on ideal fixtures | `tests/integration/` real-data tier (marker `realdata`) with sentinel facts | `tests/integration/` | the realdata tests themselves; skipped without `HEVAJRA_RAW_DIR` |

The review's reproduction checks were ported as tests (critique B11): the cases of #1 (denials),
#6 (anchor conflict, `unit/test_align_dp.py::test_repro6_anchors_on_both_sides_without_overlap_are_a_conflict`),
#7 (anchor and numeral false positives, in the fixtures above) and #14 (anchor asymmetry,
realdata) are named in their tests; the others are covered by the rows above.

## 8. Deferred and not built

| Item | State |
|---|---|
| Batches API pre-warm of the cache | **Deferred** (no `llm/batch.py`); all calls are synchronous streams |
| Drift canary | **Cut**; the evaluation ledger lets instrument versions be compared |
| Effort sweep on dev gold | **Not automated**; change `config/llm.yaml: tasks.collate.effort` and rerun `collate --chapters ...` + `evaluate --gold dev` (each effort is a new cache key and digest) |
| P-negation perturbation | **Built**: `perturb --negation` draws the dev-gold `equivalent` pairs negated on both sides, strips the witness negator and writes the polarity recall to `evaluation/perturbations.json: negation`; shown in the report at level >= 1; never gated |
| Confidence reliability (Brier score) | **Built**: `evaluation/scores.py: confidence_reliability` (Brier of the nominal high/medium/low probabilities vs the constant predictor, reliability table) on the units with a confidence, in `scores*.json: sources.<source>.reliability` and the report at level >= 1. Confidence stays a stratum unless it beats the constant on dev gold; no stage uses it otherwise |
| Sanskrit reference and readings | **Built**: `ingest` loads `data/reference/<sanskrit_reference>.tsv` when present (Sanskrit mode: Derge and T0892 both aligned witnesses, same functions per witness, scope lines adapted); `stats` reads `data/reference/readings/` for E3 (`pipeline/e3.py`) with the A2 bound. Not done: committed collator examples for the sa-bo and sa-zh pairs (the researcher writes them; `collate` refuses a pair without examples), Sanskrit-specific dev regions and test windows (the committed `gold.dev_regions` use Derge coordinates such as `around: "D418:17b.6"`), sentinels stated on Sanskrit loci (sentinels on Derge -> Chinese loci are not checked against other pairs), and T2/T3 per language pair (the T2 component coder keeps its Tibetan -> Chinese examples; T3 renders its prompt for the reference language, but the topics digest in `prereg` is computed for the configured Derge reference) |
| E3 under the Derge reference | Readings keyed by Derge segment ids are read, but E3 stays NOT_ESTIMABLE (`G4: reference is the co-witness`); the decomposition is written to `stats/details.json: e3` as description only |
| E3 circular-shift null and per-manuscript odds ratios | **Deferred** until real manuscript readings exist; O, E, O-E and phi against the independence base rate are computed |
| E4 "Delta from human-verified status only" | `twophase.known_outcomes` gives the verified outcomes; the `stats` stage does not compute this Delta. Every other E4 diagnostic is written (`stats/details.json: e4`, see [data formats](data-formats.md)) |
| Witness distance / UPGMA | **Built** for the Sanskrit reference: `stats` writes `details.json: distance` (Sanskrit reference, Derge, T0892); skipped under the Derge reference (two texts) |
| `d_comp` cell dimension from T2 | **Deferred** until per-code precision and recall on double-coded dev pairs are reported |
| Chinese report `summary.zh.md` | **Dropped** by researcher decision; reports are English only |
| Automatic retry or window split after refusal/truncation | **Not built by design**: units go to the human `resolve` queue |
| Further witness columns (Them spangs ma Kangyur, Tangut, Ming witnesses) | Out of scope for v0.3 (registry entries only); the Derge is the only second column, under the Sanskrit reference |
