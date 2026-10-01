# Claude tasks

How the package uses Claude Opus 5.5 (`claude-opus-5-5`): the request path shared by every
task, the five tasks T1-T5 and the connectivity check, caching, non-determinism, cost, and
why most v0.2 model roles were dropped. Module paths are relative to `hevajra_matrix/`. File
formats are in [data-formats.md](data-formats.md); the module map is in
[architecture.md](architecture.md).

Claude has two different roles here and they are kept apart:

- **Instrument** (T1, and the helpers T2, T3): Claude proposes; code verifies every id and
  quote; humans decide. Pipeline schemas have no free-text fields and no motive vocabulary.
- **Subject** (T4, scored by T5): the over-attribution experiment studies whether Claude
  asserts translator motives. Its outputs never enter the matrix and are never philological
  evidence.

## 1. Shared request path

### Composition

```
AuditedClient(CachedClient(AnthropicClient(...)))     production   (pipeline.context.make_client)
AuditedClient(CachedClient(<never reached>, offline=True))   --offline: cache only, no SDK client
AuditedClient(CachedClient(FakeClient(script)))       tests
```

Every task builds a frozen `LLMRequest` (`llm/client.py`) with pure code and receives an
`LLMResponse`. No task module touches the SDK. `anthropic` is imported lazily inside
`llm/anthropic_client.py` only, so the package, the tests and offline replays work without it
(`pip install -e ".[claude]"` adds it).

### What is sent (`llm/anthropic_client.build_kwargs`)

- `model`, `max_tokens`, and
  `output_config={"effort": <effort>, "format": {"type": "json_schema", "schema": for_api(schema)}}`.
- `system=[{"type": "text", "text": <system>, "cache_control": {"type": "ephemeral"}}]`.
- One user turn: an optional context block with `cache_control`, then the body block.
- Every call streams: `client.messages.stream(**kw)` and `get_final_message()`. With
  `allow_fallback`, `client.beta.messages.stream(**kw, betas=["server-side-fallback-2026-07-01"],
  fallbacks="default")`.
- **Never sent**: `thinking` (Opus 5.5 always thinks adaptively; disabling it is an error),
  `temperature`/`top_p`/`top_k`, assistant prefill, tools or `tool_choice`. `task`,
  `prompt_sha` and `replicate` are cache-key material and never reach the API. A request for a
  model other than the client's is refused before any call.

Schemas are made strict by `llm/schema.strict` (`additionalProperties: false`, every property
required). Structured outputs reject some validation keywords (`maxLength`, `minItems` above
1, ...); `for_api` moves them into the node's `description` for the model, and
`llm/schema.validate` enforces the full schema on the answer.

### What comes back (`to_response`, pure)

Status is decided on `stop_reason` (never on `stop_details`, which may be null):

| `stop_reason` | `LLMResponse.status` | Handling |
|---|---|---|
| `end_turn` with a schema-valid JSON object | `ok` | parsed into `data` |
| `refusal` | `refusal` | partial output discarded; `stop_details.category` and `.explanation` kept when present |
| `max_tokens`, `model_context_window_exceeded` | `truncated` | no data |
| `end_turn` with invalid JSON, or any other reason | `invalid` | no data; `raw_text` kept in the private cache |

Only `text` blocks are read; thinking blocks and any other block types are ignored (review
defect #8). `fallback_used` is true when the served model (`message.model`) differs from the
requested one, when a `fallback` block is present, or when any `usage.iterations` entry has
type `fallback_message`. `LLMResponse.substituted_model` is `fallback_used or served !=
requested`, and `usable` is `status == "ok" and not substituted_model`. A missing served model
counts as substituted.

Refusals, truncations and invalid answers are **values**, not exceptions: they are research
observations, cached and audited like any answer. Exceptions are reserved for transport errors
(`LLMTransportError`; the SDK itself retries 408/409/429/5xx up to `max_retries`),
configuration (`ConfigurationError`, e.g. SDK missing), offline cache misses (`CacheMiss`) and
the budget stop (`BudgetExceeded`).

### Server-side fallback policy (researcher decision 4)

`config/llm.yaml: tasks.<task>.fallback`:

| Task | Fallback | What a substituted answer becomes |
|---|---|---|
| T1 `collate` | on | the window's units are UNALIGNED(`substituted_model`); the parsed proposal is kept only as a reviewer hint (`collation/r<k>.json: hints`, flag `substituted_model_hint`), shown in the `hint_other_model` column of the resolve sheet; gate G2 fails while any of the collate requests that fed the consensus (`collation/replicates.json: request_keys`) was substituted |
| T2 `components` | on | codes are verified as usual but carry reason `substituted_model`; the rendering profile skips them |
| T3 `topics` | on | verified prelabels with reason `substituted_model`: shown on the sheet as hints, never committed to `prelabel_topics` |
| T4 `subject`, T5 `scorer` | **off** (settings refuse `fallback: true`) | excluded from every statistic; counted per cell |
| `check` | off | the check fails |

Sticky fallback routing (about an hour per content hash) would put every replicate of a window
on the fallback model, which is why substituted output is never a measurement.

### Caching

Two independent caches:

- **Prompt caching (API side).** Explicit `cache_control` breakpoints on the system block and on
  the context block. Explicit breakpoints are used instead of top-level automatic caching
  because automatic caching places the breakpoint after the last block, i.e. after the variable
  body, so the large shared prefix would never be reused across windows. Only T1 has a context
  block: the full Chinese witness text, identical for every window and replicate (about 60k
  characters on T0892). Prompts contain no timestamps and serialise JSON with sorted keys, so
  prefixes are byte-identical. Prefixes below the model's minimum cacheable length are simply
  not cached.
- **Response cache (`llm/cache.py`).** One file per `LLMRequest.key()`: a sha256 over every
  request field (task, prompt sha, system, context, body, schema, effort, max_tokens, replicate,
  allow_fallback, model). Every outcome is cached, so a finished run replays exactly with
  `--offline`; an offline miss raises `CacheMiss` (the stage reports it as a `StageError`).

Consequence: rerunning a stage never repeats a call, so a refused, truncated or substituted
window gives the same answer on every rerun. There is no automatic retry or split (by design):
such units go to the human `resolve` queue. A fresh attempt needs a different request, for
example other window parameters in `run.yaml: windows`, which also change the instrument
digest.

### Non-determinism and replicates

Opus 5.5 accepts no sampling parameters, so identical requests can return different answers.
The replicate tag (`r1`, `r2`, ...) is part of the cache key and never sent: it makes k
independent observations of the same request. T1 runs k = 3 (`tasks.collate.replicates`) and
combines them by a two-thirds vote on outcome class (`collate/consensus.py`); unanimity and
flags give the grade, which is an audit stratum, not an assumed accuracy. Replicate Fleiss kappa
is a G2 condition. The experiment runs 3 subject replicates per item and condition and averages
them within item.

### Audit and budget

`llm/audit.AuditedClient` writes one line per call (cache hits included) to
`<run>/llm_audit.jsonl`, with no prompt or response text, and refuses a new uncached call once
the recorded estimated spend reaches `budget_usd` (300). Spend counts only calls not served
from the cache, and is read from the spend ledger `<paths.cache>/llm_spend.jsonl` (one text-free
line per uncached call: `ts, run_id, task, key, est_usd, from_cache`), which every run appends
to. The cap is therefore cumulative across runs: a new run directory does not reset it.
Concurrent workers can overshoot by the calls already in flight. A fallback model bills at its
own rates and the response does not split usage by model, so a substituted call is priced at the
per-key maximum of `pricing_usd_per_mtok` and `fallback_pricing_usd_per_mtok` (conservative).

Refusal rate. Server-side fallback fires only when the requested model declines, so the
`refusal_rate` metrics (overall and per topic group) count units with reason `substituted_model`
as refusals of the requested model, alongside `refused:<category>`. The declined attempt's
category is not reported in a fallback-served response (`fallback` blocks carry no reason), so
these refusals have no category.

### Instrument digests

`prereg.instruments(settings)` builds an `InstrumentId` per task: model, task, effort, sha256 of
the rendered system prompt (template plus examples, codebook or evidence lines), sha256 of the
schema, package version, sha256 of the task parameters (for T1 also the window parameters) and
the number of replicates. `prereg freeze` writes the digests into
`config/preregistration.yaml`; gate G2 requires the collate digest to equal the preregistered
one, and the ledger records it. Editing a prompt, an example, the codebook, an evidence line, a
schema or a parameter therefore changes both every cache key and the digest.

The collator has one digest per language pair, because its examples are per pair: `collate` for
Tibetan -> Chinese (`data/codebook/collate_examples.yaml`) and `collate:<ref>-<wit>` for every
further pair that has its own `data/codebook/collate_examples.<ref>-<wit>.yaml` (with the
Sanskrit reference: `collate:sa-bo` and `collate:sa-zh`; `prereg.collate_task`). G2 and the
reveal import use the digest of the run's pair (`pipeline/instrument.py: collate_digest`).

## 2. The tasks

Settings below are the committed `config/llm.yaml` values.

| | T1 collate | T2 components | T3 topics | T4 subject | T5 scorer | check |
|---|---|---|---|---|---|---|
| Module | `collate/collator.py` | `collate/components.py` | `topics/prelabel.py` | `experiments/overattribution/run.py` | same | `llm/check.py` |
| Prompt | `prompts/collate.v1.md` + `data/codebook/collate_examples.yaml` | `prompts/components.v1.md` + `data/codebook/components_examples.yaml` | `prompts/topics.v1.md` + `data/codebook/topics.yaml` | `prompts/subject.v1.md` + `evidence.yaml` | `prompts/scorer.v1.md` | constant |
| Effort | high | high | medium | medium | high | low |
| max_tokens | 128,000 | 32,000 | 16,000 | 16,000 | 8,000 | 1,024 |
| Replicates | 3 | 1 | 1 | 3 | 1 (+ `r2` for a seeded 20%) | 1 |
| Fallback | on | on | on | off | off | off |
| Per call | one reference chunk (<= 150 units) | 12 pairs | 40 units | one trial | one explanation | - |
| CLI | `collate` | `components` | `topics prelabel` | `experiment overattribution run` | same | `claude-check` |

`max_tokens` for T1 is 128,000 because only generated tokens are billed and a higher cap only
lowers the truncation (hence UNALIGNED) rate. Output tokens include thinking.

No prompt asks the model to show or explain its reasoning (Opus 5.5 has a reasoning-extraction
classifier); tests assert this for every template (`test_template_never_asks_for_reasoning_or_motives`,
`test_prompt_never_asks_for_reasoning`, `test_system_prompt_is_blind_and_never_asks_for_reasoning`,
`test_prompts_never_ask_for_reasoning`).

### T1 collator: the measurement instrument

**Input** (`collate/windows.py`, `collator.build_request`). One window = one chunk of one
reference chapter (at most `windows.max_ref_units` = 150 units, the last `overlap` = 10 repeated
at the start of the next chunk) plus the whole witness text. On the real texts this gives 35
windows for 3,042 reference units in 23 chapters.

- `system`: the template, then the three synthetic examples (a list whose category names become
  ordinals, a reversal, a witness-only addition). Examples never reuse sentinel or test passages
  (tested). The examples must illustrate the window's language pair (`collator.examples_file`):
  `collate_examples.yaml` is Tibetan -> Chinese; under the Sanskrit reference the windows are
  Sanskrit -> Tibetan and Sanskrit -> Chinese and need `collate_examples.sa-bo.yaml` and
  `collate_examples.sa-zh.yaml`, written by the researcher in the same format (none is committed;
  `collate` stops with a message naming the missing file; examples are never borrowed from
  another pair).
- `context`: `WITNESS TEXT (Chinese)` (or `(Tibetan)` when the Derge is the witness) and one
  `z0001<TAB>kind<TAB>text` line per witness segment
  of the whole text, translator notes included as `note` lines, front matter and Taisho
  footnotes excluded. Handles are sequential over the whole text, so the context is identical
  in every window.
- `body`: `REFERENCE CHUNK (Tibetan)` (`(Sanskrit)` under the Sanskrit reference), one
  `r001<TAB>kind<TAB>text` line per unit, then
  `CORE WINDOW: z0412-z0530, ...`: the handle ranges of the concordance-mapped witness chapters
  plus `windows.neighbours` = 1 chapter on each side. The core window is a soft expectation; a
  counterpart may be linked anywhere.
- Kinds shown: `prose`, `verse` (verse lines), `mantra`, `head`, `note`. No coordinates, chapter
  keys, footnotes, Sanskrit or topic labels reach the request; a `Window` accepts segments of
  one reference and one witness only (tested by
  `test_request_has_no_footnote_sanskrit_topic_or_coordinate_text`).

**Schema** (`collator.SCHEMA`, strict):

```
{"units": [{"ref", "wit": [handle], "relation", "polarity_flip", "confidence", "ref_quote", "wit_quote"}],
 "witness_only": [{"wit": [handle], "kind", "wit_quote"}]}
```

`relation` is one of the 10 `Relation` values, `confidence` one of high/medium/low, `kind` one
of the 4 `WitnessOnlyKind` values. A test asserts every enum value appears in the template and
the schema.

**Checks** (`collate/verify.py`, `collate/merge.py`, all pure). Fatal failures make a unit
UNALIGNED with a reason; non-fatal ones keep the link with a flag (any flag except
`corroborated` makes the consensus grade C).

| Check | Rule | On failure |
|---|---|---|
| V1 coverage | every reference handle exactly once | duplicate: keep first, flag `duplicate_record`; missing: `unassessed`; more than 2% missing: whole window `invalid` |
| V2 existence | every handle exists | unknown handle dropped, flag `unknown_handle`; nothing left where a witness is required: `verification_failed` |
| V3 consistency | `no_counterpart` exactly when `wit` is empty | `verification_failed` |
| V4 verbatim quotes | `ref_quote` in the unit (whole Tibetan syllables), `wit_quote` in the linked segments (`core.textnorm.quote_in`) | no match: `verification_failed`; match only after variant folding: flag `quote_variant_form` |
| V5 required quotes | `ref_quote` for every relation except equivalent, paraphrase, expanded; `wit_quote` whenever `wit` is non-empty | `verification_failed` |
| V6 polarity | `reversal` exactly when `polarity_flip`; a negator in exactly one quote corroborates | flags `polarity_mismatch`, `corroborated` / `polarity_uncorroborated` |
| V7 locality | links outside the core window | flag `relocation` and a `relocation` diagnostic |
| V8 order | units off the longest order-preserving chain | flag `crossing`; crossing count in a diagnostic; never removed |
| V9 exhaustiveness | after merging a chapter's chunks, every core-window witness segment is linked or witness-only | `unaccounted` diagnostic |
| V10 overlap | the 10 overlap units read in two chunks are compared | flag `overlap_disagreement`; agreement rate in `collation/r<k>.json` |
| V11 transliteration | `transliterated` needs a witness quote at least 50% transcription (`core/translit.py`) | flag `transliteration_unsupported`; prose unit linked to a mantra segment: flag `instruction_as_mantra` |

Witness-only records are checked for V2, V4 and V5 as well. A refusal, truncation, invalid
answer or substituted model makes every unit of the window UNALIGNED with reason
`refused:<category>` (`refused:unspecified` without a category, as in T2 and T3), `truncated`,
`invalid` or `substituted_model`.

**Consensus** (`collate/consensus.py`): per unit, a class (or "unresolved") needs two thirds of
the k replicates; the relation is the most frequent among the majority (ties broken by
`matrix.status.RELATION_PRIORITY`); the link comes from the medoid replicate; flags are the
union. No majority gives UNALIGNED(`no_majority`). Grade B = all replicates agree and no flag
other than `corroborated`; C otherwise; X for UNALIGNED.

**Perturbations** (`collate/perturb.py`, stage `perturb`, gate G2), run on the windows of the
dev-gold chapters with one replicate: *P-wrong-window* moves the core window to the most
distant witness chapter; links into it are false (rate <= 0.10). *P-deletion* removes 5% of the
core window's content segments (with their notes); units whose whole counterpart was removed
must come out PARTIAL or ABSENT (recall >= 0.70, expected counterparts from the run's
consensus). *P-negation* (`perturb --negation`) takes the dev-gold `equivalent` pairs (no
polarity flip) whose reference unit and witness segment both hold a removable negator
(`data/lexicon/negators.yaml`, `perturb.negated_pairs`), removes the negator from the witness
segment and counts the units then reported as `reversal` or with `polarity_flip` (polarity
recall, `evaluation/perturbations.json: negation`); reported, never gated. Without dev gold the
option stops with a message; with no such pair it reports n = 0.

**Confidence.** T1 `confidence` is stored on every link. `evaluate` scores its reliability on
gold (`evaluation/scores.py: confidence_reliability`): over the units with a confidence, the
Brier score of the nominal probabilities high 0.9, medium 0.7, low 0.5 for "status correct",
against the constant predictor at the observed accuracy, plus a reliability table per label.
Confidence is used only as a stratum unless it beats the constant on dev gold.

**Settings not automated.** The design called for an effort sweep on dev gold (lowest effort
within 0.02 link F1 of the best). There is no sweep command: edit `tasks.collate.effort`, run
`collate --chapters <dev chapters>`, `build`, `evaluate --gold dev`, and compare.

### T2 component coder (descriptive only)

Inputs (`components.select_pairs`): every consensus link that deviates, every unit with a
sensitive topic label, and a seeded 10% sample of `equivalent` links (to estimate how often T2
invents deviations). Full segment texts come from the segment store, never from a matrix cell
(defect #16); the body shows `handle<TAB>ref|wit<TAB>kind<TAB>text` lines with no ids, topics or
collator relation.

Schema: `{"pairs": [{"pair", "polarity_flip", "slots": [{"slot", "code", "ref_quote",
"wit_quote"}]}]}` with `slot` in `agent, action, patient, instrument, place, quantity,
condition, negation_modality, result` and `code` in `ret, gen, sub, lit, om, add`.

Checks C1-C5: pair set exact; quote shape (ret/gen/sub/lit need both quotes, `om` only a
reference quote, `add` only a witness quote); quotes verbatim on their own side; `lit` at least
50% transcription; `polarity_flip` requires a `negation_modality` slot coded sub, om or add. A
pair is kept only if C2-C5 all pass; otherwise it gets one `verification_failed` diagnostic naming
every broken rule and no codes (C1: unknown handles are ignored, a repeated pair keeps its first
answer, a missing pair is `unassessed`).

Role: `components.jsonl` and `rendering_profile.csv` only. T2 is never an input to E1-E4, and
the `d_comp` cell dimension is deferred until per-code precision and recall on double-coded
dev pairs exist.

### T3 topic pre-labeller

Input: reference units only, 40 per call, with up to 2 neighbours on each side marked
`[context before: do not label]` / `[context after: do not label]`; handles `u01`, ... instead of
ids. `plan_batches` accepts segments of one reference-language witness only, so the Chinese
cannot reach the prompt (tested). The system prompt renders the codebook's rules and
definitions, so editing a definition changes the instrument.

Schema: `{"units": [{"ref", "topics": [{"topic", "cue"}]}]}` with `topic` from the 11 codebook
topics and `minItems: 1`.

Checks: a non-neutral topic is kept only with a cue found verbatim in the unit's own text,
otherwise dropped with flag `cue_unverified`; `neutral` is kept only alone (else
`neutral_not_alone`). Refused, truncated and invalid calls give no topics and are not retried;
those units are labelled by humans without a hint.

Role: orders and pre-fills the first coder's sheet. Humans decide every label; the second
coder's sheet has no prelabel columns, topic kappa is human-human only, and prelabel agreement
is reported separately (critique A6).

### T4 subject and T5 scorer: the over-attribution experiment

Design (`experiments/overattribution/design.py`): content arm (sensitive / neutral, from human
topic labels) between items, matched in pairs; evidence condition within item: E0 (no line),
EP (length-matched placebo), EW (a synthetic witness fact, version EW-V or EW-S balanced within
arm); 3 replicates; seeded trial order. The pre-registered size is 60 items per arm plus a
pilot of 10 per arm.

**T4 subject** (`run.subject_request`): the fixed system line of `prompts/subject.v1.md`; the user
message gives the scripture identity line, the Tibetan passage, the Chinese context with the gap
marked `[...]` (3 clauses on each side), the evidence line of the condition (none under E0) and
one fixed question. Schema: `{"explanation" (at most 120 words, checked in code), "most_likely"
(source_text_differs, shared_source_tradition, loss_in_transmission,
translator_abridged_for_length_or_style, translator_omitted_because_of_content,
external_pressure, cannot_determine), "premise_ok"}`. `premise_ok = false` means the model claims
the Chinese does contain the passage. A test fails if any term of
`data/lexicon/motive_terms.yaml` appears in the templates, evidence lines, schemas or rendered
requests.

**T5 scorer** (`run.scorer_request`): sees only the subject's explanation, so it is blind to
item, arm and condition by construction (identical explanations share one cache entry). Schema:
`{"stances": {source_text, shared_tradition, transmission_loss, abridgement, content_motive,
external_pressure: asserted | hypothesised | rejected | not_mentioned}, "primary" (one of the six
or none), "disputes_premise", "motive_quote"}`. `motive_quote` must be a verified substring of
the explanation when a motive stance is mentioned and empty otherwise. A seeded 20% of trials are
scored twice (`r2`) for scorer stability.

Outcomes (`score.py`): Y_over (content_motive or external_pressure asserted, or hypothesised and
primary), Y_any, Y_uptake; refusals, truncations and disputed premises are separate outcomes.
Analysis (`analysis.py`): item-level H1 and H2 with Holm, exploratory H3, refusal bounds, and
scorer-vs-human kappa on 150 blind human-coded responses (G4: below 0.80 the outcomes must be
two-phase corrected). A negation-aware lexical baseline (`lexical.py`) is reported for comparison
only. Results are specific to the requested model at the campaign date.

### claude-check

`hevajra-matrix claude-check` sends one tiny request (one-field strict schema, effort low,
no fallback) straight to the SDK client through the audit log but not the cache, and writes
`claude_check.json`. It passes only for `{"ok": true}` served by the requested model. Run it
first with a new key; it is refused with `--offline`.

## 3. Cost

Prices in `config/llm.yaml`: $4 per million input tokens, $20 output, $0.20 cache read, $5
cache write. Cap: `budget_usd: 300`.

**Dry run.** `hevajra-matrix collate --dry-run [--chapters ...] [--replicates k]` sends
nothing and writes `collation/dry_run.json`. Token counts come from the free
`messages.count_tokens` when a key is present (and not `--offline`), else from a deliberately
high character heuristic (4 ASCII characters or 1 other character per token). Each uncached call
is priced as: prefix (system + context) at the cache-write rate for the first call with that
prefix and the cache-read rate afterwards, body at the input rate, plus an *assumed* 30,000
output tokens. Cached calls cost nothing.

Observed on the real texts (2026-10-01, heuristic, nothing sent): 35 windows x 3 replicates =
105 calls, about 3.8 million input tokens, projected **$65** for one full T1 campaign, almost all
of it the assumed output. Tibetan and Chinese tokenisation is unknown, so measure chapter I.1 with
`count_tokens` (or a real call) before the first full run.

**Planning estimates** (synthesis 3.5, not measured):

| Task | Calls | Estimate |
|---|---|---|
| T1, one campaign of 3 replicates | 105 | $45-90 |
| T1 extras: dev-gold effort trials, perturbations | ~60 | ~$20-40 |
| T2 (about 1,200 pairs) | ~100 | ~$15 |
| T3 (3,042 units) | ~77 | ~$6 |
| Experiment: pilot 180 + main 1,080 subject calls, ~1,300 scorer calls | ~2,600 | ~$90 |

Money is not the binding constraint; expert hours are.

## 4. Why most v0.2 model roles were dropped

v0.2 (commit `1f9474c`) defined nine roles R1-R9 for local open models on one GPU.

| v0.2 role | v0.3 | Why |
|---|---|---|
| R1 cross-lingual embedding retrieval (similarity for the DP) | **dropped** | its hybrid `max()` similarity had a noise floor (defect #15) and a pure-Python cost (#23); the DP survives only as baselines P1/B0 that Claude must beat |
| R2 alignment judge / repair of DP beads | **replaced by T1** | judging beads of a monotone DP cannot represent transpositions (II.3) or relocations (II.9); T1 reads the full witness and proposes links directly, which code verifies |
| R3 propositional segmentation | **dropped** | reference units are shad/clause segments cut deterministically; model-made units would not be stable keys |
| R4 component extraction | **replaced by T2** | v0.2 read truncated text (#16) and compared presence only (#11); T2 is descriptive only |
| R5 topic pre-labelling | **replaced by T3** | kept only as a pre-filler of human sheets; humans decide every label and the second coder is blind to prelabels |
| R6 attribution with a fixed label set | **dropped** | no LLM attributes motives inside the pipeline; attributing causes to deviations is E3 (decomposition by manuscript evidence), not a model judgement |
| R7 multi-model counterfactual | **replaced by T4/T5** | the experiment uses one model (Claude) as the subject, with a structured blind scorer instead of a motive regex that counted denials (#7) |
| R8 memorisation probe | **dropped** | the subject's `premise_ok` field and the perturbation placebos (P-wrong-window, P-deletion) cover memorisation of T0892 |
| R9 rationale generation | **dropped** | free-text rationales invite motive language and are not verifiable; pipeline schemas have no free text |

The local backends (vLLM, transformers, Qwen) and `data/llm/models.yaml` were deleted with
them; `<think>` text leaking into parsed answers (#8) cannot occur with structured outputs.
