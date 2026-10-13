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
AuditedClient(CachedClient(AnthropicClient(...)))     backend api (default)   (pipeline.context.make_client)
AuditedClient(CachedClient(ClaudeCodeClient(...)))    backend subscription
AuditedClient(CachedClient(<never reached>, offline=True))   --offline: cache only, no network client
AuditedClient(CachedClient(FakeClient(script)))       tests
```

### Backends (`config/llm.yaml: backend`, CLI `--backend`)

| | `api` (default) | `subscription` |
|---|---|---|
| Client | `llm/anthropic_client.py`, official SDK | `llm/claude_code_client.py`, one `claude -p` process per request |
| Credentials | `ANTHROPIC_API_KEY` (prepaid Console credits) | a first-party Claude Max/Pro login. The child gets an allowlisted environment, so a parent `ANTHROPIC_API_KEY` cannot reach it. The CLI itself prefers, in order: a cloud session's API key file (`/home/claude/.claude/remote/.api_key`, always refused), `CLAUDE_CODE_OAUTH_TOKEN` (from `claude setup-token`), a cloud session's own OAuth token (same directory), the login stored under the configuration directory (`/login`); policy settings can add an API key helper. So before the first request the client asks the CLI (`claude auth status --json`, local, no model request) and runs only on `authMethod` `oauth_token` or `claude.ai` with `apiProvider` `firstParty`. The reported method is recorded per response (`client_info.auth`) and in the manifest (`llm.logins`). |
| Request | system and context blocks with cache breakpoints; `output_config` with effort and a strict JSON schema (constrained decoding) | `--system-prompt-file` (instructions, then the context), body on stdin, `--tools ""`, `--setting-sources ""`, `--strict-mcp-config`, `--no-session-persistence`, `--json-schema` (validate and re-prompt), `--model`, `--effort`, `--max-turns`, `CLAUDE_CODE_MAX_OUTPUT_TOKENS`, `CLAUDE_CODE_NO_MODEL_FALLBACK=1`, `DISABLE_AUTOUPDATER=1`; run in an empty temporary directory, in the parent's process group (a signal to the group stops it; the CLI's signal handlers also stop retries and new runs in worker threads). The CLI still prepends its identity line ("You are a Claude agent, built on Anthropic's Claude Agent SDK.") and an attribution header, and adds its own reminder turns for the structured answer. |
| Concurrency | `workers` threads | the same; the first request of a pass not yet in the response cache runs alone, so the others read the prompt cache it wrote (both backends) |
| Fallback | server-side fallback per task (`fallbacks="default"`) | off: no `--fallback-model`, and `CLAUDE_CODE_NO_MODEL_FALLBACK` makes the CLI fail (`ModelUnavailable`, not retried) instead of answering with another model; output from any other model in `modelUsage` still marks the response `fallback_used` (only an exact id, a date suffix or a context tag counts as the requested model) |
| Refusal | `stop_reason == "refusal"` with a category | `stop_reason == "refusal"` or `safety_stops > 0` (no category) |
| Truncation | `stop_reason` `max_tokens` / `model_context_window_exceeded` | the same, or `api_error` `max_output_tokens` (the CLI sets `is_error` and a synthetic `stop_reason` `stop_sequence`; stored as a `truncated` value with `stop_reason` `max_tokens`). Trusted only on a run that answered (subtype `success`, `error_max_turns`, `error_max_structured_output_retries`). The CLI first asks the model to continue after a capped message, so a truncation can end as an answer: `client_info` records `num_turns` and `output_over_max_tokens`. |
| Invalid | schema-invalid answer | the same, and `error_max_turns` / `error_max_structured_output_retries` (cached `invalid` values although the CLI sets `is_error`) |
| Run failures | SDK errors after retries | any other `is_error` or subtype raises `LLMTransportError`, retried `subscription.max_retries` (1) times; not retried: an exhausted allowance (`UsageLimitReached`: `api_error` `usage_limit_reached` / `model_requires_usage_credits`, or "You've hit your ... limit"), an unavailable model (`ModelUnavailable`), and a run longer than its limit (`CliTimeout`: `subscription.timeout_s`, 7200 s, or `max_tokens` / 15 tok/s if larger). Rerunning resumes from the cache. A transient 429 ("not your usage limit") is retried. |
| Provenance | - | `client_info` on every response and audit line: CLI version, login method, seconds, subtype, turns, terminal reason, the CLI's raw stop reason; the manifest's `llm` summary lists the CLI versions and logins of the run's calls. `subscription.cli_version` pins the CLI exactly (enforced, part of the digest, with its own cache directory `<cache>/subscription/cli-<version>/`) |
| Cache | `<cache>/<model>/...` | `<cache>/subscription/<model>/...` |
| Budget | `budget_usd` against `<cache>/llm_spend.jsonl` | nothing is billed; the API-equivalent cost goes to `llm_spend.subscription.jsonl`, capped only by `subscription.budget_usd_equivalent` |
| Dry-run tokens | free `count_tokens` with a key | character heuristic |

The two backends are different instruments: Claude Code wraps the request and enforces the
schema differently. The backend (with `subscription.tools`, `max_turns` and a pinned
`cli_version`) therefore enters every instrument digest, `prereg freeze` records it as
`llm_backend`, and a run directory is bound to the backend of its first LLM call
(`llm_backend.txt`): later commands without `--backend` adopt it, a different explicit one is
refused; the binding is written once the first response has arrived, so a failed attempt
binds nothing. A study must keep one backend from development gold to the final report.
Anthropic's terms allow the subscription login for the purchaser's own individual use only;
a shared server should use the API backend, or the login of the person who runs it.

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
schema, package version, sha256 of the task parameters (for T1 also the window parameters and
the version of the code checks, `collate/verify.py: VERIFY_VERSION`) and the number of
replicates. `prereg freeze` writes the digests into `config/preregistration.yaml`; gate G2
requires the collate digest to equal the preregistered one, and the ledger records it. Editing a
prompt, an example, the codebook, an evidence line, a schema or a parameter therefore changes
both every cache key and the digest. A change to what the code checks V1-V12 accept or flag
bumps `VERIFY_VERSION`, whether it is made in their code (`collate/verify.py`,
`collate/quotes.py`, `collate/merge.py` for V9 and V10, and the `core/textnorm.py` and
`core/translit.py` functions and thresholds they call) or in the word lists they read
(`data/lexicon/negators.yaml`, `data/lexicon/variant_forms.yaml`). The bump changes the collate
digest but no request or cache key, so cached answers replay and are verified by the new checks.
The components (T2) digest does not carry `VERIFY_VERSION`, although its checks C3 and C5 share
`collate/quotes.py` and the two lexicons.

The collator has one digest per language pair, because its examples are per pair: `collate` for
Tibetan -> Chinese (`data/codebook/collate_examples.yaml`) and `collate:<ref>-<wit>` for every
further pair that has its own `data/codebook/collate_examples.<ref>-<wit>.yaml` (with the
Sanskrit reference: `collate:sa-bo` and `collate:sa-zh`; `prereg.collate_task`). G2 and the
reveal import use the digest of the run's pair (`pipeline/instrument.py: collate_digest`).

The collate digests are those of the configured prompt version (`tasks.collate.prompt`, below):
under collate.v2 the prompt sha covers the v2 template, the v2 examples and the block of
verified standing renderings, and the schema sha the `convention` field. The version name
itself is not hashed into `params_sha` (it selects what is hashed), so the collate.v1 prompt,
schema and cache keys are exactly those of the runs made before collate.v2 existed (pinned in
`tests/unit/test_collate_v2.py`: system prompt sha `3ab88636...`, schema sha `ed99b249...`).
The collate.v1 digest is not: `VERIFY_VERSION` 2 (the 2026-10-13 revision of V4, V11 and V12,
under which an offline replay of the v1 pilot resolves one more unit and drops two V11 flags)
entered `params_sha` (pinned in `tests/unit/test_pipeline_collate_v2.py`).

## 2. The tasks

Settings below are the committed `config/llm.yaml` values.

| | T1 collate | T2 components | T3 topics | T4 subject | T5 scorer | check |
|---|---|---|---|---|---|---|
| Module | `collate/collator.py`, `collate/prompt.py` | `collate/components.py` | `topics/prelabel.py` | `experiments/overattribution/run.py` | same | `llm/check.py` |
| Prompt | `prompts/collate.v1.md` + `data/codebook/collate_examples.yaml` (default); `prompts/collate.v2.md` + `data/codebook/collate.v2/` | `prompts/components.v1.md` + `data/codebook/components_examples.yaml` | `prompts/topics.v1.md` + `data/codebook/topics.yaml` | `prompts/subject.v1.md` + `evidence.yaml` | `prompts/scorer.v1.md` | constant |
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

**Prompt versions** (`collate/prompt.py: VERSIONS`). `config/llm.yaml: tasks.collate.prompt`
selects the version; an unknown name is a configuration error.

| | `collate.v1` (default) | `collate.v2` |
|---|---|---|
| Template | `prompts/collate.v1.md` | `prompts/collate.v2.md` |
| Examples | `data/codebook/collate_examples[.<ref>-<wit>].yaml` | `data/codebook/collate.v2/examples[.<ref>-<wit>].yaml` |
| Glossary | none | `data/codebook/collate.v2/standing_renderings[.<ref>-<wit>].yaml`, required for each pair (`entries: []` allowed) |
| Schema | `SCHEMA_V1` (`collator.SCHEMA`) | `SCHEMA_V2`: `convention` before `relation` in every unit |
| Link source | `<model>:collate.v1:<tag>` | `<model>:collate.v2:<tag>` |

The task name stays `collate` for both. v1 stays the default until the dev-gold comparison of
v1 and v2 (docs-zh/09-collate-v2-draft.md, section 7) supports a switch. `collation/
replicates.json` records the version a run collated with (absent = v1) and the `prompt_sha` and
`schema_sha` of its requests, and `build`, `evaluate`, `perturb` and `review reveal` refuse a
configured version that differs, or the same version with other shas (a text revised in place:
template, examples or glossary) (`pipeline/instrument.py: check_prompt`): one run directory
holds one instrument. A run collated before the shas were recorded is checked by its
`request_keys` instead: the current plan of its chapters and replicate count must send the same
requests, so a changed template, example, glossary entry or request parameter is caught too (the
runs of the superseded collate.v2 texts are refused); a run without request keys is checked by
name only. These stages rewrite `manifest.json: instrument_digests` from the current
configuration. A switch after `prereg freeze` needs `prereg freeze --amend`.

collate.v2 adds a second dimension next to the relation, the **convention**, for differences that
are conventions of the translation rather than changes of content:

- `standing_rendering`: the unit contains a reference term of a verified glossary entry and the
  witness uses that entry's rendering (e.g. Hevajra as kong zhi jin gang); the relation is judged
  as if the rendering were the reference term.
- `transcribed_name`: the witness renders a proper name (deity, person, place, channel,
  implement) or a syllable or letter treated as sound (seed syllables, alphabet letters) by
  sound, the rest of the unit by meaning; the relation is judged as if the name had been
  translated.
- `none` otherwise. `transliterated` narrows to content words, explanatory text and mantras
  rendered by sound (a class noun such as dakini or yogini rendered by sound stays
  `transliterated`); a list merged into one collective expression is `generalised` for every
  merged member, while ordinals in place of the names stay `category_name_omitted`.

A convention only sets aside the name or rendering itself: any other loss, reversal or
replacement in the unit is still recorded, and a convention never turns one into `equivalent`.
The v2 examples illustrate ordinals vs a collective expression; a standing rendering with a
reversal and a no-counterpart unit; transcribed names vs transliterated content (a content
word, and a class noun rendered by sound as `transliterated` with convention `none`), a mantra
and a witness-only line. Each v2 example lists its own synthetic standing renderings, which never
overlap the real glossary.

**Glossary.** Only entries with `status: verified` (and `verified_by`) are shown, after the
examples, under `## Standing renderings of this witness` as one `STANDING RENDERINGS` block (one
line `form / form = rendering / rendering` per entry; `(none)` without an entry). Counts, notes
and names never reach the model. V12 reads exactly the object shown in the prompt
(`Examples.glossary`), and the review sheets list the same entries. The file names its
reference and witness text ids; a run on other texts stops with an error. Only a change to the
set of verified entries changes the prompt, hence the digest and every cache key: verifying an
entry, or editing, retiring or removing a verified one (an instrument change: redo the dev-gold
comparison, `prereg freeze --amend` after the freeze). Adding, editing, retiring or removing a
`proposed` entry changes nothing the model sees (`load_examples` still refuses a form of any
entry that collides with an example's own standing renderings). Every entry needs `status`;
`verified_by` is a non-blank name for `verified` and null otherwise. Verifying an entry trips
`tests/unit/test_collate_v2.py::test_the_shipped_glossary_has_no_verified_entry_so_the_block_reads_none`,
which pins the shipped empty glossary and is updated in the same commit
(`data/codebook/collate.v2/README.md`). The committed file
lists only `hevajra`, still `proposed`, so the active glossary is empty and every
`standing_rendering` claim is flagged `convention_unsupported` until an expert signs it.

**Input** (`collate/windows.py`, `collator.build_request`). One window = one chunk of one
reference chapter (at most `windows.max_ref_units` = 150 units, the last `overlap` = 10 repeated
at the start of the next chunk) plus the whole witness text. On the real texts this gives 35
windows for 3,042 reference units in 23 chapters.

- `system`: the template, then the three synthetic examples (v1: a list whose category names
  become ordinals, a reversal, a witness-only addition; v2: see above), then under v2 the
  STANDING RENDERINGS block. Examples never reuse sentinel or test passages
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
of the 4 `WitnessOnlyKind` values. `SCHEMA_V2` adds `"convention"` (`none`,
`standing_rendering`, `transcribed_name`) to every unit, before `relation`. An answer is
validated against the schema actually sent. A test asserts every enum value appears in the
template and the schema of its version.

**Checks** (`collate/verify.py`, `collate/merge.py`, all pure). Fatal failures make a unit
UNALIGNED with a reason; non-fatal ones keep the link with a flag (any flag except
`corroborated` makes the consensus grade C).

| Check | Rule | On failure |
|---|---|---|
| V1 coverage | every reference handle exactly once | duplicate: keep first, flag `duplicate_record`; missing: `unassessed`; more than 2% missing: whole window `invalid` |
| V2 existence | every handle exists | unknown handle dropped, flag `unknown_handle`; nothing left where a witness is required: `verification_failed` |
| V3 consistency | `no_counterpart` exactly when `wit` is empty | `verification_failed` |
| V4 verbatim quotes | `ref_quote` in the unit, `wit_quote` in the linked segments (`core.textnorm.quote_in`; a Tibetan quote covers whole syllables but may stop where the letters of a particle fused to its last syllable begin, -s, -r, -'i, -'o, -'ang, -'am, as V12 does: the unit's -por quoted as -po. The test is on letters alone, so a final -s or -r is dropped whatever it is: la for las, na for nas and rnam for rnams pass, because spelling cannot tell a fused particle from a final letter; any other letters left out of a syllable fail. After a final a-chung the particle's own a-chung is the quote's (nam mkha' in nam mkha'i); where -r or -s replaces the a-chung (sems dpar) the quote stops where the text's letters stop (sems dp), never at the restored citation form. The only consumer that reads meaning from a Tibetan quote, the negator check (V6, C5), reads it completed to whole syllables; `collate/quotes.py: quote_match`, also used by C3) | no match: `verification_failed`; match only after variant folding: flag `quote_variant_form` |
| V5 required quotes | `ref_quote` for every relation except equivalent, paraphrase, expanded; `wit_quote` whenever `wit` is non-empty | `verification_failed` |
| V6 polarity | `reversal` exactly when `polarity_flip`; a negator in exactly one quote corroborates (a Tibetan quote read completed to whole syllables of its text, `collate/quotes.py: completed_quote`: mi cut from mi'am is no negator; a quote that stops at a syllable boundary inside an exclusion context, mi quoted from mi rnams, still reads a negator: known and left open) | flags `polarity_mismatch`, `corroborated` / `polarity_uncorroborated` |
| V7 locality | links outside the core window | flag `relocation` and a `relocation` diagnostic |
| V8 order | units off the longest order-preserving chain | flag `crossing`; crossing count in a diagnostic; never removed |
| V9 exhaustiveness | after merging a chapter's chunks, every core-window witness segment is linked or witness-only | `unaccounted` diagnostic |
| V10 overlap | the 10 overlap units read in two chunks are compared | flag `overlap_disagreement`; agreement rate in `collation/r<k>.json` |
| V11 transliteration | `transliterated` needs a witness quote that contains a transcription: at least 50% transcription, or `MIN_TRANSCRIBED_RUN` (2) transcription characters together (`core/translit.py: contains_transcription`), since one quote also shows the rest of the unit (an ordinal, a gloss); under v2 `transliterated` covers content and mantras only | flag `transliteration_unsupported`; prose unit linked to a mantra segment: flag `instruction_as_mantra` |
| V12 conventions (v2) | the claimed convention becomes a link flag of the same name. `standing_rendering`: both quotes non-empty, and one verified entry has a reference form in `ref_quote` and one of its own renderings in `wit_quote` (quote keys, variant forms folded; Tibetan matched from a syllable start, the last syllable may stop where a fused particle -s, -r, -'i, -'o, -'ang, -'am begins, by V4's letter test). `transcribed_name`: a non-empty `ref_quote` and a `wit_quote` that passes the V11 test | flag `convention_unsupported` beside the convention flag; never fatal; grade C |

Witness-only records are checked for V2, V4 and V5 as well. A refusal, truncation, invalid
answer or substituted model makes every unit of the window UNALIGNED with reason
`refused:<category>` (`refused:unspecified` without a category, as in T2 and T3), `truncated`,
`invalid` or `substituted_model`.

**Consensus** (`collate/consensus.py`): per unit, a class (or "unresolved") needs two thirds of
the k replicates; the relation is the most frequent among the majority (ties broken by
`matrix.status.RELATION_PRIORITY`); the link comes from the medoid replicate; flags are the
union. No majority gives UNALIGNED(`no_majority`). Grade B = all replicates agree and no flag
other than `corroborated` and the convention flags; C otherwise; X for UNALIGNED. The convention
is voted separately, not taken from the flag union: a convention needs two thirds of the k
replicates among the resolved readings, else `none`; the medoid is taken among the majority
readings with the voted relation, preferring those holding the elected convention.
`convention_unsupported` is kept only when a reading holding the elected convention carried it
(an outvoted claim is dropped), and it is also set by the consensus itself when no majority
reading with the voted relation holds the elected convention: the medoid's own quotes were then
never claimed for that convention. Convention flags never lower a grade; `convention_unsupported`
gives C.
`collation/integrity.json: convention_kappa` (Fleiss kappa of the replicates' conventions) is
descriptive, never gated. A v1 consensus (every reading `none`) is byte-identical to the one
made before collate.v2.

**Conventions in the measurement.** The relation-to-status mapping is unchanged and a convention
never changes a status, D_any or D_cov; the primary outcome stays `any` or `cov`. The outcome
`any_conv` (D_any_conv: any deviation, or any transcribed, part-transcribed or standing-rendered
name in the unit) is computed only as the E1 sensitivity estimate `E1_any_conv` (final column)
and, for a collate.v2 run, its Manski bounds. It is wider than collate.v1's D_any (measured on
I.1 in docs-zh/09 section 9.2) and is not directly the caliber of published counts that include
transcriptions (omission + rewriting + transliteration). `E1_any_conv` imputes the convention
jointly with the outcome class (`stats/anyconv.py`): verified units keep their human class and
human convention, and every unverified unit drawn non-deviating carries a convention with the
share its stratum's phase-2 sample verdicts record. Machine conventions never enter (as
machine classes do not), so the estimate is valid for a collate.v1 run too, where conventions
come in through the human verdicts only, and on a v2 run the machine's convention errors are
calibrated by the sample. A sample verdict without an annotated convention counts as none, and
so does a gold or resolve decision without one; `stats/details.json: any_conv` reports both
counts. It equals `E1_any` when no non-deviating verified unit and no non-deviating phase-2
sample verdict records a convention.
The Manski bounds of `any_conv` read the built cells and are computed only for a run collated
with collate.v2. Human gold and verdicts record the relation and the convention separately, so
both calibers come from one gold. On gold, `evaluate` adds `relation_kappa`, `relation_kappa_shared`, `dany_conv_kappa`, convention
precision, recall and F1 (overall and per convention) and `convention_unsupported_rate`.
`relation_kappa`, `relation_kappa_shared` and `dany_conv_kappa` are computed for every source (a
source without conventions is read by its own D_any for D_any_conv), which is what makes the
v1-vs-v2 comparison possible; only the convention precision, recall, F1 and
`convention_unsupported_rate` are null for sources that produce no conventions (the controls,
the placebo, external alignments, a v1 consensus). `relation_kappa_shared` is the relation kappa
on the units the v1 and v2 codebooks code alike (linked gold units whose gold convention is
annotated `none` and whose gold relation is neither `generalised` nor `category_name_omitted`);
the adoption rule of docs-zh/09 section 7 compares it, while the overall `relation_kappa` is
descriptive (a v1 source scored on v2 gold is structurally lower on it). That rule, fixed on
2026-10-13 before any dev result, is decided by `hevajra-matrix adopt` (below) on the dev units
outside I.1, where v2 was tuned, with each gold `?` read by its note marker (`scores.dev.json:
without_window["I.1"]` leaves every `?` out, so it matches only adopt's sensitivity figures; all
dev units are reported, descriptive only; the three I.7 units
D417:8b.5.2-8b.5.4 the second v2 revision was written for are in-sample too, so convention
precision is also reported without them). v2 is adopted when (1) its `relation_kappa_shared`
is at least v1's minus 0.05, a point-estimate tolerance, not a non-inferiority test (about 1.5
standard deviations of the run-to-run difference between two 3-replicate consensuses, and
about one sampling standard error); (2) its convention precision is at least 0.8; (3) a
replicate-kappa floor symmetric in the two versions holds, read from the dev run's
`collation/integrity.json: fleiss_kappa` (all collated units of the dev chapters, I.1
included): if v1 reaches 0.80, v2 must too; if v1 does not, the floor does not decide (the version
chosen is then tuned on the dev set before `prereg freeze` if it is below 0.80). The three I.7
units stay in condition (2) because three of about 262 units can move precision only a little,
and the separate figure shows whether they do. Any prompt or effort tuning after the comparison
first names the regions it may look at, and the comparison is then redone on the other regions
only. A `convention_unsupported` on a consensus link may come from the consensus rule above rather than
from a replicate's V12, and then counts in `convention_unsupported_rate` too. No gate uses a convention metric. Sentinels of kind `relation_in` may restrict conventions with
`expect.conventions`.

**Applying the adoption rule.** One command applies it once the dev gold exists and both dev
runs are collated and built: `hevajra-matrix adopt --v1-run <v1 run dir> --v2-run <v2 run dir>`
(`evaluation/adoption.py`, `pipeline/adoption.py`; `--run-dir` does not apply). It reads the
version each run collated with from its `collation/replicates.json`, so `config/llm.yaml` need
match neither run, and refuses (exit status 2): a run of the other version or not collated; a run
collated with another text of its version (its recorded `prompt_sha` and `schema_sha` differ from
the current template, examples and verified glossary, e.g. after verifying a glossary entry, or
are missing); a run whose chapters are not exactly the preregistered dev chapters or whose
replicates are not 3; a run without a consensus (`build` not run); two runs of different texts;
and a missing dev gold. It scores both consensuses on the dev gold as `evaluate` does, takes the
floor from the preregistration's `min_replicate_kappa`, prints each condition with its numbers
and the decision (`adopt collate.v2`, `keep collate.v1` or `not ready`, the last with exit status
1), and writes `evaluation/adoption.json` into the v2 run. Deferred gold cells (`?`, stored as
convention "" with a note marker) are read by `data/codebook/collate.v2/note_markers.yaml`: a
part-transcribed name (Q7) as `transcribed_name`, an explained word (Q8) and a class noun (Q3) as
`none`; this default reading is primary, and the same conditions with the `?` rows left out are
reported as a sensitivity result. The decision is `not ready` while a dev region is not (fully) in
the gold, while a `?` marked for the Hevajra glossary entry remains (until the entry is verified
or rejected and the `?` replaced by a decided convention), while a `?` carries no known marker,
or when a value of condition (1) or (2) is undefined or a replicate kappa is missing. Also
reported, descriptive only: the figures on all dev windows, convention precision without the
three I.7 units, convention recall, `dany_kappa`, `dany_conv_kappa`, the D_any level of each
version against the gold level over the shared units, and the paired window-bootstrap interval of
the v2-v1 difference in `relation_kappa_shared` outside I.1.

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
no fallback) straight to the backend's client through the audit log but not the cache, and writes
`claude_check.json` (with the backend and, on the subscription backend, `claude --version`).
`tools/server_check.sh [api|subscription] [--pilot]` runs fetch, ingest, `claude-check` and
`collate --dry-run` in one run directory on a machine with Claude access and packs the
text-free results (check summary, projection, audit log, manifest, G0) into one archive. It passes only for `{"ok": true}` served by the requested model. Run it
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
