# Contributing

Rules and recipes for working on `hevajra_matrix`. The package is meant to be maintained for
years by one or two researchers, so the rules favour code that is easy to read and to check
over code that is clever. See [architecture.md](architecture.md) for the module map,
[data-formats.md](data-formats.md) for file formats and [llm-tasks.md](llm-tasks.md) for the
Claude tasks.

## 1. Setup

```bash
pip install -e ".[dev]"            # PyYAML + pytest; the package imports without the SDK
pip install -e ".[dev,claude]"     # adds the official anthropic SDK (only llm/anthropic_client.py uses it)
hevajra-matrix --help              # or: python -m hevajra_matrix --help
```

Python >= 3.10. Runtime dependencies are the standard library and PyYAML; `anthropic` is
optional and imported lazily. Do not add dependencies (no pydantic, numpy, jsonschema, pandas):
the statistics, schema validation and bootstraps are small enough to keep in plain Python.

## 2. Language policy

| Where | Rule |
|---|---|
| `*.py` (comments, docstrings **and** string literals) | no non-Latin script |
| `*.md` outside `docs-zh/` (including `hevajra_matrix/prompts/*.md` and these docs) | no non-Latin script |
| `config/*.yaml` | no non-Latin script at all |
| YAML under `data/` | keys and comments English; values may be any language |
| file and directory names | ASCII |
| `data/annotations/` | no run of more than 60 non-Latin characters (licence rule: ids and short quotes only) |

"Non-Latin" means CJK (including CJK punctuation and fullwidth forms), Tibetan, Devanagari,
Mongolian and Tangut code points. IAST transliteration (Latin with diacritics) is allowed
everywhere, so Sanskrit can be written as *śmaśāna*; Tibetan in docs and comments is written
in Wylie (*stong pa nyid*), Chinese in pinyin or by description ("the Chinese term for 'cold
forest'"), or referred to by its data file.

**Where multilingual material goes.**

- Linguistic resources: `data/lexicon/*.yaml`, loaded by `core/lexicon.py` or the module that
  owns the file.
- Codebooks and few-shot examples: `data/codebook/`.
- Test inputs: `data/fixtures/` (YAML, CSV, TXT, XML), loaded by the tests.
- A single script constant in code: a `\u` escape with an English comment, e.g.
  `TSHEG = "\u0f0b"  # Tibetan intersyllabic tsheg`.

**Enforcement.** `tests/unit/test_language_policy.py` checks all of the above on every test run
(one parametrised test per `.py` file, plus Markdown, config YAML, data YAML keys and comments,
file names and the annotation licence scan). Before committing a `.py` file, also run
`grep -nP '[^\x00-\x7f]' <file>`: prefer pure ASCII in code even where IAST would pass. Chinese
user documentation lives only in `docs-zh/`.

## 3. Tests

### Tiers and markers

| Tier | Location | Marker | Runs when | What it needs |
|---|---|---|---|---|
| unit | `tests/unit/` | none | always (default) | nothing: no network, reads `data/fixtures/`, `data/lexicon/` and the committed data |
| realdata | `tests/integration/` | `@pytest.mark.realdata` | `HEVAJRA_RAW_DIR` names a directory | the fetched source texts |
| live | `tests/live/` | `@pytest.mark.live` | `ANTHROPIC_API_KEY` is set (and the SDK installed) | one `claude-check`-sized API call, a fraction of a cent |

`tests/conftest.py` adds the skip markers, so a plain `pytest` runs every tier and skips what it
cannot run. The unit tier has about 1,960 tests and takes about half a minute.

```bash
pytest                                        # unit tier (others skipped)
pytest tests/unit/test_collate_verify.py -q   # one area
pytest tests/unit/test_language_policy.py     # the language policy only

hevajra-matrix fetch                          # CBETA T18n0892.xml + Esukhia Derge vol. 80 -> data/raw/
HEVAJRA_RAW_DIR=data/raw pytest tests/integration -m realdata

ANTHROPIC_API_KEY=... pytest tests/live -m live    # never in CI
```

The realdata tier checks the ingest counts (521 notes in 7 classes, 181 footnotes of which 16
from Tokyo 335, 3 paratext segments, 26 edit marks, 23 Derge chapters, 0 duplicate ids), the
ingest-stage sentinels, that every sentinel locus resolves, the baselines (all chapters in under
60 seconds, anchor asymmetry far below v0.2), the dev regions and test windows, and a level-0
report on the real texts.

Claude-dependent behaviour is tested in the unit tier with `llm/fake.py`:
`FakeClient(script)` plus `ok_response`, `refusal_response`, `truncated_response`,
`invalid_response` and `substituted_response`, which build responses exactly as
`AnthropicClient` would. A dict returned by a script must be valid against the request schema;
return `invalid_response(...)` to test a malformed answer. `tests/unit/test_pipeline_e2e.py` runs
the whole pipeline on fixtures with an echo collator and replays it `--offline` from the cache.
`tests/unit/test_llm_anthropic_sdk.py` exercises the real SDK against an in-process mock
transport and skips when `anthropic` (or its HTTP mock) is not installed.

### Writing tests

- File names `tests/unit/test_<area>_<topic>.py`; realdata tests in `tests/integration/` with
  `@pytest.mark.realdata` and the `raw_dir` fixture.
- Multilingual inputs go in `data/fixtures/`, never in the test file.
- Test behaviour through public functions; hand-computed expected values for statistics; one
  test per rule for verification code (V1-V11, C1-C5, gate conditions).
- Every refusal path, truncation, invalid answer and substituted model needs a test where a
  task handles LLM output.
- Randomised code takes an explicit seed and is tested for reproducibility.

## 4. Style

- **KISS.** Plain functions and frozen dataclasses. No frameworks, no DAG engine, no
  meta-programming (enums are written out; a test checks that enum, prompt and schema agree).
- **Functional core, thin shell.** Domain logic is pure functions over frozen dataclasses
  (`@dataclass(frozen=True)`, tuples and `frozenset` instead of lists and sets, `MappingProxyType`
  for read-only maps). File and network I/O happen only in `core/io.py`, the loaders,
  `llm/cache.py`, `llm/audit.py`, `llm/anthropic_client.py` and `pipeline/`.
- **SOLID, sparingly.** One responsibility per module (<= about 400 lines). A `Protocol` only
  where two real implementations exist (`LLMClient`, `Similarity`). Each consumer owns a small
  frozen parameter dataclass built with `config.from_mapping`, which rejects unknown keys; there is
  no central configuration object.
- **Fail loudly.** Loaders validate strictly and name the file, line and allowed values. A check
  that cannot be evaluated fails rather than passes. Research outcomes (a refusal, an UNALIGNED
  unit, a NOT_ESTIMABLE estimand) are values, never exceptions and never zeros.
- **Honest docstrings.** English; say what the function does and why, state assumptions, and say
  what is not done. Comments are sparse and explain reasons, not syntax.
- **Type hints** on every public function. `from __future__ import annotations` at the top.
- **Frozen contracts.** `core/types.py`, `llm/client.py`, `config.py`, `matrix/status.py`,
  `config/*.yaml`, `tests/conftest.py` and `tests/unit/test_language_policy.py` are shared by every
  module; change them deliberately, in their own commit, and update every consumer.

## 5. Recipes

### Add a new witness

The pipeline currently handles one reference (Derge, `run.yaml: witnesses.reference`) and one
target (T0892, `witnesses.target`); `pipeline/texts.py` calls `ingest/derge.py` and
`ingest/cbeta.py` directly. A new witness therefore needs code, not only data:

1. **Registry.** Add an entry to `data/registry/witnesses.yaml` (required `id, lang, layer, role,
   status, title`; say what is unverified in `notes`). For a Sanskrit manuscript that will feed
   E3, add it to `data/registry/sa_manuscripts.yaml` instead, with `to_verify` for unknown facts.
2. **Ingester.** Write `ingest/<source>.py` with `parse(path, witness, data_dir) -> IngestResult`.
   Build ids with `core.ids.assign_ids` from the witness's own coordinates (never a running
   counter), set `fingerprint` with `core.textnorm.fingerprint`, keep every note, footnote and
   resolved reading (`Footnote`, `Variant`, `kind="note"` or `"paratext"`), fill `report` with
   the counts G0 should check, and set `local_chapter`. Put any script-specific markers in
   `data/lexicon/`. A Sanskrit reference already has `ingest/sanskrit.load_reference`.
3. **Concordance.** Add the witness's `locals` and its `spans` for every reference chapter in
   `data/registry/concordance.yaml`, each with `status`, `evidence` and `source`.
4. **Pipeline.** Extend `pipeline/texts.raw_paths` and `ingest` (and `run.yaml: sources`) to parse
   the new file; for a new reference, switch `witnesses.reference` and make sure
   `Concordance.assign_reference_chapters` sets `Segment.chapter`.
5. **Prompts.** T1's few-shot examples are tied to one language pair; `collator.build_request`
   refuses a window whose languages differ from `data/codebook/collate_examples.yaml`. A new pair
   needs its own synthetic examples (and a new instrument digest).
6. **Tests and G0.** A mini fixture in `data/fixtures/` with unit tests for each structure, a
   realdata test for the counts, ingest-stage sentinels for facts you have verified, and the
   expected counts in `config/preregistration.yaml: gates.g0.expected` (witness -> report key ->
   count; the default is `evaluation/gate.DEFAULT_G0_EXPECTED`).

### Add a new aligner or control

Every aligner produces a `core.types.Alignment` with a unique `source` string; that is all
scoring needs (`evaluation/gold.score`).

- **An alignment computed elsewhere** (MITRA-E, DharmaNexus, a spreadsheet): convert it to our
  segment ids and import it, no code needed:
  `hevajra-matrix baselines import --tsv my.tsv --name mitra-e`. Format: `ref_id, wit_ids,
  relation` (see `align/external.py`). It becomes control `external:mitra-e`, is scored on the
  same gold units as every other control, and competes as "best control" in G1.
- **A new DP variant:** implement the `align.similarity.Similarity` protocol (`score(ref, wit)` in
  [0, 1] and `conflict(ref, wit)`), add it with a new `SOURCE_*` name and output file to
  `pipeline/texts.baselines` and `BASELINE_FILES`, and add a label in
  `report/markdown.SOURCE_LABELS`.
- **Anything else:** a pure function `(...) -> Alignment`, a stage that writes it with
  `pipeline.store.write_alignment` into `alignments/`, and registration in
  `pipeline/measure.control_alignments`.

Controls never populate the matrix (`matrix/build.py` takes the Claude consensus only). Test the
aligner on `data/fixtures/` and score it against a small synthetic gold set.

### Add a new LLM task

Follow the shape of `topics/prelabel.py` or `collate/components.py`:

1. **Settings.** A frozen dataclass with `from_config(llm)` reading `config/llm.yaml:
   tasks.<task>` through `config.from_mapping` (`effort, max_tokens, replicates, fallback`, plus
   task keys). Decide the fallback policy explicitly: a task whose output is ever *measured* must
   refuse `fallback: true`.
2. **Prompt.** `hevajra_matrix/prompts/<task>.v1.md`, English, static content first. Never ask
   the model to show or explain its reasoning; never put motive vocabulary
   (`data/lexicon/motive_terms.yaml`) into a pipeline prompt; no coordinates, only prompt-local
   handles. Few-shot examples are synthetic and live in `data/codebook/`.
3. **Schema.** A module-level `SCHEMA = strict({...})` without free-text fields where the answer
   feeds the pipeline; every claim about the text carries a verbatim quote that code can check.
4. **Pure request builder.** `build_request(...) -> LLMRequest` with `prompt_sha =
   sha256_text(system)`, stable prefix in `system` (and `context` when large and shared), the
   variable part in `body`. Build it from the minimal inputs so blindness holds by construction
   (e.g. accept reference segments only), and test that forbidden text cannot reach the request.
5. **Pure parse and verify.** Never raise on a model answer: `refusal`, `truncated`, `invalid` and
   `response.substituted_model` become values with a reason. Check every handle and quote in code
   (`core.textnorm.quote_in`, `collate.verify.quote_match`).
6. **Instrument.** Add the task to `prereg.instruments` so its digest is frozen and recorded.
7. **Stage and CLI.** A function in `pipeline/` that uses `make_client(ctx)` (audit, cache, budget,
   offline replay come for free), writes its outputs into the run directory and calls
   `record_stage`; a subcommand in `cli.py`; a task block in `config/llm.yaml`.
8. **Tests** with `FakeClient`: every status path, the substituted-model path, the verification
   rules, the request layout and blindness, "no reasoning requested", and a cache replay.
9. **Docs.** A section in [llm-tasks.md](llm-tasks.md) and its output files in
   [data-formats.md](data-formats.md).

### Change a prompt, codebook, schema or task parameter

Any such change alters every cache key of the task and its instrument digest. Before the
preregistration is frozen this is routine (old cache entries simply stop being hit). After
`prereg freeze`, record the change with `hevajra-matrix prereg freeze --amend "<reason>"`, which
recomputes the digests and appends to `amendments`; the ledger then shows the test set scored by
more than one instrument version and the report reads "exploratory".

## 6. Commit hygiene

- **Never commit** `data/raw/`, `data/reference/*` (except its README), `runs/` (run outputs, review
  sheets, the LLM response cache), or any file with source text beyond short quotes. They are in
  `.gitignore`; the annotation licence scan backs this up for `data/annotations/`.
- **Committed human data is append-or-amend only.** `data/ledger/test_evaluations.jsonl` is
  append-only and written only by `evaluate`. Gold, verdict and topic CSVs are written by
  `review import` / `topics import`, not by hand; if you must hand-edit one, run the unit tests
  (the loaders validate) and say so in the commit message.
- **Preregistration.** `config/preregistration.yaml` is changed after freezing only through
  `prereg freeze --amend`; commit the frozen file together with the code version whose digests it
  holds.
- **One concern per commit.** Keep frozen-contract changes, data changes (lexicon, registry,
  sentinels) and code changes in separate commits where possible; a sentinel moves from `proposed`
  to `verified` only with `verified_by` naming the researcher.
- **Before committing:** `pytest` (unit tier) green, the language-policy test green, and the
  realdata tier when ingest, alignment, sentinels or the concordance changed.
- **Messages** in English, imperative mood, saying what changed and why. Cite the review defect
  (#n), critique amendment (A1-A7, B8-B16) or open question (Qn) a change resolves.
- The v0.2 code is preserved at commit `1f9474c`; refer to it by hash rather than restoring files
  from it wholesale.
