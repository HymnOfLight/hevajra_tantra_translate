# hevajra_tantra_translate

Validated, evidence-graded collation of the *Hevajratantra* witnesses: the Tibetan
Derge translation (Toh 417-418), the Song Chinese translation (Taisho T0892) and,
when supplied, Sanskrit editions and manuscripts.

Chinese-language documentation for researchers: [`docs-zh/README.md`](docs-zh/README.md).

## Method in one paragraph

Claude Opus 5.5 reads each reference chapter against the Chinese witness and proposes,
for every unit, its counterpart, the textual relation (equivalent, abridged, substitution,
reversal, category name omitted, transliterated, no counterpart, witness-only material)
and verbatim quotes. Code verifies every id and every quote. Humans decide, blind first.
No rate is reported until the instrument passes a pre-registered gate on a sealed,
probability-sampled blind test set and beats content-free controls (length-only
alignment, shuffled links) on the same units. Prevalence estimates are corrected for
misclassification with a two-phase design (verification of machine positives plus a
stratified audit of machine negatives). Status is derived from content evidence only,
never from length. No model attributes motives inside the pipeline; whether models
over-attribute motives is studied in a separate experiment.

## Layout

| Path | Content |
|---|---|
| `hevajra_matrix/` | Python package (English only) |
| `tests/` | `unit/` (default), `integration/` (real texts), `live/` (real API) |
| `config/` | Run parameters, Claude settings, pre-registration |
| `data/` | All multilingual research data; see [`data/README.md`](data/README.md) |
| `docs/` | English developer documentation |
| `docs-zh/` | Chinese user documentation |

## Quick start

```bash
pip install -e ".[dev]"            # add ",claude" to install the Anthropic SDK
hevajra-matrix fetch               # CBETA T18n0892.xml + Esukhia Derge vol. 80 into data/raw/
hevajra-matrix run                 # every stage that is possible with the data at hand
pytest                             # unit tier
HEVAJRA_RAW_DIR=data/raw pytest    # plus real-data checks
```

With an API key (`ANTHROPIC_API_KEY`), start with `hevajra-matrix claude-check`, then
`hevajra-matrix collate --dry-run` to project the cost before any real collation.

## Licence boundary

Only coordinates, labels, numbers, code and short quotations are committed. Source
texts, Sanskrit editions, model weights and the LLM response cache are not.
