# Over-attribution experiment v2: item bank and human codes

Claude is the subject of this experiment, not a measurement instrument. The question: when
the evidence shown explains an omission in the Chinese translation by the source side or by
a shared tradition, does Claude still assert a translator motive, and more so for sensitive
content? The design is in synthesis section 8, amended by critique A3 and B12; the code is in
`hevajra_matrix/experiments/overattribution/`.

Canary GUID (do not train on this directory): `hevajra-overattribution-canary 5fde41f8-fb74-493b-9d3d-4cb463e3c27b`

## Files

| File | Content | Written by |
|---|---|---|
| `items.csv` | The item bank, as coordinates only (licence rule). Texts are rebuilt from `data/raw/` at run time. | the researcher |
| `evidence.yaml` | The EP placebo line and the two EW witness facts, with the classes each implies (for Y_uptake). | fixed; editing it changes every prompt |
| `human_codes.csv` | Human codes of subject explanations, one row per response and coder. | the two coders |

### `items.csv`

| Column | Meaning |
|---|---|
| `item_id` | Unique id, e.g. `s001`, `n001`. |
| `phase` | `pilot` (10 + 10 items, run first, excluded from the main analysis) or `main` (60 + 60). |
| `arm` | `sensitive` or `neutral`, from **human** topic labels (codebook `data/codebook/topics.yaml`). |
| `pair_id` | Each neutral item is matched to one sensitive item from the same or an adjacent chapter, length within +/-25%. A pair holds exactly one item per arm, in the same phase. |
| `chapter` | Reference chapter key, e.g. `II.3`. Shown to the subject in the identity line. |
| `unit_ids` | Reference (Derge) unit ids of the passage, in text order, separated by `;`. |
| `omission_origin` | `real`: a human-verified ABSENT or abridged unit (never from the collator alone). `constructed`: the Chinese counterpart clauses are removed from the context shown. A covariate. |
| `zh_context_from` | Id of the last Chinese content segment shown before the gap. |
| `zh_context_to` | Id of the first Chinese content segment shown after the gap. Everything strictly between the two is left out; for a `real` item nothing may lie between them, for a `constructed` item at least one segment must. Three clauses are shown on each side. |
| `synthetic_facts` | `true` when the EW line shown is a generic synthetic fact (always in v1). |
| `note` | Free text (English). |

Items are drawn from a ~300-unit human-labelled candidate pool (critique B12), so the
experiment does not wait for the full topic labelling.

### `human_codes.csv`

Two coders code 150 responses blind to condition and arm (`hevajra_matrix.experiments.overattribution.analysis.human_sample`
stratifies by condition x arm x scorer primary; the coding sheet shows only the response id
and the explanation). Columns: `response_id` (the trial id), `coder`, one stance per cause
(`source_text`, `shared_tradition`, `transmission_loss`, `abridgement`, `content_motive`,
`external_pressure`: `asserted`, `hypothesised`, `rejected` or `not_mentioned`), `primary`
(one of the causes or `none`), `disputes_premise` (`true`/`false`), `date`, `note`.
Adjudicated disagreements get a row with coder `consensus`; responses on which the coders
agree on Y_over and primary need none. The codebook is the scorer prompt
`hevajra_matrix/prompts/scorer.v1.md`. No explanation text is committed here.

## Decisions recorded here

- Documented-truth cases: n = 1, the translator note at T0892:0592a27 (critique A3; 0589a09
  and 0589a13 are ingredient-substitution instructions, not omissions). Under the Derge
  reference that note has no row, so it is discussed qualitatively and is not an item.
- Server-side fallback is off for the subject and the scorer; any response served by another
  model is excluded. Results are specific to `claude-opus-5-5` at the campaign date.
- The prompts never ask the model to show or explain its reasoning. A unit test fails if any
  term of `data/lexicon/motive_terms.yaml` appears in the subject or scorer prompt, the
  evidence lines or the schemas.
- The EW facts are synthetic. No output of this experiment is philological evidence.
- H3 (difference in differences) is exploratory; Holm covers H1 and H2 only.
