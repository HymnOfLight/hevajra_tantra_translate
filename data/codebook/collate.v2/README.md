# collate.v2 codebook

The codebook of the T1 collator's prompt version `collate.v2`, prompted by the I.1 pilot
of 2026-10-10 (docs-zh/08-pilot-I1.md). It is used when `config/llm.yaml` sets
`tasks.collate.prompt: collate.v2`; the configured default stays `collate.v1`
(`hevajra_matrix/prompts/collate.v1.md` with `../collate_examples.yaml`) until the
development-gold comparison of v1 and v2 (docs-zh/09-collate-v2-draft.md, section 7)
supports the switch.

| File | Content |
|---|---|
| `examples.yaml` | three synthetic bo -> zh examples in the v1 format plus a `convention` per unit and per-example `standing_renderings` |
| `standing_renderings.yaml` | the glossary of the witness's standing renderings against the reference, with counts and the terms considered but not listed |
| `examples.<ref>-<wit>.yaml`, `standing_renderings.<ref>-<wit>.yaml` | the same for another language pair (e.g. `sa-zh`); both files are required for a pair to have a v2 instrument |

The template is `hevajra_matrix/prompts/collate.v2.md`: v1 plus two conventions
(`standing_rendering`, `transcribed_name`), a narrower `transliterated`, and the list rule
separating `category_name_omitted` from `generalised`. The answer schema adds a
`convention` to every unit.

## The glossary

Only entries with `status: verified` (and a `verified_by`) are shown to the collator,
after the examples, under STANDING RENDERINGS; while none is verified the block reads
`(none)`, and every `standing_rendering` claim is then flagged `convention_unsupported` by
V12. The code check V12 reads exactly the entries shown in the prompt. Verifying,
adding or retiring an entry changes the prompt, so it is an instrument change: a new
digest and a new cache (and `prereg freeze --amend` after the freeze). Counts, notes and
the verifier's name are never shown to the model.

The file names its reference and witness text ids. The glossary belongs to that witness:
a run whose texts differ from them stops with an error rather than using another
witness's renderings.

The committed file lists one entry, `hevajra`, with `status: proposed`: the active glossary
is empty until an expert signs it.

## Verifying an entry

1. Check the entry's counts and passages against the texts.
2. Set `status: verified` and `verified_by: <real name>` (`verified_by` is required for, and
   only for, `verified`). To reject an entry, set `status: retired` or move it to
   `considered_not_listed` with a `reason`.
3. The whole file is validated on load (`core/lexicon.py: load_standing_renderings`); run
   `python -m pytest -q tests/unit/test_collate_v2.py`.
4. The new prompt is a new instrument: collate a new run, repeat the dev-gold comparison, and
   after `prereg freeze` record the change with `prereg freeze --amend "<reason>"`.

The review sheets list the same verified entries (`standing_renderings.csv` beside every gold,
blind and reveal sheet), and a human `standing_rendering` decision is checked by the same rule
as V12.

## Switching to collate.v2

Set `tasks.collate.prompt: collate.v2` in `config/llm.yaml` and collate a new run directory: the
v2 requests have new cache keys, and a run directory keeps the version it collated with
(`evaluate`, `perturb` and `review reveal` refuse a configured version that differs). A
language pair other than Tibetan -> Chinese needs both `examples.<ref>-<wit>.yaml` and
`standing_renderings.<ref>-<wit>.yaml` here. See docs/llm-tasks.md (T1, prompt versions) and
docs-zh/09-collate-v2-draft.md, section 6.

## Provenance

Links of a v2 run carry the source `<model>:collate.v2:<replicate>`, and
`collation/replicates.json` records the prompt version (absent means collate.v1).
