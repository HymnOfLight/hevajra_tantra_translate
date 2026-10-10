# collate.v2 draft (not in use)

A proposed revision of the T1 collator's codebook and prompt, prompted by the I.1 pilot
of 2026-10-10 (docs-zh/08-pilot-I1.md). Nothing in this folder is read by the pipeline:
the active instrument is still `hevajra_matrix/prompts/collate.v1.md` with
`data/codebook/collate_examples.yaml`, and `tests/unit/test_collate_v2_draft.py` checks
that it stays that way while keeping the draft well-formed.

| File | Content |
|---|---|
| `prompt.md` | the full v2 prompt: v1 plus two conventions (`standing_rendering`, `transcribed_name`), a narrower `transliterated`, and the list rule separating `category_name_omitted` from `generalised` |
| `examples.yaml` | three synthetic examples in the v1 format plus a `convention` per unit and per-example `standing_renderings`; they pass the v1 code checks V1-V11 |
| `standing_renderings.yaml` | the proposed glossary of the witness's standing renderings, with counts and the terms considered but not listed; only `verified` entries would be shown to the collator |

The rationale, the questions for the researcher and the implementation plan are in
`docs-zh/09-collate-v2-draft.md` (Chinese). Adopting the draft is an instrument change:
new prompt, schema and examples mean new instrument digests, a new cache, and the
development-gold comparison of v1 and v2 described there, before `prereg freeze`.
