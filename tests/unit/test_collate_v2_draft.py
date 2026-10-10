"""The collate.v2 DRAFT (data/codebook/drafts/collate.v2/) stays well-formed and inert.

Nothing in the draft is read by the pipeline yet. These tests keep it ready for adoption:
its examples pass the code checks V1-V11 that v1 examples pass (with the new
`convention` field set aside, since the checks do not know it yet), never reuse sentinel
passages, quote what their conventions claim, and the glossary is well-formed.
"""

from __future__ import annotations

import yaml

from hevajra_matrix.collate import collator
from hevajra_matrix.collate.merge import merge_chapter
from hevajra_matrix.collate.verify import CheckLexicon, verify
from hevajra_matrix.core.textnorm import for_quote
from hevajra_matrix.core.translit import transliteration_charset

from test_collate_collator import _example_window
from test_collate_support import DATA, lexicon, raw

DRAFT = DATA / "codebook" / "drafts" / "collate.v2"
CONVENTIONS = {"none", "standing_rendering", "transcribed_name"}


def examples() -> list[dict]:
    doc = yaml.safe_load((DRAFT / "examples.yaml").read_text(encoding="utf-8"))
    assert doc["schema_version"] == 1 and (doc["reference_lang"], doc["witness_lang"]) == ("bo", "zh")
    return doc["examples"]


def test_draft_examples_pass_the_v1_code_checks() -> None:
    items = examples()
    assert len(items) == collator.N_EXAMPLES
    for n, ex in enumerate(items, start=1):
        answer = {**ex["answer"], "units": [{k: v for k, v in u.items() if k != "convention"}
                                            for u in ex["answer"]["units"]]}
        w = _example_window(n, {k: ex[k] for k in ("reference", "witness")})
        lex = CheckLexicon(lexicon().negators, lexicon().variants, transliteration_charset(w.text))
        result = verify(raw(w, answer), w, lex, "example")
        assert dict(result.unresolved) == {}, (n, result.diagnostics)
        assert [d.kind for d in result.diagnostics] == [], n
        assert not [d for d in merge_chapter([w], [result]).diagnostics if d.kind == "unaccounted"], n


def test_draft_conventions_are_quoted_and_listed() -> None:
    used = set()
    for ex in examples():
        listed = ex.get("standing_renderings") or []
        for u in ex["answer"]["units"]:
            assert u["convention"] in CONVENTIONS
            used.add(u["convention"])
            if u["convention"] != "none":
                assert u["ref_quote"] and u["wit_quote"], u
            if u["convention"] == "standing_rendering":
                assert any(s["reference"] in u["ref_quote"] and s["witness"] in u["wit_quote"] for s in listed), u
    assert used == CONVENTIONS, "every convention is illustrated"


def test_draft_examples_never_reuse_sentinel_passages() -> None:
    sentinels = yaml.safe_load((DATA / "sentinels" / "sentinels.yaml").read_text(encoding="utf-8"))
    quotes = [(q, "bo" if any(0x0F00 <= ord(c) <= 0x0FFF for c in q) else "zh")
              for s in sentinels["sentinels"] for q in (s.get("quotes") or {}).values() if q]
    lines = [(line["text"], lang) for ex in examples()
             for side, lang in (("witness", "zh"), ("reference", "bo")) for line in ex[side]]
    for q, qlang in quotes:
        for text, lang in lines:
            if lang == qlang:
                a, b = for_quote(q, lang), for_quote(text, lang)
                assert a not in b and (len(b) < 4 or b not in a), (q, text)


def test_draft_glossary_is_well_formed() -> None:
    doc = yaml.safe_load((DRAFT / "standing_renderings.yaml").read_text(encoding="utf-8"))
    assert doc["schema_version"] == 1 and doc["entries"]
    for e in doc["entries"]:
        assert e["reference_forms"] and e["witness_renderings"] and e["status"] in {"proposed", "verified", "retired"}
        assert (e["status"] == "verified") == bool(e.get("verified_by"))
    ids = [e["id"] for e in doc["entries"] + doc["considered_not_listed"]]
    assert len(ids) == len(set(ids))


def test_the_draft_is_not_read_by_the_pipeline() -> None:
    assert collator.example_pairs(DATA) == [collator.DEFAULT_PAIR], "drafts are not an examples file"
    assert collator.TEMPLATE_FILE == "collate.v1.md"
