"""collate.v2 and researcher decision Q3: a word for a class or kind of being (dakini,
yogini, ...) rendered by sound is not a proper name, so it is ``transliterated`` with
convention none, in the template and in the examples; and the examples never show a form
of the real glossary file, listed or only considered.

Multilingual strings come from the codebook files (data/codebook/collate.v2/); no
non-Latin literal appears here.
"""

from __future__ import annotations

from hevajra_matrix.collate.collator import PROMPT_V2, load_template

from test_collate_v2 import glossary_doc, v2_doc


def _bullet(text: str, value: str) -> str:
    """The template's bullet for ``value``, with its line breaks folded."""
    start = text.index(f"\n- `{value}`:")
    end = text.find("\n- ", start + 1)
    return " ".join(text[start:end if end != -1 else None].split())


def test_v2_template_keeps_class_nouns_out_of_transcribed_name() -> None:
    """Decision Q3: a class noun rendered by sound is transliterated, convention none."""
    text = load_template(PROMPT_V2)
    name = _bullet(text, "transcribed_name")
    assert "A word for a class or kind of being" in name and "dakini or yogini" in name
    assert ("is not a proper name, even when it refers to deities: rendered by sound it is the relation "
            "`transliterated` with convention `none`.") in name
    content = _bullet(text, "transliterated")
    assert "words for a class or kind of being (such as dakini or yogini)" in content
    assert "reference words that are not proper names" in content


def test_v2_example_3_shows_a_class_noun_by_sound_as_transliterated() -> None:
    """Besides the mantra, two prose units are rendered by sound without being names (a
    content word and a class of beings): both are transliterated with convention none."""
    ex = v2_doc()["examples"][2]
    kinds = {f"r{n:03d}": line["kind"] for n, line in enumerate(ex["reference"], start=1)}
    by_sound = [u for u in ex["answer"]["units"] if u["relation"] == "transliterated" and kinds[u["ref"]] == "prose"]
    assert len(by_sound) == 2 and {u["convention"] for u in by_sound} == {"none"}
    assert "class of beings" in ex["comment"] and "not a proper name" in ex["comment"]
    names = [u for u in ex["answer"]["units"] if u["convention"] == "transcribed_name"]
    assert names and not {u["ref_quote"] for u in names} & {u["ref_quote"] for u in by_sound}


def test_v2_examples_never_show_a_form_of_the_glossary_file() -> None:
    """No reference form or witness rendering of standing_renderings.yaml, listed or only
    considered (dakini, yogini, ...), appears anywhere in the examples."""
    doc = glossary_doc()
    forms = {f for e in doc["entries"] for f in (*e["reference_forms"], *e["witness_renderings"])}
    forms |= {f for e in doc.get("considered_not_listed") or []
              for f in (*e.get("reference_forms", []), *e.get("witness_renderings_seen", []))}
    lines = [line["text"] for ex in v2_doc()["examples"] for side in ("witness", "reference") for line in ex[side]]
    assert forms and not [(f, t) for f in forms for t in lines if f in t]
