"""collate.v2: the prompt-version registry, the v2 template, schema, examples and glossary,
requests and parsing under v2, and the guarantee that collate.v1 stays byte-identical.

Multilingual strings come from the codebook files (data/codebook/collate.v2/); no
non-Latin literal appears here.
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from hevajra_matrix.collate import collator
from hevajra_matrix.collate.collator import (
    PROMPT_V1,
    PROMPT_V2,
    SCHEMA,
    SCHEMA_V1,
    SCHEMA_V2,
    VERSIONS,
    RawCollation,
    TaskSettings,
    Unresolved,
    build_request,
    collate,
    example_pairs,
    examples_file,
    glossary_file,
    load_examples,
    load_glossary,
    load_template,
    parse,
    render_glossary,
    schema_for,
    source_name,
    system_prompt,
)
from hevajra_matrix.collate.merge import merge_chapter
from hevajra_matrix.collate.prompt import example_glossary
from hevajra_matrix.collate.verify import CheckLexicon, verify
from hevajra_matrix.config import ConfigError
from hevajra_matrix.core.lexicon import LexiconError, StandingEntry, StandingRenderings
from hevajra_matrix.core.textnorm import for_quote
from hevajra_matrix.core.translit import transliteration_charset
from hevajra_matrix.core.types import CONVENTION_FLAGS, Convention, Relation, WitnessOnlyKind
from hevajra_matrix.llm.client import canonical_json, sha256_text
from hevajra_matrix.llm.fake import FakeClient, ok_response

from test_collate_collator import _example_window
from test_collate_support import DATA, echo_script, gold_answer, lexicon, raw, window, windows

V2_DIR = DATA / "codebook" / "collate.v2"
V1_PROMPT_SHA = "3ab88636d241c758519ec8ac3064c148f82987fcc351500757ccbc96f1533fd5"
V1_SCHEMA_SHA = "ed99b249404bedbb9eeee76e77300223198fee329fee2308d48819ef1e5d3ff9"
V2_SETTINGS = TaskSettings(prompt=PROMPT_V2)


def v2_doc() -> dict:
    return yaml.safe_load((V2_DIR / "examples.yaml").read_text(encoding="utf-8"))


def glossary_doc() -> dict:
    return yaml.safe_load((V2_DIR / "standing_renderings.yaml").read_text(encoding="utf-8"))


def v2_examples(data: Path = DATA) -> collator.Examples:
    return load_examples(data, prompt=PROMPT_V2)


def v2_request(w=None, examples: collator.Examples | None = None):
    return build_request(w or window(), V2_SETTINGS, "r1", examples or v2_examples(), load_template(PROMPT_V2))


def v2_root(tmp_path: Path) -> Path:
    """A throw-away data dir holding both versions' default-pair codebooks."""
    (tmp_path / "codebook").mkdir()
    shutil.copy(DATA / "codebook" / "collate_examples.yaml", tmp_path / "codebook")
    shutil.copytree(V2_DIR, tmp_path / "codebook" / "collate.v2")
    return tmp_path


def write_yaml(path: Path, doc: dict) -> None:
    path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")


# --------------------------------------------------------------------------- v1 unchanged
def test_v1_prompt_and_schema_are_byte_identical_to_the_pilot() -> None:
    """The pilot cache and the v1 digest were written with exactly these."""
    assert sha256_text(system_prompt(load_template(), load_examples(DATA))) == V1_PROMPT_SHA
    assert sha256_text(canonical_json(SCHEMA_V1)) == V1_SCHEMA_SHA
    assert SCHEMA is SCHEMA_V1 and schema_for() is SCHEMA_V1
    assert collator.TASK == "collate" and collator.PROMPT_ID == PROMPT_V1
    assert collator.TEMPLATE_FILE == "collate.v1.md"
    assert collator.EXAMPLES_FILE == Path("codebook") / "collate_examples.yaml"
    assert load_examples(DATA).glossary is None and load_examples(DATA).prompt == PROMPT_V1


def test_v1_source_name_and_settings_default() -> None:
    assert TaskSettings().prompt == PROMPT_V1
    assert source_name("claude-opus-5-5", "r2") == "claude-opus-5-5:collate.v1:r2"
    assert source_name("claude-opus-5-5", "r2", PROMPT_V2) == "claude-opus-5-5:collate.v2:r2"


def test_v1_answer_parses_without_convention() -> None:
    w = window()
    parsed = parse(ok_response(build_request(w, TaskSettings(), "r1", load_examples(DATA), load_template()),
                               gold_answer(w)), w)
    assert isinstance(parsed, RawCollation)
    assert {u.convention for u in parsed.units} == {Convention.NONE}


# --------------------------------------------------------------------------- registry and settings
def test_versions_registry() -> None:
    assert set(VERSIONS) == {PROMPT_V1, PROMPT_V2}
    assert not VERSIONS[PROMPT_V1].conventions and VERSIONS[PROMPT_V2].conventions
    assert schema_for(PROMPT_V2) is SCHEMA_V2
    with pytest.raises(ValueError, match="unknown collate prompt"):
        schema_for("collate.v3")


def test_task_settings_prompt() -> None:
    llm = {"tasks": {"collate": {"prompt": "collate.v2"}}}
    assert TaskSettings.from_config(llm).prompt == PROMPT_V2
    assert TaskSettings.from_config({}).prompt == PROMPT_V1
    with pytest.raises(ConfigError, match="tasks.collate.prompt must be one of"):
        TaskSettings.from_config({"tasks": {"collate": {"prompt": "collate.v3"}}})


def test_repository_config_keeps_v1_active() -> None:
    llm = yaml.safe_load((DATA.parent / "config" / "llm.yaml").read_text(encoding="utf-8"))
    assert TaskSettings.from_config(llm).prompt == PROMPT_V1


def test_file_layout_per_version() -> None:
    assert examples_file("bo", "zh", PROMPT_V2) == Path("codebook") / "collate.v2" / "examples.yaml"
    assert examples_file("sa", "zh", PROMPT_V2) == Path("codebook") / "collate.v2" / "examples.sa-zh.yaml"
    assert examples_file("sa", "zh") == Path("codebook") / "collate_examples.sa-zh.yaml"
    assert glossary_file("bo", "zh") == Path("codebook") / "collate.v2" / "standing_renderings.yaml"
    assert glossary_file("sa", "bo") == Path("codebook") / "collate.v2" / "standing_renderings.sa-bo.yaml"
    assert glossary_file("bo", "zh", PROMPT_V1) is None
    assert example_pairs(DATA, PROMPT_V2) == [collator.DEFAULT_PAIR]
    assert example_pairs(DATA) == [collator.DEFAULT_PAIR], "the v1 glob never sees collate.v2/"
    assert not (DATA / "codebook" / "drafts").exists(), "the draft was adopted, not copied"


# --------------------------------------------------------------------------- template and schema
def test_v2_schema_adds_the_convention_before_the_relation() -> None:
    items = SCHEMA_V2["properties"]["units"]["items"]
    assert items["required"] == ["ref", "wit", "convention", "relation", "polarity_flip", "confidence",
                                 "ref_quote", "wit_quote"]
    assert items["properties"]["convention"]["enum"] == [c.value for c in Convention]
    assert items["additionalProperties"] is False
    v1_items = SCHEMA_V1["properties"]["units"]["items"]
    assert {k: v for k, v in items["properties"].items() if k != "convention"} == v1_items["properties"]
    assert SCHEMA_V2["properties"]["witness_only"] == SCHEMA_V1["properties"]["witness_only"]


def test_v2_template_defines_every_value_and_keeps_the_rules() -> None:
    text = load_template(PROMPT_V2)
    assert text.startswith("# Collation task (collate.v2)\n") and "DRAFT" not in text
    assert text.isascii()
    for value in [r.value for r in Relation] + [k.value for k in WitnessOnlyKind] + [c.value for c in Convention]:
        assert f"`{value}`" in text, value
    lowered = text.lower()
    for phrase in ("reasoning", "explain", "step by step", "think", "justify", "rationale"):
        assert phrase not in lowered, phrase
    assert "motive" in lowered and "STANDING RENDERINGS" in text
    assert load_template(PROMPT_V1) != text


# --------------------------------------------------------------------------- examples
def test_v2_examples_pass_every_code_check_with_their_own_glossary() -> None:
    doc = v2_doc()
    assert doc["schema_version"] == 1 and (doc["reference_lang"], doc["witness_lang"]) == ("bo", "zh")
    assert len(doc["examples"]) == collator.N_EXAMPLES
    for n, ex in enumerate(doc["examples"], start=1):
        w = _example_window(n, ex)
        lex = CheckLexicon(lexicon().negators, lexicon().variants, transliteration_charset(w.text),
                           example_glossary(ex, f"example {n}"))
        result = verify(raw(w, ex["answer"]), w, lex, "example")
        assert dict(result.unresolved) == {}, (n, result.diagnostics)
        assert [d.kind for d in result.diagnostics] == [], n
        for link, u in zip(result.alignment.by_ref().values(), ex["answer"]["units"]):
            assert link.flags <= {"corroborated"} | CONVENTION_FLAGS, (n, u["ref"], link.flags)
            claimed = {u["convention"]} - {"none"}
            assert link.flags & CONVENTION_FLAGS == claimed, (n, u["ref"])
        assert not [d for d in merge_chapter([w], [result]).diagnostics if d.kind == "unaccounted"], n


def test_v2_examples_illustrate_every_convention_and_the_list_rule() -> None:
    used, relations = set(), set()
    for ex in v2_doc()["examples"]:
        for u in ex["answer"]["units"]:
            used.add(u["convention"])
            relations.add(u["relation"])
            if u["convention"] != "none":
                assert u["ref_quote"] and u["wit_quote"], u
    assert used == {c.value for c in Convention}
    assert {"category_name_omitted", "generalised", "transliterated", "reversal"} <= relations


def test_v2_examples_never_reuse_sentinel_passages() -> None:
    sentinels = yaml.safe_load((DATA / "sentinels" / "sentinels.yaml").read_text(encoding="utf-8"))
    quotes = [(q, "bo" if any(0x0F00 <= ord(c) <= 0x0FFF for c in q) else "zh")
              for s in sentinels["sentinels"] for q in (s.get("quotes") or {}).values() if q]
    lines = [(line["text"], lang) for ex in v2_doc()["examples"]
             for side, lang in (("witness", "zh"), ("reference", "bo")) for line in ex[side]]
    for q, qlang in quotes:
        for text, lang in lines:
            if lang == qlang:
                a, b = for_quote(q, lang), for_quote(text, lang)
                assert a not in b and (len(b) < 4 or b not in a), (q, text)


def test_v2_examples_render_their_own_glossary_inside_the_example() -> None:
    text = v2_examples().text
    ex2 = v2_doc()["examples"][1]
    own = ex2["standing_renderings"][0]
    block = text[text.index("### Example 2"):text.index("### Example 3")]
    line = f"{own['reference']} = {own['witness']}"
    assert block.index("STANDING RENDERINGS (Tibetan -> Chinese)") < block.index(line) < block.index("WITNESS TEXT")
    assert "STANDING RENDERINGS" not in text[:text.index("### Example 2")], "example 1 lists none"


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d["examples"][1]["answer"]["units"][0].update(wit_quote=""), "not both quotes"),
    (lambda d: d["examples"][1]["answer"]["units"][0].update(ref_quote=""), "not both quotes"),
    (lambda d: d["examples"][1].pop("standing_renderings"), "does not list"),
    (lambda d: d["examples"][1].update(standing_renderings=[]), "non-empty list"),
    (lambda d: d["examples"][1]["standing_renderings"][0].update(extra="x"), "non-empty reference and witness"),
    (lambda d: d["examples"][0]["answer"]["units"][0].pop("convention"), "schema"),
    (lambda d: d["examples"][0]["answer"]["units"][0].update(convention="habit"), "schema"),
    (lambda d: d["examples"][0].update(extra="x"), "keys"),
])
def test_load_v2_examples_rejects_malformed_files(tmp_path: Path, mutate, message: str) -> None:
    root = v2_root(tmp_path)
    doc = v2_doc()
    mutate(doc)
    write_yaml(root / "codebook" / "collate.v2" / "examples.yaml", doc)
    with pytest.raises(ValueError, match=message):
        load_examples(root, prompt=PROMPT_V2)


def test_a_v1_examples_file_is_not_a_v2_one(tmp_path: Path) -> None:
    root = v2_root(tmp_path)
    shutil.copy(DATA / "codebook" / "collate_examples.yaml", root / "codebook" / "collate.v2" / "examples.yaml")
    with pytest.raises(ValueError, match="schema"):
        load_examples(root, prompt=PROMPT_V2)
    doc = yaml.safe_load((DATA / "codebook" / "collate_examples.yaml").read_text(encoding="utf-8"))
    doc["examples"][0]["standing_renderings"] = v2_doc()["examples"][1]["standing_renderings"]
    write_yaml(root / "codebook" / "collate_examples.yaml", doc)
    with pytest.raises(ValueError, match="keys"):
        load_examples(root)


def test_examples_may_not_show_a_form_of_the_real_glossary(tmp_path: Path) -> None:
    root = v2_root(tmp_path)
    real = glossary_doc()["entries"][0]          # still proposed: the rule holds whatever the status
    assert real["status"] == "proposed"
    doc = v2_doc()
    own = doc["examples"][1]["standing_renderings"]
    own.append({"reference": own[0]["reference"], "witness": real["witness_renderings"][-1]})
    write_yaml(root / "codebook" / "collate.v2" / "examples.yaml", doc)
    with pytest.raises(ValueError, match="real standing renderings"):
        load_examples(root, prompt=PROMPT_V2)


# --------------------------------------------------------------------------- glossary
def test_the_shipped_glossary_has_no_verified_entry_so_the_block_reads_none() -> None:
    glossary, texts, forms = load_glossary(DATA, "bo", "zh")
    assert glossary == StandingRenderings() and texts == ("bo_derge_D417_418", "zh_T0892_song")
    assert set(glossary_doc()["entries"][0]["reference_forms"]) <= forms["ref"]
    ex = v2_examples()
    assert ex.glossary == glossary and ex.glossary_texts == texts
    system = system_prompt(load_template(PROMPT_V2), ex)
    assert system.endswith("\n\n## Standing renderings of this witness\n\n"
                           "STANDING RENDERINGS (Tibetan -> Chinese)\n(none)\n")
    for form in glossary_doc()["entries"][0]["witness_renderings"]:
        assert form not in system


def test_only_verified_entries_reach_the_prompt(tmp_path: Path) -> None:
    root = v2_root(tmp_path)
    doc = glossary_doc()
    entry = doc["entries"][0]
    second = {**entry, "id": "second", "reference_forms": entry["reference_forms"][:1],
              "witness_renderings": entry["witness_renderings"][:1], "status": "proposed", "verified_by": None}
    doc["entries"] = [{**entry, "reference_forms": entry["reference_forms"][1:],
                       "witness_renderings": entry["witness_renderings"][1:], "status": "verified",
                       "verified_by": "A. Researcher"}, second]
    write_yaml(root / "codebook" / "collate.v2" / "standing_renderings.yaml", doc)
    ex = v2_examples(root)
    assert [e.id for e in ex.glossary.entries] == ["hevajra"]
    system = system_prompt(load_template(PROMPT_V2), ex)
    block = system[system.index("## Standing renderings of this witness"):]
    assert " / ".join(entry["reference_forms"][1:]) + " = " + entry["witness_renderings"][1] in block
    assert entry["reference_forms"][0] not in block, "the proposed entry is not shown"
    for hidden in ("A. Researcher", "hevajra", "Hevajra", "counts", str(entry["counts"][0]["n"])):
        assert hidden not in block
    assert sha256_text(system) != sha256_text(system_prompt(load_template(PROMPT_V2), v2_examples())), \
        "verifying an entry changes the instrument"


def test_render_glossary_lines() -> None:
    entries = StandingRenderings((StandingEntry("a", ("x", "y"), ("p",)), StandingEntry("b", ("z",), ("q\tr",))))
    assert render_glossary(entries, "bo", "zh").splitlines() == [
        "STANDING RENDERINGS (Tibetan -> Chinese)", "x / y = p", "z = q r"]
    assert render_glossary(StandingRenderings(), "sa", "zh").splitlines() == [
        "STANDING RENDERINGS (Sanskrit -> Chinese)", "(none)"]


@pytest.mark.parametrize("mutate, error, message", [
    (lambda d: d.update(schema_version=2), LexiconError, "schema_version"),
    (lambda d: d.pop("reference"), ValueError, "reference and witness"),
    (lambda d: d["entries"][0].update(status="verified"), LexiconError, "verified_by"),
    (lambda d: d["entries"][0].update(colour="red"), LexiconError, "unknown key"),
])
def test_a_malformed_glossary_is_refused(tmp_path: Path, mutate, error, message: str) -> None:
    root = v2_root(tmp_path)
    doc = glossary_doc()
    mutate(doc)
    write_yaml(root / "codebook" / "collate.v2" / "standing_renderings.yaml", doc)
    with pytest.raises(error, match=message):
        load_examples(root, prompt=PROMPT_V2)


def test_the_glossary_file_is_required_under_v2(tmp_path: Path) -> None:
    root = v2_root(tmp_path)
    (root / "codebook" / "collate.v2" / "standing_renderings.yaml").unlink()
    with pytest.raises(FileNotFoundError, match=r"standing_renderings\.yaml is missing"):
        load_examples(root, prompt=PROMPT_V2)
    load_examples(root)                          # v1 never reads it


# --------------------------------------------------------------------------- Sanskrit pairs
def _sanskrit_v2(root: Path, pair: str) -> None:
    doc = yaml.safe_load((DATA / "fixtures" / "sa_mini" / f"collate_examples.{pair}.yaml").read_text(encoding="utf-8"))
    for ex in doc["examples"]:
        ex["answer"]["units"] = [{**u, "convention": "none"} for u in ex["answer"]["units"]]
    write_yaml(root / "codebook" / "collate.v2" / f"examples.{pair}.yaml", doc)


def test_sanskrit_pairs_under_v2(tmp_path: Path) -> None:
    root = v2_root(tmp_path)
    shutil.copy(DATA / "fixtures" / "sa_mini" / "collate_examples.sa-zh.yaml", root / "codebook")
    _sanskrit_v2(root, "sa-zh")
    assert example_pairs(root, PROMPT_V2) == [("bo", "zh"), ("sa", "zh")]
    assert example_pairs(root) == [("bo", "zh"), ("sa", "zh")], "v1 reads its own sa-zh file only"
    with pytest.raises(FileNotFoundError, match=r"standing_renderings\.sa-zh\.yaml is missing"):
        load_examples(root, "sa", "zh", PROMPT_V2)
    write_yaml(root / "codebook" / "collate.v2" / "standing_renderings.sa-zh.yaml",
               {"schema_version": 1, "reference": "sa_test", "witness": "zh_T0892_song", "entries": []})
    ex = load_examples(root, "sa", "zh", PROMPT_V2)
    assert (ex.reference_lang, ex.prompt, ex.glossary, ex.glossary_texts) == (
        "sa", PROMPT_V2, StandingRenderings(), ("sa_test", "zh_T0892_song"))
    assert "STANDING RENDERINGS (Sanskrit -> Chinese)\n(none)" in system_prompt(load_template(PROMPT_V2), ex)
    with pytest.raises(FileNotFoundError, match=r"collate\.v2/examples\.sa-bo\.yaml is missing: .* format of "
                                                 r"examples\.yaml"):
        load_examples(root, "sa", "bo", PROMPT_V2)


# --------------------------------------------------------------------------- requests and parsing
def test_v2_request_uses_the_v2_schema_and_system() -> None:
    req = v2_request()
    assert req.task == "collate" and req.schema == SCHEMA_V2
    assert req.system == system_prompt(load_template(PROMPT_V2), v2_examples())
    assert req.prompt_sha == sha256_text(req.system) != V1_PROMPT_SHA
    v1 = build_request(window(), TaskSettings(), "r1", load_examples(DATA), load_template())
    assert (req.context, req.body) == (v1.context, v1.body), "only the static prefix and schema differ"
    assert req.key() != v1.key()


def test_settings_and_examples_must_be_the_same_version() -> None:
    with pytest.raises(ValueError, match="collate.v2 but the examples are those of collate.v1"):
        build_request(window(), V2_SETTINGS, "r1", load_examples(DATA), load_template(PROMPT_V2))
    with pytest.raises(ValueError, match="collate.v1 but the examples are those of collate.v2"):
        build_request(window(), TaskSettings(), "r1", v2_examples(), load_template())


def _v2_answer(w, convention: str = "none") -> dict:
    answer = gold_answer(w)
    return {**answer, "units": [{**u, "convention": convention} for u in answer["units"]]}


def test_parse_v2_answer_carries_the_convention() -> None:
    w = window()
    parsed = parse(ok_response(v2_request(w), _v2_answer(w, "transcribed_name")), w, SCHEMA_V2)
    assert isinstance(parsed, RawCollation)
    assert {u.convention for u in parsed.units} == {Convention.TRANSCRIBED_NAME}


def test_an_answer_is_valid_only_under_the_schema_that_was_sent() -> None:
    w = window()
    assert parse(ok_response(v2_request(w), _v2_answer(w)), w) == Unresolved(w.key, "invalid")
    v1_answer = ok_response(build_request(w, TaskSettings(), "r1", load_examples(DATA), load_template()),
                            gold_answer(w))
    assert parse(v1_answer, w, SCHEMA_V2) == Unresolved(w.key, "invalid")


def test_collate_parses_with_the_request_schema() -> None:
    ws = windows()
    client = FakeClient(echo_script)
    out = collate(ws, client, V2_SETTINGS, ["r1"], v2_examples(), load_template(PROMPT_V2))
    assert all(isinstance(r, RawCollation) for r in out["r1"])
    assert {u.convention for r in out["r1"] for u in r.units} == {Convention.NONE}
    assert all(req.schema == SCHEMA_V2 for req in client.requests)


def test_examples_dataclass_defaults_keep_v1_construction() -> None:
    ex = collator.Examples("bo", "zh", "text")
    assert (ex.prompt, ex.glossary, ex.glossary_texts) == (PROMPT_V1, None, None)
    assert replace(load_examples(DATA), reference_lang="sa").prompt == PROMPT_V1
