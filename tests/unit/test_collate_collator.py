"""collate.collator: request layout and content, parsing of every response outcome,
settings, the few-shot examples and the template/schema consistency."""

from __future__ import annotations

import re
import threading
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from hevajra_matrix.collate import collator
from hevajra_matrix.collate.collator import (
    SCHEMA,
    RawCollation,
    TaskSettings,
    Unresolved,
    build_request,
    build_requests,
    collate,
    load_examples,
    parse,
    replicate_tags,
    source_name,
)
from hevajra_matrix.collate.merge import merge_chapter, verify_all
from hevajra_matrix.collate.verify import CheckLexicon, verify
from hevajra_matrix.collate.windows import Window
from hevajra_matrix.collate.windows import WindowParams, plan
from hevajra_matrix.config import ConfigError
from hevajra_matrix.core.textnorm import for_quote
from hevajra_matrix.core.translit import transliteration_charset
from hevajra_matrix.core.types import Relation, Segment, WitnessOnlyKind
from hevajra_matrix.ingest import cbeta, derge
from hevajra_matrix.llm.client import LLMTransportError, sha256_text
from hevajra_matrix.llm.fake import (
    FakeClient,
    invalid_response,
    ok_response,
    refusal_response,
    substituted_response,
    truncated_response,
)
from hevajra_matrix.llm.schema import strict
from hevajra_matrix.registry import load_concordance

from test_collate_support import (
    DATA,
    SETTINGS,
    examples,
    gold_answer,
    lexicon,
    raw,
    reference,
    template,
    window,
    windows,
)


def _request(w=None, replicate: str = "r1"):
    return build_request(w or window(), SETTINGS, replicate, examples(), template())


# --------------------------------------------------------------------------- layout
def test_request_layout_system_context_body() -> None:
    w = window()
    req = _request(w)
    assert req.task == "collate"
    assert req.system.startswith(template().rstrip())
    assert examples().text in req.system
    assert req.prompt_sha == sha256_text(collator.system_prompt(template(), examples()))
    context_lines = req.context.splitlines()
    assert context_lines[0] == "WITNESS TEXT (Chinese)"
    assert len(context_lines) == 1 + len(w.text)
    for line, (handle, sid) in zip(context_lines[1:], w.wit_handles.items()):
        h, kind, text = line.split("\t")
        assert (h, text) == (handle, w.segment[sid].text)
        assert kind in {"prose", "verse", "mantra", "head", "note"}
    body = req.body.splitlines()
    assert body[0] == "REFERENCE CHUNK (Tibetan)"
    assert [line.split("\t")[0] for line in body[1:1 + len(w.units)]] == list(w.ref_handles)
    assert body[-1] == "CORE WINDOW: z0001-z0009"


def test_prefix_is_shared_by_every_window_and_replicate() -> None:
    reqs = [_request(w, tag) for w in windows() for tag in ("r1", "r2")]
    assert len({(r.system, r.context) for r in reqs}) == 1, "system and context are the cached prefix"
    assert len({r.key() for r in reqs}) == len(reqs), "replicate and body change the cache key"


def test_request_settings_and_schema() -> None:
    req = _request()
    assert req.max_tokens == 128_000
    assert req.effort == "high" and req.model == "claude-opus-5-5" and req.replicate == "r1"
    assert req.schema == SCHEMA == strict(SCHEMA)
    custom = build_request(window(), TaskSettings(effort="low", max_tokens=2000, fallback=True), "r3",
                           examples(), template())
    assert (custom.effort, custom.max_tokens, custom.allow_fallback, custom.replicate) == ("low", 2000, True, "r3")


def test_tabs_and_newlines_inside_text_cannot_break_the_line_format() -> None:
    w = window()
    odd = replace(w.units[0], text="a\tb\nc")
    req = build_request(replace(w, units=(odd, *w.units[1:])), SETTINGS, "r1", examples(), template())
    assert "r001\tprose\ta b c" in req.body.splitlines()


def test_language_pair_must_match_the_examples() -> None:
    w = window()
    swapped = replace(examples(), reference_lang="sa")
    with pytest.raises(ValueError, match="examples"):
        build_request(w, SETTINGS, "r1", swapped, template())


# --------------------------------------------------------------------------- what never reaches the model
def _ingested_windows():
    """Windows built from the ingest fixtures, which carry Taisho footnotes."""
    zh = cbeta.parse(DATA / "fixtures" / "mini_cbeta.xml", "zh_T0892_song", DATA)
    bo = derge.parse(DATA / "fixtures" / "mini_derge.txt", ["D417", "D418"], "bo_derge_D417_418", DATA)
    conc = load_concordance(DATA / "registry" / "concordance.yaml")
    units = conc.assign_reference_chapters(bo.segments, "bo_derge_D417_418")
    return zh, plan(units, zh.segments, conc, WindowParams(150, 10, 1))


def test_request_has_no_footnote_sanskrit_topic_or_coordinate_text() -> None:
    zh, ws = _ingested_windows()
    assert zh.footnotes, "the fixture must carry footnotes for this test to mean anything"
    topics = yaml.safe_load((DATA / "codebook" / "topics.yaml").read_text(encoding="utf-8"))
    labels = [t for t in re.findall(r"\b[a-z]+_[a-z_]+\b", str(topics)) if t not in {"polarity_flip"}]
    for w in ws:
        req = build_request(w, SETTINGS, "r1", examples(), template())
        prompt = "\n".join([req.system, req.context or "", req.body])
        dynamic = "\n".join([req.context or "", req.body])
        for fn in zh.footnotes:
            assert fn.text not in prompt
            if fn.sa_text:
                assert fn.sa_text not in prompt
        for label in labels:
            assert label not in dynamic
        # no coordinates: Taisho lines, Derge folio lines, segment ids, chapter keys
        assert not re.search(r"\d{4}[abc]\d{2}", dynamic)
        assert not re.search(r"\b\d+[ab]\.\d\b", dynamic)
        assert not re.search(r"T0892|D41[78]|pin\d|\b(I|II)\.\d", dynamic)
        # only reference-language units and witness-language segments are printed
        assert all(u.lang == "bo" for u in w.units) and all(s.lang == "zh" for s in w.text)


def test_translator_notes_are_in_the_context_and_front_matter_is_not() -> None:
    zh, ws = _ingested_windows()
    context = build_request(ws[0], SETTINGS, "r1", examples(), template()).context
    notes = [s for s in zh.segments if s.kind == "note"]
    meta = [s for s in zh.segments if s.kind == "meta" and s.text.strip()]
    assert all(f"\tnote\t{n.text}" in context for n in notes)
    assert not any(f"\t{m.text}" in context for m in meta)


# --------------------------------------------------------------------------- parsing
def test_parse_ok_answer() -> None:
    w = window()
    req = _request(w)
    parsed = parse(ok_response(req, gold_answer(w)), w)
    assert isinstance(parsed, RawCollation)
    assert parsed.window == w.key
    assert [u.ref for u in parsed.units] == list(w.ref_handles)
    assert parsed.units[5].relation is Relation.REVERSAL
    assert {x.kind for x in parsed.witness_only} >= {WitnessOnlyKind.ADDITION, WitnessOnlyKind.TRANSLATOR_NOTE}


@pytest.mark.parametrize("category, reason", [("cyber", "refused:cyber"), (None, "refused:unspecified")])
def test_parse_refusal(category: str | None, reason: str) -> None:
    w = window()
    parsed = parse(refusal_response(_request(w), category=category), w)
    assert parsed == Unresolved(w.key, reason)


def test_parse_truncated_and_invalid() -> None:
    w = window()
    assert parse(truncated_response(_request(w)), w) == Unresolved(w.key, "truncated")
    assert parse(invalid_response(_request(w)), w) == Unresolved(w.key, "invalid")


def test_parse_ok_status_with_schema_violating_data_is_invalid() -> None:
    w = window()
    good = ok_response(_request(w), gold_answer(w))
    bad = replace(good, data={"units": [{"ref": "r001"}], "witness_only": []})
    assert parse(bad, w) == Unresolved(w.key, "invalid")
    assert parse(replace(good, data=None), w) == Unresolved(w.key, "invalid")


def test_parse_substituted_model_keeps_the_proposal_only_as_a_hint() -> None:
    w = window()
    parsed = parse(substituted_response(_request(w), gold_answer(w)), w)
    assert isinstance(parsed, Unresolved) and parsed.reason == "substituted_model"
    assert isinstance(parsed.hint, RawCollation) and len(parsed.hint.units) == len(w.units)
    # a fallback flagged only through usage iterations counts as well
    same_model = replace(ok_response(_request(w), gold_answer(w)), fallback_used=True)
    assert parse(same_model, w).reason == "substituted_model"


def test_substituted_answer_never_enters_the_alignment() -> None:
    w = window()
    parsed = parse(substituted_response(_request(w), gold_answer(w)), w)
    result, _ = verify_all([w], [parsed], lexicon(), "src")
    assert result.alignment.links == ()
    assert set(result.unresolved.values()) == {"substituted_model"}
    assert result.hints and all("substituted_model_hint" in h.flags for h in result.hints)


# --------------------------------------------------------------------------- running
def test_collate_returns_results_in_window_order_per_replicate() -> None:
    ws = windows(max_ref_units=3, overlap=1)
    order: list[str] = []
    lock = threading.Lock()

    def script(req):
        with lock:
            order.append(req.body)
        w = next(w for w in ws if _request(w, req.replicate).key() == req.key())
        return gold_answer(w)

    client = FakeClient(script)
    out = collate(ws, client, SETTINGS, replicate_tags(2), examples(), template(), workers=4)
    assert list(out) == ["r1", "r2"]
    for tag, results in out.items():
        assert [r.window for r in results] == [w.key for w in ws]
        assert all(isinstance(r, RawCollation) for r in results)
    assert len(client.requests) == 2 * len(ws)


def test_collate_mixed_outcomes_keep_their_slot() -> None:
    ws = windows()

    def script(req):
        if "r001\tprose\t" + ws[1].units[0].text in req.body:
            return refusal_response(req, category="bio")
        return ok_response(req, gold_answer(ws[0]))

    out = collate(ws, FakeClient(script), SETTINGS, ["r1"], examples(), template())
    assert isinstance(out["r1"][0], RawCollation)
    assert out["r1"][1] == Unresolved(ws[1].key, "refused:bio")


def test_collate_propagates_transport_errors() -> None:
    def script(req):
        raise LLMTransportError("down")

    with pytest.raises(LLMTransportError):
        collate(windows(), FakeClient(script), SETTINGS, ["r1"], examples(), template(), workers=2)
    with pytest.raises(ValueError):
        collate(windows(), FakeClient(script), SETTINGS, ["r1"], examples(), template(), workers=0)


def test_build_requests_is_replicate_major() -> None:
    jobs = build_requests(windows(), SETTINGS, ["r1", "r2"], examples(), template())
    assert [(tag, w.key) for tag, w, _ in jobs] == [("r1", "I.1.w1"), ("r1", "I.2.w1"), ("r2", "I.1.w1"),
                                                     ("r2", "I.2.w1")]


def test_replicate_tags_and_source_name() -> None:
    assert replicate_tags(3) == ("r1", "r2", "r3")
    assert source_name("claude-opus-5-5", "r2") == "claude-opus-5-5:collate.v1:r2"


# --------------------------------------------------------------------------- settings
def test_task_settings_from_config() -> None:
    llm = {"model": "claude-opus-5-5", "tasks": {"collate": {"effort": "medium", "max_tokens": 128000,
                                                             "replicates": 3, "fallback": True}}}
    s = TaskSettings.from_config(llm)
    assert (s.effort, s.max_tokens, s.replicates, s.fallback, s.model) == ("medium", 128000, 3, True,
                                                                            "claude-opus-5-5")
    assert TaskSettings.from_config({}).max_tokens == 128_000


@pytest.mark.parametrize("task", [
    {"effort": "extreme"}, {"max_tokens": 0}, {"replicates": True}, {"fallback": "yes"}, {"model": "x"},
    {"temperature": 0.0},
])
def test_task_settings_reject_bad_values(task: dict) -> None:
    with pytest.raises(ConfigError):
        TaskSettings.from_config({"tasks": {"collate": task}})


def test_repository_config_is_accepted() -> None:
    llm = yaml.safe_load((DATA.parent / "config" / "llm.yaml").read_text(encoding="utf-8"))
    TaskSettings.from_config(llm)


# --------------------------------------------------------------------------- template, schema, examples
def test_every_relation_and_witness_only_kind_is_in_template_and_schema() -> None:
    unit_props = SCHEMA["properties"]["units"]["items"]["properties"]
    wo_props = SCHEMA["properties"]["witness_only"]["items"]["properties"]
    assert set(unit_props["relation"]["enum"]) == {r.value for r in Relation}
    assert set(wo_props["kind"]["enum"]) == {k.value for k in WitnessOnlyKind}
    for value in [r.value for r in Relation] + [k.value for k in WitnessOnlyKind]:
        assert f"`{value}`" in template(), f"{value} is not defined in the template"


def test_schema_is_strict_as_specified() -> None:
    assert SCHEMA["required"] == ["units", "witness_only"]
    assert SCHEMA["properties"]["units"]["items"]["required"] == [
        "ref", "wit", "relation", "polarity_flip", "confidence", "ref_quote", "wit_quote"]
    assert SCHEMA["properties"]["witness_only"]["items"]["required"] == ["wit", "kind", "wit_quote"]
    assert all(node["additionalProperties"] is False for node in (
        SCHEMA, SCHEMA["properties"]["units"]["items"], SCHEMA["properties"]["witness_only"]["items"]))


def test_template_never_asks_for_reasoning_or_motives() -> None:
    text = template().lower()
    for phrase in ("reasoning", "explain", "step by step", "think", "justify", "rationale"):
        assert phrase not in text, phrase
    assert "motive" in text, "the template states the no-motives rule"


def test_template_is_english_only() -> None:
    assert template().isascii()


def _example_window(n: int, ex: dict) -> Window:
    """A window holding exactly what example ``n`` shows (its core window is the whole witness)."""
    def segs(side: str, witness: str, lang: str, prefix: str, **fields) -> tuple[Segment, ...]:
        return tuple(Segment(id=f"{prefix}:{n}a{i}.1", witness=witness, lang=lang, text=line["text"],
                             start="", end="", kind=line["kind"], **fields)
                     for i, line in enumerate(ex[side], start=1))
    return Window(key=f"example{n}", chapter="I.1", units=segs("reference", "ref", "bo", "EXR", chapter="I.1"),
                  text=segs("witness", "wit", "zh", "EXW", local_chapter="ex"), core_locals=frozenset({"ex"}))


def test_examples_are_three_and_pass_every_code_check() -> None:
    doc = yaml.safe_load((DATA / "codebook" / "collate_examples.yaml").read_text(encoding="utf-8"))
    assert len(doc["examples"]) == 3
    relations = set()
    for n, ex in enumerate(doc["examples"], start=1):
        w = _example_window(n, ex)
        lex = CheckLexicon(lexicon().negators, lexicon().variants, transliteration_charset(w.text))
        result = verify(raw(w, ex["answer"]), w, lex, "example")
        assert dict(result.unresolved) == {}, (n, result.diagnostics)
        assert all(link.flags <= {"corroborated"} for link in result.alignment.links), n
        assert [d.kind for d in result.diagnostics] == [], n
        merged = merge_chapter([w], [result])
        assert not [d for d in merged.diagnostics if d.kind == "unaccounted"], n
        relations |= {link.relation for link in result.alignment.links}
    assert {Relation.CATEGORY_NAME_OMITTED, Relation.REVERSAL, WitnessOnlyKind.ADDITION} <= relations


def test_examples_never_reuse_sentinel_passages() -> None:
    doc = yaml.safe_load((DATA / "codebook" / "collate_examples.yaml").read_text(encoding="utf-8"))
    sentinels = yaml.safe_load((DATA / "sentinels" / "sentinels.yaml").read_text(encoding="utf-8"))
    quotes = [(q, "bo" if any(0x0F00 <= ord(c) <= 0x0FFF for c in q) else "zh")
              for s in sentinels["sentinels"] for q in (s.get("quotes") or {}).values() if q]
    lines = [(line["text"], lang) for ex in doc["examples"]
             for side, lang in (("witness", "zh"), ("reference", "bo")) for line in ex[side]]
    for q, qlang in quotes:
        for text, lang in lines:
            if lang != qlang:
                continue
            a, b = for_quote(q, lang), for_quote(text, lang)
            assert a not in b, f"example line reuses sentinel quote {q!r}"
            assert len(b) < 4 or b not in a, f"example line {text!r} occurs in a sentinel quote"


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d["examples"].pop(), "exactly 3"),
    (lambda d: d.update(schema_version=2), "exactly 3"),
    (lambda d: d["examples"][0]["answer"]["units"].pop(), "once each"),
    (lambda d: d["examples"][0]["answer"]["units"][0].update(wit=["z0099"]), "unknown"),
    (lambda d: d["examples"][0]["answer"]["units"][0].update(relation="omitted"), "schema"),
    (lambda d: d["examples"][0].update(extra="x"), "keys"),
    (lambda d: d.update(extra=1), "keys"),
])
def test_load_examples_rejects_malformed_files(tmp_path: Path, mutate, message: str) -> None:
    doc = yaml.safe_load((DATA / "codebook" / "collate_examples.yaml").read_text(encoding="utf-8"))
    mutate(doc)
    (tmp_path / "codebook").mkdir()
    (tmp_path / "codebook" / "collate_examples.yaml").write_text(yaml.safe_dump(doc, allow_unicode=True),
                                                                 encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_examples(tmp_path)


def test_reference_fixture_is_tibetan() -> None:
    assert {s.lang for s in reference()} == {"bo"}
