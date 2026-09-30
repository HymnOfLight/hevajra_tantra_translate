"""collate.components: settings, pair selection, requests (full text), outputs, examples.

Multilingual inputs come from data/fixtures/components_texts.yaml; the verification rules
are tested in test_components_verify.py.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import yaml

from hevajra_matrix.collate import components as C
from hevajra_matrix.collate.verify import FLAG_CORROBORATED, CheckLexicon
from hevajra_matrix.config import ConfigError, load_settings
from hevajra_matrix.core.types import (
    Alignment,
    Cell,
    Grade,
    Link,
    Relation,
    Segment,
    Status,
)
from hevajra_matrix.llm.client import sha256_text
from hevajra_matrix.llm.fake import FakeClient, ok_response, substituted_response

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIXTURE = yaml.safe_load((DATA / "fixtures" / "components_texts.yaml").read_text(encoding="utf-8"))
EXAMPLES = yaml.safe_load((DATA / "codebook" / "components_examples.yaml").read_text(encoding="utf-8"))


def _segments(records, witness: str, lang: str) -> list[Segment]:
    return [Segment(r["id"], witness, lang, r["text"], r["id"], r["id"], r["kind"]) for r in records]


REF = _segments(FIXTURE["reference_units"], FIXTURE["reference_witness"], "bo")
WIT = _segments(FIXTURE["witness_units"], FIXTURE["witness"], "zh")
STORE = {s.id: s for s in REF + WIT}
KINDS = {s.id: s.kind for s in REF}
R_LAMP, R_YOGIN, R_SKULL, R_HERUKA, R_LONG, R_MANTRA, R_ABSENT = (s.id for s in REF)
W_LAMP, W_YOGIN, W_SKULL, W_SALUTE, W_NAME, W_LONG, W_MANTRA = (s.id for s in WIT)


@pytest.fixture(scope="module")
def settings() -> C.ComponentTaskSettings:
    return C.ComponentTaskSettings.from_config(load_settings(ROOT).llm)


@pytest.fixture(scope="module")
def system() -> str:
    return C.load_system(DATA)


@pytest.fixture(scope="module")
def lexicon() -> CheckLexicon:
    return CheckLexicon.load(DATA, WIT)


def _pair(ref: str, *wit: str, relation: str = "substitution", flip: bool = False) -> C.Pair:
    return C.Pair(ref, tuple(wit), relation, flip, frozenset({C.SELECT_DEVIATION}))


# --------------------------------------------------------------------------- settings
def test_settings_come_from_llm_yaml(settings):
    assert (settings.effort, settings.max_tokens, settings.pairs_per_call) == ("high", 32000, 12)
    assert settings.fallback is True and settings.replicates == 1
    assert settings.model == load_settings(ROOT).llm["model"]


@pytest.mark.parametrize("section", [
    {"effort": "extreme"}, {"pairs_per_call": 0}, {"max_tokens": True}, {"fallback": "yes"},
    {"unknown": 1}, {"model": "other"},
])
def test_settings_reject_bad_values(section):
    with pytest.raises(ConfigError):
        C.ComponentTaskSettings.from_config({"model": "claude-opus-5-5", "tasks": {"components": section}})


# --------------------------------------------------------------------------- selection
def _link(ref: str, relation: Relation, wit: tuple[str, ...] = ("w",), flip: bool = False) -> Link:
    return Link(ref, () if relation is Relation.NO_COUNTERPART else wit, relation, polarity_flip=flip)


def _alignment(*links: Link) -> Alignment:
    return Alignment("claude:consensus", "bo", "zh", links)


def test_select_pairs_takes_deviations_and_skips_nondeviations():
    alignment = _alignment(
        _link("u1", Relation.SUBSTITUTION), _link("u2", Relation.PARAPHRASE), _link("u3", Relation.ABRIDGED),
        _link("u4", Relation.EQUIVALENT, flip=True), _link("u5", Relation.TRANSLITERATED),
        _link("u6", Relation.TRANSLITERATED), _link("u7", Relation.NO_COUNTERPART), _link("u8", Relation.EXPANDED),
    )
    kinds = {"u5": "prose", "u6": "mantra"}
    pairs = C.select_pairs(alignment, {}, random.Random(1), ref_kinds=kinds, sample_fraction=0.0)
    assert [p.ref_id for p in pairs] == ["u1", "u3", "u4", "u5"]      # input order kept
    assert all(p.selected_as == {C.SELECT_DEVIATION} for p in pairs)
    assert pairs[2].polarity_flip is True and pairs[0].relation == "substitution"


def test_select_pairs_takes_every_sensitive_unit():
    alignment = _alignment(_link("u1", Relation.EQUIVALENT), _link("u2", Relation.SUBSTITUTION),
                           _link("u3", Relation.EQUIVALENT))
    topics = {"u1": {"flesh_food"}, "u2": {"harm", "ritual"}, "u3": {"neutral"}}
    pairs = C.select_pairs(alignment, topics, random.Random(1), ref_kinds={}, sample_fraction=0.0)
    assert {p.ref_id: p.selected_as for p in pairs} == {
        "u1": {C.SELECT_SENSITIVE}, "u2": {C.SELECT_SENSITIVE, C.SELECT_DEVIATION}}


def test_select_pairs_skips_units_without_witness_text_even_if_sensitive():
    alignment = _alignment(_link("u1", Relation.NO_COUNTERPART))
    assert C.select_pairs(alignment, {"u1": {"harm"}}, random.Random(1), ref_kinds={}) == []


def test_equivalent_sample_is_ten_percent_seeded_and_disjoint_from_other_strata():
    links = [_link(f"u{i:02d}", Relation.EQUIVALENT) for i in range(40)]
    topics = {f"u{i:02d}": {"sexual"} for i in range(10)}             # 10 sensitive, 30 plain equivalent
    run = [C.select_pairs(_alignment(*links), topics, random.Random(seed), ref_kinds={}) for seed in (7, 7, 8)]
    sampled = [{p.ref_id for p in r if C.SELECT_EQUIVALENT_SAMPLE in p.selected_as} for r in run]
    assert len(sampled[0]) == 3 and sampled[0] == sampled[1]
    assert not sampled[0] & set(topics)
    assert all(p.selected_as == {C.SELECT_EQUIVALENT_SAMPLE} for p in run[0] if p.ref_id in sampled[0])
    assert len(sampled[2]) == 3
    assert [p.ref_id for p in run[0]] == sorted(p.ref_id for p in run[0])


def test_equivalent_sample_size_rounds_half_up_and_excludes_other_nondeviations():
    links = [_link(f"u{i}", Relation.EQUIVALENT) for i in range(5)] + [_link("p", Relation.PARAPHRASE)]
    pairs = C.select_pairs(_alignment(*links), {}, random.Random(0), ref_kinds={})
    assert len(pairs) == 1 and pairs[0].ref_id != "p"                   # 0.1 * 5 = 0.5 -> 1
    assert C.select_pairs(_alignment(*links), {}, random.Random(0), ref_kinds={}, sample_fraction=1.0)[-1].ref_id == "u4"
    with pytest.raises(ValueError):
        C.select_pairs(_alignment(*links), {}, random.Random(0), ref_kinds={}, sample_fraction=1.5)


def test_select_pairs_reads_matrix_cells_and_skips_uncounted_ones():
    cells = [
        Cell("u1", "zh", Status.PRESENT, Grade.A, relation="generalised", wit_ids=("w1",)),
        Cell("u2", "zh", Status.UNALIGNED, Grade.X, reason="refused:bio", wit_ids=("w2",)),
        Cell("u3", "zh", Status.PARTIAL, Grade.B, relation="abridged", wit_ids=("w3", "w4")),
    ]
    pairs = C.select_pairs(cells, {"u2": {"harm"}}, random.Random(0), ref_kinds={})
    assert [(p.ref_id, p.wit_ids) for p in pairs] == [("u1", ("w1",)), ("u3", ("w3", "w4"))]


def test_select_pairs_rejects_an_invalid_topic_set():
    with pytest.raises(ValueError):
        C.select_pairs(_alignment(_link("u1", Relation.EQUIVALENT)), {"u1": {"porn"}}, random.Random(0), ref_kinds={})


# --------------------------------------------------------------------------- batches and requests
def _pairs() -> list[C.Pair]:
    return [_pair(R_LAMP, W_LAMP, relation="reversal", flip=True), _pair(R_YOGIN, W_YOGIN),
            _pair(R_SKULL, W_SKULL), _pair(R_HERUKA, W_SALUTE, W_NAME), _pair(R_LONG, W_LONG)]


def test_request_body_holds_the_full_segment_texts(settings, system):
    """Defect #16: texts come whole from the segment store, never truncated."""
    (request,) = C.build_requests(_pairs(), STORE, settings, system)
    long_ref, long_wit = STORE[R_LONG].text, STORE[W_LONG].text
    assert len(long_ref) > 80 and len(long_wit) > 80
    assert long_ref in request.body and long_wit in request.body
    for p in _pairs():
        for sid in (p.ref_id, *p.wit_ids):
            assert STORE[sid].text in request.body


def test_request_body_layout_and_blindness(settings, system):
    (request,) = C.build_requests(_pairs(), STORE, settings, system)
    lines = [ln.split("\t") for ln in request.body.splitlines() if "\t" in ln]
    assert [(h, side) for h, side, _, _ in lines] == [
        ("p01", "ref"), ("p01", "wit"), ("p02", "ref"), ("p02", "wit"), ("p03", "ref"), ("p03", "wit"),
        ("p04", "ref"), ("p04", "wit"), ("p04", "wit"), ("p05", "ref"), ("p05", "wit")]
    assert "REFERENCE LANGUAGE: Tibetan" in request.body and "WITNESS LANGUAGE: Chinese" in request.body
    for leak in ("D418", "T0892", "0596", "reversal", "substitution", "sensitive", "deviation", "bo_derge"):
        assert leak not in request.body and leak not in request.system


def test_request_settings_and_prompt_identity(settings, system):
    requests = C.build_requests(_pairs(), STORE, settings, system, pairs_per_call=2)
    assert len(requests) == 3
    r = requests[0]
    assert (r.task, r.effort, r.max_tokens, r.allow_fallback, r.model) == (
        "components", "high", 32000, True, settings.model)
    assert r.prompt_sha == sha256_text(system) and r.system == system and r.context is None
    assert r.schema == C.SCHEMA
    assert len({x.key() for x in requests}) == 3
    assert C.build_requests(_pairs(), STORE, settings, system, replicate="r2")[0].key() != \
        C.build_requests(_pairs(), STORE, settings, system)[0].key()


def test_plan_batches_uses_consecutive_chunks():
    batches = C.plan_batches(_pairs(), STORE, 2)
    assert [[i.pair.ref_id for i in b.items] for b in batches] == [
        [R_LAMP, R_YOGIN], [R_SKULL, R_HERUKA], [R_LONG]]
    assert list(batches[1].handles()) == ["p01", "p02"]
    assert batches[1].items[1].wit_text == f"{STORE[W_SALUTE].text} {STORE[W_NAME].text}"


def test_plan_batches_fails_loudly_on_missing_or_empty_witness_text():
    with pytest.raises(ValueError, match="segment store"):
        C.plan_batches([_pair(R_LAMP, "T0892:9999z99.1")], STORE, 12)
    with pytest.raises(ValueError):
        C.plan_batches([_pair(R_LAMP)], STORE, 12)
    with pytest.raises(ValueError):
        C.plan_batches(_pairs(), STORE, 0)


def test_a_batch_takes_one_language_per_side():
    mixed = {**STORE, "S:1": Segment("S:1", "sa_x", "sa", "vajra", "1", "1", "prose")}
    with pytest.raises(ValueError, match="reference language"):
        C.plan_batches([_pair(R_LAMP, W_LAMP), _pair("S:1", W_YOGIN)], mixed, 12)


# --------------------------------------------------------------------------- prompt and schema
def test_schema_enums_match_the_vocabulary_and_the_template():
    slot = C.SCHEMA["properties"]["pairs"]["items"]["properties"]["slots"]["items"]["properties"]
    assert slot["slot"]["enum"] == list(C.SLOTS) and slot["code"]["enum"] == list(C.CODES)
    template = C.load_template()
    for name in C.SLOTS + C.CODES:
        assert f"`{name}`" in template


def test_prompt_never_asks_for_reasoning(system):
    lowered = system.lower()
    for phrase in ("reasoning", "explain", "step by step", "think", "justify", "rationale"):
        assert phrase not in lowered


# --------------------------------------------------------------------------- examples
def test_examples_render_into_the_system_prompt(system):
    assert system.startswith(C.load_template().rstrip())
    assert system.count("### Example") == len(EXAMPLES["examples"])
    assert system.count("ANSWER\n{") == len(EXAMPLES["examples"])


def test_every_example_answer_passes_the_checks():
    lexicon = CheckLexicon.load(DATA, WIT)             # the fixture mantra transcribes the example name
    seen_codes, seen_flip = set(), False
    for n, ex in enumerate(EXAMPLES["examples"], start=1):
        batch, answer = C.example_batch(ex, EXAMPLES["reference_lang"], EXAMPLES["witness_lang"], f"example {n}")
        codes, diagnostics = C.verify_answer(answer, batch, lexicon)
        assert diagnostics == [], diagnostics
        assert sum(len(p["slots"]) for p in answer["pairs"]) == len(codes)
        assert all(c.flags <= {FLAG_CORROBORATED} for c in codes)
        seen_codes |= {c.code for c in codes}
        seen_flip |= any(c.polarity_flip for c in codes)
    assert seen_codes == set(C.CODES) and seen_flip


def test_load_examples_rejects_an_answer_out_of_order(tmp_path):
    doc = yaml.safe_load((DATA / "codebook" / "components_examples.yaml").read_text(encoding="utf-8"))
    doc["examples"][0]["answer"]["pairs"].reverse()
    (tmp_path / "codebook").mkdir()
    (tmp_path / "codebook" / "components_examples.yaml").write_text(yaml.safe_dump(doc, allow_unicode=True),
                                                                    encoding="utf-8")
    with pytest.raises(ValueError, match="in order"):
        C.load_examples(tmp_path)


# --------------------------------------------------------------------------- end to end and outputs
def _answer(batch: C.ComponentBatch) -> dict:
    return {"pairs": [{"pair": h, **FIXTURE["answers"][i.pair.ref_id]} for h, i in batch.handles().items()]}


def _script(request):
    batch = next(b for b in C.plan_batches(_pairs(), STORE, 2) if C.render_body(b) == request.body)
    return _answer(batch)


def test_code_components_runs_every_batch_with_a_fake_client(settings, system, lexicon):
    client = FakeClient(_script)
    small = C.ComponentTaskSettings(pairs_per_call=2, model=settings.model)
    codes, diagnostics = C.code_components(_pairs(), STORE, client, small, system, lexicon, workers=2)
    assert diagnostics == []
    assert len(client.requests) == 3
    assert [c.ref_id for c in codes] == [p.ref_id for p in _pairs()
                                         for _ in FIXTURE["answers"][p.ref_id]["slots"]]
    assert all(c.usable for c in codes)
    assert STORE[R_LONG].text in client.requests[2].body


def _codes(lexicon, response_for=ok_response) -> list[C.SlotCode]:
    codes = []
    for batch in C.plan_batches(_pairs(), STORE, 12):
        request = C.build_request(batch, C.ComponentTaskSettings(), "system")
        codes += C.verify(response_for(request, _answer(batch)), batch, lexicon)[0]
    return codes


def test_rendering_profile_groups_normalised_quotes_and_counts_units(lexicon):
    codes = _codes(lexicon)
    lamp_slot = FIXTURE["answers"][R_LAMP]["slots"][1]                   # patient: the lamp
    duplicate = C.SlotCode("u9", ("w9",), "patient", "sub", lamp_slot["ref_quote"] + FIXTURE["shad"],
                           lamp_slot["wit_quote"], False)            # the same quote with a trailing shad
    rows = C.rendering_profile(codes + [duplicate], "bo", "zh")
    lamp = next(r for r in rows if r.rendering_quote == lamp_slot["wit_quote"])
    assert lamp.count == 2 and lamp.units == (R_LAMP, "u9")
    assert lamp.codes == (("ret", 1), ("sub", 1)) and lamp.slots == (("patient", 2),)
    assert rows[0] is lamp                                              # most frequent first
    omitted = next(r for r in rows if r.rendering_quote == "")
    assert omitted.codes == (("om", 1),) and omitted.units == (R_YOGIN,)
    assert sum(r.count for r in rows) == len(codes) + 1
    assert set(lamp.csv_row()) == set(C.PROFILE_COLUMNS)
    assert lamp.csv_row()["units"] == f"{R_LAMP};u9" and lamp.csv_row()["n_units"] == 2


def test_rendering_profile_skips_substituted_model_hints(lexicon):
    hints = _codes(lexicon, substituted_response)
    assert hints and not any(c.usable for c in hints)
    assert C.rendering_profile(hints, "bo", "zh") == []


def test_code_record_is_json_ready(lexicon):
    record = C.code_record(_codes(lexicon)[3])
    assert tuple(record) == C.CODE_COLUMNS
    assert record["slot"] == "negation_modality" and record["flags"] == [FLAG_CORROBORATED]
    assert record["wit_ids"] == [W_LAMP] and record["reason"] is None


def test_invention_rate_counts_only_the_equivalent_sample():
    sample = C.Pair("a", ("w",), "equivalent", False, frozenset({C.SELECT_EQUIVALENT_SAMPLE}))
    sample2 = C.Pair("b", ("w",), "equivalent", False, frozenset({C.SELECT_EQUIVALENT_SAMPLE}))
    both = C.Pair("c", ("w",), "equivalent", False, frozenset({C.SELECT_EQUIVALENT_SAMPLE, C.SELECT_SENSITIVE}))
    codes = [C.SlotCode("a", ("w",), "action", "ret", "x", "y", False),
             C.SlotCode("b", ("w",), "action", "ret", "x", "y", False),
             C.SlotCode("b", ("w",), "place", "add", "", "z", False),
             C.SlotCode("c", ("w",), "place", "om", "x", "", False),
             C.SlotCode("a", ("w",), "place", "sub", "x", "y", False, reason="substituted_model")]
    assert C.invention_rate([sample, sample2, both], codes) == (1, 2)

