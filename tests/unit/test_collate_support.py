"""Shared builders for the collate tests, plus sanity tests of the fixture they read.

The other ``test_collate_*`` modules import from here (pytest puts this directory on
``sys.path``). All multilingual text comes from ``data/fixtures/collate_texts.yaml``;
quotes are cut from segment texts, so no non-Latin literal appears in a .py file.

``echo_script`` is the fake T1 collator of the end-to-end tests: it reads the request
(nothing else), recognises each line's text in the fixture, and answers with the fixture's
gold links, quoting whole segment texts.
"""

from __future__ import annotations

import re
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import yaml

from hevajra_matrix.collate.collator import (
    SCHEMA,
    Examples,
    RawCollation,
    RawUnit,
    RawWitnessOnly,
    TaskSettings,
    load_examples,
    load_template,
)
from hevajra_matrix.collate.verify import CheckLexicon
from hevajra_matrix.collate.windows import Window, WindowParams, plan
from hevajra_matrix.core.types import Relation, Segment, WitnessOnlyKind
from hevajra_matrix.llm.client import LLMRequest
from hevajra_matrix.registry import ChapterSpan, Concordance

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIXTURE = DATA / "fixtures" / "collate_texts.yaml"
SOURCE = "claude-opus-5-5:collate.v1:r1"
_LINE_RE = re.compile(r"^([rz]\d+)\t([a-z_]+)\t(.*)$")


# --------------------------------------------------------------------------- fixture
@lru_cache(maxsize=1)
def _doc() -> Mapping[str, Any]:
    return yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))


def _segments(side: str) -> tuple[Segment, ...]:
    block = _doc()[side]
    return tuple(
        Segment(id=s["id"], witness=block["witness"], lang=block["lang"], text=s["text"], start=s["id"],
                end=s["id"], kind=s["kind"], local_chapter=s.get("local"), chapter=s.get("chapter"),
                extra=MappingProxyType({"host": s["host"]} if "host" in s else {}))
        for s in block["segments"])


def reference() -> tuple[Segment, ...]:
    return _segments("reference")


def witness() -> tuple[Segment, ...]:
    return _segments("witness")


def gold() -> dict[str, tuple[tuple[str, ...], Relation]]:
    """Reference unit id -> (witness segment ids, relation)."""
    return {g["ref"]: (tuple(g["wit"]), Relation(g["relation"])) for g in _doc()["gold"]}


def gold_witness_only() -> dict[str, WitnessOnlyKind]:
    return {sid: WitnessOnlyKind(kind) for sid, kind in _doc()["gold_witness_only"].items()}


def quote(name: str) -> str:
    return _doc()["quotes"][name]


def seg(sid: str) -> Segment:
    return {s.id: s for s in (*reference(), *witness())}[sid]


# --------------------------------------------------------------------------- concordance and windows
LOCALS = ("pin1", "pin2", "pin3")


def concordance(mapping: Mapping[str, Sequence[str]] | None = None, witness_id: str = "zh_test",
                locals_: Sequence[str] = LOCALS) -> Concordance:
    """A concordance for the fixture: I.1 -> pin1 and I.2 -> pin2 unless ``mapping`` says otherwise."""
    mapping = mapping or {"I.1": ["pin1"], "I.2": ["pin2"]}
    spans = {ref: MappingProxyType({witness_id: tuple(ChapterSpan(local, "proposed", evidence=("test",),
                                                                  source="test") for local in locals)})
             for ref, locals in mapping.items()}
    return Concordance("test", MappingProxyType({witness_id: tuple(locals_)}), MappingProxyType(spans))


def windows(max_ref_units: int = 150, overlap: int = 0, neighbours: int = 0,
            conc: Concordance | None = None) -> list[Window]:
    return plan(reference(), witness(), conc or concordance(), WindowParams(max_ref_units, overlap, neighbours))


def window(chapter: str = "I.1", **kwargs: Any) -> Window:
    return next(w for w in windows(**kwargs) if w.chapter == chapter)


@lru_cache(maxsize=1)
def lexicon() -> CheckLexicon:
    return CheckLexicon.load(DATA, witness())


@lru_cache(maxsize=1)
def examples() -> Examples:
    return load_examples(DATA)


@lru_cache(maxsize=1)
def template() -> str:
    return load_template()


SETTINGS = TaskSettings()


# --------------------------------------------------------------------------- answers
def unit(w: Window, uid: str, wit: Sequence[str] = (), relation: Relation | str = Relation.EQUIVALENT,
         polarity_flip: bool = False, ref_quote: str | None = None, wit_quote: str | None = None,
         confidence: str = "high") -> dict[str, Any]:
    """One answer record for unit ``uid`` linking segment ids ``wit`` (converted to handles).

    Quotes default to the whole unit text and the first linked segment's text.
    """
    relation = Relation(relation)
    return {
        "ref": w.handle_of[uid],
        "wit": [w.handle_of[i] for i in wit],
        "relation": relation.value,
        "polarity_flip": polarity_flip,
        "confidence": confidence,
        "ref_quote": w.segment[uid].text if ref_quote is None else ref_quote,
        "wit_quote": (w.segment[wit[0]].text if wit else "") if wit_quote is None else wit_quote,
    }


def witness_only(w: Window, wit: Sequence[str], kind: WitnessOnlyKind | str,
                 wit_quote: str | None = None) -> dict[str, Any]:
    return {"wit": [w.handle_of[i] for i in wit], "kind": WitnessOnlyKind(kind).value,
            "wit_quote": w.segment[wit[0]].text if wit_quote is None else wit_quote}


def gold_answer(w: Window) -> dict[str, Any]:
    """The fixture gold for ``w``: every unit linked as in the gold, and every core-window
    segment not linked by the chunk declared witness-only (its gold kind, else
    ``belongs_elsewhere``)."""
    g = gold()
    units = []
    for u in w.units:
        wit, relation = g[u.id]
        units.append(unit(w, u.id, wit, relation, polarity_flip=relation is Relation.REVERSAL))
    linked = {i for u in w.units for i in g[u.id][0]}
    kinds = gold_witness_only()
    wo = [witness_only(w, [s.id], kinds.get(s.id, WitnessOnlyKind.BELONGS_ELSEWHERE))
          for s in w.text if s.id in w.core and s.id not in linked]
    return {"units": units, "witness_only": wo}


def raw(w: Window, answer: Mapping[str, Any]) -> RawCollation:
    """A parsed answer, built without a response (for verify tests)."""
    return RawCollation(
        w.key,
        tuple(RawUnit(u["ref"], tuple(u["wit"]), Relation(u["relation"]), u["polarity_flip"], u["confidence"],
                      u["ref_quote"], u["wit_quote"]) for u in answer["units"]),
        tuple(RawWitnessOnly(tuple(x["wit"]), WitnessOnlyKind(x["kind"]), x["wit_quote"])
              for x in answer["witness_only"]),
    )


def request_lines(text: str) -> list[tuple[str, str, str]]:
    return [m.groups() for m in map(_LINE_RE.match, text.splitlines()) if m]


def echo_script(request: LLMRequest) -> dict[str, Any]:
    """Fake collator: recover the window from the request text and answer with the gold.

    A unit whose gold counterpart is not in the request's witness text is answered
    ``no_counterpart``: a perfect reader of a text with deletions.
    """
    by_text = {s.text: s.id for s in (*reference(), *witness())}
    handle_of = {by_text[text]: h for h, _, text in request_lines(request.context or "")}
    handle_of |= {by_text[text]: h for h, _, text in request_lines(request.body)}
    ref_ids = [by_text[text] for _, _, text in request_lines(request.body)]
    core_line = next(line for line in request.body.splitlines() if line.startswith("CORE WINDOW: "))
    core: set[str] = set()
    for part in core_line.removeprefix("CORE WINDOW: ").split(", "):
        first, _, last = part.partition("-")
        core |= {h for h in handle_of.values() if h.startswith("z") and first <= h <= (last or first)}
    g, kinds, text_of = gold(), gold_witness_only(), {s.id: s.text for s in witness()}
    units = []
    for uid in ref_ids:
        wit, relation = g[uid]
        if all(i in handle_of for i in wit):
            units.append({"ref": handle_of[uid], "wit": [handle_of[i] for i in wit], "relation": relation.value,
                          "polarity_flip": relation is Relation.REVERSAL, "confidence": "high",
                          "ref_quote": seg(uid).text, "wit_quote": text_of[wit[0]]})
        else:   # the counterpart was deleted from the text shown (deletion perturbation)
            units.append({"ref": handle_of[uid], "wit": [], "relation": Relation.NO_COUNTERPART.value,
                          "polarity_flip": False, "confidence": "high", "ref_quote": seg(uid).text,
                          "wit_quote": ""})
    linked = {i for uid in ref_ids for i in g[uid][0]}
    wo = [{"wit": [handle_of[sid]], "kind": kinds.get(sid, WitnessOnlyKind.BELONGS_ELSEWHERE).value,
           "wit_quote": text_of[sid]}
          for sid in text_of if sid in handle_of and handle_of[sid] in core and sid not in linked]
    return {"units": units, "witness_only": wo}


# --------------------------------------------------------------------------- fixture sanity
def test_fixture_gold_covers_every_alignable_unit() -> None:
    alignable = [s.id for s in reference() if s.kind != "colophon"]
    assert sorted(gold()) == sorted(alignable)


def test_fixture_gold_answers_are_schema_valid() -> None:
    from hevajra_matrix.llm.schema import validate
    for w in windows():
        assert validate(gold_answer(w), SCHEMA) == []


def test_fixture_segments_are_distinct_texts() -> None:
    texts = [s.text for s in (*reference(), *witness())]
    assert len(texts) == len(set(texts)), "echo_script identifies lines by their text"


def test_concordance_helper_builds_core_windows() -> None:
    assert concordance().core_window("I.1", "zh_test", 0) == frozenset({"pin1"})
    assert concordance().core_window("I.1", "zh_test", 1) == frozenset({"pin1", "pin2"})


def test_replace_keeps_window_valid() -> None:
    w = window()
    assert replace(w, key="x").units == w.units
