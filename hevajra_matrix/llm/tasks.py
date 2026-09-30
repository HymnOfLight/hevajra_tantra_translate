"""LLM tasks. Every function here turns a model call into a *proposal* that lives in
the derived layer and is checked mechanically before anything reaches the matrix
(docs/04 §2–3).

Roles implemented
    R2  alignment judge / repair            ``judge_bead``
    R3  propositional segmentation          ``segment_propositions``
    R4  lexicon-constrained component extraction  ``extract_components``, ``component_deviation``
    R5  reference-side topic pre-labelling   ``label_topic``
    R6  attribution with fixed label set     ``judge_attribution``
    R7  multi-model counterfactual           ``counterfactual_across_models``
    R8  memorisation / contamination probe   ``contamination_probe``
    R9  rationale generation (motive-guarded) ``explain_cell``

Hard rules enforced in code, not in prompts:
    * component values must be substrings of the source segment (no paraphrase);
    * attribution labels must come from ``attribution.LABELS``; motive words in any
      free-text field downgrade the answer to ``abstain`` and are logged;
    * topic labels must come from ``TOPICS``;
    * all outputs carry ``model_id`` and prompt/response hashes via ``CallLog``.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

from ..anchors import AnchorLexicon
from ..attribution import LABELS, CounterfactualHarness, CounterfactualItem, asserts_motive
from .backends import LLMBackend

# --------------------------------------------------------------------------- helpers
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def parse_json(text: str) -> dict:
    """Tolerant JSON extraction: strips <think> blocks and code fences, finds the first object."""
    t = _THINK_RE.sub("", text).strip()
    m = _FENCE_RE.search(t)
    if m:
        t = m.group(1).strip()
    if not t.startswith("{"):
        i = t.find("{")
        j = t.rfind("}")
        if i < 0 or j < 0:
            raise ValueError(f"no JSON object in model output: {text[:120]!r}")
        t = t[i:j + 1]
    return json.loads(t)


def _sys(lang_hint: str = "") -> str:
    return ("You are a careful philological assistant. You never speculate about the motives of "
            "translators or scribes. You answer only in the requested JSON. " + lang_hint).strip()


# --------------------------------------------------------------------------- R4 component extraction
SLOTS = ("actor", "action", "patient", "instrument", "negation", "condition", "quantity", "result")

COMPONENT_SCHEMA = {
    "type": "object",
    "properties": {**{s: {"type": "array", "items": {"type": "string"}} for s in SLOTS},
                   "lexicon_keys": {"type": "array", "items": {"type": "string"}}},
    "required": list(SLOTS),
}

COMPONENT_PROMPT = (
    "Segment ({lang}):\n{text}\n\n"
    "Known lexicon anchors detected in this segment: {anchors}\n\n"
    "Decompose the segment into propositional components. For each slot list the *exact substrings* of "
    "the segment that fill it (copy characters verbatim, do not translate, do not paraphrase, leave the "
    "list empty if the slot is not expressed):\n"
    "  actor, action, patient, instrument, negation, condition, quantity, result.\n"
    "Also list which of the lexicon anchors above are actually used (lexicon_keys)."
)


@dataclass
class ComponentSet:
    text: str
    lang: str
    model_id: str
    slots: dict[str, list[str]] = field(default_factory=dict)
    lexicon_keys: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)   # values not found verbatim in the text

    def present(self) -> set[str]:
        return {s for s, v in self.slots.items() if v}

    def to_row(self) -> dict:
        return asdict(self)


def extract_components(backend: LLMBackend, text: str, lang: str, lexicon: AnchorLexicon | None = None) -> ComponentSet:
    anchors = sorted(lexicon.extract(text, lang)) if lexicon is not None else []
    prompt = COMPONENT_PROMPT.format(lang=lang, text=text, anchors=", ".join(anchors) or "(none)")
    raw = backend.generate(prompt, system=_sys(), json_schema=COMPONENT_SCHEMA, max_tokens=512)
    data = parse_json(raw)
    cs = ComponentSet(text=text, lang=lang, model_id=backend.model_id)
    for s in SLOTS:
        vals = data.get(s) or []
        keep: list[str] = []
        for v in vals:
            v = str(v).strip()
            if v and v in text:
                keep.append(v)
            elif v:
                cs.rejected.append(f"{s}:{v}")
        cs.slots[s] = keep
    allowed = set(anchors)
    cs.lexicon_keys = [k for k in (data.get("lexicon_keys") or []) if k in allowed]
    return cs


def component_deviation(ref: ComponentSet, wit: ComponentSet) -> dict[str, float | int]:
    """d_comp: per slot, 1 if the reference expresses the slot and the witness does not.

    Returns the loss rate over reference-filled slots plus per-slot flags and the
    count of witness-only slots (additions). Negation loss is reported separately
    because a dropped negation inverts meaning (docs/02 §4.6).
    """
    rp, wp = ref.present(), wit.present()
    lost = sorted(rp - wp)
    added = sorted(wp - rp)
    out: dict[str, float | int] = {
        "n_ref_slots": len(rp), "n_lost": len(lost), "n_added": len(added),
        "d_comp": (len(lost) / len(rp)) if rp else 0.0,
        "negation_lost": int("negation" in lost),
        "quantity_lost": int("quantity" in lost),
    }
    for s in SLOTS:
        out[f"lost_{s}"] = int(s in lost)
    return out


# --------------------------------------------------------------------------- R3 propositional segmentation
SEGMENT_SCHEMA = {"type": "object", "properties": {"propositions": {"type": "array", "items": {"type": "string"}}},
                  "required": ["propositions"]}

SEGMENT_PROMPT = (
    "Text ({lang}):\n{text}\n\n"
    "Split this text into minimal propositions (one predication each). Return the propositions as exact, "
    "contiguous, non-overlapping substrings in original order, covering the whole text. Do not alter, "
    "translate or normalise any character."
)


def segment_propositions(backend: LLMBackend, text: str, lang: str) -> list[str]:
    """Returns the proposed cut; falls back to [text] when the proposal does not tile the input."""
    raw = backend.generate(SEGMENT_PROMPT.format(lang=lang, text=text), system=_sys(), json_schema=SEGMENT_SCHEMA, max_tokens=768)
    props = [str(p) for p in parse_json(raw).get("propositions", []) if str(p).strip()]
    pos = 0
    for p in props:
        i = text.find(p, pos)
        if i < 0:
            return [text]
        pos = i + len(p)
    covered = sum(len(p) for p in props)
    return props if props and covered >= 0.9 * len(text.replace(" ", "")) else [text]


# --------------------------------------------------------------------------- R2 alignment judge
VERDICTS = ("accept", "shift_left", "shift_right", "split", "merge", "make_null", "unsure")
JUDGE_SCHEMA = {"type": "object",
                "properties": {"verdict": {"type": "string", "enum": list(VERDICTS)},
                               "confidence": {"type": "number"},
                               "shared_content": {"type": "array", "items": {"type": "string"}},
                               "note": {"type": "string"}},
                "required": ["verdict", "confidence"]}

JUDGE_PROMPT = (
    "Two texts are candidate counterparts produced by an automatic aligner (bead shape {shape}).\n\n"
    "Reference ({ref_lang}):\n{ref}\n\n"
    "Witness ({wit_lang}):\n{wit}\n\n"
    "Preceding witness segment: {wit_prev}\nFollowing witness segment: {wit_next}\n\n"
    "Judge whether they express the same content. Verdicts: accept | shift_left (witness counterpart is "
    "the preceding segment) | shift_right | split (witness covers more than the reference) | merge (witness "
    "covers less; next segment continues it) | make_null (no counterpart exists) | unsure. "
    "List the concrete shared content elements (names, numbers, actions) you relied on."
)


@dataclass
class BeadJudgement:
    verdict: str
    confidence: float
    shared_content: list[str]
    note: str
    model_id: str


def judge_bead(backend: LLMBackend, ref: str, wit: str, ref_lang: str, wit_lang: str, shape: str = "1:1",
               wit_prev: str = "", wit_next: str = "") -> BeadJudgement:
    raw = backend.generate(JUDGE_PROMPT.format(shape=shape, ref_lang=ref_lang, ref=ref, wit_lang=wit_lang, wit=wit,
                                               wit_prev=wit_prev or "(none)", wit_next=wit_next or "(none)"),
                           system=_sys(), json_schema=JUDGE_SCHEMA, max_tokens=400)
    d = parse_json(raw)
    verdict = d.get("verdict") if d.get("verdict") in VERDICTS else "unsure"
    try:
        conf = max(0.0, min(1.0, float(d.get("confidence", 0.0))))
    except (TypeError, ValueError):
        conf = 0.0
    return BeadJudgement(verdict=verdict, confidence=conf, shared_content=[str(x) for x in d.get("shared_content", [])],
                         note=str(d.get("note", "")), model_id=backend.model_id)


def judge_review_rows(backend: LLMBackend, rows: Iterable[dict], ref_lang: str, wit_lang: str,
                      only_shapes: Sequence[str] = ("1:0", "0:1", "1:3", "3:1", "2:2"), min_confidence: float = 0.6) -> list[dict]:
    """Fill ``review_status``/``review_note`` of alignment_review.csv rows with model verdicts.

    Only suspicious bead shapes are sent by default. The status is prefixed ``llm:`` so
    that a human reviewer can distinguish it from their own decision; the evidence grade
    of the cell is *not* changed by this function.
    """
    rows = list(rows)
    for k, r in enumerate(rows):
        if r.get("shape") not in only_shapes or r.get("review_status"):
            continue
        prev_ = rows[k - 1]["wit_text"] if k > 0 else ""
        next_ = rows[k + 1]["wit_text"] if k + 1 < len(rows) else ""
        j = judge_bead(backend, r.get("ref_text", ""), r.get("wit_text", ""), ref_lang, wit_lang, r.get("shape", ""), prev_, next_)
        r["review_status"] = f"llm:{j.verdict}" if j.confidence >= min_confidence else "llm:unsure"
        r["review_note"] = f"[{j.model_id} c={j.confidence:.2f}] {'; '.join(j.shared_content)} {j.note}".strip()
    return rows


# --------------------------------------------------------------------------- R5 topic labels (reference side only)
TOPICS = ("sexual_yoga", "violent_rite", "transgressive_substance", "charnel_imagery", "mantra_ritual",
          "doctrine_philosophy", "iconography", "narrative_frame", "other")
TOPIC_SCHEMA = {"type": "object", "properties": {"topic": {"type": "string", "enum": list(TOPICS)},
                                                 "cues": {"type": "array", "items": {"type": "string"}}},
                "required": ["topic"]}
TOPIC_PROMPT = (
    "Reference unit {unit} ({lang}):\n{text}\n\n"
    "Assign exactly one topic label from: {topics}. Quote the words (verbatim substrings) that motivate the label. "
    "Label the content of this unit only; you are not told, and must not guess, how any translation renders it."
)


def label_topic(backend: LLMBackend, unit: str, text: str, lang: str) -> dict:
    raw = backend.generate(TOPIC_PROMPT.format(unit=unit, lang=lang, text=text, topics=", ".join(TOPICS)),
                           system=_sys(), json_schema=TOPIC_SCHEMA, max_tokens=200)
    d = parse_json(raw)
    topic = d.get("topic") if d.get("topic") in TOPICS else "other"
    cues = [c for c in (d.get("cues") or []) if isinstance(c, str) and c in text]
    return {"unit": unit, "topic": topic, "cues": cues, "model_id": backend.model_id}


# --------------------------------------------------------------------------- R6 attribution judge
ATTR_SCHEMA = {"type": "object",
               "properties": {"distribution": {"type": "object", "properties": {k: {"type": "number"} for k in LABELS}},
                              "cited_facts": {"type": "array", "items": {"type": "integer"}},
                              "note": {"type": "string"}},
               "required": ["distribution"]}
ATTR_PROMPT = (
    "A reference unit is not fully retained by witness {witness}.\n\n"
    "Numbered facts (the ONLY admissible evidence):\n{facts}\n\n"
    "Distribute probability over exactly these labels: {labels}.\n"
    "  vorlage_attested          – some independent witness of the source text also lacks it\n"
    "  shared_with_cowitness     – an independent translation from the same source also lacks it\n"
    "  lacuna                    – the witness is physically damaged / illegible here\n"
    "  unexplained_at_translation – only this witness lacks it and no fact above explains that\n"
    "  abstain                   – the facts are insufficient\n"
    "Cite the fact numbers you used. Do not name or infer any motive."
)


@dataclass
class AttributionJudgement:
    unit: str
    witness: str
    distribution: dict[str, float]
    cited_facts: list[int]
    note: str
    model_id: str
    motive_flag: bool = False

    @property
    def top(self) -> str:
        return max(self.distribution.items(), key=lambda kv: kv[1])[0]


def judge_attribution(backend: LLMBackend, unit: str, witness: str, facts: Sequence[str]) -> AttributionJudgement:
    numbered = "\n".join(f"[{i + 1}] {f}" for i, f in enumerate(facts)) or "[1] (no facts available)"
    raw = backend.generate(ATTR_PROMPT.format(witness=witness, facts=numbered, labels=", ".join(LABELS)),
                           system=_sys(), json_schema=ATTR_SCHEMA, max_tokens=300)
    d = parse_json(raw)
    dist_in = d.get("distribution") or {}
    dist = {}
    for k in LABELS:
        try:
            v = float(dist_in.get(k, 0.0))
        except (TypeError, ValueError):
            v = 0.0
        dist[k] = max(0.0, v)
    total = sum(dist.values())
    dist = {k: v / total for k, v in dist.items()} if total > 0 else {"abstain": 1.0}
    note = str(d.get("note", ""))
    cited = [int(i) for i in d.get("cited_facts", []) if isinstance(i, (int, float)) and 1 <= int(i) <= len(facts)]
    flag = asserts_motive(note)
    if flag:
        # motive language is inadmissible: downgrade, keep the note for the audit log
        dist = {"abstain": 1.0}
    if not facts:
        dist = {"abstain": 1.0}
    return AttributionJudgement(unit=unit, witness=witness, distribution=dist, cited_facts=cited, note=note,
                                model_id=backend.model_id, motive_flag=flag)


# --------------------------------------------------------------------------- R7 counterfactual across models
def counterfactual_across_models(backends: Sequence[LLMBackend], items: Sequence[CounterfactualItem]) -> list[dict]:
    """Run the over-attribution counterfactual for each model; returns one row per model."""
    rows = []
    for b in backends:
        res = CounterfactualHarness(lambda p, b=b: b.generate(p, system=None, max_tokens=200)).run(items)
        rows.append({"model_id": b.model_id, "n": res.n, "motive_rate_without_fact": res.rate_a,
                     "motive_rate_with_fact": res.rate_b, "drop": res.diff})
    return rows


# --------------------------------------------------------------------------- R8 contamination probe
@dataclass
class ProbeItem:
    source: str      # e.g. "84000_en", "snellgrove1959_en", "willemen1983_en", "cbeta_T0892_zh", "derge_bo"
    ref: str         # unit id / coordinate
    prefix: str
    continuation: str


def _ngrams(s: str, n: int) -> set[str]:
    s = re.sub(r"\s+", " ", s.strip())
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def ngram_overlap(a: str, b: str, n: int = 4) -> float:
    ga, gb = _ngrams(a, n), _ngrams(b, n)
    return len(ga & gb) / len(ga) if ga else 0.0


PROBE_PROMPT = "Continue the following passage exactly as it appears in its published source. Output only the continuation.\n\n{prefix}"


def contamination_probe(backend: LLMBackend, items: Sequence[ProbeItem], n: int = 4, threshold: float = 0.5) -> dict:
    """Cloze-style memorisation test. High overlap means the model has seen the exact text,
    so any judgement it makes about that witness is contaminated and must be reported as such
    (docs/04 §5). Continuations are compared by character n-gram recall against the truth."""
    per = []
    for it in items:
        out = backend.generate(PROBE_PROMPT.format(prefix=it.prefix), system=None, max_tokens=max(32, len(it.continuation) // 2 + 16))
        out = _THINK_RE.sub("", out)
        ov = ngram_overlap(it.continuation, out[: len(it.continuation) * 2], n)
        per.append({"source": it.source, "ref": it.ref, "overlap": round(ov, 3), "memorised": ov >= threshold})
    by_source: dict[str, list[float]] = {}
    for r in per:
        by_source.setdefault(r["source"], []).append(r["overlap"])
    return {"model_id": backend.model_id, "n": len(per), "threshold": threshold,
            "mean_overlap": (sum(r["overlap"] for r in per) / len(per)) if per else 0.0,
            "memorised_rate": (sum(r["memorised"] for r in per) / len(per)) if per else 0.0,
            "by_source": {k: {"n": len(v), "mean_overlap": sum(v) / len(v), "memorised_rate": sum(x >= threshold for x in v) / len(v)}
                          for k, v in by_source.items()},
            "items": per}


def probe_items_from_segments(texts: Sequence[tuple[str, str, str]], prefix_chars: int = 40, min_chars: int = 80) -> list[ProbeItem]:
    """(source, ref, text) → ProbeItem with a fixed-length prefix; short texts are skipped."""
    out = []
    for source, ref, text in texts:
        t = text.strip()
        if len(t) < min_chars:
            continue
        out.append(ProbeItem(source=source, ref=ref, prefix=t[:prefix_chars], continuation=t[prefix_chars:]))
    return out


# --------------------------------------------------------------------------- R9 rationale (guarded)
EXPLAIN_PROMPT = (
    "Describe, in at most three sentences and without speculating about anyone's intentions, how the witness "
    "text relates to the reference unit. Mention only observable differences (missing, added, reordered, "
    "reworded, transliterated content).\n\nReference {unit}:\n{ref}\n\nWitness {witness}:\n{wit}\n\n"
    "Matrix record: status={status}, bead={bead}, d_len={d_len}, d_lit={d_lit}."
)


def explain_cell(backend: LLMBackend, unit: str, witness: str, ref: str, wit: str, status: str, bead: str,
                 d_len: float | None, d_lit: float | None) -> dict:
    text = backend.generate(EXPLAIN_PROMPT.format(unit=unit, witness=witness, ref=ref, wit=wit or "(no counterpart)",
                                                  status=status, bead=bead,
                                                  d_len="" if d_len is None else f"{d_len:.2f}",
                                                  d_lit="" if d_lit is None else f"{d_lit:.2f}"),
                            system=_sys(), max_tokens=200)
    text = _THINK_RE.sub("", text).strip()
    return {"unit": unit, "witness": witness, "rationale": text, "motive_flag": asserts_motive(text), "model_id": backend.model_id}
