"""Cross-lingual anchors: language-neutral keys found in a segment of any witness.

An anchor is a word or name whose translation is stable across the witnesses, e.g.
``term:kapāla`` (skull cup) or ``name:vajragarbha``, plus every numeral (``num:32``).
The B0 baseline reduces each segment to its set of anchor keys (its *bag*) and scores
candidate beads by the overlap of bags (``align.similarity.AnchorSimilarity``).

The lexicon is ``data/lexicon/anchors.yaml`` (schema documented at the top of that file).
Matching rules, all on the tokens of ``core.textnorm.chunks`` (so a match never crosses
punctuation):

    bo  a form matches whole syllables only, so the syllable for "flesh" does not match
        inside the syllable for "know" (v0.2 matched substrings of the Unicode string);
    zh  a form matches a character sequence; the loader rejects single-character forms,
        because single characters recur inside transliterations (the character for "sun"
        inside the transliteration of vajra);
    sa  each word of a form matches the beginning of a word, a crude stand-in for
        inflection: "vajra" matches "vajrasya" and "vajrasattva" (so list the compound
        as a form of its own when it is a different anchor), but a stem whose final
        vowel changes is missed (the locative "śmaśāne" of "śmaśāna").

Forms are matched leftmost-longest: at each position the longest form starting there
wins and its tokens are consumed, so the name Hevajra does not also yield ``term:vajra``.
A form does not count where it overlaps an occurrence of one of its anchor's
``exclude`` contexts (e.g. the syllable for "empowerment" inside the word for "lord").
Numerals come from ``core.textnorm.numerals`` with the exclusion contexts of
``data/lexicon/numerals.yaml`` (e.g. the Tibetan word for "emptiness" is not 1000).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..core.lexicon import LexiconError, NumeralTable, load_numerals, read_lexicon
from ..core.textnorm import chunks, numerals

LANGS: tuple[str, ...] = ("sa", "bo", "zh")          # every anchor has forms in all three
TYPES = frozenset({"name", "term", "mantra", "place"})
NUMERAL_PREFIX = "num"
_FIELDS = {"key", "type", "note", "exclude", *LANGS}
_KEY_RE = re.compile(r"[^\s:]+")


@dataclass(frozen=True)
class Anchor:
    """One lexicon entry: surface forms and exclusion contexts per language."""

    key: str                                       # full key, "<type>:<name>"
    forms: Mapping[str, tuple[str, ...]]           # language -> forms
    exclusions: Mapping[str, tuple[str, ...]]      # language -> contexts where no form counts


@dataclass(frozen=True)
class _Pattern:
    """A form compiled to tokens, with its anchor's exclusion contexts."""

    key: str
    tokens: tuple[str, ...]
    exclusions: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class AnchorLexicon:
    """The anchor entries plus the numeral table; the pattern index is derived once.

    Build it with ``load_anchor_lexicon`` or from ``parse_anchors``, which validate the
    entries the index relies on (every form one token sequence, no form in two anchors).
    """

    anchors: tuple[Anchor, ...]
    numerals: NumeralTable
    _index: Mapping[str, Mapping[str, tuple[_Pattern, ...]]] = field(
        init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_index", _build_index(self.anchors))

    def keys(self) -> frozenset[str]:
        return frozenset(a.key for a in self.anchors)


# --------------------------------------------------------------------------- loading
def load_anchor_lexicon(data_dir: Path) -> AnchorLexicon:
    """Read ``data/lexicon/anchors.yaml`` and ``numerals.yaml``; malformed entries raise."""
    doc = read_lexicon(data_dir, "anchors")
    if set(doc) != {"anchors"}:
        raise LexiconError(f"anchors.yaml: expected exactly one top-level key 'anchors', got {sorted(doc)}")
    return AnchorLexicon(anchors=parse_anchors(doc["anchors"]), numerals=load_numerals(data_dir))


def parse_anchors(records: Any) -> tuple[Anchor, ...]:
    """Validate the ``anchors`` list of anchors.yaml (see the schema in that file)."""
    if not isinstance(records, (list, tuple)):
        raise LexiconError("anchors.yaml: 'anchors' must be a list of records")
    out: list[Anchor] = []
    owner: dict[tuple[str, tuple[str, ...]], str] = {}   # (language, form tokens) -> key
    for rec in records:
        anchor = _parse_record(rec)
        if any(a.key == anchor.key for a in out):
            raise LexiconError(f"anchors.yaml: duplicate key {anchor.key!r}")
        for lang, forms in anchor.forms.items():
            for form in forms:
                other = owner.setdefault((lang, tuple(chunks(form, lang)[0])), anchor.key)
                if other != anchor.key:
                    raise LexiconError(f"anchors.yaml: {lang} form {form!r} belongs to {other} and {anchor.key}")
        out.append(anchor)
    return tuple(out)


def _parse_record(rec: Any) -> Anchor:
    if not isinstance(rec, Mapping):
        raise LexiconError(f"anchors.yaml: expected a record, got {rec!r}")
    where = f"anchors.yaml: entry {rec.get('key')!r}"
    unknown = sorted(set(rec) - _FIELDS)
    if unknown:
        raise LexiconError(f"{where}: unknown field(s) {unknown}; allowed: {sorted(_FIELDS)}")
    name, kind = rec.get("key"), rec.get("type")
    if not isinstance(name, str) or not _KEY_RE.fullmatch(name):
        raise LexiconError(f"{where}: 'key' must be a non-empty string without spaces or ':'")
    if kind not in TYPES:
        raise LexiconError(f"{where}: 'type' must be one of {sorted(TYPES)}")
    if not isinstance(rec.get("note", ""), str):
        raise LexiconError(f"{where}: 'note' must be a string")
    forms = {lang: _forms(rec.get(lang), lang, f"{where}.{lang}", required=True) for lang in LANGS}
    exclude = rec.get("exclude", {})
    if not isinstance(exclude, Mapping) or set(exclude) - set(LANGS):
        raise LexiconError(f"{where}.exclude: must map some of {list(LANGS)} to lists of contexts")
    exclusions = {lang: _forms(v, lang, f"{where}.exclude.{lang}", required=False) for lang, v in exclude.items()}
    return Anchor(key=f"{kind}:{name}", forms=MappingProxyType(forms), exclusions=MappingProxyType(exclusions))


def _forms(items: Any, lang: str, where: str, required: bool) -> tuple[str, ...]:
    """A list of distinct forms, each one token sequence without punctuation."""
    if not isinstance(items, (list, tuple)) or (required and not items):
        raise LexiconError(f"{where}: expected a {'non-empty ' if required else ''}list of forms")
    out = []
    for form in items:
        if not isinstance(form, str):
            raise LexiconError(f"{where}: {form!r} is not a string (quote YAML words such as no, on)")
        parts = chunks(form, lang)
        if len(parts) != 1:
            raise LexiconError(f"{where}: {form!r} is empty or contains punctuation")
        if lang == "zh" and len(parts[0]) < 2:
            raise LexiconError(f"{where}: single-character Chinese form {form!r} is not allowed")
        out.append(form.strip())
    if len(set(out)) != len(out):
        raise LexiconError(f"{where}: duplicate forms")
    return tuple(out)


def _build_index(anchors: Sequence[Anchor]) -> Mapping[str, Mapping[str, tuple[_Pattern, ...]]]:
    """language -> index token -> patterns, longest first (token count, then characters)."""
    by_lang: dict[str, list[tuple[int, _Pattern]]] = {}
    for order, anchor in enumerate(anchors):
        for lang, forms in anchor.forms.items():
            excl = tuple(tuple(chunks(c, lang)[0]) for c in anchor.exclusions.get(lang, ()))
            for form in forms:
                pattern = _Pattern(anchor.key, tuple(chunks(form, lang)[0]), excl)
                by_lang.setdefault(lang, []).append((order, pattern))
    index: dict[str, Mapping[str, tuple[_Pattern, ...]]] = {}
    for lang, entries in by_lang.items():
        entries.sort(key=lambda e: (-len(e[1].tokens), -len("".join(e[1].tokens)), e[0]))
        buckets: dict[str, list[_Pattern]] = {}
        for _, pattern in entries:
            buckets.setdefault(_index_token(pattern.tokens[0], lang), []).append(pattern)
        index[lang] = MappingProxyType({k: tuple(v) for k, v in buckets.items()})
    return MappingProxyType(index)


# --------------------------------------------------------------------------- extraction
def extract(text: str, lang: str, lexicon: AnchorLexicon) -> frozenset[str]:
    """Anchor keys of ``text``: lexicon keys (leftmost-longest, exclusions applied) and numerals.

    Languages other than sa, bo and zh have no forms and no numerals, so they yield
    the empty set.
    """
    keys: set[str] = set()
    index = lexicon._index.get(lang, {})
    for tokens in chunks(text, lang):
        i = 0
        while i < len(tokens):
            hit = _match_at(tokens, i, index.get(_index_token(tokens[i], lang), ()), lang)
            if hit is None:
                i += 1
            else:
                keys.add(hit.key)
                i += len(hit.tokens)
    keys.update(f"{NUMERAL_PREFIX}:{n}" for n in numerals(text, lang, lexicon.numerals))
    return frozenset(keys)


def _index_token(token: str, lang: str) -> str:
    """Bucket key: the whole token, or its first letter for prefix-matched Sanskrit."""
    return token[:1] if lang == "sa" else token


def _match_at(tokens: Sequence[str], i: int, patterns: Sequence[_Pattern], lang: str) -> _Pattern | None:
    """The first (longest) pattern occurring at ``i`` and not overlapping an exclusion."""
    for p in patterns:
        n = len(p.tokens)
        if _occurs(tokens, i, p.tokens, lang) and not any(
            _occurs(tokens, s, ctx, lang)
            for ctx in p.exclusions
            for s in range(max(0, i - len(ctx) + 1), i + n)
        ):
            return p
    return None


def _occurs(tokens: Sequence[str], i: int, seq: Sequence[str], lang: str) -> bool:
    if i + len(seq) > len(tokens):
        return False
    if lang == "sa":
        return all(tokens[i + k].startswith(t) for k, t in enumerate(seq))
    return all(tokens[i + k] == t for k, t in enumerate(seq))
