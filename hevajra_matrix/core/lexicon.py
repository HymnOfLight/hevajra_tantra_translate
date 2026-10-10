"""Loaders for the deterministic linguistic resources in ``data/lexicon/*.yaml``.

Each resource is a small frozen value built by its own loader; there is deliberately no
central lexicon object, so a module depends only on the files it actually reads.

    numerals.yaml       -> NumeralTable   (core.textnorm.numerals)
    negators.yaml       -> Negators       (polarity checks, negation-aware baselines)
    variant_forms.yaml  -> VariantForms   (core.textnorm.fold_variants)
    codebook/collate.v2/standing_renderings*.yaml
                        -> StandingRenderings (load_standing_renderings; the collate.v2
                           glossary, read from an explicit path)
    any other file      -> read_lexicon() returns it as a read-only mapping; its owner
                           module interprets it (anchors, Derge markers, note classes, ...)

Source-language forms are YAML *values* (the language policy keeps keys English), so
tables are written as lists of records such as ``{form: ..., value: 32}``. Every string
is NFC-normalised on load; Latin-script forms are also lower-cased, matching the
tokeniser in ``core.textnorm``. Loaders reject unknown keys and malformed records.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .io import read_yaml
from .textnorm import chunks

_NAME_RE = re.compile(r"[a-z][a-z0-9_]*")


class LexiconError(ValueError):
    """A lexicon file is missing required entries or contains malformed ones."""


@dataclass(frozen=True)
class ZhNumerals:
    digits: Mapping[str, int]          # single characters, 0-9
    units: Mapping[str, int]           # single characters: 10, 100, 1000, 10000, ...
    exclusions: tuple[str, ...]        # contexts in which numeral characters are not numerals


@dataclass(frozen=True)
class BoNumerals:
    words: Mapping[str, int]           # tsheg-joined syllables -> value (units, tens, 100, ...)
    teen_prefixes: frozenset[str]      # "ten" syllables that combine with a following unit
    connectors: frozenset[str]         # syllables joining a base >= 20 to a unit
    exclusions: tuple[str, ...]        # tsheg-joined syllable contexts
    fused_particles: tuple[str, ...] = ()    # particles written inside a numeral's last syllable
    multipliers: frozenset[str] = frozenset()  # "X <multiplier> Y" is X * Y (X >= 100)
    fractions: frozenset[str] = frozenset()    # a numeral followed by one of these is not a count


@dataclass(frozen=True)
class SaNumerals:
    stems: Mapping[str, int]           # word-initial IAST stems
    exclusions: tuple[str, ...]        # word-initial forms that are never numerals


@dataclass(frozen=True)
class NumeralTable:
    zh: ZhNumerals
    bo: BoNumerals
    sa: SaNumerals


@dataclass(frozen=True)
class Negators:
    """Negation words per language, with contexts in which a form is not a negator."""

    forms: Mapping[str, tuple[str, ...]]
    exclusions: Mapping[str, tuple[str, ...]]

    def for_lang(self, lang: str) -> tuple[str, ...]:
        return self.forms.get(lang, ())

    def exclusions_for(self, lang: str) -> tuple[str, ...]:
        return self.exclusions.get(lang, ())


@dataclass(frozen=True)
class VariantForms:
    """Per language, variant form -> class representative (folding is idempotent)."""

    forms: Mapping[str, Mapping[str, str]]

    def for_lang(self, lang: str) -> Mapping[str, str]:
        return self.forms.get(lang, MappingProxyType({}))


# --------------------------------------------------------------------------- generic
def lexicon_path(data_dir: Path, name: str) -> Path:
    if not _NAME_RE.fullmatch(name):
        raise ValueError(f"bad lexicon name {name!r}: use lower-case ASCII, digits and '_'")
    return Path(data_dir) / "lexicon" / f"{name}.yaml"


def read_lexicon(data_dir: Path, name: str) -> Mapping[str, Any]:
    """``data/lexicon/<name>.yaml`` as a deeply read-only mapping with NFC strings."""
    path = lexicon_path(data_dir, name)
    doc = read_yaml(path)
    if not isinstance(doc, Mapping):
        raise LexiconError(f"{path}: top level must be a mapping")
    return _freeze(doc)


def _freeze(node: Any) -> Any:
    if isinstance(node, Mapping):
        return MappingProxyType({_freeze(k): _freeze(v) for k, v in node.items()})
    if isinstance(node, (list, tuple)):
        return tuple(_freeze(v) for v in node)
    if isinstance(node, str):
        return unicodedata.normalize("NFC", node)
    return node


# --------------------------------------------------------------------------- numerals
def load_numerals(data_dir: Path) -> NumeralTable:
    doc = read_lexicon(data_dir, "numerals")
    _only(doc, {"zh", "bo", "sa"}, "numerals")
    zh = _section(doc, "zh", {"digits", "units", "exclusions"})
    bo = _section(doc, "bo", {"words", "teen_prefixes", "connectors", "exclusions", "fused_particles",
                              "multipliers", "fractions"})
    sa = _section(doc, "sa", {"stems", "exclusions"})
    table = NumeralTable(
        zh=ZhNumerals(
            digits=_values(zh.get("digits"), "numerals.zh.digits", "zh"),
            units=_values(zh.get("units"), "numerals.zh.units", "zh"),
            exclusions=_strings(zh.get("exclusions", ()), "numerals.zh.exclusions", "zh"),
        ),
        bo=BoNumerals(
            words=_values(bo.get("words"), "numerals.bo.words", "bo"),
            teen_prefixes=frozenset(_strings(bo.get("teen_prefixes", ()), "numerals.bo.teen_prefixes", "bo")),
            connectors=frozenset(_strings(bo.get("connectors", ()), "numerals.bo.connectors", "bo")),
            exclusions=_strings(bo.get("exclusions", ()), "numerals.bo.exclusions", "bo"),
            fused_particles=_strings(bo.get("fused_particles", ()), "numerals.bo.fused_particles", "bo"),
            multipliers=frozenset(_strings(bo.get("multipliers", ()), "numerals.bo.multipliers", "bo")),
            fractions=frozenset(_strings(bo.get("fractions", ()), "numerals.bo.fractions", "bo")),
        ),
        sa=SaNumerals(
            stems=_values(sa.get("stems"), "numerals.sa.stems", "sa"),
            exclusions=_strings(sa.get("exclusions", ()), "numerals.sa.exclusions", "sa"),
        ),
    )
    _check_tokens([*table.zh.digits, *table.zh.units], "zh", "numerals.zh digits/units", single=True)
    _check_tokens(table.zh.exclusions, "zh", "numerals.zh.exclusions")
    _check_tokens(table.bo.words, "bo", "numerals.bo.words")
    _check_tokens([*table.bo.teen_prefixes, *table.bo.connectors, *table.bo.fused_particles,
                   *table.bo.multipliers, *table.bo.fractions], "bo", "numerals.bo single syllables", single=True)
    _check_tokens(table.bo.exclusions, "bo", "numerals.bo.exclusions")
    _check_tokens([*table.sa.stems, *table.sa.exclusions], "sa", "numerals.sa", single=True)
    overlap = set(table.zh.digits) & set(table.zh.units)
    if overlap:
        raise LexiconError(f"numerals.zh: characters listed as both digit and unit: {sorted(overlap)}")
    return table


# --------------------------------------------------------------------------- negators
def load_negators(data_dir: Path) -> Negators:
    doc = read_lexicon(data_dir, "negators")
    _only(doc, {"forms", "exclusions"}, "negators")
    forms = doc.get("forms", {})
    exclusions = doc.get("exclusions", {})
    for key, node in (("forms", forms), ("exclusions", exclusions)):
        if not isinstance(node, Mapping):
            raise LexiconError(f"negators.{key} must map a language code to a list of forms")
    negators = Negators(
        forms=MappingProxyType({lang: _strings(v, f"negators.forms.{lang}", lang) for lang, v in forms.items()}),
        exclusions=MappingProxyType(
            {lang: _strings(v, f"negators.exclusions.{lang}", lang) for lang, v in exclusions.items()}
        ),
    )
    for key, table in (("forms", negators.forms), ("exclusions", negators.exclusions)):
        for lang, items in table.items():
            _check_tokens(items, lang, f"negators.{key}.{lang}")
    return negators


# --------------------------------------------------------------------------- standing renderings
STANDING_STATUSES = frozenset({"proposed", "verified", "retired"})


@dataclass(frozen=True)
class StandingEntry:
    """One verified standing rendering: reference forms and their witness renderings."""

    id: str
    reference_forms: tuple[str, ...]
    witness_renderings: tuple[str, ...]


@dataclass(frozen=True)
class StandingRenderings:
    """The verified standing renderings of one witness against one reference (collate.v2).

    The collator shows exactly these entries to the model and V12 checks claims against
    exactly these, so both must receive the same object. Empty until a researcher verifies
    an entry.
    """

    entries: tuple[StandingEntry, ...] = ()

    @classmethod
    def empty(cls) -> "StandingRenderings":
        return cls()


def load_standing_renderings(path: Path, reference: str, witness: str, reference_lang: str = "bo",
                             witness_lang: str = "zh") -> StandingRenderings:
    """Read a standing-renderings file and return its **verified** entries.

    The file names its reference and witness; a file for another pair is an error, never
    silently empty. Every entry (proposed, verified or retired) and every
    ``considered_not_listed`` record is validated; only verified entries are returned. An
    entry is verified exactly when it names who verified it.
    """
    where = str(path)
    doc = _freeze(read_yaml(path))
    if not isinstance(doc, Mapping):
        raise LexiconError(f"{where}: expected a mapping")
    _only(doc, {"schema_version", "reference", "witness", "entries", "considered_not_listed"}, where)
    if doc.get("schema_version") != 1:
        raise LexiconError(f"{where}: schema_version 1 required")
    if (doc.get("reference"), doc.get("witness")) != (reference, witness):
        raise LexiconError(f"{where}: lists renderings of {doc.get('witness')!r} against {doc.get('reference')!r}, "
                           f"not of {witness!r} against {reference!r}")
    entries, seen = [], set()
    for rec in doc.get("entries") or ():
        entry, status = _standing_entry(rec, where, reference_lang, witness_lang, seen)
        if status == "verified":
            entries.append(entry)
    for rec in doc.get("considered_not_listed") or ():
        if not isinstance(rec, Mapping) or not isinstance(rec.get("id"), str) or not rec.get("reason"):
            raise LexiconError(f"{where}: considered_not_listed records need an id and a reason")
        if rec["id"] in seen:
            raise LexiconError(f"{where}: duplicate id {rec['id']!r}")
        seen.add(rec["id"])
    return StandingRenderings(tuple(entries))


def _standing_entry(rec: Any, where: str, ref_lang: str, wit_lang: str,
                    seen: set[str]) -> tuple[StandingEntry, str]:
    allowed = {"id", "reference_forms", "sanskrit", "witness_renderings", "counts", "note", "status", "verified_by"}
    if not isinstance(rec, Mapping):
        raise LexiconError(f"{where}: an entry must be a mapping")
    _only(rec, allowed, f"{where}: entry {rec.get('id')!r}")
    ident = rec.get("id")
    if not isinstance(ident, str) or not _NAME_RE.fullmatch(ident) or ident in seen:
        raise LexiconError(f"{where}: missing, malformed or duplicate entry id {ident!r}")
    seen.add(ident)
    status, by = rec.get("status"), rec.get("verified_by")
    if status not in STANDING_STATUSES:
        raise LexiconError(f"{where}: entry {ident}: status must be one of {sorted(STANDING_STATUSES)}")
    named = isinstance(by, str) and bool(by.strip())
    if (status == "verified" and not named) or (status != "verified" and by is not None):
        raise LexiconError(f"{where}: entry {ident}: verified_by (the verifier's name, a non-blank string) "
                           "is required for status verified and must be null otherwise")
    refs = _strings(rec.get("reference_forms"), f"{where}: entry {ident} reference_forms", ref_lang)
    wits = _strings(rec.get("witness_renderings"), f"{where}: entry {ident} witness_renderings", wit_lang)
    _check_tokens(refs, ref_lang, f"{where}: entry {ident} reference_forms")
    _check_tokens(wits, wit_lang, f"{where}: entry {ident} witness_renderings")
    if not refs or not wits:
        raise LexiconError(f"{where}: entry {ident} needs reference_forms and witness_renderings")
    return StandingEntry(ident, refs, wits), status


# --------------------------------------------------------------------------- variant forms
def load_variant_forms(data_dir: Path) -> VariantForms:
    doc = read_lexicon(data_dir, "variant_forms")
    by_lang: dict[str, Mapping[str, str]] = {}
    for lang, records in doc.items():
        table: dict[str, str] = {}
        for rec in _records(records, f"variant_forms.{lang}", {"form", "canonical"}):
            where = f"variant_forms.{lang}"
            form, canonical = _norm(rec["form"], lang, where), _norm(rec["canonical"], lang, where)
            if not form or not canonical or form == canonical:
                raise LexiconError(f"variant_forms.{lang}: bad record {dict(rec)}")
            if table.get(form, canonical) != canonical:
                raise LexiconError(f"variant_forms.{lang}: {form!r} has two representatives")
            table[form] = canonical
        chained = sorted(set(table.values()) & set(table))
        if chained:
            raise LexiconError(f"variant_forms.{lang}: representatives also listed as forms: {chained}")
        by_lang[lang] = MappingProxyType(table)
    return VariantForms(forms=MappingProxyType(by_lang))


# --------------------------------------------------------------------------- helpers
def _only(node: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(node) - allowed)
    if unknown:
        raise LexiconError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def _section(doc: Mapping[str, Any], key: str, allowed: set[str]) -> Mapping[str, Any]:
    node = doc.get(key)
    if not isinstance(node, Mapping):
        raise LexiconError(f"numerals.{key} must be a mapping")
    _only(node, allowed, f"numerals.{key}")
    return node


def _norm(text: Any, lang: str, where: str) -> str:
    """NFC (done by ``_freeze``) plus lower case for Latin-script languages."""
    if not isinstance(text, str):
        raise LexiconError(f"{where}: expected a string, got {text!r} (quote YAML words such as no, yes, on)")
    text = text.strip()
    return text if lang in ("zh", "bo") else text.lower()


def _check_tokens(forms: Any, lang: str, where: str, single: bool = False) -> None:
    """Every form must tokenise to one chunk (no punctuation), and to one token if ``single``."""
    for form in forms:
        parts = chunks(form, lang)
        if len(parts) != 1 or (single and len(parts[0]) != 1):
            what = "one token" if single else "a token sequence without punctuation"
            raise LexiconError(f"{where}: {form!r} is not {what}")


def _strings(items: Any, where: str, lang: str) -> tuple[str, ...]:
    """A list of non-empty, distinct strings (normalised by ``_norm``)."""
    if not isinstance(items, tuple):
        raise LexiconError(f"{where}: expected a list of strings")
    out = tuple(_norm(x, lang, where) for x in items)
    if any(not x for x in out) or len(set(out)) != len(out):
        raise LexiconError(f"{where}: empty or duplicate entries")
    return out


def _values(items: Any, where: str, lang: str) -> Mapping[str, int]:
    """A list of ``{form, value}`` records with distinct forms and non-negative int values."""
    table: dict[str, int] = {}
    for rec in _records(items, where, {"form", "value"}):
        form, value = _norm(rec["form"], lang, where), rec["value"]
        if not form or isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise LexiconError(f"{where}: bad record {dict(rec)}")
        if form in table:
            raise LexiconError(f"{where}: duplicate form {form!r}")
        table[form] = value
    return MappingProxyType(table)


def _records(items: Any, where: str, fields: set[str]) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(items, tuple):
        raise LexiconError(f"{where}: expected a list of records with keys {sorted(fields)}")
    for rec in items:
        if not isinstance(rec, Mapping) or set(rec) != fields:
            raise LexiconError(f"{where}: expected a record with keys {sorted(fields)}, got {rec!r}")
    return items
