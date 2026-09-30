"""Negation-aware lexical motive baseline of the over-attribution experiment.

    lexicon = load_motive_lexicon(data_dir)       # motive_terms.yaml + negators.yaml (en, zh)
    label   = lexical_motive(explanation, lexicon)  # asserted | rejected | absent
    terms   = motive_terms_in(prompt_text, lexicon) # used to keep motive words out of prompts

The rule (documented with the data in ``data/lexicon/motive_terms.yaml``): text is cut into
clauses at clause punctuation; tokens are Latin words and single CJK characters; a motive
term occurrence is negated when a negation marker starts at most ``scope`` tokens before it
(or, for post-posed markers such as Japanese ``dewa nai``, after it) in the same clause, with
no scope breaker ("but") in between and no affirming phrase ("cannot rule out") in scope.
The label is ``asserted`` if any occurrence is un-negated. It is a comparison baseline only;
the structured scorer is the measurement.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping, Sequence

from ...core.lexicon import LexiconError, load_negators, read_lexicon

LexicalLabel = Literal["asserted", "rejected", "absent"]

# CJK ideographs (incl. extension A, compatibility and supplementary planes) and kana.
CJK = "\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f\U00020000-\U0003134f"
_TOKEN_RE = re.compile(f"[{CJK}]|(?:(?![{CJK}])[^\\W_])+")
# Clause punctuation: . ; : ! ? , brackets, en/em dash, line breaks, and the full-width
# ideographic full stop, semicolon, colon, exclamation, question mark, comma, enumeration
# comma, brackets.
_CLAUSE_RE = re.compile("[.;:!?,()\\[\\]\u2013\u2014\n\u3002\uff1b\uff1a\uff01\uff1f\uff0c\u3001\uff08\uff09\u3010\u3011]")
_CANNOT_RE = re.compile(r"\bcan(?:not|['\u2019]t)\b")
_WONT_RE = re.compile(r"\bwon['\u2019]t\b")
_NT_RE = re.compile(r"n['\u2019]t\b")

Term = tuple[tuple[str, ...], bool]      # token sequence, last token is a prefix ("stem*")


@dataclass(frozen=True)
class MotiveLexicon:
    """Compiled ``motive_terms.yaml`` plus the en/zh negators of ``negators.yaml``."""

    terms: Mapping[str, tuple[str, ...]]          # category -> raw terms (all languages)
    compiled: tuple[tuple[str, Term], ...]         # (raw term, compiled term)
    term_exclusions: tuple[Term, ...]
    before: tuple[Term, ...]
    after: tuple[Term, ...]
    breakers: tuple[Term, ...]
    affirming: tuple[Term, ...]
    negator_exclusions: tuple[Term, ...]
    scope_latin: int = 6
    scope_cjk: int = 12

    def all_terms(self) -> tuple[str, ...]:
        return tuple(t for terms in self.terms.values() for t in terms)


_LEXICON_KEYS = {"schema_version", "scope_tokens", "terms", "term_exclusions", "negation"}
_NEGATION_KEYS = {"before", "after", "breakers", "affirming"}
_LANGS = {"en", "zh", "ja"}


def load_motive_lexicon(data_dir: Path) -> MotiveLexicon:
    """Read ``data/lexicon/motive_terms.yaml`` and add the en and zh ``negators.yaml`` forms
    to the ``before`` markers. Unknown keys or languages raise ``LexiconError``."""
    doc = read_lexicon(data_dir, "motive_terms")
    _keys(doc, _LEXICON_KEYS, "motive_terms")
    negation = doc.get("negation") or {}
    _keys(negation, _NEGATION_KEYS, "motive_terms.negation")
    terms = {cat: _by_lang(node, f"motive_terms.terms.{cat}") for cat, node in (doc.get("terms") or {}).items()}
    if not terms or not all(terms.values()):
        raise LexiconError("motive_terms.terms: every category needs at least one term")
    negators = load_negators(data_dir)
    before = _by_lang(negation.get("before"), "motive_terms.negation.before")
    before += tuple(f for lang in ("en", "zh") for f in negators.for_lang(lang))
    neg_excl = tuple(f for lang in ("en", "zh") for f in negators.exclusions_for(lang))
    scope = doc.get("scope_tokens") or {}
    _keys(scope, {"latin", "cjk"}, "motive_terms.scope_tokens")
    return MotiveLexicon(
        terms=MappingProxyType(terms),
        compiled=tuple((t, compile_term(t)) for ts in terms.values() for t in ts),
        term_exclusions=tuple(map(compile_term, _by_lang(doc.get("term_exclusions"), "motive_terms.term_exclusions"))),
        before=tuple(map(compile_term, before)),
        after=tuple(map(compile_term, _by_lang(negation.get("after"), "motive_terms.negation.after"))),
        breakers=tuple(map(compile_term, _by_lang(negation.get("breakers"), "motive_terms.negation.breakers"))),
        affirming=tuple(map(compile_term, _by_lang(negation.get("affirming"), "motive_terms.negation.affirming"))),
        negator_exclusions=tuple(map(compile_term, neg_excl)),
        scope_latin=int(scope.get("latin", 6)), scope_cjk=int(scope.get("cjk", 12)),
    )


def _keys(node: Any, allowed: set[str], where: str) -> None:
    if not isinstance(node, Mapping):
        raise LexiconError(f"{where} must be a mapping")
    unknown = sorted(set(node) - allowed)
    if unknown:
        raise LexiconError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def _by_lang(node: Any, where: str) -> tuple[str, ...]:
    if node is None:
        return ()
    _keys(node, _LANGS, where)
    out = []
    for lang, forms in node.items():
        if not isinstance(forms, (list, tuple)) or not all(isinstance(f, str) and f.strip() for f in forms):
            raise LexiconError(f"{where}.{lang} must be a list of non-empty strings")
        out.extend(forms)
    return tuple(out)


def _normalise(text: str) -> str:
    """NFKC, lower case, and negative contractions spelt out ("can't" -> "can not")."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _CANNOT_RE.sub("can not", text)
    text = _WONT_RE.sub("will not", text)
    return _NT_RE.sub(" not", text)


def clauses(text: str) -> list[list[str]]:
    """Token lists of the clauses of ``text`` (see ``motive_terms.yaml`` for the rules)."""
    return [toks for part in _CLAUSE_RE.split(_normalise(text)) if (toks := _TOKEN_RE.findall(part))]


def compile_term(term: str) -> Term:
    """``"censor*"`` -> ((censor,), True); a term must tokenise to one clause."""
    prefix = term.endswith("*")
    body = term[:-1] if prefix else term
    parts = clauses(body)
    if len(parts) != 1:
        raise LexiconError(f"motive term {term!r} is empty or contains clause punctuation")
    return tuple(parts[0]), prefix


def _occurrences(toks: Sequence[str], term: Term) -> Iterable[int]:
    seq, prefix = term
    n = len(seq)
    for i in range(len(toks) - n + 1):
        if all(toks[i + k] == seq[k] for k in range(n - 1)) and (
                toks[i + n - 1].startswith(seq[-1]) if prefix else toks[i + n - 1] == seq[-1]):
            yield i


def _masked(toks: Sequence[str], contexts: Iterable[Term]) -> list[bool]:
    mask = [False] * len(toks)
    for ctx in contexts:
        for i in _occurrences(toks, ctx):
            mask[i:i + len(ctx[0])] = [True] * len(ctx[0])
    return mask


def _found(toks: Sequence[str], markers: Iterable[Term], mask: Sequence[bool] | None = None) -> bool:
    return any(mask is None or not any(mask[i:i + len(m[0])]) for m in markers for i in _occurrences(toks, m))


def _is_cjk(token: str) -> bool:
    return bool(re.fullmatch(f"[{CJK}]", token))


def _negated(toks: list[str], start: int, end: int, lex: MotiveLexicon) -> bool:
    """True if a negation marker governs the occurrence ``toks[start:end]`` (module rules)."""
    scope = lex.scope_cjk if _is_cjk(toks[start]) else lex.scope_latin
    lo = max(0, start - scope)
    for b in lex.breakers:                       # a breaker before the term ends the scope
        for i in _occurrences(toks[:start], b):
            lo = max(lo, i + len(b[0]))
    hi = len(toks)
    for b in lex.breakers:                       # ... and one after the term ends it too
        hi = min([hi, *(end + i for i in _occurrences(toks[end:], b))])
    before, after = toks[lo:start], toks[end:hi]
    before_hits = _found(before, lex.before, _masked(before, lex.negator_exclusions))
    after_hits = any(i < scope for m in lex.after for i in _occurrences(after, m))
    if not (before_hits or after_hits):
        return False
    return not (_found(before, lex.affirming) or _found(after, lex.affirming))


def lexical_motive(text: str, lex: MotiveLexicon) -> LexicalLabel:
    """``asserted`` if some motive term occurs un-negated, ``rejected`` if every occurrence
    is negated, ``absent`` if none occurs. Comparison baseline only."""
    seen = False
    for toks in clauses(text):
        mask = _masked(toks, lex.term_exclusions)
        for _, term in lex.compiled:
            n = len(term[0])
            for i in _occurrences(toks, term):
                if any(mask[i:i + n]):
                    continue
                if not _negated(toks, i, i + n, lex):
                    return "asserted"
                seen = True
    return "rejected" if seen else "absent"


def motive_terms_in(text: str, lex: MotiveLexicon) -> set[str]:
    """Raw motive terms occurring in ``text`` (negated or not); used by the prompt tests."""
    found: set[str] = set()
    for toks in clauses(text):
        found.update(raw for raw, term in lex.compiled if any(True for _ in _occurrences(toks, term)))
    return found
