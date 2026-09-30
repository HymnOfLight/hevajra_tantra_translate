"""Language-aware text normalisation: quote keys, fingerprints, lengths, numerals, terms.

All functions are pure and deterministic. Language-specific *data* (numeral words,
exclusion contexts, variant forms, negators) lives in ``data/lexicon/*.yaml`` and is
passed in by the caller (see ``core.lexicon``); only Unicode-level rules live here.

Tokenisation (shared by every function below)
    Text is NFC-normalised and lower-cased, then cut into *chunks* at punctuation,
    symbols and control characters; a term never matches across a chunk boundary.
    Inside a chunk the tokens are
        zh     single characters (whitespace is dropped),
        bo     syllables (tsheg, non-breaking tsheg and whitespace separate them;
               shad and other Tibetan punctuation end the chunk),
        other  words (runs between whitespace), e.g. IAST Sanskrit or English.
    Matching on whole tokens is what keeps a Tibetan syllable from matching inside a
    longer syllable, and a Sanskrit numeral from matching inside another word.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from functools import lru_cache
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

if TYPE_CHECKING:  # type-only import: core.lexicon itself imports nothing from here
    from .lexicon import BoNumerals, NumeralTable, SaNumerals, ZhNumerals

TSHEG = "\u0f0b"        # Tibetan intersyllabic tsheg
TSHEG_NB = "\u0f0c"     # Tibetan non-breaking tsheg
FINGERPRINT_HEX = 12    # hex digits of sha1 kept in a fingerprint (48 bits)
MIN_NUMERAL = 2         # a bare "one" (and zero) is never reported; see ``numerals``

# Characters that in Tibetan script occur (almost) only in Sanskrit transliteration.
BO_SANSKRIT_MARKERS = frozenset(
    "\u0f71"                                # vowel sign aa (long a)
    "\u0f7e"                                # anusvara
    "\u0f83"                                # candrabindu
    "\u0f7b\u0f7d"                          # vowel signs ai, au
    "\u0f4a\u0f4b\u0f4c\u0f4d\u0f4e"        # retroflex tta, ttha, dda, ddha, nna
    "\u0f65"                                # ssa (retroflex sibilant)
    "\u0f9a\u0f9b\u0f9c\u0f9d\u0f9e"        # subjoined retroflex tta .. nna
    "\u0fb5"                                # subjoined ssa
    "\u0fb7"                                # subjoined ha (aspirates gh, dh, bh, ...)
    "\u0f7f"                                # visarga
)
# The paluta mark (U+0F85, avagraha) is Unicode punctuation, so ``chunks`` treats it as
# a boundary and it can never occur inside a syllable; it is therefore not a marker.

# IAST vowel nuclei: diphthongs first, then a, long a, i, long i, u, long u,
# vocalic r, long vocalic r, vocalic l, long vocalic l, e, o.
_SA_VOWEL_RE = re.compile("ai|au|[a\u0101i\u012bu\u016b\u1e5b\u1e5d\u1e37\u1e39eo]")


# --------------------------------------------------------------------------- tokenisation
def _char_class(ch: str) -> str:
    """"space", "ignore" (zero-width format characters such as ZWSP, ZWJ, BOM, soft
    hyphen), "boundary" (punctuation, symbols, controls) or "text" (letters, marks,
    digits, private-use characters that may encode rare glyphs)."""
    if ch.isspace():
        return "space"
    category = unicodedata.category(ch)
    if category == "Cf":
        return "ignore"
    if category[0] in "PS" or category == "Cc":
        return "boundary"
    return "text"


def chunks(text: str, lang: str) -> list[list[str]]:
    """Token lists between boundaries, as described in the module docstring."""
    text = unicodedata.normalize("NFC", text).lower()
    out: list[list[str]] = []
    tokens: list[str] = []
    buf: list[str] = []

    def end_token() -> None:
        if buf:
            tokens.append("".join(buf))
            buf.clear()

    def end_chunk() -> None:
        end_token()
        if tokens:
            out.append(tokens.copy())
            tokens.clear()

    for ch in text:
        kind = "space" if ch in (TSHEG, TSHEG_NB) else _char_class(ch)
        if kind == "boundary":
            end_chunk()
        elif kind == "space":
            end_token()
        elif kind == "text":
            if lang == "zh":
                tokens.append(ch)
            else:
                buf.append(ch)
    end_chunk()
    return out


@lru_cache(maxsize=4096)
def _term_tokens(term: str, lang: str) -> tuple[str, ...]:
    """A lexicon term as one token sequence; a term must not contain a boundary."""
    parts = chunks(term, lang)
    if len(parts) != 1:
        raise ValueError(f"lexicon term {term!r} ({lang}) is empty or contains punctuation")
    return tuple(parts[0])


def _occurrences(tokens: Sequence[str], term: tuple[str, ...]) -> Iterable[int]:
    n = len(term)
    return (i for i in range(len(tokens) - n + 1) if tuple(tokens[i:i + n]) == term)


def _masked(tokens: Sequence[str], exclusions: Iterable[str], lang: str) -> list[bool]:
    """True for every token covered by an occurrence of an exclusion context."""
    mask = [False] * len(tokens)
    for context in exclusions:
        term = _term_tokens(context, lang)
        for i in _occurrences(tokens, term):
            mask[i:i + len(term)] = [True] * len(term)
    return mask


# --------------------------------------------------------------------------- quotes and keys
def for_quote(text: str, lang: str) -> str:
    """Key for verbatim-quote checks: ``for_quote(q) in for_quote(t)`` means q is quoted from t.

    NFC, lower case, whitespace and punctuation (CJK, ASCII, Tibetan shad and head marks)
    dropped. Tibetan keeps exactly one tsheg between syllables, so tsheg, non-breaking
    tsheg, shad and spacing variants all fold to the same key.
    """
    joiner = TSHEG if lang == "bo" else ""
    return joiner.join(token for chunk in chunks(text, lang) for token in chunk)


def quote_in(quote: str, text: str, lang: str) -> bool:
    """True if ``quote`` occurs verbatim in ``text`` after ``for_quote`` normalisation.

    A Tibetan quote must cover whole syllables (a quote ending inside a syllable is not
    verbatim). An empty quote proves nothing and is never contained.
    """
    q, t = for_quote(quote, lang), for_quote(text, lang)
    if not q:
        return False
    if lang == "bo":
        return f"{TSHEG}{q}{TSHEG}" in f"{TSHEG}{t}{TSHEG}"
    return q in t


def fingerprint(text: str, lang: str) -> str:
    """First 12 hex digits of sha1 over ``for_quote``: changes with the words, not with punctuation."""
    digest = hashlib.sha1(for_quote(text, lang).encode("utf-8")).hexdigest()
    return digest[:FINGERPRINT_HEX]


def fold_variants(text: str, table: Mapping[str, str]) -> str:
    """Replace every listed variant form by its class representative (longest form first).

    Apply it to both strings of a comparison; ``core.lexicon.VariantForms`` guarantees
    that folding twice changes nothing. Matching is case-sensitive and the lexicon
    stores Latin-script forms in lower case, so for a quote check fold the ``for_quote``
    keys: ``fold_variants(for_quote(q, lang), table)``. The result is NFC.
    """
    text = unicodedata.normalize("NFC", text)
    if not table:
        return text
    pattern = _variant_pattern(tuple(sorted(table, key=len, reverse=True)))
    return pattern.sub(lambda m: table[m.group(0)], text)


@lru_cache(maxsize=64)
def _variant_pattern(forms: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile("|".join(re.escape(f) for f in forms))


def find_terms(text: str, lang: str, terms: Iterable[str], exclusions: Iterable[str] = ()) -> set[str]:
    """The ``terms`` that occur in ``text`` as whole token sequences.

    An occurrence overlapping an occurrence of an ``exclusions`` context does not count.
    Used for negators and other closed word lists; matching is case-insensitive.
    """
    exclusions = tuple(exclusions)
    found: set[str] = set()
    for tokens in chunks(text, lang):
        mask = _masked(tokens, exclusions, lang)
        for term in terms:
            seq = _term_tokens(term, lang)
            if any(not any(mask[i:i + len(seq)]) for i in _occurrences(tokens, seq)):
                found.add(term)
    return found


# --------------------------------------------------------------------------- lengths
def bo_syllables(text: str) -> list[str]:
    """Tibetan syllables in order, without tsheg, shad or other punctuation."""
    return [token for chunk in chunks(text, "bo") for token in chunk]


def bo_sanskrit_syllable_ratio(text: str) -> float:
    """Share of syllables containing a Sanskrit-only character (0.0 for empty text)."""
    syllables = bo_syllables(text)
    if not syllables:
        return 0.0
    return sum(1 for s in syllables if BO_SANSKRIT_MARKERS.intersection(s)) / len(syllables)


def length(text: str, lang: str) -> int:
    """Language-appropriate length.

    zh: characters without punctuation; bo: syllables; sa: vowel nuclei of the IAST
    text (a syllable count); any other language: words.
    """
    if lang == "sa":
        return len(_SA_VOWEL_RE.findall(unicodedata.normalize("NFC", text).lower()))
    return sum(len(chunk) for chunk in chunks(text, lang))


# --------------------------------------------------------------------------- numerals
def numerals(text: str, lang: str, table: NumeralTable) -> set[int]:
    """Numbers written as numeral words in ``text`` (zh, bo and sa; empty for other languages).

    Matching respects token boundaries, and a numeral inside an exclusion context of the
    table (e.g. the Tibetan word for emptiness, whose first syllable also means 1000) is
    ignored. Values below ``MIN_NUMERAL`` are dropped: a bare "one" is usually an
    article or quantifier ("all", "once", "together"), not a count, while 11, 21, 100
    etc. are kept.
    """
    if lang == "zh":
        found = _zh_numerals(text, table.zh)
    elif lang == "bo":
        found = _bo_numerals(text, table.bo)
    elif lang == "sa":
        found = _sa_numerals(text, table.sa)
    else:
        return set()
    return {v for v in found if v >= MIN_NUMERAL}


def zh_parse_numeral(run: str, table: ZhNumerals) -> int | None:
    """Value of a Chinese numeral string, e.g. 32 for three-ten-two, 108 for one-hundred-eight.

    Returns None if a character is not a numeral. A unit below 10,000 multiplies the digit
    before it (1 when there is none); 10,000 and above multiply everything accumulated
    since the previous such unit (1 when nothing is).
    """
    if not run or any(ch not in table.digits and ch not in table.units for ch in run):
        return None
    total = section = digit = 0
    for ch in run:
        if ch in table.digits:
            digit = table.digits[ch]
            continue
        unit = table.units[ch]
        if unit >= 10000:
            total += ((section + digit) or 1) * unit
            section = 0
        else:
            section += (digit or 1) * unit
        digit = 0
    return total + section + digit


def _zh_numerals(text: str, table: ZhNumerals) -> set[int]:
    """Maximal runs of unmasked numeral characters inside a chunk, each parsed as one number."""
    out: set[int] = set()
    for tokens in chunks(text, "zh"):
        mask = _masked(tokens, table.exclusions, "zh")
        run: list[str] = []
        for ch, masked in zip([*tokens, ""], [*mask, True]):   # the sentinel flushes the last run
            if not masked and (ch in table.digits or ch in table.units):
                run.append(ch)
                continue
            value = zh_parse_numeral("".join(run), table)
            if value is not None:
                out.add(value)
            run = []
    return out


def _bo_numerals(text: str, table: BoNumerals) -> set[int]:
    """Syllable-level parser for numeral words, teens and base + connector + unit forms.

    Covered forms: a numeral word from the table (one or more syllables, longest first);
    a teen prefix followed by a unit (15 as "ten, five"); a base of 20 or more followed
    by a connector syllable and a unit (32 as "thirty, connector, two"; 108 as
    "hundred, connector, eight"). Masked syllables never take part in a numeral.
    """
    words = {_term_tokens(w, "bo"): v for w, v in table.words.items()}
    out: set[int] = set()
    for syl in chunks(text, "bo"):
        mask = _masked(syl, table.exclusions, "bo")
        i = 0
        while i < len(syl):
            value, i = _bo_numeral_at(syl, mask, i, words, table)
            if value is not None:
                out.add(value)
    return out


def _bo_numeral_at(syl: Sequence[str], mask: Sequence[bool], i: int,
                   words: Mapping[tuple[str, ...], int], table: BoNumerals) -> tuple[int | None, int]:
    """(value or None, index after it) for a numeral starting at syllable ``i``."""
    if mask[i]:
        return None, i + 1
    if syl[i] in table.teen_prefixes:
        unit = _bo_word_at(syl, mask, i + 1, words)
        if unit is not None and unit[0] < 10:
            return 10 + unit[0], i + 1 + unit[1]
        return 10, i + 1
    hit = _bo_word_at(syl, mask, i, words)
    if hit is None:
        return None, i + 1
    value, j = hit[0], i + hit[1]
    if value >= 20 and j < len(syl) and not mask[j] and syl[j] in table.connectors:
        unit = _bo_word_at(syl, mask, j + 1, words)
        if unit is not None and unit[0] < 10:
            return value + unit[0], j + 1 + unit[1]
    return value, j


def _bo_word_at(syl: Sequence[str], mask: Sequence[bool], i: int,
                words: Mapping[tuple[str, ...], int]) -> tuple[int, int] | None:
    """(value, syllable count) of the longest unmasked numeral word starting at ``i``."""
    longest = max((len(w) for w in words), default=0)
    for n in range(min(longest, len(syl) - i), 0, -1):
        seq = tuple(syl[i:i + n])
        if seq in words and not any(mask[i:i + n]):
            return words[seq], n
    return None


def _sa_numerals(text: str, table: SaNumerals) -> set[int]:
    """A word counts when it begins with a numeral stem (longest stem wins) and does not
    begin with an exclusion form.

    Stems are matched only at the start of a word, so numerals inside compounds are
    missed on purpose: matching anywhere inside a word turned 'laksana' (mark) into
    100000 and 'advitiya' (non-dual) into 2 in v0.2.
    """
    stems = sorted(table.stems, key=len, reverse=True)
    out: set[int] = set()
    for tokens in chunks(text, "sa"):
        for word in tokens:
            if any(word.startswith(x) for x in table.exclusions):
                continue
            stem = next((s for s in stems if word.startswith(s)), None)
            if stem is not None:
                out.add(table.stems[stem])
    return out
