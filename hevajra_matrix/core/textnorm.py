"""Language-specific normalisation: lengths, numerals, transliteration cues.

Everything here is deliberately simple and auditable; no model calls.
"""

from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------- Chinese
ZH_PUNCT = set("，。、；：？！「」『』（）〔〕【】《》〈〉—…‧·．,.;:?!()[]<>\"' \u3000\n\t")

ZH_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "兩": 2, "两": 2, "三": 3, "四": 4, "五": 5,
             "六": 6, "七": 7, "八": 8, "九": 9}
ZH_UNITS = {"十": 10, "百": 100, "千": 1000, "萬": 10000, "万": 10000}
_ZH_NUM_RE = re.compile("[零〇一二兩两三四五六七八九十百千萬万]+")


def zh_strip(text: str) -> str:
    return "".join(ch for ch in text if ch not in ZH_PUNCT)


def zh_length(text: str) -> int:
    return len(zh_strip(text))


def zh_parse_numeral(s: str) -> int | None:
    """Parse a Chinese numeral like 三十二 → 32. Returns None for non-numerals."""
    if not s or not all(ch in ZH_DIGITS or ch in ZH_UNITS for ch in s):
        return None
    total, section, num = 0, 0, 0
    for ch in s:
        if ch in ZH_DIGITS:
            num = ZH_DIGITS[ch]
        else:
            unit = ZH_UNITS[ch]
            if unit >= 10000:
                section = (section + (num if num else 0)) * unit
                total += section
                section, num = 0, 0
            else:
                section += (num if num else 1) * unit
                num = 0
    return total + section + num


def zh_numerals(text: str) -> set[int]:
    out: set[int] = set()
    for m in _ZH_NUM_RE.finditer(text):
        v = zh_parse_numeral(m.group(0))
        # single "一" is too frequent as a non-numeral ("一切", "一時"), keep >= 2 or multi-char
        if v is not None and (v >= 2 or len(m.group(0)) > 1):
            out.add(v)
    return out


# --------------------------------------------------------------------------- Tibetan
TSHEG = "\u0f0b"
SHAD_CHARS = "\u0f0d\u0f0e\u0f0f\u0f10\u0f11\u0f14"
_BO_SYL_SPLIT = re.compile(r"[\u0f0b\u0f0c\s" + SHAD_CHARS + r"]+")

# Characters that in canonical Tibetan appear (almost) only in Sanskrit transliteration.
BO_SANSKRIT_MARKERS = set(
    "\u0f71"  # ཱ  long a
    "\u0f7e"  # ཾ  anusvara
    "\u0f83"  # ྃ  candrabindu
    "\u0f7b\u0f7d"  # ཻ ཽ  ai au
    "\u0f4a\u0f4b\u0f4c\u0f4d\u0f4e"  # ཊ ཋ ཌ ཌྷ ཎ retroflex
    "\u0f65"  # ཥ  ṣa
    "\u0f9a\u0f9b\u0f9c\u0f9d\u0f9e"  # subjoined retroflex
    "\u0fb5"  # subjoined ṣa
    "\u0fb7"  # ྷ subjoined ha (gh, dh, bh, ...)
    "\u0f85"  # ྅ avagraha
    "\u0f7f"  # ཿ visarga
)

BO_NUMERALS = {
    "གཅིག": 1, "གཉིས": 2, "གསུམ": 3, "བཞི": 4, "ལྔ": 5, "དྲུག": 6, "བདུན": 7, "བརྒྱད": 8, "དགུ": 9,
    "བཅུ": 10, "བཅོ": 10, "ཉི་ཤུ": 20, "སུམ་ཅུ": 30, "བཞི་བཅུ": 40, "ལྔ་བཅུ": 50, "དྲུག་ཅུ": 60,
    "བདུན་ཅུ": 70, "བརྒྱད་ཅུ": 80, "དགུ་བཅུ": 90, "བརྒྱ": 100, "སྟོང": 1000, "ཁྲི": 10000, "འབུམ": 100000,
}
_BO_TENS_CONNECTOR = {"རྩ", "རྩེ", "ར", "སོ", "ཞེ", "ང", "རེ", "དོན", "གྱ", "གོ"}


def bo_syllables(text: str) -> list[str]:
    return [s for s in _BO_SYL_SPLIT.split(text) if s]


def bo_length(text: str) -> int:
    return len(bo_syllables(text))


def bo_sanskrit_syllable_ratio(text: str) -> float:
    syls = bo_syllables(text)
    if not syls:
        return 0.0
    n = sum(1 for s in syls if any(ch in BO_SANSKRIT_MARKERS for ch in s))
    return n / len(syls)


def bo_numerals(text: str) -> set[int]:
    """Very small Tibetan numeral parser covering the forms that occur in the tantra.

    Handles: units, teens (བཅུ་གཅིག / བཅོ་ལྔ / བཅོ་བརྒྱད), tens with connector
    (སུམ་ཅུ་རྩ་གཉིས = 32), and hundreds/thousands as isolated anchors.
    """
    syls = bo_syllables(text)
    out: set[int] = set()
    i = 0
    while i < len(syls):
        two = TSHEG.join(syls[i:i + 2])
        one = syls[i]
        if two in BO_NUMERALS and BO_NUMERALS[two] >= 20:
            base = BO_NUMERALS[two]
            j = i + 2
            if j < len(syls) and syls[j] in _BO_TENS_CONNECTOR and j + 1 < len(syls) and syls[j + 1] in BO_NUMERALS and BO_NUMERALS[syls[j + 1]] < 10:
                out.add(base + BO_NUMERALS[syls[j + 1]])
                i = j + 2
                continue
            out.add(base)
            i += 2
            continue
        if one in ("བཅུ", "བཅོ"):
            if i + 1 < len(syls) and syls[i + 1] in BO_NUMERALS and BO_NUMERALS[syls[i + 1]] < 10:
                out.add(10 + BO_NUMERALS[syls[i + 1]])
                i += 2
                continue
            out.add(10)
            i += 1
            continue
        if one in BO_NUMERALS and BO_NUMERALS[one] != 1:
            out.add(BO_NUMERALS[one])
        i += 1
    return out


# --------------------------------------------------------------------------- Sanskrit (IAST)
_IAST_VOWEL_RE = re.compile(r"ai|au|[aāiīuūṛṝḷḹeo]")

# Stems rather than full forms so that sandhi variants match (dvātriṃśat / dvātriṃśan …).
# Longest stem wins and consumes its span, so "tri" inside "dvātriṃśa" is not double-counted.
SA_NUMERALS = {
    "dvi": 2, "dvau": 2, "tri": 3, "traya": 3, "catur": 4, "catvāri": 4, "catuḥ": 4, "pañca": 5,
    "ṣaṣ": 6, "ṣaṭ": 6, "ṣaḍ": 6, "sapta": 7, "aṣṭa": 8, "aṣṭau": 8, "nava": 9, "daśa": 10,
    "ekādaśa": 11, "dvādaśa": 12, "ṣoḍaśa": 16, "viṃśati": 20, "caturviṃśati": 24, "catuḥviṃśati": 24,
    "dvātriṃśa": 32, "catuḥṣaṣṭi": 64, "śata": 100, "sahasra": 1000, "lakṣa": 100000,
}
_SA_NUM_KEYS = sorted(SA_NUMERALS, key=len, reverse=True)


def sa_normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text.strip().lower())
    text = re.sub(r"[|।॥\d\.\-\(\)\[\]]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sa_length(text: str) -> int:
    """Syllable count ≈ number of vowel nuclei."""
    return len(_IAST_VOWEL_RE.findall(sa_normalize(text)))


def sa_numerals(text: str) -> set[int]:
    t = sa_normalize(text)
    consumed = [False] * len(t)
    out: set[int] = set()
    for k in _SA_NUM_KEYS:
        start = 0
        while True:
            i = t.find(k, start)
            if i < 0:
                break
            if not any(consumed[i:i + len(k)]):
                out.add(SA_NUMERALS[k])
                for j in range(i, i + len(k)):
                    consumed[j] = True
            start = i + len(k)
    return out


# --------------------------------------------------------------------------- dispatch
def length(text: str, lang: str) -> int:
    if lang == "zh":
        return zh_length(text)
    if lang == "bo":
        return bo_length(text)
    if lang == "sa":
        return sa_length(text)
    return len(text.split())


def numerals(text: str, lang: str) -> set[int]:
    if lang == "zh":
        return zh_numerals(text)
    if lang == "bo":
        return bo_numerals(text)
    if lang == "sa":
        return sa_numerals(text)
    return set()
