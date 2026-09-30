"""Cross-lingual anchor extraction (terms, names, mantras, numerals).

Anchors are the B0 baseline's only cross-lingual signal: a segment in any
language is reduced to a bag of language-neutral anchor keys such as
``term:vajragarbha`` or ``num:32``. Similarity between bags is what the
aligner uses in addition to length.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import normalize
from .registry import DATA_DIR


@dataclass
class Term:
    key: str
    type: str
    forms: dict[str, list[str]]
    zh_transliteration: bool = False


@dataclass
class AnchorLexicon:
    terms: list[Term]
    zh_translit_chars: set[str] = field(default_factory=set)
    _index: dict[str, list[tuple[str, Term]]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for t in self.terms:
            for lang, forms in t.forms.items():
                # longest forms first so that 大悲空智金剛 wins over 金剛
                for f in sorted(forms, key=len, reverse=True):
                    self._index.setdefault(lang, []).append((f, t))
        for lang in self._index:
            self._index[lang].sort(key=lambda x: len(x[0]), reverse=True)

    # ------------------------------------------------------------------ extraction
    def extract(self, text: str, lang: str) -> set[str]:
        """Return anchor keys for a segment: term:*, mantra:*, name:*, num:*."""
        keys: set[str] = set()
        t = text if lang != "sa" else normalize.sa_normalize(text)
        consumed = [False] * len(t)
        for form, term in self._index.get(lang, []):
            start = 0
            while True:
                i = t.find(form, start)
                if i < 0:
                    break
                if not any(consumed[i:i + len(form)]):
                    keys.add(f"{term.type}:{term.key}")
                    for j in range(i, i + len(form)):
                        consumed[j] = True
                start = i + len(form)
        for n in normalize.numerals(text, lang):
            keys.add(f"num:{n}")
        return keys

    def zh_transliteration_density(self, text: str) -> float:
        """Share of Chinese characters that belong to transliterated stretches.

        A character counts as transliteration if it lies inside a matched
        ``zh_transliteration: true`` term or if it belongs to the
        transliteration character set and sits in a run of ≥2 such characters.
        """
        s = normalize.zh_strip(text)
        if not s:
            return 0.0
        flags = [False] * len(s)
        for form, term in self._index.get("zh", []):
            if not term.zh_transliteration:
                continue
            start = 0
            while True:
                i = s.find(form, start)
                if i < 0:
                    break
                for j in range(i, i + len(form)):
                    flags[j] = True
                start = i + len(form)
        run: list[int] = []
        for i, ch in enumerate(s + "\0"):
            if ch in self.zh_translit_chars:
                run.append(i)
            else:
                if len(run) >= 2:
                    for j in run:
                        flags[j] = True
                run = []
        return sum(flags) / len(s)

    def transliteration_density(self, text: str, lang: str) -> float:
        if lang == "zh":
            return self.zh_transliteration_density(text)
        if lang == "bo":
            return normalize.bo_sanskrit_syllable_ratio(text)
        return 0.0


def load_lexicon(path: Path | None = None) -> AnchorLexicon:
    path = path or DATA_DIR / "anchors" / "terms.yaml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    terms = []
    for t in doc["terms"]:
        forms = {"sa": [normalize.sa_normalize(t["sa"].split(" ")[0])]}
        if "bo" in t:
            forms["bo"] = list(t["bo"])
        if "zh" in t:
            forms["zh"] = list(t["zh"])
        terms.append(Term(key=t["sa"].split(" ")[0], type=t.get("type", "term"), forms=forms,
                          zh_transliteration=bool(t.get("zh_transliteration", False))))
    chars = set("".join(doc.get("zh_transliteration_chars", "").split()))
    return AnchorLexicon(terms=terms, zh_translit_chars=chars)


def anchor_similarity(a: set[str], b: set[str]) -> float:
    """Jaccard on anchor bags; numerals weigh double because they are rarer and sharper."""
    if not a and not b:
        return 0.0

    def w(k: str) -> float:
        return 2.0 if k.startswith("num:") else 1.0

    inter = sum(w(k) for k in a & b)
    union = sum(w(k) for k in a | b)
    return inter / union if union else 0.0
