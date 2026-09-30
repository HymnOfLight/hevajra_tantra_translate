"""Load witness and chapter-concordance registries (data/*.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

STATISTICAL_LAYERS = {"primary", "edition", "indirect"}


@dataclass
class Witness:
    id: str
    lang: str
    layer: str
    source_lang: str | None
    role: str
    independence: str = "unknown"
    status: str = "to_verify"
    title: str = ""
    date: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def counts_in_statistics(self) -> bool:
        return self.layer in STATISTICAL_LAYERS

    @property
    def is_sanskrit(self) -> bool:
        return self.lang == "sa" and self.layer in {"edition", "primary"}


def load_witnesses(path: Path | None = None) -> dict[str, Witness]:
    path = path or DATA_DIR / "witnesses.yaml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    out: dict[str, Witness] = {}
    for w in doc["witnesses"]:
        out[w["id"]] = Witness(
            id=w["id"],
            lang=w["lang"],
            layer=w["layer"],
            source_lang=w.get("source_lang"),
            role=w.get("role", ""),
            independence=w.get("independence", "unknown"),
            status=w.get("status", "to_verify"),
            title=w.get("title", ""),
            date=str(w.get("date", "")),
            raw=w,
        )
    return out


@dataclass
class ChapterMap:
    ref: str
    sa_title: str
    bo: dict[str, Any]
    zh: dict[str, Any]

    def witness_chapter(self, lang: str) -> int | None:
        d = self.bo if lang == "bo" else self.zh if lang == "zh" else {}
        key = "chapter" if lang == "bo" else "pin"
        return d.get(key)


def load_chapter_concordance(path: Path | None = None) -> list[ChapterMap]:
    path = path or DATA_DIR / "concordance" / "chapters.yaml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        ChapterMap(ref=c["ref"], sa_title=c.get("sa_title", ""), bo=c.get("bo", {}), zh=c.get("zh", {}))
        for c in doc["chapters"]
    ]


def chapter_lookup(concordance: list[ChapterMap], lang: str) -> dict[int, str]:
    """Map a witness-side chapter number to a reference chapter key.

    For Tibetan the number is per-Toh (1–11 → I.*, 1–12 → II.*), so callers
    must pass the part-qualified key themselves; here we return the mapping
    keyed by (part, number) encoded as part*100+number for bo, or by pin for zh.
    """
    out: dict[int, str] = {}
    for c in concordance:
        n = c.witness_chapter(lang)
        if n is None:
            continue
        if lang == "bo":
            part = 1 if c.ref.startswith("I.") else 2
            out[part * 100 + n] = c.ref
        else:
            out[n] = c.ref
    return out
