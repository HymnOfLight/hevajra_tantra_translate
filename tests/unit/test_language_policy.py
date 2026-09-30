"""Repository language policy, enforced mechanically.

    * ``.py`` files: no non-Latin script anywhere (comments, docstrings and literals).
      Script constants are written as ``\\u`` escapes or live in ``data/lexicon/``.
    * Markdown outside ``docs-zh/`` (including prompt templates): no non-Latin script.
    * ``config/*.yaml``: no non-Latin script at all.
    * YAML under ``data/``: keys and comments are English; values may be any language.
    * File and directory names are ASCII.
    * ``data/annotations/`` holds only ids and short quotes (licence rule).

IAST transliteration is Latin script and therefore allowed everywhere.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {".git", "runs", "raw", "__pycache__", ".pytest_cache", "build", "dist"}

# CJK (radicals .. unified ideographs, symbols, kana), compatibility ideographs and forms,
# fullwidth forms, supplementary ideographs, Tibetan, Devanagari, Mongolian, Tangut.
NON_LATIN = re.compile(
    "[\u2e80-\u9fff\uf900-\ufaff\ufe30-\ufe4f\uff00-\uffef"
    "\U00020000-\U0003134f\u0f00-\u0fff\u0900-\u097f\u1800-\u18af\U00017000-\U00018aff]"
)
LONG_NON_LATIN_RUN = re.compile(NON_LATIN.pattern + "{61,}")


def _files(pattern: str, under: Path = ROOT) -> list[Path]:
    return [p for p in under.rglob(pattern) if not (set(p.relative_to(ROOT).parts) & SKIP_DIRS)]


def _first_hit(text: str) -> str | None:
    m = NON_LATIN.search(text)
    if m is None:
        return None
    line = text.count("\n", 0, m.start()) + 1
    return f"line {line}: {text.splitlines()[line - 1].strip()[:60]!r}"


def _yaml_comment(line: str) -> str:
    """Return the comment part of a YAML line, ignoring '#' inside quotes."""
    quote: str | None = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1].isspace()):
            return line[i:]
    return ""


def _yaml_keys(node: object) -> list[str]:
    if isinstance(node, dict):
        return [str(k) for k in node] + [k for v in node.values() for k in _yaml_keys(v)]
    if isinstance(node, list):
        return [k for v in node for k in _yaml_keys(v)]
    return []


@pytest.mark.parametrize("path", _files("*.py"), ids=lambda p: str(p.relative_to(ROOT)))
def test_python_files_are_english_only(path: Path) -> None:
    hit = _first_hit(path.read_text(encoding="utf-8"))
    assert hit is None, f"non-Latin script in {path.relative_to(ROOT)} at {hit}"


def test_markdown_outside_docs_zh_is_english_only() -> None:
    bad = []
    for path in _files("*.md"):
        rel = path.relative_to(ROOT)
        if rel.parts[0] == "docs-zh":
            continue
        hit = _first_hit(path.read_text(encoding="utf-8"))
        if hit:
            bad.append(f"{rel} {hit}")
    assert not bad, "\n".join(bad)


def test_config_yaml_is_english_only() -> None:
    bad = []
    for path in _files("*.yaml", ROOT / "config"):
        hit = _first_hit(path.read_text(encoding="utf-8"))
        if hit:
            bad.append(f"{path.relative_to(ROOT)} {hit}")
    assert not bad, "\n".join(bad)


def test_data_yaml_keys_and_comments_are_english() -> None:
    bad = []
    for path in _files("*.yaml", ROOT / "data"):
        text = path.read_text(encoding="utf-8")
        for n, line in enumerate(text.splitlines(), start=1):
            if NON_LATIN.search(_yaml_comment(line)):
                bad.append(f"{path.relative_to(ROOT)}:{n} comment")
        for key in _yaml_keys(yaml.safe_load(text)):
            if NON_LATIN.search(key):
                bad.append(f"{path.relative_to(ROOT)} key {key!r}")
    assert not bad, "\n".join(bad[:50])


def test_file_and_directory_names_are_ascii() -> None:
    bad = [str(p.relative_to(ROOT)) for p in _files("*") if not p.name.isascii()]
    assert not bad, "\n".join(bad)


def test_annotations_hold_only_short_quotes() -> None:
    annotations = ROOT / "data" / "annotations"
    bad = []
    for path in _files("*", annotations):
        if path.is_file() and path.suffix in {".csv", ".yaml", ".jsonl", ".json", ".tsv"}:
            if LONG_NON_LATIN_RUN.search(path.read_text(encoding="utf-8")):
                bad.append(str(path.relative_to(ROOT)))
    assert not bad, "long source-text runs found (licence rule): " + ", ".join(bad)
