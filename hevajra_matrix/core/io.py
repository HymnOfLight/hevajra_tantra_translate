"""File I/O helpers: the only place in ``core`` that touches the file system.

Conventions shared by every stage
    * UTF-8 everywhere. CSV readers accept a byte-order mark (spreadsheets add one);
      ``write_csv(..., bom=True)`` writes one so that Excel shows Chinese and Tibetan
      correctly in review sheets. Machine files are written without a BOM.
    * Whole-file writers write to a temporary sibling and rename it into place, so an
      interrupted run never leaves a half-written file behind.
    * JSON is written with ``ensure_ascii=False`` (source-language text stays readable).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import yaml

if TYPE_CHECKING:
    from ..config import Settings

MANIFEST_NAME = "manifest.json"


def read_yaml(path: Path) -> Any:
    """Parsed YAML document (``None`` for an empty file); only safe YAML is accepted."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    """Rows as dicts keyed by the header; a leading byte-order mark is ignored."""
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str], bom: bool = False) -> None:
    """Write ``rows`` with exactly ``columns``; a key outside ``columns`` is an error.

    Missing keys and ``None`` values are written as empty cells.
    """
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(columns), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: "" if v is None else v for k, v in row.items()})
    _atomic_write(Path(path), buf.getvalue(), encoding="utf-8-sig" if bom else "utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """One JSON object per non-blank line; the error names the offending line."""
    records = []
    with Path(path).open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n}: invalid JSON ({exc.msg})") from exc
    return records


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    _atomic_write(Path(path), "".join(_json_line(r) for r in records))


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    """Append one record (for logs and ledgers that must never be rewritten)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(_json_line(record))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_commit(root: Path) -> str | None:
    """HEAD commit of the repository at ``root``, or None if git or the repository is absent."""
    return _git(root, "rev-parse", "HEAD")


def git_dirty(root: Path) -> bool | None:
    """True if the working tree differs from HEAD (untracked files included; ignored files,
    such as ``runs/`` and ``data/raw/``, excluded); None if git is unavailable."""
    status = _git(root, "status", "--porcelain")
    return None if status is None else bool(status)


def write_manifest(run_dir: Path, settings: Settings, inputs: Mapping[str, Path],
                   extra: Mapping[str, Any] | None = None) -> Path:
    """Write ``<run_dir>/manifest.json``: what produced this run, so it can be reproduced.

    Records the package version, git commit and dirty flag, Python version, the sha256 of
    every config file and of every input (``inputs`` maps a label to a path), and the
    stage-specific ``extra`` entries (instrument digests, served models, cache statistics,
    ...), which must not reuse a standard key.
    """
    from .. import __version__

    manifest: dict[str, Any] = {
        "package_version": __version__,
        "git_commit": git_commit(settings.root),
        "git_dirty": git_dirty(settings.root),
        "python": platform.python_version(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_sha256": dict(sorted(settings.shas.items())),
        "prereg_sha256": settings.shas.get("preregistration.yaml"),
        "inputs": {label: {"path": str(p), "sha256": sha256_file(p)} for label, p in sorted(inputs.items())},
    }
    clash = sorted(set(extra or {}) & set(manifest))
    if clash:
        raise ValueError(f"manifest extra entries reuse standard keys: {clash}")
    manifest.update(extra or {})
    path = Path(run_dir) / MANIFEST_NAME
    _atomic_write(path, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return path


# --------------------------------------------------------------------------- internals
def _json_line(record: Mapping[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False) + "\n"


def _atomic_write(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding=encoding, newline="")
    os.replace(tmp, path)


def _git(root: Path, *args: str) -> str | None:
    """Output of a read-only git command, or None on any failure.

    ``--no-optional-locks`` keeps ``git status`` from refreshing the index, so a manifest
    written while someone else commits in the same checkout cannot collide with them.
    """
    try:
        done = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(root), *args],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None
