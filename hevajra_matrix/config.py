"""Run configuration: three YAML files under ``<root>/config`` loaded once per run.

    config/run.yaml               paths, witnesses, windows, DP parameters, review budget
    config/llm.yaml               model, pricing, budget cap, per-task LLM settings
    config/preregistration.yaml   estimands, gold/test sampling, gate thresholds, stats

This module only locates, reads and fingerprints the files. Each consumer owns a
small frozen parameter dataclass and builds it with ``from_mapping``, which rejects
unknown keys, so a typo in a threshold fails loudly instead of silently using a
default. Keeping each parameter schema next to the code that uses it avoids a
central configuration god-object.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml

CONFIG_FILES = {"run": "run.yaml", "llm": "llm.yaml", "prereg": "preregistration.yaml"}

T = TypeVar("T")


class ConfigError(ValueError):
    """Missing, malformed or unknown configuration."""


@dataclass(frozen=True)
class Settings:
    root: Path
    run: Mapping[str, Any]
    llm: Mapping[str, Any]
    prereg: Mapping[str, Any]
    shas: Mapping[str, str]          # config file name -> sha256 of its bytes

    def path(self, key: str) -> Path:
        """Resolve ``run.paths.<key>`` against the repository root."""
        paths = self.run.get("paths", {})
        if key not in paths:
            raise ConfigError(f"run.yaml: paths.{key} is not defined")
        p = Path(paths[key])
        return p if p.is_absolute() else self.root / p


def find_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (default: cwd) to the directory holding ``config/run.yaml``."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "config" / CONFIG_FILES["run"]).is_file():
            return candidate
    raise ConfigError(f"no config/{CONFIG_FILES['run']} found above {here}; pass --root")


def load_settings(root: Path | None = None) -> Settings:
    root = find_root(root) if root is None or not (root / "config").is_dir() else root.resolve()
    loaded: dict[str, Mapping[str, Any]] = {}
    shas: dict[str, str] = {}
    for name, filename in CONFIG_FILES.items():
        path = root / "config" / filename
        if not path.is_file():
            raise ConfigError(f"missing configuration file {path}")
        raw = path.read_bytes()
        shas[filename] = hashlib.sha256(raw).hexdigest()
        doc = yaml.safe_load(raw.decode("utf-8")) or {}
        if not isinstance(doc, Mapping):
            raise ConfigError(f"{path}: top level must be a mapping")
        loaded[name] = doc
    return Settings(root=root, run=loaded["run"], llm=loaded["llm"], prereg=loaded["prereg"], shas=shas)


def from_mapping(cls: type[T], data: Mapping[str, Any] | None, where: str) -> T:
    """Build the frozen dataclass ``cls`` from ``data``; unknown keys raise ``ConfigError``.

    Missing keys fall back to the dataclass defaults. Lists become tuples so that the
    result stays hashable and immutable.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    data = dict(data or {})
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(names)}")
    kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in data.items()}
    try:
        return cls(**kwargs)
    except TypeError as exc:  # missing required field
        raise ConfigError(f"{where}: {exc}") from exc
