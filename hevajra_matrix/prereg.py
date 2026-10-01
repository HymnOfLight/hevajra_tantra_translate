"""Instrument digests and ``prereg freeze`` (synthesis 2, 5.5; critique A7).

An instrument is everything that decides what an LLM task measures: model, task, effort,
the rendered static prompt, the output schema, the code version, the task parameters and
the number of replicates (``core.types.InstrumentId``). Its digest changes whenever any of
these change, which is what makes a validated instrument re-validatable:

    collate     T1, the measurement instrument (gate G2 compares its digest with the prereg);
                one more digest ``collate:<ref>-<wit>`` per extra language pair that has its
                own examples file (the Sanskrit reference: ``collate:sa-bo``, ``collate:sa-zh``)
    topics      T3 pre-labeller (orders human work; frozen for provenance)
    components  T2 component coder (descriptive)
    subject     T4 experiment subject
    scorer      T5 experiment scorer

``freeze`` writes the digests and ``frozen: true`` into ``config/preregistration.yaml``.
A frozen file is only changed through ``--amend "<reason>"``, which recomputes the digests
and appends ``{date, reason, changed}`` to ``amendments``. The file is rewritten in place:
only the lines of the top-level keys ``frozen``, ``instrument_digests`` and ``amendments``
are replaced, so every other key, its order and its comments stay as they were; if that
targeted edit cannot be verified by re-parsing, the whole document is dumped instead
(keys and order kept, comments lost).
"""

from __future__ import annotations

import re
from dataclasses import asdict
from datetime import date as Date
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import __version__
from .collate import collator
from .collate import components
from .collate.windows import WindowParams
from .config import CONFIG_FILES, ConfigError, Settings, load_settings
from .core.types import InstrumentId
from .experiments.overattribution import design as exp_design
from .experiments.overattribution import run as exp_run
from .experiments.overattribution import score as exp_score
from .llm.client import canonical_json, sha256_text
from .registry import load_witnesses
from .topics import TopicTaskSettings, load_codebook, prelabel_schema, render_system

PREREG_FILE = CONFIG_FILES["prereg"]
MANAGED_KEYS = ("frozen", "instrument_digests", "amendments")


class PreregError(RuntimeError):
    """Freezing is refused (already frozen without an amendment, or a malformed file)."""


# --------------------------------------------------------------------------- digests
def _params_sha(*parts: Any) -> str:
    return sha256_text(canonical_json([asdict(p) if hasattr(p, "__dataclass_fields__") else p for p in parts]))


def _instrument(task: str, model: str, effort: str, system: str, schema: Any, params: tuple[Any, ...],
                replicates: int) -> InstrumentId:
    return InstrumentId(model=model, task=task, effort=effort, prompt_sha=sha256_text(system),
                        schema_sha=sha256_text(canonical_json(schema)), code_version=__version__,
                        params_sha=_params_sha(*params), replicates=replicates)


def instruments(settings: Settings) -> dict[str, InstrumentId]:
    """The current ``InstrumentId`` of every LLM task, from code, prompts, data and config."""
    llm, data = settings.llm, settings.path("data")
    reference = str(settings.run["witnesses"]["reference"])
    ref_lang = load_witnesses(data / "registry" / "witnesses.yaml")[reference].lang

    t1 = collator.TaskSettings.from_config(llm)
    template = collator.load_template()
    collate = {collate_task(*pair): _instrument(
        "collate", t1.model, t1.effort, collator.system_prompt(template, collator.load_examples(data, *pair)),
        collator.SCHEMA, (t1, WindowParams.from_config(settings.run)), t1.replicates)
        for pair in collator.example_pairs(data)}
    t3 = TopicTaskSettings.from_config(llm)
    codebook = load_codebook(data / "codebook" / "topics.yaml")
    t2 = components.ComponentTaskSettings.from_config(llm)
    t4 = exp_run.ExperimentTaskSettings.from_config(llm, exp_run.SUBJECT_TASK)
    t5 = exp_run.ExperimentTaskSettings.from_config(llm, exp_run.SCORER_TASK)
    evidence = exp_design.load_evidence(data / "experiments" / "overattribution" / "evidence.yaml")
    subject_prompt = exp_run.SUBJECT_PROMPT.read_text(encoding="utf-8") + canonical_json(
        {k: v.text for k, v in evidence.items()})
    return {
        **collate,
        "topics": _instrument("topics", t3.model, t3.effort, render_system(codebook, ref_lang),
                              prelabel_schema(codebook), (t3,), t3.replicates),
        "components": _instrument("components", t2.model, t2.effort, components.load_system(data),
                                  components.SCHEMA, (t2,), t2.replicates),
        "subject": _instrument("subject", t4.model, t4.effort, subject_prompt, exp_score.subject_schema(), (t4,),
                               t4.replicates),
        "scorer": _instrument("scorer", t5.model, t5.effort, exp_run.SCORER_PROMPT.read_text(encoding="utf-8"),
                              exp_score.scorer_schema(), (t5,), t5.replicates),
    }


def collate_task(ref_lang: str, wit_lang: str) -> str:
    """Digest key of the collator for one language pair: ``collate`` for the default pair
    (Tibetan -> Chinese), ``collate:<ref>-<wit>`` for any other pair with its own examples."""
    return "collate" if (ref_lang, wit_lang) == collator.DEFAULT_PAIR else f"collate:{ref_lang}-{wit_lang}"


def instrument_digests(settings: Settings) -> dict[str, str]:
    """Task -> sha256 digest of its ``InstrumentId``."""
    return {task: ident.digest() for task, ident in instruments(settings).items()}


# --------------------------------------------------------------------------- freeze
def freeze(root: Path, amend: str | None = None, today: Date | None = None) -> dict[str, Any]:
    """Freeze the preregistration (or amend a frozen one); returns the new document.

    Refuses a frozen file without ``amend``, and an empty amendment reason.
    """
    settings = load_settings(root)
    path = settings.root / "config" / PREREG_FILE
    doc = dict(settings.prereg)
    missing = [k for k in MANAGED_KEYS if k not in doc]
    if missing:
        raise PreregError(f"{path}: missing top-level key(s) {missing}")
    if amend is not None and not amend.strip():
        raise PreregError("an amendment needs a reason")
    frozen = doc.get("frozen") is True
    if frozen and amend is None:
        raise PreregError(f"{path} is already frozen; change it only with --amend \"<reason>\"")
    old = dict(doc.get("instrument_digests") or {})
    digests = instrument_digests(settings)
    amendments = list(doc.get("amendments") or [])
    if amend is not None:
        changed = sorted(t for t in digests if old.get(t) != digests[t])
        amendments.append({"date": (today or Date.today()).isoformat(), "reason": " ".join(amend.split()),
                           "changed": changed, "was_frozen": frozen})
    new = {**doc, "frozen": True, "instrument_digests": digests, "amendments": amendments}
    _write(path, new)
    return new


def _write(path: Path, doc: Mapping[str, Any]) -> None:
    text = path.read_text(encoding="utf-8")
    edited = text
    for key in MANAGED_KEYS:
        edited = replace_top_level(edited, key, doc[key])
    try:
        ok = yaml.safe_load(edited) == dict(doc)
    except yaml.YAMLError:
        ok = False
    if not ok:
        edited = "# Written by `hevajra-matrix prereg freeze` (comments could not be preserved).\n" + yaml.safe_dump(
            dict(doc), sort_keys=False, allow_unicode=False, width=100)
    path.write_text(edited, encoding="utf-8")


def replace_top_level(text: str, key: str, value: Any) -> str:
    """Replace the block of top-level ``key`` (its line plus indented continuation lines)
    with ``value`` dumped as YAML; a trailing comment on the key's line is kept."""
    lines = text.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if re.match(rf"{re.escape(key)}\s*:", line)), None)
    if start is None:
        raise ConfigError(f"top-level key {key!r} not found")
    end = start + 1
    while end < len(lines) and (lines[end].startswith((" ", "\t", "-")) or not lines[end].strip()):
        end += 1
    while end > start + 1 and not lines[end - 1].strip():       # keep blank lines before the next key
        end -= 1
    comment = re.search(r"\s#.*$", lines[start].rstrip("\n"))
    dumped = yaml.safe_dump({key: value}, sort_keys=False, allow_unicode=False, width=100, default_flow_style=False)
    first, *rest = dumped.splitlines()
    note = comment.group(0).strip() if comment else ""
    if note and not rest:
        first = f"{first:<38} {note}"
    block = "\n".join(([note] if note and rest else []) + [first, *rest]) + "\n"
    return "".join(lines[:start]) + block + "".join(lines[end:])
