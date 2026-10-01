"""Shared helpers for the pipeline, CLI and report tests (no tests here).

``make_root`` builds a throw-away repository root: a copy of ``config/`` and ``data/`` with
the mini CBETA and Derge fixtures installed as the raw texts, and a preregistration whose
dev regions and test windows fit those tiny texts. ``echo_script`` is a FakeClient
collator that answers every T1 request with verifiable links (quotes copied from the
prompt), and ``fill_cache`` writes its answers into the response cache through
``CachedClient``, so that the pipeline can then run ``--offline``.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import yaml

from hevajra_matrix.config import load_settings
from hevajra_matrix.llm.cache import CachedClient
from hevajra_matrix.llm.client import LLMRequest
from hevajra_matrix.llm.fake import FakeClient
from hevajra_matrix.pipeline import RunContext, plan_collation
from hevajra_matrix.pipeline.context import load_texts
from hevajra_matrix.topics import TopicTaskSettings, build_request, load_codebook, plan_batches

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "data" / "fixtures"
MINI_PREREG = {"dev_regions": [{"chapter": "I.1"}],
               "test_windows": {"n_windows": 1, "width": 2, "second_annotator_windows": 1, "seed": 7}}
_LINE = re.compile(r"^([rz]\d+)\t(\w+)\t(.*)$")
_CORE = re.compile(r"^CORE WINDOW: (.*)$", re.M)


def make_root(tmp_path: Path, gold: dict[str, Any] | None = None) -> Path:
    """A repository root under ``tmp_path`` whose raw texts are the mini fixtures."""
    root = tmp_path / "repo"
    shutil.copytree(REPO / "config", root / "config")
    shutil.copytree(REPO / "data", root / "data", ignore=shutil.ignore_patterns("raw", "reference", "__pycache__"))
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / "mini_cbeta.xml", raw / "T18n0892.xml")
    shutil.copy(FIXTURES / "mini_derge.txt", raw / "derge_rgyud_bum_nga.txt")
    path = root / "config" / "preregistration.yaml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    doc["gold"] = gold or MINI_PREREG
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return root


SANSKRIT = FIXTURES / "sa_mini"


def make_sanskrit_root(tmp_path: Path, readings: bool = True) -> Path:
    """``make_root`` plus the synthetic Sanskrit reference (and its manuscript readings) in
    ``data/reference/`` and the test-only sa->bo / sa->zh collator examples."""
    root = make_root(tmp_path)
    reference = root / "data" / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    shutil.copy(SANSKRIT / "sa_snellgrove1959.tsv", reference)
    if readings:
        shutil.copytree(SANSKRIT / "readings", reference / "readings")
    for pair in ("sa-bo", "sa-zh"):
        shutil.copy(SANSKRIT / f"collate_examples.{pair}.yaml", root / "data" / "codebook")
    return root


def fill_all(ctx: RunContext) -> int:
    """``fill_cache`` for every aligned witness of the run."""
    return sum(fill_cache(one) for one in ctx.each_witness())


def context(root: Path, run_dir: Path, offline: bool = False) -> RunContext:
    settings = load_settings(root)
    return RunContext(settings.root, run_dir, settings, offline)


def _lines(block: str) -> list[tuple[str, str, str]]:
    return [m.groups() for line in block.splitlines() if (m := _LINE.match(line))]


def _core_handles(body: str, witness: list[tuple[str, str, str]]) -> list[tuple[str, str]]:
    """(handle, text) of the core-window content lines, in order."""
    found = _CORE.search(body)
    ranges = [] if not found or found.group(1) == "(empty)" else [
        tuple(part.split("-")) if "-" in part else (part, part) for part in found.group(1).split(", ")]
    order = [h for h, _, _ in witness]
    inside = {h for a, b in ranges for h in order[order.index(a):order.index(b) + 1]}
    return [(h, text) for h, kind, text in witness if h in inside and kind in ("prose", "verse", "mantra")]


def echo_script(request: LLMRequest) -> dict[str, Any]:
    """A faithful-looking collator: unit i links core line i (cycling); every fourth unit,
    and every unit of a window with an empty core, has no counterpart."""
    witness = _lines(request.context or "")
    core = _core_handles(request.body, witness)
    units = []
    for i, (handle, _, text) in enumerate(_lines(request.body)):
        if not core or i % 4 == 3:
            units.append({"ref": handle, "wit": [], "relation": "no_counterpart", "polarity_flip": False,
                          "confidence": "medium", "ref_quote": text, "wit_quote": ""})
        else:
            z, z_text = core[i % len(core)]
            units.append({"ref": handle, "wit": [z], "relation": "equivalent", "polarity_flip": False,
                          "confidence": "high", "ref_quote": "", "wit_quote": z_text})
    return {"units": units, "witness_only": []}


def neutral_prelabels(request: LLMRequest) -> dict[str, Any]:
    """A T3 answer labelling every unit to label ``neutral``."""
    units = request.body.split("\n\n")[1].splitlines()[1:]
    return {"units": [{"ref": line.split("\t")[0], "topics": [{"topic": "neutral", "cue": ""}]} for line in units]}


def fill_prelabel_cache(ctx: RunContext) -> int:
    """Store ``neutral_prelabels`` answers for every T3 request of the run."""
    texts = load_texts(ctx)
    settings = TopicTaskSettings.from_config(ctx.settings.llm)
    codebook = load_codebook(ctx.data_dir / "codebook" / "topics.yaml")
    cache = CachedClient(FakeClient(neutral_prelabels), ctx.settings.path("cache"))
    batches = plan_batches(list(texts.units), settings.units_per_call)
    for batch in batches:
        cache.complete(build_request(batch, codebook, settings))
    return len(batches)


def fill_cache(ctx: RunContext, replicates: int | None = None) -> int:
    """Answer every T1 request of the run with ``echo_script`` and store it in the cache."""
    cache = CachedClient(FakeClient(echo_script), ctx.settings.path("cache"))
    requests = [r for _, _, r in plan_collation(ctx, replicates=replicates).requests()]
    for request in requests:
        cache.complete(request)
    return len(requests)
