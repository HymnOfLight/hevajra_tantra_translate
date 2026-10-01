"""Human-work stages: topic pre-labels and sheets, sampling, and the review sheet round trip.

Sheets carry licensed text and are written under ``<run_dir>/review/`` only; imports write
the committed files under ``data/annotations/`` (ids and short quotes only).
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date as Date
from pathlib import Path
from typing import Any, Sequence

from ..core.io import read_jsonl, write_jsonl
from ..core.types import Verdict
from ..evaluation import gold as gold_sets
from ..review import sampling, sheets, topic_sheets
from ..review import verdicts as verdict_files
from ..topics import (
    Prelabel,
    TopicTaskSettings,
    build_request,
    load_labels,
    parse_prelabels,
    plan_batches,
    write_labels,
)
from . import human_data as ann
from .context import RunContext, StageError, Texts, has_api_key, load_texts, make_client, record_stage, require
from .instrument import read_consensus
from .measure import read_built_cells

PRELABELS = ("topics", "prelabels.jsonl")
SECOND_CODER_FRACTION = 0.2
REVIEW_TASKS = ("gold", "verify", "audit", "resolve", "reveal", "topics", "topics_second")
RESOLVE_SUFFIX = "_resolve"


# --------------------------------------------------------------------------- topics (T3 and sheets)
def topics_prelabel(ctx: RunContext) -> dict[str, int]:
    """T3 pre-labels of every reference unit (reference text only), as reviewer hints."""
    texts = load_texts(ctx)
    if not ctx.offline and not has_api_key():
        raise StageError("topics prelabel needs ANTHROPIC_API_KEY, or --offline with a filled cache")
    settings = TopicTaskSettings.from_config(ctx.settings.llm)
    batches = plan_batches(list(texts.units), settings.units_per_call)
    codebook = ann.codebook(ctx)
    client = make_client(ctx)
    requests = [build_request(b, codebook, settings) for b in batches]
    with ThreadPoolExecutor(max_workers=int(ctx.settings.llm.get("workers", 1))) as pool:
        responses = list(pool.map(client.complete, requests))
    labels = {uid: p for b, r in zip(batches, responses) for uid, p in parse_prelabels(r, b).items()}
    write_jsonl(ctx.path(*PRELABELS), ({"unit_id": p.unit_id, "topics": sorted(p.topics), "cues": [list(c) for c in p.cues],
                                        "flags": sorted(p.flags), "reason": p.reason} for p in labels.values()))
    record_stage(ctx, "topics-prelabel")
    usable = sum(1 for p in labels.values() if p.usable)
    print(f"topics prelabel: {len(labels)} units in {len(batches)} calls, {usable} usable pre-labels")
    return {"units": len(labels), "usable": usable}


def read_prelabels(ctx: RunContext) -> dict[str, Prelabel]:
    path = ctx.path(*PRELABELS)
    if not path.is_file():
        return {}
    return {r["unit_id"]: Prelabel(r["unit_id"], frozenset(r["topics"]), tuple(tuple(c) for c in r["cues"]),
                                   frozenset(r["flags"]), r.get("reason")) for r in read_jsonl(path)}


# --------------------------------------------------------------------------- sampling
def sample(ctx: RunContext, kind: str, batch: str | None = None, force: bool = False) -> Path:
    """Draw ``windows`` (dev regions + test windows, committed), ``verification`` or ``audit``
    (a plan ``review/plan_<batch>.csv`` in the run directory)."""
    texts = load_texts(ctx)
    if kind == "windows":
        return _sample_windows(ctx, texts, force)
    if kind not in ("verification", "audit"):
        raise StageError(f"sample: unknown kind {kind!r}; use windows, verification or audit")
    machine = read_built_cells(ctx, machine=True)
    groups = ann.load_topics(ctx, texts).groups()
    decided = {v.unit_id for v in ann.current_verdicts(ann.review_verdicts(ctx, texts), texts) if v.final_relation}
    decided |= {v.unit_id for v in ann.gold_verdicts(ctx, texts)}
    if kind == "verification":
        spec = sampling.VerificationSpec.from_config(ctx.settings.prereg)
        items = sampling.draw_verification(machine, groups, spec, ref_kinds=texts.ref_kinds, verified=decided)
    else:
        spec_a = sampling.AuditSpec.from_config(ctx.settings.prereg)
        items = sampling.draw_audit(machine, groups, spec_a, ref_kinds=texts.ref_kinds, verified=decided)
    path = ann.plan_path(ctx, batch or ("verify" if kind == "verification" else "audit"))
    if path.exists() and not force:
        raise StageError(f"{path} exists; a plan is drawn once (pass --force to redraw it)")
    sampling.write_plan(items, path)
    record_stage(ctx, f"sample-{kind}")
    print(f"sample {kind}: {len(items)} items -> {path}")
    return path


def _sample_windows(ctx: RunContext, texts: Texts, force: bool) -> Path:
    gold_cfg = ctx.settings.prereg.get("gold") or {}
    spec = gold_cfg.get("test_windows") or {}
    regions = gold_sets.dev_region_units(list(texts.units), gold_cfg.get("dev_regions") or [])
    dev = [gold_sets.region_window(rid, units, texts.chapter_of) for rid, units in regions.items()]
    test = gold_sets.draw_test_windows(texts.units_by_chapter(), int(spec["n_windows"]), int(spec["width"]),
                                       {u for units in regions.values() for u in units}, int(spec["seed"]),
                                       drawn_at=Date.today().isoformat())
    path = ann.gold_dir(ctx, texts.witness_id) / gold_sets.WINDOWS_FILE
    if path.exists() and not force:
        raise StageError(f"{path} exists; test windows are drawn once (pass --force to redraw them)")
    gold_sets.write_windows([*dev, *test], path)
    print(f"sample windows: {len(dev)} dev regions and {len(test)} test windows -> {path}")
    return path


# --------------------------------------------------------------------------- export
def review_export(ctx: RunContext, task: str, batch: str | None = None, gold_set: str | None = None,
                  hours: float | None = None, competence: Sequence[str] = ()) -> list[Path]:
    """Write the sheets of one batch under ``review/`` (see ``review.sheets``)."""
    if task not in REVIEW_TASKS:
        raise StageError(f"review export: unknown task {task!r}; use one of {', '.join(REVIEW_TASKS)}")
    texts = load_texts(ctx)
    out = ctx.path("review")
    segments = [*texts.reference, *texts.witness]
    if task == "gold":
        return _export_gold(ctx, texts, gold_set or "", out, segments)
    if task in ("topics", "topics_second"):
        items = _topic_items(ctx, texts, task)
        if not items:
            raise StageError(f"review export: no unit left for {task}")
        default = "first" if task == "topics" else "second"          # topics_first.csv, topics_second.second.csv
        return sheets.export(items, task, out, segments, (), batch=batch or default, prelabels=read_prelabels(ctx))
    batch = batch or ("audit" if task == "audit" else "verify")      # resolve items are in the verification plan
    machine = read_built_cells(ctx, machine=True)
    if task == "reveal":
        blind = verdict_files.read_file(require(ann.verdict_dir(ctx, texts.witness_id) / f"{batch}.csv", "review import"))
        items = [sampling.ReviewItem(v.item_id, v.task, v.unit_id, v.stratum) for v in blind]
        consensus = read_consensus(ctx)
        return sheets.export(items, "reveal", out, segments, machine, batch=batch, blind=blind,
                             links=consensus.by_ref() if consensus else {})
    items = [i for i in sampling.read_plan(require(ann.plan_path(ctx, batch), "sample")) if i.task == task]
    if hours is not None:
        done = {v.item_id for v in ann.review_verdicts(ctx, texts) if v.blind_relation}
        items = sampling.queue(items, hours * 60, competence, sampling.ReviewParams.from_config(ctx.settings.run), done)
    if not items:
        raise StageError(f"review export: no {task} item in plan_{batch}.csv (or none fits the budget)")
    sheet = batch if task != "resolve" else f"{batch}{RESOLVE_SUFFIX}"      # a plan's resolve items get their own sheet
    paths = sheets.export(items, task, out, segments, machine, batch=sheet)
    print(f"review export {task}: {len(items)} items -> {paths[0]}")
    return paths


def _export_gold(ctx: RunContext, texts: Texts, gold_set: str, out: Path, segments: list) -> list[Path]:
    if gold_set not in gold_sets.SETS:
        raise StageError(f"review export --task gold needs --set {' | '.join(gold_sets.SETS)}")
    windows = gold_sets.read_windows(require(ann.gold_dir(ctx, texts.witness_id) / gold_sets.WINDOWS_FILE,
                                             "sample windows"))
    regions = gold_sets.dev_region_units(list(texts.units), (ctx.settings.prereg.get("gold") or {}).get("dev_regions")
                                         or [])
    by_chapter = texts.units_by_chapter()
    neighbours = int((ctx.settings.run.get("windows") or {}).get("neighbours", 1))
    chosen = [w for w in windows if (w.seed is None) == (gold_set == "dev")]
    n_second = int(((ctx.settings.prereg.get("gold") or {}).get("test_windows") or {}).get("second_annotator_windows", 0))
    if gold_set == "test_second":
        chosen = chosen[:n_second]
    paths: list[Path] = []
    for w in chosen:
        units = regions.get(w.window_id) if w.seed is None else gold_sets.window_units(w, by_chapter)
        if not units:
            raise StageError(f"window {w.window_id}: no units (dev regions changed since windows.csv was drawn?)")
        locals_ = sorted({loc for ch in w.chapter.split("+")
                          for loc in texts.concordance.core_window(ch, texts.witness_id, neighbours)})
        paths += sheets.export([sampling.make_item("gold", u) for u in units], "gold", out / "gold", segments, (),
                               batch=f"{gold_set}_{w.window_id}", witness_chapters=locals_)
    print(f"review export gold: {len(chosen)} {gold_set} window(s) -> {out / 'gold'}")
    return paths


def _topic_items(ctx: RunContext, texts: Texts, task: str) -> list[sampling.ReviewItem]:
    labels = ann.load_topics(ctx, texts).labels
    if task == "topics":
        units = [s.id for s in texts.units if not (labels.get(s.id) and labels[s.id].topics)]
    else:
        pool = sorted(u for u, x in labels.items() if x.topics and not x.second_topics)
        seed = int((ctx.settings.prereg.get("stats") or {}).get("seed", 0))
        k = round(SECOND_CODER_FRACTION * sum(1 for x in labels.values() if x.topics)) - \
            sum(1 for x in labels.values() if x.second_topics)
        order = {s.id: i for i, s in enumerate(texts.units)}
        units = sorted(random.Random(f"{seed}:topics_second").sample(pool, max(0, min(k, len(pool)))), key=order.get)
    return [sampling.make_item(task, u) for u in units]


# --------------------------------------------------------------------------- import
def review_import(ctx: RunContext, task: str, path: Path, annotator: str, date: str | None = None,
                  minutes: float | None = None) -> Path:
    """Import a filled sheet into the committed annotations; returns the file written."""
    texts = load_texts(ctx)
    date = date or Date.today().isoformat()
    params = sampling.ReviewParams.from_config(ctx.settings.run)
    langs = dict(ref_lang=texts.reference[0].lang, wit_lang=texts.witness[0].lang)
    if task in ("topics", "topics_second"):
        return _import_topics(ctx, texts, path, annotator, date)
    if task == "gold":
        return _import_gold(ctx, texts, sheets.import_(path, "gold", params=params, annotator=annotator, date=date,
                                                      minutes=minutes, **langs))
    batch = sheets.batch_of(path, ".reveal.csv" if task == "reveal" else ".blind.csv")
    target = ann.verdict_dir(ctx, texts.witness_id) / f"{batch}.csv"
    if task == "reveal":
        from ..prereg import instrument_digests   # local import: prereg imports every task module
        blind = verdict_files.read_file(require(target, "review import"))
        new = sheets.import_(path, "reveal", params=params, date=date, blind=blind,
                             instrument_digest=instrument_digests(ctx.settings)["collate"], **langs)
        verdict_files.save(new, target)
    else:
        plan_file = ann.plan_path(ctx, batch.removesuffix(RESOLVE_SUFFIX) if task == "resolve" else batch)
        plan = sampling.read_plan(plan_file) if plan_file.is_file() else []
        new = sheets.import_(path, task, params=params, annotator=annotator, date=date, items=plan, minutes=minutes,
                             **langs)
        old = verdict_files.read_file(target) if target.is_file() else []
        verdict_files.save(_merge(old, new), target)
    print(f"review import {task}: {len(new)} verdict(s) -> {target}")
    return target


def _merge(old: Sequence[Verdict], new: Sequence[Verdict]) -> list[Verdict]:
    """Old verdicts of the batch with those re-imported replaced by item id (order kept)."""
    fresh = {v.item_id: v for v in new}
    return [fresh.pop(v.item_id, v) for v in old] + list(fresh.values())


def _import_gold(ctx: RunContext, texts: Texts, verdicts: list[Verdict]) -> Path:
    if not verdicts:
        raise StageError("the gold sheet holds no decision")
    batch = verdicts[0].batch_id
    set_name = max((s for s in gold_sets.SETS if batch.startswith(f"{s}_")), key=len, default=None)
    if set_name is None:
        raise StageError(f"gold batch {batch!r} does not start with one of {gold_sets.SETS}")
    new = gold_sets.from_verdicts(verdicts, set_name, texts.witness_id, texts.reference_id)
    target = ann.gold_dir(ctx, texts.witness_id) / f"{set_name}.csv"
    known = {w.window_id for w in gold_sets.read_windows(require(target.with_name(gold_sets.WINDOWS_FILE),
                                                                 "sample windows"))}
    unknown = sorted({r.window_id for r in new.rows} - known)
    if unknown:
        raise StageError(f"gold window(s) {unknown} are not in {gold_sets.WINDOWS_FILE}; nothing was imported")
    old = gold_sets.load(target, texts.reference_id) if target.is_file() else None
    windows = {r.window_id for r in new.rows}
    rows = tuple(r for r in (old.rows if old else ()) if r.window_id not in windows) + new.rows
    gold_sets.save(replace(new, rows=rows), target)
    print(f"review import gold: {len(new.rows)} row(s) of window(s) {sorted(windows)} -> {target}")
    return target


def _import_topics(ctx: RunContext, texts: Texts, path: Path, coder: str, date: str) -> Path:
    codebook = ann.codebook(ctx)
    imported = topic_sheets.import_topics(path, codebook, coder=coder, date=date, prelabels=read_prelabels(ctx))
    target = ann.topic_labels_path(ctx, texts.reference_id)
    existing = load_labels(target, codebook) if target.is_file() else {}
    merged = topic_sheets.merge_topic_labels(existing, imported)
    order = {s.id: i for i, s in enumerate(texts.units)}
    write_labels(target, sorted(merged.values(), key=lambda x: order.get(x.unit_id, len(order))))
    print(f"review import topics: {len(imported)} label(s) -> {target}")
    return target


# --------------------------------------------------------------------------- status
def review_status(ctx: RunContext) -> dict[str, Any]:
    """Coverage per stratum (planned / blind / final), gold rows and topic labels."""
    texts = load_texts(ctx)
    verdicts = ann.review_verdicts(ctx, texts)
    coverage = sampling.coverage(ann.review_plans(ctx), verdicts)
    lines = sampling.format_coverage(coverage)
    gold = {name: len(g.rows) for name in gold_sets.SETS if (g := ann.load_gold(ctx, texts, name)) is not None}
    topics = ann.load_topics(ctx, texts)
    labelled = sum(1 for x in topics.labels.values() if x.topics)
    print("\n".join(lines) if lines else "review status: no review plan in this run")
    print(f"gold rows: {gold or 'none'}; topic labels {labelled}/{topics.n_units} ({len(topics.stale)} stale); "
          f"verdicts {len(verdicts)} ({len(verdicts) - len(ann.current_verdicts(verdicts, texts))} stale or orphan-row)")
    return {"coverage": coverage, "gold": gold, "topics_labelled": labelled}

