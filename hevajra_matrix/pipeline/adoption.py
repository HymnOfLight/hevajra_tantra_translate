"""Claude-free stage ``adopt``: the collate.v1/v2 adoption rule of docs-zh/09 section 7
(``evaluation.adoption``) applied to a built collate.v1 run and a built collate.v2 run on the
repository's dev gold; writes ``evaluation/adoption.json`` into the v2 run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..collate.prompt import PROMPT_V1, PROMPT_V2, load_examples, load_template, schema_for, system_prompt
from ..collate.prompt import version as prompt_version
from ..core.io import read_jsonl, read_yaml
from ..evaluation import adoption as rule
from ..evaluation import gate as gates
from ..evaluation.gold import GoldSet, dev_region_units
from ..llm.client import canonical_json, sha256_text
from .context import RunContext, StageError, Texts, load_texts, record_stage, require
from .human_data import gold_dir, load_gold
from .instrument import COLLATED_SHAS, CONSENSUS_FILE, collated_prompt, dev_chapters, text_langs
from .store import diagnostic_from_dict, read_alignment

ADOPTION_FILE = "adoption.json"
NOTE_MARKERS_FILE = ("codebook", "collate.v2", "note_markers.yaml")    # under data/


def load_note_markers(data_dir: Path) -> tuple[rule.NoteMarker, ...]:
    """``data/codebook/collate.v2/note_markers.yaml``, validated (``evaluation.adoption.note_markers``)."""
    path = Path(data_dir).joinpath(*NOTE_MARKERS_FILE)
    if not path.is_file():
        raise StageError(f"{path} is missing (the note markers of deferred gold conventions)")
    return rule.note_markers(read_yaml(path), str(path))


def adopt(v1: RunContext, v2: RunContext) -> dict[str, Any]:
    """Score the consensus of a collate.v1 run and of a collate.v2 run on the dev gold and apply
    the adoption rule; write ``evaluation/adoption.json`` into the v2 run and record the stage
    there.

    The version a run collated with is read from its ``collation/replicates.json`` (never from
    ``config/llm.yaml``, which need match neither run). Refused: a run collated with another
    version, not at all, or with another text of its version (its recorded ``COLLATED_SHAS``
    differ from the current template, examples and glossary, or are missing); a run whose
    chapters are not exactly the preregistered dev chapters or whose replicates are not
    ``REPLICATES``; a run without a consensus (``build`` not run); two runs whose reference
    units differ; and a witness without dev gold. A dev region the gold does not cover yet makes
    the decision "not ready". The floor of condition 3 is the preregistered
    ``gates.g2.min_replicate_kappa``; the bootstrap of the descriptive v2-v1 difference uses
    ``gates.g1.n_boot`` and ``stats.seed``."""
    runs = {"v1": v1, "v2": v2}
    chapters = sorted(dev_chapters(v2.settings.prereg))
    texts = {label: _check_run(ctx, label, PROMPT_V1 if label == "v1" else PROMPT_V2, chapters)
             for label, ctx in runs.items()}
    one, two = texts["v1"], texts["v2"]
    if (one.reference_id, one.witness_id, one.ref_kinds) != (two.reference_id, two.witness_id, two.ref_kinds):
        raise StageError(f"{v1.run_dir} and {v2.run_dir} hold different texts or reference units; compare two runs "
                         "of the same ingest")
    gold = load_gold(v2, two, "dev")
    if gold is None:
        raise StageError(f"no dev gold for {two.witness_id} yet ({gold_dir(v2, two.witness_id) / 'dev.csv'}); "
                         "the adoption rule is decided on it")
    regions = dev_region_units(list(two.units), (v2.settings.prereg.get("gold") or {}).get("dev_regions") or [])
    gaps = rule.gold_gaps(gold, regions)
    markers = load_note_markers(v2.data_dir)
    specs = gates.GateSpecs.from_prereg(v2.settings.prereg)
    scored = {label: _score(ctx, gold, markers, texts[label]) for label, ctx in runs.items()}
    figures = {label: s.figures() for label, s in scored.items()}
    kappas = {label: _read_json(ctx.path("collation", "integrity.json")).get("fleiss_kappa")
              for label, ctx in runs.items()}
    doc = {"decision": rule.decide(figures["v1"], figures["v2"], kappas["v1"], kappas["v2"],
                                   specs.g2.min_replicate_kappa, gaps),
           "relation_kappa_shared_difference": rule.kappa_difference(
               scored["v1"].readings[rule.DEFAULT], scored["v2"].readings[rule.DEFAULT], specs.g1.n_boot, specs.seed),
           "gold": {"set": gold.name, "witness": gold.witness, "units": len(gold.units()),
                    "windows": [w.window_id for w in gold.windows], "dev_regions": list(regions),
                    "missing": gaps, "dev_chapters": chapters},
           "runs": {label: _provenance(ctx, kappas[label]) for label, ctx in runs.items()},
           "figures": figures}
    path = v2.path("evaluation", ADOPTION_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    record_stage(v2, "adopt")
    _summary(doc, path)
    return doc


def _check_run(ctx: RunContext, label: str, version: str, chapters: list[str]) -> Texts:
    """Refuse a run that is not a section-7 run of ``version``; returns its texts."""
    index = _read_json(ctx.path("collation", "replicates.json"))
    found, where = collated_prompt(ctx), f"--{label}-run {ctx.run_dir}"
    if found != version:
        raise StageError(f"{where} " + (f"was collated with {found}, not {version}" if found
                                        else f"has not been collated (no collation/replicates.json); collate it with "
                                             f"{version}"))
    texts = load_texts(ctx)
    current = _current_text(ctx, version, texts)
    changed = [f"{k} {str(index.get(k) or 'none')[:12]} recorded, {current[k][:12]} now" for k in COLLATED_SHAS
               if index.get(k) != current[k]]
    if changed:
        raise StageError(f"{where} was not collated with the current {version} text ({'; '.join(changed)}): its "
                         "template, examples or verified glossary changed since (docs-zh/09 section 6.4), or the run "
                         "predates the recorded text; collate the dev chapters anew in a new run directory")
    found_chapters, replicates = sorted(index.get("chapters") or ()), len(index.get("tags") or ())
    if found_chapters != chapters or replicates != rule.REPLICATES:
        raise StageError(f"{where} collated chapters {found_chapters} with {replicates} replicate(s); the rule "
                         f"compares runs of the preregistered dev chapters {chapters}, collated whole with "
                         f"{rule.REPLICATES} replicates")
    require(ctx.path("alignments", CONSENSUS_FILE), "build")
    return texts


def _current_text(ctx: RunContext, version: str, texts: Texts) -> dict[str, str]:
    """``COLLATED_SHAS`` of ``version`` as the repository holds it now (as ``collate`` records them)."""
    try:
        examples = load_examples(ctx.data_dir, *text_langs(texts), prompt=version)
    except FileNotFoundError as exc:
        raise StageError(str(exc)) from exc
    return {"prompt_sha": sha256_text(system_prompt(load_template(version), examples)),
            "schema_sha": sha256_text(canonical_json(schema_for(version)))}


def _score(ctx: RunContext, gold: GoldSet, markers: tuple[rule.NoteMarker, ...], texts: Texts) -> rule.RunScores:
    """The run's consensus scored as ``evaluate`` scores it (reasons, diagnostics, conventions)."""
    diagnostics_path = ctx.path("collation", "diagnostics.jsonl")
    diagnostics = [diagnostic_from_dict(d) for d in read_jsonl(diagnostics_path)] if diagnostics_path.is_file() else []
    return rule.score_run(read_alignment(ctx.path("alignments", CONSENSUS_FILE)), gold, markers,
                          ref_kinds=texts.ref_kinds,
                          unresolved=_read_json(ctx.path("collation", "consensus.json")).get("reasons"),
                          diagnostics=diagnostics, conventions=prompt_version(str(collated_prompt(ctx))).conventions)


def _provenance(ctx: RunContext, kappa: float | None) -> dict[str, Any]:
    index = _read_json(ctx.path("collation", "replicates.json"))
    return {"run_dir": str(ctx.run_dir), "prompt": collated_prompt(ctx), "prompt_sha": index.get("prompt_sha"),
            "schema_sha": index.get("schema_sha"), "replicates": len(index.get("tags") or ()),
            "chapters": index.get("chapters"), "fleiss_kappa": kappa}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _outcome(c: Mapping[str, Any]) -> str:
    return {True: "passed", False: "failed", None: "undefined"}[c["passed"]]


def _summary(doc: Mapping[str, Any], path: Path) -> None:
    d = doc["decision"]
    k, p, r = (d["conditions"][name] for name in ("relation_kappa_shared", "convention_precision", "replicate_kappa"))
    v2 = doc["figures"]["v2"]
    outside = "outside " + ", ".join(rule.TUNING_WINDOWS)
    diff = doc["relation_kappa_shared_difference"]
    deferred = ", ".join(f"{m} {n}" for m, n in v2["deferred"].items())
    print(f"adopt: dev gold {doc['gold']['units']} units, {len(doc['gold']['missing'])} dev region(s) not covered; "
          f"deferred '?' by note marker: {deferred} ({len(v2['blocked'])} blocked, {len(v2['unmarked'])} without "
          "a known marker)")
    print(f"  (1) relation_kappa_shared {outside}: v2 {_fmt(k['v2'])} >= v1 {_fmt(k['v1'])} - {rule.TOLERANCE}: "
          f"{_outcome(k)}")
    print(f"  (2) convention precision {outside}: v2 {_fmt(p['v2'])} >= {p['threshold']}: {_outcome(p)} "
          f"(without the in-sample units {_fmt(p['v2_without_in_sample'])}, not decisive)")
    print(f"  (3) replicate kappa floor {r['threshold']}: v1 {_fmt(r['v1'])}, v2 {_fmt(r['v2'])}: {_outcome(r)}"
          + ("" if r["applies"] or r["passed"] is None else " (v1 below the floor: does not decide)"))
    print(f"  decision: {d['decision']}" + (f" ({'; '.join(d['not_ready'])})" if d["not_ready"] else ""))
    print(f"  sensitivity, '?' left out: {d['sensitivity']['decision']}")
    if d["note"]:
        print(f"  note: {d['note']}")
    print(f"  v2 - v1 relation_kappa_shared {outside}: {_fmt(diff['point'])} [{_fmt(diff['lo'])}, {_fmt(diff['hi'])}] "
          "(paired window bootstrap, descriptive)")
    print(f"adopt: {path}")
