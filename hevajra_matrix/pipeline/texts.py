"""Claude-free stages on the texts: fetch, ingest (with G0 and ingest-stage sentinels) and
the baselines (P1, B0, imported external alignments)."""

from __future__ import annotations

import json
import urllib.request
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from ..align.anchors import load_anchor_lexicon
from ..align.dp import SOURCE_ANCHOR, SOURCE_ZERO, DPParams, align_groups
from ..align.external import load_tsv
from ..align.similarity import AnchorSimilarity, ZeroSimilarity
from ..core.ids import REF_CHAPTERS
from ..core.io import sha256_file, write_csv, write_jsonl
from ..core.types import Alignment, Segment
from ..evaluation import gate as gates
from ..evaluation import sentinels as sentinel_checks
from ..ingest import cbeta, derge, sanskrit
from ..registry import load_concordance
from .context import CONCORDANCE_FILE, RunContext, StageError, all_segments, load_texts, record_stage, segments_file
from .store import sentinel_result_to_dict, write_alignment, write_segments

# Source files (synthesis 2, stage 1); URLs as in v0.2 (commit 1f9474c). The Derge URL is the
# percent-encoded Tibetan file name of volume 80 (rgyud 'bum, nga) in the Esukhia repository.
SOURCES: Mapping[str, Mapping[str, str]] = {
    "T18n0892.xml": {
        "url": "https://raw.githubusercontent.com/cbeta-org/xml-p5/master/T/T18/T18n0892.xml",
        "license": "CC BY-NC-SA 4.0 (CBETA)",
        "witness": "zh_T0892_song",
    },
    "derge_rgyud_bum_nga.txt": {
        "url": "https://raw.githubusercontent.com/Esukhia/derge-kangyur/master/text/080_%E0%BD%A2%E0%BE%92%E0"
               "%BE%B1%E0%BD%B4%E0%BD%91%E0%BC%8B%E0%BD%A0%E0%BD%96%E0%BD%B4%E0%BD%98%E0%BC%8D_%E0%BD%84.txt",
        "license": "see the Esukhia/derge-kangyur README (to verify)",
        "witness": "bo_derge_D417_418",
    },
}
SENTINELS_FILE = Path("sentinels") / "sentinels.yaml"
VARIANT_COLUMNS = ("segment_id", "locus", "reading", "alternative", "kind")
EXTERNAL_PREFIX = "external:"
GAP_KIND = "gap"          # a Sanskrit reference unit flagged LACUNA or ABSENT: no text, no matrix row


# --------------------------------------------------------------------------- fetch
def fetch(raw_dir: Path, force: bool = False, opener: Callable[[str], bytes] | None = None) -> dict[str, Any]:
    """Download the source texts into ``raw_dir`` and record their sha256 in ``manifest.json``.

    Existing files are kept unless ``force``; their hashes are recorded either way.
    ``opener`` (URL -> bytes) replaces the network in tests.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    fetch_url = opener or _download
    for name, meta in SOURCES.items():
        dest = raw_dir / name
        if force or not dest.is_file():
            print(f"fetch   {meta['url']}")
            dest.write_bytes(fetch_url(meta["url"]))
        else:
            print(f"exists  {dest}")
        manifest[name] = {**meta, "sha256": sha256_file(dest), "bytes": dest.stat().st_size,
                          "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        print(f"        sha256 {manifest[name]['sha256'][:16]}  {manifest[name]['bytes']} bytes")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    return manifest


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:      # fixed https URLs (SOURCES)
        return response.read()


# --------------------------------------------------------------------------- ingest
def raw_paths(ctx: RunContext, raw_dir: Path | None = None) -> dict[str, Path]:
    """Label -> raw source file of the reference and the witness (``run.yaml: sources``)."""
    base = raw_dir or ctx.settings.path("raw")
    sources = ctx.settings.run.get("sources") or {}
    out = {"cbeta": base / str(sources.get("cbeta_file", "")), "derge": base / str(sources.get("derge_file", ""))}
    missing = [str(p) for p in out.values() if not p.is_file()]
    if missing:
        raise StageError(f"raw source file(s) missing: {missing}; run `hevajra-matrix fetch` or pass --raw-dir")
    return out


def ingest(ctx: RunContext, raw_dir: Path | None = None) -> dict[str, Mapping[str, Any]]:
    """Parse the texts; write ``ingest/*`` and report G0 and the ingest-stage sentinels.

    Returns witness -> ingest report. The Derge segments get their reference chapter from
    the concordance. When ``data/reference/<witnesses.sanskrit_reference>.tsv`` exists, its
    units are written too and become the reference of the run (``RunContext.sanskrit_mode``);
    units flagged LACUNA or ABSENT have no text and are kept with kind ``gap`` (no matrix row).
    """
    paths = raw_paths(ctx, raw_dir)
    data = ctx.data_dir
    toh = list(ctx.settings.run["witnesses"].get("derge_toh") or [])
    zh = cbeta.parse(paths["cbeta"], ctx.default_witness, data)
    bo = derge.parse(paths["derge"], toh, ctx.derge, data)
    conc = load_concordance(data / CONCORDANCE_FILE)
    tibetan = conc.assign_reference_chapters(bo.segments, ctx.derge)
    write_segments(segments_file(ctx, ctx.derge), tibetan)
    write_segments(segments_file(ctx, ctx.default_witness), zh.segments)
    out = ctx.path("ingest")
    write_jsonl(out / "footnotes.jsonl", (asdict(f) for f in zh.footnotes))
    write_csv(out / "variants.csv", [asdict(v) for v in (*bo.variants, *zh.variants)], VARIANT_COLUMNS)
    reports = {ctx.derge: dict(bo.report), ctx.default_witness: dict(zh.report)}
    _write_json(out / "witness_meta.json", {ctx.derge: dict(bo.metadata), ctx.default_witness: dict(zh.metadata)})
    inputs = {"raw:cbeta": paths["cbeta"], "raw:derge": paths["derge"]}
    sa_units = sanskrit_reference(ctx)
    if sa_units is not None:
        reports[ctx.sanskrit_reference] = sanskrit_report(sa_units)
        inputs[f"reference:{ctx.sanskrit_reference}"] = sanskrit_path(ctx)
    _write_json(out / "report.json", reports)
    texts = load_texts(ctx)
    results = ingest_sentinels(ctx)
    g0 = g0_reasons(reports, results, ctx.settings.prereg)
    _write_json(out / "g0.json", {"passed": not g0, "reasons": g0})
    record_stage(ctx, "ingest", inputs)
    for witness, report in reports.items():
        print(f"ingest  {witness}: {report.get('segments')} segments, {report.get('content_segments')} content")
    print(f"ingest  reference {texts.reference_id}: {len(texts.units)} units in {len(texts.units_by_chapter())} "
          f"chapters; aligned witnesses {', '.join(ctx.witnesses)}; {len(zh.footnotes)} footnotes")
    _print_sentinels(results)
    print("G0      passed" if not g0 else "G0      failed:\n  " + "\n  ".join(g0))
    return reports


def sanskrit_path(ctx: RunContext) -> Path:
    return ctx.settings.path("reference") / f"{ctx.sanskrit_reference}.tsv"


def sanskrit_reference(ctx: RunContext) -> list[Segment] | None:
    """Load the Sanskrit reference into the run when its TSV exists, else make sure no stale
    copy is left (the run is then in Derge mode)."""
    target = segments_file(ctx, ctx.sanskrit_reference)
    path = sanskrit_path(ctx)
    if not path.is_file():
        target.unlink(missing_ok=True)
        return None
    try:
        units = sanskrit.load_reference(path, ctx.sanskrit_reference)
    except ValueError as exc:
        raise StageError(str(exc)) from exc
    units = [replace(u, kind=GAP_KIND) if u.extra.get("flag") else u for u in units]
    write_segments(target, units)
    return units


def sanskrit_report(units: list[Segment]) -> dict[str, Any]:
    """Ingest counts of the Sanskrit reference (``report.json``)."""
    content = [u for u in units if u.kind != GAP_KIND]
    return {"segments": len(units), "content_segments": len(content),
            "chapters": sorted({str(u.chapter) for u in content}, key=REF_CHAPTERS.index),
            "flagged": dict(Counter(u.extra["flag"] for u in units if u.kind == GAP_KIND)), "duplicate_ids": []}


def ingest_sentinels(ctx: RunContext) -> list[sentinel_checks.SentinelResult]:
    """Check the ingest-stage sentinels on every ingested text; write ``ingest/sentinels.jsonl``."""
    results = sentinel_checks.check(sentinel_checks.load(ctx.data_dir / SENTINELS_FILE), "ingest",
                                    all_segments(ctx))
    write_jsonl(ctx.path("ingest", "sentinels.jsonl"), (sentinel_result_to_dict(r) for r in results))
    return results


def g0_reasons(reports: Mapping[str, Mapping[str, Any]], ingest_results: list[sentinel_checks.SentinelResult],
               prereg: Mapping[str, Any]) -> list[str]:
    """The G0 part of ``evaluation.gate.evaluate`` (the other gates need later stages)."""
    report = gates.evaluate({}, None, gates.Integrity(ingest_reports=reports, sentinels={"ingest": ingest_results}),
                            gates.Perturbations(), gates.Calibration(), prereg, "", gates.LedgerSummary())
    return [r for r in report.reasons if r.startswith("G0:")]


def _print_sentinels(results: list[sentinel_checks.SentinelResult]) -> None:
    counts = Counter((r.status, r.passed) for r in results)
    print(f"sentinels ingest: verified {counts[('verified', True)]} passed / {counts[('verified', False)]} failed; "
          f"proposed {counts[('proposed', True)]} passed / {counts[('proposed', False)]} failed")
    for r in results:
        if not r.passed:
            print(f"  {'BLOCKING ' if r.blocking else ''}{r.sentinel_id}: {r.detail}")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- baselines
BASELINE_FILES = {SOURCE_ZERO: "dp_zero.jsonl", SOURCE_ANCHOR: "dp_anchor.jsonl"}


def baselines(ctx: RunContext) -> dict[str, Alignment]:
    """P1 (length-only DP) and B0 (anchor DP) over every concordance chapter group."""
    texts = load_texts(ctx)
    params = DPParams.from_config(ctx.settings.run)
    groups = texts.groups()
    sims = {SOURCE_ZERO: ZeroSimilarity(),
            SOURCE_ANCHOR: AnchorSimilarity.fit(load_anchor_lexicon(ctx.data_dir),
                                                [*texts.units, *texts.witness_content])}
    out = {}
    for source, sim in sims.items():
        alignment = align_groups(texts.units, texts.witness_content, groups, sim, params, source=source,
                                 reference=texts.reference_id, witness=texts.witness_id)
        write_alignment(ctx.path("alignments", BASELINE_FILES[source]), alignment)
        out[source] = alignment
        linked = sum(1 for link in alignment.links if link.ref_id and link.wit_ids)
        print(f"baselines {source}: {len(alignment.by_ref())} units, {linked} linked, "
              f"{len(alignment.witness_only())} witness-only runs")
    record_stage(ctx, "baselines")
    return out


def external_file(name: str) -> str:
    return f"external_{name}.jsonl"


def baselines_import(ctx: RunContext, tsv: Path, name: str) -> Alignment:
    """Import a precomputed external alignment (critique A4) as control ``external:<name>``.

    Every id must be a segment id of this run's texts: an external alignment has to be
    converted to our ids before it can be scored on the same units as the other controls.
    """
    if not name.replace("-", "").replace("_", "").isalnum():
        raise StageError(f"--name must be letters, digits, '-' or '_', got {name!r}")
    texts = load_texts(ctx)
    alignment = load_tsv(tsv, texts.reference_id, texts.witness_id, f"{EXTERNAL_PREFIX}{name}")
    units, wit = set(texts.ref_kinds), {s.id for s in texts.witness}
    bad = sorted({link.ref_id for link in alignment.links if link.ref_id and link.ref_id not in units}
                 | {w for link in alignment.links for w in link.wit_ids if w not in wit})
    if bad:
        raise StageError(f"{tsv}: {len(bad)} id(s) unknown in this run's texts, e.g. {bad[:3]}")
    write_alignment(ctx.path("alignments", external_file(name)), alignment)
    record_stage(ctx, "baselines-import", {f"external:{name}": tsv})
    print(f"baselines import {name}: {len(alignment.by_ref())} units from {tsv}")
    return alignment
