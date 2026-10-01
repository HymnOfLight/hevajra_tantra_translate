"""A whole campaign on the real texts with a simulated Claude and simulated annotators.

Every stage runs through the CLI, offline, in a temporary repository root: ingest,
baselines, collate (3 replicates), build, sample windows, blind gold (dev, test,
test_second), perturb, prereg freeze, evaluate, topics prelabel and both topic sheets,
sample verification and audit, blind review, resolve and reveal, build, evaluate, stats
and report. The report level must climb 0 -> 1 -> 2 as the gates allow.

The simulation: the truth is the B0 anchor baseline with seeded relation changes (absent,
substitution, reversal, abridged, ...), longer links B0 misses, and the verified sentinel
facts. Claude is the truth with systematic errors (B0 copied, false absences), per-replicate
noise, invalid quotes, omitted units, one refused window and one refused-and-truncated
window. Its answers are written into the response cache through ``CachedClient`` over a
``FakeClient`` before the offline stages read them. Annotators answer from the truth.

Skipped unless HEVAJRA_RAW_DIR points at the fetched files.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from hevajra_matrix.cli import main
from hevajra_matrix.collate import collator
from hevajra_matrix.collate.perturb import delete_segments, wrong_window
from hevajra_matrix.config import load_settings
from hevajra_matrix.evaluation import sentinels as sentinel_checks
from hevajra_matrix.ingest import CONTENT_KINDS
from hevajra_matrix.llm.cache import CachedClient
from hevajra_matrix.llm.client import LLMRequest
from hevajra_matrix.llm.fake import FakeClient, ok_response, refusal_response, truncated_response
from hevajra_matrix.pipeline import RunContext, plan_collation
from hevajra_matrix.pipeline.context import load_texts
from hevajra_matrix.pipeline.instrument import dev_chapters, far_chapter
from hevajra_matrix.pipeline.store import read_alignment
from hevajra_matrix.topics import TopicTaskSettings, build_request, load_codebook, plan_batches

pytestmark = pytest.mark.realdata
REPO = Path(__file__).resolve().parents[2]
TSHEG = "\u0f0b"                      # Tibetan intersyllabic mark
BAD_QUOTE = "\u9451\u9f98\u9451"      # three CJK characters found nowhere in T0892
SENSITIVE = ("harm", "sexual", "female_agent", "ritual", "flesh_food")
DEVIATING = {"no_counterpart", "abridged", "generalised", "substitution", "reversal", "category_name_omitted"}
CJK_RUN = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]{31,}")
TIBETAN_RUN = re.compile(r"[\u0f00-\u0fff]{61,}")


def h(*parts: object) -> int:
    """A stable pseudo-random number in [0, 1000) for the simulation's choices."""
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) % 1000


# --------------------------------------------------------------------------- the simulated truth
def simulated_truth(ctx: RunContext) -> dict[str, Any]:
    texts = load_texts(ctx)
    b0 = read_alignment(ctx.path("alignments", "dp_anchor.jsonl")).by_ref()
    content = [s for s in texts.witness if s.kind in CONTENT_KINDS]
    position = {s.id: i for i, s in enumerate(content)}
    local = {s.id: s.local_chapter for s in texts.witness}
    kind = {s.id: s.kind for s in texts.witness}
    units: dict[str, dict[str, Any]] = {}
    for u in texts.units:
        wits = list(b0[u.id].wit_ids) if u.id in b0 else []
        x = h(u.id, "truth")
        rel = "equivalent"
        for bound, name in ((50, "no_counterpart"), (80, "substitution"), (90, "reversal"), (150, "abridged"),
                            (200, "generalised"), (260, "paraphrase"), (275, "category_name_omitted")):
            if x < bound:
                rel = name
                break
        if not wits or rel == "no_counterpart":
            rel, wits = "no_counterpart", []
        elif 300 <= x < 400 and position[wits[-1]] + 1 < len(content):
            nxt = content[position[wits[-1]] + 1].id
            if local[nxt] == local[wits[-1]]:
                wits.append(nxt)                       # a longer link that B0 misses
        if u.kind == "mantra" and wits and all(kind[w] == "mantra" for w in wits):
            rel = "transliterated"
        units[u.id] = {"rel": rel, "wit": wits, "flip": rel == "reversal"}
    # the verified sentinel facts hold in the truth
    loaded = {s.id: s for s in sentinel_checks.load(ctx.data_dir / "sentinels" / "sentinels.yaml")}
    tx = sentinel_checks._texts([*texts.reference, *texts.witness])

    def span(sid: str, side: str) -> list[str]:
        return [s.id for s in sentinel_checks._range(tx, loaded[sid], side)]

    def spread(sid: str, rel: str, only_absent: bool = False) -> None:
        ref, wit = span(sid, "ref_"), span(sid, "wit_")
        for i, uid in enumerate(ref):
            if not only_absent or units[uid]["rel"] == "no_counterpart":
                units[uid] = {"rel": rel, "wit": [wit[min(len(wit) - 1, i * len(wit) // len(ref))]],
                              "flip": rel == "reversal"}

    spread("S3a_II9_opening_in_zh18", "equivalent")
    spread("S3c_II9_kill_to_subdue", "substitution")
    spread("S2a_II3_vows_reversed", "reversal")
    spread("S1a_I7_places_kept", "equivalent", only_absent=True)
    gathas = set(span("S4_zh_final_gathas_witness_only", "wit_"))
    for t in units.values():
        t["wit"] = [w for w in t["wit"] if w not in gathas]
        if not t["wit"]:
            t.update(rel="no_counterpart", flip=False)
    linked = {w for t in units.values() for w in t["wit"]}
    only = {s.id: ("addition" if s.kind in CONTENT_KINDS else "translator_note") for s in texts.witness
            if s.local_chapter and s.id not in linked and (s.kind in CONTENT_KINDS or s.kind == "note")}
    topics = {u.id: ["mantra_control"] if u.kind == "mantra" else
              [SENSITIVE[h(u.id, "t2") % 5]] if h(u.id, "topic") < 200 else ["neutral"] for u in texts.units}
    return {"units": units, "only": only, "topics": topics, "b0": {u: list(l.wit_ids) for u, l in b0.items()}}


def claude_says(truth: dict[str, Any], uid: str, tag: str) -> tuple[str, list[str], bool]:
    t = truth["units"][uid]
    rel, wits, flip = t["rel"], list(t["wit"]), t["flip"]
    x, b0 = h(uid, "claude"), truth["b0"].get(uid) or []
    if x < 30 and b0:
        rel, wits, flip = "equivalent", list(b0), False        # copies a B0 error
    elif x < 40:
        rel, wits, flip = "no_counterpart", [], False          # a false absence
    if h(uid, tag, "noise") < 20 and rel != "no_counterpart":
        rel, flip = ("paraphrase" if rel == "equivalent" else "equivalent"), False
    return rel, wits, flip


def collate_answer(truth: dict[str, Any], window, tag: str, text: dict[str, str]) -> dict[str, Any]:
    units = []
    for handle, uid in window.ref_handles.items():
        if h(uid, tag, "missing") == 999:
            continue                                            # an omitted unit (V1)
        rel, wits, flip = claude_says(truth, uid, tag)
        wits = [w for w in wits if w in window.handle_of]
        if not wits:
            rel, flip = "no_counterpart", False
        quote = text[wits[0]] if wits else ""
        if wits and h(uid, tag, "quote") < 3:
            quote = BAD_QUOTE                                   # fails V4
        units.append({"ref": handle, "wit": [window.handle_of[w] for w in wits], "relation": rel,
                      "polarity_flip": flip, "confidence": "high", "ref_quote": text[uid], "wit_quote": quote})
    only = [{"wit": [window.handle_of[s]], "kind": k, "wit_quote": text[s]} for s, k in truth["only"].items()
            if s in window.core and s in window.handle_of]
    return {"units": units, "witness_only": only}


def prefill(ctx: RunContext, truth: dict[str, Any]) -> None:
    """Write every T1 (collate and perturbation) and T3 answer into the response cache."""
    texts = load_texts(ctx)
    text = {s.id: s.text for s in (*texts.reference, *texts.witness)}
    cplan = plan_collation(ctx, texts=texts)
    jobs = {r.key(): (tag, w, cplan.windows.index(w)) for tag, w, r in cplan.requests()}

    def collate_script(req: LLMRequest):
        tag, w, n = jobs[req.key()]
        if (tag, n) in (("r2", 7), ("r2", 25)):
            return refusal_response(req)
        if (tag, n) == ("r3", 25):
            return truncated_response(req)
        return ok_response(req, collate_answer(truth, w, tag, text))

    cache = CachedClient(FakeClient(collate_script), ctx.settings.path("cache"))
    for _, _, r in cplan.requests():
        cache.complete(r)
    seed = int(ctx.settings.prereg["stats"]["seed"])
    base = plan_collation(ctx, ",".join(dev_chapters(ctx.settings.prereg)), 1, texts)
    order = texts.concordance.local_order[texts.witness_id]
    windows = [wrong_window(w, [far_chapter(w, order)])[0] for w in base.windows]
    windows += [delete_segments(w, 0.05, seed)[0] for w in base.windows]
    perturbed = {collator.build_request(w, base.settings, "r1", base.examples, base.template).key(): w
                 for w in windows}
    cache = CachedClient(FakeClient(lambda r: collate_answer(truth, perturbed[r.key()], "r1", text)),
                         ctx.settings.path("cache"))
    for w in windows:
        cache.complete(collator.build_request(w, base.settings, "r1", base.examples, base.template))
    settings = TopicTaskSettings.from_config(ctx.settings.llm)
    batches = plan_batches(list(texts.units), settings.units_per_call)
    requests = {build_request(b, load_codebook(ctx.data_dir / "codebook" / "topics.yaml"), settings).key(): b
                for b in batches}

    def topics_script(req: LLMRequest):
        rows = [{"ref": handle, "topics": [{"topic": t, "cue": "" if t == "neutral" else seg.text.split(TSHEG)[0]}
                                           for t in truth["topics"][seg.id]]}
                for handle, seg in requests[req.key()].handles().items()]
        return {"units": rows}

    cache = CachedClient(FakeClient(topics_script), ctx.settings.path("cache"))
    for b in batches:
        cache.complete(build_request(b, load_codebook(ctx.data_dir / "codebook" / "topics.yaml"), settings))


# --------------------------------------------------------------------------- the simulated annotators
def rewrite(path: Path, fill) -> None:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows, columns = list(reader), list(reader.fieldnames or ())
    for row in rows:
        fill(row)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def annotate_gold(truth: dict[str, Any], ref_sheet: Path, second: bool) -> None:
    wit_sheet = ref_sheet.with_name(ref_sheet.name.replace(".ref.csv", ".wit.csv"))
    with wit_sheet.open(encoding="utf-8-sig", newline="") as fh:
        handle = {r["seg_id"]: r["handle"] for r in csv.DictReader(fh)}
    order = {seg: i for i, seg in enumerate(handle)}
    spans: list[int] = []

    def fill(row: dict[str, str]) -> None:
        t = truth["units"][row["unit_id"]]
        rel = t["rel"]
        if second and h(row["unit_id"], "second") < 50 and rel in ("equivalent", "paraphrase"):
            rel = "paraphrase" if rel == "equivalent" else "equivalent"
        if not all(w in handle for w in t["wit"]):
            rel, wits = "unresolved", []
        else:
            wits = t["wit"]
        spans.extend(order[w] for w in wits)
        row.update(links=" ".join(handle[w] for w in wits), relation=rel, polarity_flip="1" if t["flip"] else "0")

    rewrite(ref_sheet, fill)
    lo, hi = (min(spans), max(spans)) if spans else (1, 0)
    rewrite(wit_sheet, lambda row: row.update(witness_only=truth["only"].get(row["seg_id"], ""))
            if lo <= order[row["seg_id"]] <= hi else None)


def annotate_blind(truth: dict[str, Any], sheet: Path) -> None:
    def fill(row: dict[str, str]) -> None:
        uid = row["unit_id"]
        if uid.startswith("+"):
            kind = truth["only"].get(uid[1:])
            row.update(blind_relation=kind or "has_counterpart", blind_wit_loci=uid[1:] if kind else "")
        else:
            t = truth["units"][uid]
            row.update(blind_relation=t["rel"], blind_wit_loci=" ".join(t["wit"]))

    rewrite(sheet, fill)


def annotate_reveal(sheet: Path) -> None:
    """The reviewer adopts some machine readings after the reveal (automation bias)."""
    def fill(row: dict[str, str]) -> None:
        machine = row["machine_relation"]
        if (machine and ":" not in machine and machine != row["blind_relation"] and h(row["unit_id"], "reveal") < 300
                and (row["machine_wit_loci"] or machine == "no_counterpart") and not row["unit_id"].startswith("+")):
            row.update(final_relation=machine, final_wit_loci=row["machine_wit_loci"], revised_reason="machine")

    rewrite(sheet, fill)


def annotate_topics(truth: dict[str, Any], sheet: Path) -> None:
    second = sheet.name.endswith(".second.csv")

    def fill(row: dict[str, str]) -> None:
        topics = truth["topics"][row["unit_id"]]
        if second and h(row["unit_id"], "coder2") < 50:
            topics = ["neutral"] if topics != ["neutral"] else ["ritual"]
        row["topics"] = ";".join(topics)

    rewrite(sheet, fill)


# --------------------------------------------------------------------------- the campaign
def make_root(tmp: Path, raw: Path) -> Path:
    root = tmp / "repo"
    shutil.copytree(REPO / "config", root / "config")
    shutil.copytree(REPO / "data", root / "data", ignore=shutil.ignore_patterns("raw", "reference", "__pycache__"))
    shutil.copytree(raw, root / "data" / "raw")
    path = root / "config" / "preregistration.yaml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    # With the committed cap (450) the census of machine positives (absences plus every
    # witness-only row, translator notes included) leaves no room for the sampled classes,
    # so their strata are never calibrated and G3 cannot pass; a researcher would amend the
    # cap before freezing. See the verification section of the preregistration.
    doc["verification"]["cap_units"] = 1500
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def campaign(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp = tmp_path_factory.mktemp("campaign")
    root = make_root(tmp, Path(os.environ["HEVAJRA_RAW_DIR"]))
    run = tmp / "run"
    levels: dict[str, int] = {}

    def cli(*args: str) -> None:
        assert main([*args, "--root", str(root), "--run-dir", str(run)]) == 0, args

    def level() -> int:
        return json.loads((run / "evaluation" / "gate.json").read_text(encoding="utf-8"))["level"]

    cli("ingest")
    cli("baselines")
    settings = load_settings(root)
    ctx = RunContext(settings.root, run, settings)
    truth = simulated_truth(ctx)
    prefill(ctx, truth)
    cli("collate", "--offline")
    cli("build")
    cli("sample", "windows")
    for gold_set in ("dev", "test", "test_second"):
        cli("review", "export", "--task", "gold", "--set", gold_set)
    for sheet in sorted((run / "review" / "gold").glob("*.ref.csv")):
        second = sheet.name.startswith("test_second")
        annotate_gold(truth, sheet, second)
        cli("review", "import", "--task", "gold", "--file", str(sheet), "--annotator", "b" if second else "a")
    cli("evaluate", "--gold", "dev")
    levels["dev"] = level()
    cli("perturb", "--offline")
    cli("prereg", "freeze")
    cli("evaluate")
    levels["test"] = level()
    cli("topics", "prelabel", "--offline")
    cli("topics", "export")
    annotate_topics(truth, run / "review" / "topics_first.csv")
    cli("topics", "import", "--file", str(run / "review" / "topics_first.csv"), "--annotator", "c1")
    cli("topics", "export", "--second")
    annotate_topics(truth, run / "review" / "topics_second.second.csv")
    cli("topics", "import", "--file", str(run / "review" / "topics_second.second.csv"), "--annotator", "c2")
    cli("sample", "verification")
    cli("sample", "audit")
    for task, sheet in (("verify", "verify"), ("resolve", "verify_resolve"), ("audit", "audit")):
        cli("review", "export", "--task", task)
        annotate_blind(truth, run / "review" / f"{sheet}.blind.csv")
        cli("review", "import", "--task", task, "--file", str(run / "review" / f"{sheet}.blind.csv"),
            "--annotator", "r")
    for batch in ("verify", "audit"):
        cli("review", "export", "--task", "reveal", "--batch", batch)
        annotate_reveal(run / "review" / f"{batch}.reveal.csv")
        cli("review", "reveal", "--file", str(run / "review" / f"{batch}.reveal.csv"), "--annotator", "r")
    cli("build")
    cli("evaluate")
    levels["reviewed"] = level()
    cli("stats")
    cli("report")
    return {"root": root, "run": run, "truth": truth, "levels": levels}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_report_level_climbs_as_the_gates_allow(campaign: dict[str, Any]) -> None:
    assert campaign["levels"] == {"dev": 0, "test": 1, "reviewed": 2}
    gate = _json(campaign["run"] / "evaluation" / "gate.json")
    assert gate["confirmatory"] is True, gate["scope_note"]
    assert gate["not_estimable"] == {"E3": "G4: no manuscript column; reference is the co-witness"}
    ledger = campaign["root"] / "data" / "ledger" / "test_evaluations.jsonl"
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1, "re-gating is not a new test scoring"
    assert "Report level 2 (CALIBRATED)" in (campaign["run"] / "summary.md").read_text(encoding="utf-8")


def test_perturbations_score_only_the_perturbed_windows(campaign: dict[str, Any]) -> None:
    p = _json(campaign["run"] / "evaluation" / "perturbations.json")
    assert p["wrong_window"]["hits"] == 0 and p["wrong_window"]["n"] > 0
    assert p["deletion"]["n"] > 0 and p["deletion"]["hits"] == p["deletion"]["n"]


def test_sentinels_at_every_stage(campaign: dict[str, Any]) -> None:
    results = [json.loads(line) for line in
               (campaign["run"] / "evaluation" / "sentinels.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["stage"] for r in results} == {"ingest", "proposal", "final"}
    assert all(r["passed"] for r in results if r["blocking"]), [r for r in results if r["blocking"] and not r["passed"]]


def test_counts_add_up_and_the_estimate_finds_the_truth(campaign: dict[str, Any]) -> None:
    run, truth = campaign["run"], campaign["truth"]
    cells = list(csv.DictReader((run / "matrix" / "cells.csv").open(encoding="utf-8")))
    units = [c for c in cells if c["row_type"] == "unit"]
    assert len(units) == len(truth["units"]) == 3042
    assert not [c for c in units if c["status"] == "UNALIGNED"], "every unresolved unit was resolved"
    est = _json(run / "stats" / "estimates.json")
    share = sum(1 for t in truth["units"].values() if t["rel"] in DEVIATING) / len(truth["units"])
    e1 = est["E1_any"]
    assert e1["not_estimable"] is None and e1["lo"] - 0.02 <= share <= e1["hi"] + 0.02, (share, e1)
    assert est["E4"]["not_estimable"] is None and est["E5"]["not_estimable"] is None


def test_no_source_text_in_the_committed_annotations(campaign: dict[str, Any]) -> None:
    data = campaign["root"] / "data"
    for path in [*(data / "annotations").rglob("*.csv"), *(data / "ledger").rglob("*.jsonl")]:
        text = path.read_text(encoding="utf-8-sig")
        assert not CJK_RUN.search(text) and not TIBETAN_RUN.search(text), path
