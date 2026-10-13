"""``hevajra-matrix adopt`` end to end: a collate.v1 run and a collate.v2 run of the mini texts,
collated offline (FakeClient answers from the cache) and built, a small dev gold written from
the v2 consensus over the preregistered dev regions; adoption.json in the v2 run, the summary,
the recorded stage, the exit status, the floor and kappas wiring, "not ready" (a blocked or
unmarked ``?``, a dev region not in the gold) and the refusals (wrong version, another text of
the version, other chapters or replicates, no consensus, no dev gold, --run-dir)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import yaml

from hevajra_matrix.cli import main
from hevajra_matrix.core.types import convention_of
from hevajra_matrix.evaluation import adoption as A
from hevajra_matrix.evaluation import gold as G
from hevajra_matrix.llm.cache import CachedClient
from hevajra_matrix.llm.client import LLMRequest
from hevajra_matrix.llm.fake import FakeClient
from hevajra_matrix.matrix.build import segment_fingerprint
from hevajra_matrix.pipeline import plan_collation
from hevajra_matrix.pipeline.context import load_texts
from hevajra_matrix.pipeline.store import read_alignment

from test_pipeline_collate_v2 import select_prompt
from test_pipeline_support import MINI_PREREG, context, echo_script, fill_cache, make_root

KONGZHI = "\u7a7a\u667a"     # note marker of the Hevajra glossary entry: blocks
HALF = "\u534a\u97f3"        # note marker of Q7 (a part-transcribed name): reads transcribed_name
EXPLAINED = "\u91ca\u8bcd"   # note marker of Q8 (an explained word): reads none
CLASS_NOUN = "\u7c7b\u540d"  # note marker of Q3 (a class noun): reads none

# The dev regions of the test project: every chapter of the mini texts, so that both runs collate
# exactly the dev chapters. The tuning window I.1 holds the first two units of chapter I.1 here,
# so that the units outside it include a transcribed name and shared units of two relations.
DEV_REGIONS = [{"chapter": "I.1", "first_units": 2, "id": "I.1"}, {"chapter": "I.1", "last_units": 2, "id": "I.1-tail"},
               {"chapter": "I.2"}, {"chapter": "II.1"}, {"chapter": "II.2"}]
WINDOWS = (("I.1", 2), ("I.1-tail", 2), ("I.2", 2), ("II.1", 3), ("II.2", 1))   # region id -> units


def echo_v2(request: LLMRequest) -> dict:
    """``echo_script`` under the v2 schema: of the linked units, the second of every three is a
    paraphrase and the third a transcribed name."""
    answer = echo_script(request)
    answer["units"] = [{**u, "relation": "paraphrase" if u["wit"] and n % 3 == 1 else u["relation"],
                        "convention": "transcribed_name" if u["wit"] and n % 3 == 2 else "none"}
                       for n, u in enumerate(answer["units"])]
    return answer


def cli(root: Path, *args: str) -> int:
    return main([*args, "--root", str(root), "--offline"])


def adopt(root: Path, v1: Path, v2: Path) -> int:
    return cli(root, "adopt", "--v1-run", str(v1), "--v2-run", str(v2))


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """(root, v1 run, v2 run), each collated with 3 replicates and built; config selects v2."""
    tmp = tmp_path_factory.mktemp("adopt")
    root, v1, v2 = make_root(tmp, {**MINI_PREREG, "dev_regions": DEV_REGIONS}), tmp / "v1", tmp / "v2"
    for run in (v1, v2):
        assert main(["ingest", "--root", str(root), "--run-dir", str(run)]) == 0
    fill_cache(context(root, v1))
    assert cli(root, "collate", "--run-dir", str(v1)) == 0 and cli(root, "build", "--run-dir", str(v1)) == 0
    select_prompt(root, "collate.v2")
    ctx = context(root, v2)
    cache = CachedClient(FakeClient(echo_v2), ctx.settings.path("cache"))
    for _, _, request in plan_collation(ctx).requests():
        cache.complete(request)
    assert cli(root, "collate", "--run-dir", str(v2)) == 0 and cli(root, "build", "--run-dir", str(v2)) == 0
    return root, v1, v2


def write_gold(root: Path, run: Path, cells: dict[str, tuple[str, str]] | None = None,
               leave_out: str = "") -> G.GoldSet:
    """Dev gold copied from the run's consensus over the dev regions (``leave_out``: a region not
    imported yet); ``cells``: unit -> (convention, note) written instead of the consensus
    convention."""
    texts = load_texts(context(root, run))
    links = read_alignment(run / "alignments" / "claude.jsonl").by_ref()
    regions = G.dev_region_units(list(texts.units), DEV_REGIONS)
    assert [(rid, len(units)) for rid, units in regions.items()] == list(WINDOWS)
    assert [s.id for s in texts.units] == [u for units in regions.values() for u in units], "every unit is dev"
    by_id = {s.id: s for s in texts.units}
    rows, windows = [], []
    for window_id, units in regions.items():
        if window_id == leave_out:
            continue
        segments = [by_id[u] for u in units]
        windows.append(G.TestWindow(window_id, units[0], units[-1], len(units), str(segments[0].chapter)))
        for seg in segments:
            link = links[seg.id]
            convention, note = (cells or {}).get(seg.id, (convention_of(link.flags).value, ""))
            rows.append(G.GoldRow(set_name="dev", window_id=window_id, row_type="ref", unit_id=seg.id,
                                  fingerprint=segment_fingerprint(seg), wit_ids=link.wit_ids,
                                  relation=str(link.relation), convention=convention, note=note, annotator="ann",
                                  date="2026-10-13"))
    gold = G.GoldSet("dev", texts.witness_id, tuple(rows), tuple(windows), texts.reference_id)
    path = root / "data" / "annotations" / "gold" / texts.witness_id / "dev.csv"
    G.save(gold, path)
    G.write_windows(windows, path.with_name("windows.csv"))
    return gold


def first_unit(gold: G.GoldSet, window: str) -> str:
    return next(r.unit_id for r in gold.rows if r.window_id == window)


def adoption(run: Path) -> dict:
    return json.loads((run / "evaluation" / "adoption.json").read_text(encoding="utf-8"))


def copy_run(run: Path, to: Path, edit=None) -> Path:
    """A copy of ``run`` at ``to``; ``edit(replicates_index, integrity)`` may change its
    ``collation/replicates.json`` and ``collation/integrity.json`` in place."""
    shutil.copytree(run, to)
    if edit is not None:
        paths = [to / "collation" / name for name in ("replicates.json", "integrity.json")]
        docs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        edit(*docs)
        for path, doc in zip(paths, docs):
            path.write_text(json.dumps(doc), encoding="utf-8")
    return to


def test_adopt_writes_the_decision_into_the_v2_run(runs, capsys) -> None:
    root, v1, v2 = runs
    gold = write_gold(root, v2)
    outside = [r for r in gold.rows if r.window_id not in A.TUNING_WINDOWS and r.relation != "no_counterpart"]
    assert sorted((r.relation, r.convention) for r in outside) == [
        ("equivalent", "none"), ("equivalent", "transcribed_name"), ("paraphrase", "none")], "the fake's pattern"
    write_gold(root, v2, {first_unit(gold, "I.1"): ("", f"{EXPLAINED} x")})
    capsys.readouterr()
    assert adopt(root, v1, v2) == 0
    out = capsys.readouterr().out
    doc = adoption(v2)
    decision = doc["decision"]
    assert decision["decision"] == A.ADOPT and decision["not_ready"] == [] and decision["note"] is None
    kappa, precision, floor = (decision["conditions"][c] for c in
                               ("relation_kappa_shared", "convention_precision", "replicate_kappa"))
    assert (kappa["v1"], kappa["v2"], kappa["passed"]) == (0.0, 1.0, True), "v1 never paraphrases"
    assert (precision["v2"], precision["v2_without_in_sample"], precision["passed"]) == (1.0, 1.0, True)
    integrity = [json.loads((run / "collation" / "integrity.json").read_text())["fleiss_kappa"] for run in (v1, v2)]
    assert [floor["v1"], floor["v2"], floor["threshold"], floor["passed"]] == [*integrity, 0.8, True]
    assert decision["sensitivity"]["decision"] == A.ADOPT
    figures = doc["figures"]
    assert figures["v2"]["deferred"] == {KONGZHI: 0, HALF: 0, EXPLAINED: 1, CLASS_NOUN: 0}
    assert figures["v2"]["unmarked"] == [] and figures["v2"]["blocked"] == []
    assert figures["v1"]["conventions"] is False and figures["v2"]["conventions"] is True
    default, excluded = (figures["v2"]["readings"][r][A.ALL] for r in (A.DEFAULT, A.EXCLUDED))
    assert default["convention_annotated"] == excluded["convention_annotated"] + 1, "the '?' read as its default"
    assert default["shared_units"] == excluded["shared_units"] + 1
    outside_default = figures["v2"]["readings"][A.DEFAULT][A.OUTSIDE]
    assert outside_default["windows"] == ["I.1-tail", "I.2", "II.1", "II.2"] and outside_default["units"] == 8
    assert doc["gold"]["missing"] == {} and doc["gold"]["dev_regions"] == [w for w, _ in WINDOWS]
    assert doc["gold"]["dev_chapters"] == ["I.1", "I.2", "II.1", "II.2"]
    assert doc["runs"]["v1"]["prompt"] == "collate.v1" and doc["runs"]["v2"]["replicates"] == 3
    assert doc["runs"]["v2"]["chapters"] == doc["gold"]["dev_chapters"]
    assert doc["relation_kappa_shared_difference"]["point"] == 1.0
    assert "adopt" in json.loads((v2 / "manifest.json").read_text())["stages"]
    assert f"decision: {A.ADOPT}" in out and "(1) relation_kappa_shared outside I.1: v2 1.000 >= v1 0.000" in out


def test_config_need_not_match_either_run(runs) -> None:
    root, v1, v2 = runs
    write_gold(root, v2)
    path = root / "config" / "llm.yaml"
    text = path.read_text(encoding="utf-8")
    try:
        path.write_text(text.replace("prompt: collate.v2}", "prompt: collate.v1}"), encoding="utf-8")
        assert adopt(root, v1, v2) == 0 and adoption(v2)["decision"]["decision"] == A.ADOPT
    finally:
        path.write_text(text, encoding="utf-8")


def test_the_floor_comes_from_the_preregistration_and_each_kappa_from_its_run(runs, tmp_path: Path) -> None:
    root, v1, v2 = runs
    write_gold(root, v2)
    one = copy_run(v1, tmp_path / "v1", lambda _, integrity: integrity.update(fleiss_kappa=0.9))
    two = copy_run(v2, tmp_path / "v2", lambda _, integrity: integrity.update(fleiss_kappa=0.85))
    path = root / "config" / "preregistration.yaml"
    text = path.read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    doc["gates"]["g2"]["min_replicate_kappa"] = 0.88
    try:
        path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        assert adopt(root, one, two) == 0
    finally:
        path.write_text(text, encoding="utf-8")
    decision = adoption(two)["decision"]
    assert decision["conditions"]["replicate_kappa"] == {"v1": 0.9, "v2": 0.85, "threshold": 0.88, "applies": True,
                                                         "passed": False}
    assert decision["decision"] == A.KEEP and decision["rule"]["min_replicate_kappa"] == 0.88


def test_a_unit_blocked_by_the_glossary_marker_is_not_ready(runs, capsys) -> None:
    root, v1, v2 = runs
    blocked = first_unit(write_gold(root, v2), "I.2")
    write_gold(root, v2, {blocked: ("", f"{KONGZHI} {HALF}")})
    capsys.readouterr()
    assert adopt(root, v1, v2) == 1, "exit 1: the rule cannot decide yet"
    doc = adoption(v2)
    assert doc["decision"]["decision"] == A.NOT_READY and blocked in doc["decision"]["not_ready"][0]
    assert doc["figures"]["v2"]["blocked"] == [blocked] and doc["figures"]["v2"]["deferred"][KONGZHI] == 1
    assert "decision: not ready" in capsys.readouterr().out


def test_an_unmarked_question_mark_or_a_dev_region_not_in_the_gold_is_not_ready(runs) -> None:
    root, v1, v2 = runs
    unmarked = first_unit(write_gold(root, v2), "I.2")
    write_gold(root, v2, {unmarked: ("", "see T:1.1")})
    assert adopt(root, v1, v2) == 1
    doc = adoption(v2)
    assert doc["figures"]["v2"]["unmarked"] == [unmarked]
    assert doc["decision"]["decision"] == A.NOT_READY
    assert any("no known marker" in r and unmarked in r for r in doc["decision"]["not_ready"])
    write_gold(root, v2, leave_out="II.2")
    assert adopt(root, v1, v2) == 1
    doc = adoption(v2)
    assert doc["gold"]["missing"] == {"II.2": 1} and doc["decision"]["decision"] == A.NOT_READY
    assert doc["decision"]["not_ready"] == [
        "the dev gold does not cover every dev region yet: II.2 (1 unit(s) missing)"]


def test_adopt_refuses_runs_of_the_other_version(runs, capsys) -> None:
    root, v1, v2 = runs
    write_gold(root, v2)
    capsys.readouterr()
    assert adopt(root, v2, v1) == 2
    assert f"--v1-run {v2} was collated with collate.v2, not collate.v1" in capsys.readouterr().err
    assert adopt(root, v1, v1) == 2
    assert f"--v2-run {v1} was collated with collate.v1, not collate.v2" in capsys.readouterr().err


@pytest.mark.parametrize("edit", [lambda index, _: index.update(prompt_sha="0" * 64),
                                  lambda index, _: index.pop("schema_sha")], ids=["revised text", "not recorded"])
def test_adopt_refuses_a_run_of_another_text_of_its_version(runs, tmp_path: Path, capsys, edit) -> None:
    root, v1, v2 = runs
    write_gold(root, v2)
    stale = copy_run(v2, tmp_path / "stale", edit)
    capsys.readouterr()
    assert adopt(root, v1, stale) == 2
    assert f"--v2-run {stale} was not collated with the current collate.v2 text" in capsys.readouterr().err


@pytest.mark.parametrize("option", [("--chapters", "I.1,I.2"), ("--replicates", "2")])
def test_adopt_refuses_a_run_of_other_chapters_or_replicates(runs, tmp_path: Path, capsys, option) -> None:
    root, v1, v2 = runs
    write_gold(root, v2)
    part = tmp_path / "part"
    assert main(["ingest", "--root", str(root), "--run-dir", str(part)]) == 0
    assert cli(root, "collate", "--run-dir", str(part), *option) == 0, "answered from the v2 cache"
    assert cli(root, "build", "--run-dir", str(part)) == 0
    capsys.readouterr()
    assert adopt(root, v1, part) == 2
    assert "the rule compares runs of the preregistered dev chapters ['I.1', 'I.2', 'II.1', 'II.2'], collated " \
           "whole with 3 replicates" in capsys.readouterr().err


def test_adopt_refuses_a_run_without_consensus(runs, tmp_path: Path, capsys) -> None:
    root, v1, v2 = runs
    write_gold(root, v2)
    unbuilt = copy_run(v1, tmp_path / "unbuilt")
    (unbuilt / "alignments" / "claude.jsonl").unlink()
    capsys.readouterr()
    assert adopt(root, unbuilt, v2) == 2
    assert "run `hevajra-matrix build` first" in capsys.readouterr().err


def test_adopt_refuses_without_dev_gold_and_with_a_run_dir(runs, capsys) -> None:
    root, v1, v2 = runs
    write_gold(root, v2)
    capsys.readouterr()
    assert cli(root, "adopt", "--v1-run", str(v1), "--v2-run", str(v2), "--run-dir", str(v2)) == 2
    assert "--run-dir does not apply" in capsys.readouterr().err
    next((root / "data" / "annotations" / "gold").glob("*/dev.csv")).unlink()
    assert adopt(root, v1, v2) == 2
    assert "no dev gold for zh_T0892_song yet" in capsys.readouterr().err
