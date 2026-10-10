"""evaluate and perturb under collate.v2: the consensus of a v2 run gets convention scores
(with intervals) in ``scores.<set>.json``, every other source and a v1 consensus report
n/a, and a configured prompt version other than the collated one is refused."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from hevajra_matrix.cli import main
from hevajra_matrix.core.types import convention_of
from hevajra_matrix.llm.cache import CachedClient
from hevajra_matrix.llm.client import LLMRequest
from hevajra_matrix.llm.fake import FakeClient
from hevajra_matrix.pipeline import StageError, perturb, plan_collation
from hevajra_matrix.pipeline.instrument import check_prompt, collated_prompt
from hevajra_matrix.pipeline.measure import already_ledgered
from hevajra_matrix.pipeline.store import read_alignment

from test_pipeline_collate_v2 import select_prompt
from test_pipeline_support import context, echo_script, fill_cache, make_root

CONVENTION_METRICS = ("convention_precision", "convention_recall", "convention_f1", "convention_unsupported_rate",
                      "convention_precision:transcribed_name", "convention_recall:transcribed_name")


def echo_names(request: LLMRequest) -> dict:
    """``echo_script`` under a v2 schema: every second linked unit is a transcribed name."""
    answer = echo_script(request)
    if "convention" in request.schema["properties"]["units"]["items"]["properties"]:
        answer["units"] = [{**u, "convention": "transcribed_name" if u["wit"] and n % 2 else "none"}
                           for n, u in enumerate(answer["units"])]
    return answer


def cli(root: Path, run: Path, *args: str) -> int:
    return main([*args, "--root", str(root), "--run-dir", str(run), "--offline"])


def finished_run(tmp_path: Path, prompt: str | None) -> tuple[Path, Path]:
    """An offline run (collate, build, evaluate) with dev gold copied from the consensus,
    conventions included."""
    root, run = make_root(tmp_path), tmp_path / "run"
    assert main(["ingest", "--root", str(root), "--run-dir", str(run)]) == 0
    if prompt is None:
        fill_cache(context(root, run))
    else:
        select_prompt(root, prompt)
        ctx = context(root, run)
        cache = CachedClient(FakeClient(echo_names), ctx.settings.path("cache"))
        for _, _, request in plan_collation(ctx).requests():
            cache.complete(request)
    assert cli(root, run, "run") == 0
    assert cli(root, run, "sample", "windows") == 0
    assert cli(root, run, "review", "export", "--task", "gold", "--set", "dev") == 0
    links = read_alignment(run / "alignments" / "claude.jsonl").by_ref()
    for ref_sheet in sorted((run / "review" / "gold").glob("dev_*.ref.csv")):
        with ref_sheet.with_name(ref_sheet.name.replace(".ref.csv", ".wit.csv")).open(encoding="utf-8-sig") as fh:
            handle = {r["seg_id"]: r["handle"] for r in csv.DictReader(fh)}
        with ref_sheet.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            rows, columns = list(reader), reader.fieldnames
        for row in rows:
            link = links[row["unit_id"]]
            row.update(links=" ".join(handle[w] for w in link.wit_ids), relation=str(link.relation),
                       convention=convention_of(link.flags).value)
        with ref_sheet.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        assert cli(root, run, "review", "import", "--task", "gold", "--file", str(ref_sheet), "--annotator", "ann",
                   "--date", "2026-09-30") == 0
    return root, run


def dev_scores(root: Path, run: Path) -> dict:
    assert cli(root, run, "evaluate", "--gold", "dev") == 0
    return json.loads((run / "evaluation" / "scores.dev.json").read_text(encoding="utf-8"))["sources"]


@pytest.fixture(scope="module")
def v2_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return finished_run(tmp_path_factory.mktemp("v2"), "collate.v2")


def test_a_v2_consensus_is_scored_on_conventions_with_intervals(v2_run: tuple[Path, Path]) -> None:
    root, run = v2_run
    sources = dev_scores(root, run)
    claude = sources["claude:consensus"]
    assert claude["counts"]["gold_conventions"] > 0
    assert claude["counts"]["pred_conventions"] == claude["counts"]["gold_conventions"]
    for name in ("convention_precision", "convention_recall", "convention_f1", "relation_kappa",
                 "convention_precision:transcribed_name", "convention_recall:transcribed_name"):
        assert claude["metrics"][name] == 1.0, name
        assert claude["intervals"][name]["point"] == 1.0 and claude["intervals"][name]["lo"] == 1.0, name
    # echo_script quotes no reference text, so V12 could not confirm any transcribed name
    assert claude["metrics"]["convention_unsupported_rate"] == 1.0
    assert claude["metrics"]["convention_precision:standing_rendering"] is None
    for source in ("dp:zero", "dp:anchor", "placebo:shuffled"):
        assert source in sources, source
        assert all(sources[source]["metrics"][name] is None for name in CONVENTION_METRICS), source
        assert sources[source]["counts"]["pred_conventions"] is None
        assert sources[source]["metrics"]["dany_conv_kappa"] is not None, "measured on its own D_any"


def test_evaluate_and_perturb_refuse_another_prompt_version(v2_run: tuple[Path, Path], capsys) -> None:
    root, run = v2_run
    ctx = context(root, run)
    assert collated_prompt(ctx) == check_prompt(ctx) == "collate.v2"
    path = root / "config" / "llm.yaml"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("prompt: collate.v2}", "prompt: collate.v1}"), encoding="utf-8")
    try:
        capsys.readouterr()
        assert cli(root, run, "evaluate", "--gold", "dev") == 2
        assert "was collated with collate.v2" in capsys.readouterr().err
        with pytest.raises(StageError, match="tasks.collate.prompt: collate.v1"):
            perturb(context(root, run))
        assert cli(root, run, "evaluate", "--gold", "dev", "--baselines-only") == 0, "no consensus is scored"
    finally:
        path.write_text(text, encoding="utf-8")


def test_a_v1_consensus_reports_no_convention_metrics(tmp_path: Path) -> None:
    root, run = finished_run(tmp_path, None)
    assert json.loads((run / "collation" / "integrity.json").read_text())["convention_kappa"] is None
    ctx = context(root, run)
    assert collated_prompt(ctx) == "collate.v1"
    claude = dev_scores(root, run)["claude:consensus"]
    assert all(claude["metrics"][name] is None for name in CONVENTION_METRICS)
    assert claude["counts"]["gold_conventions"] == 0 and claude["counts"]["pred_conventions"] is None
    # gold annotated "none" everywhere: the two calibers coincide
    assert claude["metrics"]["dany_conv_kappa"] == claude["metrics"]["dany_kappa"]
    assert claude["metrics"]["relation_kappa"] == 1.0


def test_a_run_from_before_the_prompt_key_counts_as_v1(tmp_path: Path) -> None:
    root, run = make_root(tmp_path), tmp_path / "run"
    ctx = context(root, run)
    assert collated_prompt(ctx) is None and check_prompt(ctx) is None
    index = run / "collation" / "replicates.json"
    index.parent.mkdir(parents=True)
    index.write_text(json.dumps({"tags": ["r1"]}), encoding="utf-8")
    assert collated_prompt(ctx) == check_prompt(ctx) == "collate.v1"
    select_prompt(root, "collate.v2")
    with pytest.raises(StageError, match="was collated with collate.v1"):
        check_prompt(context(root, run))


def test_metrics_added_later_do_not_make_a_new_test_scoring(tmp_path: Path) -> None:
    """The ledger compares the metrics both records carry: a record written before the
    convention metrics existed still matches a re-gating of the same output."""
    old = {"instrument_digest": "d", "prereg_sha256": "p", "gate": True,
           "metrics": {"claude:consensus": {"link_f1": 0.8, "refusal_rate:sensitive": 0.1}}}
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps(old) + "\n", encoding="utf-8")
    new = {**old, "metrics": {"claude:consensus": {"link_f1": 0.8, "refusal_rate:sensitive": 0.3,
                                                   "convention_precision": None, "relation_kappa": 0.7}}}
    assert already_ledgered(ledger, new)
    assert not already_ledgered(ledger, {**new, "metrics": {"claude:consensus": {"link_f1": 0.7}}})
    assert not already_ledgered(ledger, {**new, "instrument_digest": "other"})
    assert not already_ledgered(ledger, {**new, "metrics": {**new["metrics"], "dp:zero": {"link_f1": 0.5}}})


def test_integrity_records_the_replicate_agreement_on_conventions(v2_run: tuple[Path, Path]) -> None:
    """Descriptive (no gate): identical replicates agree perfectly on their conventions."""
    _, run = v2_run
    integrity = json.loads((run / "collation" / "integrity.json").read_text())
    assert integrity["replicates"] > 1 and integrity["convention_kappa"] == 1.0
