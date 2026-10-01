"""Stage behaviour on the mini fixtures: client composition, skipping collate without a key,
the dry-run cost projection, baselines-only evaluation on dev gold, external imports,
fetch, and the experiment's refusal to run on an empty item bank."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hevajra_matrix.align.external import write_tsv
from hevajra_matrix.cli import main
from hevajra_matrix.core.io import read_jsonl
from hevajra_matrix.evaluation import gold as gold_sets
from hevajra_matrix.llm.anthropic_client import AnthropicClient
from hevajra_matrix.llm.audit import AuditedClient
from hevajra_matrix.llm.cache import CachedClient
from hevajra_matrix.llm.client import LLMRequest
from hevajra_matrix.pipeline import StageError, make_client
from hevajra_matrix.pipeline.experiment import outcome_from_record
from hevajra_matrix.pipeline.instrument import estimate_tokens, far_chapter, project_cost
from hevajra_matrix.pipeline.store import read_alignment
from hevajra_matrix.pipeline.texts import fetch
from hevajra_matrix.experiments.overattribution.score import TrialOutcome, outcome_record

from test_pipeline_support import context, make_root

PRICING = {"input": 4.0, "output": 20.0, "cache_read": 0.2, "cache_write": 5.0}


@pytest.fixture
def ingested(tmp_path: Path) -> tuple[Path, Path]:
    root, run = make_root(tmp_path), tmp_path / "run"
    assert main(["ingest", "--root", str(root), "--run-dir", str(run)]) == 0
    return root, run


def cli(root: Path, run: Path, *args: str) -> int:
    return main([*args, "--root", str(root), "--run-dir", str(run)])


# --------------------------------------------------------------------------- client composition
def test_make_client_composes_audit_cache_and_sdk_client(ingested: tuple[Path, Path]) -> None:
    ctx = context(*ingested)
    client = make_client(ctx)
    assert isinstance(client, AuditedClient) and isinstance(client.inner, CachedClient)
    assert isinstance(client.inner.inner, AnthropicClient) and client.inner.inner.model == "claude-opus-5-5"
    assert client.inner.inner._sdk is None, "the SDK client is created lazily, on the first call"
    assert client.inner.root == ctx.settings.path("cache") and not client.inner.offline
    assert client.log_path == ctx.run_dir / "llm_audit.jsonl"
    assert client.budget_usd == 300 and client.run_id == ctx.run_dir.name


def test_offline_client_never_builds_an_anthropic_client(ingested: tuple[Path, Path], monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("AnthropicClient constructed offline")

    monkeypatch.setattr("hevajra_matrix.pipeline.context.AnthropicClient", forbidden)
    client = make_client(context(*ingested, offline=True))
    assert client.inner.offline and not isinstance(client.inner.inner, AnthropicClient)


# --------------------------------------------------------------------------- no key
def test_collate_without_key_or_offline_says_what_to_do(ingested, monkeypatch, capsys) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert cli(*ingested, "collate") == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_run_without_key_skips_collate_and_still_reports(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    root, run = make_root(tmp_path), tmp_path / "run"
    assert cli(root, run, "run") == 0
    out = capsys.readouterr().out
    assert "collate: skipped (no ANTHROPIC_API_KEY and not --offline)" in out
    assert "stats: skipped" in out
    text = (run / "summary.md").read_text(encoding="utf-8")
    assert "No instrument output in this run" in text and "| outcome | P1 length-only DP | B0 anchor DP |" in text
    assert not (run / "llm_audit.jsonl").exists()


# --------------------------------------------------------------------------- dry run
def test_dry_run_projects_cost_with_the_heuristic_and_sends_nothing(ingested, monkeypatch, capsys) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    root, run = ingested
    assert cli(root, run, "collate", "--dry-run", "--replicates", "2") == 0
    result = json.loads((run / "collation" / "dry_run.json").read_text(encoding="utf-8"))
    assert result["token_count_method"] == "heuristic" and result["calls"] == 2 * result["windows"]
    assert result["projected_usd"] > 0 and result["calls_to_send"] == result["calls"]
    assert "nothing was sent" in capsys.readouterr().out
    assert not (run / "llm_audit.jsonl").exists() and not (root / "runs" / "llm-cache").exists()


def test_dry_run_uses_count_tokens_when_a_key_is_present(ingested, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    calls = []

    def count(self, request):
        calls.append(request.key())
        return 1000

    monkeypatch.setattr(AnthropicClient, "count_tokens", count)
    monkeypatch.setattr(AnthropicClient, "complete", lambda self, r: pytest.fail("dry run sent a request"))
    root, run = ingested
    assert cli(root, run, "collate", "--dry-run") == 0
    result = json.loads((run / "collation" / "dry_run.json").read_text(encoding="utf-8"))
    assert result["token_count_method"] == "count_tokens"
    assert result["input_tokens"] == 1000 * result["calls"], "replicates share one count per window"
    assert len(set(calls)) == result["windows"]


def _request(body: str, replicate: str = "r1") -> LLMRequest:
    return LLMRequest(task="collate", prompt_sha="p", system="S" * 300, context="C" * 700, body=body,
                      schema={"type": "object"}, effort="high", max_tokens=10, replicate=replicate)


def test_project_cost_writes_the_prefix_once_and_skips_cached_calls() -> None:
    reqs = [_request("B" * 1000), _request("B" * 1000, "r2"), _request("X" * 1000)]
    result = project_cost(reqs, PRICING, count=lambda r: 2000, cached=lambda r: r.replicate == "r2",
                          output_tokens=100)
    assert result["calls"] == 3 and result["cached_calls"] == 1 and result["input_tokens"] == 4000
    # prefix = 1000 of 2000 tokens; first call writes it, the second reads it
    expected = (1000 * 5.0 + 1000 * 4.0 + 100 * 20.0 + 1000 * 0.2 + 1000 * 4.0 + 100 * 20.0) / 1e6
    assert result["projected_usd"] == round(expected, 2)


def test_estimate_tokens_is_conservative_for_non_latin_script() -> None:
    assert estimate_tokens("abcd" * 10) == 10
    assert estimate_tokens("\u0f40\u0f0b\u4e00") == 3  # Tibetan ka, tsheg, CJK one


# --------------------------------------------------------------------------- controls and dev gold
def test_baselines_import_adds_an_external_control(ingested: tuple[Path, Path], tmp_path: Path) -> None:
    root, run = ingested
    assert cli(root, run, "baselines") == 0
    tsv = tmp_path / "mitra.tsv"
    write_tsv(read_alignment(run / "alignments" / "dp_zero.jsonl"), tsv)
    assert cli(root, run, "baselines", "import", "--tsv", str(tsv), "--name", "mitra-e") == 0
    imported = read_alignment(run / "alignments" / "external_mitra-e.jsonl")
    assert imported.source == "external:mitra-e" and len(imported.by_ref()) == 10
    bad = tmp_path / "bad.tsv"
    bad.write_text("ref_id\twit_ids\nD417:9a.1.1\t\n", encoding="utf-8")
    assert cli(root, run, "baselines", "import", "--tsv", str(bad), "--name", "bad") == 2


def test_dev_gold_round_trip_and_baselines_only_evaluation(ingested, monkeypatch, capsys) -> None:
    """sample windows -> gold sheets -> filled from P1 -> import -> evaluate --gold dev --baselines-only."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    root, run = ingested
    assert cli(root, run, "baselines") == 0
    assert cli(root, run, "sample", "windows") == 0
    gold_dir = root / "data" / "annotations" / "gold" / "zh_T0892_song"
    windows = gold_sets.read_windows(gold_dir / "windows.csv")
    assert [w.window_id for w in windows if w.seed is None] == ["I.1"] and len(windows) == 2
    assert cli(root, run, "sample", "windows") == 2, "windows are drawn once"
    assert cli(root, run, "review", "export", "--task", "gold", "--set", "dev") == 0
    ref_sheet = run / "review" / "gold" / "dev_I.1.ref.csv"
    _fill_gold_from(ref_sheet, read_alignment(run / "alignments" / "dp_zero.jsonl"))
    assert cli(root, run, "review", "import", "--task", "gold", "--file", str(ref_sheet), "--annotator", "ann",
               "--date", "2026-09-30") == 0
    dev = gold_sets.load(gold_dir / "dev.csv")
    assert len(dev.units()) == 4
    capsys.readouterr()
    assert cli(root, run, "evaluate", "--gold", "dev", "--baselines-only") == 0
    scores = json.loads((run / "evaluation" / "scores.json").read_text(encoding="utf-8"))
    assert set(scores["sources"]) == {"dp:zero", "dp:anchor"} and scores["baselines_only"]
    assert scores["sources"]["dp:zero"]["metrics"]["link_f1"] == 1.0
    assert not (root / "data" / "ledger" / "test_evaluations.jsonl").exists(), "dev gold is never ledgered"
    assert "evaluate dev dp:zero: 4 units, link F1 1.000" in capsys.readouterr().out


def _fill_gold_from(ref_sheet: Path, alignment) -> None:
    """Answer a gold sheet with the links of ``alignment`` (handles from the wit sheet)."""
    import csv

    wit_sheet = ref_sheet.with_name(ref_sheet.name.replace(".ref.csv", ".wit.csv"))
    with wit_sheet.open(encoding="utf-8-sig", newline="") as fh:
        handle = {r["seg_id"]: r["handle"] for r in csv.DictReader(fh)}
    with ref_sheet.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows, columns = list(reader), reader.fieldnames
    links = alignment.by_ref()
    for row in rows:
        link = links[row["unit_id"]]
        row["links"] = " ".join(handle[w] for w in link.wit_ids)
        row["relation"] = str(link.relation)
    with ref_sheet.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- perturbation helper
def test_far_chapter_picks_the_most_distant_present_chapter() -> None:
    order = [f"pin{i}" for i in range(1, 8)]
    window = SimpleNamespace(key="w", core_locals=frozenset({"pin2", "pin3"}),
                             text=[SimpleNamespace(local_chapter=c) for c in ("pin1", "pin2", "pin3", "pin6", None)])
    assert far_chapter(window, order) == "pin6"
    lonely = SimpleNamespace(key="w", core_locals=frozenset({"pin1"}), text=[SimpleNamespace(local_chapter="pin1")])
    with pytest.raises(StageError, match="outside its core window"):
        far_chapter(lonely, order)


# --------------------------------------------------------------------------- fetch, experiment, check
def test_fetch_records_sha256_and_keeps_existing_files(tmp_path: Path) -> None:
    seen: list[str] = []

    def opener(url: str) -> bytes:
        seen.append(url)
        return url.encode("ascii")

    manifest = fetch(tmp_path / "raw", opener=opener)
    assert len(seen) == 2 and set(manifest) == {"T18n0892.xml", "derge_rgyud_bum_nga.txt"}
    assert all(len(m["sha256"]) == 64 and m["url"].startswith("https://") for m in manifest.values())
    fetch(tmp_path / "raw", opener=opener)
    assert len(seen) == 2, "existing files are not downloaded again"
    on_disk = json.loads((tmp_path / "raw" / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["T18n0892.xml"]["sha256"] == manifest["T18n0892.xml"]["sha256"]


def test_experiment_refuses_an_empty_item_bank(ingested, capsys) -> None:
    assert cli(*ingested, "experiment", "overattribution", "plan") == 2
    assert "fill the item bank" in capsys.readouterr().err


def test_outcome_records_round_trip() -> None:
    outcome = TrialOutcome(trial_id="i1:E0:r1", item_id="i1", arm="sensitive", condition="E0", evidence="none",
                           replicate="r1", omission_origin="real", status="ok", scorer_status="ok",
                           stances={"content_motive": "asserted"}, y_over=True, flags=frozenset({"x"}))
    record = json.loads(json.dumps(outcome_record(outcome)))
    assert outcome_from_record(record) == outcome


def test_claude_check_is_refused_offline(ingested, capsys) -> None:
    assert main(["claude-check", "--offline", "--root", str(ingested[0])]) == 2
    assert "cannot run --offline" in capsys.readouterr().err


def test_stage_without_its_input_names_the_producing_command(ingested, capsys) -> None:
    assert cli(*ingested, "build") == 2
    assert "hevajra-matrix collate" in capsys.readouterr().err
    assert read_jsonl(ingested[1] / "ingest" / "sentinels.jsonl")


def test_experiment_score_analyses_recorded_responses(ingested) -> None:
    from hevajra_matrix.core.io import write_jsonl
    from hevajra_matrix.pipeline.experiment import experiment_score

    root, run = ingested
    stances = {"source_text": "not_mentioned", "shared_tradition": "not_mentioned",
               "transmission_loss": "not_mentioned", "abridgement": "not_mentioned",
               "content_motive": "asserted", "external_pressure": "not_mentioned"}
    records = []
    for pair in range(6):
        for arm in ("sensitive", "neutral"):
            item = f"{arm[0]}{pair}"
            for condition in ("E0", "EP", "EW"):
                over = arm == "sensitive" and condition != "EW"
                records.append(outcome_record(TrialOutcome(
                    trial_id=f"{item}:{condition}:r1", item_id=item, arm=arm, condition=condition,
                    evidence="none", replicate="r1", omission_origin="real", status="ok", scorer_status="ok",
                    served_model="claude-opus-5-5", explanation=f"explanation {item} {condition}",
                    primary="content_motive" if over else "source_text", stances=stances, disputes_premise=False,
                    y_over=over, y_any=over, y_uptake=not over, premise_ok=True)))
    write_jsonl(run / "experiments" / "overattribution" / "responses.jsonl", records)
    payload = experiment_score(context(root, run), n_boot=50)
    results = json.loads((run / "experiments" / "overattribution" / "results.json").read_text(encoding="utf-8"))
    assert results == json.loads(json.dumps(payload, default=list))
    h1 = next(t for t in results["tests"] if t["name"] == "H1")
    assert h1["role"] == "confirmatory" and h1["estimate"] == 1.0
    assert (run / "experiments" / "overattribution" / "human_coding_sheet.csv").is_file()


ITEMS = """item_id,phase,arm,pair_id,chapter,unit_ids,omission_origin,zh_context_from,zh_context_to,synthetic_facts,note
s1,main,sensitive,p1,I.1,D417:1b.2.1,constructed,T0892:0587c13.1,T0892:0587c13.3,true,
n1,main,neutral,p1,I.1,D417:1b.1.6,real,T0892:0587c11.1,T0892:0587c11.2,true,
"""


def _experiment_script(request):
    from hevajra_matrix.llm.fake import ok_response

    if request.task == "subject":
        return ok_response(request, {"explanation": "The source text may differ here.",
                                     "most_likely": "source_text_differs", "premise_ok": True})
    stances = dict.fromkeys(("source_text", "shared_tradition", "transmission_loss", "abridgement",
                             "content_motive", "external_pressure"), "not_mentioned")
    return ok_response(request, {"stances": {**stances, "source_text": "hypothesised"}, "primary": "source_text",
                                 "disputes_premise": False, "motive_quote": ""})


def test_experiment_plan_and_run_on_a_tiny_bank(ingested, monkeypatch) -> None:
    from hevajra_matrix.llm.fake import FakeClient

    root, run = ingested
    (root / "data" / "experiments" / "overattribution" / "items.csv").write_text(ITEMS, encoding="utf-8")
    assert cli(root, run, "experiment", "overattribution", "plan") == 0
    trials = read_jsonl(run / "experiments" / "overattribution" / "trials.jsonl")
    assert len(trials) == 2 * 3 * 3 and sorted(t["order"] for t in trials) == list(range(18))
    fake = FakeClient(_experiment_script)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("hevajra_matrix.pipeline.experiment.make_client", lambda ctx: fake)
    assert cli(root, run, "experiment", "overattribution", "run") == 0
    responses = read_jsonl(run / "experiments" / "overattribution" / "responses.jsonl")
    assert len(responses) == 18 and all(r["status"] == "ok" and r["scorer_status"] == "ok" for r in responses)
    assert {r.task for r in fake.requests} == {"subject", "scorer"}
    assert not any(r.allow_fallback for r in fake.requests), "fallback is off for the experiment"


def test_components_stage_writes_codes_profile_and_diagnostics(tmp_path: Path, monkeypatch) -> None:
    from hevajra_matrix.llm.fake import FakeClient

    from test_pipeline_support import fill_cache

    root, run = make_root(tmp_path), tmp_path / "run"
    assert cli(root, run, "ingest") == 0
    fill_cache(context(root, run))
    assert main(["run", "--offline", "--root", str(root), "--run-dir", str(run)]) == 0
    fake = FakeClient(lambda request: {"pairs": []})
    monkeypatch.setattr("hevajra_matrix.pipeline.instrument.make_client", lambda ctx: fake)
    assert main(["components", "--offline", "--root", str(root), "--run-dir", str(run)]) == 0
    out = run / "components"
    assert (out / "components.jsonl").is_file() and (out / "rendering_profile.csv").is_file()
    diagnostics = read_jsonl(out / "diagnostics.jsonl")
    assert diagnostics and all(d["kind"] for d in diagnostics), "unanswered pairs are reported, not dropped"
    assert {r.task for r in fake.requests} == {"components"}


def test_perturb_needs_a_distant_witness_chapter(tmp_path: Path, capsys) -> None:
    from test_pipeline_support import fill_cache

    root, run = make_root(tmp_path), tmp_path / "run"
    assert cli(root, run, "ingest") == 0
    fill_cache(context(root, run))
    assert main(["run", "--offline", "--root", str(root), "--run-dir", str(run)]) == 0
    capsys.readouterr()
    assert main(["perturb", "--offline", "--root", str(root), "--run-dir", str(run)]) == 2
    assert "no witness chapter outside its core window" in capsys.readouterr().err
