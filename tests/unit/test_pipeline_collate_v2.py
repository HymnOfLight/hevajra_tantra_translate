"""The collate stage and the instrument digests under a selected prompt version: v2 plans,
requests, check lexicon, provenance and the glossary's witness check; v1 digests and
provenance unchanged by the version field; the collated text recorded and checked."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from hevajra_matrix.cli import main
from hevajra_matrix.collate.collator import PROMPT_V1, PROMPT_V2, SCHEMA_V1, SCHEMA_V2
from hevajra_matrix.config import load_settings
from hevajra_matrix.llm.cache import CachedClient
from hevajra_matrix.llm.client import LLMRequest, canonical_json, sha256_text
from hevajra_matrix.llm.fake import FakeClient
from hevajra_matrix.pipeline import StageError, plan_collation
from hevajra_matrix.pipeline.context import load_texts
from hevajra_matrix.pipeline.instrument import check_prompt
from hevajra_matrix.pipeline.store import read_alignment
from hevajra_matrix.prereg import instruments

from test_pipeline_support import REPO, context, echo_script, make_root

# Changed from 581b3cac... when collate.verify.VERIFY_VERSION (2) joined the collate parameters,
# so that a change to the code checks changes the digest; prompt, schema and request keys did not.
V1_COLLATE_PARAMS_SHA = "e65ffb04e787c0c869863132d0095565b6fb0d85d4f28eb2536e1f6d359803b2"


def select_prompt(root: Path, prompt: str) -> None:
    path = root / "config" / "llm.yaml"
    text = path.read_text(encoding="utf-8")
    assert "prompt: collate.v1}" in text
    path.write_text(text.replace("prompt: collate.v1}", f"prompt: {prompt}}}"), encoding="utf-8")


def echo_v2(request: LLMRequest) -> dict:
    """``echo_script`` with the convention a v2 schema asks for."""
    answer = echo_script(request)
    if "convention" in request.schema["properties"]["units"]["items"]["properties"]:
        answer["units"] = [{**u, "convention": "none"} for u in answer["units"]]
    return answer


def collate_offline(root: Path, run: Path) -> None:
    ctx = context(root, run)
    cache = CachedClient(FakeClient(echo_v2), ctx.settings.path("cache"))
    for _, _, request in plan_collation(ctx, replicates=1).requests():
        cache.complete(request)
    assert main(["collate", "--offline", "--replicates", "1", "--root", str(root), "--run-dir", str(run)]) == 0


@pytest.fixture
def ingested(tmp_path: Path) -> tuple[Path, Path]:
    root, run = make_root(tmp_path), tmp_path / "run"
    assert main(["ingest", "--root", str(root), "--run-dir", str(run)]) == 0
    return root, run


def test_v2_plan_requests_lexicon_and_provenance(ingested: tuple[Path, Path]) -> None:
    root, run = ingested
    select_prompt(root, PROMPT_V2)
    ctx = context(root, run)
    cplan = plan_collation(ctx)
    assert cplan.settings.prompt == cplan.examples.prompt == PROMPT_V2
    assert cplan.template.startswith("# Collation task (collate.v2)")
    assert all(request.schema == SCHEMA_V2 and request.task == "collate" for _, _, request in cplan.requests())
    lexicon = cplan.lexicon(ctx, load_texts(ctx))
    assert lexicon.standing_renderings is cplan.examples.glossary, "V12 reads the glossary shown in the prompt"
    collate_offline(root, run)
    index = json.loads((run / "collation" / "replicates.json").read_text(encoding="utf-8"))
    assert index["prompt"] == PROMPT_V2 and index["tags"] == ["r1"]
    alignment = read_alignment(run / "alignments" / "claude.r1.jsonl")
    assert alignment.links and {link.source for link in alignment.links} == {"claude-opus-5-5:collate.v2:r1"}


def test_v1_stays_the_default_with_its_provenance(ingested: tuple[Path, Path]) -> None:
    root, run = ingested
    cplan = plan_collation(context(root, run))
    assert cplan.settings.prompt == PROMPT_V1 and cplan.examples.glossary is None
    assert all(request.schema == SCHEMA_V1 for _, _, request in cplan.requests())
    collate_offline(root, run)
    assert json.loads((run / "collation" / "replicates.json").read_text(encoding="utf-8"))["prompt"] == PROMPT_V1
    alignment = read_alignment(run / "alignments" / "claude.r1.jsonl")
    assert {link.source for link in alignment.links} == {"claude-opus-5-5:collate.v1:r1"}
    assert not any({"standing_rendering", "transcribed_name", "convention_unsupported"} & link.flags
                   for link in alignment.links)


def test_a_glossary_written_for_another_witness_is_refused(ingested: tuple[Path, Path]) -> None:
    root, run = ingested
    select_prompt(root, PROMPT_V2)
    path = root / "data" / "codebook" / "collate.v2" / "standing_renderings.yaml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    doc["witness"] = "zh_other_witness"
    path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    with pytest.raises(StageError, match="never borrowed"):
        plan_collation(context(root, run))


def test_a_missing_glossary_stops_the_plan(ingested: tuple[Path, Path]) -> None:
    root, run = ingested
    select_prompt(root, PROMPT_V2)
    (root / "data" / "codebook" / "collate.v2" / "standing_renderings.yaml").unlink()
    with pytest.raises(StageError, match=r"standing_renderings\.yaml is missing"):
        plan_collation(context(root, run))


def _edit_examples_in_place(root: Path) -> None:
    """Change the v1 examples' wording without a new version name (the prompt text changes)."""
    path = root / "data" / "codebook" / "collate_examples.yaml"
    text = path.read_text(encoding="utf-8")
    assert "rendered in other words" in text
    path.write_text(text.replace("rendered in other words", "said otherwise"), encoding="utf-8")


def test_a_run_records_its_collated_text_and_refuses_another_under_the_same_name(ingested) -> None:
    root, run = ingested
    collate_offline(root, run)
    index = json.loads((run / "collation" / "replicates.json").read_text(encoding="utf-8"))
    request = plan_collation(context(root, run), replicates=1).requests()[0][2]
    assert (index["prompt_sha"], index["schema_sha"]) == (request.prompt_sha, sha256_text(canonical_json(SCHEMA_V1)))
    assert check_prompt(context(root, run)) == PROMPT_V1
    _edit_examples_in_place(root)
    with pytest.raises(StageError, match=f"another collate.v1 text \\(prompt_sha {index['prompt_sha'][:12]} "
                                         "recorded, [0-9a-f]{12} configured\\): its template, examples or glossary"):
        check_prompt(context(root, run))


def test_build_refuses_a_text_revised_under_the_same_name(ingested, capsys) -> None:
    root, run = ingested
    collate_offline(root, run)
    _edit_examples_in_place(root)
    assert main(["build", "--root", str(root), "--run-dir", str(run), "--offline"]) != 0
    assert "another collate.v1 text" in capsys.readouterr().err
    assert not (run / "alignments" / "claude.jsonl").exists()


def test_a_run_from_before_the_recorded_text_is_checked_by_name_only(ingested) -> None:
    root, run = ingested
    collate_offline(root, run)
    path = run / "collation" / "replicates.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({k: v for k, v in index.items() if k not in ("prompt_sha", "schema_sha")}),
                    encoding="utf-8")
    _edit_examples_in_place(root)
    assert check_prompt(context(root, run)) == PROMPT_V1


# --------------------------------------------------------------------------- digests
def test_the_prompt_field_does_not_enter_the_v1_digest() -> None:
    settings = load_settings(REPO)
    v1 = instruments(settings)["collate"]
    assert v1.params_sha == V1_COLLATE_PARAMS_SHA
    llm = {**settings.llm, "tasks": {**settings.llm["tasks"],
                                     "collate": {**settings.llm["tasks"]["collate"], "prompt": PROMPT_V2}}}
    v2 = instruments(replace(settings, llm=llm))["collate"]
    assert v2.params_sha == v1.params_sha, "the version is captured by the prompt and schema shas"
    assert (v2.prompt_sha, v2.schema_sha) != (v1.prompt_sha, v1.schema_sha) and v2.digest() != v1.digest()
    without = {**settings.llm, "tasks": {**settings.llm["tasks"], "collate": {
        k: v for k, v in settings.llm["tasks"]["collate"].items() if k != "prompt"}}}
    assert instruments(replace(settings, llm=without))["collate"] == v1, "an explicit collate.v1 changes nothing"
