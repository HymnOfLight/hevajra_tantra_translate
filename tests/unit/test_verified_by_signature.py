"""The verification signature (``verified_by``) of the glossary, the sentinels and the topic
codebook: status verified needs the verifier's name as a non-blank string; any other status
needs verified_by null. A truthy non-name (whitespace, true, a number) is not a signature."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from hevajra_matrix.collate.collator import PROMPT_V2, load_examples
from hevajra_matrix.core.lexicon import LexiconError
from hevajra_matrix.evaluation import sentinels as S
from hevajra_matrix.topics import CodebookError, load_codebook

from test_collate_v2 import glossary_doc, v2_root, write_yaml

DATA = Path(__file__).resolve().parents[2] / "data"
NOT_A_NAME = [None, "", " ", "\t\n", True, 1, ["A. Researcher"]]
NOT_NULL = ["A. Researcher", "", " ", False, 0]


def _glossary(tmp_path: Path, status: str, by) -> Path:
    root = v2_root(tmp_path)
    doc = glossary_doc()
    doc["entries"][0].update(status=status, verified_by=by)
    write_yaml(root / "codebook" / "collate.v2" / "standing_renderings.yaml", doc)
    return root


# --------------------------------------------------------------------------- glossary
@pytest.mark.parametrize("by", NOT_A_NAME)
def test_a_verified_glossary_entry_needs_a_name(tmp_path: Path, by) -> None:
    with pytest.raises(LexiconError, match="verified_by"):
        load_examples(_glossary(tmp_path, "verified", by), prompt=PROMPT_V2)


@pytest.mark.parametrize("status", ["proposed", "retired"])
@pytest.mark.parametrize("by", NOT_NULL)
def test_an_unverified_glossary_entry_carries_no_name(tmp_path: Path, status: str, by) -> None:
    with pytest.raises(LexiconError, match="verified_by"):
        load_examples(_glossary(tmp_path, status, by), prompt=PROMPT_V2)


@pytest.mark.parametrize("status, by, shown", [("verified", "A. Researcher", 1), ("proposed", None, 0),
                                               ("retired", None, 0)])
def test_a_well_signed_glossary_entry_loads(tmp_path: Path, status: str, by, shown: int) -> None:
    ex = load_examples(_glossary(tmp_path, status, by), prompt=PROMPT_V2)
    assert len(ex.glossary.entries) == shown


# --------------------------------------------------------------------------- sentinels
def _sentinels(tmp_path: Path, status: str, by) -> Path:
    path = tmp_path / "sentinels.yaml"
    path.write_text(yaml.safe_dump({"schema_version": S.SCHEMA_VERSION, "sentinels": [
        {"id": "a", "status": status, "verified_by": by, "source": "s", "stage": "ingest", "check": "adjacent",
         "loci": {}, "quotes": {}, "expect": {}}]}, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize("by", NOT_A_NAME)
def test_a_verified_sentinel_needs_a_name(tmp_path: Path, by) -> None:
    with pytest.raises(S.SentinelError, match="names who verified"):
        S.load(_sentinels(tmp_path, "verified", by))


@pytest.mark.parametrize("status", ["proposed", "retired"])
@pytest.mark.parametrize("by", NOT_NULL)
def test_an_unverified_sentinel_carries_no_name(tmp_path: Path, status: str, by) -> None:
    with pytest.raises(S.SentinelError, match="verified_by must be null"):
        S.load(_sentinels(tmp_path, status, by))


@pytest.mark.parametrize("status, by", [("verified", "code review"), ("proposed", None), ("retired", None)])
def test_a_well_signed_sentinel_loads(tmp_path: Path, status: str, by) -> None:
    (loaded,) = S.load(_sentinels(tmp_path, status, by))
    assert (loaded.status, loaded.verified_by) == (status, by)


def test_the_committed_sentinels_are_well_signed() -> None:
    assert S.load(DATA / "sentinels" / "sentinels.yaml")


# --------------------------------------------------------------------------- topic codebook
@pytest.fixture(scope="module")
def raw() -> dict:
    return yaml.safe_load((DATA / "codebook" / "topics.yaml").read_text(encoding="utf-8"))


def _codebook(tmp_path: Path, raw: dict, status: str, by) -> Path:
    doc = copy.deepcopy(raw)
    doc.update(status=status, verified_by=by)
    path = tmp_path / "topics.yaml"
    path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize("by", NOT_A_NAME)
def test_a_verified_codebook_needs_a_name(tmp_path: Path, raw: dict, by) -> None:
    with pytest.raises(CodebookError, match="needs verified_by"):
        load_codebook(_codebook(tmp_path, raw, "verified", by))


@pytest.mark.parametrize("by", NOT_NULL)
def test_a_draft_codebook_carries_no_name(tmp_path: Path, raw: dict, by) -> None:
    with pytest.raises(CodebookError, match="verified_by must be null"):
        load_codebook(_codebook(tmp_path, raw, "draft", by))


def test_the_committed_codebook_is_well_signed() -> None:
    cb = load_codebook(DATA / "codebook" / "topics.yaml")
    assert cb.status == "draft" and cb.verified_by is None
